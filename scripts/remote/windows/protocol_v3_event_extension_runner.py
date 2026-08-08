#!/usr/bin/env python3
"""Fail-closed runner for the nine strict v3 event extension tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts.remote.windows.protocol_v3_runner import (
    Invocation,
    Manifest,
    ManifestError,
    ProtocolV3Inputs,
    Task,
    load_manifest,
    run_queue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_MANIFEST = (
    PROJECT_ROOT / "config/protocol_v3/table4_event_extensions_v1.json"
)
SUPPORT_MANIFEST = (
    PROJECT_ROOT / "config/protocol_v3/event_extension_support_v1.json"
)
METHODS = ("bist", "igstgnn", "ndr_net")
REGIONS = ("Alameda", "ContraCosta", "Orange")


@dataclass(frozen=True)
class VerifiedExtensionSupport:
    support_root: Path
    node_order: Path
    manifest_sha256: str


@dataclass(frozen=True)
class ExtensionInputs:
    support_root: Path
    node_order: Path
    support_manifest_sha256: str
    igstgnn_data_parallel: bool
    max_microbatch_size: int
    cache_dir: Path = Path("outputs/dist_net/region_data_v3_noleak_dedup")
    graph_dir: Path = Path("outputs/region_graphs")
    python_executable: str = sys.executable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, binding: object, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"extension support {label} is missing or a symlink")
    if not isinstance(binding, dict):
        raise ManifestError(f"extension support binding is invalid for {label}")
    expected_bytes = binding.get("bytes")
    expected_sha256 = binding.get("sha256")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ManifestError(f"extension support binding is invalid for {label}")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
        raise ManifestError(f"extension support file does not match binding: {label}")


def verify_extension_support(
    *,
    support_manifest_path: Path | str,
    support_root: Path | str,
    node_order: Path | str,
) -> VerifiedExtensionSupport:
    manifest_path = Path(support_manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestError("extension support manifest is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("extension support manifest is invalid") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("support_id")
        != "protocol_v3_event_extensions_official_model_support_v1"
    ):
        raise ManifestError("extension support manifest identity is invalid")
    root = Path(support_root).resolve()
    order = Path(node_order).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ManifestError("extension support root is missing or a symlink")
    _verify_file(order, payload.get("node_order"), "node_order.npy")
    regions = payload.get("regions")
    if not isinstance(regions, dict) or set(regions) != {
        "Alameda", "Contra_Costa", "Orange"
    }:
        raise ManifestError("extension support region set is invalid")
    for region, files in regions.items():
        if not isinstance(files, dict) or set(files) != {
            "sensors.csv", "desc_mapping.json", "type_mapping.json", "adj_matrix.npy"
        }:
            raise ManifestError(f"extension support files are invalid for {region}")
        for filename, binding in files.items():
            _verify_file(root / region / filename, binding, f"{region}/{filename}")
    return VerifiedExtensionSupport(
        support_root=root,
        node_order=order,
        manifest_sha256=_sha256(manifest_path),
    )


def load_extension_manifest(
    path: Path | str,
    *,
    require_complete_matrix: bool = True,
) -> Manifest:
    manifest = load_manifest(path, require_production_matrix=False)
    expected_training = {
        "epochs": 30,
        "patience": 20,
        "device": "cuda:0",
        "seed": 42,
        "checkpoint_metric": "global_validation_masked_mae_all_valid_points",
        "evaluation_mask": "explicit_v3_flow_mask",
        "partial_batch": "retain",
        "training_analysis_labels": False,
    }
    if dict(manifest.training) != expected_training:
        raise ManifestError("v3 extension training contract is invalid")
    expected = {
        (region, method, 42, "event")
        for region in REGIONS
        for method in METHODS
    }
    actual = {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    }
    if require_complete_matrix:
        if len(manifest.tasks) != 9 or actual != expected:
            raise ManifestError("v3 extension manifest must contain the exact nine tasks")
    elif len(actual) != len(manifest.tasks) or not actual <= expected:
        raise ManifestError("v3 extension shard is not a canonical task subset")
    for task in manifest.tasks:
        if task.kind != "event_extension" or task.variant != "v3_event_aligned":
            raise ManifestError(f"v3 extension task binding is invalid: {task.task_id}")
    return manifest


def build_extension_invocation(
    task: Task,
    manifest: Manifest,
    extension: ExtensionInputs,
    attempt_dir: Path,
) -> Invocation:
    if task.model not in METHODS:
        raise ManifestError(f"unsupported v3 extension method: {task.model}")
    if extension.max_microbatch_size < 0:
        raise ManifestError("max_microbatch_size must be nonnegative")
    effective_microbatch_size = extension.max_microbatch_size
    if (
        not effective_microbatch_size
        and task.region == "Orange"
        and task.model in {"igstgnn", "ndr_net"}
    ):
        effective_microbatch_size = 8
    work = attempt_dir / "work"
    command = [
        extension.python_executable,
        "-X", "utf8",
        "scripts/train_protocol_v3_event_extension.py",
        "--method", task.model,
        "--region", task.region,
        "--seed", str(task.seed),
        "--epochs", str(manifest.training["epochs"]),
        "--patience", str(manifest.training["patience"]),
        "--device", str(manifest.training["device"]),
        "--data-dir", str(extension.cache_dir),
        "--graph-dir", str(extension.graph_dir),
        "--support-root", str(extension.support_root),
        "--node-order", str(extension.node_order),
        "--out-dir", str(work),
    ]
    if effective_microbatch_size:
        command += ["--max-microbatch-size", str(effective_microbatch_size)]
    if task.model == "igstgnn" and extension.igstgnn_data_parallel:
        command.append("--igstgnn-data-parallel")
    config = {
        "kind": task.kind,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "sampling_protocol": task.sampling_protocol,
        **dict(manifest.training),
        "support_root": str(extension.support_root),
        "node_order": str(extension.node_order),
        "support_manifest_sha256": extension.support_manifest_sha256,
        "early_stopping_patience": (
            None if task.model == "ndr_net" else int(manifest.training["patience"])
        ),
        "upstream_curriculum_applied": task.model == "igstgnn",
        "igstgnn_data_parallel": bool(
            task.model == "igstgnn" and extension.igstgnn_data_parallel
        ),
        "max_microbatch_size": effective_microbatch_size,
    }
    return Invocation(
        command=tuple(command),
        result_relpath=Path("work") / task.region,
        required_artifacts=(
            "summary.json", "ckpt_best.pt", "test_predictions.npz", "train.log"
        ),
        required_metrics=("all",),
        config=config,
    )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--support-manifest", type=Path, default=SUPPORT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--allow-shard", action="store_true")
    parser.add_argument("--igstgnn-data-parallel", action="store_true")
    parser.add_argument("--max-microbatch-size", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_extension_manifest(
            args.manifest,
            require_complete_matrix=not args.allow_shard,
        )
        root = args.repo_root.resolve()
        protocol = ProtocolV3Inputs(
            results_root=_resolve(root, args.results_root or manifest.default_results_root),
            cache_dir=_resolve(root, args.cache_dir or manifest.default_cache_dir),
            graph_dir=_resolve(root, args.graph_dir or manifest.default_graph_dir),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
        support = verify_extension_support(
            support_manifest_path=_resolve(root, args.support_manifest),
            support_root=_resolve(root, args.support_root),
            node_order=_resolve(root, args.node_order),
        )
        extension = ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=args.igstgnn_data_parallel,
            max_microbatch_size=args.max_microbatch_size,
            cache_dir=protocol.cache_dir,
            graph_dir=protocol.graph_dir,
            python_executable=protocol.python_executable,
        )
    except ManifestError as exc:
        print(f"CONFIG_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    return run_queue(
        manifest,
        protocol,
        repo_root=root,
        command_builder=lambda task, current, _inputs, attempt: build_extension_invocation(
            task, current, extension, attempt
        ),
        verify_git=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
