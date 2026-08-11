#!/usr/bin/env python3
"""Fail-closed runner for the three strict v3 event BiST-NDR tasks."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.remote.windows.protocol_v3_event_extension_runner import (  # noqa: E402
    ExtensionInputs,
    SUPPORT_MANIFEST,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import (  # noqa: E402
    Invocation,
    Manifest,
    ManifestError,
    ProtocolV3Inputs,
    Task,
    load_manifest,
    run_queue,
)


CANONICAL_MANIFEST = (
    PROJECT_ROOT / "config/protocol_v3/table4_event_bist_ndr_v1.json"
)
METHOD_PROFILE = (
    PROJECT_ROOT / "config/protocol_v3/event_bist_ndr_profile_v1.json"
)
REGIONS = ("Alameda", "ContraCosta", "Orange")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bist_ndr_manifest(path: Path | str) -> Manifest:
    manifest = load_manifest(path, require_production_matrix=False)
    expected_training = {
        "epochs": 30,
        "patience": 20,
        "device": "cuda:0",
        "seed": 42,
        "checkpoint_metric": "global_validation_masked_mae_all_valid_points",
        "evaluation_mask": "explicit_v3_flow_mask",
        "partial_batch": "retain",
        "traffic_loading": "eager",
        "training_analysis_labels": False,
        "method_profile": "config/protocol_v3/event_bist_ndr_profile_v1.json",
        "method_profile_sha256": _sha256(METHOD_PROFILE),
    }
    if dict(manifest.training) != expected_training:
        raise ManifestError("v3 BiST-NDR training contract is invalid")
    expected = {(region, "bist_ndr", 42, "event") for region in REGIONS}
    actual = {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    }
    if len(manifest.tasks) != 3 or actual != expected:
        raise ManifestError("v3 BiST-NDR manifest must contain the exact three tasks")
    for task in manifest.tasks:
        if task.kind != "event_extension" or task.variant != "v3_event_bist_ndr":
            raise ManifestError(f"v3 BiST-NDR task binding is invalid: {task.task_id}")
    return manifest


def build_bist_ndr_invocation(
    task: Task,
    manifest: Manifest,
    extension: ExtensionInputs,
    attempt_dir: Path,
) -> Invocation:
    if task.model != "bist_ndr":
        raise ManifestError(f"unsupported v3 BiST-NDR method: {task.model}")
    if extension.max_microbatch_size < 0:
        raise ManifestError("max_microbatch_size must be nonnegative")
    effective_microbatch_size = extension.max_microbatch_size
    if not effective_microbatch_size and task.region == "Orange":
        effective_microbatch_size = 8
    work = attempt_dir / "work"
    command = [
        extension.python_executable,
        "-X",
        "utf8",
        "scripts/train_protocol_v3_event_extension.py",
        "--method",
        task.model,
        "--variant",
        task.variant,
        "--region",
        task.region,
        "--seed",
        str(task.seed),
        "--epochs",
        str(manifest.training["epochs"]),
        "--patience",
        str(manifest.training["patience"]),
        "--device",
        str(manifest.training["device"]),
        "--data-dir",
        str(extension.cache_dir),
        "--graph-dir",
        str(extension.graph_dir),
        "--support-root",
        str(extension.support_root),
        "--node-order",
        str(extension.node_order),
        "--out-dir",
        str(work),
        "--traffic-loading",
        str(manifest.training["traffic_loading"]),
        "--run-contract",
        "formal30",
    ]
    if effective_microbatch_size:
        command += ["--max-microbatch-size", str(effective_microbatch_size)]
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
        "early_stopping_patience": None,
        "upstream_curriculum_applied": False,
        "igstgnn_data_parallel": False,
        "max_microbatch_size": effective_microbatch_size,
        "run_contract": "formal30",
        "claim_scope": "same_protocol_project_method_extension",
    }
    return Invocation(
        command=tuple(command),
        result_relpath=Path("work") / task.region,
        required_artifacts=(
            "summary.json",
            "ckpt_best.pt",
            "test_predictions.npz",
            "train.log",
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
    parser.add_argument("--max-microbatch-size", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_bist_ndr_manifest(args.manifest)
        root = args.repo_root.resolve()
        protocol = ProtocolV3Inputs(
            results_root=_resolve(
                root, args.results_root or manifest.default_results_root
            ),
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
            igstgnn_data_parallel=False,
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
        command_builder=lambda task, current, _inputs, attempt: (
            build_bist_ndr_invocation(task, current, extension, attempt)
        ),
        verify_git=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
