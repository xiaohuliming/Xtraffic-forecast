#!/usr/bin/env python3
"""Fail-closed runner for the supported v3 Table 4 baseline subset."""

from __future__ import annotations

import argparse
import sys
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


BASELINE_METHODS = (
    "hl",
    "lstm",
    "dcrnn",
    "agcrn",
    "stgcn",
    "graphwavenet",
    "astgcn",
    "sttn",
    "dstagnn",
    "dgcrn",
    "d2stgnn",
)
ZOO_METHODS = frozenset(BASELINE_METHODS) - {"graphwavenet"}
REGIONS = ("Alameda", "ContraCosta", "Orange")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "protocol_v3"
    / "table4_fullwindow_baselines_phase1_v1.json"
)


def _training_value(manifest: Manifest, name: str, expected_type: type) -> object:
    value = manifest.training.get(name)
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise ManifestError(f"training.{name} has an invalid type")
    return value


def load_baseline_manifest(path: Path | str) -> Manifest:
    manifest = load_manifest(path, require_production_matrix=False)
    expected_training = {
        "epochs": 30,
        "patience": 6,
        "device": "cuda:0",
        "batch_size": 32,
        "stride": 1,
        "train_frac": 0.7,
        "val_frac": 0.15,
    }
    if dict(manifest.training) != expected_training:
        raise ManifestError(
            "v3 baseline phase 1 training contract must equal "
            f"{expected_training!r}"
        )
    expected = {
        (region, method, 42, "full_window")
        for region in REGIONS
        for method in BASELINE_METHODS
    }
    actual = {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    }
    if len(manifest.tasks) != 33 or actual != expected:
        raise ManifestError(
            "v3 baseline phase 1 must contain the exact 33-task supported "
            "Table 4 subset"
        )
    for task in manifest.tasks:
        expected_kind = (
            "graphwavenet" if task.model == "graphwavenet" else "baseline_zoo"
        )
        if task.kind != expected_kind or task.variant != "base":
            raise ManifestError(
                f"v3 baseline task binding is invalid for {task.task_id}"
            )
    return manifest


def build_baseline_invocation(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    attempt_dir: Path,
) -> Invocation:
    epochs = int(_training_value(manifest, "epochs", int))
    patience = int(_training_value(manifest, "patience", int))
    batch_size = int(_training_value(manifest, "batch_size", int))
    stride = int(_training_value(manifest, "stride", int))
    train_frac = float(_training_value(manifest, "train_frac", float))
    val_frac = float(_training_value(manifest, "val_frac", float))
    device = str(_training_value(manifest, "device", str))
    work = attempt_dir / "work"
    common = [
        run_inputs.python_executable,
        "-X",
        "utf8",
    ]
    if task.kind == "baseline_zoo" and task.model in ZOO_METHODS:
        command = common + [
            "scripts/train_baseline_zoo.py",
            "--model",
            task.model,
        ]
    elif task.kind == "graphwavenet" and task.model == "graphwavenet":
        command = common + ["scripts/train_graphwavenet.py"]
    else:
        raise ManifestError(f"unsupported v3 baseline task: {task.task_id}")
    command += [
        "--region",
        task.region,
        "--seed",
        str(task.seed),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--data-dir",
        str(run_inputs.cache_dir),
        "--graph-dir",
        str(run_inputs.graph_dir),
        "--out-dir",
        str(work),
        "--protocol",
        "full_window",
        "--stride",
        str(stride),
        "--train_frac",
        str(train_frac),
        "--val_frac",
        str(val_frac),
    ]
    artifacts = ["summary.json"]
    if task.model != "hl":
        artifacts.append("ckpt_best.pt")
    artifacts.extend(["test_predictions.npz", "train.log"])
    config = {
        "kind": task.kind,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "sampling_protocol": task.sampling_protocol,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "device": device,
        "cache_dir": str(run_inputs.cache_dir.resolve()),
        "graph_dir": str(run_inputs.graph_dir.resolve()),
        "stride": stride,
        "train_frac": train_frac,
        "val_frac": val_frac,
    }
    return Invocation(
        command=tuple(command),
        result_relpath=Path("work_fullwindow") / task.region,
        required_artifacts=tuple(artifacts),
        required_metrics=("all",),
        config=config,
    )


def _resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_baseline_manifest(args.manifest)
        repo_root = args.repo_root.resolve()
        run_inputs = ProtocolV3Inputs(
            results_root=_resolve(
                repo_root,
                args.results_root or manifest.default_results_root,
            ),
            cache_dir=_resolve(
                repo_root,
                args.cache_dir or manifest.default_cache_dir,
            ),
            graph_dir=_resolve(
                repo_root,
                args.graph_dir or manifest.default_graph_dir,
            ),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
    except ManifestError as exc:
        print(f"CONFIG_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    return run_queue(
        manifest,
        run_inputs,
        repo_root=repo_root,
        command_builder=build_baseline_invocation,
        verify_git=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
