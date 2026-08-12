#!/usr/bin/env python3
"""Fail-closed runner for the three-region v3 event FDN supplement."""

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


REGIONS = ("Alameda", "ContraCosta", "Orange")
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "config/protocol_v3/table4_event_fdn_v1.json"
)


def load_fdn_manifest(path: Path | str) -> Manifest:
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
    if manifest.training != expected_training:
        raise ManifestError(
            "v3 event FDN training contract must equal "
            f"{expected_training!r}"
        )
    expected = {
        (region, "fourier_dual_net", "learnable_k3", 42, "event")
        for region in REGIONS
    }
    actual = {
        (
            task.region,
            task.model,
            task.variant,
            task.seed,
            task.sampling_protocol,
        )
        for task in manifest.tasks
    }
    if len(manifest.tasks) != 3 or actual != expected:
        raise ManifestError(
            "v3 event FDN matrix must contain learnable-K3 seed 42 for "
            "exactly Alameda, ContraCosta, and Orange"
        )
    if any(task.kind != "fourier_dual_net" for task in manifest.tasks):
        raise ManifestError("v3 event FDN tasks must use fourier_dual_net kind")
    return manifest


def build_fdn_invocation(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    attempt_dir: Path,
) -> Invocation:
    work = attempt_dir / "work"
    epochs = int(manifest.training["epochs"])
    patience = int(manifest.training["patience"])
    batch_size = int(manifest.training["batch_size"])
    device = str(manifest.training["device"])
    command = (
        run_inputs.python_executable,
        "-X",
        "utf8",
        "scripts/train_fourier_dual_net.py",
        "--region",
        task.region,
        "--seed",
        str(task.seed),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--data_dir",
        str(run_inputs.cache_dir),
        "--graph_dir",
        str(run_inputs.graph_dir),
        "--out_dir",
        str(work),
        "--tag",
        "learnable_K3",
        "--protocol",
        "event",
        "--decomp_mode",
        "learnable",
        "--K",
        "3",
    )
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
        "decomp_mode": "learnable",
        "K": 3,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "t_max": "epochs_times_training_batches",
            "eta_min": 0.00001,
            "step_unit": "optimizer_step",
        },
    }
    return Invocation(
        command=command,
        result_relpath=Path("work") / "learnable_K3" / task.region,
        required_artifacts=(
            "summary.json",
            "ckpt_best.pt",
            "test_predictions.npz",
            "train.log",
            "decomp_mask.json",
        ),
        required_metrics=("all", "affected", "unaffected"),
        config=config,
    )


def _resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
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
        manifest = load_fdn_manifest(args.manifest)
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
        command_builder=build_fdn_invocation,
        verify_git=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
