#!/usr/bin/env python3
"""Generate the formal go/no-go report from a complete FDN benchmark v1 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from scripts.fdn_benchmark_v1 import strict_json_write
from scripts.remote.windows.fdn_benchmark_v1_runner import (
    Task,
    derive_config_hash,
    load_manifest,
    sha256_file,
    validate_result,
)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _daily_window_differences(
    learnable: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
) -> np.ndarray:
    for key in ("actual", "mask", "forecast_origin"):
        if not np.array_equal(learnable[key], control[key]):
            raise ValueError(f"paired prediction artifacts differ in {key}")
    mask = learnable["mask"]

    def window_mae(prediction: np.ndarray) -> np.ndarray:
        absolute = np.abs(prediction.astype(np.float64) - learnable["actual"].astype(np.float64))
        counts = mask.sum(axis=(1, 2))
        if np.any(counts <= 0):
            raise ValueError("paired prediction artifact contains an empty test window")
        return (absolute * mask).sum(axis=(1, 2), dtype=np.float64) / counts

    difference = window_mae(learnable["prediction"]) - window_mae(control["prediction"])
    day = learnable["forecast_origin"] // 288
    return np.asarray(
        [difference[day == value].mean(dtype=np.float64) for value in np.unique(day)],
        dtype=np.float64,
    )


def _hierarchical_bootstrap(
    day_differences_by_seed: list[np.ndarray],
    *,
    iterations: int = 20_000,
    seed: int = 20260822,
) -> tuple[float, float]:
    if len(day_differences_by_seed) < 2 or any(values.size < 2 for values in day_differences_by_seed):
        raise ValueError("hierarchical bootstrap requires multiple seeds and day blocks")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    n_seeds = len(day_differences_by_seed)
    for iteration in range(iterations):
        sampled_seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        seed_means = []
        for index in sampled_seed_indices:
            values = day_differences_by_seed[int(index)]
            selected = rng.integers(0, values.size, size=values.size)
            seed_means.append(values[selected].mean(dtype=np.float64))
        draws[iteration] = np.mean(seed_means, dtype=np.float64)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def generate_report(
    *,
    manifest_path: Path,
    results_root: Path,
    data_dir: Path,
    output_dir: Path,
    code_sha: str,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    queue = json.loads((results_root / "queue_status.json").read_text(encoding="utf-8"))
    if queue.get("state") != "completed" or queue.get("completed_tasks") != 24:
        raise ValueError("benchmark matrix is not complete")
    if queue.get("failed_count") != 0 or queue.get("code_sha") != code_sha:
        raise ValueError("benchmark queue identity is invalid")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_rows: list[dict[str, object]] = []
    summaries: dict[tuple[str, int, str], Mapping[str, object]] = {}
    task_by_cell: dict[tuple[str, int, str], Task] = {}
    validation_receipts: dict[str, object] = {}
    for task in manifest.tasks:
        config_hash = derive_config_hash(task, manifest, code_sha)
        result_dir = results_root / "tasks" / task.task_id
        receipt = validate_result(
            result_dir,
            task,
            manifest,
            code_sha=code_sha,
            config_hash=config_hash,
            data_dir=data_dir,
        )
        summary = receipt["summary"]
        metrics = summary["test_metrics"]
        cell = (task.dataset, task.seed, task.variant)
        summaries[cell] = summary
        task_by_cell[cell] = task
        validation_receipts[task.task_id] = receipt
        run_rows.append(
            {
                "dataset": task.dataset,
                "seed": task.seed,
                "variant": task.variant,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape_percent": metrics["mape_percent"],
                "h3_mae": metrics["h3"]["mae"],
                "h6_mae": metrics["h6"]["mae"],
                "h12_mae": metrics["h12"]["mae"],
                "best_epoch": summary["best_epoch"],
                "parameters": summary["parameters"],
                "training_time_seconds": summary["training_time_seconds"],
                "peak_memory_bytes": summary["peak_memory_bytes"],
                "task_id": task.task_id,
            }
        )
    run_rows.sort(key=lambda row: (str(row["dataset"]), str(row["variant"]), int(row["seed"])))
    _write_csv(output_dir / "runs.csv", run_rows)

    aggregate_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["dataset"]), str(row["variant"]))].append(row)
    for (dataset, variant), rows in sorted(grouped.items()):
        values = np.asarray([float(row["mae"]) for row in rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_seeds": int(values.size),
                "mae_mean": float(values.mean()),
                "mae_sample_std": float(values.std(ddof=1)),
                "mae_min": float(values.min()),
                "mae_max": float(values.max()),
            }
        )
    _write_csv(output_dir / "aggregate.csv", aggregate_rows)

    paired_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    negative_pairs = 0
    negative_dataset_means = 0
    significant_datasets = 0
    for dataset in ("pems04", "pems08"):
        seed_differences: list[float] = []
        day_differences: list[np.ndarray] = []
        for seed in (42, 1, 2):
            learnable = summaries[(dataset, seed, "fdn_learnable_k3")]
            control = summaries[(dataset, seed, "dual_gwn")]
            difference = float(learnable["test_metrics"]["mae"]) - float(
                control["test_metrics"]["mae"]
            )
            seed_differences.append(difference)
            if difference < 0:
                negative_pairs += 1
            paired_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "fdn_learnable_mae": learnable["test_metrics"]["mae"],
                    "dual_gwn_mae": control["test_metrics"]["mae"],
                    "difference_fdn_minus_dual": difference,
                }
            )
            learnable_task = task_by_cell[(dataset, seed, "fdn_learnable_k3")]
            control_task = task_by_cell[(dataset, seed, "dual_gwn")]
            day_differences.append(
                _daily_window_differences(
                    _load_prediction(
                        results_root / "tasks" / learnable_task.task_id / "test_predictions.npz"
                    ),
                    _load_prediction(
                        results_root / "tasks" / control_task.task_id / "test_predictions.npz"
                    ),
                )
            )
        mean_difference = float(np.mean(seed_differences))
        if mean_difference < 0:
            negative_dataset_means += 1
        lower, upper = _hierarchical_bootstrap(day_differences)
        if upper < 0:
            significant_datasets += 1
        bootstrap_rows.append(
            {
                "dataset": dataset,
                "mean_seed_paired_difference": mean_difference,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "day_blocks_per_seed": "/".join(str(values.size) for values in day_differences),
                "bootstrap_iterations": 20000,
            }
        )
    _write_csv(output_dir / "paired.csv", paired_rows)
    _write_csv(output_dir / "block_bootstrap.csv", bootstrap_rows)
    go_no_go = manifest.raw["go_no_go"]
    decision = (
        "continue"
        if negative_dataset_means == 2
        and negative_pairs >= int(go_no_go["minimum_negative_seed_pairs"])
        and significant_datasets >= int(go_no_go["minimum_datasets_ci_upper_below_zero"])
        else "stop_independent_method_paper"
    )
    report = {
        "status": "verified_go_no_go_report",
        "matrix_id": manifest.matrix_id,
        "protocol_version": manifest.raw["protocol_version"],
        "code_sha": code_sha,
        "manifest_sha256": sha256_file(manifest.path),
        "queue_status_sha256": sha256_file(results_root / "queue_status.json"),
        "validated_tasks": len(validation_receipts),
        "negative_seed_pairs": negative_pairs,
        "negative_dataset_means": negative_dataset_means,
        "datasets_ci_upper_below_zero": significant_datasets,
        "decision": decision,
        "validation_receipts_digest": hashlib_sha256(validation_receipts),
    }
    strict_json_write(output_dir / "report.json", report)
    markdown = [
        "# FDN benchmark v1 go/no-go report",
        "",
        f"- Decision: `{decision}`",
        f"- Negative seed pairs: `{negative_pairs}/6`",
        f"- Negative dataset means: `{negative_dataset_means}/2`",
        f"- Datasets with bootstrap upper bound below zero: `{significant_datasets}/2`",
        "",
        "The independent replicate is a training seed. The confidence interval uses a hierarchical bootstrap over seeds and 288-origin day blocks.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def hashlib_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = generate_report(
        manifest_path=args.manifest.resolve(),
        results_root=args.results_root.resolve(),
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        code_sha=args.code_sha.lower(),
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
