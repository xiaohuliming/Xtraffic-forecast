#!/usr/bin/env python3
"""Audit experiment artifacts for the impact-guided forecasting project.

The audit is intentionally lightweight and read-only. It checks whether the
reported paper numbers can be traced to saved configs/metrics/logs, and records
the main leakage/reproducibility risks that should be resolved before a paper
submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-output",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay"),
    )
    parser.add_argument(
        "--normal-output",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/normal_stgnn_forecaster"),
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/ablation_summary/learned_normal_ablation_table.csv"),
    )
    parser.add_argument(
        "--seed-summary-csv",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/decay_seed_robustness/seed_robustness_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/experiment_audit"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    return str(value)


def metric_row(metrics: dict[str, Any], split: str) -> dict[str, float]:
    row = metrics["metrics"][split]
    return {
        "all_baseline": float(row["all_candidates_baseline_robust_mae"]),
        "all_model": float(row["all_candidates_model_robust_mae"]),
        "all_gain_pct": float(row["all_candidates_improvement_pct"]),
        "affected_baseline": float(row["affected_candidates_baseline_robust_mae"]),
        "affected_model": float(row["affected_candidates_model_robust_mae"]),
        "affected_gain_pct": float(row["affected_candidates_improvement_pct"]),
        "affected_node_rate": float(row["affected_node_rate"]),
    }


def audit_cache(cache_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    attrs: dict[str, Any] = {}
    with h5py.File(cache_path, "r") as h5:
        split = h5["split"][:]
        for name, code in SPLIT_TO_CODE.items():
            rows.append({"split": name, "samples": int(np.sum(split == code))})
        for key, value in h5.attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, np.generic):
                attrs[key] = value.item()
            else:
                attrs[key] = value
        attrs["datasets"] = sorted([key for key in h5.keys() if key != "stats"])
        attrs["has_stats_group"] = "stats" in h5
        if "stats" in h5:
            attrs["stats_datasets"] = sorted(h5["stats"].keys())
        attrs["sample_shape_y_residual"] = tuple(int(v) for v in h5["y_residual"].shape)
        attrs["sample_shape_hist_residual"] = tuple(int(v) for v in h5["hist_residual"].shape)
    return pd.DataFrame(rows), attrs


def audit_training_log(path: Path) -> dict[str, Any]:
    log = pd.read_csv(path)
    if log.empty:
        return {"epochs": 0}
    best_idx = log["val_loss"].idxmin()
    return {
        "epochs": int(len(log)),
        "first_train_loss": float(log.iloc[0]["train_loss"]),
        "last_train_loss": float(log.iloc[-1]["train_loss"]),
        "first_val_loss": float(log.iloc[0]["val_loss"]),
        "last_val_loss": float(log.iloc[-1]["val_loss"]),
        "best_epoch": int(log.loc[best_idx, "epoch"]),
        "best_val_loss": float(log.loc[best_idx, "val_loss"]),
        "val_still_decreasing": bool(log.iloc[-1]["val_loss"] <= log.iloc[0]["val_loss"]),
    }


def normal_summary(normal_output: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(normal_output.glob("*/metrics.json")):
        data = load_json(metrics_path)
        test = data["metrics"]["test"]
        rows.append(
            {
                "region": data["region"],
                "train_samples": data["samples"]["train"],
                "val_samples": data["samples"]["val"],
                "test_samples": data["samples"]["test"],
                "test_blend_robust": test["blend_robust_mae"],
                "test_model_robust": test["model_robust_mae"],
                "test_gain_pct": test["robust_improvement_pct"],
            }
        )
    return pd.DataFrame(rows)


def check_ablation_rows(ablation_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ablation_csv)
    rows = []
    for row in df.itertuples(index=False):
        metrics_path = Path(str(row.metrics_path))
        status = "missing"
        all_model = np.nan
        affected_model = np.nan
        if metrics_path.exists():
            data = load_json(metrics_path)
            test = metric_row(data, "test")
            all_model = test["all_model"]
            affected_model = test["affected_model"]
            status = "ok"
        rows.append(
            {
                "model": row.model,
                "metrics_path": str(metrics_path),
                "status": status,
                "csv_all_model": row.test_all_model_robust_mae,
                "json_all_model": all_model,
                "csv_affected_model": row.test_affected_model_robust_mae,
                "json_affected_model": affected_model,
                "all_abs_diff": abs(float(row.test_all_model_robust_mae) - all_model)
                if np.isfinite(all_model)
                else np.nan,
                "affected_abs_diff": abs(float(row.test_affected_model_robust_mae) - affected_model)
                if np.isfinite(affected_model)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_output = args.main_output.resolve()
    config = load_json(main_output / "config.json")
    metrics = load_json(main_output / "metrics.json")
    test = metric_row(metrics, "test")
    val = metric_row(metrics, "val")
    train = metric_row(metrics, "train")
    cache_path = Path(metrics.get("cache_path") or config["cache_path"]).resolve()
    split_df, cache_attrs = audit_cache(cache_path)
    log_stats = audit_training_log(main_output / "training_log.csv")
    normal_df = normal_summary(args.normal_output.resolve())
    ablation_df = check_ablation_rows(args.ablation_csv.resolve())
    seed_df = pd.read_csv(args.seed_summary_csv.resolve()) if args.seed_summary_csv.exists() else pd.DataFrame()

    split_df.to_csv(out_dir / "cache_split_counts.csv", index=False)
    normal_df.to_csv(out_dir / "normal_branch_summary.csv", index=False)
    ablation_df.to_csv(out_dir / "ablation_trace_check.csv", index=False)

    source_trace_ok = bool(
        not ablation_df.empty
        and (ablation_df["status"] == "ok").all()
        and (ablation_df["all_abs_diff"].fillna(1.0) < 1e-9).all()
        and (ablation_df["affected_abs_diff"].fillna(1.0) < 1e-9).all()
    )

    lines = [
        "# Experiment Integrity Audit",
        "",
        "## Main Result Trace",
        "",
        f"- main_output: `{main_output}`",
        f"- cache_path: `{cache_path}`",
        f"- normal_model_dir: `{config.get('normal_model_dir')}`",
        f"- normal_inference_scope: `{config.get('normal_inference_scope')}`",
        f"- epochs: {config.get('epochs')}",
        f"- max_train_samples: {config.get('max_train_samples')}",
        f"- use_normal_delta: {config.get('use_normal_delta')}",
        f"- use_normal_delta_abs: {config.get('use_normal_delta_abs')}",
        f"- use_dual_hist_residual: {config.get('use_dual_hist_residual')}",
        f"- use_temporal_decay_head: {config.get('use_temporal_decay_head')}",
        "",
        "## Split Counts",
        "",
        split_df.to_markdown(index=False),
        "",
        "## Main Metrics",
        "",
        pd.DataFrame(
            [
                {"split": "train", **train},
                {"split": "val", **val},
                {"split": "test", **test},
            ]
        ).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Training Log",
        "",
        pd.DataFrame([log_stats]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Normal Branch Test Metrics",
        "",
        normal_df.to_markdown(index=False, floatfmt=".4f") if not normal_df.empty else "No normal branch metrics found.",
        "",
        "## Table Trace Check",
        "",
        f"- ablation CSV rows match their referenced `metrics.json`: `{source_trace_ok}`",
        "",
        ablation_df[
            [
                "model",
                "status",
                "csv_all_model",
                "json_all_model",
                "csv_affected_model",
                "json_affected_model",
                "all_abs_diff",
                "affected_abs_diff",
            ]
        ].to_markdown(index=False, floatfmt=".10f"),
        "",
        "## Seed Robustness",
        "",
        seed_df.to_markdown(index=False, floatfmt=".4f") if not seed_df.empty else "No seed summary found.",
        "",
        "## Leakage And Reproducibility Notes",
        "",
        "- Primary forecast target is the normalized residual `actual_future - normal_pred`; this is a supervised target and is not present in model inputs.",
        "- Main cache uses time-based split codes: train = first 70%, val = next 15%, test = final 15% of the regional 2023 timeline.",
        "- Statistical normal baseline and blend alphas are rebuilt with `train_valid` only, where all timestamps after the 70% train boundary are masked out.",
        "- `global_context` contains incident metadata/time features, not derived severity/recovery/spread labels.",
        "- Future-derived impact labels (`event_aux`, `node_affected`, and heatmap labels) are stored in the cache. In the current main run, `event_aux_weight=0.05` and `node_aux_weight=0.03`, so these labels are used as auxiliary training supervision, not inference inputs.",
        "- The raw impact labels were generated by `build_impact_labels.py`; that standalone label-generation step builds its baseline over the full year. This is acceptable for exploratory labels, but for a stricter paper run we should either rebuild labels with train-only baselines per split or set auxiliary weights to zero and keep those labels only for reporting.",
        "- The final model was trained for 5 epochs and the validation loss was still improving, so the current result is a strong stage result but not a convergence claim.",
        "- The learned normal branch was evaluated in local candidate-subgraph mode for the incident cache. A full-region normal inference diagnostic exists, but the final table should clearly state which mode is used.",
        "",
        "## Cache Attributes",
        "",
        "```json",
        json.dumps(cache_attrs, indent=2, ensure_ascii=False, default=str),
        "```",
    ]

    report_path = out_dir / "audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    report_path = write_report(parse_args())
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
