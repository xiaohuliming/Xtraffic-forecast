#!/usr/bin/env python3
"""Summarize seed robustness for learned-normal model variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RUNS_BY_VARIANT = {
    "uncertainty": [
        (7, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty/metrics.json")),
        (11, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty_seed_11/metrics.json")),
        (23, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty_seed_23/metrics.json")),
    ],
    "decay": [
        (7, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/metrics.json")),
        (11, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay_seed_11/metrics.json")),
        (23, Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay_seed_23/metrics.json")),
    ],
}


def load_row(seed: int, path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload["metrics"]["test"]
    val = payload["metrics"]["val"]
    return {
        "seed": seed,
        "test_all_model_robust_mae": test["all_candidates_model_robust_mae"],
        "test_all_baseline_robust_mae": test["all_candidates_baseline_robust_mae"],
        "test_all_improvement_pct": test["all_candidates_improvement_pct"],
        "test_affected_model_robust_mae": test["affected_candidates_model_robust_mae"],
        "test_affected_baseline_robust_mae": test["affected_candidates_baseline_robust_mae"],
        "test_affected_improvement_pct": test["affected_candidates_improvement_pct"],
        "test_unaffected_model_robust_mae": test["unaffected_candidates_model_robust_mae"],
        "val_all_model_robust_mae": val["all_candidates_model_robust_mae"],
        "val_affected_model_robust_mae": val["affected_candidates_model_robust_mae"],
        "horizon_06_test_all_model_robust_mae": test.get("horizon_06_all_candidates_model_robust_mae", float("nan")),
        "horizon_06_test_affected_model_robust_mae": test.get("horizon_06_affected_candidates_model_robust_mae", float("nan")),
        "horizon_12_test_all_model_robust_mae": test.get("horizon_12_all_candidates_model_robust_mae", float("nan")),
        "horizon_12_test_affected_model_robust_mae": test.get("horizon_12_affected_candidates_model_robust_mae", float("nan")),
        "residual_beta": payload.get("residual_beta", float("nan")),
        "metrics_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(RUNS_BY_VARIANT), default="uncertainty")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for seed, path in RUNS_BY_VARIANT[args.variant] if (row := load_row(seed, path)) is not None]
    if not rows:
        raise FileNotFoundError("No seed robustness metrics found.")

    default_output = Path(f"outputs/impact_guided_next_stage/{args.variant}_seed_robustness")
    if args.variant == "uncertainty":
        default_output = Path("outputs/impact_guided_next_stage/seed_robustness")
    out_dir = (args.output_dir or default_output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("seed")
    metric_cols = [
        "test_all_model_robust_mae",
        "test_affected_model_robust_mae",
        "test_unaffected_model_robust_mae",
        "test_all_improvement_pct",
        "test_affected_improvement_pct",
    ]
    optional_metric_cols = [
        "horizon_06_test_affected_model_robust_mae",
        "horizon_12_test_affected_model_robust_mae",
    ]
    metric_cols.extend([col for col in optional_metric_cols if df[col].notna().any()])
    summary = df[metric_cols].agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})

    df.to_csv(out_dir / "seed_robustness_runs.csv", index=False)
    summary.to_csv(out_dir / "seed_robustness_summary.csv", index=False)

    display_cols = [
        "seed",
        "test_all_model_robust_mae",
        "test_affected_model_robust_mae",
        "test_unaffected_model_robust_mae",
        "residual_beta",
    ]
    display_cols[4:4] = [col for col in optional_metric_cols if df[col].notna().any()]
    lines = [f"# Seed Robustness: {args.variant}", ""]
    lines.append("## Runs")
    lines.append("")
    lines.append(df[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    (out_dir / "seed_robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {out_dir / 'seed_robustness_runs.csv'}")
    print(f"wrote {out_dir / 'seed_robustness_summary.csv'}")
    print(f"wrote {out_dir / 'seed_robustness_summary.md'}")


if __name__ == "__main__":
    main()
