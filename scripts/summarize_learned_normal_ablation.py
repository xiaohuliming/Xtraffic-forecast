#!/usr/bin/env python3
"""Summarize learned-normal residual ablations into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RUNS = [
    (
        "statistical normal + residual STGNN",
        "outputs/full_candidate_stgnn_heatmap_model/ablation_sigma_3_00_undirected/metrics.json",
        "statistical blend",
        "statistical residual",
    ),
    (
        "learned normal",
        "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal/metrics.json",
        "learned normal STGNN",
        "future residual target only",
    ),
    (
        "+ normal_delta",
        "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_delta/metrics.json",
        "learned normal STGNN",
        "normal_delta",
    ),
    (
        "+ dual historical residual",
        "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/metrics.json",
        "learned normal STGNN",
        "normal_delta + dual history",
    ),
    (
        "+ disagreement proxy",
        "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty/metrics.json",
        "learned normal STGNN",
        "normal_delta + abs(normal_delta) + dual history",
    ),
    (
        "+ temporal decay head",
        "outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/metrics.json",
        "learned normal STGNN",
        "normal_delta + abs(normal_delta) + dual history + temporal gate",
    ),
]


def load_test_row(name: str, path: Path, normal_branch: str, features: str) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload["metrics"]["test"]
    return {
        "model": name,
        "normal_branch": normal_branch,
        "incident_branch_inputs": features,
        "test_all_baseline_robust_mae": test["all_candidates_baseline_robust_mae"],
        "test_all_model_robust_mae": test["all_candidates_model_robust_mae"],
        "test_all_improvement_pct": test["all_candidates_improvement_pct"],
        "test_affected_baseline_robust_mae": test["affected_candidates_baseline_robust_mae"],
        "test_affected_model_robust_mae": test["affected_candidates_model_robust_mae"],
        "test_affected_improvement_pct": test["affected_candidates_improvement_pct"],
        "test_unaffected_model_robust_mae": test["unaffected_candidates_model_robust_mae"],
        "residual_beta": payload.get("residual_beta", float("nan")),
        "metrics_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/impact_guided_next_stage/ablation_summary"))
    args = parser.parse_args()

    rows = []
    for name, path_str, normal_branch, features in DEFAULT_RUNS:
        row = load_test_row(name, Path(path_str), normal_branch, features)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No metrics.json files found for the configured ablation runs.")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learned_normal_ablation_table.csv", index=False)

    display_cols = [
        "model",
        "incident_branch_inputs",
        "test_all_baseline_robust_mae",
        "test_all_model_robust_mae",
        "test_all_improvement_pct",
        "test_affected_baseline_robust_mae",
        "test_affected_model_robust_mae",
        "test_affected_improvement_pct",
    ]
    lines = ["# Learned Normal Ablation Table", ""]
    lines.append(df[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Notes:")
    lines.append("- robust MAE is computed on normalized residual space.")
    lines.append("- Optional enhancement rows are included only after their metrics files exist.")
    (out_dir / "learned_normal_ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'learned_normal_ablation_table.csv'}")
    print(f"wrote {out_dir / 'learned_normal_ablation_table.md'}")


if __name__ == "__main__":
    main()
