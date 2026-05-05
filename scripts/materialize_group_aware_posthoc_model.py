#!/usr/bin/env python3
"""Write a standard model directory for a selected group-aware posthoc config."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def robust_improvement(model_mae: float, baseline_mae: float) -> float:
    if baseline_mae == 0.0:
        return 0.0
    return 100.0 * (baseline_mae - model_mae) / baseline_mae


def main() -> None:
    args = parse_args()
    source_dir = args.source_model_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads((selection_dir / "selected_config.json").read_text(encoding="utf-8"))
    source_metrics = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))
    test_groups = pd.read_csv(selection_dir / "test_group_metrics.csv")
    overall = test_groups[test_groups["group"] == "overall"]
    if overall.empty:
        raise ValueError(f"missing overall row in {selection_dir / 'test_group_metrics.csv'}")
    overall_row = overall.iloc[0]

    ckpt = torch.load(source_dir / "model.pt", map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    ckpt_args = dict(ckpt_args)
    ckpt_args["normal_veto_scale"] = float(selected["normal_veto_scale"])
    ckpt_args["normal_veto_temperature"] = float(selected["normal_veto_temperature"])
    ckpt_args["residual_beta"] = float(selected["residual_beta"])
    ckpt_args["posthoc_selection_variant"] = "group_aware_severity_recovery"
    ckpt_args["posthoc_selection_dir"] = str(selection_dir)
    ckpt["args"] = ckpt_args
    ckpt["residual_beta"] = float(selected["residual_beta"])
    torch.save(ckpt, output_dir / "model.pt")

    source_test = source_metrics.get("metrics", {}).get("test", {})
    all_model = float(overall_row["all_candidates_model_robust_mae"])
    affected_model = float(overall_row["affected_candidates_model_robust_mae"])
    unaffected_model = float(overall_row["unaffected_candidates_model_robust_mae"])
    all_baseline = float(source_test.get("all_candidates_baseline_robust_mae", 0.0))
    affected_baseline = float(source_test.get("affected_candidates_baseline_robust_mae", 0.0))
    unaffected_baseline = float(source_test.get("unaffected_candidates_baseline_robust_mae", 0.0))
    metrics = {
        "test": {
            "all_candidates_model_robust_mae": all_model,
            "all_candidates_baseline_robust_mae": all_baseline,
            "all_candidates_improvement_pct": robust_improvement(all_model, all_baseline),
            "affected_candidates_model_robust_mae": affected_model,
            "affected_candidates_baseline_robust_mae": affected_baseline,
            "affected_candidates_improvement_pct": robust_improvement(affected_model, affected_baseline),
            "unaffected_candidates_model_robust_mae": unaffected_model,
            "unaffected_candidates_baseline_robust_mae": unaffected_baseline,
            "unaffected_candidates_improvement_pct": robust_improvement(unaffected_model, unaffected_baseline),
        }
    }
    payload = {
        "metrics": metrics,
        "samples": source_metrics.get("samples", {}),
        "eval_samples": source_metrics.get("eval_samples", {}),
        "normal_veto_scale": float(selected["normal_veto_scale"]),
        "normal_veto_temperature": float(selected["normal_veto_temperature"]),
        "residual_beta": float(selected["residual_beta"]),
        "cache_path": selected["cache_path"],
        "source_model_dir": str(source_dir),
        "selection_dir": str(selection_dir),
        "selection_groups": selected["selection_groups"],
        "selection_weights": selected["selection_weights"],
        "selection_group_score": float(selected["group_score"]),
        "selection_overall_all_mae": float(selected["overall_all_mae"]),
        "selection_overall_affected_mae": float(selected["overall_affected_mae"]),
        "selection_overall_unaffected_mae": float(selected["overall_unaffected_mae"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for filename in [
        "selected_config.json",
        "test_group_metrics.csv",
        "val_group_sweep.csv",
        "val_selection_scores.csv",
        "summary.md",
    ]:
        src = selection_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)

    config = {
        "source_model_dir": str(source_dir),
        "selection_dir": str(selection_dir),
        "normal_veto_scale": float(selected["normal_veto_scale"]),
        "normal_veto_temperature": float(selected["normal_veto_temperature"]),
        "residual_beta": float(selected["residual_beta"]),
        "posthoc_selection_variant": "group_aware_severity_recovery",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote materialized posthoc model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
