#!/usr/bin/env python3
"""Summarize impact-correction adapter candidates and recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RUNS = {
    7: "groupaware_impact_correction_adapter_tailtarget_seed_7/group_metrics.csv",
    11: "groupaware_impact_correction_adapter_tailtarget_seed_11/group_metrics.csv",
    23: "groupaware_impact_correction_adapter_tailtarget_smoke_seed_23/group_metrics.csv",
}

ANOMGATE_RUNS = {
    7: "groupaware_impact_correction_adapter_highfocus_anomgate05_seed_7/group_metrics.csv",
    11: "groupaware_impact_correction_adapter_highfocus_anomgate05_seed_11/group_metrics.csv",
    23: "groupaware_impact_correction_adapter_highfocus_anomgate05_smoke_seed_23/group_metrics.csv",
}

MAX05_RUNS = {
    7: "groupaware_impact_correction_adapter_max05_seed_7/full_eval/group_metrics.csv",
    11: "groupaware_impact_correction_adapter_max05_seed_11/full_eval/group_metrics.csv",
    23: "groupaware_impact_correction_adapter_max05_smoke_seed_23/full_eval/group_metrics.csv",
}

SEED23_VARIANTS = {
    "balanced_default": "groupaware_impact_correction_adapter_tailtarget_smoke_seed_23/group_metrics.csv",
    "anomgate05": "groupaware_impact_correction_adapter_highfocus_anomgate05_smoke_seed_23/group_metrics.csv",
    "max05_conservative": "groupaware_impact_correction_adapter_max05_smoke_seed_23/full_eval/group_metrics.csv",
    "highfocus_affected": "groupaware_impact_correction_adapter_highfocus_finalselect_smoke_seed_23/full_eval/group_metrics.csv",
    "max12_not_recommended": "groupaware_impact_correction_adapter_max12_smoke_seed_23/full_eval/group_metrics.csv",
}

GROUP_ORDER = [
    "overall",
    "severity_low",
    "severity_mid",
    "severity_high",
    "recovery_short_lt30",
    "recovery_mid_30_90",
    "recovery_long_ge90",
    "severity_high_and_long",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path("outputs/impact_guided_next_stage"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/impact_correction_final_candidates"),
    )
    return parser.parse_args()


def read_group_metrics(root: Path, rel_path: str) -> pd.DataFrame:
    path = root / rel_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def order_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order"] = out["group"].map({group: idx for idx, group in enumerate(GROUP_ORDER)})
    return out.sort_values(["order", "group"]).drop(columns=["order"])


def summarize_run_set(root: Path, run_paths: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed, rel_path in run_paths.items():
        df = read_group_metrics(root, rel_path)
        df = df.copy()
        df["seed"] = seed
        rows.append(df)
    per_seed = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "source_all_mae",
        "adapter_all_mae",
        "all_delta",
        "source_affected_mae",
        "adapter_affected_mae",
        "affected_delta",
        "source_unaffected_mae",
        "adapter_unaffected_mae",
        "unaffected_delta",
    ]
    mean_df = per_seed.groupby("group", as_index=False).agg(
        samples=("samples", "first"),
        **{col: (col, "mean") for col in numeric_cols},
    )
    return per_seed, order_groups(mean_df)


def summarize_default(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return summarize_run_set(root, DEFAULT_RUNS)


def summarize_anomgate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return summarize_run_set(root, ANOMGATE_RUNS)


def summarize_max05(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return summarize_run_set(root, MAX05_RUNS)


def summarize_seed23_variants(root: Path) -> pd.DataFrame:
    rows = []
    for variant, rel_path in SEED23_VARIANTS.items():
        df = read_group_metrics(root, rel_path)
        df = df.copy()
        df["variant"] = variant
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def variant_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = df.pivot(index="group", columns="variant", values=metric).reset_index()
    return order_groups(wide)


def candidate_table(
    default_mean: pd.DataFrame,
    anomgate_mean: pd.DataFrame,
    max05_mean: pd.DataFrame,
    seed23_variants: pd.DataFrame,
) -> pd.DataFrame:
    default_overall = default_mean[default_mean["group"] == "overall"].iloc[0]
    anomgate_overall = anomgate_mean[anomgate_mean["group"] == "overall"].iloc[0]
    anomgate = anomgate_mean[anomgate_mean["group"] == "severity_high_and_long"].iloc[0]
    max05_overall = max05_mean[max05_mean["group"] == "overall"].iloc[0]
    max05 = max05_mean[max05_mean["group"] == "severity_high_and_long"].iloc[0]
    highfocus = seed23_variants[
        (seed23_variants["variant"] == "highfocus_affected") & (seed23_variants["group"] == "severity_high_and_long")
    ].iloc[0]
    max12 = seed23_variants[
        (seed23_variants["variant"] == "max12_not_recommended") & (seed23_variants["group"] == "severity_high_and_long")
    ].iloc[0]
    rows = [
        {
            "candidate": "anomgate05",
            "role": "new all-oriented main candidate",
            "evidence_scope": "3-seed mean",
            "overall_all_delta": anomgate_overall["all_delta"],
            "overall_affected_delta": anomgate_overall["affected_delta"],
            "overall_unaffected_delta": anomgate_overall["unaffected_delta"],
            "high_and_long_all_delta": anomgate["all_delta"],
            "high_and_long_affected_delta": anomgate["affected_delta"],
            "high_and_long_unaffected_delta": anomgate["unaffected_delta"],
            "decision": "Promote as strongest all/unaffected candidate; document high-risk affected per-seed caveat.",
        },
        {
            "candidate": "balanced_default",
            "role": "previous balanced default",
            "evidence_scope": "3-seed mean",
            "overall_all_delta": default_overall["all_delta"],
            "overall_affected_delta": default_overall["affected_delta"],
            "overall_unaffected_delta": default_overall["unaffected_delta"],
            "high_and_long_all_delta": default_mean[default_mean["group"] == "severity_high_and_long"].iloc[0]["all_delta"],
            "high_and_long_affected_delta": default_mean[default_mean["group"] == "severity_high_and_long"].iloc[0]["affected_delta"],
            "high_and_long_unaffected_delta": default_mean[default_mean["group"] == "severity_high_and_long"].iloc[0]["unaffected_delta"],
            "decision": "Keep as simpler/stabler backup; weaker than anomgate on 3-seed mean.",
        },
        {
            "candidate": "max05_conservative",
            "role": "low-correction conservative ablation",
            "evidence_scope": "3-seed mean",
            "overall_all_delta": max05_overall["all_delta"],
            "overall_affected_delta": max05_overall["affected_delta"],
            "overall_unaffected_delta": max05_overall["unaffected_delta"],
            "high_and_long_all_delta": max05["all_delta"],
            "high_and_long_affected_delta": max05["affected_delta"],
            "high_and_long_unaffected_delta": max05["unaffected_delta"],
            "decision": "Stable but weaker than balanced; keep as ablation, not backup.",
        },
        {
            "candidate": "highfocus_affected",
            "role": "affected-oriented ablation",
            "evidence_scope": "seed23",
            "overall_all_delta": seed23_variants[
                (seed23_variants["variant"] == "highfocus_affected") & (seed23_variants["group"] == "overall")
            ].iloc[0]["all_delta"],
            "overall_affected_delta": seed23_variants[
                (seed23_variants["variant"] == "highfocus_affected") & (seed23_variants["group"] == "overall")
            ].iloc[0]["affected_delta"],
            "overall_unaffected_delta": seed23_variants[
                (seed23_variants["variant"] == "highfocus_affected") & (seed23_variants["group"] == "overall")
            ].iloc[0]["unaffected_delta"],
            "high_and_long_all_delta": highfocus["all_delta"],
            "high_and_long_affected_delta": highfocus["affected_delta"],
            "high_and_long_unaffected_delta": highfocus["unaffected_delta"],
            "decision": "Not default; use only to show affected-vs-unaffected tradeoff.",
        },
        {
            "candidate": "max12_not_recommended",
            "role": "rejected strong-correction ablation",
            "evidence_scope": "seed23",
            "overall_all_delta": seed23_variants[
                (seed23_variants["variant"] == "max12_not_recommended") & (seed23_variants["group"] == "overall")
            ].iloc[0]["all_delta"],
            "overall_affected_delta": seed23_variants[
                (seed23_variants["variant"] == "max12_not_recommended") & (seed23_variants["group"] == "overall")
            ].iloc[0]["affected_delta"],
            "overall_unaffected_delta": seed23_variants[
                (seed23_variants["variant"] == "max12_not_recommended") & (seed23_variants["group"] == "overall")
            ].iloc[0]["unaffected_delta"],
            "high_and_long_all_delta": max12["all_delta"],
            "high_and_long_affected_delta": max12["affected_delta"],
            "high_and_long_unaffected_delta": max12["unaffected_delta"],
            "decision": "Reject as default; high-risk all/unaffected regress.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    default_per_seed, default_mean = summarize_default(root)
    anomgate_per_seed, anomgate_mean = summarize_anomgate(root)
    max05_per_seed, max05_mean = summarize_max05(root)
    seed23_variants = summarize_seed23_variants(root)
    candidates = candidate_table(default_mean, anomgate_mean, max05_mean, seed23_variants)

    default_per_seed.to_csv(output_dir / "balanced_default_per_seed_group_metrics.csv", index=False)
    default_mean.to_csv(output_dir / "balanced_default_3seed_mean_group_metrics.csv", index=False)
    anomgate_per_seed.to_csv(output_dir / "anomgate05_per_seed_group_metrics.csv", index=False)
    anomgate_mean.to_csv(output_dir / "anomgate05_3seed_mean_group_metrics.csv", index=False)
    max05_per_seed.to_csv(output_dir / "max05_per_seed_group_metrics.csv", index=False)
    max05_mean.to_csv(output_dir / "max05_3seed_mean_group_metrics.csv", index=False)
    seed23_variants.to_csv(output_dir / "seed23_variant_group_metrics_long.csv", index=False)
    for metric in ["all_delta", "affected_delta", "unaffected_delta"]:
        variant_wide(seed23_variants, metric).to_csv(output_dir / f"seed23_variant_{metric}_wide.csv", index=False)
    candidates.to_csv(output_dir / "candidate_recommendations.csv", index=False)

    display_cols = [
        "candidate",
        "role",
        "evidence_scope",
        "overall_all_delta",
        "overall_affected_delta",
        "overall_unaffected_delta",
        "high_and_long_all_delta",
        "high_and_long_affected_delta",
        "high_and_long_unaffected_delta",
        "decision",
    ]
    lines = [
        "# Impact Correction Final Candidates",
        "",
        "Negative delta means the adapter is better than its source checkpoint.",
        "",
        "## Recommendation Table",
        "",
        candidates[display_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Anomgate05 3-Seed Mean",
        "",
        anomgate_mean[
            [
                "group",
                "samples",
                "all_delta",
                "affected_delta",
                "unaffected_delta",
                "source_affected_mae",
                "adapter_affected_mae",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Balanced Default 3-Seed Mean",
        "",
        default_mean[
            [
                "group",
                "samples",
                "all_delta",
                "affected_delta",
                "unaffected_delta",
                "source_affected_mae",
                "adapter_affected_mae",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Max05 Conservative 3-Seed Mean",
        "",
        max05_mean[
            [
                "group",
                "samples",
                "all_delta",
                "affected_delta",
                "unaffected_delta",
                "source_affected_mae",
                "adapter_affected_mae",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Seed23 Variant Affected Delta",
        "",
        variant_wide(seed23_variants, "affected_delta").to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Seed23 Variant Unaffected Delta",
        "",
        variant_wide(seed23_variants, "unaffected_delta").to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
