#!/usr/bin/env python3
"""Analyze generated XTraffic incident impact labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("outputs/impact_labels"),
        help="Directory produced by build_impact_labels.py.",
    )
    return parser.parse_args()


def load_events(label_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(label_dir.glob("*/event_labels.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No event_labels.csv files found under {label_dir}")
    return pd.concat(frames, ignore_index=True)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    sub = df[[group_col, value_col]].dropna()
    if sub.empty:
        return float("nan")
    values = sub[value_col].to_numpy(dtype=float)
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for _, group in sub.groupby(group_col):
        vals = group[value_col].to_numpy(dtype=float)
        ss_between += len(vals) * (vals.mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def spearman_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    corr = df[cols].corr(method="spearman")
    rows = []
    base = "severity_any_z_auc_topk"
    for col in cols:
        if col == base:
            continue
        rows.append({"pair": f"{base} vs {col}", "spearman": corr.loc[base, col]})
    return pd.DataFrame(rows)


def write_summary(label_dir: Path, df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Impact Label Analysis")
    lines.append("")
    lines.append(f"- total_events: {len(df)}")
    lines.append(f"- regions: {', '.join(sorted(df['region'].unique()))}")
    lines.append("")

    region_summary = (
        df.groupby("region")
        .agg(
            events=("incident_id", "count"),
            severity_p50=("severity_any_z_auc_topk", "median"),
            severity_p90=("severity_any_z_auc_topk", lambda x: x.quantile(0.9)),
            recovery_p50=("recovery_time_min", "median"),
            recovery_censored_rate=("recovery_censored", "mean"),
            spread_nodes_p50=("spread_nodes", "median"),
            upstream_dominant_rate=(
                "directionality_class",
                lambda x: (x == "upstream_dominant").mean(),
            ),
            downstream_dominant_rate=(
                "directionality_class",
                lambda x: (x == "downstream_dominant").mean(),
            ),
        )
        .reset_index()
    )
    lines.append("## Region Summary")
    lines.append("")
    lines.append(region_summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    cols = [
        "severity_any_z_auc_topk",
        "severity_flow_auc_topk",
        "severity_speed_auc_topk",
        "severity_occ_auc_topk",
        "duration_min",
        "recovery_time_min",
        "spread_nodes",
        "spread_pm",
    ]
    lines.append("## Spearman Correlations")
    lines.append("")
    lines.append(spearman_table(df, cols).to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    eta_rows = []
    for target in ["severity_any_z_auc_topk", "recovery_time_min", "spread_nodes"]:
        eta_rows.append(
            {
                "target": target,
                "eta_squared_by_type": eta_squared(df, "type", target),
                "eta_squared_by_region": eta_squared(df, "region", target),
                "eta_squared_by_area": eta_squared(df, "area", target),
            }
        )
    eta_df = pd.DataFrame(eta_rows)
    lines.append("## Label Variance Explained By Categorical Fields")
    lines.append("")
    lines.append(eta_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append(
        "Interpretation: low eta-squared for `type` means incident type alone "
        "does not explain much impact-label variance."
    )
    lines.append("")

    lines.append("## Severity By Incident Type")
    lines.append("")
    by_type = (
        df.groupby("type")
        .agg(
            events=("incident_id", "count"),
            duration_p50=("duration_min", "median"),
            severity_p50=("severity_any_z_auc_topk", "median"),
            severity_p90=("severity_any_z_auc_topk", lambda x: x.quantile(0.9)),
            recovery_p50=("recovery_time_min", "median"),
            spread_nodes_p50=("spread_nodes", "median"),
        )
        .sort_values("severity_p50", ascending=False)
        .reset_index()
    )
    lines.append(by_type.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## Top Areas Per Region")
    lines.append("")
    for region, region_df in df.groupby("region"):
        lines.append(f"### {region}")
        area_counts = region_df["area"].value_counts().head(12).reset_index()
        area_counts.columns = ["area", "events"]
        lines.append(area_counts.to_markdown(index=False))
        lines.append("")

    lines.append("## Most Severe Examples")
    lines.append("")
    severe_cols = [
        "region",
        "incident_id",
        "dt",
        "type",
        "area",
        "severity_any_z_auc_topk",
        "recovery_time_min",
        "spread_nodes",
        "directionality",
    ]
    severe = df.sort_values("severity_any_z_auc_topk", ascending=False).head(20)[severe_cols]
    lines.append(severe.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    out = label_dir / "analysis_summary.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    label_dir = args.label_dir.resolve()
    df = load_events(label_dir)
    write_summary(label_dir, df)


if __name__ == "__main__":
    main()
