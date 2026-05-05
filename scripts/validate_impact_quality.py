#!/usr/bin/env python3
"""Validate impact-label quality with statistics and diagnostic plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-root",
        type=Path,
        default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"),
        help="Directory containing <Region>/event_labels.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_quality"),
        help="Directory for quality report and plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12000,
        help="Maximum points for scatter plots.",
    )
    return parser.parse_args()


def load_events(event_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(event_root.glob("*/event_labels.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No event_labels.csv files found under {event_root}")
    return pd.concat(frames, ignore_index=True)


def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    sub = df[[group_col, value_col]].dropna()
    if sub.empty:
        return float("nan")
    values = sub[value_col].to_numpy(float)
    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in sub.groupby(group_col):
        vals = group[value_col].to_numpy(float)
        between += len(vals) * (vals.mean() - values.mean()) ** 2
    return float(between / total)


def type_col(df: pd.DataFrame) -> str:
    return "primary_type" if "primary_type" in df.columns else "type"


def scatter_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=7)


def save_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("severity_any_z_auc_topk", "Severity robust z AUC"),
        ("recovery_time_min", "Recovery time (min)"),
        ("spread_nodes", "Spread nodes"),
        ("directionality", "Directionality"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for region, group in df.groupby("region"):
            vals = group[col].dropna()
            ax.hist(vals, bins=40, alpha=0.45, label=region)
        ax.set_title(title)
        ax.set_ylabel("events")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "label_distributions.png", dpi=180)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, out_dir: Path, sample_size: int) -> None:
    sub = scatter_sample(df, sample_size)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        sub["severity_any_z_auc_topk"],
        sub["recovery_time_min"],
        s=8,
        alpha=0.25,
    )
    axes[0].set_xlabel("severity_any_z_auc_topk")
    axes[0].set_ylabel("recovery_time_min")
    axes[0].set_title("Severity vs Recovery")

    axes[1].scatter(
        sub["severity_any_z_auc_topk"],
        sub["spread_nodes"],
        s=8,
        alpha=0.25,
    )
    axes[1].set_xlabel("severity_any_z_auc_topk")
    axes[1].set_ylabel("spread_nodes")
    axes[1].set_title("Severity vs Spread")
    fig.tight_layout()
    fig.savefig(out_dir / "severity_vs_recovery_spread.png", dpi=180)
    plt.close(fig)


def save_type_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    col = type_col(df)
    order = (
        df.groupby(col)["severity_any_z_auc_topk"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [df.loc[df[col] == name, "severity_any_z_auc_topk"].dropna().to_numpy() for name in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("severity_any_z_auc_topk")
    ax.set_title("Severity overlap across incident types")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "severity_by_type_boxplot.png", dpi=180)
    plt.close(fig)


def save_channel_corr(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "severity_flow_auc_topk",
        "severity_speed_auc_topk",
        "severity_occ_auc_topk",
        "severity_any_z_auc_topk",
        "recovery_time_min",
        "spread_nodes",
    ]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)), labels=cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels=cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlations among impact labels")
    fig.tight_layout()
    fig.savefig(out_dir / "impact_label_correlation.png", dpi=180)
    plt.close(fig)


def write_report(df: pd.DataFrame, event_root: Path, out_dir: Path) -> None:
    tcol = type_col(df)
    cols = [
        "severity_any_z_auc_topk",
        "severity_flow_auc_topk",
        "severity_speed_auc_topk",
        "severity_occ_auc_topk",
        "duration_min",
        "recovery_time_min",
        "spread_nodes",
        "spread_pm",
        "directionality",
    ]
    corr = df[cols].corr(method="spearman")
    corr_rows = []
    for col in cols:
        if col != "severity_any_z_auc_topk":
            corr_rows.append(
                {
                    "pair": f"severity_any_z_auc_topk vs {col}",
                    "spearman": corr.loc["severity_any_z_auc_topk", col],
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    eta_rows = []
    for target in ["severity_any_z_auc_topk", "recovery_time_min", "spread_nodes"]:
        eta_rows.append(
            {
                "target": target,
                f"eta_squared_by_{tcol}": eta_squared(df, tcol, target),
                "eta_squared_by_region": eta_squared(df, "region", target),
            }
        )
    eta_df = pd.DataFrame(eta_rows)

    region = (
        df.groupby("region")
        .agg(
            events=("region", "count"),
            severity_p50=("severity_any_z_auc_topk", "median"),
            severity_p90=("severity_any_z_auc_topk", lambda x: x.quantile(0.9)),
            recovery_p50=("recovery_time_min", "median"),
            spread_nodes_p50=("spread_nodes", "median"),
            upstream_rate=("directionality_class", lambda x: (x == "upstream_dominant").mean()),
            downstream_rate=("directionality_class", lambda x: (x == "downstream_dominant").mean()),
        )
        .reset_index()
    )

    by_type = (
        df.groupby(tcol)
        .agg(
            events=(tcol, "count"),
            severity_p50=("severity_any_z_auc_topk", "median"),
            severity_iqr=(
                "severity_any_z_auc_topk",
                lambda x: x.quantile(0.75) - x.quantile(0.25),
            ),
            recovery_p50=("recovery_time_min", "median"),
            spread_nodes_p50=("spread_nodes", "median"),
        )
        .sort_values("severity_p50", ascending=False)
        .reset_index()
    )

    lines = [
        "# Impact Label Quality Report",
        "",
        f"- event_root: `{event_root}`",
        f"- events: {len(df)}",
        f"- type column: `{tcol}`",
        "",
        "## Region Summary",
        "",
        region.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Spearman Correlations",
        "",
        corr_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Variance Explained",
        "",
        eta_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Low eta-squared for incident type supports the claim that type labels alone are weak impact supervision.",
        "",
        "## Impact By Type",
        "",
        by_type.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Plots",
        "",
        "- `label_distributions.png`",
        "- `severity_vs_recovery_spread.png`",
        "- `severity_by_type_boxplot.png`",
        "- `impact_label_correlation.png`",
        "",
    ]
    with (out_dir / "quality_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    event_root = args.event_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_events(event_root)
    save_histograms(df, out_dir)
    save_scatter(df, out_dir, args.sample_size)
    save_type_boxplot(df, out_dir)
    save_channel_corr(df, out_dir)
    write_report(df, event_root, out_dir)
    print(f"wrote {out_dir / 'quality_report.md'}", flush=True)


if __name__ == "__main__":
    main()
