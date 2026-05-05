#!/usr/bin/env python3
"""Plot traffic curves and spatial impact bars for selected incident cases."""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_impact_labels import load_region_traffic


REGIONS = ("Alameda", "ContraCosta", "Orange")
CHANNEL_NAMES = ("flow", "occupancy", "speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("archive"),
        help="Path to XTraffic archive.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("outputs/impact_labels"),
        help="Raw label directory produced by build_impact_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_case_plots"),
        help="Directory for case figures.",
    )
    parser.add_argument("--pre-steps", type=int, default=12)
    parser.add_argument("--post-steps", type=int, default=36)
    parser.add_argument("--top-nodes", type=int, default=12)
    return parser.parse_args()


def sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:120]


def select_cases(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in events.groupby("region"):
        valid = group.dropna(subset=["severity_any_z_auc_topk"])
        if valid.empty:
            continue

        rows.append(("top_severity", valid.sort_values("severity_any_z_auc_topk", ascending=False).iloc[0]))

        p75 = valid["severity_any_z_auc_topk"].quantile(0.75)
        strong = valid[valid["severity_any_z_auc_topk"] >= p75]
        if not strong.empty:
            up = strong.dropna(subset=["directionality"])
            if not up.empty:
                rows.append(("upstream_dominant", up.sort_values("directionality", ascending=False).iloc[0]))
                rows.append(("downstream_dominant", up.sort_values("directionality", ascending=True).iloc[0]))

        spread = valid.sort_values(["spread_nodes", "severity_any_z_auc_topk"], ascending=False)
        rows.append(("high_spread", spread.iloc[0]))

        mild = valid.iloc[(valid["severity_any_z_auc_topk"] - valid["severity_any_z_auc_topk"].median()).abs().argsort()[:1]]
        if len(mild):
            rows.append(("median_case", mild.iloc[0]))

    out = []
    seen = set()
    for case_type, row in rows:
        key = (row["region"], str(row["incident_id"]))
        if (case_type, key) in seen:
            continue
        seen.add((case_type, key))
        item = row.to_dict()
        item["case_type"] = case_type
        out.append(item)
    return pd.DataFrame(out)


def same_slot_baseline(
    traffic: np.ndarray,
    times: pd.DatetimeIndex,
    sensor_local_idx: int,
    target_indices: np.ndarray,
    event_start: int,
) -> np.ndarray:
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    baseline = np.full((len(target_indices), 3), np.nan, dtype=np.float32)

    for pos, idx in enumerate(target_indices):
        if idx < 0 or idx >= len(times):
            continue
        mask = (day_kind == day_kind[idx]) & (tod == tod[idx])
        # Exclude the event day and adjacent day to avoid leaking the plotted
        # incident into the reference curve.
        mask &= np.abs(np.arange(len(times)) - event_start) > 288
        vals = traffic[mask, sensor_local_idx, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            baseline[pos] = np.nanmedian(vals, axis=0)
    return baseline


def plot_case(
    case: pd.Series,
    traffic: np.ndarray,
    times: pd.DatetimeIndex,
    node_labels: pd.DataFrame,
    out_dir: Path,
    pre_steps: int,
    post_steps: int,
    top_nodes: int,
) -> Path | None:
    incident_id = str(case["incident_id"])
    case_nodes = node_labels[node_labels["incident_id"].astype(str) == incident_id].copy()
    if case_nodes.empty:
        return None

    case_nodes = case_nodes.sort_values("any_z_auc", ascending=False)
    start = int(case["start_idx"])
    lo = max(0, start - pre_steps)
    hi = min(traffic.shape[0], start + post_steps)
    idx = np.arange(lo, hi)
    rel_min = (idx - start) * 5

    min_obs = min(10, max(4, int(math.ceil(len(idx) * 0.35))))
    top_node = None
    for _, candidate in case_nodes.head(60).iterrows():
        candidate_idx = int(candidate["region_node_idx"])
        candidate_actual = traffic[idx, candidate_idx, :]
        flow_ok = np.isfinite(candidate_actual[:, 0]).sum() >= min_obs
        speed_ok = np.isfinite(candidate_actual[:, 2]).sum() >= min_obs
        occ_ok = np.isfinite(candidate_actual[:, 1]).sum() >= min_obs
        if flow_ok and (speed_ok or occ_ok):
            top_node = candidate
            break
    if top_node is None:
        top_node = case_nodes.iloc[0]

    sensor_idx = int(top_node["region_node_idx"])

    actual = traffic[idx, sensor_idx, :]
    baseline = same_slot_baseline(traffic, times, sensor_idx, idx, start)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.15], hspace=0.42)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    for channel, ax in zip([0, 2, 1], axes[:3]):
        ax.plot(rel_min, actual[:, channel], label="actual", color="#1f77b4", linewidth=1.8)
        ax.plot(rel_min, baseline[:, channel], label="same-slot median baseline", color="#d62728", linestyle="--")
        ax.axvline(0, color="black", linewidth=1.1)
        ax.set_ylabel(CHANNEL_NAMES[channel])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    bar_nodes = case_nodes.head(top_nodes).copy()
    sign = bar_nodes["side"].map({"upstream": -1, "downstream": 1, "at_incident": 0}).fillna(0)
    signed_dist = bar_nodes["pm_dist"].astype(float) * sign.astype(float)
    colors = bar_nodes["side"].map(
        {"upstream": "#d62728", "downstream": "#2ca02c", "at_incident": "#7f7f7f"}
    ).fillna("#7f7f7f")
    axes[3].bar(np.arange(len(bar_nodes)), bar_nodes["any_z_auc"], color=colors)
    axes[3].set_xticks(
        np.arange(len(bar_nodes)),
        [f"{d:.2f}" for d in signed_dist],
        rotation=35,
        ha="right",
    )
    axes[3].set_ylabel("node any_z_auc")
    axes[3].set_xlabel("signed PM distance (upstream negative, downstream positive)")
    axes[3].grid(axis="y", alpha=0.25)

    title = (
        f"{case['case_type']} | {case['region']} | incident {incident_id} | "
        f"{case['dt']} | type={case['type']} | severity={case['severity_any_z_auc_topk']:.2f} | "
        f"recovery={case['recovery_time_min']:.0f}m | spread={case['spread_nodes']}"
    )
    fig.suptitle(title, fontsize=11)
    out_path = out_dir / f"{sanitize(case['region'] + '_' + case['case_type'] + '_' + incident_id)}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    label_dir = args.label_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = []
    for region in REGIONS:
        events.append(pd.read_csv(label_dir / region / "event_labels.csv", dtype={"incident_id": str}))
    event_df = pd.concat(events, ignore_index=True)
    cases = select_cases(event_df)
    cases.to_csv(out_dir / "selected_cases.csv", index=False)

    plot_rows = []
    for region in REGIONS:
        region_cases = cases[cases["region"] == region]
        if region_cases.empty:
            continue
        sensors = pd.read_csv(label_dir / region / "region_sensors.csv")
        region_node_idx = sensors["node_idx"].to_numpy(dtype=np.int32)
        print(f"[{region}] loading traffic", flush=True)
        traffic, times = load_region_traffic(data_dir, region_node_idx)
        node_labels = pd.read_csv(label_dir / region / "node_labels.csv", dtype={"incident_id": str})

        for _, case in region_cases.iterrows():
            path = plot_case(
                case=case,
                traffic=traffic,
                times=times,
                node_labels=node_labels,
                out_dir=out_dir,
                pre_steps=args.pre_steps,
                post_steps=args.post_steps,
                top_nodes=args.top_nodes,
            )
            if path:
                plot_rows.append(
                    {
                        "region": region,
                        "incident_id": case["incident_id"],
                        "case_type": case["case_type"],
                        "plot": str(path),
                    }
                )

    pd.DataFrame(plot_rows).to_csv(out_dir / "plot_index.csv", index=False)
    lines = ["# Impact Case Plots", ""]
    for row in plot_rows:
        lines.append(
            f"- {row['region']} {row['case_type']} incident {row['incident_id']}: `{row['plot']}`"
        )
    lines.append("")
    with (out_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
