#!/usr/bin/env python3
"""Aggregate raw XTraffic incident-record labels into event-level benchmarks.

The raw CHP incident table may contain multiple records for the same local
event. This script creates several transparent aggregation profiles so that we
can compare label behavior under different event definitions instead of hiding
the benchmark-construction choice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("Alameda", "ContraCosta", "Orange")

AREA_PROFILES = {
    # These are CHP areas that predominantly cover the selected county sensors.
    # We keep this profile separate from the all-match profile because CHP area
    # names do not perfectly coincide with county boundaries.
    "Alameda": {"Oakland", "Hayward", "Dublin", "Castro Valley"},
    "ContraCosta": {"Contra Costa"},
    "Orange": {
        "Santa Ana",
        "Orange County FSP",
        "Westminster",
        "Capistrano",
        "OCCC",
        "OCFSP",
        "OC",
    },
}

PROFILES = {
    "sensor_window_all": {
        "description": "All matched records grouped by region + start_idx + anchor_sensor_id.",
        "area_filter": False,
        "max_anchor_pm_dist": None,
        "require_positive_duration": False,
    },
    "region_area_sensor_window": {
        "description": "Area-filtered records grouped by region + start_idx + anchor_sensor_id.",
        "area_filter": True,
        "max_anchor_pm_dist": None,
        "require_positive_duration": False,
    },
    "region_area_close_sensor_window": {
        "description": "Area-filtered close-anchor positive-duration records grouped by region + start_idx + anchor_sensor_id.",
        "area_filter": True,
        "max_anchor_pm_dist": 0.5,
        "require_positive_duration": True,
    },
}


META_FIRST_COLS = [
    "region",
    "start_idx",
    "dt",
    "anchor_sensor_id",
    "anchor_region_idx",
    "fwy",
    "direction",
]

META_MEDIAN_COLS = [
    "incident_abs_pm",
    "latitude",
    "longitude",
    "anchor_pm_dist",
    "duration_min",
]

MAX_COLS = [
    "candidate_nodes",
    "severity_flow_auc_topk",
    "severity_speed_auc_topk",
    "severity_occ_auc_topk",
    "severity_flow_peak_topk",
    "severity_speed_peak_topk",
    "severity_occ_peak_topk",
    "severity_flow_z_auc_topk",
    "severity_speed_z_auc_topk",
    "severity_occ_z_auc_topk",
    "severity_any_z_auc_topk",
    "severity_any_z_peak_topk",
    "recovery_time_min",
    "recovery_censored",
    "time_to_peak_min",
    "spread_nodes",
    "spread_pm",
    "spread_radius_pm",
    "upstream_impact_z_auc_topk",
    "downstream_impact_z_auc_topk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("outputs/impact_labels"),
        help="Directory produced by build_impact_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_labels_aggregated"),
        help="Directory for aggregated outputs.",
    )
    parser.add_argument(
        "--write-node-labels",
        action="store_true",
        help="Also aggregate node_labels.csv for each profile. This is larger and slower.",
    )
    return parser.parse_args()


def mode_or_join(values: pd.Series, max_items: int = 8) -> str:
    items = [str(x) for x in values.dropna().tolist() if str(x) and str(x) != "nan"]
    if not items:
        return ""
    counts = Counter(items)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ";".join([name for name, _ in ordered[:max_items]])


def severity_class(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 1.0:
        return "weak"
    if value < 2.0:
        return "mild"
    if value < 3.0:
        return "moderate"
    return "severe"


def directionality_class(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value > 0.3:
        return "upstream_dominant"
    if value < -0.3:
        return "downstream_dominant"
    return "balanced"


def filter_profile(df: pd.DataFrame, region: str, profile: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    if profile["area_filter"]:
        out = out[out["area"].isin(AREA_PROFILES[region])]
    max_anchor = profile["max_anchor_pm_dist"]
    if max_anchor is not None:
        out = out[out["anchor_pm_dist"] <= float(max_anchor)]
    if profile["require_positive_duration"]:
        out = out[out["duration_min"] > 0]
    return out


def aggregate_events(df: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    df = df.copy()
    group_cols = ["region", "start_idx", "anchor_sensor_id"]
    df["agg_id"] = (
        profile_name
        + "__"
        + df["region"].astype(str)
        + "__"
        + df["start_idx"].astype(str)
        + "__"
        + df["anchor_sensor_id"].astype(str)
    )

    rows: list[dict[str, object]] = []
    for agg_id, group in df.groupby("agg_id", sort=False):
        # Use the strongest raw record as representative for directionality and
        # categorical classes, while max-pooling quantitative impact labels.
        idx = group["severity_any_z_auc_topk"].fillna(-np.inf).idxmax()
        strongest = group.loc[idx]
        row: dict[str, object] = {
            "agg_id": agg_id,
            "profile": profile_name,
            "raw_record_count": int(len(group)),
            "incident_ids": ";".join(group["incident_id"].astype(str).head(32)),
            "primary_type": mode_or_join(group["type"], max_items=1),
            "all_types": mode_or_join(group["type"]),
            "primary_area": mode_or_join(group["area"], max_items=1),
            "all_areas": mode_or_join(group["area"]),
            "primary_description": mode_or_join(group["description"], max_items=1),
        }
        for col in META_FIRST_COLS:
            row[col] = group[col].iloc[0]
        for col in META_MEDIAN_COLS:
            row[col] = group[col].median()
        for col in MAX_COLS:
            row[col] = group[col].max()

        row["directionality"] = strongest["directionality"]
        row["directionality_class"] = directionality_class(float(row["directionality"]))
        row["severity_class_z"] = severity_class(float(row["severity_any_z_auc_topk"]))
        rows.append(row)

    agg = pd.DataFrame(rows)
    agg = agg.sort_values(["region", "start_idx", "anchor_sensor_id"]).reset_index(drop=True)
    return agg


def aggregate_nodes(raw_nodes: pd.DataFrame, event_map: pd.DataFrame) -> pd.DataFrame:
    key = event_map[["region", "incident_id", "agg_id"]].copy()
    key["incident_id"] = key["incident_id"].astype(str)
    raw_nodes = raw_nodes.copy()
    raw_nodes["incident_id"] = raw_nodes["incident_id"].astype(str)
    merged = raw_nodes.merge(key, on=["region", "incident_id"], how="inner")
    if merged.empty:
        return merged

    group_cols = ["agg_id", "region", "sensor_id", "region_node_idx"]
    agg_dict = {
        "pm_dist": "min",
        "side": lambda x: mode_or_join(x, max_items=1),
        "affected": "max",
    }
    for col in [
        "flow_auc",
        "speed_auc",
        "occ_auc",
        "flow_peak",
        "speed_peak",
        "occ_peak",
        "flow_z_auc",
        "speed_z_auc",
        "occ_z_auc",
        "any_z_auc",
        "any_z_peak",
    ]:
        agg_dict[col] = "max"
    out = merged.groupby(group_cols, as_index=False).agg(agg_dict)
    return out


def summarize_profile(profile_dir: Path, profile_name: str, frames: list[pd.DataFrame]) -> None:
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lines: list[str] = [f"# Aggregation Profile: {profile_name}", ""]
    lines.append(PROFILES[profile_name]["description"])
    lines.append("")
    if df.empty:
        lines.append("No events.")
    else:
        summary = (
            df.groupby("region")
            .agg(
                events=("agg_id", "count"),
                raw_records=("raw_record_count", "sum"),
                raw_per_event=("raw_record_count", "mean"),
                severity_p50=("severity_any_z_auc_topk", "median"),
                severity_p90=("severity_any_z_auc_topk", lambda x: x.quantile(0.9)),
                recovery_p50=("recovery_time_min", "median"),
                spread_nodes_p50=("spread_nodes", "median"),
                close_anchor_p50=("anchor_pm_dist", "median"),
            )
            .reset_index()
        )
        lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
        lines.append("")
        lines.append("## Severity Class")
        lines.append("")
        counts = df.groupby(["region", "severity_class_z"]).size().reset_index(name="events")
        lines.append(counts.to_markdown(index=False))
    with (profile_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_global_summary(output_dir: Path, profile_summaries: list[dict[str, object]]) -> None:
    summary = pd.DataFrame(profile_summaries)
    lines = ["# Aggregated Impact Labels", ""]
    lines.append(summary.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## Profiles")
    lines.append("")
    for name, profile in PROFILES.items():
        lines.append(f"- `{name}`: {profile['description']}")
    lines.append("")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    label_dir = args.label_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_summaries: list[dict[str, object]] = []

    for profile_name, profile in PROFILES.items():
        profile_dir = output_dir / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_frames: list[pd.DataFrame] = []

        for region in REGIONS:
            event_path = label_dir / region / "event_labels.csv"
            raw = pd.read_csv(event_path, dtype={"incident_id": str})
            filtered = filter_profile(raw, region, profile)
            agg = aggregate_events(filtered, profile_name)
            region_dir = profile_dir / region
            region_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(region_dir / "event_labels.csv", index=False)
            profile_frames.append(agg)

            event_map = filtered[["region", "incident_id"]].copy()
            event_map["incident_id"] = event_map["incident_id"].astype(str)
            event_map["agg_id"] = (
                profile_name
                + "__"
                + filtered["region"].astype(str)
                + "__"
                + filtered["start_idx"].astype(str)
                + "__"
                + filtered["anchor_sensor_id"].astype(str)
            )
            if args.write_node_labels:
                node_path = label_dir / region / "node_labels.csv"
                raw_nodes = pd.read_csv(node_path, dtype={"incident_id": str})
                node_agg = aggregate_nodes(raw_nodes, event_map)
                node_agg.to_csv(region_dir / "node_labels.csv", index=False)

            profile_summaries.append(
                {
                    "profile": profile_name,
                    "region": region,
                    "raw_records_after_filter": int(len(filtered)),
                    "aggregated_events": int(len(agg)),
                    "raw_per_event": float(len(filtered) / max(len(agg), 1)),
                }
            )
            print(
                f"[{profile_name}/{region}] raw={len(filtered)} agg={len(agg)}",
                flush=True,
            )

        summarize_profile(profile_dir, profile_name, profile_frames)
        with (profile_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    write_global_summary(output_dir, profile_summaries)
    print(f"wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
