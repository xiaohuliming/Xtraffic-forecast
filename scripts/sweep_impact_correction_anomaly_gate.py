#!/usr/bin/env python3
"""Posthoc sweep anomaly-gate settings for an impact correction adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_dual_branch_group_metrics import read_event_groups
from evaluate_impact_correction_adapter import load_adapter
from train_full_candidate_stgnn_heatmap_model import CHANNELS, compute_stats, make_loader, split_indices
from train_impact_residual_model import choose_device


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

TARGETS = ["all", "affected", "unaffected"]


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/anomaly_gate_posthoc_sweep"),
    )
    parser.add_argument("--thresholds", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--floors", default="0.1,0.25,0.5")
    parser.add_argument("--temperatures", default="0.25")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def empty_sums() -> dict[str, float]:
    out = {}
    for target in TARGETS:
        out[f"{target}_source_sum"] = 0.0
        out[f"{target}_adapter_sum"] = 0.0
        out[f"{target}_count"] = 0.0
    return out


def add_metric_sums(
    sums: dict[str, float],
    source_abs: torch.Tensor,
    adapter_abs: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> None:
    for target, mask in masks.items():
        count = float(mask.sum().detach().cpu())
        if count <= 0.0:
            continue
        sums[f"{target}_source_sum"] += float(source_abs[mask].sum().detach().cpu())
        sums[f"{target}_adapter_sum"] += float(adapter_abs[mask].sum().detach().cpu())
        sums[f"{target}_count"] += count


def finalize_rows(
    sums_by_key: dict[tuple[float, float, float, str], dict[str, float]],
    group_labels: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for (threshold, floor, temperature, group), sums in sums_by_key.items():
        row: dict[str, float | str] = {
            "threshold": threshold,
            "floor": floor,
            "temperature": temperature,
            "group": group,
            "label": group_labels[group],
        }
        for target in TARGETS:
            count = max(sums[f"{target}_count"], 1.0)
            source_mae = sums[f"{target}_source_sum"] / count
            adapter_mae = sums[f"{target}_adapter_sum"] / count
            row[f"source_{target}_mae"] = source_mae
            row[f"adapter_{target}_mae"] = adapter_mae
            row[f"{target}_delta"] = adapter_mae - source_mae
        rows.append(row)
    df = pd.DataFrame(rows)
    df["group_order"] = df["group"].map({group: idx for idx, group in enumerate(GROUP_ORDER)})
    return df.sort_values(["threshold", "floor", "temperature", "group_order"]).drop(columns=["group_order"])


def main() -> None:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    floors = parse_float_list(args.floors)
    temperatures = parse_float_list(args.temperatures)
    combos = [(threshold, floor, temperature) for threshold in thresholds for floor in floors for temperature in temperatures]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.adapter_dir.resolve()
    device = choose_device(args.device)
    model, cache_path = load_adapter(adapter_dir, device)
    stats = compute_stats(cache_path)
    split_idx = split_indices(cache_path)[args.split]
    groups = read_event_groups(cache_path, split_idx)
    group_masks = {group: mask for group, (mask, _label) in groups.items()}
    group_labels = {group: label for group, (_mask, label) in groups.items()}

    sums_by_key = {
        (threshold, floor, temperature, group): empty_sums()
        for threshold, floor, temperature in combos
        for group in GROUP_ORDER
    }

    loader = make_loader(cache_path, split_idx, stats, batch_size=args.batch_size, shuffle=False)
    position = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (
                hist,
                hist_normal,
                node,
                global_context,
                normal_delta,
                y,
                y_mask,
                _impact,
                _impact_mask,
                _event_aux,
                node_affected,
                node_valid,
            ) = [item.to(device) for item in batch]
            if int(getattr(model, "hist_input_channels", len(CHANNELS))) > int(hist.shape[-1]):
                hist = torch.cat([hist, hist_normal], dim=-1)
            _pred_y, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            batch_size = int(y.shape[0])
            batch_positions = np.arange(position, position + batch_size)
            position += batch_size

            source_pred = details["source_pred"]
            source_abs = torch.abs(source_pred - y)
            raw_correction = details["raw_correction"]
            normal = model.base_beta * details["normal_residual"]
            incident = model.base_beta * details["incident_residual"]
            anomaly = torch.abs(incident - normal)

            for threshold, floor, temperature in combos:
                gate = floor + (1.0 - floor) * torch.sigmoid((anomaly - threshold) / max(temperature, 1e-6))
                adapter_abs = torch.abs(source_pred + raw_correction * gate - y)
                for group in GROUP_ORDER:
                    sample_mask_np = group_masks[group][batch_positions]
                    if not sample_mask_np.any():
                        continue
                    sample_mask = torch.as_tensor(sample_mask_np, device=device, dtype=torch.bool)[:, None, None, None]
                    valid_mask = y_mask.bool() & sample_mask
                    affected_mask = valid_mask & node_affected[:, None, :, None].bool()
                    unaffected_mask = valid_mask & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
                    add_metric_sums(
                        sums_by_key[(threshold, floor, temperature, group)],
                        source_abs,
                        adapter_abs,
                        {
                            "all": valid_mask,
                            "affected": affected_mask,
                            "unaffected": unaffected_mask,
                        },
                    )

    df = finalize_rows(sums_by_key, group_labels)
    df.to_csv(output_dir / "group_metrics.csv", index=False)
    focus = df[df["group"].isin(["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"])]
    focus_cols = [
        "threshold",
        "floor",
        "temperature",
        "group",
        "all_delta",
        "affected_delta",
        "unaffected_delta",
    ]
    lines = [
        "# Anomaly Gate Posthoc Sweep",
        "",
        f"- adapter_dir: `{adapter_dir}`",
        f"- split: `{args.split}`",
        "",
        focus[focus_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter_dir": str(adapter_dir),
                "split": args.split,
                "thresholds": thresholds,
                "floors": floors,
                "temperatures": temperatures,
                "batch_size": args.batch_size,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote anomaly gate sweep to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
