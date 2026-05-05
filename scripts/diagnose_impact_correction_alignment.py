#!/usr/bin/env python3
"""Diagnose correction-target alignment for impact correction adapters."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/impact_correction_alignment"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-abs-target", type=float, default=1e-6)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def empty_sums() -> dict[str, float]:
    return {
        "count": 0.0,
        "active_count": 0.0,
        "target_abs_sum": 0.0,
        "raw_abs_sum": 0.0,
        "correction_abs_sum": 0.0,
        "source_abs_sum": 0.0,
        "corrected_abs_sum": 0.0,
        "improvement_sum": 0.0,
        "sign_match_count": 0.0,
        "beneficial_count": 0.0,
        "harmful_count": 0.0,
    }


def update_sums(
    sums: dict[str, float],
    target: torch.Tensor,
    raw_correction: torch.Tensor,
    correction: torch.Tensor,
    mask: torch.Tensor,
    min_abs_target: float,
) -> None:
    count = float(mask.sum().detach().cpu())
    if count <= 0.0:
        return
    target_masked = target[mask]
    raw_masked = raw_correction[mask]
    correction_masked = correction[mask]
    source_abs = torch.abs(target_masked)
    corrected_abs = torch.abs(target_masked - correction_masked)
    improvement = source_abs - corrected_abs
    active = (source_abs > min_abs_target) & (torch.abs(correction_masked) > 1e-8)
    sign_match = active & ((target_masked * correction_masked) > 0.0)
    beneficial = improvement > 0.0
    harmful = improvement < 0.0

    sums["count"] += count
    sums["active_count"] += float(active.sum().detach().cpu())
    sums["target_abs_sum"] += float(source_abs.sum().detach().cpu())
    sums["raw_abs_sum"] += float(torch.abs(raw_masked).sum().detach().cpu())
    sums["correction_abs_sum"] += float(torch.abs(correction_masked).sum().detach().cpu())
    sums["source_abs_sum"] += float(source_abs.sum().detach().cpu())
    sums["corrected_abs_sum"] += float(corrected_abs.sum().detach().cpu())
    sums["improvement_sum"] += float(improvement.sum().detach().cpu())
    sums["sign_match_count"] += float(sign_match.sum().detach().cpu())
    sums["beneficial_count"] += float(beneficial.sum().detach().cpu())
    sums["harmful_count"] += float(harmful.sum().detach().cpu())


def finalize(sums_by_key: dict[tuple[str, str], dict[str, float]], group_labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (group, target), sums in sums_by_key.items():
        count = max(sums["count"], 1.0)
        active_count = max(sums["active_count"], 1.0)
        rows.append(
            {
                "group": group,
                "label": group_labels[group],
                "target": target,
                "count": int(sums["count"]),
                "target_abs_mean": sums["target_abs_sum"] / count,
                "raw_abs_mean": sums["raw_abs_sum"] / count,
                "correction_abs_mean": sums["correction_abs_sum"] / count,
                "source_abs_mean": sums["source_abs_sum"] / count,
                "corrected_abs_mean": sums["corrected_abs_sum"] / count,
                "mean_improvement": sums["improvement_sum"] / count,
                "sign_match_rate": sums["sign_match_count"] / active_count,
                "beneficial_rate": sums["beneficial_count"] / count,
                "harmful_rate": sums["harmful_count"] / count,
            }
        )
    df = pd.DataFrame(rows)
    df["group_order"] = df["group"].map({group: idx for idx, group in enumerate(GROUP_ORDER)})
    df["target_order"] = df["target"].map({target: idx for idx, target in enumerate(TARGETS)})
    return df.sort_values(["group_order", "target_order"]).drop(columns=["group_order", "target_order"])


def main() -> None:
    args = parse_args()
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
    sums_by_key = {(group, target): empty_sums() for group in GROUP_ORDER for target in TARGETS}

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
            target_correction = y - source_pred
            raw_correction = details["raw_correction"]
            correction = details["correction"]

            for group in GROUP_ORDER:
                sample_mask_np = group_masks[group][batch_positions]
                if not sample_mask_np.any():
                    continue
                sample_mask = torch.as_tensor(sample_mask_np, device=device, dtype=torch.bool)[:, None, None, None]
                valid_mask = y_mask.bool() & sample_mask
                affected_mask = valid_mask & node_affected[:, None, :, None].bool()
                unaffected_mask = valid_mask & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
                for target_name, mask in {
                    "all": valid_mask,
                    "affected": affected_mask,
                    "unaffected": unaffected_mask,
                }.items():
                    update_sums(
                        sums_by_key[(group, target_name)],
                        target_correction,
                        raw_correction,
                        correction,
                        mask,
                        args.min_abs_target,
                    )

    df = finalize(sums_by_key, group_labels)
    df.to_csv(output_dir / "alignment_metrics.csv", index=False)
    focus = df[df["group"].isin(["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"])]
    cols = [
        "group",
        "target",
        "target_abs_mean",
        "correction_abs_mean",
        "mean_improvement",
        "sign_match_rate",
        "beneficial_rate",
        "harmful_rate",
    ]
    lines = [
        "# Impact Correction Alignment Diagnostics",
        "",
        f"- adapter_dir: `{adapter_dir}`",
        f"- split: `{args.split}`",
        "",
        focus[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter_dir": str(adapter_dir),
                "split": args.split,
                "batch_size": args.batch_size,
                "device": str(device),
                "min_abs_target": args.min_abs_target,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote alignment diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
