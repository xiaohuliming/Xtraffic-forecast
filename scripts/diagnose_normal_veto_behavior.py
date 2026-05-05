#!/usr/bin/env python3
"""Summarize how trained normal-veto heads change the fused proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from sweep_sttis_gate_posthoc import resolve_cache_path
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


SUBSETS = ("all", "affected", "unaffected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/normal_veto_behavior_summary.csv"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def empty_stats() -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for subset in SUBSETS:
        stats[subset] = {
            "count": 0.0,
            "normal_abs": 0.0,
            "incident_abs": 0.0,
            "base_fused_abs": 0.0,
            "final_abs": 0.0,
            "base_gate": 0.0,
            "effective_gate": 0.0,
            "normal_veto": 0.0,
            "normal_veto_amount": 0.0,
            "normal_advantage": 0.0,
            "normal_better_than_base": 0.0,
            "normal_better_than_incident": 0.0,
            "final_better_than_base": 0.0,
        }
    return stats


def add_subset_stats(
    stats: dict[str, dict[str, float]],
    subset: str,
    mask: torch.Tensor,
    values: dict[str, torch.Tensor],
) -> None:
    count = float(mask.sum().item())
    if count <= 0:
        return
    row = stats[subset]
    row["count"] += count
    for key, value in values.items():
        row[key] += float(value[mask].sum().detach().cpu())


def summarize_model(model_dir: Path, split: str, batch_size: int, device: torch.device) -> list[dict[str, float | str]]:
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    residual_beta = float(ckpt.get("residual_beta", 1.0))
    model = make_model(ckpt, cache_path, device)
    if not hasattr(model, "normal_veto_scale"):
        raise TypeError(f"{model_dir} is not a normal-veto checkpoint")

    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)[split]
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sums = empty_stats()
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
                _idx,
            ) = [item.to(device) for item in batch]
            if int(getattr(model, "hist_input_channels", hist.shape[-1])) > hist.shape[-1]:
                hist = torch.cat([hist, hist_normal], dim=-1)
            pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal_abs = torch.abs(residual_beta * details["normal_residual"] - y)
            incident_abs = torch.abs(residual_beta * details["incident_residual"] - y)
            base_fused_abs = torch.abs(residual_beta * details["base_fused_residual"] - y)
            final_abs = torch.abs(residual_beta * pred_y - y)
            values = {
                "normal_abs": normal_abs,
                "incident_abs": incident_abs,
                "base_fused_abs": base_fused_abs,
                "final_abs": final_abs,
                "base_gate": details["base_gate"],
                "effective_gate": details["gate"],
                "normal_veto": details["normal_veto"],
                "normal_veto_amount": details["normal_veto_amount"],
                "normal_advantage": base_fused_abs - normal_abs,
                "normal_better_than_base": (normal_abs < base_fused_abs).to(y.dtype),
                "normal_better_than_incident": (normal_abs < incident_abs).to(y.dtype),
                "final_better_than_base": (final_abs < base_fused_abs).to(y.dtype),
            }
            masks = {
                "all": y_mask.bool(),
                "affected": y_mask.bool() & node_affected[:, None, :, None].bool(),
                "unaffected": y_mask.bool()
                & (~node_affected[:, None, :, None].bool())
                & node_valid[:, None, :, None].bool(),
            }
            for subset, mask in masks.items():
                add_subset_stats(sums, subset, mask, values)

    rows: list[dict[str, float | str]] = []
    for subset, row in sums.items():
        count = max(row["count"], 1.0)
        rows.append(
            {
                "model_dir": str(model_dir),
                "split": split,
                "subset": subset,
                "count": row["count"],
                "normal_mae": row["normal_abs"] / count,
                "incident_mae": row["incident_abs"] / count,
                "base_fused_mae": row["base_fused_abs"] / count,
                "final_mae": row["final_abs"] / count,
                "base_gate_mean": row["base_gate"] / count,
                "effective_gate_mean": row["effective_gate"] / count,
                "normal_veto_mean": row["normal_veto"] / count,
                "normal_veto_amount_mean": row["normal_veto_amount"] / count,
                "normal_advantage_mean": row["normal_advantage"] / count,
                "normal_better_than_base_rate": row["normal_better_than_base"] / count,
                "normal_better_than_incident_rate": row["normal_better_than_incident"] / count,
                "final_better_than_base_rate": row["final_better_than_base"] / count,
                "residual_beta": residual_beta,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    rows: list[dict[str, float | str]] = []
    for model_dir in args.model_dirs:
        print(f"diagnosing {model_dir}", flush=True)
        rows.extend(summarize_model(model_dir.resolve(), args.split, args.batch_size, device))
    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    summary_path = args.output_csv.with_suffix(".md")
    lines = [
        "# Normal-veto behavior summary",
        "",
        df.to_markdown(index=False, floatfmt=".6f"),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
