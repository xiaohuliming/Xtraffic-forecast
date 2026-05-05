#!/usr/bin/env python3
"""Diagnose localization signals used by impact correction adapters."""

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

SIGNALS = [
    "node_prob",
    "branch_delta_abs",
    "source_abs",
    "correction_abs",
    "correction_signed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/impact_correction_signal_diagnostics"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def signal_stats(values: pd.Series) -> dict[str, float | int]:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "q50": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
        }
    q10, q25, q50, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "q10": float(q10),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "q90": float(q90),
    }


def collect_node_signals(adapter_dir: Path, split: str, batch_size: int, device: torch.device) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    model, cache_path = load_adapter(adapter_dir, device)
    stats = compute_stats(cache_path)
    split_idx = split_indices(cache_path)[split]
    groups = read_event_groups(cache_path, split_idx)
    loader = make_loader(cache_path, split_idx, stats, batch_size=batch_size, shuffle=False)
    rows = []
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
                _y,
                _y_mask,
                _impact,
                _impact_mask,
                _event_aux,
                node_affected,
                node_valid,
            ) = [item.to(device) for item in batch]
            if int(getattr(model, "hist_input_channels", len(CHANNELS))) > int(hist.shape[-1]):
                hist = torch.cat([hist, hist_normal], dim=-1)
            _pred_y, _pred_impact, _pred_event_aux, pred_node_logits, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            source_pred = details["source_pred"]
            normal = model.base_beta * details["normal_residual"]
            incident = model.base_beta * details["incident_residual"]
            correction = details["correction"]

            node_prob = torch.sigmoid(pred_node_logits)
            branch_delta_abs = torch.abs(incident - normal).mean(dim=(1, 3))
            source_abs = torch.abs(source_pred).mean(dim=(1, 3))
            correction_abs = torch.abs(correction).mean(dim=(1, 3))
            correction_signed = correction.mean(dim=(1, 3))
            valid = node_valid.bool()
            affected = node_affected.bool()
            batch_size_actual = int(valid.shape[0])
            sample_positions = np.arange(position, position + batch_size_actual, dtype=np.int64)
            position += batch_size_actual

            tensors = {
                "node_prob": node_prob,
                "branch_delta_abs": branch_delta_abs,
                "source_abs": source_abs,
                "correction_abs": correction_abs,
                "correction_signed": correction_signed,
            }
            flat_valid = valid.detach().cpu().numpy().reshape(-1)
            flat_affected = affected.detach().cpu().numpy().reshape(-1)
            node_count = int(valid.shape[1])
            base = {
                "sample_position": np.repeat(sample_positions, node_count)[flat_valid],
                "node_type": np.where(flat_affected[flat_valid], "affected", "unaffected"),
            }
            frame = pd.DataFrame(base)
            for name, tensor in tensors.items():
                frame[name] = tensor.detach().cpu().numpy().reshape(-1)[flat_valid].astype(np.float64)
            rows.append(frame)
    group_masks = {group: mask for group, (mask, _label) in groups.items()}
    group_labels = {group: label for group, (_mask, label) in groups.items()}
    return pd.concat(rows, ignore_index=True), group_masks, group_labels


def summarize(df: pd.DataFrame, group_masks: dict[str, np.ndarray], group_labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        group_df = df[group_masks[group][df["sample_position"].to_numpy(dtype=np.int64)]]
        for node_type in ["affected", "unaffected"]:
            part = group_df[group_df["node_type"] == node_type]
            for signal in SIGNALS:
                rows.append(
                    {
                        "group": group,
                        "group_label": group_labels[group],
                        "node_type": node_type,
                        "signal": signal,
                        **signal_stats(part[signal]),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, stats: pd.DataFrame, adapter_dir: Path) -> None:
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"]
    focus_signals = ["node_prob", "branch_delta_abs", "correction_abs"]
    focus = stats[stats["group"].isin(focus_groups) & stats["signal"].isin(focus_signals)]
    cols = ["group", "node_type", "signal", "count", "mean", "q10", "q25", "q50", "q75", "q90"]
    lines = [
        "# Impact Correction Signal Diagnostics",
        "",
        f"- adapter_dir: `{adapter_dir}`",
        "",
        "Signals are node-level averages over forecast horizon and channels where applicable.",
        "",
        focus[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.adapter_dir.resolve()
    device = choose_device(args.device)
    node_df, group_masks, group_labels = collect_node_signals(adapter_dir, args.split, args.batch_size, device)
    stats = summarize(node_df, group_masks, group_labels)
    node_df.to_csv(output_dir / "node_signal_values.csv", index=False)
    stats.to_csv(output_dir / "node_signal_stats.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "adapter_dir": str(adapter_dir),
                "split": args.split,
                "batch_size": args.batch_size,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, stats, adapter_dir)
    print(f"wrote signal diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
