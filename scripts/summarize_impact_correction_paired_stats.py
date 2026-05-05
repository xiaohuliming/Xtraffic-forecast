#!/usr/bin/env python3
"""Compute paired sample-level statistics for impact correction adapters."""

from __future__ import annotations

import argparse
import json
import math
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

TARGET_ORDER = ["all", "affected", "unaffected"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels aligned with --adapter-dirs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/impact_correction_paired_stats"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def sample_mae(abs_error: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    batch = int(abs_error.shape[0])
    flat_error = abs_error.reshape(batch, -1)
    flat_mask = mask.reshape(batch, -1).to(abs_error.dtype)
    counts = flat_mask.sum(dim=1)
    sums = (flat_error * flat_mask).sum(dim=1)
    values = sums / counts.clamp_min(1.0)
    out = values.detach().cpu().numpy().astype(np.float64)
    count_np = counts.detach().cpu().numpy().astype(np.float64)
    out[count_np <= 0] = np.nan
    return out, count_np


def normal_approx_p(mean: float, se: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0.0:
        return float("nan")
    z = abs(mean) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def paired_stats(values: pd.Series) -> dict[str, float | int]:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "samples": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "std_delta": float("nan"),
            "se_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "improve_rate": float("nan"),
            "worse_rate": float("nan"),
            "p_normal_approx": float("nan"),
        }
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "samples": n,
        "mean_delta": mean,
        "median_delta": median,
        "std_delta": std,
        "se_delta": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "improve_rate": float((arr < 0.0).mean()),
        "worse_rate": float((arr > 0.0).mean()),
        "p_normal_approx": normal_approx_p(mean, se),
    }


def collect_sample_rows(
    adapter_dir: Path,
    label: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    model, cache_path = load_adapter(adapter_dir, device)
    stats = compute_stats(cache_path)
    split_idx = split_indices(cache_path)[split]
    loader = make_loader(cache_path, split_idx, stats, batch_size=batch_size, shuffle=False)
    groups = read_event_groups(cache_path, split_idx)

    rows: list[dict[str, float | int | str]] = []
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
            adapter_pred, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            source_pred = details["source_pred"]
            adapter_abs = torch.abs(adapter_pred - y)
            source_abs = torch.abs(source_pred - y)
            all_mask = y_mask.bool()
            affected_mask = all_mask & node_affected[:, None, :, None].bool()
            unaffected_mask = all_mask & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
            masks = {
                "all": all_mask,
                "affected": affected_mask,
                "unaffected": unaffected_mask,
            }
            batch_size_actual = int(y.shape[0])
            batch_indices = split_idx[position : position + batch_size_actual]
            position += batch_size_actual

            per_target: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for target, mask in masks.items():
                adapter_mae, counts = sample_mae(adapter_abs, mask)
                source_mae, _ = sample_mae(source_abs, mask)
                per_target[target] = (source_mae, adapter_mae, adapter_mae - source_mae, counts)

            for row_idx, sample_index in enumerate(batch_indices):
                row: dict[str, float | int | str] = {
                    "label": label,
                    "sample_position": int(position - batch_size_actual + row_idx),
                    "sample_index": int(sample_index),
                }
                for target in TARGET_ORDER:
                    source_mae, adapter_mae, delta, counts = per_target[target]
                    row[f"{target}_source_mae"] = float(source_mae[row_idx])
                    row[f"{target}_adapter_mae"] = float(adapter_mae[row_idx])
                    row[f"{target}_delta"] = float(delta[row_idx])
                    row[f"{target}_count"] = float(counts[row_idx])
                rows.append(row)

    group_masks = {group: mask for group, (mask, _label) in groups.items()}
    group_labels = {group: group_label for group, (_mask, group_label) in groups.items()}
    return pd.DataFrame(rows), group_masks, group_labels


def summarize_samples(samples: pd.DataFrame, group_masks: dict[str, np.ndarray], group_labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        mask = group_masks[group]
        group_df = samples.loc[mask].reset_index(drop=True)
        for target in TARGET_ORDER:
            stats = paired_stats(group_df[f"{target}_delta"])
            rows.append(
                {
                    "group": group,
                    "group_label": group_labels[group],
                    "target": target,
                    **stats,
                    "source_sample_mae": float(group_df[f"{target}_source_mae"].mean(skipna=True)),
                    "adapter_sample_mae": float(group_df[f"{target}_adapter_mae"].mean(skipna=True)),
                }
            )
    return pd.DataFrame(rows)


def summarize_seed_means(per_seed_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, target), part in per_seed_stats.groupby(["group", "target"], sort=False):
        rows.append(
            {
                "group": group,
                "target": target,
                "seeds": int(part["label"].nunique()),
                "seed_mean_delta": float(part["mean_delta"].mean()),
                "seed_std_delta": float(part["mean_delta"].std(ddof=1)) if part.shape[0] > 1 else 0.0,
                "seed_min_delta": float(part["mean_delta"].min()),
                "seed_max_delta": float(part["mean_delta"].max()),
                "all_seeds_improved": bool((part["mean_delta"] < 0.0).all()),
                "mean_improve_rate": float(part["improve_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def ordered_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group_order"] = out["group"].map({group: idx for idx, group in enumerate(GROUP_ORDER)})
    out["target_order"] = out["target"].map({target: idx for idx, target in enumerate(TARGET_ORDER)})
    return out.sort_values(["group_order", "target_order"]).drop(columns=["group_order", "target_order"])


def write_summary(output_dir: Path, per_seed: pd.DataFrame, seed_mean: pd.DataFrame) -> None:
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"]
    focus_targets = ["all", "affected", "unaffected"]
    focus_seed = per_seed[per_seed["group"].isin(focus_groups) & per_seed["target"].isin(focus_targets)]
    focus_mean = seed_mean[seed_mean["group"].isin(focus_groups) & seed_mean["target"].isin(focus_targets)]
    seed_cols = [
        "label",
        "group",
        "target",
        "samples",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "improve_rate",
        "p_normal_approx",
    ]
    mean_cols = [
        "group",
        "target",
        "seeds",
        "seed_mean_delta",
        "seed_std_delta",
        "all_seeds_improved",
        "mean_improve_rate",
    ]
    lines = [
        "# Impact Correction Paired Sample Stats",
        "",
        "Delta is adapter sample MAE minus source sample MAE; negative is better.",
        "The confidence interval is a normal approximation over event-level sample deltas, so it is supporting evidence rather than a full seed-level significance test.",
        "",
        "## Seed-Mean Focus",
        "",
        ordered_stats(focus_mean)[mean_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-Seed Focus",
        "",
        ordered_stats(focus_seed)[seed_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) not in {0, len(args.adapter_dirs)}:
        raise ValueError("--labels must be omitted or have the same length as --adapter-dirs")
    labels = args.labels if args.labels else [path.name for path in args.adapter_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    sample_frames = []
    stat_frames = []
    configs = []
    for adapter_dir, label in zip(args.adapter_dirs, labels, strict=True):
        adapter_dir = adapter_dir.resolve()
        samples, group_masks, group_labels = collect_sample_rows(adapter_dir, label, args.split, args.batch_size, device)
        sample_frames.append(samples)
        stats = summarize_samples(samples, group_masks, group_labels)
        stats.insert(0, "label", label)
        stat_frames.append(stats)
        configs.append({"label": label, "adapter_dir": str(adapter_dir)})

    sample_df = pd.concat(sample_frames, ignore_index=True)
    per_seed_stats = ordered_stats(pd.concat(stat_frames, ignore_index=True))
    seed_mean_stats = ordered_stats(summarize_seed_means(per_seed_stats))

    sample_df.to_csv(output_dir / "sample_level_deltas.csv", index=False)
    per_seed_stats.to_csv(output_dir / "per_seed_paired_stats.csv", index=False)
    seed_mean_stats.to_csv(output_dir / "seed_mean_paired_stats.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "batch_size": args.batch_size,
                "runs": configs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, per_seed_stats, seed_mean_stats)
    print(f"wrote paired stats to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
