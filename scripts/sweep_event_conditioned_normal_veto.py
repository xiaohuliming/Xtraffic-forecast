#!/usr/bin/env python3
"""Sweep event-conditioned normal-veto inference calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from sweep_sttis_gate_posthoc import resolve_cache_path
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/event_conditioned_normal_veto_seed_23"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--normal-veto-temperatures", default="0.75,1.0,1.5")
    parser.add_argument("--base-scales", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--severity-boosts", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--recovery-boosts", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--event-temperatures", default="1.0")
    parser.add_argument(
        "--event-source",
        choices=["pred", "true"],
        default="pred",
        help="Use predicted event_aux signals or true standardized event_aux signals for oracle diagnostics.",
    )
    parser.add_argument("--event-boost-max", type=float, default=3.0)
    parser.add_argument("--betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def cap_indices(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def empty_sums(betas: list[float]) -> dict[float, dict[str, float]]:
    return {
        beta: {
            "all_model": 0.0,
            "all_base": 0.0,
            "all_count": 0.0,
            "aff_model": 0.0,
            "aff_base": 0.0,
            "aff_count": 0.0,
            "unaff_model": 0.0,
            "unaff_base": 0.0,
            "unaff_count": 0.0,
            "amount_sum": 0.0,
            "amount_aff_sum": 0.0,
            "amount_unaff_sum": 0.0,
            "event_boost_sum": 0.0,
            "event_boost_aff_sum": 0.0,
            "event_boost_unaff_sum": 0.0,
        }
        for beta in betas
    }


def update_sums(
    sums: dict[float, dict[str, float]],
    residual: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
    amount: torch.Tensor,
    event_boost: torch.Tensor,
    betas: list[float],
) -> None:
    base_abs = y.abs()
    boost_map = event_boost.expand_as(y)
    for beta in betas:
        model_abs = (beta * residual - y).abs()
        for prefix, label in [("all", "all"), ("aff", "affected"), ("unaff", "unaffected")]:
            mask = masks[label]
            count = mask.sum().item()
            if count <= 0:
                continue
            sums[beta][f"{prefix}_model"] += float(model_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_base"] += float(base_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_count"] += float(count)
        sums[beta]["amount_sum"] += float(amount[masks["all"]].sum().detach().cpu())
        sums[beta]["amount_aff_sum"] += float(amount[masks["affected"]].sum().detach().cpu())
        sums[beta]["amount_unaff_sum"] += float(amount[masks["unaffected"]].sum().detach().cpu())
        sums[beta]["event_boost_sum"] += float(boost_map[masks["all"]].sum().detach().cpu())
        sums[beta]["event_boost_aff_sum"] += float(boost_map[masks["affected"]].sum().detach().cpu())
        sums[beta]["event_boost_unaff_sum"] += float(boost_map[masks["unaffected"]].sum().detach().cpu())


def summarize(
    split: str,
    normal_veto_temperature: float,
    base_scale: float,
    severity_boost: float,
    recovery_boost: float,
    event_temperature: float,
    sums: dict[float, dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for beta, vals in sums.items():
        row: dict[str, float | str] = {
            "split": split,
            "normal_veto_temperature": normal_veto_temperature,
            "base_scale": base_scale,
            "severity_boost": severity_boost,
            "recovery_boost": recovery_boost,
            "event_temperature": event_temperature,
            "beta": beta,
        }
        for prefix, label in [("all", "all"), ("aff", "affected"), ("unaff", "unaffected")]:
            model_mae = vals[f"{prefix}_model"] / max(vals[f"{prefix}_count"], 1.0)
            base_mae = vals[f"{prefix}_base"] / max(vals[f"{prefix}_count"], 1.0)
            row[f"{label}_mae"] = model_mae
            row[f"{label}_baseline_mae"] = base_mae
            row[f"{label}_gain_pct"] = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else math.nan
        row["amount_mean"] = vals["amount_sum"] / max(vals["all_count"], 1.0)
        row["affected_amount_mean"] = vals["amount_aff_sum"] / max(vals["aff_count"], 1.0)
        row["unaffected_amount_mean"] = vals["amount_unaff_sum"] / max(vals["unaff_count"], 1.0)
        row["event_boost_mean"] = vals["event_boost_sum"] / max(vals["all_count"], 1.0)
        row["affected_event_boost_mean"] = vals["event_boost_aff_sum"] / max(vals["aff_count"], 1.0)
        row["unaffected_event_boost_mean"] = vals["event_boost_unaff_sum"] / max(vals["unaff_count"], 1.0)
        rows.append(row)
    return rows


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    cache_path: Path,
    indices: np.ndarray,
    batch_size: int,
    normal_veto_temperatures: list[float],
    base_scales: list[float],
    severity_boosts: list[float],
    recovery_boosts: list[float],
    event_temperatures: list[float],
    event_source: str,
    event_boost_max: float,
    betas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    keys = [
        (normal_temp, scale, sev_boost, rec_boost, event_temp)
        for normal_temp in normal_veto_temperatures
        for scale in base_scales
        for sev_boost in severity_boosts
        for rec_boost in recovery_boosts
        for event_temp in event_temperatures
    ]
    all_sums = {key: empty_sums(betas) for key in keys}
    old_temperature = float(getattr(model, "normal_veto_temperature", 1.0))
    old_scale = float(getattr(model, "normal_veto_scale", 1.0))
    model.eval()
    try:
        with torch.no_grad():
            for normal_temp in normal_veto_temperatures:
                model.normal_veto_temperature = float(normal_temp)
                model.normal_veto_scale = 1.0
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
                        event_aux,
                        node_affected,
                        node_valid,
                        _idx,
                    ) = [item.to(device) for item in batch]
                    if int(getattr(model, "hist_input_channels", hist.shape[-1])) > hist.shape[-1]:
                        hist = torch.cat([hist, hist_normal], dim=-1)
                    _pred_y, _pred_impact, pred_event, _pred_node, details = model(
                        hist,
                        node,
                        global_context,
                        normal_delta,
                        return_details=True,
                    )
                    normal = details["normal_residual"]
                    base_fused = details["base_fused_residual"]
                    score = details["normal_veto"]
                    if event_source == "true":
                        severity_signal = torch.relu(event_aux[:, 0])
                        recovery_signal = torch.relu(event_aux[:, 1])
                    elif event_source == "pred":
                        severity_signal = torch.relu(pred_event[:, 0])
                        recovery_signal = torch.relu(pred_event[:, 1])
                    else:
                        raise ValueError(f"unsupported event_source: {event_source}")
                    masks = {
                        "all": y_mask.bool(),
                        "affected": y_mask.bool() & node_affected[:, None, :, None].bool(),
                        "unaffected": y_mask.bool()
                        & (~node_affected[:, None, :, None].bool())
                        & node_valid[:, None, :, None].bool(),
                    }
                    for scale in base_scales:
                        for sev_boost in severity_boosts:
                            for rec_boost in recovery_boosts:
                                for event_temp in event_temperatures:
                                    boost = 1.0 + (
                                        sev_boost * severity_signal + rec_boost * recovery_signal
                                    ) / max(event_temp, 1e-6)
                                    if event_boost_max > 0.0:
                                        boost = boost.clamp(max=event_boost_max)
                                    boost = boost[:, None, None, None]
                                    amount = (scale * score * boost).clamp(0.0, 1.0)
                                    residual = (1.0 - amount) * base_fused + amount * normal
                                    key = (normal_temp, scale, sev_boost, rec_boost, event_temp)
                                    update_sums(all_sums[key], residual, y, masks, amount, boost, betas)
    finally:
        model.normal_veto_temperature = old_temperature
        model.normal_veto_scale = old_scale
    rows = []
    for key, sums in all_sums.items():
        rows.extend(summarize(split, *key, sums))
    return pd.DataFrame(rows)


def select_row(val_df: pd.DataFrame, metric: str, all_tolerance: float) -> pd.Series:
    if metric not in val_df.columns:
        raise KeyError(f"selection metric not found: {metric}")
    best_all = float(val_df["all_mae"].min())
    eligible = val_df[val_df["all_mae"] <= best_all + all_tolerance]
    if eligible.empty:
        eligible = val_df
    return eligible.loc[eligible[metric].idxmin()]


def write_summary(
    output_dir: Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected: pd.Series,
    event_source: str,
) -> None:
    key_cols = [
        "normal_veto_temperature",
        "base_scale",
        "severity_boost",
        "recovery_boost",
        "event_temperature",
        "beta",
    ]
    mask = np.ones(len(test_df), dtype=bool)
    for col in key_cols:
        mask &= np.isclose(test_df[col].astype(float), float(selected[col]))
    test_row = test_df[mask].iloc[0]
    show_cols = key_cols + [
        "all_mae",
        "affected_mae",
        "unaffected_mae",
        "affected_amount_mean",
        "affected_event_boost_mean",
    ]
    lines = [
        "# Event-Conditioned Normal-Veto Sweep",
        "",
        f"This changes only inference-time normal-veto calibration using `{event_source}` event severity/recovery signals.",
        "",
        "## Validation-selected result",
        "",
    ]
    for col in key_cols:
        lines.append(f"- {col}: `{float(selected[col]):.4g}`")
    lines.extend(
        [
            f"- validation all / affected MAE: `{float(selected['all_mae']):.6f}` / `{float(selected['affected_mae']):.6f}`",
            f"- test all / affected / unaffected MAE: `{float(test_row['all_mae']):.6f}` / `{float(test_row['affected_mae']):.6f}` / `{float(test_row['unaffected_mae']):.6f}`",
            "",
            "## Top validation affected rows",
            "",
            val_df.sort_values("affected_mae")[show_cols].head(12).to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Top test affected rows",
            "",
            test_df.sort_values("affected_mae")[show_cols].head(12).to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)
    model = make_model(ckpt, cache_path, device)
    if not hasattr(model, "normal_veto_temperature"):
        raise TypeError(f"{model_dir} is not a normal-veto checkpoint")
    splits = split_indices(cache_path)
    eval_indices = {
        split: cap_indices(indices, args.max_samples, args.seed + offset)
        for offset, (split, indices) in enumerate(splits.items())
    }
    normal_veto_temperatures = parse_float_list(args.normal_veto_temperatures)
    base_scales = parse_float_list(args.base_scales)
    severity_boosts = parse_float_list(args.severity_boosts)
    recovery_boosts = parse_float_list(args.recovery_boosts)
    event_temperatures = parse_float_list(args.event_temperatures)
    betas = parse_float_list(args.betas)
    val_df = evaluate_split(
        "val",
        model,
        cache_path,
        eval_indices["val"],
        args.batch_size,
        normal_veto_temperatures,
        base_scales,
        severity_boosts,
        recovery_boosts,
        event_temperatures,
        args.event_source,
        args.event_boost_max,
        betas,
        device,
    )
    test_df = evaluate_split(
        "test",
        model,
        cache_path,
        eval_indices["test"],
        args.batch_size,
        normal_veto_temperatures,
        base_scales,
        severity_boosts,
        recovery_boosts,
        event_temperatures,
        args.event_source,
        args.event_boost_max,
        betas,
        device,
    )
    val_df.to_csv(output_dir / "val_event_conditioned_normal_veto_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_event_conditioned_normal_veto_sweep.csv", index=False)
    selected = select_row(val_df, args.selection_metric, args.all_val_tolerance)
    write_summary(output_dir, val_df, test_df, selected, args.event_source)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "cache_path": str(cache_path), "device": str(device)}, f, indent=2, ensure_ascii=False, default=str)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
