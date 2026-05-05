#!/usr/bin/env python3
"""Case studies comparing two dual-branch normal-veto models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, cap_indices, make_model, torch_load
from compare_dual_branch_group_metrics import residual_beta
from train_full_candidate_stgnn_heatmap_model import CHANNELS, compute_stats, split_indices
from train_impact_residual_model import choose_device
from visualize_dual_branch_gate_cases import masked_channel_mean, masked_mae_by_sample, sort_nodes_by_position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid"),
    )
    parser.add_argument(
        "--focused-model-dir",
        type=Path,
        default=Path(
            "outputs/impact_guided_next_stage/"
            "dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/focused_veto_case_studies_seed_23"),
    )
    parser.add_argument("--base-label", default="element normal-veto")
    parser.add_argument("--candidate-label", default="focused impact-veto")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-success", type=int, default=4)
    parser.add_argument("--num-boundary", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_cache_path(model_dir: Path, ckpt: dict[str, object]) -> Path:
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        metrics_path = model_dir / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cache_path = Path(payload["cache_path"])
    return cache_path.resolve()


def model_args_from_ckpt(ckpt: dict[str, object]) -> dict[str, object]:
    model_args = ckpt.get("args", {})
    return model_args if isinstance(model_args, dict) else {}


def prepare_hist(
    hist: torch.Tensor,
    hist_normal: torch.Tensor,
    model_args: dict[str, object],
) -> torch.Tensor:
    if bool(model_args.get("use_dual_hist_residual", False)):
        return torch.cat([hist, hist_normal], dim=-1)
    return hist


def forward_with_details(
    model: torch.nn.Module,
    beta: float,
    hist: torch.Tensor,
    hist_normal: torch.Tensor,
    node: torch.Tensor,
    global_context: torch.Tensor,
    normal_delta: torch.Tensor,
    model_args: dict[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hist_in = prepare_hist(hist, hist_normal, model_args)
    pred_y, _pred_impact, _pred_event, _pred_node, details = model(
        hist_in,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    return beta * pred_y, details


def event_group_columns(event_aux: np.ndarray, severity_q: tuple[float, float]) -> dict[str, np.ndarray]:
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    spread_nodes = np.expm1(event_aux[:, 2])
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    spread_nodes[~np.isfinite(spread_nodes)] = 0.0
    q33, q66 = severity_q
    return {
        "severity": severity,
        "recovery_min": recovery_min,
        "spread_nodes": spread_nodes,
        "severity_low": severity <= q33,
        "severity_mid": (severity > q33) & (severity <= q66),
        "severity_high": severity > q66,
        "recovery_short_lt30": recovery_min < 30.0,
        "recovery_mid_30_90": (recovery_min >= 30.0) & (recovery_min < 90.0),
        "recovery_long_ge90": recovery_min >= 90.0,
    }


def masked_mean_by_sample(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=(1, 2, 3)).astype(np.float64)
    summed = np.where(mask, values, 0.0).sum(axis=(1, 2, 3)).astype(np.float64)
    return np.divide(summed, np.maximum(count, 1.0), out=np.full_like(summed, np.nan), where=count > 0)


def score_split(
    base_model: torch.nn.Module,
    focused_model: torch.nn.Module,
    cache_path: Path,
    selected_indices: np.ndarray,
    base_beta: float,
    focused_beta: float,
    base_args: dict[str, object],
    focused_args: dict[str, object],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=selected_indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    rows: list[dict[str, float | int | bool]] = []

    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        event_aux_all = h5["event_aux"][selected_indices].astype(np.float32)
        severity = np.expm1(event_aux_all[:, 0])
        severity[~np.isfinite(severity)] = 0.0
        severity_q = tuple(float(x) for x in np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0]))

        for batch_idx, batch in enumerate(loader, start=1):
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
                idx,
            ) = batch
            hist = hist.to(device)
            hist_normal = hist_normal.to(device)
            node = node.to(device)
            global_context = global_context.to(device)
            normal_delta = normal_delta.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            node_affected = node_affected.to(device)
            node_valid = node_valid.to(device)

            base_pred, base_details = forward_with_details(
                base_model,
                base_beta,
                hist,
                hist_normal,
                node,
                global_context,
                normal_delta,
                base_args,
            )
            focused_pred, focused_details = forward_with_details(
                focused_model,
                focused_beta,
                hist,
                hist_normal,
                node,
                global_context,
                normal_delta,
                focused_args,
            )

            idx_np = idx.detach().cpu().numpy().astype(np.int64)
            y_np = y.detach().cpu().numpy()
            y_mask_np = y_mask.detach().cpu().numpy().astype(bool)
            affected_np = node_affected.detach().cpu().numpy().astype(bool)
            valid_np = node_valid.detach().cpu().numpy().astype(bool)

            all_mask = y_mask_np
            affected_mask = y_mask_np & affected_np[:, None, :, None]
            unaffected_mask = y_mask_np & (~affected_np[:, None, :, None]) & valid_np[:, None, :, None]
            masks = {"all": all_mask, "affected": affected_mask, "unaffected": unaffected_mask}

            base_error = np.abs(base_pred.detach().cpu().numpy() - y_np)
            focused_error = np.abs(focused_pred.detach().cpu().numpy() - y_np)
            base_veto = base_details["normal_veto_amount"].detach().cpu().numpy()
            focused_veto = focused_details["normal_veto_amount"].detach().cpu().numpy()
            base_gate = base_details["gate"].detach().cpu().numpy()
            focused_gate = focused_details["gate"].detach().cpu().numpy()

            base_mae: dict[str, np.ndarray] = {}
            focused_mae: dict[str, np.ndarray] = {}
            base_veto_mean: dict[str, np.ndarray] = {}
            focused_veto_mean: dict[str, np.ndarray] = {}
            base_gate_mean: dict[str, np.ndarray] = {}
            focused_gate_mean: dict[str, np.ndarray] = {}
            counts: dict[str, np.ndarray] = {}
            for subset, mask in masks.items():
                base_mae[subset], counts[subset] = masked_mae_by_sample(base_error, mask)
                focused_mae[subset], _ = masked_mae_by_sample(focused_error, mask)
                base_veto_mean[subset] = masked_mean_by_sample(base_veto, mask)
                focused_veto_mean[subset] = masked_mean_by_sample(focused_veto, mask)
                base_gate_mean[subset] = masked_mean_by_sample(base_gate, mask)
                focused_gate_mean[subset] = masked_mean_by_sample(focused_gate, mask)

            raw_event = h5["event_aux"][idx_np].astype(np.float32)
            groups = event_group_columns(raw_event, severity_q)
            region_code = h5["region_code"][idx_np]
            affected_nodes = h5["node_affected"][idx_np].sum(axis=1)
            valid_nodes = h5["node_valid"][idx_np].sum(axis=1)

            for i, sample_idx in enumerate(idx_np):
                row: dict[str, float | int | bool] = {
                    "sample_idx": int(sample_idx),
                    "region_code": int(region_code[i]),
                    "affected_nodes": float(affected_nodes[i]),
                    "valid_nodes": float(valid_nodes[i]),
                    "affected_elements": float(counts["affected"][i]),
                    "severity": float(groups["severity"][i]),
                    "severity_log_auc": float(raw_event[i, 0]),
                    "recovery_min": float(groups["recovery_min"][i]),
                    "spread_nodes": float(groups["spread_nodes"][i]),
                    "spread_log_nodes": float(raw_event[i, 2]),
                    "severity_low": bool(groups["severity_low"][i]),
                    "severity_mid": bool(groups["severity_mid"][i]),
                    "severity_high": bool(groups["severity_high"][i]),
                    "recovery_short_lt30": bool(groups["recovery_short_lt30"][i]),
                    "recovery_mid_30_90": bool(groups["recovery_mid_30_90"][i]),
                    "recovery_long_ge90": bool(groups["recovery_long_ge90"][i]),
                }
                for subset in masks:
                    row[f"base_{subset}_mae"] = float(base_mae[subset][i])
                    row[f"focused_{subset}_mae"] = float(focused_mae[subset][i])
                    row[f"{subset}_gain"] = float(base_mae[subset][i] - focused_mae[subset][i])
                    row[f"{subset}_delta"] = float(focused_mae[subset][i] - base_mae[subset][i])
                    row[f"base_veto_{subset}_mean"] = float(base_veto_mean[subset][i])
                    row[f"focused_veto_{subset}_mean"] = float(focused_veto_mean[subset][i])
                    row[f"veto_{subset}_diff"] = float(focused_veto_mean[subset][i] - base_veto_mean[subset][i])
                    row[f"base_gate_{subset}_mean"] = float(base_gate_mean[subset][i])
                    row[f"focused_gate_{subset}_mean"] = float(focused_gate_mean[subset][i])
                    row[f"gate_{subset}_diff"] = float(focused_gate_mean[subset][i] - base_gate_mean[subset][i])
                rows.append(row)

            if batch_idx % 20 == 0:
                done = min(batch_idx * batch_size, selected_indices.size)
                print(f"scored {done}/{selected_indices.size}", flush=True)

    return pd.DataFrame(rows)


def pick_top_unique(
    df: pd.DataFrame,
    mask: pd.Series,
    category: str,
    sort_col: str,
    ascending: bool,
    limit: int,
    seen: set[int],
) -> pd.DataFrame:
    candidates = df[mask & (df["affected_elements"] > 0) & np.isfinite(df[sort_col])].copy()
    candidates = candidates.sort_values(
        [sort_col, "affected_elements", "severity", "recovery_min"],
        ascending=[ascending, False, False, False],
    )
    selected = []
    for _, row in candidates.iterrows():
        sample_idx = int(row["sample_idx"])
        if sample_idx in seen:
            continue
        out = row.copy()
        out["category"] = category
        selected.append(out)
        seen.add(sample_idx)
        if len(selected) >= limit:
            break
    if not selected:
        return pd.DataFrame(columns=list(df.columns) + ["category"])
    return pd.DataFrame(selected)


def pick_cases(df: pd.DataFrame, num_success: int, num_boundary: int) -> pd.DataFrame:
    seen: set[int] = set()
    severe_or_long = df["severity_high"] | df["recovery_long_ge90"]
    both = df["severity_high"] & df["recovery_long_ge90"]

    success_parts = []
    first_limit = min(max(1, num_success // 2), num_success)
    first = pick_top_unique(
        df,
        both,
        "success_high_severity_long_recovery",
        "affected_gain",
        False,
        first_limit,
        seen,
    )
    if not first.empty:
        success_parts.append(first)

    success_count = sum(len(part) for part in success_parts)
    if success_count < num_success:
        part = pick_top_unique(
            df,
            df["severity_high"],
            "success_high_severity",
            "affected_gain",
            False,
            num_success - success_count,
            seen,
        )
        if not part.empty:
            success_parts.append(part)
    success_count = sum(len(part) for part in success_parts)
    if success_count < num_success:
        part = pick_top_unique(
            df,
            df["recovery_long_ge90"],
            "success_long_recovery",
            "affected_gain",
            False,
            num_success - success_count,
            seen,
        )
        if not part.empty:
            success_parts.append(part)

    boundary = pick_top_unique(
        df,
        severe_or_long,
        "boundary_severe_or_long",
        "affected_gain",
        True,
        num_boundary,
        seen,
    )
    parts = [*success_parts]
    if not boundary.empty:
        parts.append(boundary)
    selected = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if selected.empty:
        selected = df[(df["affected_elements"] > 0) & np.isfinite(df["affected_gain"])].copy()
        selected = selected.sort_values("affected_gain", ascending=False).head(num_success)
        selected["category"] = "fallback_top_gain"
    selected = selected.head(num_success + num_boundary).copy()
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def load_single_case(
    model: torch.nn.Module,
    beta: float,
    cache_path: Path,
    sample_idx: int,
    stats: object,
    model_args: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=np.asarray([sample_idx], dtype=np.int64), stats=stats)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
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
        _node_affected,
        _node_valid,
        _idx,
    ) = batch
    hist = hist.to(device)
    hist_normal = hist_normal.to(device)
    node = node.to(device)
    global_context = global_context.to(device)
    normal_delta = normal_delta.to(device)
    with torch.no_grad():
        pred, details = forward_with_details(
            model,
            beta,
            hist,
            hist_normal,
            node,
            global_context,
            normal_delta,
            model_args,
        )
    details_np = {key: value.detach().cpu().numpy()[0] for key, value in details.items() if value.ndim >= 4}
    return pred.detach().cpu().numpy()[0], details_np


def visualize_case(
    base_model: torch.nn.Module,
    focused_model: torch.nn.Module,
    cache_path: Path,
    sample_idx: int,
    stats: object,
    base_beta: float,
    focused_beta: float,
    base_args: dict[str, object],
    focused_args: dict[str, object],
    device: torch.device,
    output_dir: Path,
    metrics_row: pd.Series,
    base_label: str,
    candidate_label: str,
) -> Path:
    base_pred, base_details = load_single_case(base_model, base_beta, cache_path, sample_idx, stats, base_args, device)
    focused_pred, focused_details = load_single_case(
        focused_model,
        focused_beta,
        cache_path,
        sample_idx,
        stats,
        focused_args,
        device,
    )
    with h5py.File(cache_path, "r") as h5:
        y = h5["y_residual"][sample_idx].astype(np.float32)
        y_mask = h5["y_mask"][sample_idx].astype(bool)
        affected = h5["node_affected"][sample_idx].astype(bool)
        valid = h5["node_valid"][sample_idx].astype(bool)
        position = h5["node_signed_pm_raw"][sample_idx].astype(np.float32)
        node_idx = h5["node_idx"][sample_idx].astype(np.int32)
        region_code = int(h5["region_code"][sample_idx])
        raw_event = h5["event_aux"][sample_idx].astype(np.float32)

    order = sort_nodes_by_position(position, valid)
    order = order[valid[order]]
    if order.size == 0:
        order = sort_nodes_by_position(position, valid)
    valid_order = valid[order]
    affected_order = affected[order]
    labels = [str(int(n)) if v else "" for n, v in zip(node_idx[order], valid_order)]

    base_error = np.abs(base_pred - y)
    focused_error = np.abs(focused_pred - y)
    gain = base_error - focused_error
    veto_diff = focused_details["normal_veto_amount"] - base_details["normal_veto_amount"]

    target_map = masked_channel_mean(np.abs(y), y_mask)[:, order]
    base_error_map = masked_channel_mean(base_error, y_mask)[:, order]
    focused_error_map = masked_channel_mean(focused_error, y_mask)[:, order]
    gain_map = masked_channel_mean(gain, y_mask)[:, order]
    veto_diff_map = masked_channel_mean(veto_diff, y_mask)[:, order]
    for arr in [target_map, base_error_map, focused_error_map, gain_map, veto_diff_map]:
        arr[:, ~valid_order] = np.nan

    affected_node_mask = affected[:, None] & valid[:, None]
    flow_idx = CHANNELS.index("flow")
    horizon = np.arange(1, y.shape[0] + 1)
    curve_mask = y_mask[:, :, flow_idx] & affected[None, :] & valid[None, :]
    curve_count = curve_mask.sum(axis=1)
    y_curve = np.divide(
        np.where(curve_mask, y[:, :, flow_idx], 0.0).sum(axis=1),
        np.maximum(curve_count, 1),
        out=np.full(y.shape[0], np.nan, dtype=np.float64),
        where=curve_count > 0,
    )
    base_curve = np.divide(
        np.where(curve_mask, base_pred[:, :, flow_idx], 0.0).sum(axis=1),
        np.maximum(curve_count, 1),
        out=np.full(y.shape[0], np.nan, dtype=np.float64),
        where=curve_count > 0,
    )
    focused_curve = np.divide(
        np.where(curve_mask, focused_pred[:, :, flow_idx], 0.0).sum(axis=1),
        np.maximum(curve_count, 1),
        out=np.full(y.shape[0], np.nan, dtype=np.float64),
        where=curve_count > 0,
    )
    if not affected_node_mask.any():
        y_curve[:] = np.nan
        base_curve[:] = np.nan
        focused_curve[:] = np.nan

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(13.5, 13.4),
        gridspec_kw={"height_ratios": [0.25, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    affected_strip = np.where(affected_order[None, :], 1.0, 0.0)
    affected_strip[:, ~valid_order] = np.nan
    axes[0].imshow(affected_strip, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    axes[0].set_yticks([])
    axes[0].set_title("Affected candidate nodes")

    axes[1].plot(horizon, y_curve, marker="o", linewidth=1.6, label="target residual")
    axes[1].plot(horizon, base_curve, marker="s", linewidth=1.3, label=base_label)
    axes[1].plot(horizon, focused_curve, marker="^", linewidth=1.3, label=candidate_label)
    axes[1].axhline(0.0, color="0.65", linewidth=0.8)
    axes[1].set_ylabel("Flow residual")
    axes[1].set_title("Affected-node mean flow residual")
    axes[1].legend(loc="best", fontsize=8)

    vmax_target = np.nanpercentile(target_map, 95) if np.isfinite(target_map).any() else 1.0
    im_target = axes[2].imshow(
        target_map,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(vmax_target, 1e-6),
    )
    axes[2].set_ylabel("Horizon")
    axes[2].set_title("Absolute target residual")
    fig.colorbar(im_target, ax=axes[2], fraction=0.018, pad=0.01)

    vmax_error = np.nanpercentile(np.concatenate([base_error_map.ravel(), focused_error_map.ravel()]), 95)
    vmax_error = max(float(vmax_error), 1e-6) if np.isfinite(vmax_error) else 1.0
    im_base = axes[3].imshow(base_error_map, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0.0, vmax=vmax_error)
    axes[3].set_ylabel("Horizon")
    axes[3].set_title(f"{base_label} absolute error")
    fig.colorbar(im_base, ax=axes[3], fraction=0.018, pad=0.01)

    im_focus = axes[4].imshow(
        focused_error_map,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax_error,
    )
    axes[4].set_ylabel("Horizon")
    axes[4].set_title(f"{candidate_label} absolute error")
    fig.colorbar(im_focus, ax=axes[4], fraction=0.018, pad=0.01)

    vmax_gain = np.nanpercentile(np.abs(gain_map), 95) if np.isfinite(gain_map).any() else 1.0
    im_gain = axes[5].imshow(
        gain_map,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-max(vmax_gain, 1e-6),
        vmax=max(vmax_gain, 1e-6),
    )
    axes[5].set_ylabel("Horizon")
    axes[5].set_title(f"{base_label} error minus {candidate_label} error (positive = {candidate_label} better)")
    fig.colorbar(im_gain, ax=axes[5], fraction=0.018, pad=0.01)

    vmax_veto = np.nanpercentile(np.abs(veto_diff_map), 95) if np.isfinite(veto_diff_map).any() else 1.0
    im_veto = axes[6].imshow(
        veto_diff_map,
        aspect="auto",
        interpolation="nearest",
        cmap="PiYG",
        vmin=-max(vmax_veto, 1e-6),
        vmax=max(vmax_veto, 1e-6),
    )
    axes[6].set_ylabel("Horizon")
    axes[6].set_title(f"{candidate_label} normal-veto amount minus {base_label} normal-veto amount")
    fig.colorbar(im_veto, ax=axes[6], fraction=0.018, pad=0.01)

    for ax in axes[2:]:
        ax.set_yticks(np.arange(0, y.shape[0], 2))
        ax.set_yticklabels([str(i + 1) for i in range(0, y.shape[0], 2)])
    for ax in axes:
        ax.set_xticks(np.arange(0, len(labels), 4))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 4)], rotation=45, ha="right", fontsize=7)
    axes[1].set_xticks(horizon)
    axes[-1].set_xlabel("Candidate sensors sorted by signed postmile")

    category = str(metrics_row.get("category", "case"))
    title = (
        f"Case {int(metrics_row['rank'])} ({category}): sample {sample_idx}, region {region_code}, "
        f"affected MAE {metrics_row['base_affected_mae']:.3f} -> {metrics_row['focused_affected_mae']:.3f} "
        f"(gain {metrics_row['affected_gain']:.3f})"
    )
    subtitle = (
        f"severity={metrics_row['severity']:.3f}, recovery={metrics_row['recovery_min']:.1f} min, "
        f"spread={metrics_row['spread_nodes']:.1f}, affected_nodes={int(affected.sum())}, "
        f"raw_event=({raw_event[0]:.3f}, {raw_event[1]:.3f}, {raw_event[2]:.3f})"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=12)
    safe_category = category.replace("/", "_").replace(" ", "_")
    path = output_dir / f"case_{int(metrics_row['rank']):02d}_{safe_category}_sample_{sample_idx}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    cases: pd.DataFrame,
    paths: list[Path],
    score_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    base_beta: float,
    focused_beta: float,
) -> None:
    table_cols = [
        "rank",
        "category",
        "sample_idx",
        "region_code",
        "severity_high",
        "recovery_long_ge90",
        "affected_nodes",
        "severity",
        "recovery_min",
        "base_affected_mae",
        "focused_affected_mae",
        "affected_gain",
        "veto_affected_diff",
        "gate_affected_diff",
    ]
    case_table = cases[[col for col in table_cols if col in cases.columns]].copy()
    group_rows = []
    for group, mask in {
        "overall": np.ones(len(score_df), dtype=bool),
        "severity_high": score_df["severity_high"].to_numpy(dtype=bool),
        "recovery_long_ge90": score_df["recovery_long_ge90"].to_numpy(dtype=bool),
        "severity_high_and_long": (score_df["severity_high"] & score_df["recovery_long_ge90"]).to_numpy(dtype=bool),
    }.items():
        sub = score_df[mask]
        if sub.empty:
            continue
        group_rows.append(
            {
                "group": group,
                "samples": len(sub),
                "affected_gain_mean": sub["affected_gain"].mean(),
                "affected_gain_median": sub["affected_gain"].median(),
                "veto_affected_diff_mean": sub["veto_affected_diff"].mean(),
                "gate_affected_diff_mean": sub["gate_affected_diff"].mean(),
            }
        )
    group_table = pd.DataFrame(group_rows)
    lines = [
        "# Dual-Branch Veto Case Studies",
        "",
        f"- split: `{args.split}`",
        f"- base model: `{args.base_model_dir}`",
        f"- candidate model: `{args.focused_model_dir}`",
        f"- base label: `{args.base_label}`",
        f"- candidate label: `{args.candidate_label}`",
        f"- base residual beta: `{base_beta:.4f}`",
        f"- candidate residual beta: `{focused_beta:.4f}`",
        "",
        "Positive affected_gain means the candidate model has lower affected-node MAE.",
        "",
        "## Group Snapshot",
        "",
        group_table.to_markdown(index=False, floatfmt=".6f") if not group_table.empty else "No group rows.",
        "",
        "## Selected Cases",
        "",
        case_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- The curve compares affected-node mean flow residuals over the 12 forecast horizons.",
            "- The gain heatmap is positive where the candidate model reduces local error.",
            "- The veto-difference heatmap shows how much more or less the candidate detector pulls the fused residual toward the normal branch.",
        ]
    )
    (output_dir / "case_study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_case in output_dir.glob("case_*.png"):
        old_case.unlink()
    device = choose_device(args.device)

    base_dir = args.base_model_dir.resolve()
    focused_dir = args.focused_model_dir.resolve()
    base_ckpt = torch_load(base_dir / "model.pt")
    focused_ckpt = torch_load(focused_dir / "model.pt")
    cache_path = resolve_cache_path(base_dir, base_ckpt)
    focused_cache = resolve_cache_path(focused_dir, focused_ckpt)
    if focused_cache != cache_path:
        raise ValueError(f"model cache mismatch: {cache_path} vs {focused_cache}")

    base_args = model_args_from_ckpt(base_ckpt)
    focused_args = model_args_from_ckpt(focused_ckpt)
    base_beta = residual_beta(base_dir, base_ckpt)
    focused_beta = residual_beta(focused_dir, focused_ckpt)
    indices = split_indices(cache_path)[args.split]
    selected_indices = cap_indices(indices, args.max_samples, args.seed)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"base: {base_dir}", flush=True)
    print(f"candidate: {focused_dir}", flush=True)
    print(f"samples: {selected_indices.size}", flush=True)

    base_model = make_model(base_ckpt, cache_path, device)
    focused_model = make_model(focused_ckpt, cache_path, device)
    score_df = score_split(
        base_model=base_model,
        focused_model=focused_model,
        cache_path=cache_path,
        selected_indices=selected_indices,
        base_beta=base_beta,
        focused_beta=focused_beta,
        base_args=base_args,
        focused_args=focused_args,
        batch_size=args.batch_size,
        device=device,
    )
    score_df.to_csv(output_dir / "candidate_case_metrics.csv", index=False)

    cases = pick_cases(score_df, args.num_success, args.num_boundary)
    cases.to_csv(output_dir / "selected_cases.csv", index=False)

    stats = compute_stats(cache_path)
    paths = []
    for _, row in cases.iterrows():
        path = visualize_case(
            base_model=base_model,
            focused_model=focused_model,
            cache_path=cache_path,
            sample_idx=int(row["sample_idx"]),
            stats=stats,
            base_beta=base_beta,
            focused_beta=focused_beta,
            base_args=base_args,
            focused_args=focused_args,
            device=device,
            output_dir=output_dir,
            metrics_row=row,
            base_label=args.base_label,
            candidate_label=args.candidate_label,
        )
        paths.append(path)
        print(f"wrote {path.name}", flush=True)

    write_report(cases, paths, score_df, output_dir, args, base_beta, focused_beta)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_dir": str(base_dir),
                "focused_model_dir": str(focused_dir),
                "base_label": args.base_label,
                "candidate_label": args.candidate_label,
                "cache_path": str(cache_path),
                "split": args.split,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "num_success": args.num_success,
                "num_boundary": args.num_boundary,
                "base_beta": base_beta,
                "focused_beta": focused_beta,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote case studies to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
