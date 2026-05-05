#!/usr/bin/env python3
"""Train a lightweight candidate-node STGNN residual forecaster.

This script upgrades the candidate-neighborhood MLP scaffold to a small
spatiotemporal graph model:

1. A GRU encodes each candidate node's historical normalized residual.
2. Dynamic graph propagation uses signed postmile distance and node validity.
3. A node-wise decoder predicts future incident residuals.

The candidate set is still selected only by road-aligned spatial proximity, not
by impact labels. Impact labels are used only for optional auxiliary losses and
for stratified reporting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_impact_labels import (
    build_baseline_valid_mask,
    build_matches,
    build_robust_baseline,
    load_incidents_2023,
    load_region_traffic,
    load_sensor_meta,
    region_specs,
)
from train_candidate_residual_model import (
    build_candidate_lookup,
    build_node_context,
    cap_train_samples,
    make_event_aux,
    robust_mae_from_arrays,
    select_candidate_nodes,
)
from train_impact_residual_model import (
    choose_device,
    json_safe_args,
    make_event_context,
    make_time_features,
    split_name,
    standardize_aux,
    standardize_train_val_test,
)
from validate_forecast_error_against_impact import fit_blend_alphas, parse_incident_ids


CHANNELS = ("flow", "occupancy", "speed")


@dataclass
class SampleArrays:
    hist_residual: np.ndarray
    node_context: np.ndarray
    global_context: np.ndarray
    y_residual: np.ndarray
    y_mask: np.ndarray
    y_event_aux_raw: np.ndarray
    y_node_affected: np.ndarray
    node_valid: np.ndarray
    normal_pred: np.ndarray
    actual_future: np.ndarray
    future_scale: np.ndarray
    split: np.ndarray
    region: np.ndarray


class DirectionalGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.all_proj = nn.Linear(hidden_dim, hidden_dim)
        self.left_proj = nn.Linear(hidden_dim, hidden_dim)
        self.right_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj_all: torch.Tensor,
        adj_left: torch.Tensor,
        adj_right: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        all_msg = torch.bmm(adj_all, h)
        left_msg = torch.bmm(adj_left, h)
        right_msg = torch.bmm(adj_right, h)
        out = (
            self.self_proj(h)
            + self.all_proj(all_msg)
            + self.left_proj(left_msg)
            + self.right_proj(right_msg)
        )
        out = self.norm(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        return out * valid.unsqueeze(-1)


class CandidateSTGNNResidual(nn.Module):
    def __init__(
        self,
        channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.channels = channels
        self.graph_sigma = graph_sigma
        self.temporal_encoder = nn.GRU(
            input_size=channels,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(node_context_dim + global_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [DirectionalGraphLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(graph_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * channels),
        )
        self.event_aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.node_aux_head = nn.Linear(hidden_dim, 1)
        final = self.decoder[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def build_adjacency(self, node_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signed_pm = node_context[:, :, 0]
        valid = node_context[:, :, 6].clamp(0.0, 1.0)
        diff = torch.abs(signed_pm[:, :, None] - signed_pm[:, None, :])
        valid_pair = valid[:, :, None] * valid[:, None, :]
        adj = torch.exp(-diff / max(self.graph_sigma, 1e-6)) * valid_pair

        eye = torch.eye(adj.shape[1], device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye * valid[:, :, None]

        source_pos = signed_pm[:, None, :]
        target_pos = signed_pm[:, :, None]
        left_mask = (source_pos < target_pos).to(adj.dtype)
        right_mask = (source_pos > target_pos).to(adj.dtype)
        adj_left = adj * left_mask
        adj_right = adj * right_mask

        def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
            return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return row_normalize(adj), row_normalize(adj_left), row_normalize(adj_right), valid

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, input_steps, nodes, channels = hist_residual.shape
        temporal_in = hist_residual.permute(0, 2, 1, 3).reshape(batch_size * nodes, input_steps, channels)
        _, h_last = self.temporal_encoder(temporal_in)
        h = h_last[-1].reshape(batch_size, nodes, -1)

        global_rep = global_context[:, None, :].expand(-1, nodes, -1)
        ctx = self.context_proj(torch.cat([node_context, global_rep], dim=-1))
        h = self.input_norm(h + ctx)

        adj_all, adj_left, adj_right, valid = self.build_adjacency(node_context)
        h = h * valid.unsqueeze(-1)
        for layer in self.graph_layers:
            h = h + layer(h, adj_all, adj_left, adj_right, valid)
            h = h * valid.unsqueeze(-1)

        residual = self.decoder(h).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = residual.permute(0, 2, 1, 3).contiguous()

        pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(h).squeeze(-1)
        return residual, event_aux, node_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument(
        "--event-root",
        type=Path,
        default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"),
    )
    parser.add_argument("--raw-label-dir", type=Path, default=Path("outputs/impact_labels"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/candidate_stgnn_residual_model/first_pass"),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["Alameda", "ContraCosta", "Orange"],
    )
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--max-candidate-nodes", type=int, default=16)
    parser.add_argument("--sample-offsets", nargs="+", type=int, default=[0, 6, 12])
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-sigma", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-aux-weight", type=float, default=0.10)
    parser.add_argument("--node-aux-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def build_region_samples(
    region_name: str,
    data_dir: Path,
    event_root: Path,
    raw_label_dir: Path,
    meta: pd.DataFrame,
    inc: pd.DataFrame,
    args: argparse.Namespace,
) -> SampleArrays:
    region = region_specs()[region_name]
    region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
    region_meta = region_meta.reset_index(drop=True)
    region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)

    print(f"[{region_name}] loading traffic", flush=True)
    traffic, times = load_region_traffic(data_dir, region_node_idx)
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    total_steps = traffic.shape[0]

    print(f"[{region_name}] fitting normal forecaster", flush=True)
    matches = build_matches(
        inc=inc,
        region_meta=region_meta,
        times=times,
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
        baseline_mask_extra_steps=args.baseline_mask_extra_steps,
    )
    baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)
    train_valid = baseline_valid.copy()
    train_valid[int(total_steps * 0.70) :, :] = False
    baseline, scale, _ = build_robust_baseline(
        traffic=traffic,
        times=times,
        baseline_valid=train_valid,
        min_count=args.min_baseline_count,
    )
    alphas = fit_blend_alphas(
        traffic=traffic,
        times=times,
        train_valid=train_valid,
        baseline=baseline,
        input_steps=args.input_steps,
        horizon_steps=args.horizon_steps,
    )

    print(f"[{region_name}] building STGNN samples", flush=True)
    events = pd.read_csv(event_root / region_name / "event_labels.csv")
    raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
    candidate_lookup = build_candidate_lookup(raw_nodes)

    hist_rows: list[np.ndarray] = []
    node_ctx_rows: list[np.ndarray] = []
    global_ctx_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    y_mask_rows: list[np.ndarray] = []
    event_aux_rows: list[np.ndarray] = []
    node_affected_rows: list[np.ndarray] = []
    node_valid_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    actual_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    split_rows: list[str] = []
    region_rows: list[str] = []

    for row in events.itertuples(index=False):
        incident_ids = parse_incident_ids(row.incident_ids)
        candidates = select_candidate_nodes(
            incident_ids=incident_ids,
            candidate_lookup=candidate_lookup,
            anchor_region_idx=int(row.anchor_region_idx),
            max_nodes=args.max_candidate_nodes,
        )
        node_idx, node_valid, node_affected, node_ctx = build_node_context(
            candidates=candidates,
            max_nodes=args.max_candidate_nodes,
            pm_radius=args.candidate_pm_radius,
            anchor_idx=int(row.anchor_region_idx),
        )

        for offset in args.sample_offsets:
            sample_start = int(row.start_idx) + int(offset)
            input_start = sample_start - args.input_steps
            future_end = sample_start + args.horizon_steps
            if input_start < 0 or future_end > total_steps:
                continue

            input_idx = np.arange(input_start, sample_start, dtype=np.int32)
            future_idx = np.arange(sample_start, future_end, dtype=np.int32)

            hist = traffic[input_idx][:, node_idx, :]
            hist_base = baseline[day_kind[input_idx], tod[input_idx]][:, node_idx, :]
            hist_scale = scale[day_kind[input_idx], tod[input_idx]][:, node_idx, :]
            fut_base = baseline[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
            fut_scale = scale[day_kind[future_idx], tod[future_idx]][:, node_idx, :]
            actual_future = traffic[future_idx][:, node_idx, :]
            last_obs = traffic[sample_start - 1, node_idx, :]

            normal_pred = np.empty_like(actual_future, dtype=np.float32)
            for h in range(args.horizon_steps):
                normal_pred[h] = fut_base[h] + alphas[h][None, :] * (last_obs - fut_base[h])

            with np.errstate(divide="ignore", invalid="ignore"):
                hist_res_z = (hist - hist_base) / hist_scale
                y_res_z = (actual_future - normal_pred) / fut_scale
            valid_node_mask = node_valid.astype(bool)
            hist_res_z[:, ~valid_node_mask, :] = 0.0
            y_mask = np.isfinite(y_res_z)
            y_mask[:, ~valid_node_mask, :] = False
            hist_res_z[~np.isfinite(hist_res_z)] = 0.0
            y_res_z[~np.isfinite(y_res_z)] = 0.0
            if not y_mask.any():
                continue

            normal_pred[:, ~valid_node_mask, :] = 0.0
            actual_future[:, ~valid_node_mask, :] = 0.0
            fut_scale[:, ~valid_node_mask, :] = 1.0

            global_ctx = np.concatenate(
                [
                    make_event_context(row, offset, args.horizon_steps),
                    make_time_features(times, sample_start),
                ]
            ).astype(np.float32)
            global_ctx[~np.isfinite(global_ctx)] = 0.0

            hist_rows.append(hist_res_z.astype(np.float32))
            node_ctx_rows.append(node_ctx.astype(np.float32))
            global_ctx_rows.append(global_ctx)
            y_rows.append(y_res_z.astype(np.float32))
            y_mask_rows.append(y_mask.astype(np.float32))
            event_aux_rows.append(make_event_aux(row))
            node_affected_rows.append(node_affected.astype(np.float32))
            node_valid_rows.append(node_valid.astype(np.float32))
            normal_rows.append(normal_pred.astype(np.float32))
            actual_rows.append(actual_future.astype(np.float32))
            scale_rows.append(fut_scale.astype(np.float32))
            split_rows.append(split_name(sample_start, total_steps))
            region_rows.append(region_name)

    return SampleArrays(
        hist_residual=np.asarray(hist_rows, dtype=np.float32),
        node_context=np.asarray(node_ctx_rows, dtype=np.float32),
        global_context=np.asarray(global_ctx_rows, dtype=np.float32),
        y_residual=np.asarray(y_rows, dtype=np.float32),
        y_mask=np.asarray(y_mask_rows, dtype=np.float32),
        y_event_aux_raw=np.asarray(event_aux_rows, dtype=np.float32),
        y_node_affected=np.asarray(node_affected_rows, dtype=np.float32),
        node_valid=np.asarray(node_valid_rows, dtype=np.float32),
        normal_pred=np.asarray(normal_rows, dtype=np.float32),
        actual_future=np.asarray(actual_rows, dtype=np.float32),
        future_scale=np.asarray(scale_rows, dtype=np.float32),
        split=np.asarray(split_rows),
        region=np.asarray(region_rows),
    )


def concat_samples(parts: list[SampleArrays]) -> SampleArrays:
    return SampleArrays(
        hist_residual=np.concatenate([p.hist_residual for p in parts], axis=0),
        node_context=np.concatenate([p.node_context for p in parts], axis=0),
        global_context=np.concatenate([p.global_context for p in parts], axis=0),
        y_residual=np.concatenate([p.y_residual for p in parts], axis=0),
        y_mask=np.concatenate([p.y_mask for p in parts], axis=0),
        y_event_aux_raw=np.concatenate([p.y_event_aux_raw for p in parts], axis=0),
        y_node_affected=np.concatenate([p.y_node_affected for p in parts], axis=0),
        node_valid=np.concatenate([p.node_valid for p in parts], axis=0),
        normal_pred=np.concatenate([p.normal_pred for p in parts], axis=0),
        actual_future=np.concatenate([p.actual_future for p in parts], axis=0),
        future_scale=np.concatenate([p.future_scale for p in parts], axis=0),
        split=np.concatenate([p.split for p in parts], axis=0),
        region=np.concatenate([p.region for p in parts], axis=0),
    )


def standardize_structured_inputs(
    samples: SampleArrays,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    hist = samples.hist_residual.copy()
    hist_mean = hist[train_mask].mean(axis=(0, 1, 2), keepdims=True)
    hist_std = hist[train_mask].std(axis=(0, 1, 2), keepdims=True)
    hist_std[hist_std < 1e-6] = 1.0
    hist = (hist - hist_mean) / hist_std
    hist[~np.isfinite(hist)] = 0.0

    node_context = samples.node_context.copy()
    node_mean = node_context[train_mask].mean(axis=(0, 1), keepdims=True)
    node_std = node_context[train_mask].std(axis=(0, 1), keepdims=True)
    node_std[:, :, 6] = 1.0
    node_std[node_std < 1e-6] = 1.0
    node_context = (node_context - node_mean) / node_std
    # Preserve the fields used for graph construction.
    node_context[:, :, 0] = samples.node_context[:, :, 0]
    node_context[:, :, 6] = samples.node_context[:, :, 6]
    node_context[~np.isfinite(node_context)] = 0.0

    global_context, global_mean, global_std = standardize_train_val_test(samples.global_context, train_mask)
    stats = {
        "hist_mean": hist_mean.astype(np.float32),
        "hist_std": hist_std.astype(np.float32),
        "node_mean": node_mean.astype(np.float32),
        "node_std": node_std.astype(np.float32),
        "global_mean": global_mean,
        "global_std": global_std,
    }
    return hist.astype(np.float32), node_context.astype(np.float32), global_context.astype(np.float32), stats


def make_loader(
    hist: np.ndarray,
    node_context: np.ndarray,
    global_context: np.ndarray,
    samples: SampleArrays,
    event_aux: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(hist[mask]),
        torch.from_numpy(node_context[mask]),
        torch.from_numpy(global_context[mask]),
        torch.from_numpy(samples.y_residual[mask]),
        torch.from_numpy(samples.y_mask[mask]),
        torch.from_numpy(event_aux[mask]),
        torch.from_numpy(samples.y_node_affected[mask]),
        torch.from_numpy(samples.node_valid[mask]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(
    model: CandidateSTGNNResidual,
    batch: tuple[torch.Tensor, ...],
    event_aux_weight: float,
    node_aux_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        hist,
        node_context,
        global_context,
        y,
        y_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    pred_y, pred_event_aux, pred_node_logits = model(hist, node_context, global_context)
    res = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = (res * y_mask).sum() / y_mask.sum().clamp_min(1.0)
    event_aux_loss = nn.functional.smooth_l1_loss(pred_event_aux, event_aux)
    node_bce = nn.functional.binary_cross_entropy_with_logits(
        pred_node_logits,
        node_affected,
        reduction="none",
    )
    node_aux_loss = (node_bce * node_valid).sum() / node_valid.sum().clamp_min(1.0)
    loss = residual_loss + event_aux_weight * event_aux_loss + node_aux_weight * node_aux_loss
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "event_aux_loss": float(event_aux_loss.detach().cpu()),
        "node_aux_loss": float(node_aux_loss.detach().cpu()),
    }


def evaluate_loader(
    model: CandidateSTGNNResidual,
    loader: DataLoader,
    event_aux_weight: float,
    node_aux_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "residual_loss": 0.0, "event_aux_loss": 0.0, "node_aux_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            loss, parts = compute_loss(model, batch, event_aux_weight, node_aux_weight, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def predict_residuals(
    model: CandidateSTGNNResidual,
    hist: np.ndarray,
    node_context: np.ndarray,
    global_context: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(hist), batch_size):
            hb = torch.from_numpy(hist[start : start + batch_size]).to(device)
            nb = torch.from_numpy(node_context[start : start + batch_size]).to(device)
            gb = torch.from_numpy(global_context[start : start + batch_size]).to(device)
            pred_y, _, _ = model(hb, nb, gb)
            preds.append(pred_y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_forecast_metrics(
    samples: SampleArrays,
    pred_residual: np.ndarray,
    sample_mask: np.ndarray,
    residual_beta: float,
) -> dict[str, float]:
    pred = samples.normal_pred[sample_mask] + residual_beta * pred_residual * samples.future_scale[sample_mask]
    actual = samples.actual_future[sample_mask]
    scale = samples.future_scale[sample_mask]
    y_mask = samples.y_mask[sample_mask].astype(bool)
    node_affected = samples.y_node_affected[sample_mask].astype(bool)
    affected_mask = y_mask & node_affected[:, None, :, None]
    unaffected_mask = (
        y_mask
        & (~node_affected[:, None, :, None])
        & samples.node_valid[sample_mask][:, None, :, None].astype(bool)
    )

    def pair(mask: np.ndarray, prefix: str) -> dict[str, float]:
        model_mae = robust_mae_from_arrays(pred, actual, scale, mask)
        baseline_mae = robust_mae_from_arrays(samples.normal_pred[sample_mask], actual, scale, mask)
        if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
            improvement = float("nan")
        else:
            improvement = 100.0 * (baseline_mae - model_mae) / baseline_mae
        return {
            f"{prefix}_model_robust_mae": model_mae,
            f"{prefix}_baseline_robust_mae": baseline_mae,
            f"{prefix}_improvement_pct": improvement,
        }

    out = {}
    out.update(pair(y_mask, "all_candidates"))
    out.update(pair(affected_mask, "affected_candidates"))
    out.update(pair(unaffected_mask, "unaffected_candidates"))
    out["affected_node_rate"] = float(node_affected[samples.node_valid[sample_mask].astype(bool)].mean())
    return out


def select_residual_beta(
    samples: SampleArrays,
    pred_residual_all: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for beta in np.linspace(0.0, 1.0, 21):
        metrics = compute_forecast_metrics(
            samples=samples,
            pred_residual=pred_residual_all[val_mask],
            sample_mask=val_mask,
            residual_beta=float(beta),
        )
        rows.append({"residual_beta": float(beta), **metrics})
    df = pd.DataFrame(rows)
    idx = df["all_candidates_model_robust_mae"].idxmin()
    return float(df.loc[idx, "residual_beta"]), df


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Candidate STGNN residual model training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    samples: SampleArrays,
    log_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    region_metrics: pd.DataFrame,
    residual_beta: float,
) -> None:
    test = metrics["test"]
    lines = ["# 候选节点 STGNN 事故残差模型", ""]
    lines.append("## 一句话结论")
    lines.append("")
    lines.append(
        "这一版把候选节点 MLP 升级为轻量 STGNN：每个节点先用 GRU 编码历史 residual，"
        "再基于 signed postmile 距离构造局部图进行空间传播，最后预测未来事故 residual。"
    )
    lines.append("")
    lines.append(
        f"测试集上，全候选邻域 robust MAE 从 `{test['all_candidates_baseline_robust_mae']:.4f}` "
        f"降到 `{test['all_candidates_model_robust_mae']:.4f}`，"
        f"相对提升 `{test['all_candidates_improvement_pct']:.2f}%`。"
    )
    lines.append("")
    lines.append(
        f"只看受影响候选节点，robust MAE 从 `{test['affected_candidates_baseline_robust_mae']:.4f}` "
        f"降到 `{test['affected_candidates_model_robust_mae']:.4f}`，"
        f"相对提升 `{test['affected_candidates_improvement_pct']:.2f}%`。"
    )
    lines.append("")

    lines.append("## 和候选节点 MLP 的区别")
    lines.append("")
    lines.append("- MLP：把候选节点拍平成一个向量，空间关系只能隐式学习。")
    lines.append("- STGNN：保留节点维度，用 postmile 距离和上下游相对位置构造局部图，显式做空间传播。")
    lines.append("- 两者都不使用 impact label 来选择输入节点。")
    lines.append("")

    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 区域: {', '.join(args.regions)}")
    lines.append(f"- input_steps: {args.input_steps}")
    lines.append(f"- horizon_steps: {args.horizon_steps}")
    lines.append(f"- max_candidate_nodes: {args.max_candidate_nodes}")
    lines.append(f"- hidden_dim: {args.hidden_dim}")
    lines.append(f"- graph_layers: {args.graph_layers}")
    lines.append(f"- graph_sigma: {args.graph_sigma}")
    lines.append(f"- event_aux_weight: {args.event_aux_weight}")
    lines.append(f"- node_aux_weight: {args.node_aux_weight}")
    lines.append(f"- residual_beta: {residual_beta:.2f}")
    lines.append("")

    split_counts = pd.Series(samples.split).value_counts().rename_axis("split").reset_index(name="samples")
    lines.append("## 样本数量")
    lines.append("")
    lines.append(split_counts.to_markdown(index=False))
    lines.append("")

    metric_df = pd.DataFrame([{"split": split, **values} for split, values in metrics.items()])
    keep_cols = [
        "split",
        "all_candidates_model_robust_mae",
        "all_candidates_baseline_robust_mae",
        "all_candidates_improvement_pct",
        "affected_candidates_model_robust_mae",
        "affected_candidates_baseline_robust_mae",
        "affected_candidates_improvement_pct",
        "affected_node_rate",
    ]
    lines.append("## 预测指标")
    lines.append("")
    lines.append(metric_df[keep_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if not region_metrics.empty:
        lines.append("## 各地区测试集指标")
        lines.append("")
        lines.append(region_metrics.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    best = log_df.loc[log_df["val_loss"].idxmin()]
    lines.append("## 训练情况")
    lines.append("")
    lines.append(f"- 最佳轮数 best_epoch: {int(best['epoch'])}")
    lines.append(f"- 最佳验证损失 best_val_loss: {best['val_loss']:.4f}")
    lines.append("")

    lines.append("## 仍然存在的限制")
    lines.append("")
    lines.append("- 这个 STGNN 仍然是轻量原型，局部图来自 signed postmile 距离，还不是完整路网拓扑。")
    lines.append("- 当前最多保留 16 个候选节点，后续应扩展到完整候选集合。")
    lines.append("- 需要和 candidate MLP、no-aux STGNN、strong forecasting backbone 做系统消融。")
    lines.append("")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    event_root = args.event_root.resolve()
    raw_label_dir = args.raw_label_dir.resolve()

    print(f"using device: {device}", flush=True)
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)

    parts = []
    for region_name in args.regions:
        parts.append(
            build_region_samples(
                region_name=region_name,
                data_dir=data_dir,
                event_root=event_root,
                raw_label_dir=raw_label_dir,
                meta=meta,
                inc=inc,
                args=args,
            )
        )
    samples = concat_samples(parts)
    if samples.hist_residual.size == 0:
        raise RuntimeError("No samples were built.")

    train_mask_full = samples.split == "train"
    val_mask = samples.split == "val"
    test_mask = samples.split == "test"
    train_mask = cap_train_samples(train_mask_full, args.max_train_samples, args.seed)

    hist, node_context, global_context, input_stats = standardize_structured_inputs(samples, train_mask)
    event_aux_std, event_aux_mean, event_aux_stddev = standardize_aux(samples.y_event_aux_raw, train_mask)

    train_loader = make_loader(
        hist=hist,
        node_context=node_context,
        global_context=global_context,
        samples=samples,
        event_aux=event_aux_std,
        mask=train_mask,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        hist=hist,
        node_context=node_context,
        global_context=global_context,
        samples=samples,
        event_aux=event_aux_std,
        mask=val_mask,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CandidateSTGNNResidual(
        channels=len(CHANNELS),
        node_context_dim=node_context.shape[-1],
        global_context_dim=global_context.shape[-1],
        horizon_steps=args.horizon_steps,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        graph_sigma=args.graph_sigma,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _ = compute_loss(
                model=model,
                batch=batch,
                event_aux_weight=args.event_aux_weight,
                node_aux_weight=args.node_aux_weight,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            event_aux_weight=args.event_aux_weight,
            node_aux_weight=args.node_aux_weight,
            device=device,
        )
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_residual_loss": val_metrics["residual_loss"],
                "val_event_aux_loss": val_metrics["event_aux_loss"],
                "val_node_aux_loss": val_metrics["node_aux_loss"],
            }
        )
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_metrics['loss']:.4f}", flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    pred_all = predict_residuals(
        model=model,
        hist=hist,
        node_context=node_context,
        global_context=global_context,
        batch_size=args.batch_size,
        device=device,
    )
    residual_beta, beta_df = select_residual_beta(samples=samples, pred_residual_all=pred_all, val_mask=val_mask)
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)

    metrics: dict[str, dict[str, float]] = {}
    for name, mask in [("train", train_mask_full), ("val", val_mask), ("test", test_mask)]:
        metrics[name] = compute_forecast_metrics(
            samples=samples,
            pred_residual=pred_all[mask],
            sample_mask=mask,
            residual_beta=residual_beta,
        )

    region_rows = []
    for region_name in args.regions:
        mask = test_mask & (samples.region == region_name)
        if not np.any(mask):
            continue
        row = {"region": region_name, "samples": int(mask.sum())}
        row.update(
            compute_forecast_metrics(
                samples=samples,
                pred_residual=pred_all[mask],
                sample_mask=mask,
                residual_beta=residual_beta,
            )
        )
        region_rows.append(row)
    region_metrics = pd.DataFrame(region_rows)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_stats": input_stats,
            "event_aux_mean": event_aux_mean,
            "event_aux_std": event_aux_stddev,
            "args": json_safe_args(args),
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "region_metrics": region_rows,
                "samples": pd.Series(samples.split).value_counts().to_dict(),
                "residual_beta": residual_beta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["device"] = str(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_report(
        output_dir=output_dir,
        args=args,
        samples=samples,
        log_df=log_df,
        metrics=metrics,
        region_metrics=region_metrics,
        residual_beta=residual_beta,
    )
    print(f"wrote model outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
