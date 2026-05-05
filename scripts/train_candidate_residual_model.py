#!/usr/bin/env python3
"""Train a candidate-neighborhood incident residual forecaster.

This script removes the main limitation of the first residual scaffold: it no
longer selects the top-k most impacted nodes by derived labels. Instead, each
incident is represented by a fixed-size spatial candidate neighborhood around
the incident anchor. Nodes are selected and ordered by road position/distance,
which is available from incident metadata and sensor metadata at inference.

The model still remains deliberately lightweight. It is meant to validate the
candidate-node formulation before we move to a full spatiotemporal graph model.
"""

from __future__ import annotations

import argparse
import json
import math
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
from train_impact_residual_model import (
    choose_device,
    fit_blend_alphas,
    json_safe_args,
    make_event_context,
    make_time_features,
    split_name,
    standardize_aux,
    standardize_train_val_test,
)
from validate_forecast_error_against_impact import parse_incident_ids


CHANNELS = ("flow", "occupancy", "speed")


@dataclass
class SampleArrays:
    x: np.ndarray
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


class CandidateResidualMLP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        residual_dim: int,
        event_aux_dim: int,
        node_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(hidden_dim, residual_dim)
        self.event_aux_head = nn.Linear(hidden_dim, event_aux_dim)
        self.node_aux_head = nn.Linear(hidden_dim, node_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.residual_head(h), self.event_aux_head(h), self.node_aux_head(h)


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
        default=Path("outputs/candidate_residual_model/first_pass"),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["Alameda", "ContraCosta", "Orange"],
    )
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument(
        "--max-candidate-nodes",
        type=int,
        default=16,
        help="Nearest candidate sensors kept per event. Selection is spatial, not impact-ranked.",
    )
    parser.add_argument(
        "--sample-offsets",
        nargs="+",
        type=int,
        default=[0, 6, 12],
    )
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-aux-weight", type=float, default=0.10)
    parser.add_argument("--node-aux-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap for faster experiments; 0 means no cap.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    return parser.parse_args()


def robust_mae_from_arrays(pred: np.ndarray, actual: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(pred - actual) / scale
    vals = err[np.isfinite(err) & mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def side_to_signed_pm(side: object, pm_dist: float) -> float:
    text = str(side)
    if text == "upstream":
        return -float(pm_dist)
    if text == "downstream":
        return float(pm_dist)
    return 0.0


def build_candidate_lookup(raw_nodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_nodes = raw_nodes.copy()
    raw_nodes["incident_id"] = raw_nodes["incident_id"].astype(str)
    raw_nodes["signed_pm_dist"] = [
        side_to_signed_pm(side, pm)
        for side, pm in zip(raw_nodes["side"], raw_nodes["pm_dist"])
    ]
    lookup: dict[str, pd.DataFrame] = {}
    for incident_id, group in raw_nodes.groupby("incident_id", sort=False):
        grouped = (
            group.groupby("region_node_idx", as_index=False)
            .agg(
                sensor_id=("sensor_id", "first"),
                pm_dist=("pm_dist", "min"),
                signed_pm_dist=("signed_pm_dist", "median"),
                side=("side", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                affected=("affected", "max"),
            )
        )
        lookup[str(incident_id)] = grouped
    return lookup


def select_candidate_nodes(
    incident_ids: list[str],
    candidate_lookup: dict[str, pd.DataFrame],
    anchor_region_idx: int,
    max_nodes: int,
) -> pd.DataFrame:
    pieces = []
    for incident_id in incident_ids:
        item = candidate_lookup.get(incident_id)
        if item is not None and not item.empty:
            pieces.append(item)
    if pieces:
        candidates = pd.concat(pieces, ignore_index=True)
        candidates = (
            candidates.groupby("region_node_idx", as_index=False)
            .agg(
                sensor_id=("sensor_id", "first"),
                pm_dist=("pm_dist", "min"),
                signed_pm_dist=("signed_pm_dist", "median"),
                side=("side", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                affected=("affected", "max"),
            )
        )
    else:
        candidates = pd.DataFrame(
            [
                {
                    "region_node_idx": int(anchor_region_idx),
                    "sensor_id": -1,
                    "pm_dist": 0.0,
                    "signed_pm_dist": 0.0,
                    "side": "at_incident",
                    "affected": 0,
                }
            ]
        )

    if not (candidates["region_node_idx"] == int(anchor_region_idx)).any():
        anchor_row = pd.DataFrame(
            [
                {
                    "region_node_idx": int(anchor_region_idx),
                    "sensor_id": -1,
                    "pm_dist": 0.0,
                    "signed_pm_dist": 0.0,
                    "side": "at_incident",
                    "affected": 0,
                }
            ]
        )
        candidates = pd.concat([candidates, anchor_row], ignore_index=True)

    # If there are too many candidates, keep the nearest road-neighborhood
    # nodes. This is spatial selection, not impact-label selection.
    if len(candidates) > max_nodes:
        candidates = candidates.sort_values(["pm_dist", "region_node_idx"]).head(max_nodes)
    candidates = candidates.sort_values(["signed_pm_dist", "pm_dist", "region_node_idx"]).reset_index(drop=True)
    return candidates


def build_node_context(candidates: pd.DataFrame, max_nodes: int, pm_radius: float, anchor_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_idx = np.full(max_nodes, int(anchor_idx), dtype=np.int32)
    node_valid = np.zeros(max_nodes, dtype=np.float32)
    node_affected = np.zeros(max_nodes, dtype=np.float32)
    ctx = np.zeros((max_nodes, 8), dtype=np.float32)

    n = min(max_nodes, len(candidates))
    for pos, row in enumerate(candidates.itertuples(index=False)):
        if pos >= n:
            break
        idx = int(row.region_node_idx)
        signed_pm = float(row.signed_pm_dist)
        side = str(row.side)
        node_idx[pos] = idx
        node_valid[pos] = 1.0
        node_affected[pos] = float(row.affected)
        ctx[pos] = np.asarray(
            [
                signed_pm / max(pm_radius, 1e-6),
                abs(signed_pm) / max(pm_radius, 1e-6),
                float(side == "upstream"),
                float(side == "downstream"),
                float(side == "at_incident"),
                float(idx == int(anchor_idx)),
                node_valid[pos],
                float(pos) / max(max_nodes - 1, 1),
            ],
            dtype=np.float32,
        )
    return node_idx, node_valid, node_affected, ctx


def make_event_aux(row: object) -> np.ndarray:
    values = np.asarray(
        [
            float(np.log1p(max(row.severity_any_z_auc_topk, 0.0))),
            float(row.recovery_time_min) / 180.0,
            float(np.log1p(max(row.spread_nodes, 0.0))),
        ],
        dtype=np.float32,
    )
    values[~np.isfinite(values)] = 0.0
    return values


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

    print(f"[{region_name}] building candidate-neighborhood samples", flush=True)
    events = pd.read_csv(event_root / region_name / "event_labels.csv")
    raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
    candidate_lookup = build_candidate_lookup(raw_nodes)

    x_rows: list[np.ndarray] = []
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
            metric_mask = y_mask & np.isfinite(normal_pred) & np.isfinite(actual_future) & np.isfinite(fut_scale)
            if not metric_mask.any():
                continue

            event_features = make_event_context(row, offset, args.horizon_steps)
            time_features = make_time_features(times, sample_start)
            x = np.concatenate(
                [
                    hist_res_z.reshape(-1),
                    node_ctx.reshape(-1),
                    event_features,
                    time_features,
                ]
            ).astype(np.float32)
            x[~np.isfinite(x)] = 0.0

            normal_pred[:, ~valid_node_mask, :] = 0.0
            actual_future[:, ~valid_node_mask, :] = 0.0
            fut_scale[:, ~valid_node_mask, :] = 1.0

            x_rows.append(x)
            y_rows.append(y_res_z.reshape(-1).astype(np.float32))
            y_mask_rows.append(y_mask.reshape(-1).astype(np.float32))
            event_aux_rows.append(make_event_aux(row))
            node_affected_rows.append(node_affected.astype(np.float32))
            node_valid_rows.append(node_valid.astype(np.float32))
            normal_rows.append(normal_pred.astype(np.float32))
            actual_rows.append(actual_future.astype(np.float32))
            scale_rows.append(fut_scale.astype(np.float32))
            split_rows.append(split_name(sample_start, total_steps))
            region_rows.append(region_name)

    return SampleArrays(
        x=np.asarray(x_rows, dtype=np.float32),
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
        x=np.concatenate([p.x for p in parts], axis=0),
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


def cap_train_samples(mask: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0:
        return mask
    idx = np.flatnonzero(mask)
    if idx.size <= max_samples:
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(idx, size=max_samples, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[keep] = True
    return out


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    event_aux: np.ndarray,
    node_affected: np.ndarray,
    node_valid: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x[mask]),
        torch.from_numpy(y[mask]),
        torch.from_numpy(y_mask[mask]),
        torch.from_numpy(event_aux[mask]),
        torch.from_numpy(node_affected[mask]),
        torch.from_numpy(node_valid[mask]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(
    model: CandidateResidualMLP,
    batch: tuple[torch.Tensor, ...],
    event_aux_weight: float,
    node_aux_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    x, y, y_mask, event_aux, node_affected, node_valid = [item.to(device) for item in batch]
    pred_y, pred_event_aux, pred_node_logits = model(x)

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
    model: CandidateResidualMLP,
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
    model: CandidateResidualMLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred_y, _, _ = model(xb)
            preds.append(pred_y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_forecast_metrics(
    samples: SampleArrays,
    pred_residual_flat: np.ndarray,
    sample_mask: np.ndarray,
    horizon_steps: int,
    max_candidate_nodes: int,
    residual_beta: float,
) -> dict[str, float]:
    shape = (-1, horizon_steps, max_candidate_nodes, len(CHANNELS))
    pred_residual = pred_residual_flat.reshape(shape)
    pred = samples.normal_pred[sample_mask] + residual_beta * pred_residual * samples.future_scale[sample_mask]
    actual = samples.actual_future[sample_mask]
    scale = samples.future_scale[sample_mask]
    y_mask = samples.y_mask[sample_mask].reshape(shape).astype(bool)
    node_affected = samples.y_node_affected[sample_mask].astype(bool)
    affected_mask = y_mask & node_affected[:, None, :, None]
    unaffected_mask = y_mask & (~node_affected[:, None, :, None]) & samples.node_valid[sample_mask][:, None, :, None].astype(bool)

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
    pred_residual_flat: np.ndarray,
    val_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for beta in np.linspace(0.0, 1.0, 21):
        metrics = compute_forecast_metrics(
            samples=samples,
            pred_residual_flat=pred_residual_flat[val_mask],
            sample_mask=val_mask,
            horizon_steps=args.horizon_steps,
            max_candidate_nodes=args.max_candidate_nodes,
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
    ax.set_title("Candidate residual model training")
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
    lines = ["# 候选节点事故残差模型", ""]
    lines.append("## 一句话结论")
    lines.append("")
    test = metrics["test"]
    lines.append(
        "这一版不再用影响标签挑选 top-k 传感器，而是在事故附近按空间距离选取固定大小的候选节点邻域，"
        "让模型对整个候选邻域预测事故残差。"
    )
    lines.append("")
    lines.append(
        f"测试集上，全候选邻域 robust MAE 从 `{test['all_candidates_baseline_robust_mae']:.4f}` "
        f"降到 `{test['all_candidates_model_robust_mae']:.4f}`，"
        f"相对提升 `{test['all_candidates_improvement_pct']:.2f}%`。"
    )
    lines.append("")
    lines.append(
        f"如果只看候选邻域中确实受影响的节点，robust MAE 从 "
        f"`{test['affected_candidates_baseline_robust_mae']:.4f}` 降到 "
        f"`{test['affected_candidates_model_robust_mae']:.4f}`，"
        f"相对提升 `{test['affected_candidates_improvement_pct']:.2f}%`。"
    )
    lines.append("")

    lines.append("## 和上一版 top-k 实验的区别")
    lines.append("")
    lines.append("- 上一版：用节点影响标签选出最明显受影响的 5 个传感器，再评估残差修正效果。")
    lines.append("- 这一版：不用影响强度排序，只按事故位置附近的空间邻域选传感器。")
    lines.append("- 因此这一版更接近真实推理场景，但仍然是 MLP scaffold，不是最终 STGNN。")
    lines.append("")

    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 区域: {', '.join(args.regions)}")
    lines.append(f"- 历史输入步数 input_steps: {args.input_steps}")
    lines.append(f"- 未来预测步数 horizon_steps: {args.horizon_steps}")
    lines.append(f"- 最大候选节点数 max_candidate_nodes: {args.max_candidate_nodes}")
    lines.append(f"- 候选节点选择方式: 按事故点附近 postmile 空间距离选取，不按影响强度选取")
    lines.append(f"- 事故内采样偏移 sample_offsets: {args.sample_offsets}")
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
    lines.append("- 当前模型还是 MLP，空间传播只通过固定顺序的候选节点隐式表达，没有显式图结构。")
    lines.append("- 当前默认最多保留 16 个候选节点，是为了控制内存和先验证口径；后续应改成流式 Dataset 或 STGNN 来覆盖完整候选集合。")
    lines.append("- 现在的 node affected 标签只作为辅助监督和分层评估使用，没有用于选择候选节点。")
    lines.append("")

    lines.append("## 下一步建议")
    lines.append("")
    lines.append("- 做消融：关闭 event/node impact auxiliary loss，看 impact supervision 是否真的带来增益。")
    lines.append("- 把 MLP 换成 candidate-node STGNN，让模型显式学习上下游传播。")
    lines.append("- 进一步从最多 16 个候选节点扩展到完整候选集合。")
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
    if samples.x.size == 0:
        raise RuntimeError("No samples were built.")

    train_mask_full = samples.split == "train"
    val_mask = samples.split == "val"
    test_mask = samples.split == "test"
    train_mask = cap_train_samples(train_mask_full, args.max_train_samples, args.seed)

    x_std, x_mean, x_stddev = standardize_train_val_test(samples.x, train_mask)
    event_aux_std, event_aux_mean, event_aux_stddev = standardize_aux(samples.y_event_aux_raw, train_mask)

    train_loader = make_loader(
        x=x_std,
        y=samples.y_residual,
        y_mask=samples.y_mask,
        event_aux=event_aux_std,
        node_affected=samples.y_node_affected,
        node_valid=samples.node_valid,
        mask=train_mask,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        x=x_std,
        y=samples.y_residual,
        y_mask=samples.y_mask,
        event_aux=event_aux_std,
        node_affected=samples.y_node_affected,
        node_valid=samples.node_valid,
        mask=val_mask,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CandidateResidualMLP(
        x_dim=x_std.shape[1],
        residual_dim=samples.y_residual.shape[1],
        event_aux_dim=event_aux_std.shape[1],
        node_dim=args.max_candidate_nodes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
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

    pred_all = predict_residuals(model, x_std, args.batch_size, device)
    residual_beta, beta_df = select_residual_beta(
        samples=samples,
        pred_residual_flat=pred_all,
        val_mask=val_mask,
        args=args,
    )
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)

    metrics: dict[str, dict[str, float]] = {}
    for name, mask in [("train", train_mask_full), ("val", val_mask), ("test", test_mask)]:
        metrics[name] = compute_forecast_metrics(
            samples=samples,
            pred_residual_flat=pred_all[mask],
            sample_mask=mask,
            horizon_steps=args.horizon_steps,
            max_candidate_nodes=args.max_candidate_nodes,
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
                pred_residual_flat=pred_all[mask],
                sample_mask=mask,
                horizon_steps=args.horizon_steps,
                max_candidate_nodes=args.max_candidate_nodes,
                residual_beta=residual_beta,
            )
        )
        region_rows.append(row)
    region_metrics = pd.DataFrame(region_rows)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_stddev,
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
