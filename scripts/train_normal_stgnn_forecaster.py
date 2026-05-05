#!/usr/bin/env python3
"""Train a lightweight normal-traffic STGNN forecaster.

This script is the first learned normal branch for the impact-guided project.
It trains only on windows that are mostly outside incident masks, then learns a
correction over the transparent statistical blend baseline used by the current
residual models.

The model is intentionally compact:
1. A GRU encodes each sensor's recent multichannel traffic sequence.
2. Static graph propagation uses same-freeway/same-direction postmile neighbors.
3. A node-wise decoder predicts a normalized correction to the blend baseline.

Models are trained per region because Alameda, Contra Costa, and Orange have
different sensor counts.
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
from torch.utils.data import DataLoader, Dataset

from build_impact_labels import (
    build_baseline_valid_mask,
    build_matches,
    build_robust_baseline,
    load_incidents_2023,
    load_region_traffic,
    load_sensor_meta,
    normalize_direction,
    region_specs,
)
from train_impact_residual_model import choose_device, json_safe_args
from validate_forecast_error_against_impact import fit_blend_alphas


CHANNELS = ("flow", "occupancy", "speed")
CHANNEL_MINS = np.asarray([5.0, 0.005, 2.0], dtype=np.float32)
SPLITS = ("train", "val", "test")


@dataclass
class RegionData:
    traffic: np.ndarray
    times: pd.DatetimeIndex
    valid: np.ndarray
    baseline: np.ndarray
    scale: np.ndarray
    alphas: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray
    node_context: np.ndarray
    adj_all: np.ndarray
    adj_left: np.ndarray
    adj_right: np.ndarray
    split_starts: dict[str, np.ndarray]


class NormalWindowDataset(Dataset):
    def __init__(
        self,
        data: RegionData,
        starts: np.ndarray,
        input_steps: int,
        horizon_steps: int,
    ) -> None:
        self.data = data
        self.starts = np.asarray(starts, dtype=np.int32)
        self.input_steps = input_steps
        self.horizon_steps = horizon_steps
        self.day_kind = (data.times.dayofweek.to_numpy() >= 5).astype(np.int8)
        self.tod = ((data.times.hour.to_numpy() * 60 + data.times.minute.to_numpy()) // 5).astype(np.int16)

    def __len__(self) -> int:
        return int(self.starts.size)

    def _time_features(self, start: int) -> np.ndarray:
        ts = self.data.times[start]
        slot = (ts.hour * 60 + ts.minute) / 1440.0
        dow = ts.dayofweek / 7.0
        return np.asarray(
            [
                math.sin(2 * math.pi * slot),
                math.cos(2 * math.pi * slot),
                math.sin(2 * math.pi * dow),
                math.cos(2 * math.pi * dow),
                float(ts.dayofweek >= 5),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        start = int(self.starts[item])
        input_idx = np.arange(start - self.input_steps, start, dtype=np.int32)
        future_idx = np.arange(start, start + self.horizon_steps, dtype=np.int32)

        x_raw = self.data.traffic[input_idx]
        y_raw = self.data.traffic[future_idx]
        x_mask = np.isfinite(x_raw) & self.data.valid[input_idx, :, None]
        y_mask = np.isfinite(y_raw) & self.data.valid[future_idx, :, None]

        mean = self.data.channel_mean.reshape(1, 1, -1)
        std = self.data.channel_std.reshape(1, 1, -1)
        x = (x_raw - mean) / std
        y = (y_raw - mean) / std
        x[~x_mask] = 0.0
        y[~np.isfinite(y)] = 0.0

        future_base = self.data.baseline[self.day_kind[future_idx], self.tod[future_idx]]
        future_scale = self.data.scale[self.day_kind[future_idx], self.tod[future_idx]]
        prev_idx = start - 1
        prev_base = self.data.baseline[self.day_kind[prev_idx], self.tod[prev_idx]]
        last_obs = self.data.traffic[prev_idx].copy()
        last_valid = np.isfinite(last_obs) & self.data.valid[prev_idx, :, None]
        last_obs = np.where(last_valid, last_obs, prev_base)

        blend = np.empty_like(y_raw, dtype=np.float32)
        for h in range(self.horizon_steps):
            blend[h] = future_base[h] + self.data.alphas[h][None, :] * (last_obs - future_base[h])
        blend[~np.isfinite(blend)] = 0.0
        future_scale[~np.isfinite(future_scale)] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (y_raw - blend) / std
        y[~np.isfinite(y)] = 0.0

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(self._time_features(start)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(y_raw.astype(np.float32)),
            torch.from_numpy(y_mask.astype(np.float32)),
            torch.from_numpy(blend.astype(np.float32)),
            torch.from_numpy(future_scale.astype(np.float32)),
        )


class StaticGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, use_directional: bool) -> None:
        super().__init__()
        self.use_directional = use_directional
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.all_proj = nn.Linear(hidden_dim, hidden_dim)
        if use_directional:
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
    ) -> torch.Tensor:
        all_msg = torch.einsum("ij,bjh->bih", adj_all, h)
        out = self.self_proj(h) + self.all_proj(all_msg)
        if self.use_directional:
            left_msg = torch.einsum("ij,bjh->bih", adj_left, h)
            right_msg = torch.einsum("ij,bjh->bih", adj_right, h)
            out = out + self.left_proj(left_msg) + self.right_proj(right_msg)
        out = self.norm(out)
        out = torch.nn.functional.gelu(out)
        return self.dropout(out)


class NormalSTGNN(nn.Module):
    def __init__(
        self,
        channels: int,
        node_context: np.ndarray,
        adj_all: np.ndarray,
        adj_left: np.ndarray,
        adj_right: np.ndarray,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        use_directional: bool,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.channels = channels
        self.register_buffer("node_context", torch.from_numpy(node_context.astype(np.float32)))
        self.register_buffer("adj_all", torch.from_numpy(adj_all.astype(np.float32)))
        self.register_buffer("adj_left", torch.from_numpy(adj_left.astype(np.float32)))
        self.register_buffer("adj_right", torch.from_numpy(adj_right.astype(np.float32)))
        self.temporal_encoder = nn.GRU(channels, hidden_dim, batch_first=True)
        self.context_proj = nn.Sequential(
            nn.Linear(node_context.shape[1] + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [StaticGraphLayer(hidden_dim, dropout, use_directional) for _ in range(graph_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * channels),
        )
        final = self.decoder[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, time_context: torch.Tensor) -> torch.Tensor:
        batch_size, input_steps, nodes, channels = x.shape
        temporal_in = x.permute(0, 2, 1, 3).reshape(batch_size * nodes, input_steps, channels)
        _, h_last = self.temporal_encoder(temporal_in)
        h = h_last[-1].reshape(batch_size, nodes, -1)

        node_ctx = self.node_context.unsqueeze(0).expand(batch_size, -1, -1)
        time_ctx = time_context[:, None, :].expand(-1, nodes, -1)
        h = self.input_norm(h + self.context_proj(torch.cat([node_ctx, time_ctx], dim=-1)))

        for layer in self.graph_layers:
            h = h + layer(h, self.adj_all, self.adj_left, self.adj_right)

        out = self.decoder(h).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        return out.permute(0, 2, 1, 3).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/normal_stgnn_forecaster/first_pass"))
    parser.add_argument("--regions", nargs="+", default=["Alameda", "ContraCosta", "Orange"])
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--min-valid-fraction", type=float, default=0.90)
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument("--graph-sigma", type=float, default=1.5)
    parser.add_argument("--graph-directional", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-eval-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def numeric_series(series: pd.Series, default: float = 0.0) -> np.ndarray:
    vals = pd.to_numeric(series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    arr = vals.to_numpy(dtype=np.float32)
    arr[~np.isfinite(arr)] = default
    return arr


def standardize_feature(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(arr[finite].mean())
    std = float(arr[finite].std())
    if std < 1e-6:
        std = 1.0
    out = (arr - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def make_node_context(region_meta: pd.DataFrame) -> np.ndarray:
    direction = region_meta["Direction"].map(normalize_direction).fillna("").astype(str)
    features = [
        standardize_feature(pd.to_numeric(region_meta["Abs PM"], errors="coerce").to_numpy(np.float32)),
        standardize_feature(pd.to_numeric(region_meta["Lat"], errors="coerce").to_numpy(np.float32)),
        standardize_feature(pd.to_numeric(region_meta["Lng"], errors="coerce").to_numpy(np.float32)),
        numeric_series(region_meta["Design Speed Limit"], default=65.0) / 100.0,
        numeric_series(region_meta["Lane Width"], default=12.0) / 12.0,
        numeric_series(region_meta["Road Width"], default=24.0) / 50.0,
        (direction == "N").to_numpy(np.float32),
        (direction == "S").to_numpy(np.float32),
        (direction == "E").to_numpy(np.float32),
        (direction == "W").to_numpy(np.float32),
    ]
    return np.stack(features, axis=1).astype(np.float32)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = matrix.sum(axis=1, keepdims=True)
    denom[denom < 1e-6] = 1.0
    return (matrix / denom).astype(np.float32)


def build_static_graph(region_meta: pd.DataFrame, topk: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(region_meta)
    adj_all = np.zeros((n, n), dtype=np.float32)
    adj_left = np.zeros_like(adj_all)
    adj_right = np.zeros_like(adj_all)
    work = region_meta.copy()
    work["Direction"] = work["Direction"].map(normalize_direction)
    work["Fwy"] = pd.to_numeric(work["Fwy"], errors="coerce")
    work["Abs PM"] = pd.to_numeric(work["Abs PM"], errors="coerce")

    for _, group in work.groupby(["Fwy", "Direction"], dropna=True):
        idx = group.index.to_numpy(dtype=np.int32)
        pm = group["Abs PM"].to_numpy(dtype=np.float32)
        finite = np.isfinite(pm)
        idx = idx[finite]
        pm = pm[finite]
        if idx.size == 0:
            continue
        for pos, target in enumerate(idx):
            dist = np.abs(pm - pm[pos])
            order = np.argsort(dist)[: max(1, min(topk + 1, idx.size))]
            weights = np.exp(-dist[order] / max(sigma, 1e-6)).astype(np.float32)
            sources = idx[order]
            adj_all[target, sources] = np.maximum(adj_all[target, sources], weights)
            adj_left[target, sources[pm[order] < pm[pos]]] = weights[pm[order] < pm[pos]]
            adj_right[target, sources[pm[order] > pm[pos]]] = weights[pm[order] > pm[pos]]

    eye = np.eye(n, dtype=np.float32)
    adj_all = np.maximum(adj_all, eye)
    adj_left = np.maximum(adj_left, eye)
    adj_right = np.maximum(adj_right, eye)
    return row_normalize(adj_all), row_normalize(adj_left), row_normalize(adj_right)


def split_bounds(total_steps: int) -> dict[str, tuple[int, int]]:
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)
    return {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, total_steps)}


def filter_starts(
    starts: np.ndarray,
    valid: np.ndarray,
    input_steps: int,
    horizon_steps: int,
    min_valid_fraction: float,
) -> np.ndarray:
    kept: list[int] = []
    for start in starts:
        window = valid[start - input_steps : start + horizon_steps]
        if float(window.mean()) >= min_valid_fraction:
            kept.append(int(start))
    return np.asarray(kept, dtype=np.int32)


def cap_starts(starts: np.ndarray, limit: int, seed: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int32)
    if limit <= 0 or starts.size <= limit:
        return starts
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(starts, size=limit, replace=False)).astype(np.int32)


def make_split_starts(
    valid: np.ndarray,
    input_steps: int,
    horizon_steps: int,
    sample_stride: int,
    min_valid_fraction: float,
    max_train_samples: int,
    max_eval_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    total_steps = valid.shape[0]
    out: dict[str, np.ndarray] = {}
    for split, (lo, hi) in split_bounds(total_steps).items():
        start_lo = max(input_steps, lo)
        start_hi = min(hi - horizon_steps + 1, total_steps - horizon_steps + 1)
        if start_hi <= start_lo:
            out[split] = np.asarray([], dtype=np.int32)
            continue
        candidates = np.arange(start_lo, start_hi, sample_stride, dtype=np.int32)
        filtered = filter_starts(candidates, valid, input_steps, horizon_steps, min_valid_fraction)
        limit = max_train_samples if split == "train" else max_eval_samples
        out[split] = cap_starts(filtered, limit, seed + len(out))
    return out


def compute_channel_stats(traffic: np.ndarray, valid: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    train = traffic[:train_end]
    mask = valid[:train_end, :, None] & np.isfinite(train)
    sums = np.where(mask, train, 0.0).sum(axis=(0, 1), dtype=np.float64)
    counts = mask.sum(axis=(0, 1)).astype(np.float64)
    mean = sums / np.maximum(counts, 1.0)
    sq = np.where(mask, np.square(train), 0.0).sum(axis=(0, 1), dtype=np.float64)
    var = sq / np.maximum(counts, 1.0) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-6))
    std = np.maximum(std, CHANNEL_MINS)
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_region(region_name: str, args: argparse.Namespace, meta: pd.DataFrame, inc: pd.DataFrame) -> tuple[pd.DataFrame, RegionData]:
    spec = region_specs()[region_name]
    region_meta = meta[(meta["County"] == spec.county) & (meta["Type"] == "Mainline")].copy().reset_index(drop=True)
    region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)
    print(f"[{region_name}] loading traffic for {len(region_meta)} mainline sensors", flush=True)
    traffic, times = load_region_traffic(args.data_dir, region_node_idx)

    print(f"[{region_name}] building incident mask and normal baseline", flush=True)
    matches = build_matches(
        inc=inc,
        region_meta=region_meta,
        times=times,
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
        baseline_mask_extra_steps=args.baseline_mask_extra_steps,
    )
    valid = build_baseline_valid_mask(traffic.shape[:2], matches)
    bounds = split_bounds(traffic.shape[0])
    train_end = bounds["train"][1]
    train_valid = valid.copy()
    train_valid[train_end:, :] = False
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
    channel_mean, channel_std = compute_channel_stats(traffic, valid, train_end)
    split_starts = make_split_starts(
        valid=valid,
        input_steps=args.input_steps,
        horizon_steps=args.horizon_steps,
        sample_stride=args.sample_stride,
        min_valid_fraction=args.min_valid_fraction,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )
    adj_all, adj_left, adj_right = build_static_graph(region_meta, topk=args.graph_topk, sigma=args.graph_sigma)
    node_context = make_node_context(region_meta)
    data = RegionData(
        traffic=traffic,
        times=times,
        valid=valid,
        baseline=baseline,
        scale=scale,
        alphas=alphas,
        channel_mean=channel_mean,
        channel_std=channel_std,
        node_context=node_context,
        adj_all=adj_all,
        adj_left=adj_left,
        adj_right=adj_right,
        split_starts=split_starts,
    )
    return region_meta, data


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(
    model: NormalSTGNN,
    loader: DataLoader,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    mean = torch.as_tensor(channel_mean.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    std = torch.as_tensor(channel_std.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    sums = {
        "model_abs": 0.0,
        "blend_abs": 0.0,
        "model_sq": 0.0,
        "blend_sq": 0.0,
        "model_mape": 0.0,
        "blend_mape": 0.0,
        "model_robust": 0.0,
        "blend_robust": 0.0,
        "count": 0.0,
    }
    denom_floor = torch.as_tensor(CHANNEL_MINS.reshape(1, 1, 1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            x, time_ctx, _y_norm, y_raw, mask, blend, scale = [item.to(device) for item in batch]
            pred_correction = model(x, time_ctx)
            pred = blend + pred_correction * std
            diff_model = pred - y_raw
            diff_blend = blend - y_raw
            valid = mask.bool() & torch.isfinite(y_raw) & torch.isfinite(pred) & torch.isfinite(blend)
            count = float(valid.sum().detach().cpu())
            if count <= 0:
                continue
            abs_model = torch.abs(diff_model)
            abs_blend = torch.abs(diff_blend)
            denom = torch.maximum(torch.abs(y_raw), denom_floor)
            safe_scale = torch.where(torch.isfinite(scale) & (scale > 1e-6), scale, torch.ones_like(scale))
            sums["model_abs"] += float(abs_model[valid].sum().detach().cpu())
            sums["blend_abs"] += float(abs_blend[valid].sum().detach().cpu())
            sums["model_sq"] += float(torch.square(diff_model)[valid].sum().detach().cpu())
            sums["blend_sq"] += float(torch.square(diff_blend)[valid].sum().detach().cpu())
            sums["model_mape"] += float((abs_model / denom)[valid].sum().detach().cpu())
            sums["blend_mape"] += float((abs_blend / denom)[valid].sum().detach().cpu())
            sums["model_robust"] += float((abs_model / safe_scale)[valid].sum().detach().cpu())
            sums["blend_robust"] += float((abs_blend / safe_scale)[valid].sum().detach().cpu())
            sums["count"] += count

    c = max(sums["count"], 1.0)
    model_mae = sums["model_abs"] / c
    blend_mae = sums["blend_abs"] / c
    model_robust = sums["model_robust"] / c
    blend_robust = sums["blend_robust"] / c
    return {
        "model_mae": model_mae,
        "blend_mae": blend_mae,
        "mae_improvement_pct": 100.0 * (blend_mae - model_mae) / blend_mae if blend_mae > 0 else float("nan"),
        "model_rmse": math.sqrt(sums["model_sq"] / c),
        "blend_rmse": math.sqrt(sums["blend_sq"] / c),
        "model_mape": 100.0 * sums["model_mape"] / c,
        "blend_mape": 100.0 * sums["blend_mape"] / c,
        "model_robust_mae": model_robust,
        "blend_robust_mae": blend_robust,
        "robust_improvement_pct": 100.0 * (blend_robust - model_robust) / blend_robust if blend_robust > 0 else float("nan"),
        "valid_values": sums["count"],
    }


def save_training_plot(log_df: pd.DataFrame, output_dir: Path, region_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized SmoothL1")
    ax.set_title(f"Normal STGNN training: {region_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def train_region(region_name: str, args: argparse.Namespace, meta: pd.DataFrame, inc: pd.DataFrame, device: torch.device) -> dict[str, object]:
    region_dir = args.output_dir.resolve() / region_name
    region_dir.mkdir(parents=True, exist_ok=True)
    region_meta, data = prepare_region(region_name, args, meta, inc)
    print(
        f"[{region_name}] starts: "
        + ", ".join(f"{split}={len(starts)}" for split, starts in data.split_starts.items()),
        flush=True,
    )

    train_loader = DataLoader(
        NormalWindowDataset(data, data.split_starts["train"], args.input_steps, args.horizon_steps),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        NormalWindowDataset(data, data.split_starts["val"], args.input_steps, args.horizon_steps),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = NormalSTGNN(
        channels=len(CHANNELS),
        node_context=data.node_context,
        adj_all=data.adj_all,
        adj_left=data.adj_left,
        adj_right=data.adj_right,
        horizon_steps=args.horizon_steps,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        use_directional=args.graph_directional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for batch in train_loader:
            x, time_ctx, y_norm, _y_raw, mask, _blend, _scale = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, time_ctx)
            loss = masked_smooth_l1(pred, y_norm, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        train_loss = total / max(batches, 1)
        val_metrics = evaluate(model, val_loader, data.channel_mean, data.channel_std, device)
        val_loss = val_metrics["model_robust_mae"]
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"[{region_name}] epoch {epoch:03d} train={train_loss:.4f} val_robust={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(region_dir / "training_log.csv", index=False)
    save_training_plot(log_df, region_dir, region_name)

    metrics: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        loader = DataLoader(
            NormalWindowDataset(data, data.split_starts[split], args.input_steps, args.horizon_steps),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        metrics[split] = evaluate(model, loader, data.channel_mean, data.channel_std, device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": json_safe_args(args),
            "region": region_name,
            "channel_mean": data.channel_mean,
            "channel_std": data.channel_std,
            "node_context": data.node_context,
            "region_sensor_ids": region_meta["station_id"].to_numpy(dtype=np.int64),
            "region_node_idx": region_meta["node_idx"].to_numpy(dtype=np.int32),
        },
        region_dir / "model.pt",
    )
    with (region_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "region": region_name,
                "samples": {split: int(len(starts)) for split, starts in data.split_starts.items()},
                "metrics": metrics,
                "channel_mean": data.channel_mean.tolist(),
                "channel_std": data.channel_std.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_lines = [
        f"# Normal STGNN Forecaster: {region_name}",
        "",
        "## Samples",
        "",
        pd.DataFrame([{"split": k, "samples": len(v)} for k, v in data.split_starts.items()]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        pd.DataFrame([{"split": k, **v} for k, v in metrics.items()]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "- The blend baseline is the same transparent normal predictor used by the current residual branch.",
        "- The learned model predicts a correction over that blend baseline, so zero correction recovers the baseline.",
        "- Training and evaluation losses are masked by incident-free node-time labels.",
    ]
    with (region_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return {
        "region": region_name,
        "samples": {split: int(len(starts)) for split, starts in data.split_starts.items()},
        "metrics": metrics,
        "output_dir": str(region_dir),
    }


def write_global_summary(output_dir: Path, results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        test_metrics = result["metrics"]["test"]  # type: ignore[index]
        rows.append({"region": result["region"], **test_metrics})  # type: ignore[arg-type]
    table = pd.DataFrame(rows)
    lines = [
        "# Learned Normal STGNN Summary",
        "",
        "This run trains a learned normal-traffic branch on mostly non-incident windows.",
        "The next integration step is to use these normal forecasts as the baseline for incident residual learning.",
        "",
        "## Test Metrics",
        "",
        table.to_markdown(index=False, floatfmt=".4f") if not table.empty else "No test metrics.",
        "",
        "## Region Outputs",
        "",
    ]
    for result in results:
        lines.append(f"- {result['region']}: `{result['output_dir']}`")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)

    meta = load_sensor_meta(args.data_dir)
    inc = load_incidents_2023(args.data_dir)
    results = []
    for region_name in args.regions:
        results.append(train_region(region_name, args, meta, inc, device))

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        config = json_safe_args(args)
        config["device"] = str(device)
        json.dump(config, f, indent=2, ensure_ascii=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_global_summary(output_dir, results)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
