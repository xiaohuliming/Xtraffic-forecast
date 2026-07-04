#!/usr/bin/env python3
"""Ablation A: v0c original model on XTraffic 2024.

This is a standalone pilot script for the "label-branch" ablation. It uses
2024 raw arrays directly, builds a simple train-only climatology baseline, then
trains the current best single-branch v0c idea:

    prediction = adaptive_anchor + GraphWaveNet(residual_history)

Notes:
  - 2024 monthly arrays are shaped (node, time, channel), unlike 2023.
  - The baseline here is intentionally shared by A and B. It is not meant to
    replace the full 2023 cache pipeline used in the main report.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROJECT = WORKSPACE / "Xtraffic-forecast"
DEFAULT_DATA = WORKSPACE / "data"

sys.path.insert(0, str(PROJECT / "baselines" / "GraphWaveNet"))
from model import gwnet  # noqa: E402


REGION_COUNTY = {
    "Alameda": "Alameda",
    "ContraCosta": "Contra Costa",
    "Orange": "Orange",
}
CHANNEL_FLOW = 0
CHANNEL_OCC = 1
CHANNEL_SPEED = 2


@dataclass
class MatchedIncident:
    start_idx: int
    anchor_idx: int
    candidate_idx: np.ndarray


@dataclass
class RegionBundle:
    region: str
    traffic: np.ndarray
    mask: np.ndarray
    baseline: np.ndarray
    scale: np.ndarray
    day_kind: np.ndarray
    tod: np.ndarray
    dow: np.ndarray
    samples: list[dict]
    edge_index: np.ndarray
    region_meta: pd.DataFrame


def parse_offsets(text: str) -> list[int]:
    text = str(text).strip()
    if ":" in text:
        parts = [int(x) for x in text.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
        raise ValueError(f"Bad offset range: {text}")
    return [int(x) for x in text.split(",") if x.strip()]


def normalize_direction(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()[:1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sensor_meta(data_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "sensor_meta_feature.csv", sep="\t", low_memory=False)
    node_order = np.load(data_dir / "node_order.npy")
    ordered = meta.set_index("station_id").loc[node_order].reset_index()
    ordered = ordered.rename(columns={"index": "station_id"})
    ordered["node_idx"] = np.arange(len(ordered), dtype=np.int32)
    ordered["Direction"] = ordered["Direction"].map(normalize_direction)
    for col in ["Fwy", "Abs PM", "Lat", "Lng"]:
        ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
    return ordered


def region_mainline_meta(meta: pd.DataFrame, region: str, max_nodes: int = 0) -> pd.DataFrame:
    county = REGION_COUNTY[region]
    sub = meta[(meta["County"] == county) & (meta["Type"] == "Mainline")].copy()
    sub = sub.reset_index(drop=True)
    if max_nodes and max_nodes > 0:
        sub = sub.head(max_nodes).copy()
    sub["local_idx"] = np.arange(len(sub), dtype=np.int32)
    return sub


def load_traffic_2024(data_dir: Path, region_node_idx: np.ndarray) -> tuple[np.ndarray, pd.DatetimeIndex]:
    traffic_dir = data_dir / "year_2024" / "year_2024"
    chunks = []
    for month in range(1, 13):
        path = traffic_dir / f"2024_p{month:02d}.npy"
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] != 16972:
            raise ValueError(f"Expected 2024 array as (node,time,channel), got {arr.shape}")
        sub = np.asarray(arr[region_node_idx, :, :], dtype=np.float32)  # (N,T,3)
        chunks.append(np.transpose(sub, (1, 0, 2)))  # (T,N,3)
    traffic = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    times = pd.date_range("2024-01-01", periods=traffic.shape[0], freq="5min")
    return traffic, times


def build_simple_baseline(
    traffic: np.ndarray,
    times: pd.DatetimeIndex,
    train_frac: float,
    min_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train-only median baseline by weekend/weekday and time-of-day."""
    T, N, C = traffic.shape
    cutoff = int(T * train_frac)
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    dow = times.dayofweek.to_numpy().astype(np.int8)
    baseline = np.full((2, 288, N, C), np.nan, dtype=np.float32)
    scale = np.full_like(baseline, np.nan)

    for dk in (0, 1):
        for slot in range(288):
            idx = np.flatnonzero((day_kind[:cutoff] == dk) & (tod[:cutoff] == slot))
            if idx.size == 0:
                continue
            vals = traffic[idx]
            finite = np.isfinite(vals)
            count = finite.sum(axis=0)
            vals2 = vals.copy()
            vals2[:, count < min_count] = np.nan
            with np.errstate(all="ignore"):
                med = np.nanmedian(vals2, axis=0).astype(np.float32)
                mad = np.nanmedian(np.abs(vals2 - med[None, :, :]), axis=0).astype(np.float32)
            scl = 1.4826 * mad
            scl[:, CHANNEL_FLOW] = np.maximum(scl[:, CHANNEL_FLOW], 3.0)
            scl[:, CHANNEL_OCC] = np.maximum(scl[:, CHANNEL_OCC], 0.005)
            scl[:, CHANNEL_SPEED] = np.maximum(scl[:, CHANNEL_SPEED], 2.0)
            baseline[dk, slot] = med
            scale[dk, slot] = scl

    # Fill empty slots with node-level train medians.
    for c in range(C):
        node_med = np.nanmedian(traffic[:cutoff, :, c], axis=0).astype(np.float32)
        node_med[~np.isfinite(node_med)] = 0.0
        node_scl = np.nanmedian(np.abs(traffic[:cutoff, :, c] - node_med[None, :]), axis=0)
        node_scl = (1.4826 * node_scl).astype(np.float32)
        floor = 3.0 if c == CHANNEL_FLOW else (0.005 if c == CHANNEL_OCC else 2.0)
        node_scl = np.maximum(np.where(np.isfinite(node_scl), node_scl, floor), floor)
        bad = ~np.isfinite(baseline[:, :, :, c])
        baseline[:, :, :, c][bad] = np.broadcast_to(node_med, (2, 288, N))[bad]
        bad_s = ~np.isfinite(scale[:, :, :, c])
        scale[:, :, :, c][bad_s] = np.broadcast_to(node_scl, (2, 288, N))[bad_s]

    return baseline, scale, day_kind, tod, dow


def load_incidents_2024(data_dir: Path) -> pd.DataFrame:
    inc = pd.read_csv(data_dir / "incidents_y2024.csv", sep="\t", low_memory=False)
    inc["dt_parsed"] = pd.to_datetime(inc["dt"], errors="coerce")
    inc["incident_id"] = inc["incident_id"].astype(str)
    inc["Fwy"] = pd.to_numeric(inc["Fwy"], errors="coerce")
    inc["Abs PM"] = pd.to_numeric(inc["Abs PM"], errors="coerce")
    inc["Freeway_direction"] = inc["Freeway_direction"].map(normalize_direction)
    inc = inc.dropna(subset=["dt_parsed", "Fwy", "Abs PM"])
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-01-01")
    inc = inc[(inc["dt_parsed"] >= start) & (inc["dt_parsed"] < end)].copy()
    return inc.reset_index(drop=True)


def match_incidents(
    inc: pd.DataFrame,
    region_meta: pd.DataFrame,
    total_steps: int,
    candidate_pm_radius: float,
    anchor_pm_radius: float,
) -> list[MatchedIncident]:
    groups: dict[tuple[float, str], np.ndarray] = {}
    for key, grp in region_meta.groupby(["Fwy", "Direction"], dropna=True):
        groups[(float(key[0]), str(key[1]))] = grp["local_idx"].to_numpy(dtype=np.int32)
    abs_pm = region_meta["Abs PM"].to_numpy(dtype=np.float64)
    start = pd.Timestamp("2024-01-01")
    out: list[MatchedIncident] = []

    for _, row in inc.iterrows():
        direction = normalize_direction(row["Freeway_direction"])
        key = (float(row["Fwy"]), direction)
        sensor_idx = groups.get(key)
        if sensor_idx is None or sensor_idx.size == 0:
            continue
        pm = float(row["Abs PM"])
        dist = np.abs(abs_pm[sensor_idx] - pm)
        within = dist <= candidate_pm_radius
        if not np.any(within):
            continue
        cand = sensor_idx[within]
        cand_dist = dist[within]
        anchor_pos = int(np.argmin(cand_dist))
        if float(cand_dist[anchor_pos]) > anchor_pm_radius:
            continue
        t = int((row["dt_parsed"] - start).total_seconds() // 300)
        if 0 <= t < total_steps:
            out.append(MatchedIncident(t, int(cand[anchor_pos]), cand.astype(np.int32)))
    return out


def build_edge_index(region_meta: pd.DataFrame, k: int) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for _, grp in region_meta.groupby(["Fwy", "Direction"], dropna=True):
        idx = grp["local_idx"].to_numpy(dtype=np.int64)
        pm = grp["Abs PM"].to_numpy(dtype=np.float64)
        if idx.size <= 1:
            continue
        order = np.argsort(pm)
        idx = idx[order]
        for pos, i in enumerate(idx):
            lo = max(0, pos - k)
            hi = min(idx.size, pos + k + 1)
            for j in idx[lo:hi]:
                if i != j:
                    edges.add((int(i), int(j)))
                    edges.add((int(j), int(i)))
    for i in range(len(region_meta)):
        edges.add((i, i))
    if not edges:
        raise RuntimeError("No graph edges built; check region metadata.")
    return np.asarray(sorted(edges), dtype=np.int64).T


def affected_mask_for_sample(
    traffic: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    t: int,
    candidate_idx: np.ndarray,
    horizon: int,
    threshold: float,
    N: int,
) -> np.ndarray:
    fut = np.arange(t + 1, t + 1 + horizon)
    actual = traffic[fut][:, candidate_idx, :]
    base = baseline[day_kind[fut], tod[fut]][:, candidate_idx, :]
    scl = scale[day_kind[fut], tod[fut]][:, candidate_idx, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_z = np.maximum(0, (base[:, :, CHANNEL_FLOW] - actual[:, :, CHANNEL_FLOW]) / scl[:, :, CHANNEL_FLOW])
        speed_z = np.maximum(0, (base[:, :, CHANNEL_SPEED] - actual[:, :, CHANNEL_SPEED]) / scl[:, :, CHANNEL_SPEED])
        occ_z = np.maximum(0, (actual[:, :, CHANNEL_OCC] - base[:, :, CHANNEL_OCC]) / scl[:, :, CHANNEL_OCC])
        any_z = np.nanmax(np.stack([flow_z, speed_z, occ_z], axis=0), axis=0)
        peak = np.nanmax(any_z, axis=0)
    peak[~np.isfinite(peak)] = 0.0
    mask = np.zeros(N, dtype=bool)
    mask[candidate_idx] = peak >= threshold
    return mask


def build_samples(
    matches: list[MatchedIncident],
    traffic: np.ndarray,
    baseline: np.ndarray,
    scale: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    history: int,
    horizon: int,
    offsets: list[int],
    train_frac: float,
    val_frac: float,
    affected_z: float,
) -> list[dict]:
    T, N, _ = traffic.shape
    samples: list[dict] = []
    for m in matches:
        for off in offsets:
            t = m.start_idx + int(off)
            if t < history - 1 or t + horizon >= T:
                continue
            frac = t / T
            split = "train" if frac < train_frac else ("val" if frac < train_frac + val_frac else "test")
            aff = affected_mask_for_sample(
                traffic, baseline, scale, day_kind, tod, t,
                m.candidate_idx, horizon, affected_z, N,
            )
            samples.append({"t": int(t), "split": split, "affected": aff})
    return samples


class XTraffic2024Dataset(Dataset):
    def __init__(self, bundle: RegionBundle, split: str, history: int, horizon: int,
                 max_samples: int = 0, seed: int = 42):
        self.bundle = bundle
        self.history = history
        self.horizon = horizon
        self.items = [s for s in bundle.samples if s["split"] == split]
        if max_samples and max_samples > 0 and len(self.items) > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.items), size=max_samples, replace=False)
            self.items = [self.items[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        b = self.bundle
        t = self.items[idx]["t"]
        hist = np.arange(t - self.history + 1, t + 1)
        fut = np.arange(t + 1, t + 1 + self.horizon)
        x_hist = b.traffic[hist].transpose(1, 0, 2).astype(np.float32)
        x_base = b.baseline[b.day_kind[hist], b.tod[hist]].transpose(1, 0, 2).astype(np.float32)
        y_true = b.traffic[fut].transpose(1, 0, 2).astype(np.float32)
        y_base = b.baseline[b.day_kind[fut], b.tod[fut]].transpose(1, 0, 2).astype(np.float32)
        y_mask = np.isfinite(y_true[..., CHANNEL_FLOW])
        return {
            "x_hist": np.nan_to_num(x_hist, nan=0.0),
            "x_baseline": np.nan_to_num(x_base, nan=0.0),
            "y_true": np.nan_to_num(y_true, nan=0.0),
            "y_baseline": np.nan_to_num(y_base, nan=0.0),
            "y_mask_flow": y_mask.astype(np.bool_),
            "affected_mask": self.items[idx]["affected"].astype(np.bool_),
            "sample_start": np.int64(t),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([b[k] for b in batch], axis=0)) for k in batch[0]}


def build_adj_supports(edge_index: np.ndarray, N: int, device: torch.device) -> list[torch.Tensor]:
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    A1 = deg_inv[:, None] * A
    deg_t = A.T.sum(axis=1)
    deg_t_inv = np.where(deg_t > 0, 1.0 / deg_t, 0.0)
    A2 = deg_t_inv[:, None] * A.T
    return [torch.from_numpy(A1).to(device), torch.from_numpy(A2).to(device)]


def make_gwn(num_nodes: int, supports: list[torch.Tensor], device: torch.device,
             in_dim: int, horizon: int, nhid: int, dropout: float) -> nn.Module:
    return gwnet(
        device=device, num_nodes=num_nodes, dropout=dropout, supports=supports,
        gcn_bool=True, addaptadj=True, in_dim=in_dim, out_dim=horizon,
        residual_channels=nhid, dilation_channels=nhid,
        skip_channels=nhid * 8, end_channels=nhid * 16, blocks=4, layers=2,
    )


class AdaptiveAlpha(nn.Module):
    def __init__(self, r0_init: float = 1.5, tau_init: float = 1.0):
        super().__init__()
        self.r0 = nn.Parameter(torch.tensor(float(r0_init)))
        self.raw_tau = nn.Parameter(torch.tensor(float(tau_init)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.raw_tau) + 1e-3
        return torch.exp(-F.relu(r - self.r0) / tau)


class V0COriginal(nn.Module):
    def __init__(self, num_nodes: int, supports: list[torch.Tensor], horizon: int,
                 device: torch.device, nhid: int = 32, dropout: float = 0.3):
        super().__init__()
        self.single = make_gwn(num_nodes, supports, device, 1, horizon, nhid, dropout)
        self.alpha_mod = AdaptiveAlpha()
        self.register_buffer("sd_res", torch.tensor(1.0))
        self.last_alpha: torch.Tensor | None = None

    @staticmethod
    def _to_gwnet(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _from_gwnet(out: torch.Tensor) -> torch.Tensor:
        return out.squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x_hist: torch.Tensor, x_baseline: torch.Tensor,
                y_baseline: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flow = x_hist[..., CHANNEL_FLOW]
        sig = (flow - x_baseline[..., CHANNEL_FLOW]) / self.sd_res
        residual = self._from_gwnet(self.single(self._to_gwnet(sig.unsqueeze(-1))))
        r = (flow - x_baseline[..., CHANNEL_FLOW]).abs().mean(dim=2) / self.sd_res
        alpha = self.alpha_mod(r)
        self.last_alpha = alpha.detach()
        persist = flow[:, :, -1:]
        anchor = alpha.unsqueeze(-1) * y_baseline[..., CHANNEL_FLOW] + (1.0 - alpha).unsqueeze(-1) * persist
        pred = anchor + residual * self.sd_res
        return pred, {"alpha": alpha, "r": r}


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return ((pred - target).abs() * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def compute_sd_res(bundle: RegionBundle, train_frac: float) -> float:
    T = bundle.traffic.shape[0]
    cutoff = int(T * train_frac)
    idx = np.arange(cutoff)
    base = bundle.baseline[bundle.day_kind[idx], bundle.tod[idx]][..., CHANNEL_FLOW]
    resid = bundle.traffic[:cutoff, :, CHANNEL_FLOW] - base
    valid = np.isfinite(resid)
    sd = float(np.nanstd(resid[valid]) + 1e-6) if valid.any() else 1.0
    return max(sd, 1.0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums = {"all": 0.0, "affected": 0.0, "unaffected": 0.0}
    counts = {"all": 0.0, "affected": 0.0, "unaffected": 0.0}
    alpha_aff, alpha_unaff = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x_hist"].to(device)
            xb = batch["x_baseline"].to(device)
            yb = batch["y_baseline"].to(device)
            target = batch["y_true"][..., CHANNEL_FLOW].to(device)
            mask = batch["y_mask_flow"].to(device)
            aff = batch["affected_mask"].to(device)
            pred, details = model(x, xb, yb)
            diff = (pred - target).abs()
            aff3 = aff.unsqueeze(-1).expand_as(mask)
            for name, m in [
                ("all", mask),
                ("affected", mask & aff3),
                ("unaffected", mask & ~aff3),
            ]:
                mf = m.float()
                sums[name] += float((diff * mf).sum().item())
                counts[name] += float(mf.sum().item())
            alpha = details.get("alpha")
            if alpha is not None:
                if aff.any():
                    alpha_aff.append(alpha[aff].detach().cpu())
                if (~aff).any():
                    alpha_unaff.append(alpha[~aff].detach().cpu())
    out = {k: sums[k] / max(counts[k], 1.0) for k in sums}
    if alpha_aff:
        out["alpha_affected"] = float(torch.cat(alpha_aff).mean().item())
    if alpha_unaff:
        out["alpha_unaffected"] = float(torch.cat(alpha_unaff).mean().item())
    model.train()
    return out


def prepare_bundle(args: argparse.Namespace) -> RegionBundle:
    data_dir = Path(args.data_dir).expanduser().resolve()
    meta_all = load_sensor_meta(data_dir)
    rmeta = region_mainline_meta(meta_all, args.region, args.max_nodes)
    node_idx = rmeta["node_idx"].to_numpy(dtype=np.int64)
    print(f"[data] region={args.region} nodes={len(rmeta)} loading 2024 traffic...", flush=True)
    traffic, times = load_traffic_2024(data_dir, node_idx)
    mask = np.isfinite(traffic)
    print(f"[data] traffic={traffic.shape} finite={mask.mean():.4f}", flush=True)
    print("[data] building train-only climatology baseline...", flush=True)
    baseline, scale, day_kind, tod, dow = build_simple_baseline(
        traffic, times, train_frac=args.train_frac, min_count=args.min_baseline_count,
    )
    print("[data] matching incidents...", flush=True)
    inc = load_incidents_2024(data_dir)
    matches = match_incidents(
        inc, rmeta, traffic.shape[0],
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
    )
    if args.max_events and args.max_events > 0:
        matches = matches[:args.max_events]
    print(f"[data] matched incidents={len(matches)}", flush=True)
    offsets = parse_offsets(args.sample_offsets)
    samples = build_samples(
        matches, traffic, baseline, scale, day_kind, tod,
        history=args.history, horizon=args.horizon, offsets=offsets,
        train_frac=args.train_frac, val_frac=args.val_frac,
        affected_z=args.affected_z,
    )
    print(
        "[data] samples "
        + " ".join(f"{s}={sum(x['split'] == s for x in samples)}" for s in ["train", "val", "test"]),
        flush=True,
    )
    edge_index = build_edge_index(rmeta, k=args.graph_k)
    return RegionBundle(args.region, traffic, mask, baseline, scale, day_kind, tod, dow, samples, edge_index, rmeta)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    p.add_argument("--out-dir", type=Path, default=HERE / "outputs" / "A_v0c_original_2024")
    p.add_argument("--region", default="Alameda", choices=list(REGION_COUNTY))
    p.add_argument("--history", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--sample-offsets", default="0,6,12")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--candidate-pm-radius", type=float, default=5.0)
    p.add_argument("--anchor-pm-radius", type=float, default=2.0)
    p.add_argument("--affected-z", type=float, default=2.0)
    p.add_argument("--min-baseline-count", type=int, default=8)
    p.add_argument("--graph-k", type=int, default=4)
    p.add_argument("--max-nodes", type=int, default=0, help="Debug only: cap region nodes.")
    p.add_argument("--max-events", type=int, default=0, help="Debug only: cap matched incidents.")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = prepare_bundle(args)
    train_ds = XTraffic2024Dataset(bundle, "train", args.history, args.horizon, args.max_train_samples, args.seed)
    val_ds = XTraffic2024Dataset(bundle, "val", args.history, args.horizon, args.max_val_samples, args.seed)
    test_ds = XTraffic2024Dataset(bundle, "test", args.history, args.horizon, args.max_test_samples, args.seed)
    if not train_ds or not val_ds or not test_ds:
        raise RuntimeError("Empty split. Try a different region, offsets, or max-events setting.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    supports = build_adj_supports(bundle.edge_index, bundle.traffic.shape[1], device)
    model = V0COriginal(bundle.traffic.shape[1], supports, args.horizon, device, args.nhid, args.dropout).to(device)
    sd_res = compute_sd_res(bundle, args.train_frac)
    model.sd_res.fill_(sd_res)
    print(f"[model] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} sd_res={sd_res:.3f}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs * len(train_loader), 1), eta_min=args.lr * 1e-2)
    best = float("inf")
    best_path = args.out_dir / f"{args.region}_best.pt"
    log_rows = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch["x_hist"].to(device)
            xb = batch["x_baseline"].to(device)
            yb = batch["y_baseline"].to(device)
            target = batch["y_true"][..., CHANNEL_FLOW].to(device)
            mask = batch["y_mask_flow"].to(device)
            pred, _ = model(x, xb, yb)
            loss = masked_mae(pred, target, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        val = evaluate(model, val_loader, device)
        row = {"epoch": ep, "train_mae": total / max(n, 1), **{f"val_{k}": v for k, v in val.items()}}
        log_rows.append(row)
        print(f"[ep {ep:02d}] train={row['train_mae']:.4f} val_all={val['all']:.4f} time={time.time()-t0:.1f}s", flush=True)
        if val["all"] < best:
            best = val["all"]
            torch.save({"model_state": model.state_dict(), "args": jsonable_args(args), "sd_res": sd_res}, best_path)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, device)
    summary = {
        "variant": "A_v0c_original_2024",
        "region": args.region,
        "nodes": int(bundle.traffic.shape[1]),
        "samples": {s: int(sum(x["split"] == s for x in bundle.samples)) for s in ["train", "val", "test"]},
        "best_val_all": best,
        "test": test,
        "args": jsonable_args(args),
    }
    (args.out_dir / f"{args.region}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(args.out_dir / f"{args.region}_training_log.csv", index=False)
    print("[test]", json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
