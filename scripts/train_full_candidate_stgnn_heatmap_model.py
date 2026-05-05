#!/usr/bin/env python3
"""Train full-candidate STGNN residual model with node-time impact heatmap.

This version addresses two limitations of the previous candidate STGNN:

1. It can keep the complete incident candidate set by using a disk-backed HDF5
   cache instead of concatenating all tensors in memory.
2. It adds a node-time impact heatmap auxiliary target, which is closer to the
   residual forecasting output than event-level severity/recovery/spread labels.

Candidate nodes are still selected only from road-aligned incident neighborhoods
(same freeway/direction and postmile radius). Impact labels are not used to
select input nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_impact_labels import (
    CHANNEL_FLOW,
    CHANNEL_OCC,
    CHANNEL_SPEED,
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
    make_event_aux,
    robust_mae_from_arrays,
    select_candidate_nodes,
)
from train_candidate_stgnn_residual_model import DirectionalGraphLayer
from train_impact_residual_model import (
    choose_device,
    json_safe_args,
    make_event_context,
    make_time_features,
    split_name,
)
from validate_forecast_error_against_impact import fit_blend_alphas, parse_incident_ids


CHANNELS = ("flow", "occupancy", "speed")
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}


class UndirectedGraphLayer(nn.Module):
    """Graph propagation ablation that ignores upstream/downstream direction."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.all_proj = nn.Linear(hidden_dim, hidden_dim)
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
        del adj_left, adj_right
        all_msg = torch.bmm(adj_all, h)
        out = self.self_proj(h) + self.all_proj(all_msg)
        out = self.norm(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        return out * valid.unsqueeze(-1)


@dataclass
class CacheStats:
    hist_mean: np.ndarray
    hist_std: np.ndarray
    hist_normal_mean: np.ndarray
    hist_normal_std: np.ndarray
    node_mean: np.ndarray
    node_std: np.ndarray
    global_mean: np.ndarray
    global_std: np.ndarray
    event_aux_mean: np.ndarray
    event_aux_std: np.ndarray


class H5SampleWriter:
    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.h5 = h5py.File(self.path, "w")
        self.count = 0
        self.datasets: dict[str, h5py.Dataset] = {}
        self._init_datasets()

    def _create(self, name: str, shape_tail: tuple[int, ...], dtype: str) -> None:
        chunks = (min(512, 1024), *shape_tail)
        self.datasets[name] = self.h5.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            chunks=chunks,
            dtype=dtype,
        )

    def _init_datasets(self) -> None:
        a = self.args
        n = a.max_candidate_nodes
        h = a.horizon_steps
        inp = a.input_steps
        c = len(CHANNELS)
        self._create("hist_residual", (inp, n, c), "float32")
        self._create("hist_residual_normal", (inp, n, c), "float32")
        self._create("node_idx", (n,), "int32")
        self._create("node_signed_pm_raw", (n,), "float32")
        self._create("node_abs_pm_raw", (n,), "float32")
        self._create("node_context", (n, 8), "float32")
        self._create("global_context", (17,), "float32")
        self._create("normal_delta", (h, n, c), "float32")
        self._create("y_residual", (h, n, c), "float32")
        self._create("y_mask", (h, n, c), "bool")
        self._create("impact_heatmap", (h, n), "float32")
        self._create("impact_mask", (h, n), "bool")
        self._create("event_aux", (3,), "float32")
        self._create("node_affected", (n,), "float32")
        self._create("node_valid", (n,), "float32")
        self._create("split", (), "int8")
        self._create("region_code", (), "int8")

    def append_many(self, rows: dict[str, list[np.ndarray] | list[int]]) -> None:
        if not rows["split"]:
            return
        batch = len(rows["split"])
        start = self.count
        end = start + batch
        for ds in self.datasets.values():
            ds.resize((end, *ds.shape[1:]))
        for name, values in rows.items():
            arr = np.asarray(values)
            self.datasets[name][start:end] = arr
        self.count = end

    def close(self) -> None:
        self.h5.attrs["samples"] = self.count
        self.h5.attrs["max_candidate_nodes"] = self.args.max_candidate_nodes
        self.h5.attrs["input_steps"] = self.args.input_steps
        self.h5.attrs["horizon_steps"] = self.args.horizon_steps
        self.h5.attrs["channels"] = len(CHANNELS)
        self.h5.attrs["candidate_pm_radius"] = self.args.candidate_pm_radius
        self.h5.attrs["normal_model_dir"] = "" if self.args.normal_model_dir is None else str(self.args.normal_model_dir)
        self.h5.attrs["normal_infer_batch_size"] = self.args.normal_infer_batch_size
        self.h5.attrs["normal_inference_scope"] = self.args.normal_inference_scope
        self.h5.attrs["max_cache_samples_per_split"] = self.args.max_cache_samples_per_split
        self.h5.attrs["has_normal_delta"] = True
        self.h5.attrs["has_hist_residual_normal"] = True
        self.h5.close()


class H5IncidentDataset(Dataset):
    def __init__(self, cache_path: Path, indices: np.ndarray, stats: CacheStats) -> None:
        self.cache_path = cache_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.stats = stats
        self.h5: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _file(self) -> h5py.File:
        if self.h5 is None:
            self.h5 = h5py.File(self.cache_path, "r")
        return self.h5

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        h5 = self._file()
        idx = int(self.indices[item])

        hist = h5["hist_residual"][idx].astype(np.float32)
        hist_mean = self.stats.hist_mean.reshape(1, 1, -1)
        hist_std = self.stats.hist_std.reshape(1, 1, -1)
        hist = (hist - hist_mean) / hist_std
        hist[~np.isfinite(hist)] = 0.0
        if "hist_residual_normal" in h5:
            hist_normal = h5["hist_residual_normal"][idx].astype(np.float32)
        else:
            hist_normal = np.zeros_like(h5["hist_residual"][idx].astype(np.float32))
        hist_normal_mean = self.stats.hist_normal_mean.reshape(1, 1, -1)
        hist_normal_std = self.stats.hist_normal_std.reshape(1, 1, -1)
        hist_normal = (hist_normal - hist_normal_mean) / hist_normal_std
        hist_normal[~np.isfinite(hist_normal)] = 0.0

        raw_node = h5["node_context"][idx].astype(np.float32)
        node = (raw_node - self.stats.node_mean) / self.stats.node_std
        node[:, 0] = raw_node[:, 0]
        node[:, 6] = raw_node[:, 6]
        node[~np.isfinite(node)] = 0.0

        global_context = h5["global_context"][idx].astype(np.float32)
        global_context = (global_context - self.stats.global_mean) / self.stats.global_std
        global_context[~np.isfinite(global_context)] = 0.0

        event_aux = h5["event_aux"][idx].astype(np.float32)
        event_aux = (event_aux - self.stats.event_aux_mean) / self.stats.event_aux_std
        event_aux[~np.isfinite(event_aux)] = 0.0
        if "normal_delta" in h5:
            normal_delta = h5["normal_delta"][idx].astype(np.float32)
        else:
            horizon = int(h5.attrs["horizon_steps"])
            nodes = int(h5.attrs["max_candidate_nodes"])
            channels = int(h5.attrs["channels"])
            normal_delta = np.zeros((horizon, nodes, channels), dtype=np.float32)
        normal_delta[~np.isfinite(normal_delta)] = 0.0

        return (
            torch.from_numpy(hist),
            torch.from_numpy(hist_normal.astype(np.float32)),
            torch.from_numpy(node.astype(np.float32)),
            torch.from_numpy(global_context.astype(np.float32)),
            torch.from_numpy(normal_delta.astype(np.float32)),
            torch.from_numpy(h5["y_residual"][idx].astype(np.float32)),
            torch.from_numpy(h5["y_mask"][idx].astype(np.float32)),
            torch.from_numpy(h5["impact_heatmap"][idx].astype(np.float32)),
            torch.from_numpy(h5["impact_mask"][idx].astype(np.float32)),
            torch.from_numpy(event_aux.astype(np.float32)),
            torch.from_numpy(h5["node_affected"][idx].astype(np.float32)),
            torch.from_numpy(h5["node_valid"][idx].astype(np.float32)),
        )


class FullCandidateSTGNNHeatmap(nn.Module):
    def __init__(
        self,
        channels: int,
        hist_input_channels: int,
        node_context_dim: int,
        global_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        graph_sigma: float,
        graph_mode: str,
        use_normal_delta: bool,
        use_normal_delta_abs: bool,
        use_temporal_decay_head: bool,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.channels = channels
        self.hist_input_channels = hist_input_channels
        self.graph_sigma = graph_sigma
        self.graph_mode = graph_mode
        self.use_normal_delta = use_normal_delta
        self.use_normal_delta_abs = use_normal_delta_abs
        self.use_temporal_decay_head = use_temporal_decay_head
        graph_layer_cls = DirectionalGraphLayer if graph_mode == "directional" else UndirectedGraphLayer
        self.temporal_encoder = nn.GRU(hist_input_channels, hidden_dim, batch_first=True)
        self.context_proj = nn.Sequential(
            nn.Linear(node_context_dim + global_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [graph_layer_cls(hidden_dim=hidden_dim, dropout=dropout) for _ in range(graph_layers)]
        )
        residual_input_dim = hidden_dim
        if use_normal_delta:
            residual_input_dim += horizon_steps * channels
            self.normal_delta_norm = nn.LayerNorm(horizon_steps * channels)
        if use_normal_delta_abs:
            residual_input_dim += horizon_steps * channels
            self.normal_delta_abs_norm = nn.LayerNorm(horizon_steps * channels)
        self.residual_decoder = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * channels),
        )
        if use_temporal_decay_head:
            self.temporal_decay_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, horizon_steps),
            )
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps),
        )
        self.event_aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.node_aux_head = nn.Linear(hidden_dim, 1)
        zero_init_modules = [self.residual_decoder[-1], self.impact_head[-1]]
        if use_temporal_decay_head:
            zero_init_modules.append(self.temporal_decay_head[-1])
        for module in zero_init_modules:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

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
        adj_left = adj * (source_pos < target_pos).to(adj.dtype)
        adj_right = adj * (source_pos > target_pos).to(adj.dtype)

        def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
            return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return row_normalize(adj), row_normalize(adj_left), row_normalize(adj_right), valid

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, input_steps, nodes, hist_channels = hist_residual.shape
        temporal_in = hist_residual.permute(0, 2, 1, 3).reshape(batch_size * nodes, input_steps, hist_channels)
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

        residual_input = h
        if self.use_normal_delta or self.use_normal_delta_abs:
            if normal_delta is None:
                normal_delta = torch.zeros(
                    batch_size,
                    self.horizon_steps,
                    nodes,
                    self.channels,
                    dtype=h.dtype,
                    device=h.device,
                )
        if self.use_normal_delta:
            delta_flat = normal_delta.permute(0, 2, 1, 3).reshape(batch_size, nodes, self.horizon_steps * self.channels)
            delta_flat = self.normal_delta_norm(delta_flat)
            residual_input = torch.cat([h, delta_flat], dim=-1)
        if self.use_normal_delta_abs:
            delta_abs_flat = normal_delta.abs().permute(0, 2, 1, 3).reshape(
                batch_size,
                nodes,
                self.horizon_steps * self.channels,
            )
            delta_abs_flat = self.normal_delta_abs_norm(delta_abs_flat)
            residual_input = torch.cat([residual_input, delta_abs_flat], dim=-1)
        residual = self.residual_decoder(residual_input).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        residual = residual.permute(0, 2, 1, 3).contiguous()
        if self.use_temporal_decay_head:
            decay_gate = 2.0 * torch.sigmoid(self.temporal_decay_head(h))
            residual = residual * decay_gate.permute(0, 2, 1).unsqueeze(-1)
        impact = self.impact_head(h).permute(0, 2, 1).contiguous()
        pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_aux = self.event_aux_head(pooled)
        node_logits = self.node_aux_head(h).squeeze(-1)
        return residual, impact, event_aux, node_logits


class LocalNormalGraphLayer(nn.Module):
    """Normal-branch graph layer that accepts a per-sample local adjacency."""

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
        valid: torch.Tensor,
    ) -> torch.Tensor:
        all_msg = torch.bmm(adj_all, h)
        out = self.self_proj(h) + self.all_proj(all_msg)
        if self.use_directional:
            left_msg = torch.bmm(adj_left, h)
            right_msg = torch.bmm(adj_right, h)
            out = out + self.left_proj(left_msg) + self.right_proj(right_msg)
        out = self.norm(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        return out * valid.unsqueeze(-1)


class LocalNormalSTGNN(nn.Module):
    """Checkpoint-compatible local version of the learned normal STGNN.

    The trained normal model has fixed full-region graph buffers. For incident
    samples we only need the candidate neighborhood, so this module reuses the
    learned weights and accepts sliced node context / adjacency at runtime.
    """

    def __init__(
        self,
        channels: int,
        node_context_dim: int,
        horizon_steps: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
        use_directional: bool,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.channels = channels
        self.temporal_encoder = nn.GRU(channels, hidden_dim, batch_first=True)
        self.context_proj = nn.Sequential(
            nn.Linear(node_context_dim + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.graph_layers = nn.ModuleList(
            [LocalNormalGraphLayer(hidden_dim, dropout, use_directional) for _ in range(graph_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_steps * channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_context: torch.Tensor,
        node_context: torch.Tensor,
        adj_all: torch.Tensor,
        adj_left: torch.Tensor,
        adj_right: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, input_steps, nodes, channels = x.shape
        temporal_in = x.permute(0, 2, 1, 3).reshape(batch_size * nodes, input_steps, channels)
        _, h_last = self.temporal_encoder(temporal_in)
        h = h_last[-1].reshape(batch_size, nodes, -1)

        time_ctx = time_context[:, None, :].expand(-1, nodes, -1)
        h = self.input_norm(h + self.context_proj(torch.cat([node_context, time_ctx], dim=-1)))
        h = h * node_valid.unsqueeze(-1)
        for layer in self.graph_layers:
            h = h + layer(h, adj_all, adj_left, adj_right, node_valid)
            h = h * node_valid.unsqueeze(-1)
        out = self.decoder(h).reshape(batch_size, nodes, self.horizon_steps, self.channels)
        return out.permute(0, 2, 1, 3).contiguous()


def _torch_load(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normal_time_features(times: pd.DatetimeIndex, start: int) -> np.ndarray:
    ts = times[start]
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


def normalize_local_adj(adj: np.ndarray, node_valid: np.ndarray) -> np.ndarray:
    valid = node_valid.astype(bool)
    out = np.asarray(adj, dtype=np.float32).copy()
    out[~np.isfinite(out)] = 0.0
    out *= valid[:, None] * valid[None, :]
    if valid.any():
        idx = np.flatnonzero(valid)
        out[idx, idx] = np.maximum(out[idx, idx], 1.0)
    denom = out.sum(axis=1, keepdims=True)
    denom[denom < 1e-6] = 1.0
    return (out / denom).astype(np.float32)


def build_blend_prediction_batch(
    traffic: np.ndarray,
    baseline: np.ndarray,
    day_kind: np.ndarray,
    tod: np.ndarray,
    alphas: np.ndarray,
    starts: np.ndarray,
    horizon_steps: int,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int32)
    batch = starts.size
    nodes = traffic.shape[1]
    channels = traffic.shape[2]
    out = np.empty((batch, horizon_steps, nodes, channels), dtype=np.float32)
    for i, start in enumerate(starts):
        future_idx = np.arange(int(start), int(start) + horizon_steps, dtype=np.int32)
        future_base = baseline[day_kind[future_idx], tod[future_idx]]
        last_obs = traffic[int(start) - 1]
        for h in range(horizon_steps):
            out[i, h] = future_base[h] + alphas[h][None, :] * (last_obs - future_base[h])
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


@dataclass
class LearnedNormalRegion:
    model: LocalNormalSTGNN
    channel_mean: np.ndarray
    channel_std: np.ndarray
    node_context: np.ndarray
    adj_all: np.ndarray
    adj_left: np.ndarray
    adj_right: np.ndarray
    device: torch.device

    @classmethod
    def load(cls, normal_model_dir: Path, region_name: str, device: torch.device) -> "LearnedNormalRegion":
        model_path = normal_model_dir.resolve() / region_name / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"normal model checkpoint not found: {model_path}")
        ckpt = _torch_load(model_path)
        normal_args = ckpt.get("args", {})
        state = ckpt["model_state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"unexpected normal model state in {model_path}")

        node_context = np.asarray(ckpt["node_context"], dtype=np.float32)
        state_np = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in state.items()
            if key in {"adj_all", "adj_left", "adj_right"}
        }
        model = LocalNormalSTGNN(
            channels=len(CHANNELS),
            node_context_dim=node_context.shape[1],
            horizon_steps=int(normal_args.get("horizon_steps", 12)),
            hidden_dim=int(normal_args.get("hidden_dim", 64)),
            graph_layers=int(normal_args.get("graph_layers", 2)),
            dropout=float(normal_args.get("dropout", 0.10)),
            use_directional=bool(normal_args.get("graph_directional", False)),
        )
        filtered_state = {
            key: value
            for key, value in state.items()
            if key not in {"node_context", "adj_all", "adj_left", "adj_right"}
        }
        model.load_state_dict(filtered_state, strict=True)
        model.to(device)
        model.eval()
        return cls(
            model=model,
            channel_mean=np.asarray(ckpt["channel_mean"], dtype=np.float32),
            channel_std=np.asarray(ckpt["channel_std"], dtype=np.float32),
            node_context=node_context,
            adj_all=state_np["adj_all"],
            adj_left=state_np["adj_left"],
            adj_right=state_np["adj_right"],
            device=device,
        )

    def predict(
        self,
        hist: np.ndarray,
        blend: np.ndarray,
        node_idx: np.ndarray,
        node_valid: np.ndarray,
        times: pd.DatetimeIndex,
        sample_start: int,
    ) -> np.ndarray:
        valid = node_valid.astype(bool)
        mean = self.channel_mean.reshape(1, 1, -1)
        std = self.channel_std.reshape(1, 1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (hist - mean) / std
        x[:, ~valid, :] = 0.0
        x[~np.isfinite(x)] = 0.0

        local_node_context = self.node_context[node_idx].astype(np.float32).copy()
        local_node_context[~valid] = 0.0
        local_adj_all = normalize_local_adj(self.adj_all[np.ix_(node_idx, node_idx)], node_valid)
        local_adj_left = normalize_local_adj(self.adj_left[np.ix_(node_idx, node_idx)], node_valid)
        local_adj_right = normalize_local_adj(self.adj_right[np.ix_(node_idx, node_idx)], node_valid)
        time_context = normal_time_features(times, sample_start)

        tensors = [
            torch.from_numpy(x[None].astype(np.float32)),
            torch.from_numpy(time_context[None].astype(np.float32)),
            torch.from_numpy(local_node_context[None].astype(np.float32)),
            torch.from_numpy(local_adj_all[None].astype(np.float32)),
            torch.from_numpy(local_adj_left[None].astype(np.float32)),
            torch.from_numpy(local_adj_right[None].astype(np.float32)),
            torch.from_numpy(node_valid[None].astype(np.float32)),
        ]
        tensors = [item.to(self.device) for item in tensors]
        with torch.no_grad():
            correction = self.model(*tensors).detach().cpu().numpy()[0]
        pred = blend + correction * self.channel_std.reshape(1, 1, -1)
        pred[:, ~valid, :] = blend[:, ~valid, :]
        pred[~np.isfinite(pred)] = blend[~np.isfinite(pred)]
        return pred.astype(np.float32)

    def predict_many(
        self,
        hist: np.ndarray,
        blend: np.ndarray,
        node_idx: np.ndarray,
        node_valid: np.ndarray,
        times: pd.DatetimeIndex,
        sample_start: np.ndarray,
    ) -> np.ndarray:
        valid = node_valid.astype(bool)
        mean = self.channel_mean.reshape(1, 1, 1, -1)
        std = self.channel_std.reshape(1, 1, 1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (hist - mean) / std
        x = np.where(valid[:, None, :, None], x, 0.0)
        x[~np.isfinite(x)] = 0.0

        batch_size, _, nodes, _ = x.shape
        local_node_context = np.zeros((batch_size, nodes, self.node_context.shape[1]), dtype=np.float32)
        local_adj_all = np.zeros((batch_size, nodes, nodes), dtype=np.float32)
        local_adj_left = np.zeros_like(local_adj_all)
        local_adj_right = np.zeros_like(local_adj_all)
        for i in range(batch_size):
            local_node_context[i] = self.node_context[node_idx[i]].astype(np.float32)
            local_node_context[i, ~valid[i]] = 0.0
            ix = np.ix_(node_idx[i], node_idx[i])
            local_adj_all[i] = normalize_local_adj(self.adj_all[ix], node_valid[i])
            local_adj_left[i] = normalize_local_adj(self.adj_left[ix], node_valid[i])
            local_adj_right[i] = normalize_local_adj(self.adj_right[ix], node_valid[i])
        time_context = np.stack([normal_time_features(times, int(start)) for start in sample_start]).astype(np.float32)

        tensors = [
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(time_context),
            torch.from_numpy(local_node_context),
            torch.from_numpy(local_adj_all),
            torch.from_numpy(local_adj_left),
            torch.from_numpy(local_adj_right),
            torch.from_numpy(node_valid.astype(np.float32)),
        ]
        tensors = [item.to(self.device) for item in tensors]
        with torch.no_grad():
            correction = self.model(*tensors).detach().cpu().numpy()
        pred = blend + correction * self.channel_std.reshape(1, 1, 1, -1)
        pred = np.where(valid[:, None, :, None], pred, blend)
        pred[~np.isfinite(pred)] = blend[~np.isfinite(pred)]
        return pred.astype(np.float32)

    def predict_many_full(
        self,
        full_hist: np.ndarray,
        full_blend: np.ndarray,
        node_idx: np.ndarray,
        node_valid: np.ndarray,
        times: pd.DatetimeIndex,
        sample_start: np.ndarray,
    ) -> np.ndarray:
        mean = self.channel_mean.reshape(1, 1, 1, -1)
        std = self.channel_std.reshape(1, 1, 1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (full_hist - mean) / std
        x[~np.isfinite(x)] = 0.0

        batch_size, _, nodes, _ = x.shape
        full_valid = np.ones((batch_size, nodes), dtype=np.float32)
        full_node_context = np.broadcast_to(self.node_context[None], (batch_size, *self.node_context.shape)).copy()
        full_adj_all = np.broadcast_to(self.adj_all[None], (batch_size, *self.adj_all.shape)).copy()
        full_adj_left = np.broadcast_to(self.adj_left[None], (batch_size, *self.adj_left.shape)).copy()
        full_adj_right = np.broadcast_to(self.adj_right[None], (batch_size, *self.adj_right.shape)).copy()
        time_context = np.stack([normal_time_features(times, int(start)) for start in sample_start]).astype(np.float32)

        tensors = [
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(time_context),
            torch.from_numpy(full_node_context.astype(np.float32)),
            torch.from_numpy(full_adj_all.astype(np.float32)),
            torch.from_numpy(full_adj_left.astype(np.float32)),
            torch.from_numpy(full_adj_right.astype(np.float32)),
            torch.from_numpy(full_valid),
        ]
        tensors = [item.to(self.device) for item in tensors]
        with torch.no_grad():
            correction = self.model(*tensors).detach().cpu().numpy()
        full_pred = full_blend + correction * self.channel_std.reshape(1, 1, 1, -1)
        full_pred[~np.isfinite(full_pred)] = full_blend[~np.isfinite(full_pred)]

        candidate_pred = np.empty(
            (batch_size, full_pred.shape[1], node_idx.shape[1], full_pred.shape[3]),
            dtype=np.float32,
        )
        candidate_blend = np.empty_like(candidate_pred)
        for i in range(batch_size):
            candidate_pred[i] = full_pred[i][:, node_idx[i], :]
            candidate_blend[i] = full_blend[i][:, node_idx[i], :]
        valid = node_valid.astype(bool)
        candidate_pred = np.where(valid[:, None, :, None], candidate_pred, candidate_blend)
        candidate_pred[~np.isfinite(candidate_pred)] = candidate_blend[~np.isfinite(candidate_pred)]
        return candidate_pred.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument("--event-root", type=Path, default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"))
    parser.add_argument("--raw-label-dir", type=Path, default=Path("outputs/impact_labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_candidate_stgnn_heatmap_model/first_pass"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--normal-model-dir",
        type=Path,
        default=None,
        help="Optional learned normal STGNN output directory used when rebuilding the residual cache.",
    )
    parser.add_argument(
        "--normal-infer-batch-size",
        type=int,
        default=256,
        help="Batch size for learned normal inference while rebuilding the residual cache.",
    )
    parser.add_argument(
        "--normal-inference-scope",
        choices=["local", "full"],
        default="local",
        help="Use candidate-subgraph normal inference or full-region normal inference before slicing candidates.",
    )
    parser.add_argument("--regions", nargs="+", default=["Alameda", "ContraCosta", "Orange"])
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--max-candidate-nodes", type=int, default=36)
    parser.add_argument("--sample-offsets", nargs="+", type=int, default=[0, 6, 12])
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--max-impact-z", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-mode", choices=["directional", "undirected"], default="directional")
    parser.add_argument("--graph-sigma", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-aux-weight", type=float, default=0.10)
    parser.add_argument("--event-aux-weight", type=float, default=0.05)
    parser.add_argument("--node-aux-weight", type=float, default=0.03)
    parser.add_argument(
        "--use-normal-delta",
        action="store_true",
        help="Use normalized learned-normal correction as a future known covariate for residual prediction.",
    )
    parser.add_argument(
        "--use-normal-delta-abs",
        action="store_true",
        help="Append abs(normal_delta) as a normal-branch disagreement / uncertainty proxy.",
    )
    parser.add_argument(
        "--use-dual-hist-residual",
        action="store_true",
        help="Concatenate statistical and learned-normal historical residuals as the temporal input.",
    )
    parser.add_argument(
        "--use-temporal-decay-head",
        action="store_true",
        help="Scale residual predictions with a learned node-level horizon gate initialized to identity.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--max-cache-samples-per-region",
        type=int,
        default=0,
        help="Optional cache-building cap for fast learned-normal smoke tests; 0 means no cap.",
    )
    parser.add_argument(
        "--max-cache-samples-per-split",
        type=int,
        default=0,
        help="Optional per-split cache-building cap for smoke tests; 0 means no cap.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def init_buffer() -> dict[str, list[np.ndarray] | list[int]]:
    return {
        "hist_residual": [],
        "hist_residual_normal": [],
        "node_idx": [],
        "node_signed_pm_raw": [],
        "node_abs_pm_raw": [],
        "node_context": [],
        "global_context": [],
        "normal_delta": [],
        "y_residual": [],
        "y_mask": [],
        "impact_heatmap": [],
        "impact_mask": [],
        "event_aux": [],
        "node_affected": [],
        "node_valid": [],
        "split": [],
        "region_code": [],
    }


def flush_if_needed(writer: H5SampleWriter, buffer: dict[str, list[np.ndarray] | list[int]], force: bool = False) -> None:
    if force or len(buffer["split"]) >= 512:
        writer.append_many(buffer)
        for key in buffer:
            buffer[key].clear()


def robust_impact_heatmap(actual: np.ndarray, baseline: np.ndarray, scale: np.ndarray, max_impact_z: float) -> tuple[np.ndarray, np.ndarray]:
    flow_z = np.maximum(0.0, (baseline[:, :, CHANNEL_FLOW] - actual[:, :, CHANNEL_FLOW]) / scale[:, :, CHANNEL_FLOW])
    speed_z = np.maximum(0.0, (baseline[:, :, CHANNEL_SPEED] - actual[:, :, CHANNEL_SPEED]) / scale[:, :, CHANNEL_SPEED])
    occ_z = np.maximum(0.0, (actual[:, :, CHANNEL_OCC] - baseline[:, :, CHANNEL_OCC]) / scale[:, :, CHANNEL_OCC])
    stack = np.stack([flow_z, speed_z, occ_z], axis=0)
    finite = np.isfinite(stack)
    mask = np.any(finite, axis=0)
    filled = np.where(finite, stack, -np.inf)
    any_z = np.max(filled, axis=0)
    any_z[~mask] = 0.0
    any_z = np.clip(any_z, 0.0, max_impact_z)
    any_z[~np.isfinite(any_z)] = 0.0
    return np.log1p(any_z).astype(np.float32), mask


def append_residual_sample(
    buffer: dict[str, list[np.ndarray] | list[int]],
    args: argparse.Namespace,
    hist: np.ndarray,
    hist_base: np.ndarray,
    hist_scale: np.ndarray,
    hist_residual_normal: np.ndarray,
    fut_scale: np.ndarray,
    actual_future: np.ndarray,
    normal_pred: np.ndarray,
    normal_delta: np.ndarray,
    heatmap_baseline: np.ndarray,
    node_idx: np.ndarray,
    node_valid: np.ndarray,
    node_affected: np.ndarray,
    node_ctx: np.ndarray,
    global_ctx: np.ndarray,
    event_aux: np.ndarray,
    sample_split: str,
    region_code: int,
) -> bool:
    with np.errstate(divide="ignore", invalid="ignore"):
        hist_res_z = (hist - hist_base) / hist_scale
        y_res_z = (actual_future - normal_pred) / fut_scale
    valid_node_mask = node_valid.astype(bool)
    hist_res_z[:, ~valid_node_mask, :] = 0.0
    hist_residual_normal = np.asarray(hist_residual_normal, dtype=np.float32).copy()
    hist_residual_normal[:, ~valid_node_mask, :] = 0.0
    hist_residual_normal[~np.isfinite(hist_residual_normal)] = 0.0
    y_mask = np.isfinite(y_res_z)
    y_mask[:, ~valid_node_mask, :] = False
    hist_res_z[~np.isfinite(hist_res_z)] = 0.0
    y_res_z[~np.isfinite(y_res_z)] = 0.0
    if not y_mask.any():
        return False

    heatmap, heatmap_mask = robust_impact_heatmap(
        actual=actual_future,
        baseline=heatmap_baseline,
        scale=fut_scale,
        max_impact_z=args.max_impact_z,
    )
    heatmap[:, ~valid_node_mask] = 0.0
    heatmap_mask[:, ~valid_node_mask] = False

    buffer["hist_residual"].append(hist_res_z.astype(np.float32))
    buffer["hist_residual_normal"].append(hist_residual_normal.astype(np.float32))
    buffer["node_idx"].append(node_idx.astype(np.int32))
    buffer["node_signed_pm_raw"].append((node_ctx[:, 0] * args.candidate_pm_radius).astype(np.float32))
    buffer["node_abs_pm_raw"].append((node_ctx[:, 1] * args.candidate_pm_radius).astype(np.float32))
    buffer["node_context"].append(node_ctx.astype(np.float32))
    buffer["global_context"].append(global_ctx.astype(np.float32))
    buffer["normal_delta"].append(normal_delta.astype(np.float32))
    buffer["y_residual"].append(y_res_z.astype(np.float32))
    buffer["y_mask"].append(y_mask.astype(bool))
    buffer["impact_heatmap"].append(heatmap.astype(np.float32))
    buffer["impact_mask"].append(heatmap_mask.astype(bool))
    buffer["event_aux"].append(event_aux.astype(np.float32))
    buffer["node_affected"].append(node_affected.astype(np.float32))
    buffer["node_valid"].append(node_valid.astype(np.float32))
    buffer["split"].append(SPLIT_TO_CODE[sample_split])
    buffer["region_code"].append(region_code)
    return True


def flush_learned_normal_pending(
    pending: list[dict[str, np.ndarray | int | str]],
    learned_normal: LearnedNormalRegion,
    buffer: dict[str, list[np.ndarray] | list[int]],
    writer: H5SampleWriter,
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    traffic: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    day_kind: np.ndarray | None = None,
    tod: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
) -> dict[str, int]:
    counts = {name: 0 for name in SPLIT_TO_CODE}
    if not pending:
        return counts
    if args.normal_inference_scope == "full" and any(
        item is None for item in (traffic, baseline, day_kind, tod, alphas)
    ):
        raise ValueError("full normal inference requires traffic, baseline, day_kind, tod, and alphas")

    hist = np.stack([item["hist"] for item in pending]).astype(np.float32)
    blend = np.stack([item["normal_pred"] for item in pending]).astype(np.float32)
    node_idx = np.stack([item["node_idx"] for item in pending]).astype(np.int32)
    node_valid = np.stack([item["node_valid"] for item in pending]).astype(np.float32)
    sample_start = np.asarray([item["sample_start"] for item in pending], dtype=np.int32)
    if args.normal_inference_scope == "full":
        full_hist = np.stack(
            [traffic[int(start) - args.input_steps : int(start)] for start in sample_start]  # type: ignore[index]
        ).astype(np.float32)
        full_blend = build_blend_prediction_batch(
            traffic=traffic,  # type: ignore[arg-type]
            baseline=baseline,  # type: ignore[arg-type]
            day_kind=day_kind,  # type: ignore[arg-type]
            tod=tod,  # type: ignore[arg-type]
            alphas=alphas,  # type: ignore[arg-type]
            starts=sample_start,
            horizon_steps=args.horizon_steps,
        )
        normal_pred_batch = learned_normal.predict_many_full(
            full_hist=full_hist,
            full_blend=full_blend,
            node_idx=node_idx,
            node_valid=node_valid,
            times=times,
            sample_start=sample_start,
        )
    else:
        normal_pred_batch = learned_normal.predict_many(
            hist=hist,
            blend=blend,
            node_idx=node_idx,
            node_valid=node_valid,
            times=times,
            sample_start=sample_start,
        )
    hist_residual_normal_batch = np.zeros_like(hist, dtype=np.float32)
    if args.use_dual_hist_residual:
        with np.errstate(divide="ignore", invalid="ignore"):
            hist_residual_normal_batch = (
                hist - np.stack([item["hist_base"] for item in pending]).astype(np.float32)
            ) / np.stack([item["hist_scale"] for item in pending]).astype(np.float32)
        valid_hist_items = [i for i, item in enumerate(pending) if "hist_context" in item]
        if valid_hist_items:
            hist_node_idx = np.stack([pending[i]["node_idx"] for i in valid_hist_items]).astype(np.int32)
            hist_node_valid = np.stack([pending[i]["node_valid"] for i in valid_hist_items]).astype(np.float32)
            hist_sample_start = np.asarray([pending[i]["hist_sample_start"] for i in valid_hist_items], dtype=np.int32)
            if args.normal_inference_scope == "full":
                hist_context = np.stack(
                    [
                        traffic[int(start) - args.input_steps : int(start)]  # type: ignore[index]
                        for start in hist_sample_start
                    ]
                ).astype(np.float32)
                hist_blend = build_blend_prediction_batch(
                    traffic=traffic,  # type: ignore[arg-type]
                    baseline=baseline,  # type: ignore[arg-type]
                    day_kind=day_kind,  # type: ignore[arg-type]
                    tod=tod,  # type: ignore[arg-type]
                    alphas=alphas,  # type: ignore[arg-type]
                    starts=hist_sample_start,
                    horizon_steps=args.horizon_steps,
                )
                hist_normal_pred = learned_normal.predict_many_full(
                    full_hist=hist_context,
                    full_blend=hist_blend,
                    node_idx=hist_node_idx,
                    node_valid=hist_node_valid,
                    times=times,
                    sample_start=hist_sample_start,
                )[:, : args.input_steps]
            else:
                hist_context = np.stack([pending[i]["hist_context"] for i in valid_hist_items]).astype(np.float32)
                hist_blend = np.stack([pending[i]["hist_blend_pred"] for i in valid_hist_items]).astype(np.float32)
                hist_normal_pred = learned_normal.predict_many(
                    hist=hist_context,
                    blend=hist_blend,
                    node_idx=hist_node_idx,
                    node_valid=hist_node_valid,
                    times=times,
                    sample_start=hist_sample_start,
                )
            hist_scale = np.stack([pending[i]["hist_scale"] for i in valid_hist_items]).astype(np.float32)
            hist_target = np.stack([pending[i]["hist"] for i in valid_hist_items]).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                hist_residual_normal_batch[valid_hist_items] = (hist_target - hist_normal_pred) / hist_scale
        hist_residual_normal_batch[~np.isfinite(hist_residual_normal_batch)] = 0.0

    for item, normal_pred, hist_residual_normal in zip(pending, normal_pred_batch, hist_residual_normal_batch):
        sample_split = str(item["sample_split"])
        blend_pred = item["normal_pred"]  # type: ignore[assignment]
        fut_scale = item["fut_scale"]  # type: ignore[assignment]
        with np.errstate(divide="ignore", invalid="ignore"):
            normal_delta = (normal_pred - blend_pred) / fut_scale  # type: ignore[operator]
        normal_delta[~np.isfinite(normal_delta)] = 0.0
        appended = append_residual_sample(
            buffer=buffer,
            args=args,
            hist=item["hist"],  # type: ignore[arg-type]
            hist_base=item["hist_base"],  # type: ignore[arg-type]
            hist_scale=item["hist_scale"],  # type: ignore[arg-type]
            hist_residual_normal=hist_residual_normal,
            fut_scale=fut_scale,  # type: ignore[arg-type]
            actual_future=item["actual_future"],  # type: ignore[arg-type]
            normal_pred=normal_pred,
            normal_delta=normal_delta,
            heatmap_baseline=normal_pred,
            node_idx=item["node_idx"],  # type: ignore[arg-type]
            node_valid=item["node_valid"],  # type: ignore[arg-type]
            node_affected=item["node_affected"],  # type: ignore[arg-type]
            node_ctx=item["node_ctx"],  # type: ignore[arg-type]
            global_ctx=item["global_ctx"],  # type: ignore[arg-type]
            event_aux=item["event_aux"],  # type: ignore[arg-type]
            sample_split=sample_split,
            region_code=int(item["region_code"]),
        )
        if appended:
            counts[sample_split] += 1
            flush_if_needed(writer, buffer)
    pending.clear()
    return counts


def build_cache(args: argparse.Namespace, cache_path: Path, device: torch.device) -> None:
    data_dir = args.data_dir.resolve()
    event_root = args.event_root.resolve()
    raw_label_dir = args.raw_label_dir.resolve()
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)
    writer = H5SampleWriter(cache_path, args)

    try:
        for region_code, region_name in enumerate(args.regions):
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
            learned_normal = None
            if args.normal_model_dir is not None:
                print(f"[{region_name}] loading learned normal branch", flush=True)
                learned_normal = LearnedNormalRegion.load(args.normal_model_dir, region_name, device)
                if learned_normal.model.horizon_steps != args.horizon_steps:
                    raise ValueError(
                        f"normal model horizon {learned_normal.model.horizon_steps} "
                        f"does not match residual horizon {args.horizon_steps}"
                    )

            print(f"[{region_name}] writing full-candidate samples", flush=True)
            events = pd.read_csv(event_root / region_name / "event_labels.csv")
            raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
            candidate_lookup = build_candidate_lookup(raw_nodes)
            buffer = init_buffer()
            region_samples = 0
            region_split_counts = {name: 0 for name in SPLIT_TO_CODE}
            pending_learned: list[dict[str, np.ndarray | int | str]] = []
            pending_split_counts = {name: 0 for name in SPLIT_TO_CODE}
            stop_region = False

            for row in events.itertuples(index=False):
                if stop_region:
                    break
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
                    sample_split = split_name(sample_start, total_steps)
                    if (
                        args.max_cache_samples_per_split > 0
                        and region_split_counts[sample_split] + pending_split_counts[sample_split]
                        >= args.max_cache_samples_per_split
                    ):
                        if all(
                            region_split_counts[name] + pending_split_counts[name] >= args.max_cache_samples_per_split
                            for name in region_split_counts
                        ):
                            stop_region = True
                            break
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
                    hist_context = None
                    hist_blend_pred = None
                    if learned_normal is not None and args.use_dual_hist_residual:
                        hist_context_start = input_start - args.input_steps
                        if hist_context_start >= 0:
                            hist_context_idx = np.arange(hist_context_start, input_start, dtype=np.int32)
                            hist_context = traffic[hist_context_idx][:, node_idx, :]
                            hist_last_obs = traffic[input_start - 1, node_idx, :]
                            hist_blend_pred = np.empty_like(hist, dtype=np.float32)
                            for h in range(args.input_steps):
                                hist_blend_pred[h] = hist_base[h] + alphas[h][None, :] * (hist_last_obs - hist_base[h])

                    global_ctx = np.concatenate(
                        [make_event_context(row, offset, args.horizon_steps), make_time_features(times, sample_start)]
                    ).astype(np.float32)
                    global_ctx[~np.isfinite(global_ctx)] = 0.0
                    event_aux = make_event_aux(row).astype(np.float32)

                    if learned_normal is not None:
                        pending_item: dict[str, np.ndarray | int | str] = {
                            "hist": hist.astype(np.float32),
                            "hist_base": hist_base.astype(np.float32),
                            "hist_scale": hist_scale.astype(np.float32),
                            "fut_scale": fut_scale.astype(np.float32),
                            "actual_future": actual_future.astype(np.float32),
                            "normal_pred": normal_pred.astype(np.float32),
                            "node_idx": node_idx.astype(np.int32),
                            "node_valid": node_valid.astype(np.float32),
                            "node_affected": node_affected.astype(np.float32),
                            "node_ctx": node_ctx.astype(np.float32),
                            "global_ctx": global_ctx,
                            "event_aux": event_aux,
                            "sample_split": sample_split,
                            "region_code": region_code,
                            "sample_start": sample_start,
                        }
                        if hist_context is not None and hist_blend_pred is not None:
                            pending_item["hist_context"] = hist_context.astype(np.float32)
                            pending_item["hist_blend_pred"] = hist_blend_pred.astype(np.float32)
                            pending_item["hist_sample_start"] = input_start
                        pending_learned.append(pending_item)
                        pending_split_counts[sample_split] += 1
                        if len(pending_learned) >= max(1, args.normal_infer_batch_size):
                            new_counts = flush_learned_normal_pending(
                                pending=pending_learned,
                                learned_normal=learned_normal,
                                buffer=buffer,
                                writer=writer,
                                args=args,
                                times=times,
                                traffic=traffic,
                                baseline=baseline,
                                day_kind=day_kind,
                                tod=tod,
                                alphas=alphas,
                            )
                            for name, count in new_counts.items():
                                region_split_counts[name] += count
                                region_samples += count
                            pending_split_counts = {name: 0 for name in SPLIT_TO_CODE}

                        hit_region_cap = (
                            args.max_cache_samples_per_region > 0
                            and region_samples + len(pending_learned) >= args.max_cache_samples_per_region
                        )
                        hit_split_cap = (
                            args.max_cache_samples_per_split > 0
                            and all(
                                region_split_counts[name] + pending_split_counts[name]
                                >= args.max_cache_samples_per_split
                                for name in region_split_counts
                            )
                        )
                        if hit_region_cap or hit_split_cap:
                            new_counts = flush_learned_normal_pending(
                                pending=pending_learned,
                                learned_normal=learned_normal,
                                buffer=buffer,
                                writer=writer,
                                args=args,
                                times=times,
                                traffic=traffic,
                                baseline=baseline,
                                day_kind=day_kind,
                                tod=tod,
                                alphas=alphas,
                            )
                            for name, count in new_counts.items():
                                region_split_counts[name] += count
                                region_samples += count
                            pending_split_counts = {name: 0 for name in SPLIT_TO_CODE}
                            stop_region = True
                            break
                    else:
                        appended = append_residual_sample(
                            buffer=buffer,
                            args=args,
                            hist=hist,
                            hist_base=hist_base,
                            hist_scale=hist_scale,
                            hist_residual_normal=np.zeros_like(hist, dtype=np.float32),
                            fut_scale=fut_scale,
                            actual_future=actual_future,
                            normal_pred=normal_pred,
                            normal_delta=np.zeros_like(normal_pred, dtype=np.float32),
                            heatmap_baseline=fut_base,
                            node_idx=node_idx,
                            node_valid=node_valid,
                            node_affected=node_affected,
                            node_ctx=node_ctx,
                            global_ctx=global_ctx,
                            event_aux=event_aux,
                            sample_split=sample_split,
                            region_code=region_code,
                        )
                        if not appended:
                            continue
                        region_samples += 1
                        region_split_counts[sample_split] += 1
                        flush_if_needed(writer, buffer)
                        hit_region_cap = args.max_cache_samples_per_region > 0 and region_samples >= args.max_cache_samples_per_region
                        hit_split_cap = (
                            args.max_cache_samples_per_split > 0
                            and all(count >= args.max_cache_samples_per_split for count in region_split_counts.values())
                        )
                        if hit_region_cap or hit_split_cap:
                            stop_region = True
                            break
            if learned_normal is not None and pending_learned:
                new_counts = flush_learned_normal_pending(
                    pending=pending_learned,
                    learned_normal=learned_normal,
                    buffer=buffer,
                    writer=writer,
                    args=args,
                    times=times,
                    traffic=traffic,
                    baseline=baseline,
                    day_kind=day_kind,
                    tod=tod,
                    alphas=alphas,
                )
                for name, count in new_counts.items():
                    region_split_counts[name] += count
                    region_samples += count
            flush_if_needed(writer, buffer, force=True)
            split_msg = ", ".join(f"{name}={count}" for name, count in region_split_counts.items())
            print(f"[{region_name}] region cache samples: {region_samples} ({split_msg}); total so far: {writer.count}", flush=True)
    finally:
        writer.close()


def iter_train_chunks(h5: h5py.File, chunk_size: int = 1024) -> Iterator[slice]:
    n = int(h5.attrs["samples"])
    for start in range(0, n, chunk_size):
        yield slice(start, min(n, start + chunk_size))


def safe_std(var: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(var, 0.0))
    std[std < 1e-6] = 1.0
    return std.astype(np.float32)


def compute_stats(cache_path: Path) -> CacheStats:
    with h5py.File(cache_path, "r+") as h5:
        if "stats/hist_mean" in h5:
            stats = h5["stats"]
            hist_shape = stats["hist_mean"][()].shape
            hist_normal_mean = stats["hist_normal_mean"][()] if "hist_normal_mean" in stats else np.zeros(hist_shape, dtype=np.float32)
            hist_normal_std = stats["hist_normal_std"][()] if "hist_normal_std" in stats else np.ones(hist_shape, dtype=np.float32)
            return CacheStats(
                hist_mean=stats["hist_mean"][()],
                hist_std=stats["hist_std"][()],
                hist_normal_mean=hist_normal_mean,
                hist_normal_std=hist_normal_std,
                node_mean=stats["node_mean"][()],
                node_std=stats["node_std"][()],
                global_mean=stats["global_mean"][()],
                global_std=stats["global_std"][()],
                event_aux_mean=stats["event_aux_mean"][()],
                event_aux_std=stats["event_aux_std"][()],
            )

        sums: dict[str, np.ndarray] = {}
        sqs: dict[str, np.ndarray] = {}
        counts: dict[str, float] = {}
        specs = {
            "hist": ("hist_residual", (1, 1, 1, len(CHANNELS)), (0, 1, 2)),
            "hist_normal": ("hist_residual_normal", (1, 1, 1, len(CHANNELS)), (0, 1, 2)),
            "node": ("node_context", (1, 1, 8), (0, 1)),
            "global": ("global_context", (1, 17), (0,)),
            "event_aux": ("event_aux", (1, 3), (0,)),
        }
        for key, (_, shape, _) in specs.items():
            sums[key] = np.zeros(shape[1:], dtype=np.float64)
            sqs[key] = np.zeros(shape[1:], dtype=np.float64)
            counts[key] = 0.0

        for slc in iter_train_chunks(h5):
            split = h5["split"][slc]
            mask = split == SPLIT_TO_CODE["train"]
            if not np.any(mask):
                continue
            for key, (dataset, _, axes) in specs.items():
                if dataset not in h5:
                    continue
                values = h5[dataset][slc][mask].astype(np.float64)
                sums[key] += values.sum(axis=axes)
                sqs[key] += np.square(values).sum(axis=axes)
                counts[key] += float(np.prod([values.shape[axis] for axis in axes]))

        means = {key: sums[key] / max(counts[key], 1.0) for key in sums}
        vars_ = {key: sqs[key] / max(counts[key], 1.0) - np.square(means[key]) for key in sums}
        stds = {key: safe_std(vars_[key]) for key in vars_}

        node_std = stds["node"]
        node_std[..., 6] = 1.0
        stats_group = h5.require_group("stats")
        arrays = {
            "hist_mean": means["hist"].reshape(1, 1, 1, -1).astype(np.float32),
            "hist_std": stds["hist"].reshape(1, 1, 1, -1).astype(np.float32),
            "hist_normal_mean": means["hist_normal"].reshape(1, 1, 1, -1).astype(np.float32),
            "hist_normal_std": stds["hist_normal"].reshape(1, 1, 1, -1).astype(np.float32),
            "node_mean": means["node"].reshape(1, -1).astype(np.float32),
            "node_std": node_std.reshape(1, -1).astype(np.float32),
            "global_mean": means["global"].astype(np.float32),
            "global_std": stds["global"].astype(np.float32),
            "event_aux_mean": means["event_aux"].astype(np.float32),
            "event_aux_std": stds["event_aux"].astype(np.float32),
        }
        for name, arr in arrays.items():
            if name in stats_group:
                del stats_group[name]
            stats_group.create_dataset(name, data=arr)
        return CacheStats(**arrays)


def make_loader(cache_path: Path, indices: np.ndarray, stats: CacheStats, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = H5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(
    model: FullCandidateSTGNNHeatmap,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        hist,
        hist_normal,
        node,
        global_context,
        normal_delta,
        y,
        y_mask,
        impact,
        impact_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)
    pred_y, pred_impact, pred_event_aux, pred_node_logits = model(hist, node, global_context, normal_delta)
    residual_loss_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = (residual_loss_raw * y_mask).sum() / y_mask.sum().clamp_min(1.0)
    impact_loss_raw = nn.functional.smooth_l1_loss(pred_impact, impact, reduction="none")
    impact_loss = (impact_loss_raw * impact_mask).sum() / impact_mask.sum().clamp_min(1.0)
    event_aux_loss = nn.functional.smooth_l1_loss(pred_event_aux, event_aux)
    node_bce = nn.functional.binary_cross_entropy_with_logits(pred_node_logits, node_affected, reduction="none")
    node_aux_loss = (node_bce * node_valid).sum() / node_valid.sum().clamp_min(1.0)
    loss = (
        residual_loss
        + args.heatmap_aux_weight * impact_loss
        + args.event_aux_weight * event_aux_loss
        + args.node_aux_weight * node_aux_loss
    )
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "impact_loss": float(impact_loss.detach().cpu()),
        "event_aux_loss": float(event_aux_loss.detach().cpu()),
        "node_aux_loss": float(node_aux_loss.detach().cpu()),
    }


def evaluate_loader(model: FullCandidateSTGNNHeatmap, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "residual_loss": 0.0, "impact_loss": 0.0, "event_aux_loss": 0.0, "node_aux_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            loss, parts = compute_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def forecast_metrics_for_loader(
    model: FullCandidateSTGNNHeatmap,
    loader: DataLoader,
    betas: list[float],
    device: torch.device,
) -> dict[float, dict[str, float]]:
    horizon_steps = int(getattr(model, "horizon_steps", 0))
    sums = {
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
            "affected_nodes": 0.0,
            "valid_nodes": 0.0,
            "h_all_model": np.zeros(horizon_steps, dtype=np.float64),
            "h_all_base": np.zeros(horizon_steps, dtype=np.float64),
            "h_all_count": np.zeros(horizon_steps, dtype=np.float64),
            "h_aff_model": np.zeros(horizon_steps, dtype=np.float64),
            "h_aff_base": np.zeros(horizon_steps, dtype=np.float64),
            "h_aff_count": np.zeros(horizon_steps, dtype=np.float64),
        }
        for beta in betas
    }
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
            if getattr(model, "hist_input_channels", len(CHANNELS)) > len(CHANNELS):
                hist = torch.cat([hist, hist_normal], dim=-1)
            pred_y, _pred_impact, _pred_event, _pred_node = model(hist, node, global_context, normal_delta)
            all_mask = y_mask.bool()
            affected_mask = all_mask & node_affected[:, None, :, None].bool()
            unaffected_mask = all_mask & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
            base_abs = torch.abs(y)
            for beta in betas:
                model_abs = torch.abs(beta * pred_y - y)
                for prefix, mask in [("all", all_mask), ("aff", affected_mask), ("unaff", unaffected_mask)]:
                    count = mask.sum().item()
                    if count <= 0:
                        continue
                    sums[beta][f"{prefix}_model"] += float(model_abs[mask].sum().detach().cpu())
                    sums[beta][f"{prefix}_base"] += float(base_abs[mask].sum().detach().cpu())
                    sums[beta][f"{prefix}_count"] += float(count)
                for horizon_idx in range(horizon_steps):
                    h_all = all_mask[:, horizon_idx]
                    h_aff = affected_mask[:, horizon_idx]
                    for prefix, mask in [("h_all", h_all), ("h_aff", h_aff)]:
                        count = mask.sum().item()
                        if count <= 0:
                            continue
                        sums[beta][f"{prefix}_model"][horizon_idx] += float(model_abs[:, horizon_idx][mask].sum().detach().cpu())
                        sums[beta][f"{prefix}_base"][horizon_idx] += float(base_abs[:, horizon_idx][mask].sum().detach().cpu())
                        sums[beta][f"{prefix}_count"][horizon_idx] += float(count)
                valid_count = node_valid.sum().item()
                if valid_count > 0:
                    sums[beta]["affected_nodes"] += float((node_affected * node_valid).sum().detach().cpu())
                    sums[beta]["valid_nodes"] += float(valid_count)

    out: dict[float, dict[str, float]] = {}
    for beta, vals in sums.items():
        row: dict[str, float] = {}
        for prefix, label in [("all", "all_candidates"), ("aff", "affected_candidates"), ("unaff", "unaffected_candidates")]:
            model_mae = vals[f"{prefix}_model"] / max(vals[f"{prefix}_count"], 1.0)
            base_mae = vals[f"{prefix}_base"] / max(vals[f"{prefix}_count"], 1.0)
            row[f"{label}_model_robust_mae"] = model_mae
            row[f"{label}_baseline_robust_mae"] = base_mae
            row[f"{label}_improvement_pct"] = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else float("nan")
        for horizon_idx in range(horizon_steps):
            step = horizon_idx + 1
            for prefix, label in [("h_all", "all_candidates"), ("h_aff", "affected_candidates")]:
                model_mae = vals[f"{prefix}_model"][horizon_idx] / max(vals[f"{prefix}_count"][horizon_idx], 1.0)
                base_mae = vals[f"{prefix}_base"][horizon_idx] / max(vals[f"{prefix}_count"][horizon_idx], 1.0)
                row[f"horizon_{step:02d}_{label}_model_robust_mae"] = float(model_mae)
                row[f"horizon_{step:02d}_{label}_baseline_robust_mae"] = float(base_mae)
                row[f"horizon_{step:02d}_{label}_improvement_pct"] = (
                    100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else float("nan")
                )
        row["affected_node_rate"] = vals["affected_nodes"] / max(vals["valid_nodes"], 1.0)
        out[beta] = row
    return out


def split_indices(cache_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(cache_path, "r") as h5:
        split = h5["split"][:]
    return {name: np.flatnonzero(split == code) for name, code in SPLIT_TO_CODE.items()}


def region_codes(cache_path: Path) -> np.ndarray:
    with h5py.File(cache_path, "r") as h5:
        return h5["region_code"][:]


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Full-candidate STGNN heatmap training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    metrics: dict[str, dict[str, float]],
    region_metrics: pd.DataFrame,
    split_counts: dict[str, int],
    residual_beta: float,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = ["# 完整候选集合 STGNN + 节点时间影响热力图", ""]
    lines.append("## 一句话结论")
    lines.append("")
    if args.graph_mode == "directional":
        graph_desc = "使用区分相对 postmile 左右方向的 directional graph propagation"
    else:
        graph_desc = "使用不区分上下游方向的 undirected distance graph propagation"
    lines.append(
        f"这一版使用 HDF5 磁盘缓存支持完整候选节点集合，{graph_desc}，并可加入节点-时间 impact heatmap 辅助监督。"
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
    lines.append("## 关键变化")
    lines.append("")
    lines.append(f"- 候选节点集合上限设为 {args.max_candidate_nodes}；主实验建议使用完整候选集合上限 36。")
    lines.append("- 训练数据使用 HDF5 缓存，不再一次性把所有大张量放进内存。")
    lines.append("- 新增节点-时间 impact heatmap 辅助目标，比事件级 severity/recovery/spread 更贴近模型输出。")
    if args.use_normal_delta:
        lines.append("- 新增 normal_delta 输入：把 learned normal 相对统计 blend 的修正量作为 future known covariate。")
    if args.use_normal_delta_abs:
        lines.append("- 新增 normal_delta_abs 输入：把 learned normal 与统计 blend 的差异强度作为 uncertainty/disagreement proxy。")
    if args.use_dual_hist_residual:
        lines.append("- 新增 dual historical residual 输入：同时编码统计 residual 和 learned-normal historical residual。")
    if args.use_temporal_decay_head:
        lines.append("- 新增 impact-aware temporal decay head：用节点级 horizon gate 显式调节事故 residual 的持续/衰减形状。")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 区域: {', '.join(args.regions)}")
    lines.append(f"- input_steps: {args.input_steps}")
    lines.append(f"- horizon_steps: {args.horizon_steps}")
    lines.append(f"- max_candidate_nodes: {args.max_candidate_nodes}")
    lines.append(f"- candidate_pm_radius: {args.candidate_pm_radius}")
    lines.append(f"- hidden_dim: {args.hidden_dim}")
    lines.append(f"- graph_layers: {args.graph_layers}")
    lines.append(f"- graph_mode: {args.graph_mode}")
    lines.append(f"- graph_sigma: {args.graph_sigma}")
    lines.append(f"- heatmap_aux_weight: {args.heatmap_aux_weight}")
    lines.append(f"- event_aux_weight: {args.event_aux_weight}")
    lines.append(f"- node_aux_weight: {args.node_aux_weight}")
    lines.append(f"- normal_model_dir: {args.normal_model_dir if args.normal_model_dir is not None else 'statistical_blend'}")
    lines.append(f"- normal_inference_scope: {args.normal_inference_scope}")
    lines.append(f"- use_normal_delta: {args.use_normal_delta}")
    lines.append(f"- use_normal_delta_abs: {args.use_normal_delta_abs}")
    lines.append(f"- use_dual_hist_residual: {args.use_dual_hist_residual}")
    lines.append(f"- use_temporal_decay_head: {args.use_temporal_decay_head}")
    lines.append(f"- residual_beta: {residual_beta:.2f}")
    lines.append("")

    split_df = pd.DataFrame([{"split": key, "samples": value} for key, value in split_counts.items()])
    lines.append("## 样本数量")
    lines.append("")
    lines.append(split_df.to_markdown(index=False))
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

    finite_val = log_df["val_loss"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_val.empty:
        best = log_df.loc[log_df["train_loss"].idxmin()]
        best_metric_name = "train_loss"
    else:
        best = log_df.loc[finite_val.idxmin()]
        best_metric_name = "val_loss"
    lines.append("## 训练情况")
    lines.append("")
    lines.append(f"- 最佳轮数 best_epoch: {int(best['epoch'])}")
    lines.append(f"- 选择指标 best_metric: {best_metric_name}")
    lines.append(f"- 最佳验证损失 best_val_loss: {best['val_loss']:.4f}")
    lines.append("")

    lines.append("## 仍然存在的限制")
    lines.append("")
    lines.append("- 局部图仍然主要基于 signed postmile 距离，还没有融合完整路网拓扑。")
    lines.append("- impact heatmap 辅助是否真的带来增益，需要继续跑 no-heatmap 消融验证。")
    if args.normal_model_dir is None:
        lines.append("- 当前 normal forecaster 仍然是统计型基线，最终应替换成更强的 normal backbone。")
    else:
        lines.append("- learned normal branch 在事故 cache 中使用候选子图近似 full-region normal STGNN，仍需和完整路网推理对照。")
    lines.append("")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.use_dual_hist_residual and args.input_steps > args.horizon_steps:
        raise ValueError("--use-dual-hist-residual currently requires input_steps <= horizon_steps")
    if args.use_dual_hist_residual and args.normal_model_dir is None:
        raise ValueError("--use-dual-hist-residual requires --normal-model-dir")
    if args.use_normal_delta_abs and args.normal_model_dir is None:
        raise ValueError("--use-normal-delta-abs requires --normal-model-dir")
    if args.normal_inference_scope == "full" and args.normal_model_dir is None:
        raise ValueError("--normal-inference-scope full requires --normal-model-dir")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_path is None:
        cache_path = output_dir / "full_candidate_samples.h5"
    else:
        cache_path = args.cache_path.resolve()

    if args.rebuild_cache or not cache_path.exists():
        print(f"building cache at {cache_path}", flush=True)
        build_cache(args, cache_path, device)
    else:
        print(f"using existing cache at {cache_path}", flush=True)

    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)
    train_indices_full = indices["train"]
    if args.max_train_samples > 0 and train_indices_full.size > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(rng.choice(train_indices_full, size=args.max_train_samples, replace=False))
    else:
        train_indices = train_indices_full

    train_loader = make_loader(cache_path, train_indices, stats, args.batch_size, shuffle=True)
    val_loader = make_loader(cache_path, indices["val"], stats, args.batch_size, shuffle=False)

    model = FullCandidateSTGNNHeatmap(
        channels=len(CHANNELS),
        hist_input_channels=len(CHANNELS) * (2 if args.use_dual_hist_residual else 1),
        node_context_dim=8,
        global_context_dim=17,
        horizon_steps=args.horizon_steps,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        graph_sigma=args.graph_sigma,
        graph_mode=args.graph_mode,
        use_normal_delta=args.use_normal_delta,
        use_normal_delta_abs=args.use_normal_delta_abs,
        use_temporal_decay_head=args.use_temporal_decay_head,
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
            loss, _ = compute_loss(model, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(model, val_loader, args, device)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"}})
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_metrics['loss']:.4f}", flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    beta_candidates = [float(x) for x in np.linspace(0.0, 1.0, 21)]
    val_metrics_by_beta = forecast_metrics_for_loader(model, val_loader, beta_candidates, device)
    beta_df = pd.DataFrame([{"residual_beta": beta, **metrics} for beta, metrics in val_metrics_by_beta.items()])
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)
    residual_beta = float(beta_df.loc[beta_df["all_candidates_model_robust_mae"].idxmin(), "residual_beta"])

    metrics: dict[str, dict[str, float]] = {}
    for split, idx in indices.items():
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]

    region_code_arr = region_codes(cache_path)
    region_rows = []
    for code, region_name in enumerate(args.regions):
        mask_idx = indices["test"][region_code_arr[indices["test"]] == code]
        if mask_idx.size == 0:
            continue
        loader = make_loader(cache_path, mask_idx, stats, args.batch_size, shuffle=False)
        row = {"region": region_name, "samples": int(mask_idx.size)}
        row.update(forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta])
        region_rows.append(row)
    region_metrics = pd.DataFrame(region_rows)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": json_safe_args(args),
            "cache_path": str(cache_path),
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "region_metrics": region_rows,
                "samples": split_counts,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_report(output_dir, args, metrics, region_metrics, split_counts, residual_beta, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
