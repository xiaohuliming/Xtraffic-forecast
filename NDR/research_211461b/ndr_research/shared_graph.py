"""Node-inductive NDR candidate with sparse roads and low-rank global mixing.

All fitted climatology and per-node scales are runtime inputs. Predictions and
anchors use raw flow units; residual heads use the supplied training residual
scale. No parameter or registered buffer depends on the number of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HistoryState:
    filled: torch.Tensor
    observed: torch.Tensor
    reference: torch.Tensor
    last_index: torch.Tensor
    previous_index: torch.Tensor
    age: torch.Tensor
    available: torch.Tensor


def canonical_history(
    history: torch.Tensor,
    climatology: torch.Tensor,
    observed: torch.Tensor,
) -> HistoryState:
    """Causal in-window fill; missing sentinels never participate in arithmetic."""
    if history.ndim != 3 or climatology.shape != history.shape:
        raise ValueError("history and climatology must match (B, N, T)")
    if history.shape[0] <= 0 or history.shape[1] <= 0 or history.shape[2] < 2:
        raise ValueError("history needs a nonempty batch, nodes and at least two steps")
    if observed.shape != history.shape or observed.dtype != torch.bool:
        raise ValueError("history observation mask must be matching boolean values")
    if not (history.device == climatology.device == observed.device):
        raise ValueError("history tensors must share a device")
    if not history.is_floating_point() or climatology.dtype != history.dtype:
        raise ValueError("history and climatology must share a floating dtype")
    if not torch.isfinite(climatology).all():
        raise FloatingPointError("history climatology must be finite")
    if not torch.isfinite(history[observed]).all() or (history[observed] < 0).any():
        raise FloatingPointError("observed flow must be finite and nonnegative")
    steps = history.shape[-1]
    positions = torch.arange(steps, device=history.device).view(1, 1, steps)
    indices = torch.where(observed, positions, -1)
    latest = indices.cummax(-1).values
    safe = torch.where(observed, history, torch.zeros_like(history))
    gathered = safe.gather(-1, latest.clamp_min(0))
    filled = torch.where(latest >= 0, gathered, climatology)
    last = indices.amax(-1)
    previous = torch.where(positions < last.unsqueeze(-1), indices, -1).amax(-1)
    available = last >= 0
    reference = safe.gather(-1, last.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    reference = torch.where(available, reference, climatology[..., -1])
    age = torch.where(available, steps - 1 - last, steps)
    return HistoryState(filled, observed, reference, last, previous, age, available)


def runtime_node_scale(
    node_scale: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    """Return (B, N, 1) flow-unit scales without storing region-specific state."""
    batch, nodes, _ = history.shape
    if node_scale.device != history.device or node_scale.dtype != history.dtype:
        raise ValueError("node scale must share history device and floating dtype")
    if node_scale.shape == (nodes,):
        scale = node_scale.view(1, nodes, 1).expand(batch, -1, -1)
    elif node_scale.shape == (batch, nodes):
        scale = node_scale.unsqueeze(-1)
    else:
        raise ValueError("node scale must have shape (N,) or (B, N)")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("node scale must contain finite positive training statistics")
    return scale


def history_regime_features(
    state: HistoryState,
    climatology: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Shared signed deviation, recovery and reliability features, shape (B, N, 8)."""
    residual = ((state.filled - climatology) / scale).clamp(-20, 20)
    selected = torch.where(state.observed, residual, torch.zeros_like(residual))
    count = state.observed.sum(-1)
    denominator = count.clamp_min(1).to(residual.dtype)
    mean = selected.sum(-1) / denominator
    deviation = selected.abs().sum(-1) / denominator
    centered = torch.where(state.observed, residual - mean.unsqueeze(-1), 0.0)
    variance = centered.square().sum(-1) / denominator
    standard_deviation = torch.where(count > 0, (variance + 1e-6).sqrt(), 0.0)
    last = selected.gather(-1, state.last_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    last = torch.where(state.available, last, 0.0)
    previous = selected.gather(-1, state.previous_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    gap = (state.last_index - state.previous_index).clamp_min(1).to(residual.dtype)
    slope = torch.where(state.previous_index >= 0, (last - previous) / gap, 0.0)
    recovery = -last.sign() * slope
    steps = state.filled.shape[-1]
    return torch.stack(
        (
            last,
            deviation,
            mean,
            standard_deviation,
            slope,
            recovery,
            count.to(residual.dtype) / steps,
            state.age.to(residual.dtype) / steps,
        ),
        -1,
    ).clamp(-8, 8)


class MultiScaleHistoryEncoder(nn.Module):
    """Shared temporal patch encoders at five-, fifteen- and thirty-minute scales."""

    def __init__(self, history_steps: int, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.history_steps = history_steps
        self.channels = channels
        self.patch_sizes = (1, 3, 6)
        if any(history_steps % size for size in self.patch_sizes):
            raise ValueError("history length must be divisible by the temporal patch sizes")
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(size * channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for size in self.patch_sizes
        )
        self.positions = nn.ParameterList(
            nn.Parameter(torch.empty(history_steps // size, hidden_dim))
            for size in self.patch_sizes
        )
        self.pool_queries = nn.Parameter(torch.empty(len(self.patch_sizes), hidden_dim))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.patch_sizes)))
        self.norm = nn.LayerNorm(hidden_dim)
        for position in self.positions:
            nn.init.trunc_normal_(position, std=0.02)
        nn.init.trunc_normal_(self.pool_queries, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, channels = inputs.shape
        if (steps, channels) != (self.history_steps, self.channels):
            raise ValueError("temporal input does not match the encoder recipe")
        encoded = []
        for index, size in enumerate(self.patch_sizes):
            patches = inputs.reshape(batch, nodes, steps // size, size * channels)
            values = self.projections[index](patches) + self.positions[index]
            logits = (values * self.pool_queries[index]).sum(-1) / values.shape[-1] ** 0.5
            weights = logits.softmax(-1)
            encoded.append((weights.unsqueeze(-1) * values).sum(-2))
        mixture = self.scale_logits.softmax(0)
        result = sum(weight * value for weight, value in zip(mixture, encoded))
        return self.norm(result)


def _validate_edges(edge_index: torch.Tensor, nodes: int, device: torch.device) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must be a (2, E) int64 tensor")
    if edge_index.device != device:
        raise ValueError("road graph and history must share a device")
    if edge_index.numel() and ((edge_index < 0).any() or (edge_index >= nodes).any()):
        raise ValueError("road graph contains a node outside the supplied node set")


def _mean_messages(
    values: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-target degree normalization, memory proportional to nodes and edges."""
    messages = torch.zeros_like(values)
    if sources.numel() == 0:
        return messages
    messages.index_add_(1, targets, values[:, sources])
    degree = values.new_zeros(values.shape[1])
    degree.index_add_(0, targets, values.new_ones(targets.shape))
    return messages / degree.clamp_min(1).view(1, -1, 1)


class BidirectionalRoadBlock(nn.Module):
    """Separate forward/reverse road means plus shared residual feed-forward layers."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.road_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(nodes)
        sources, targets = edge_index
        forward = _mean_messages(normalized, sources, targets)
        reverse = _mean_messages(normalized, targets, sources)
        nodes = nodes + self.road_update(torch.cat((normalized, forward, reverse), -1))
        return nodes + self.feed_forward(self.norm2(nodes))


class LatentGlobalMixer(nn.Module):
    """Cross-attention through K shared tokens, no node-by-node attention matrix."""

    def __init__(self, hidden_dim: int, tokens: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(tokens, hidden_dim))
        nn.init.trunc_normal_(self.tokens, std=0.02)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.to_tokens = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.to_nodes = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.update_strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        normalized = self.node_norm(nodes)
        token_queries = self.tokens.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        tokens, _ = self.to_tokens(token_queries, normalized, normalized, need_weights=False)
        tokens = self.token_norm(token_queries + tokens)
        update, _ = self.to_nodes(normalized, tokens, tokens, need_weights=False)
        return self.output_norm(nodes + self.update_strength.sigmoid() * update)


class SharedHistoryGate(nn.Module):
    """Dimensionless history routing, shared across all nodes and regions."""

    def __init__(self, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.full((horizon,), 2.0))
        self.b = nn.Parameter(torch.zeros(horizon))
        self.c = nn.Parameter(torch.zeros(horizon))
        self.gap_coefficient = nn.Parameter(torch.zeros(horizon))
        self.context = nn.Sequential(
            nn.Linear(hidden_dim + 8, 32), nn.SiLU(), nn.Linear(32, horizon)
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(
        self,
        encoded: torch.Tensor,
        features: torch.Tensor,
        anchor_gap: torch.Tensor,
    ) -> torch.Tensor:
        deviation = features[..., 1:2]
        absolute_slope = features[..., 4:5].abs()
        base = self.a - F.softplus(self.b) * deviation - F.softplus(self.c) * absolute_slope
        correction = self.context(torch.cat((encoded, features), -1))
        return (base + correction + self.gap_coefficient * anchor_gap.clamp(-8, 8)).sigmoid()


class SharedGraphNDR(nn.Module):
    """Node-shared dual-residual forecaster; complexity O(E + N*K) at fixed width.

    Runtime arrays must be derived from deployment-available history and
    train-only fitting. Target/affected fields are deliberately absent from the
    forecast API. Target-region climatology fitting is historical calibration,
    and does not constitute strict zero-shot evaluation.
    """

    output_units = "raw_flow"
    residual_units = "runtime_per_node_training_residual_standard_deviation"

    def __init__(
        self,
        *,
        history_steps: int = 12,
        horizon: int = 12,
        hidden_dim: int = 64,
        road_layers: int = 2,
        latent_tokens: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if type(history_steps) is not int or history_steps < 6 or history_steps % 6:
            raise ValueError("history_steps must be a positive multiple of six")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("forecast horizon must be positive")
        if type(hidden_dim) is not int or hidden_dim < 8:
            raise ValueError("hidden dimension must be at least eight")
        if type(num_heads) is not int or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden dimension must be divisible by positive head count")
        if type(road_layers) is not int or road_layers < 2:
            raise ValueError("candidate requires at least two road propagation layers")
        if latent_tokens not in (8, 16) or type(latent_tokens) is not int:
            raise ValueError("global mixing requires eight or sixteen shared tokens")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self._config = {
            "history_steps": history_steps,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "road_layers": road_layers,
            "latent_tokens": latent_tokens,
            "num_heads": num_heads,
            "dropout": float(dropout),
        }
        self.history_steps = history_steps
        self.horizon = horizon
        self.temporal = MultiScaleHistoryEncoder(history_steps, channels=7, hidden_dim=hidden_dim)
        self.roads = nn.ModuleList(BidirectionalRoadBlock(hidden_dim, dropout) for _ in range(road_layers))
        self.global_mixer = LatentGlobalMixer(hidden_dim, latent_tokens, num_heads, dropout)
        self.normal_head = nn.Linear(hidden_dim, horizon)
        self.disturbance_head = nn.Linear(hidden_dim, horizon)
        self.gate = SharedHistoryGate(hidden_dim, horizon)
        for head in (self.normal_head, self.disturbance_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        self.oracle_sup = True
        self.dual = True
        self.abank = False
        self.aux: dict[str, torch.Tensor] | None = None
        self.last_alpha: torch.Tensor | None = None
        self.last_d: torch.Tensor | None = None
        self.research_spec = {
            "variant": "shared_graph_ndr",
            "output_units": self.output_units,
            "residual_units": self.residual_units,
            "history_policy": "window_causal_observations_then_train_climatology",
            "local_normalization": "center_on_latest_observed_flow_then_divide_by_train_node_scale",
            "normalized_history_clip": 20.0,
            "regime_feature_clip": 8.0,
            "normal_anchor": "train_only_future_climatology",
            "disturbance_anchor": "latest_in_window_observation_or_current_train_climatology",
            "node_identity_parameters": False,
            "region_identity_parameters": False,
            "road_normalization": "separate_direction_per_target_degree_mean",
            "global_mixing": "node_to_latent_then_latent_to_node_cross_attention",
            "complexity": "O(E + N*K) with fixed history, horizon and hidden width",
            "dev_score": "not measured",
        }

    def get_config(self) -> dict[str, Any]:
        """Constructor recipe, intentionally independent of graph and node count."""
        return dict(self._config)

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
        history_mask: torch.Tensor,
        edge_index: torch.Tensor,
        node_scale: torch.Tensor,
    ) -> torch.Tensor:
        state = canonical_history(flow_hist, x_baseline, history_mask)
        batch, nodes, steps = flow_hist.shape
        if steps != self.history_steps or y_baseline.shape != (batch, nodes, self.horizon):
            raise ValueError("history or future climatology does not match the recipe")
        if y_baseline.device != flow_hist.device or y_baseline.dtype != flow_hist.dtype:
            raise ValueError("future climatology must share history device and dtype")
        if not torch.isfinite(y_baseline).all():
            raise FloatingPointError("future climatology must be finite")
        if time_feat.shape != (batch, steps, 2) or time_feat.device != flow_hist.device:
            raise ValueError("time features must match (B, T, 2) on the history device")
        if time_feat.dtype != flow_hist.dtype or not torch.isfinite(time_feat).all():
            raise ValueError("time features must have finite history-matching dtype")
        if not ((time_feat >= 0) & (time_feat < 1)).all():
            raise ValueError("historical time-of-day and day-of-week must be in [0, 1)")
        _validate_edges(edge_index, nodes, flow_hist.device)
        scale = runtime_node_scale(node_scale, flow_hist)
        signed = ((state.filled - x_baseline) / scale).clamp(-20, 20)
        local = ((state.filled - state.reference.unsqueeze(-1)) / scale).clamp(-20, 20)
        calendar_phase = time_feat * (2 * torch.pi)
        calendar = torch.cat((calendar_phase.sin(), calendar_phase.cos()), -1)
        calendar = calendar[:, None].expand(-1, nodes, -1, -1)
        temporal_inputs = torch.cat(
            (signed.unsqueeze(-1), local.unsqueeze(-1), history_mask.to(signed.dtype).unsqueeze(-1), calendar),
            -1,
        )
        encoded = self.temporal(temporal_inputs)
        for road_block in self.roads:
            encoded = road_block(encoded, edge_index)
        encoded = self.global_mixer(encoded)
        features = history_regime_features(state, x_baseline, scale)
        persistence = state.reference.unsqueeze(-1)
        gate = self.gate(encoded, features, (y_baseline - persistence) / scale)
        normal = y_baseline + self.normal_head(encoded) * scale
        disturbance = persistence + self.disturbance_head(encoded) * scale
        prediction = gate * normal + (1 - gate) * disturbance
        self.aux = {
            "g": gate,
            "B": y_baseline,
            "P": persistence,
            "yn": normal,
            "yd": disturbance,
            "node_scale": scale,
        }
        self.last_alpha = gate.detach().mean(-1)
        self.last_d = features[..., 1].detach()
        return prediction
