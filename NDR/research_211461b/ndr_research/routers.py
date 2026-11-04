"""History-only shared routing for the isolated NDR research experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_staeformer_deseason import HorizonGate


@dataclass(frozen=True)
class ReliableHistory:
    filled: torch.Tensor
    observed: torch.Tensor
    last_value: torch.Tensor
    last_index: torch.Tensor
    previous_index: torch.Tensor
    age: torch.Tensor
    available: torch.Tensor


def reliable_history(
    history: torch.Tensor,
    baseline: torch.Tensor,
    observed: torch.Tensor,
) -> ReliableHistory:
    """Ignore missing sentinels and fill causally from within the supplied window."""
    if history.ndim != 3 or history.shape != baseline.shape:
        raise ValueError("history and baseline must have matching (B, N, T) shapes")
    if history.shape[-1] < 2:
        raise ValueError("routing requires at least two historical steps")
    if observed.shape != history.shape or observed.dtype != torch.bool:
        raise ValueError("history observation mask must be matching boolean values")
    if not (history.device == baseline.device == observed.device):
        raise ValueError("history tensors must share a device")
    if not torch.isfinite(baseline).all():
        raise FloatingPointError("historical climatology must be finite")
    if not torch.isfinite(history[observed]).all() or (history[observed] < 0).any():
        raise FloatingPointError("observed flow must be finite and nonnegative")
    steps = history.shape[-1]
    positions = torch.arange(steps, device=history.device).view(1, 1, steps)
    indices = torch.where(observed, positions, -1)
    causal_indices = indices.cummax(dim=-1).values
    # Remove sentinels before gathering, so missing NaNs never enter arithmetic.
    safe = torch.where(observed, history, torch.zeros_like(history))
    gathered = safe.gather(-1, causal_indices.clamp_min(0))
    filled = torch.where(causal_indices >= 0, gathered, baseline)
    last_index = indices.amax(dim=-1)
    previous = torch.where(positions < last_index.unsqueeze(-1), indices, -1)
    previous_index = previous.amax(dim=-1)
    available = last_index >= 0
    last_value = safe.gather(-1, last_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    last_value = torch.where(available, last_value, baseline[..., -1])
    age = torch.where(available, steps - 1 - last_index, steps)
    return ReliableHistory(
        filled=filled,
        observed=observed,
        last_value=last_value,
        last_index=last_index,
        previous_index=previous_index,
        age=age,
        available=available,
    )


def persistence_anchor(
    state: ReliableHistory,
    baseline_history: torch.Tensor,
    *,
    reliable: bool = False,
    max_age: int = 2,
) -> torch.Tensor:
    """Optional age-limited observed anchor with a current-climatology fallback."""
    if type(max_age) is not int or not 0 <= max_age < state.filled.shape[-1]:
        raise ValueError("reference maximum age is outside the historical window")
    if not reliable:
        return state.filled[..., -1:]
    usable = state.available & (state.age <= max_age)
    anchor = torch.where(usable, state.last_value, baseline_history[..., -1])
    return anchor.unsqueeze(-1)


class MaskSignedGate(nn.Module):
    """Original monotone gate plus a zero-initialized node-shared correction."""

    feature_names = (
        "last_signed_residual",
        "observed_mean_absolute_residual",
        "observed_mean_signed_residual",
        "observed_residual_standard_deviation",
        "signed_observed_slope",
        "signed_recovery",
        "observation_fraction",
        "latest_observation_age_fraction",
        "future_anchor_gap",
        "horizon_fraction",
    )

    def __init__(self, horizon: int = 12, hidden_dim: int = 16) -> None:
        super().__init__()
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be positive")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("router hidden dimension must be positive")
        self.horizon = horizon
        self.base = HorizonGate(horizon)
        self.correction = nn.Sequential(
            nn.Linear(len(self.feature_names), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def features(
        self,
        state: ReliableHistory,
        baseline_history: torch.Tensor,
        baseline_future: torch.Tensor,
        persistence: torch.Tensor,
        node_scale: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, steps = state.filled.shape
        if baseline_future.shape != (batch, nodes, self.horizon):
            raise ValueError("future climatology does not match routing horizon")
        if persistence.shape != (batch, nodes, 1):
            raise ValueError("persistence anchor must have shape (B, N, 1)")
        if node_scale.shape != (nodes,) or not torch.isfinite(node_scale).all():
            raise ValueError("router node scales must be finite with shape (N,)")
        if not (node_scale > 0).all():
            raise ValueError("router node scales must be positive")
        if not torch.isfinite(baseline_future).all():
            raise FloatingPointError("future climatology must be finite")
        scale = node_scale.view(1, nodes, 1)
        residual = (state.filled - baseline_history) / scale
        selected = torch.where(state.observed, residual, torch.zeros_like(residual))
        count = state.observed.sum(-1)
        denominator = count.clamp_min(1).to(residual.dtype)
        mean = selected.sum(-1) / denominator
        deviation = selected.abs().sum(-1) / denominator
        centered = torch.where(
            state.observed,
            residual - mean.unsqueeze(-1),
            torch.zeros_like(residual),
        )
        variance = centered.square().sum(-1) / denominator
        standard_deviation = variance.clamp_min(0).sqrt()
        last = selected.gather(-1, state.last_index.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        last = torch.where(state.available, last, torch.zeros_like(last))
        previous = selected.gather(
            -1, state.previous_index.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        gap = (state.last_index - state.previous_index).clamp_min(1).to(last.dtype)
        slope = torch.where(state.previous_index >= 0, (last - previous) / gap, 0.0)
        # Positive means the residual is moving toward zero, negative away from it.
        recovery = -last.sign() * slope
        fraction = count.to(last.dtype) / steps
        age = state.age.to(last.dtype) / steps
        node_features = torch.stack(
            (last, deviation, mean, standard_deviation, slope, recovery, fraction, age),
            dim=-1,
        ).clamp(-8, 8)
        expanded = node_features.unsqueeze(-2).expand(-1, -1, self.horizon, -1)
        anchor_gap = ((baseline_future - persistence) / scale).clamp(-8, 8)
        horizons = torch.arange(
            1, self.horizon + 1, device=last.device, dtype=last.dtype
        ).view(1, 1, self.horizon).expand(batch, nodes, -1) / self.horizon
        return torch.cat((expanded, anchor_gap.unsqueeze(-1), horizons.unsqueeze(-1)), -1)

    def forward(
        self,
        state: ReliableHistory,
        baseline_history: torch.Tensor,
        baseline_future: torch.Tensor,
        persistence: torch.Tensor,
        global_scale: torch.Tensor,
        node_scale: torch.Tensor,
    ) -> torch.Tensor:
        if global_scale.numel() != 1 or not torch.isfinite(global_scale).all():
            raise ValueError("global residual scale must be one finite value")
        if not (global_scale > 0).all():
            raise ValueError("global residual scale must be positive")
        residual = (state.filled - baseline_history) / global_scale
        deviation = residual.abs().mean(-1)
        slope = (residual[..., -1] - residual[..., -2]).abs()
        logits = (
            self.base.a.view(1, 1, -1)
            - F.softplus(self.base.b).view(1, 1, -1) * deviation.unsqueeze(-1)
            - F.softplus(self.base.c).view(1, 1, -1) * slope.unsqueeze(-1)
        )
        features = self.features(
            state, baseline_history, baseline_future, persistence, node_scale
        )
        return torch.sigmoid(logits + self.correction(features).squeeze(-1))
