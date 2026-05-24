"""Affected / Unaffected Gate — controls per-(node, horizon) incident influence.

This is the design doc §8 spec, not a stub. Gate is small enough that the real
implementation fits in 20 lines.

Output convention:
  g ∈ [0, 1]^(B, N, T_p)
  pred = pred_normal_final + g[..., None] * delta_pred

Init: gate biases set so initial sigmoid ≈ 0.1 (gentle activation, lets incident
branch ramp up gradually).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AffectedGate(nn.Module):
    def __init__(self, hidden_dim: int, t_p: int):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.horizon_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_p),
        )
        # gentle init: sigmoid(-2.2) ≈ 0.1
        nn.init.constant_(self.node_mlp[-1].bias, -2.2)
        nn.init.constant_(self.horizon_mlp[-1].bias, -2.2)

    def forward(self, z_normal: torch.Tensor, z_incident: torch.Tensor) -> torch.Tensor:
        # z_*: (B, N, d)
        g_in = torch.cat([z_normal, z_incident], dim=-1)          # (B, N, 2d)
        g_node = torch.sigmoid(self.node_mlp(g_in)).squeeze(-1)   # (B, N)
        g_horizon = torch.sigmoid(self.horizon_mlp(g_in))         # (B, N, T_p)
        return g_node.unsqueeze(-1) * g_horizon                   # (B, N, T_p)
