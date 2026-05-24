"""NormalBranch — periodic / baseline-pattern encoder.

Real implementation (v1) per design doc §5.2 + §5.3:
  • Multi-scale temporal patching (long-focus weighted) — replaces mean-over-time stub
  • L_enc-layer sparse-attention spatial encoder over region-sampled graph
  • Region-conditioned via nn.Embedding
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .patching import MultiScalePatching
from .sparse_attn import SparseTransformerLayer, batched_edge_index


class NormalBranch(nn.Module):
    def __init__(self, c_x: int, c_meta: int, d_t: int, n_regions: int,
                 hidden_dim: int, t_h: int, t_p: int,
                 n_heads: int = 4, n_enc_layers: int = 2):
        super().__init__()
        self.c_x = c_x
        self.t_p = t_p
        self.hidden_dim = hidden_dim

        # Multi-scale patching with long-focus init weights
        self.patching = MultiScalePatching(
            c_x=c_x, d_t=d_t, hidden_dim=hidden_dim, t_h=t_h,
            init_weights=[0.5, 0.3, 0.2],
        )
        self.static_proj = nn.Linear(c_meta, hidden_dim)
        self.region_emb = nn.Embedding(n_regions, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Spatial-temporal encoder: L sparse-attention transformer layers
        self.encoder_layers = nn.ModuleList([
            SparseTransformerLayer(hidden_dim, n_heads=n_heads, ffn_mult=4)
            for _ in range(n_enc_layers)
        ])
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Init pred head (used as auxiliary monitoring; final pred comes from
        # DISTNet.pred_normal_final_head after cross-attention).
        self.pred_head = nn.Linear(hidden_dim, t_p * c_x)

    def forward(self, x_hist: torch.Tensor, x_hist_mask: torch.Tensor,
                time_enc: torch.Tensor, static_meta: torch.Tensor,
                region_code: torch.Tensor,
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs:
          x_hist        (B, N, T_h, C_x)
          x_hist_mask   (B, N, T_h, C_x) bool
          time_enc      (B, T_h, d_t)
          static_meta   (B, N, C_meta)
          region_code   (B,) long
          edge_index    (2, E) -- region sparse graph

        Outputs:
          z_normal      (B, N, d)
          pred_normal   (B, N, T_p, C_x)
        """
        B, N, T_h, C_x = x_hist.shape

        x_in = torch.cat([x_hist, x_hist_mask.float()], dim=-1)            # (B, N, T_h, 2C_x)
        e_temporal = self.patching(x_in, time_enc)                          # (B, N, d)

        s = self.static_proj(static_meta)                                   # (B, N, d)
        r = self.region_emb(region_code).unsqueeze(1).expand(-1, N, -1)     # (B, N, d)

        h = self.input_norm(e_temporal + s + r)                             # (B, N, d)

        # Encoder over batched-disjoint sparse graph
        h_flat = h.reshape(B * N, self.hidden_dim)
        big_edges = batched_edge_index(edge_index.to(h.device), B, N)
        for layer in self.encoder_layers:
            h_flat = layer(h_flat, big_edges)
        h_flat = self.encoder_norm(h_flat)
        z_normal = h_flat.reshape(B, N, self.hidden_dim)

        pred = self.pred_head(z_normal).view(B, N, self.t_p, self.c_x)
        return z_normal, pred
