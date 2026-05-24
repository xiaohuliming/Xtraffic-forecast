"""Multi-scale temporal patching per design doc §5.2.

Splits the (T_h,)-history into three parallel scales:
  long  : patch_size=6, num_patches=2   — captures within-hour phase
  mid   : patch_size=4, num_patches=3   — 20-min blocks
  short : patch_size=1, num_patches=12  — raw 5-min steps

Each scale produces (B, N, d). A learnable softmax-mix combines them.

Init weights:
  NormalBranch  : [0.5, 0.3, 0.2] (long-focus)
  IncidentBranch: [0.2, 0.3, 0.5] (short-focus)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


SCALES = [
    ("long",  6, 2),
    ("mid",   4, 3),
    ("short", 1, 12),
]


class MultiScalePatching(nn.Module):
    def __init__(self, c_x: int, d_t: int, hidden_dim: int,
                 t_h: int, init_weights: list[float] | tuple[float, ...]):
        super().__init__()
        for _, ps, np_ in SCALES:
            assert ps * np_ == t_h, f"scale {ps}*{np_} != t_h {t_h}"
        self.c_x = c_x
        self.d_t = d_t
        self.hidden_dim = hidden_dim
        self.t_h = t_h

        in_per_step = 2 * c_x  # flow + mask concatenated
        self.patch_projs = nn.ModuleList([
            nn.Linear(ps * in_per_step, hidden_dim) for _, ps, _ in SCALES
        ])
        self.time_projs = nn.ModuleList([
            nn.Linear(d_t, hidden_dim) for _ in SCALES
        ])
        self.pos_encs = nn.ParameterList([
            nn.Parameter(torch.zeros(np_, hidden_dim)) for _, _, np_ in SCALES
        ])
        for pe in self.pos_encs:
            nn.init.trunc_normal_(pe, std=0.02)

        assert len(init_weights) == len(SCALES)
        s = float(sum(init_weights))
        normalized = [w / s for w in init_weights]
        init_logits = torch.tensor([math.log(w) for w in normalized],
                                   dtype=torch.float32)
        self.scale_logits = nn.Parameter(init_logits)

    def mixing_weights(self) -> torch.Tensor:
        """Current α (size 3) — useful for monitoring."""
        return torch.softmax(self.scale_logits, dim=0)

    def forward(self, x_hist_with_mask: torch.Tensor,
                time_enc: torch.Tensor) -> torch.Tensor:
        """x_hist_with_mask: (B, N, T_h, 2*C_x); time_enc: (B, T_h, d_t).
        Output: (B, N, d) — scale-mixed temporal embedding per node.
        """
        B, N, T_h, _ = x_hist_with_mask.shape
        embeds: list[torch.Tensor] = []
        for k, (_, ps, P) in enumerate(SCALES):
            # patch the time axis
            x_p = x_hist_with_mask.reshape(B, N, P, ps * 2 * self.c_x)
            e_x = self.patch_projs[k](x_p)                                  # (B, N, P, d)
            # per-patch time_enc (mean within patch)
            t_p = time_enc.reshape(B, P, ps, self.d_t).mean(dim=2)          # (B, P, d_t)
            e_t = self.time_projs[k](t_p)                                   # (B, P, d)
            e_k = e_x + e_t.unsqueeze(1) + self.pos_encs[k].view(1, 1, P, self.hidden_dim)
            embeds.append(e_k.mean(dim=2))                                  # (B, N, d)

        α = torch.softmax(self.scale_logits, dim=0)                         # (3,)
        out = torch.zeros_like(embeds[0])
        for a, e in zip(α, embeds):
            out = out + a * e
        return out
