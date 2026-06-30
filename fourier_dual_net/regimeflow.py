"""RegimeFlowNet, minimal rung RFN-v0.

Tests the foundational hypothesis of the regime-transition direction: do speed and
occupancy residuals, as INPUT, help predict flow deviations beyond flow-only? Node-local
multi-scale causal TCN over the de-seasonalized residual; no graph, no regime gate yet.
Output is the de-seasonalized flow: q_hat = baseline_q + standardized-deviation * sd_q.
Numpy-free at import so it can be py_compiled locally; torch only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwise(nn.Module):
    """Causal depthwise 1d conv: left-pad by (k-1)*dilation, no future leakage."""

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation, groups=ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B,C,T)
        return self.conv(F.pad(x, (self.pad, 0)))


class MultiScaleBlock(nn.Module):
    """Parallel causal depthwise convs at several dilations, pointwise-mixed, residual."""

    def __init__(self, nhid: int, scales=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList([CausalDepthwise(nhid, 3, d) for d in scales])
        self.point = nn.Conv1d(nhid * len(scales), nhid, 1)
        self.norm = nn.BatchNorm1d(nhid)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B,C,T)
        h = torch.cat([b(x) for b in self.branches], dim=1)
        h = self.drop(F.gelu(self.norm(self.point(h))))
        return x + h


class RFNv0(nn.Module):
    def __init__(self, num_nodes, T_h, T_p, channels="qov",
                 nhid=64, nblocks=3, scales=(1, 2, 4), dropout=0.1):
        super().__init__()
        assert channels in ("qov", "flow")
        self.channels = channels
        self.in_ch = 6 if channels == "qov" else 2
        self.T_p = T_p
        self.proj = nn.Conv1d(self.in_ch, nhid, 1)
        self.blocks = nn.ModuleList([MultiScaleBlock(nhid, scales, dropout) for _ in range(nblocks)])
        self.head = nn.Sequential(nn.Linear(nhid, nhid), nn.GELU(), nn.Linear(nhid, T_p))
        self.register_buffer("sd", torch.ones(3))              # per-channel residual std q,o,v

    def forward(self, x_hist, x_baseline, y_baseline):
        # x_hist, x_baseline (B,N,T_h,3); y_baseline (B,N,T_p,3)
        B, N, T_h, _ = x_hist.shape
        res = (x_hist - x_baseline) / self.sd.view(1, 1, 1, 3)         # standardized residual
        delta = torch.zeros_like(res)
        delta[:, :, 1:] = res[:, :, 1:] - res[:, :, :-1]              # first difference
        if self.channels == "qov":
            feat = torch.cat([res, delta], dim=-1)                    # (B,N,T_h,6)
        else:
            feat = torch.cat([res[..., 0:1], delta[..., 0:1]], dim=-1)  # (B,N,T_h,2)
        x = feat.permute(0, 1, 3, 2).reshape(B * N, self.in_ch, T_h)
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        dq = self.head(h[:, :, -1]).reshape(B, N, self.T_p)           # standardized flow deviation
        return y_baseline[..., 0] + dq * self.sd[0]                   # (B,N,T_p) flow
