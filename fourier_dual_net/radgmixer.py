"""RAD-GMixer: Regime-Adaptive Decomposition Graph Mixer.

A lightweight, label-free, GWN-free forecaster that internalizes the v0c finding
(normal nodes trust the periodic baseline, anomalous nodes anchor on current state) as a
learnable per-node per-horizon decomposition layer, then predicts only the residual with a
multi-scale temporal mixer and (optionally) constrained graph mixers.

Rungs (flag-driven):
  RAD-v0  regime-adaptive anchor + multi-scale temporal mixer + residual decoder (no graph)
  RAD-v1  + local directed road graph mixer (upstream/downstream)   [local_graph=True]
  RAD-v2  + low-rank landmark graph mixer                           [landmark=True]

Prediction: y_hat[n,h] = A[n,h] + sd_node[n] * rho[n,h] * R[n,h]
  A = alpha * periodic_baseline + (1-alpha) * persistence   (regime-adaptive anchor)
  alpha[n,h] = sigmoid(a_h - softplus(b_h)*d_n - softplus(c_h)*u_n)   (monotone gate)
torch only; no numpy at import so it py_compiles locally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALES = (1, 2, 3, 6, 12)   # window sizes; each must divide T_h=12


class RegimeAnchor(nn.Module):
    """Per-node per-horizon blend of periodic baseline and clipped-persistence anchor."""

    def __init__(self, T_p: int):
        super().__init__()
        self.T_p = T_p
        self.a = nn.Parameter(torch.full((T_p,), 2.0))     # bias -> alpha~0.88 init
        self.b = nn.Parameter(torch.zeros(T_p))            # softplus(b) weights deviation d
        self.c = nn.Parameter(torch.zeros(T_p))            # softplus(c) weights slope u

    def forward(self, flow, y_baseline, d, u, sd_node):
        # flow (B,N,T_h); y_baseline (B,N,T_p); d,u (B,N); sd_node (N,)
        B, N, _ = flow.shape
        a = self.a.view(1, 1, -1)
        bb = F.softplus(self.b).view(1, 1, -1)
        cc = F.softplus(self.c).view(1, 1, -1)
        alpha = torch.sigmoid(a - bb * d.unsqueeze(-1) - cc * u.unsqueeze(-1))   # (B,N,T_p)
        last = flow[:, :, -1:]                                                   # (B,N,1)
        slope = flow[:, :, -1] - flow[:, :, -2]                                  # (B,N)
        m = sd_node.view(1, N)                                                   # clip slope to +-sd
        slope = torch.clamp(slope, -m, m).unsqueeze(-1)                          # (B,N,1)
        steps = torch.arange(1, self.T_p + 1, device=flow.device).view(1, 1, -1)
        persist = last + steps * slope                                          # (B,N,T_p)
        anchor = alpha * y_baseline + (1.0 - alpha) * persist
        return anchor, alpha


class MultiScaleTemporalMixer(nn.Module):
    """Per-node multi-scale direct temporal features, regime-gated sum -> (B,N,hidden)."""

    def __init__(self, T_h: int, hidden: int, scales=SCALES):
        super().__init__()
        self.scales = [s for s in scales if T_h % s == 0]
        self.proj = nn.ModuleList([nn.Linear(T_h // s, hidden) for s in self.scales])
        self.gate = nn.Sequential(nn.Linear(2, hidden), nn.GELU(),
                                  nn.Linear(hidden, len(self.scales)))

    def forward(self, r, d, u):
        # r (B,N,T_h); d,u (B,N)
        B, N, T_h = r.shape
        feats = []
        for s, proj in zip(self.scales, self.proj):
            pooled = r.reshape(B, N, T_h // s, s).mean(-1) if s > 1 else r       # (B,N,T_h/s)
            feats.append(proj(pooled))                                          # (B,N,hidden)
        feats = torch.stack(feats, dim=-2)                                      # (B,N,S,hidden)
        g = F.softmax(self.gate(torch.stack([d, u], dim=-1)), dim=-1)           # (B,N,S)
        return (g.unsqueeze(-1) * feats).sum(-2)                                # (B,N,hidden)


class LocalRoadMixer(nn.Module):
    """Directed local road mix: self + upstream + downstream sparse propagation. RAD-v1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden)
        self.w_up = nn.Linear(hidden, hidden)
        self.w_down = nn.Linear(hidden, hidden)

    def forward(self, h, A_up, A_down):
        # h (B,N,hidden); A_up,A_down (N,N) row-normalized directed adjacency
        up = torch.einsum("nm,bmc->bnc", A_up, h)
        down = torch.einsum("nm,bmc->bnc", A_down, h)
        return F.gelu(self.w_self(h) + self.w_up(up) + self.w_down(down))


class LandmarkMixer(nn.Module):
    """Low-rank global mix through M learned landmarks: O(NM). RAD-v2."""

    def __init__(self, num_nodes: int, hidden: int, n_land: int, emb_dim: int = 16):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.05)
        self.to_land = nn.Linear(emb_dim, n_land)
        self.w_g = nn.Linear(hidden, hidden)

    def forward(self, h):
        C = F.softmax(self.to_land(self.emb), dim=-1)          # (N,M)
        g = torch.einsum("nm,bnc->bmc", C, h) / (C.sum(0).view(1, -1, 1) + 1e-6)  # (B,M,hidden)
        back = torch.einsum("nm,bmc->bnc", C, g)               # (B,N,hidden)
        return h + self.w_g(back)


class RADGMixer(nn.Module):
    def __init__(self, num_nodes, T_h, T_p, hidden=48, local_graph=False, landmark=False,
                 n_land=16, dropout=0.1):
        super().__init__()
        self.T_p = T_p
        self.local_graph = bool(local_graph)
        self.landmark = bool(landmark)
        self.register_buffer("sd_node", torch.ones(num_nodes))       # per-node residual std
        self.anchor = RegimeAnchor(T_p)
        self.tmixer = MultiScaleTemporalMixer(T_h, hidden)
        self.local = LocalRoadMixer(hidden) if self.local_graph else None
        self.land = LandmarkMixer(num_nodes, hidden, n_land) if self.landmark else None
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, T_p))
        self.rho_gate = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, T_p))
        self.last_alpha = None                                        # diagnostic (B,N,T_p)

    def forward(self, x_hist, x_baseline, y_baseline, A_up=None, A_down=None):
        flow = x_hist[..., 0]                                         # (B,N,T_h)
        base_h = x_baseline[..., 0]
        y_base = y_baseline[..., 0]
        sd = self.sd_node.view(1, -1, 1)
        r = (flow - base_h) / sd                                     # (B,N,T_h) per-node std
        d = r.abs().mean(-1)                                         # (B,N) deviation strength
        u = (r[:, :, -1] - r[:, :, -2]).abs()                        # (B,N) recent slope

        anchor, alpha = self.anchor(flow, y_base, d, u, self.sd_node)
        self.last_alpha = alpha.detach()

        h = self.tmixer(r, d, u)                                     # (B,N,hidden)
        if self.local is not None:
            h = self.local(h, A_up, A_down)
        if self.land is not None:
            h = self.land(h)
        h = self.drop(h)
        R = self.decoder(h)                                          # (B,N,T_p) standardized residual
        rho = 0.5 + torch.sigmoid(self.rho_gate(torch.stack([d, u], dim=-1)))  # (B,N,T_p) in (0.5,1.5)
        return anchor + self.sd_node.view(1, -1, 1) * (rho * R)
