"""RGDN: de-seasonalized residual-guided dual-branch forecaster.

Decomposition reuses the cache climatology (baseline added back as a known skeleton);
the network predicts only the standardized deviation. Variants are flag-driven:
  deseason: subtract baseline & predict deviation (else z-scored raw flow)
  dual:     two branches (else single GWN baseline)
  main_gcn: main branch uses graph conv (else node-local TCN)
  inject:   feed neighbor-residual summary into the main branch
  adaptive: single-branch only; per-node alpha from recent residual magnitude blends
            the future climatology anchor with persistence (route B). alpha->1 on
            normal nodes recovers v0b exactly; alpha->0 on anomalous nodes anchors on
            the current level instead of the wrong periodic baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines" / "GraphWaveNet"))
from model import gwnet  # noqa: E402


def _make_gwn(num_nodes, supports, device, in_dim, T_p, nhid, gcn_bool, dropout):
    return gwnet(device=device, num_nodes=num_nodes, dropout=dropout, supports=supports,
                 gcn_bool=gcn_bool, addaptadj=gcn_bool, in_dim=in_dim, out_dim=T_p,
                 residual_channels=nhid, dilation_channels=nhid,
                 skip_channels=nhid * 8, end_channels=nhid * 16, blocks=4, layers=2)


class InjectionGraphConv(nn.Module):
    """One adaptive-adjacency diffusion step over residuals -> per-node neighbor summary.

    res (B,N,T) standardized flow residual -> (B, c_out, N, T) in gwnet channel layout.
    """

    def __init__(self, num_nodes: int, c_out: int = 4, d_emb: int = 10):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.05)
        self.e2 = nn.Parameter(torch.randn(d_emb, num_nodes) * 0.05)
        self.proj = nn.Conv2d(1, c_out, kernel_size=(1, 1))

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)   # (N,N) rows sum to 1
        summary = torch.einsum("nm,bmt->bnt", adp, res)             # (B,N,T) neighbor-weighted res
        return self.proj(summary.unsqueeze(1))                      # (B,c_out,N,T)


class AdaptiveAlpha(nn.Module):
    """Per-node anchor weight from recent residual magnitude. Physically grounded,
    near-degenerate-proof: only (r0, tau) learnable, and alpha is pinned to 1 below r0.

    alpha = exp(-relu(r - r0) / tau), r = mean_t |flow-baseline| / sd_res  (B,N).
    relu pins alpha=1 exactly for normal nodes (r<=r0), so the unaffected gain is
    untouched; only genuinely anomalous nodes (r>r0) drop alpha toward persistence.
    """

    def __init__(self, r0_init: float = 1.5, tau_init: float = 1.0):
        super().__init__()
        self.r0 = nn.Parameter(torch.tensor(float(r0_init)))
        self.raw_tau = nn.Parameter(torch.tensor(float(tau_init)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.raw_tau) + 1e-3
        return torch.exp(-F.relu(r - self.r0) / tau)              # (B,N) in (0,1]


class RGDN(nn.Module):
    def __init__(self, num_nodes, supports, T_h, T_p, device=None,
                 deseason=True, dual=True, main_gcn=False, inject=True, inject_gate=False,
                 adaptive=False, const_alpha=False, qov_input=False,
                 nhid_single=32, nhid_main=26, nhid_res=22, c_inject=4, dropout=0.3):
        super().__init__()
        self.deseason = bool(deseason)
        self.dual = bool(dual)
        self.inject = bool(inject) and self.dual
        self.use_inject_gate = bool(inject_gate) and self.inject
        self.adaptive = bool(adaptive) and self.deseason and not self.dual
        self.const_alpha = bool(const_alpha) and self.deseason and not self.dual and not self.adaptive
        self.qov_input = bool(qov_input) and self.deseason and not self.dual
        self.T_p = T_p
        self.register_buffer("sd_res", torch.tensor(1.0))
        self.register_buffer("sd_qov", torch.ones(3))             # per-channel q/o/v residual std
        self.register_buffer("flow_mu", torch.tensor(0.0))
        self.register_buffer("flow_sd", torch.tensor(1.0))
        self.last_alpha = None                                    # diagnostic: (B,N) detached

        if not self.dual:
            in_dim = 3 if self.qov_input else 1
            self.single = _make_gwn(num_nodes, supports, device, in_dim, T_p, nhid_single, True, dropout)
            self.alpha_mod = AdaptiveAlpha() if self.adaptive else None
            if self.const_alpha:
                self.alpha_logit = nn.Parameter(torch.tensor(2.2))   # sigmoid ~ 0.90 init
            return

        self.inject_mod = InjectionGraphConv(num_nodes, c_out=c_inject) if self.inject else None
        if self.use_inject_gate:
            self.inject_gate = nn.Parameter(torch.ones(1))         # learnable injection strength
        main_in = 1 + 2 + (c_inject if self.inject else 0)          # res + tod/dow + injection
        self.main_branch = _make_gwn(num_nodes, supports, device, main_in, T_p, nhid_main,
                                     main_gcn, dropout)
        self.res_branch = _make_gwn(num_nodes, supports, device, 1, T_p, nhid_res, True, dropout)

    @staticmethod
    def _to_gwnet(x):   # (B,N,T,C)->(B,C,N,T)
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _from_gwnet(out):   # (B,T_p,N,1)->(B,N,T_p)
        return out.squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x_hist, x_baseline, y_baseline, time_feat):
        flow = x_hist[..., 0]                                       # (B,N,T_h)
        if self.deseason:
            sig = (flow - x_baseline[..., 0]) / self.sd_res         # standardized residual
        else:
            sig = (flow - self.flow_mu) / self.flow_sd

        if not self.dual:
            if self.qov_input:
                sig_in = (x_hist - x_baseline) / self.sd_qov.view(1, 1, 1, 3)   # (B,N,T_h,3)
            else:
                sig_in = sig.unsqueeze(-1)                                      # (B,N,T_h,1)
            out = self._from_gwnet(self.single(self._to_gwnet(sig_in)))
            if self.adaptive:
                r = (flow - x_baseline[..., 0]).abs().mean(dim=2) / self.sd_res    # (B,N)
                alpha = self.alpha_mod(r)                                          # (B,N)
                self.last_alpha = alpha.detach()
                persist = flow[:, :, -1:]                                          # (B,N,1) last obs
                anchor = (alpha.unsqueeze(-1) * y_baseline[..., 0]
                          + (1.0 - alpha).unsqueeze(-1) * persist)                 # (B,N,T_p)
                return anchor + out * self.sd_res
            if self.const_alpha:
                alpha = torch.sigmoid(self.alpha_logit)                            # scalar
                self.last_alpha = alpha.detach().expand(flow.shape[0], flow.shape[1])
                persist = flow[:, :, -1:]
                anchor = alpha * y_baseline[..., 0] + (1.0 - alpha) * persist      # (B,N,T_p)
                return anchor + out * self.sd_res
            return self._reseason(out, y_baseline)

        B, N, T_h = sig.shape
        y_res = self._from_gwnet(self.res_branch(self._to_gwnet(sig.unsqueeze(-1))))
        tf = time_feat.unsqueeze(1).expand(B, N, T_h, 2)
        feats = [sig.unsqueeze(-1), tf]
        if self.inject:
            inj = self.inject_mod(sig)
            if self.use_inject_gate:
                inj = inj * self.inject_gate
            inj = inj.permute(0, 2, 3, 1)                           # (B,N,T_h,c_inject)
            feats.append(inj)
        y_main = self._from_gwnet(self.main_branch(self._to_gwnet(torch.cat(feats, dim=-1))))
        return self._reseason(y_main + y_res, y_baseline)

    def _reseason(self, out, y_baseline):
        if self.deseason:
            return y_baseline[..., 0] + out * self.sd_res
        return out * self.flow_sd + self.flow_mu
