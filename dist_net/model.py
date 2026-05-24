"""DIST-Net — top-level model orchestrating the two branches + cross-attn + gate.

Architecture overview (full version in docs/dist_net_design.md):

  x_hist + mask                        incident_feat + mask
       │                                       │
       │                                       │
  ┌────▼─────────┐                       ┌─────▼────────┐
  │ NormalBranch │                       │IncidentBranch│
  └────┬─────────┘                       └────┬─────────┘
       │  z_normal, pred_normal_init           │  z_incident, delta_pred
       └─────────────────┬─────────────────────┘
                         │
                ┌────────▼────────────┐
                │ BidirectionalCross  │  (uses edge_index from sparse graph)
                │      Attention      │
                └────────┬────────────┘
                         │  z_normal_updated, z_incident_updated
                  ┌──────▼──────┐
                  │AffectedGate │ → g (B, N, T_p)
                  └──────┬──────┘
                         │
                  pred = pred_normal_final + g · delta_pred

Current implementation status: all submodules are STUBS that produce correct
shapes with trainable parameters. Real implementations land per design doc
§5-§8 as we replace each stub.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cross_attn import BidirectionalCrossAttention
from .gate import AffectedGate
from .incident_branch import IncidentBranch
from .normal_branch import NormalBranch


@dataclass
class DISTNetConfig:
    c_x: int = 3
    c_meta: int = 8
    c_e: int = 13
    d_t: int = 5
    n_regions: int = 3
    hidden_dim: int = 64
    t_h: int = 12
    t_p: int = 12
    single_branch: bool = False


class DISTNet(nn.Module):
    def __init__(self, cfg: DISTNetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.normal = NormalBranch(
            c_x=cfg.c_x, c_meta=cfg.c_meta, d_t=cfg.d_t,
            n_regions=cfg.n_regions, hidden_dim=d,
            t_h=cfg.t_h, t_p=cfg.t_p,
        )
        if not cfg.single_branch:
            self.incident = IncidentBranch(
                c_x=cfg.c_x, c_meta=cfg.c_meta, c_e=cfg.c_e,
                n_regions=cfg.n_regions, hidden_dim=d,
                t_h=cfg.t_h, t_p=cfg.t_p,
            )
            self.cross_attn = BidirectionalCrossAttention(hidden_dim=d)
            self.gate = AffectedGate(hidden_dim=d, t_p=cfg.t_p)
        self.pred_normal_final_head = nn.Linear(d, cfg.t_p * cfg.c_x)

    def forward(self, batch: dict[str, torch.Tensor],
                edge_index: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Run full forward.

        batch keys (from dist_net.data collate):
          x_hist, x_hist_mask, time_enc, static_meta, region_code,
          incident_feat, incident_mask, affected_mask, y_true, y_baseline, y_mask

        edge_index: (2, E) int64, region-specific sparse graph. Required when the
        cross_attn module is replaced with the real sparse version; the stub
        ignores it.
        """
        B, N, T_h, C_x = batch["x_hist"].shape

        z_normal, _pred_normal_init = self.normal(
            x_hist=batch["x_hist"],
            x_hist_mask=batch["x_hist_mask"],
            time_enc=batch["time_enc"],
            static_meta=batch["static_meta"],
            region_code=batch["region_code"],
            edge_index=edge_index,
        )

        if self.cfg.single_branch:
            pred_normal_final = self.pred_normal_final_head(z_normal).view(
                B, N, self.cfg.t_p, self.cfg.c_x
            )
            delta_pred = torch.zeros_like(pred_normal_final)
            g = torch.zeros(B, N, self.cfg.t_p, device=z_normal.device, dtype=z_normal.dtype)
            return {
                "pred": pred_normal_final,
                "pred_normal_final": pred_normal_final,
                "delta_pred": delta_pred,
                "g": g,
                "z_normal": z_normal,
                "z_incident": torch.zeros_like(z_normal),
            }

        z_incident, delta_pred = self.incident(
            x_hist=batch["x_hist"],
            x_hist_mask=batch["x_hist_mask"],
            incident_feat=batch["incident_feat"],
            incident_mask=batch["incident_mask"],
            static_meta=batch["static_meta"],
            region_code=batch["region_code"],
            time_enc=batch["time_enc"],
            edge_index=edge_index,
            rel_feat=batch.get("rel_feat"),
        )

        z_normal_u, z_incident_u = self.cross_attn(z_normal, z_incident, edge_index)

        g = self.gate(z_normal_u, z_incident_u)

        pred_normal_final = self.pred_normal_final_head(z_normal_u).view(
            B, N, self.cfg.t_p, self.cfg.c_x
        )
        pred = pred_normal_final + g.unsqueeze(-1) * delta_pred

        return {
            "pred": pred,
            "pred_normal_final": pred_normal_final,
            "delta_pred": delta_pred,
            "g": g,
            "z_normal": z_normal_u,
            "z_incident": z_incident_u,
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Per-submodule parameter counts for inspecting model size."""
    out = {"_total": 0}
    for name, sub in model.named_children():
        n = sum(p.numel() for p in sub.parameters() if p.requires_grad)
        out[name] = n
        out["_total"] += n
    return out
