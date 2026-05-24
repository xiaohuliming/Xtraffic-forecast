"""Three-component DIST-Net loss per design doc §9.

L_total = α · L_main  +  β · L_normal  +  γ · L_incident
  L_main     — raw-flow MAE on all valid (N, T_p, C_x) positions
  L_normal   — pred_normal_final aligned to learned baseline (down-weighted on
               affected nodes since the soft label has noise there)
  L_incident — raw-flow MAE restricted to affected nodes (focus on event scope)

Default weights from §9.2:
  α = 1.0, β = 0.3, γ = 0.5, λ_aff = 0.3
"""
from __future__ import annotations

import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """All inputs broadcastable to (B, N, T_p, C_x). mask is bool/float.

    weight (optional): same broadcast shape, multiplies the absolute error
    BEFORE the mask sum. Returns scalar mean over valid positions
    (sum |err| · w · m) / (sum m), so weighting doesn't shrink the denominator.
    """
    diff = (pred - target).abs()
    if weight is not None:
        diff = diff * weight
    mask_f = mask.float()
    num = (diff * mask_f).sum()
    denom = mask_f.sum().clamp(min=1.0)
    return num / denom


def compute_losses(pred: torch.Tensor, pred_normal: torch.Tensor,
                   y_true: torch.Tensor, y_baseline: torch.Tensor,
                   y_mask: torch.Tensor, affected_mask: torch.Tensor,
                   alpha: float = 1.0, beta: float = 0.3, gamma: float = 0.5,
                   lam_aff: float = 0.3) -> dict[str, torch.Tensor]:
    """All tensor inputs shaped (B, N, T_p, C_x) except affected_mask (B, N).

    Returns dict with: L_main, L_normal, L_incident, L_total.
    """
    # L_main — all valid positions
    L_main = masked_mae(pred, y_true, y_mask)

    # L_normal — down-weight affected nodes (their baseline soft label is noisy)
    # weight: (B, N, 1, 1) broadcast over (T_p, C_x)
    aff_node = affected_mask.unsqueeze(-1).unsqueeze(-1).float()   # (B, N, 1, 1)
    weight = torch.where(aff_node > 0,
                         torch.full_like(aff_node, lam_aff),
                         torch.ones_like(aff_node))
    L_normal = masked_mae(pred_normal, y_baseline, y_mask, weight=weight)

    # L_incident — restricted to affected nodes
    inc_mask = y_mask & affected_mask.unsqueeze(-1).unsqueeze(-1)
    L_incident = masked_mae(pred, y_true, inc_mask)

    L_total = alpha * L_main + beta * L_normal + gamma * L_incident
    return {
        "L_main": L_main,
        "L_normal": L_normal,
        "L_incident": L_incident,
        "L_total": L_total,
    }
