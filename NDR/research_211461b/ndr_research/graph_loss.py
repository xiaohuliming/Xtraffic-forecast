"""Raw-flow NDR supervision with runtime scales and exact batch reductions."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

TEACHERS = ("original", "projected")


def _valid_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if target.ndim != 3 or target.shape != mask.shape or target.device != mask.device:
        raise ValueError("target and mask must match (B, N, H) on one device")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("target mask must contain only zero or one")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("loss batch has no valid target observations")
    if not torch.isfinite(target[valid]).all():
        raise FloatingPointError("observed targets must be finite")
    return valid


def graph_node_scale(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Canonical (B, N, 1) scale, deliberately absent from model state."""
    batch, nodes, _ = target.shape
    if scale.device != target.device or scale.dtype != target.dtype:
        raise ValueError("node scales must share target dtype and device")
    if scale.shape == (nodes,):
        scale = scale.view(1, nodes, 1).expand(batch, -1, -1)
    elif scale.shape == (batch, nodes):
        scale = scale.unsqueeze(-1)
    elif scale.shape == (1, nodes, 1):
        scale = scale.expand(batch, -1, -1)
    elif scale.shape != (batch, nodes, 1):
        raise ValueError("runtime node scales must have shape (N,), (B,N), or (B,N,1)")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("runtime node scales must be finite and positive")
    return scale


def graph_oracle_teacher(
    target: torch.Tensor, mask: torch.Tensor, *, baseline: torch.Tensor,
    persistence: torch.Tensor, node_scale: torch.Tensor, teacher: str = "original",
) -> dict[str, torch.Tensor]:
    """The original raw-anchor teacher, or its explicit projected ablation."""
    if teacher not in TEACHERS:
        raise ValueError(f"unsupported oracle teacher: {teacher}")
    valid = _valid_target(target, mask)
    if baseline.shape != target.shape or baseline.device != target.device:
        raise ValueError("baseline must match target shape and device")
    if persistence.shape not in (target.shape, (*target.shape[:2], 1)):
        raise ValueError("persistence must match the target nodes")
    if persistence.device != target.device:
        raise ValueError("persistence must share target device")
    if not torch.isfinite(baseline).all() or not torch.isfinite(persistence).all():
        raise FloatingPointError("raw oracle anchors must be finite")
    scale = graph_node_scale(node_scale, target)
    safe_target = torch.where(valid, target, baseline)
    baseline_error = (safe_target - baseline).abs()
    persistence_error = (safe_target - persistence).abs()
    gap = baseline - persistence
    confidence = (
        (baseline_error - persistence_error).abs()
        / (baseline_error + persistence_error + 1e-6)
        * (gap.abs() / scale).clamp(0, 1)
    )
    if teacher == "original":
        weight = persistence_error / (baseline_error + persistence_error + 1e-6)
    else:
        distinct = gap.abs() > 1e-6 * scale
        denominator = torch.where(distinct, gap, torch.ones_like(gap))
        weight = ((safe_target - persistence) / denominator).clamp(0, 1)
        weight = torch.where(distinct, weight, torch.full_like(weight, 0.5))
        confidence = torch.where(distinct, confidence, torch.zeros_like(confidence))
    return {
        "weight": weight.detach(),
        "confidence": torch.where(valid, confidence, torch.zeros_like(confidence)).detach(),
        "valid": valid,
        "node_scale": scale,
    }


def build_graph_loss_normalizers(
    target: torch.Tensor, mask: torch.Tensor, *, baseline: torch.Tensor,
    persistence: torch.Tensor, node_scale: torch.Tensor, teacher: str = "original",
) -> dict[str, torch.Tensor]:
    """Freeze all coupled constants on the full effective batch."""
    oracle = graph_oracle_teacher(target, mask, baseline=baseline, persistence=persistence,
                                  node_scale=node_scale, teacher=teacher)
    count = oracle["valid"].sum().to(target.dtype)
    return {
        "valid_count": count.detach(),
        "gate_confidence_sum": oracle["confidence"].sum().detach(),
        "mean_gate_target": ((oracle["weight"] * oracle["valid"]).sum() / (count + 1e-6)).detach(),
    }


def compute_graph_loss(
    model: Any, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, teacher: str = "original", lambda_gate: float = 1.0,
    lambda_specialization: float = 0.3,
    normalizers: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = _valid_target(target, mask)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("prediction must be finite everywhere and match target")
    if lambda_gate < 0 or lambda_specialization < 0:
        raise ValueError("oracle and specialization coefficients must be nonnegative")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    count = valid.sum().to(target.dtype)
    denominator = count if normalizers is None else normalizers["valid_count"]
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("full effective batch denominator must be finite and positive")
    primary = torch.where(valid, (prediction - safe_target).abs(), 0.0).sum() / denominator
    parts = {"mae": float(primary.detach())}
    if not model.training:
        return primary, parts
    aux = getattr(model, "aux", None)
    required = {"g", "B", "P", "yn", "yd", "node_scale"}
    if not isinstance(aux, Mapping) or not required.issubset(aux):
        raise ValueError("node-inductive NDR auxiliary anchors and runtime scales are missing")
    if any(not torch.isfinite(aux[key]).all() for key in required):
        raise FloatingPointError("node-inductive NDR auxiliary values are non-finite")
    oracle = graph_oracle_teacher(target, valid, baseline=aux["B"], persistence=aux["P"],
                                  node_scale=aux["node_scale"], teacher=teacher)
    weight, confidence, scale = oracle["weight"], oracle["confidence"], oracle["node_scale"]
    confidence_sum = confidence.sum() if normalizers is None else normalizers["gate_confidence_sum"]
    mean_target = (
        (weight * valid).sum() / (count + 1e-6)
        if normalizers is None else normalizers["mean_gate_target"]
    )
    gate_huber = F.smooth_l1_loss(aux["g"], weight, reduction="none", beta=0.25)
    gate_loss = (confidence * scale * gate_huber).sum() / (confidence_sum + 1e-6)
    normal_weight = weight / (mean_target + 1e-6)
    disturbance_weight = (1 - weight) / (1 - mean_target + 1e-6)
    specialization = torch.where(
        valid,
        normal_weight * (aux["yn"] - safe_target).abs()
        + disturbance_weight * (aux["yd"] - safe_target).abs(),
        0.0,
    ).sum() / (denominator + 1e-6)
    loss = primary + lambda_gate * gate_loss + lambda_specialization * specialization
    if not torch.isfinite(loss):
        raise FloatingPointError("graph NDR objective is non-finite")
    parts.update(gate=float(gate_loss.detach()), spec=float(specialization.detach()))
    return loss, parts
