"""Isolated NDR controls and mask/signed-recovery candidate; no training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from scripts.train_staeformer_deseason import (
    STAEformerDeseason,
    VARIANTS,
    build_ndr_loss_normalizers,
    compute_loss,
)

from .routers import MaskSignedGate, persistence_anchor, reliable_history


RESEARCH_VARIANTS = ("baseline", "projected_oracle", "sanitized_control", "mask_signed")


@dataclass(frozen=True)
class OracleTeacher:
    target: torch.Tensor
    confidence: torch.Tensor


def _checked_target_mask(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if target.shape != mask.shape or target.ndim != 3:
        raise ValueError("target and mask must have matching (B, N, H) shapes")
    if target.device != mask.device:
        raise ValueError("target and mask must share a device")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("target mask must contain only zero or one")
    valid = mask.bool()
    if not valid.any():
        raise ValueError("loss batch has no valid target observations")
    if not torch.isfinite(target[valid]).all():
        raise FloatingPointError("valid target observations must be finite")
    return valid


def projected_oracle_teacher(
    target: torch.Tensor,
    baseline: torch.Tensor,
    persistence: torch.Tensor,
    target_mask: torch.Tensor,
    node_scale: torch.Tensor,
    *,
    ambiguity_tolerance: float = 1e-6,
) -> OracleTeacher:
    """Exact pointwise MAE minimizer over convex mixtures of two raw anchors."""
    valid = _checked_target_mask(target, target_mask)
    if baseline.shape != target.shape:
        raise ValueError("teacher baseline must match target shape")
    if persistence.shape not in (target.shape, (*target.shape[:2], 1)):
        raise ValueError("teacher persistence must match nodes and horizon")
    if node_scale.shape != (target.shape[1],) or not (node_scale > 0).all():
        raise ValueError("teacher requires positive per-node residual scales")
    if not torch.isfinite(node_scale).all() or not torch.isfinite(baseline).all():
        raise FloatingPointError("teacher scales and baseline must be finite")
    if not torch.isfinite(persistence).all():
        raise FloatingPointError("teacher persistence must be finite")
    if not 0 < ambiguity_tolerance < 1:
        raise ValueError("anchor ambiguity tolerance must lie between zero and one")
    scale = node_scale.view(1, -1, 1)
    safe_target = torch.where(valid, target, baseline)
    gap = baseline - persistence
    distinct = gap.abs() > ambiguity_tolerance * scale
    denominator = torch.where(distinct, gap, torch.ones_like(gap))
    weight = ((safe_target - persistence) / denominator).clamp(0, 1)
    weight = torch.where(distinct & valid, weight, torch.full_like(weight, 0.5))
    baseline_error = (safe_target - baseline).abs()
    persistence_error = (safe_target - persistence).abs()
    confidence = (
        (baseline_error - persistence_error).abs()
        / (baseline_error + persistence_error + 1e-6)
        * (gap.abs() / scale).clamp(0, 1)
    )
    confidence = torch.where(distinct & valid, confidence, torch.zeros_like(confidence))
    return OracleTeacher(weight.detach(), confidence.detach())


class SanitizedNDR(STAEformerDeseason):
    """Original routing with the same history sanitization as the candidate."""

    def __init__(self, node_count: int) -> None:
        super().__init__(node_count, 12, 12, **dict(VARIANTS["s0g"]))

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = reliable_history(flow_hist, x_baseline, history_mask)
        return super().forward(state.filled, x_baseline, y_baseline, time_feat)


class MaskSignedNDR(STAEformerDeseason):
    """Retain the original STAEformer and two heads, changing history routing only."""

    def __init__(
        self,
        node_count: int,
        *,
        hidden_dim: int = 16,
        reliable_anchor: bool = False,
        max_reference_age: int = 2,
    ) -> None:
        super().__init__(node_count, 12, 12, **dict(VARIANTS["s0g"]))
        base_state = self.gate_mod.state_dict()
        self.gate_mod = MaskSignedGate(12, hidden_dim)
        self.gate_mod.base.load_state_dict(base_state)
        self.reliable_anchor = bool(reliable_anchor)
        if type(max_reference_age) is not int or not 0 <= max_reference_age < 12:
            raise ValueError("reference maximum age must be an integer in [0, 11]")
        self.max_reference_age = max_reference_age

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        state = reliable_history(flow_hist, x_baseline, history_mask)
        batch, nodes, steps = flow_hist.shape
        if steps != 12 or nodes != self.sd_node.numel():
            raise ValueError("history does not match the fitted NDR node count")
        if time_feat.shape != (batch, steps, 2):
            raise ValueError("time features must have shape (B, 12, 2)")
        signal = (state.filled - x_baseline) / self.sd_res
        tod = time_feat[..., 0].unsqueeze(-1).expand(batch, steps, nodes)
        dow = (time_feat[..., 1] * 7.0).round().clamp(0, 6).unsqueeze(-1).expand_as(tod)
        inputs = torch.stack((signal.permute(0, 2, 1), tod, dow), dim=-1)
        raw = self.backbone(inputs)
        persistence = persistence_anchor(
            state,
            x_baseline,
            reliable=self.reliable_anchor,
            max_age=self.max_reference_age,
        )
        gate = self.gate_mod(
            state, x_baseline, y_baseline, persistence, self.sd_res, self.sd_node
        )
        normal = y_baseline + raw[..., 0].permute(0, 2, 1) * self.sd_res
        disturbance = persistence + raw[..., 1].permute(0, 2, 1) * self.sd_res
        self.last_alpha = gate.detach().mean(-1)
        self.last_d = signal.abs().mean(-1).detach()
        self.aux = {"g": gate, "B": y_baseline, "P": persistence, "yn": normal, "yd": disturbance}
        return gate * normal + (1.0 - gate) * disturbance


def make_research_model(
    node_count: int,
    *,
    variant: str = "baseline",
    residual_std: float = 1.0,
    residual_std_by_node: Any = None,
    flow_mean: float = 0.0,
    flow_std: float = 1.0,
    reliable_anchor: bool = False,
    max_reference_age: int = 2,
    hidden_dim: int = 16,
    device: torch.device | str = "cpu",
) -> STAEformerDeseason:
    """Baseline and projected-oracle instantiate the unmodified original s0g."""
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("node count must be positive")
    if variant not in RESEARCH_VARIANTS:
        raise ValueError(f"unsupported research variant: {variant}")
    if reliable_anchor and variant != "mask_signed":
        raise ValueError("reliable anchor is an independent mask_signed option")
    if variant == "mask_signed":
        model: STAEformerDeseason = MaskSignedNDR(
            node_count,
            hidden_dim=hidden_dim,
            reliable_anchor=reliable_anchor,
            max_reference_age=max_reference_age,
        )
    elif variant == "sanitized_control":
        model = SanitizedNDR(node_count)
    else:
        model = STAEformerDeseason(node_count, 12, 12, **dict(VARIANTS["s0g"]))
    scales = torch.as_tensor(
        residual_std_by_node if residual_std_by_node is not None else torch.ones(node_count),
        dtype=torch.float32,
    )
    numbers = torch.tensor([residual_std, flow_mean, flow_std], dtype=torch.float64)
    if not torch.isfinite(numbers).all() or residual_std <= 0 or flow_std <= 0:
        raise ValueError("fitted flow statistics must be finite with positive scales")
    if scales.shape != (node_count,) or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("fitted node residual scales must be positive with shape (N,)")
    with torch.no_grad():
        model.sd_res.fill_(residual_std)
        model.sd_node.copy_(scales)
        model.flow_mu.fill_(flow_mean)
        model.flow_sd.fill_(flow_std)
    model.research_spec = {
        "variant": variant,
        "backbone": "original_STAEformer_s0g",
        "backbone_global_normalization": "unchanged",
        "dual_residual_heads": "unchanged",
        "teacher": "projected_raw_anchor" if variant == "projected_oracle" else "original_soft_distance",
        "history_policy": "window_causal_observations_then_climatology" if variant in {"mask_signed", "sanitized_control"} else "original_causal_cache",
        "router": "mask_signed_shared_mlp_zero_initial_output" if variant == "mask_signed" else "original_horizon_monotone",
        "router_hidden_dim": hidden_dim if variant == "mask_signed" else 0,
        "reliable_anchor": bool(reliable_anchor),
        "max_reference_age": max_reference_age if reliable_anchor else None,
        "stale_reference_fallback": "current_train_climatology" if reliable_anchor else None,
        "parameter_node_independence": "router_only; original_backbone_retains_node_embeddings",
    }
    return model.to(torch.device(device))


def forward_research(
    model: STAEformerDeseason,
    flow_hist: torch.Tensor,
    x_baseline: torch.Tensor,
    y_baseline: torch.Tensor,
    time_feat: torch.Tensor,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Common forecast dispatch, deliberately exposing no target or label argument."""
    if model.research_spec["variant"] in {"sanitized_control", "mask_signed"}:
        if history_mask is None:
            raise ValueError("candidate and sanitization control require history masks")
        return model(flow_hist, x_baseline, y_baseline, time_feat, history_mask)
    return model(flow_hist, x_baseline, y_baseline, time_feat)


def research_persistence(
    model: STAEformerDeseason,
    flow_hist: torch.Tensor,
    x_baseline: torch.Tensor,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The actual disturbance anchor, also used for full-batch loss constants."""
    variant = model.research_spec["variant"]
    if variant not in {"sanitized_control", "mask_signed"}:
        return flow_hist[..., -1:]
    if history_mask is None:
        raise ValueError("sanitized disturbance anchors require history masks")
    state = reliable_history(flow_hist, x_baseline, history_mask)
    return persistence_anchor(
        state,
        x_baseline,
        reliable=bool(model.research_spec["reliable_anchor"]),
        max_age=model.research_spec["max_reference_age"] or 0,
    )


def research_factory_kwargs(model: STAEformerDeseason) -> dict[str, Any]:
    """JSON-safe recipe sufficient to rebuild a checkpoint before loading its state."""
    spec = model.research_spec
    return {
        "node_count": int(model.sd_node.numel()),
        "variant": spec["variant"],
        "residual_std": float(model.sd_res.detach().cpu()),
        "residual_std_by_node": model.sd_node.detach().cpu().tolist(),
        "flow_mean": float(model.flow_mu.detach().cpu()),
        "flow_std": float(model.flow_sd.detach().cpu()),
        "reliable_anchor": bool(spec["reliable_anchor"]),
        "max_reference_age": spec["max_reference_age"] if spec["reliable_anchor"] else 2,
        "hidden_dim": spec["router_hidden_dim"] or 16,
    }


def build_research_loss_normalizers(
    model: STAEformerDeseason,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    baseline: torch.Tensor,
    persistence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute full-effective-batch constants before activation microbatching."""
    valid = _checked_target_mask(target, target_mask)
    persistence = persistence.expand_as(baseline)
    if model.research_spec["variant"] != "projected_oracle":
        return build_ndr_loss_normalizers(
            model,
            {"y_true": target.unsqueeze(-1), "y_mask": valid.unsqueeze(-1)},
            baseline=baseline,
            persistence=persistence,
            device=target.device,
        )
    teacher = projected_oracle_teacher(target, baseline, persistence, valid, model.sd_node)
    count = valid.sum().to(target.dtype)
    return {
        "valid_count": count.detach(),
        "gate_confidence_sum": teacher.confidence.sum().detach(),
        "mean_gate_target": ((teacher.target * valid).sum() / (count + 1e-6)).detach(),
    }


def compute_research_loss(
    model: STAEformerDeseason,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    lambda_gate: float = 1.0,
    lambda_specialization: float = 0.3,
    normalizers: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """All targets remain loss-only; existing variants use the exact original loss."""
    valid = _checked_target_mask(target, target_mask)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("forecast must be finite and match the target")
    if model.research_spec["variant"] != "projected_oracle":
        return compute_loss(
            model,
            prediction,
            {"y_true": target.unsqueeze(-1), "y_mask": valid.unsqueeze(-1)},
            SimpleNamespace(lambda_gate=lambda_gate, lambda_spec=lambda_specialization),
            target.device,
            normalizers=normalizers,
        )
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    local_count = valid.sum().to(target.dtype)
    denominator = local_count if normalizers is None else normalizers["valid_count"]
    # Match the original masked MAE's non-microbatch denominator convention.
    primary = torch.where(valid, (prediction - safe_target).abs(), 0.0).sum() / denominator
    parts = {"mae": float(primary.detach())}
    loss = primary
    if not model.training or model.aux is None:
        return loss, parts
    aux = model.aux
    teacher = projected_oracle_teacher(target, aux["B"], aux["P"], valid, model.sd_node)
    confidence_denominator = (
        teacher.confidence.sum() if normalizers is None else normalizers["gate_confidence_sum"]
    )
    huber = F.smooth_l1_loss(aux["g"], teacher.target, reduction="none", beta=0.25)
    gate_loss = (
        teacher.confidence * model.sd_node.view(1, -1, 1) * huber
    ).sum() / (confidence_denominator + 1e-6)
    mean_target = (
        (teacher.target * valid).sum() / (local_count + 1e-6)
        if normalizers is None else normalizers["mean_gate_target"]
    )
    normal_weight = teacher.target / (mean_target + 1e-6)
    disturbance_weight = (1 - teacher.target) / (1 - mean_target + 1e-6)
    specialization = torch.where(
        valid,
        normal_weight * (aux["yn"] - safe_target).abs()
        + disturbance_weight * (aux["yd"] - safe_target).abs(),
        0.0,
    ).sum() / (denominator + 1e-6)
    loss = loss + lambda_gate * gate_loss + lambda_specialization * specialization
    parts.update(gate=float(gate_loss.detach()), spec=float(specialization.detach()))
    return loss, parts
