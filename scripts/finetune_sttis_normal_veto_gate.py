#!/usr/bin/env python3
"""Fine-tune a conservative normal-veto head on a proposal-aware ST-TIS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from analyze_dual_branch_gate import torch_load
from train_dual_branch_gate_baseline import cap_indices, infer_cache_shapes
from train_dual_branch_sttis_gate import (
    DualBranchSTTISHierarchicalImpactNormalVetoGate,
    DualBranchSTTISImpactConditionedNormalVetoGate,
    DualBranchSTTISNodeEventNormalVetoGate,
    DualBranchSTTISNormalVetoGate,
)
from train_full_candidate_stgnn_heatmap_model import CHANNELS, compute_stats, forecast_metrics_for_loader, make_loader, split_indices
from train_impact_residual_model import choose_device, json_safe_args


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_split_list(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val", "test"}
    bad = [item for item in splits if item not in allowed]
    if bad:
        raise ValueError(f"unsupported eval split(s): {bad}")
    if "test" not in splits:
        splits.append("test")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_convexgate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Batch size for validation/test sweeps; defaults to --batch-size when <= 0.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--affected-weight", type=float, default=4.0)
    parser.add_argument("--veto-loss-weight", type=float, default=0.15)
    parser.add_argument("--regret-loss-weight", type=float, default=0.08)
    parser.add_argument("--sparsity-loss-weight", type=float, default=0.02)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--ranking-positive-margin", type=float, default=0.20)
    parser.add_argument("--ranking-negative-margin", type=float, default=0.0)
    parser.add_argument("--ranking-pairs-per-batch", type=int, default=4096)
    parser.add_argument(
        "--ranking-affected-only",
        action="store_true",
        help="Rank veto logits only on affected-node positions.",
    )
    parser.add_argument("--normal-better-margin", type=float, default=0.10)
    parser.add_argument("--normal-better-target-scale", type=float, default=0.50)
    parser.add_argument("--normal-better-weight", type=float, default=4.0)
    parser.add_argument(
        "--node-event-veto-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary loss for hierarchical node-event normal-better pretraining.",
    )
    parser.add_argument(
        "--node-event-veto-positive-fraction",
        type=float,
        default=0.05,
        help="Minimum per-node normal-better element fraction for binary node-event target.",
    )
    parser.add_argument("--node-event-veto-positive-weight", type=float, default=4.0)
    parser.add_argument(
        "--node-event-veto-target-mode",
        choices=["binary", "fraction"],
        default="binary",
        help="Train the node-event veto prior with a binary hard target or a positive-element fraction.",
    )
    parser.add_argument(
        "--node-event-pretrain-epochs",
        type=int,
        default=0,
        help="Warm up only the hierarchical node-event normal-better detector before full veto fine-tuning.",
    )
    parser.add_argument(
        "--node-event-pretrain-lr",
        type=float,
        default=0.0,
        help="Learning rate for node-event detector warmup; defaults to --lr when <= 0.",
    )
    parser.add_argument(
        "--node-event-pretrain-affected-only",
        action="store_true",
        help="During node-event warmup, train only affected candidate nodes.",
    )
    parser.add_argument(
        "--node-event-pretrain-event-focus-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier on severity/recovery event-focus weights during node-event warmup only.",
    )
    parser.add_argument(
        "--veto-negative-weight",
        type=float,
        default=0.0,
        help="Extra BCE weight on non-normal-better positions; useful for high-precision binary detectors.",
    )
    parser.add_argument(
        "--veto-target-mode",
        choices=["continuous", "binary"],
        default="continuous",
        help="Use a ramped normal-better target or a strict binary detector target.",
    )
    parser.add_argument(
        "--veto-loss-kind",
        choices=["bce", "smooth_l1", "mse"],
        default="bce",
        help="Train veto logits with BCE or train sigmoid veto scores as a regression target.",
    )
    parser.add_argument("--normal-veto-init-bias", type=float, default=-4.0)
    parser.add_argument("--train-normal-veto-temperature", type=float, default=1.0)
    parser.add_argument("--train-normal-veto-scale", type=float, default=1.0)
    parser.add_argument(
        "--normal-veto-granularity",
        choices=["element", "node_event", "hierarchical"],
        default="element",
        help="Predict veto per horizon/channel element or one node-event score broadcast to all outputs.",
    )
    parser.add_argument(
        "--normal-veto-context",
        choices=["base", "impact_aux"],
        default="base",
        help="Whether the veto detector sees only proposal/gate features or predicted incident-impact auxiliary cues.",
    )
    parser.add_argument("--impact-aux-weight", type=float, default=0.0)
    parser.add_argument("--event-aux-weight", type=float, default=0.0)
    parser.add_argument("--node-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--severity-focus-weight",
        type=float,
        default=0.0,
        help="Extra affected-position loss weight for above-average standardized severity.",
    )
    parser.add_argument(
        "--recovery-focus-weight",
        type=float,
        default=0.0,
        help="Extra affected-position loss weight for above-average standardized recovery time.",
    )
    parser.add_argument("--event-focus-temperature", type=float, default=1.0)
    parser.add_argument(
        "--event-focus-max",
        type=float,
        default=3.0,
        help="Maximum additive event focus weight; set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--severity-target-boost",
        type=float,
        default=0.0,
        help="Boost the normal-veto target on affected positions for above-average standardized severity.",
    )
    parser.add_argument(
        "--recovery-target-boost",
        type=float,
        default=0.0,
        help="Boost the normal-veto target on affected positions for above-average standardized recovery time.",
    )
    parser.add_argument(
        "--event-target-boost-max",
        type=float,
        default=2.0,
        help="Maximum multiplicative target boost minus one; set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--severity-high-focus-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples above the training severity 66th percentile.",
    )
    parser.add_argument(
        "--recovery-long-focus-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples whose recovery duration is at least 90 minutes.",
    )
    parser.add_argument(
        "--high-long-focus-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples that are both severity-high and recovery-long.",
    )
    parser.add_argument(
        "--severity-high-target-boost",
        type=float,
        default=0.0,
        help="Boost the normal-veto target on severity-high affected positions.",
    )
    parser.add_argument(
        "--recovery-long-target-boost",
        type=float,
        default=0.0,
        help="Boost the normal-veto target on recovery-long affected positions.",
    )
    parser.add_argument(
        "--high-long-target-boost",
        type=float,
        default=0.0,
        help="Boost the normal-veto target on high-and-long affected positions.",
    )
    parser.add_argument(
        "--tail-normal-better-margin-add",
        type=float,
        default=0.0,
        help="Additional normal-better margin required on severity-high or recovery-long affected positions.",
    )
    parser.add_argument(
        "--tail-veto-negative-weight",
        type=float,
        default=0.0,
        help="Extra BCE weight against normal-veto false positives on high-risk affected positions.",
    )
    parser.add_argument(
        "--tail-sparsity-weight",
        type=float,
        default=0.0,
        help="Extra sparsity penalty for normal-veto amount on high-risk affected positions.",
    )
    parser.add_argument("--proposal-feature-count", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--sweep-scales", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--sweep-temperatures", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_candidates_model_robust_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_cache_path(model_dir: Path, ckpt: dict[str, object]) -> Path:
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        cache_path = Path(data["cache_path"])
    return cache_path.resolve()


def build_model(
    ckpt: dict[str, object],
    cache_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> DualBranchSTTISNormalVetoGate:
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError("checkpoint args must be a dict")
    shapes = infer_cache_shapes(cache_path)
    if args.normal_veto_context == "impact_aux":
        if args.normal_veto_granularity == "hierarchical":
            model_cls = DualBranchSTTISHierarchicalImpactNormalVetoGate
        elif args.normal_veto_granularity == "element":
            model_cls = DualBranchSTTISImpactConditionedNormalVetoGate
        else:
            raise ValueError("impact_aux normal-veto context supports element or hierarchical granularity")
    elif args.normal_veto_granularity == "node_event":
        model_cls = DualBranchSTTISNodeEventNormalVetoGate
    elif args.normal_veto_granularity == "hierarchical":
        raise ValueError("hierarchical normal-veto currently requires --normal-veto-context impact_aux")
    else:
        model_cls = DualBranchSTTISNormalVetoGate
    model = model_cls(
        channels=shapes["channels"],
        hist_input_channels=len(CHANNELS) * (2 if bool(model_args.get("use_dual_hist_residual", True)) else 1),
        node_context_dim=shapes["node_context_dim"],
        global_context_dim=shapes["global_context_dim"],
        horizon_steps=shapes["horizon_steps"],
        hidden_dim=int(model_args.get("hidden_dim", 96)),
        graph_layers=int(model_args.get("graph_layers", 2)),
        dropout=float(model_args.get("dropout", 0.10)),
        graph_sigma=float(model_args.get("graph_sigma", 3.0)),
        graph_mode=str(model_args.get("graph_mode", "undirected")),
        use_normal_delta=bool(model_args.get("use_normal_delta", True)),
        use_normal_delta_abs=bool(model_args.get("use_normal_delta_abs", True)),
        sttis_heads=int(model_args.get("sttis_heads", 4)),
        sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
        sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
        sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
        proposal_feature_count=args.proposal_feature_count,
        normal_veto_scale=args.train_normal_veto_scale,
        normal_veto_temperature=args.train_normal_veto_temperature,
        normal_veto_init_bias=args.normal_veto_init_bias,
    )
    source_state = ckpt["model_state_dict"]
    if not isinstance(source_state, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in state and tuple(state[key].shape) == tuple(value.shape)
    }
    state.update(compatible)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def freeze_except_veto(model: nn.Module, train_aux_heads: bool = False) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("normal_veto_head.")
            or name.startswith("node_event_veto_head.")
            or name.startswith("base_gate_logit_norm.")
            or name.startswith("impact_feature_norm.")
            or name.startswith("event_aux_feature_norm.")
            or (
                train_aux_heads
                and (
                    name.startswith("impact_head.")
                    or name.startswith("event_aux_head.")
                    or name.startswith("node_aux_head.")
                )
            )
        )
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no normal-veto parameters were marked trainable")
    return trainable


def freeze_for_node_event_pretrain(model: nn.Module) -> int:
    trainable = 0
    prefixes = (
        "node_event_veto_head.",
        "base_gate_logit_norm.",
        "impact_feature_norm.",
        "event_aux_feature_norm.",
    )
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("--node-event-pretrain-epochs requires a hierarchical model with node_event_veto_head")
    return trainable


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def masked_node_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    if weights is None:
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    weighted_mask = mask_f * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def compute_tail_thresholds(cache_path: Path, indices: np.ndarray, stats: object) -> dict[str, float]:
    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][np.sort(indices)].astype(np.float32)
    severity_raw_q66 = float(np.quantile(raw_event[:, 0], 2.0 / 3.0))
    recovery_raw_90min = 90.0 / 180.0
    event_mean = np.asarray(getattr(stats, "event_aux_mean"), dtype=np.float32)
    event_std = np.asarray(getattr(stats, "event_aux_std"), dtype=np.float32)
    return {
        "severity_high_z_threshold": float((severity_raw_q66 - event_mean[0]) / max(float(event_std[0]), 1e-6)),
        "recovery_long_z_threshold": float((recovery_raw_90min - event_mean[1]) / max(float(event_std[1]), 1e-6)),
        "severity_high_raw_threshold": severity_raw_q66,
        "recovery_long_raw_threshold": recovery_raw_90min,
    }


def tail_focus_vectors(
    event_aux: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    severity_high = event_aux[:, 0] >= float(getattr(args, "severity_high_z_threshold", float("inf")))
    recovery_long = event_aux[:, 1] >= float(getattr(args, "recovery_long_z_threshold", float("inf")))
    high_long = severity_high & recovery_long
    focus = (
        args.severity_high_focus_weight * severity_high.to(event_aux.dtype)
        + args.recovery_long_focus_weight * recovery_long.to(event_aux.dtype)
        + args.high_long_focus_weight * high_long.to(event_aux.dtype)
    )
    target_boost = (
        args.severity_high_target_boost * severity_high.to(event_aux.dtype)
        + args.recovery_long_target_boost * recovery_long.to(event_aux.dtype)
        + args.high_long_target_boost * high_long.to(event_aux.dtype)
    )
    if args.event_focus_max > 0.0:
        focus = focus.clamp(max=args.event_focus_max)
    if args.event_target_boost_max > 0.0:
        target_boost = target_boost.clamp(max=args.event_target_boost_max)
    return focus, target_boost, severity_high, recovery_long, high_long


def pairwise_ranking_loss(
    logits: torch.Tensor,
    normal_advantage: torch.Tensor,
    valid_mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float, float]:
    if args.ranking_loss_weight <= 0.0:
        zero = logits.sum() * 0.0
        return zero, 0.0, 0.0
    pos_mask = valid_mask.bool() & (normal_advantage > args.ranking_positive_margin)
    neg_mask = valid_mask.bool() & (normal_advantage <= args.ranking_negative_margin)
    pos_count = float(pos_mask.sum().detach().cpu())
    neg_count = float(neg_mask.sum().detach().cpu())
    if pos_count <= 0.0 or neg_count <= 0.0:
        zero = logits.sum() * 0.0
        return zero, pos_count, neg_count
    pos_logits = logits[pos_mask].flatten()
    neg_logits = logits[neg_mask].flatten()
    pair_count = min(pos_logits.numel(), neg_logits.numel(), max(int(args.ranking_pairs_per_batch), 1))
    pos_logits = pos_logits[:pair_count]
    neg_logits = neg_logits[:pair_count]
    # Softplus is a smooth hinge: low when pos_logit exceeds neg_logit by ranking_margin.
    loss = nn.functional.softplus(args.ranking_margin - pos_logits + neg_logits).mean()
    return loss, pos_count, neg_count


def compute_veto_loss(
    model: DualBranchSTTISNormalVetoGate,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        hist,
        hist_normal,
        node,
        global_context,
        normal_delta,
        y,
        y_mask,
        impact,
        impact_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, pred_impact, pred_event_aux, pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    affected_mask = node_affected[:, None, :, None].bool() & y_mask.bool()
    weights = 1.0 + args.affected_weight * affected_mask.to(y_mask.dtype)
    severity_focus = torch.relu(event_aux[:, 0] / max(args.event_focus_temperature, 1e-6))
    recovery_focus = torch.relu(event_aux[:, 1] / max(args.event_focus_temperature, 1e-6))
    event_focus = args.severity_focus_weight * severity_focus + args.recovery_focus_weight * recovery_focus
    if args.event_focus_max > 0.0:
        event_focus = event_focus.clamp(max=args.event_focus_max)
    tail_focus, tail_target_boost, severity_high, recovery_long, high_long = tail_focus_vectors(event_aux, args)
    event_focus_map = event_focus[:, None, None, None].expand_as(y)
    tail_focus_map = tail_focus[:, None, None, None].expand_as(y)
    weights = weights * (1.0 + event_focus_map * affected_mask.to(y_mask.dtype))
    weights = weights * (1.0 + tail_focus_map * affected_mask.to(y_mask.dtype))

    residual_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_raw, y_mask, weights)

    normal_abs = torch.abs(details["normal_residual"].detach() - y)
    base_abs = torch.abs(details["base_fused_residual"].detach() - y)
    normal_advantage = base_abs - normal_abs
    high_risk = severity_high | recovery_long
    high_risk_affected = high_risk[:, None, None, None].expand_as(y) & affected_mask
    margin = args.normal_better_margin + args.tail_normal_better_margin_add * high_risk_affected.to(y.dtype)
    positive = normal_advantage > margin
    if args.veto_target_mode == "binary":
        target = positive.to(y.dtype)
    else:
        target = ((normal_advantage - margin) / max(args.normal_better_target_scale, 1e-6)).clamp(0.0, 1.0)
    event_target_boost = args.severity_target_boost * severity_focus + args.recovery_target_boost * recovery_focus
    if args.event_target_boost_max > 0.0:
        event_target_boost = event_target_boost.clamp(max=args.event_target_boost_max)
    event_target_boost = event_target_boost + tail_target_boost
    if args.event_target_boost_max > 0.0:
        event_target_boost = event_target_boost.clamp(max=args.event_target_boost_max)
    event_target_boost_map = event_target_boost[:, None, None, None].expand_as(y)
    target = (target * (1.0 + event_target_boost_map * affected_mask.to(y.dtype))).clamp(0.0, 1.0)
    veto_weights = weights * (
        1.0
        + args.normal_better_weight * positive.to(weights.dtype)
        + args.veto_negative_weight * (~positive).to(weights.dtype)
        + args.tail_veto_negative_weight * high_risk_affected.to(weights.dtype) * (~positive).to(weights.dtype)
    )
    if args.veto_loss_kind == "bce":
        veto_raw = nn.functional.binary_cross_entropy_with_logits(details["normal_veto_logits"], target, reduction="none")
    elif args.veto_loss_kind == "smooth_l1":
        veto_raw = nn.functional.smooth_l1_loss(details["normal_veto"], target, reduction="none")
    elif args.veto_loss_kind == "mse":
        veto_raw = nn.functional.mse_loss(details["normal_veto"], target, reduction="none")
    else:
        raise ValueError(f"unsupported veto_loss_kind: {args.veto_loss_kind}")
    veto_loss = masked_mean(veto_raw, y_mask, veto_weights)
    ranking_mask = affected_mask if args.ranking_affected_only else y_mask.bool()
    ranking_loss, ranking_positive_count, ranking_negative_count = pairwise_ranking_loss(
        details["normal_veto_logits"],
        normal_advantage,
        ranking_mask,
        args,
    )
    node_event_veto_loss = pred_y.sum() * 0.0
    node_event_target_mean = 0.0
    node_event_positive_rate = 0.0
    if "node_event_normal_veto_logits" in details:
        valid_elements = y_mask.bool()
        positive_elements = positive & valid_elements
        element_count = valid_elements.to(y.dtype).sum(dim=(1, 3))
        positive_count = positive_elements.to(y.dtype).sum(dim=(1, 3))
        positive_fraction = positive_count / element_count.clamp_min(1.0)
        if args.node_event_veto_target_mode == "binary":
            node_event_target = (positive_fraction >= args.node_event_veto_positive_fraction).to(y.dtype)
        elif args.node_event_veto_target_mode == "fraction":
            node_event_target = positive_fraction.clamp(0.0, 1.0)
        else:
            raise ValueError(f"unsupported node_event_veto_target_mode: {args.node_event_veto_target_mode}")
        node_event_mask = node_valid.bool() & (element_count > 0)
        node_event_weights = (
            1.0
            + args.node_event_veto_positive_weight * (node_event_target > 0.0).to(y.dtype)
            + args.affected_weight * node_affected
        )
        node_event_weights = node_event_weights * (1.0 + event_focus[:, None] * node_affected)
        node_event_raw = nn.functional.binary_cross_entropy_with_logits(
            details["node_event_normal_veto_logits"],
            node_event_target,
            reduction="none",
        )
        node_event_veto_loss = masked_node_mean(node_event_raw, node_event_mask, node_event_weights)
        with torch.no_grad():
            node_event_target_mean = float(masked_node_mean(node_event_target, node_event_mask).detach().cpu())
            node_event_positive_rate = float(
                masked_node_mean((node_event_target > 0.0).to(y.dtype), node_event_mask).detach().cpu()
            )

    best_abs = torch.minimum(normal_abs, base_abs)
    pred_abs = torch.abs(pred_y - y)
    regret_loss = masked_mean((pred_abs - best_abs).clamp_min(0.0), y_mask, veto_weights)

    sparsity_weights = weights * (1.0 + args.tail_sparsity_weight * high_risk_affected.to(weights.dtype))
    sparsity_loss = masked_mean(details["normal_veto_amount"], y_mask, sparsity_weights)
    impact_raw = nn.functional.smooth_l1_loss(pred_impact, impact, reduction="none")
    impact_loss = (impact_raw * impact_mask).sum() / impact_mask.sum().clamp_min(1.0)
    event_aux_loss = nn.functional.smooth_l1_loss(pred_event_aux, event_aux)
    node_bce = nn.functional.binary_cross_entropy_with_logits(pred_node_logits, node_affected, reduction="none")
    node_aux_loss = (node_bce * node_valid).sum() / node_valid.sum().clamp_min(1.0)
    loss = (
        residual_loss
        + args.veto_loss_weight * veto_loss
        + args.node_event_veto_loss_weight * node_event_veto_loss
        + args.ranking_loss_weight * ranking_loss
        + args.regret_loss_weight * regret_loss
        + args.sparsity_loss_weight * sparsity_loss
        + args.impact_aux_weight * impact_loss
        + args.event_aux_weight * event_aux_loss
        + args.node_aux_weight * node_aux_loss
    )
    with torch.no_grad():
        valid_count = y_mask.sum().clamp_min(1.0)
        positive_rate = (positive & y_mask.bool()).to(torch.float32).sum() / valid_count
        affected_positive_rate = (positive & affected_mask).to(torch.float32).sum() / affected_mask.to(torch.float32).sum().clamp_min(1.0)
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "veto_loss": float(veto_loss.detach().cpu()),
        "node_event_veto_loss": float(node_event_veto_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "regret_loss": float(regret_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "impact_loss": float(impact_loss.detach().cpu()),
        "event_aux_loss": float(event_aux_loss.detach().cpu()),
        "node_aux_loss": float(node_aux_loss.detach().cpu()),
        "event_focus_mean": float(masked_mean(event_focus_map, y_mask).detach().cpu()),
        "affected_event_focus_mean": float(masked_mean(event_focus_map, affected_mask.to(y_mask.dtype)).detach().cpu()),
        "tail_focus_mean": float(masked_mean(tail_focus_map, y_mask).detach().cpu()),
        "affected_tail_focus_mean": float(masked_mean(tail_focus_map, affected_mask.to(y_mask.dtype)).detach().cpu()),
        "event_target_boost_mean": float(masked_mean(event_target_boost_map, y_mask).detach().cpu()),
        "affected_event_target_boost_mean": float(masked_mean(event_target_boost_map, affected_mask.to(y_mask.dtype)).detach().cpu()),
        "severity_high_rate": float(severity_high.to(torch.float32).mean().detach().cpu()),
        "recovery_long_rate": float(recovery_long.to(torch.float32).mean().detach().cpu()),
        "high_long_rate": float(high_long.to(torch.float32).mean().detach().cpu()),
        "high_risk_affected_positive_rate": float(
            (positive & high_risk_affected).to(torch.float32).sum().detach().cpu()
            / high_risk_affected.to(torch.float32).sum().clamp_min(1.0).detach().cpu()
        ),
        "normal_advantage_mean": float(masked_mean(normal_advantage, y_mask).detach().cpu()),
        "positive_rate": float(positive_rate.detach().cpu()),
        "affected_positive_rate": float(affected_positive_rate.detach().cpu()),
        "ranking_positive_count": ranking_positive_count,
        "ranking_negative_count": ranking_negative_count,
        "node_event_target_mean": node_event_target_mean,
        "node_event_positive_rate": node_event_positive_rate,
        "normal_veto_mean": float(masked_mean(details["normal_veto_amount"], y_mask).detach().cpu()),
        "affected_normal_veto_mean": float(masked_mean(details["normal_veto_amount"], affected_mask.to(y_mask.dtype)).detach().cpu()),
        "effective_gate_mean": float(masked_mean(details["gate"], y_mask).detach().cpu()),
        "base_gate_mean": float(masked_mean(details["base_gate"], y_mask).detach().cpu()),
    }


def compute_node_event_pretrain_loss(
    model: DualBranchSTTISNormalVetoGate,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    (
        hist,
        hist_normal,
        node,
        global_context,
        normal_delta,
        y,
        y_mask,
        _impact,
        _impact_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    if "node_event_normal_veto_logits" not in details:
        raise RuntimeError("node-event pretraining requires node_event_normal_veto_logits in model details")

    normal_abs = torch.abs(details["normal_residual"].detach() - y)
    base_abs = torch.abs(details["base_fused_residual"].detach() - y)
    normal_advantage = base_abs - normal_abs
    positive = normal_advantage > args.normal_better_margin

    valid_elements = y_mask.bool()
    positive_elements = positive & valid_elements
    element_count = valid_elements.to(y.dtype).sum(dim=(1, 3))
    positive_count = positive_elements.to(y.dtype).sum(dim=(1, 3))
    positive_fraction = positive_count / element_count.clamp_min(1.0)
    if args.node_event_veto_target_mode == "binary":
        node_event_target = (positive_fraction >= args.node_event_veto_positive_fraction).to(y.dtype)
    elif args.node_event_veto_target_mode == "fraction":
        node_event_target = positive_fraction.clamp(0.0, 1.0)
    else:
        raise ValueError(f"unsupported node_event_veto_target_mode: {args.node_event_veto_target_mode}")

    severity_focus = torch.relu(event_aux[:, 0] / max(args.event_focus_temperature, 1e-6))
    recovery_focus = torch.relu(event_aux[:, 1] / max(args.event_focus_temperature, 1e-6))
    event_focus = args.severity_focus_weight * severity_focus + args.recovery_focus_weight * recovery_focus
    if args.event_focus_max > 0.0:
        event_focus = event_focus.clamp(max=args.event_focus_max)
    tail_focus, _tail_target_boost, _severity_high, _recovery_long, _high_long = tail_focus_vectors(event_aux, args)
    event_focus = event_focus + tail_focus
    if args.event_focus_max > 0.0:
        event_focus = event_focus.clamp(max=args.event_focus_max)
    event_focus = event_focus * max(float(args.node_event_pretrain_event_focus_multiplier), 0.0)

    node_event_mask = node_valid.bool() & (element_count > 0)
    if args.node_event_pretrain_affected_only:
        node_event_mask = node_event_mask & node_affected.bool()
    node_event_weights = (
        1.0
        + args.node_event_veto_positive_weight * (node_event_target > 0.0).to(y.dtype)
        + args.affected_weight * node_affected
    )
    node_event_weights = node_event_weights * (1.0 + event_focus[:, None] * node_affected)
    node_event_raw = nn.functional.binary_cross_entropy_with_logits(
        details["node_event_normal_veto_logits"],
        node_event_target,
        reduction="none",
    )
    loss = masked_node_mean(node_event_raw, node_event_mask, node_event_weights)

    with torch.no_grad():
        node_event_veto = details["node_event_normal_veto"]
        affected_node_mask = node_event_mask & node_affected.bool()
        valid_count = valid_elements.to(torch.float32).sum().clamp_min(1.0)
        affected_element_count = (valid_elements & node_affected[:, None, :, None].bool()).to(torch.float32).sum().clamp_min(1.0)
        positive_rate = positive_elements.to(torch.float32).sum() / valid_count
        affected_positive_rate = (
            (positive_elements & node_affected[:, None, :, None].bool()).to(torch.float32).sum() / affected_element_count
        )
        node_event_target_mean = masked_node_mean(node_event_target, node_event_mask)
        node_event_positive_rate = masked_node_mean((node_event_target > 0.0).to(y.dtype), node_event_mask)
        affected_node_event_positive_rate = masked_node_mean(
            (node_event_target > 0.0).to(y.dtype),
            affected_node_mask,
        )
        node_event_veto_mean = masked_node_mean(node_event_veto, node_event_mask)
        affected_node_event_veto_mean = masked_node_mean(node_event_veto, affected_node_mask)

    return loss, {
        "node_event_pretrain_loss": float(loss.detach().cpu()),
        "normal_advantage_mean": float(masked_mean(normal_advantage, y_mask).detach().cpu()),
        "positive_rate": float(positive_rate.detach().cpu()),
        "affected_positive_rate": float(affected_positive_rate.detach().cpu()),
        "node_event_target_mean": float(node_event_target_mean.detach().cpu()),
        "node_event_positive_rate": float(node_event_positive_rate.detach().cpu()),
        "affected_node_event_positive_rate": float(affected_node_event_positive_rate.detach().cpu()),
        "node_event_veto_mean": float(node_event_veto_mean.detach().cpu()),
        "affected_node_event_veto_mean": float(affected_node_event_veto_mean.detach().cpu()),
        "prediction_anchor": float(pred_y.detach().abs().mean().cpu()),
    }


def evaluate_node_event_pretrain_loss(
    model: DualBranchSTTISNormalVetoGate,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    keys = [
        "loss",
        "node_event_pretrain_loss",
        "normal_advantage_mean",
        "positive_rate",
        "affected_positive_rate",
        "node_event_target_mean",
        "node_event_positive_rate",
        "affected_node_event_positive_rate",
        "node_event_veto_mean",
        "affected_node_event_veto_mean",
        "prediction_anchor",
    ]
    totals = {key: 0.0 for key in keys}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_node_event_pretrain_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += float(value) * batch_size
            count += batch_size
    return {key: totals[key] / max(count, 1) for key in keys}


def run_node_event_pretrain(
    model: DualBranchSTTISNormalVetoGate,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    loss_args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> int:
    if args.node_event_pretrain_epochs <= 0:
        return 0
    trainable = freeze_for_node_event_pretrain(model)
    pretrain_lr = args.node_event_pretrain_lr if args.node_event_pretrain_lr > 0.0 else args.lr
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=pretrain_lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None
    rows = []
    print(f"node-event pretrain params: {trainable}", flush=True)
    for epoch in range(1, args.node_event_pretrain_epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _parts = compute_node_event_pretrain_loss(model, batch, loss_args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_node_event_pretrain_loss(model, val_loader, loss_args, device)
        val_loss = float(val_metrics["loss"])
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"node-pretrain {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(output_dir / "node_event_pretrain_log.csv", index=False)
    return trainable


def evaluate_loss(
    model: DualBranchSTTISNormalVetoGate,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    keys = [
        "loss",
        "residual_loss",
        "veto_loss",
        "node_event_veto_loss",
        "ranking_loss",
        "regret_loss",
        "sparsity_loss",
        "impact_loss",
        "event_aux_loss",
        "node_aux_loss",
        "event_focus_mean",
        "affected_event_focus_mean",
        "tail_focus_mean",
        "affected_tail_focus_mean",
        "event_target_boost_mean",
        "affected_event_target_boost_mean",
        "severity_high_rate",
        "recovery_long_rate",
        "high_long_rate",
        "high_risk_affected_positive_rate",
        "normal_advantage_mean",
        "positive_rate",
        "affected_positive_rate",
        "ranking_positive_count",
        "ranking_negative_count",
        "node_event_target_mean",
        "node_event_positive_rate",
        "normal_veto_mean",
        "affected_normal_veto_mean",
        "effective_gate_mean",
        "base_gate_mean",
    ]
    totals = {key: 0.0 for key in keys}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_veto_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += float(value) * batch_size
            count += batch_size
    return {key: totals[key] / max(count, 1) for key in keys}


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    if log_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def sweep_scale_temperature_beta(
    model: DualBranchSTTISNormalVetoGate,
    loader: torch.utils.data.DataLoader,
    scales: list[float],
    temperatures: list[float],
    betas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    old_scale = float(model.normal_veto_scale)
    old_temperature = float(model.normal_veto_temperature)
    rows = []
    try:
        for scale in scales:
            model.normal_veto_scale = float(scale)
            for temperature in temperatures:
                model.normal_veto_temperature = float(temperature)
                metrics_by_beta = forecast_metrics_for_loader(model, loader, betas, device)
                rows.extend(
                    {
                        "normal_veto_scale": scale,
                        "normal_veto_temperature": temperature,
                        "residual_beta": beta,
                        **values,
                    }
                    for beta, values in metrics_by_beta.items()
                )
    finally:
        model.normal_veto_scale = old_scale
        model.normal_veto_temperature = old_temperature
    return pd.DataFrame(rows)


def select_scale_temperature_beta(val_sweep: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float, float]:
    if args.selection_metric not in val_sweep.columns:
        raise KeyError(f"selection metric not found in sweep: {args.selection_metric}")
    all_metric = "all_candidates_model_robust_mae"
    best_all = float(val_sweep[all_metric].min())
    eligible = val_sweep[val_sweep[all_metric] <= best_all + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_sweep
    row = eligible.loc[eligible[args.selection_metric].idxmin()]
    return float(row["normal_veto_scale"]), float(row["normal_veto_temperature"]), float(row["residual_beta"])


def metrics_at(
    model: DualBranchSTTISNormalVetoGate,
    loader: torch.utils.data.DataLoader,
    scale: float,
    temperature: float,
    residual_beta: float,
    device: torch.device,
) -> dict[str, float]:
    old_scale = float(model.normal_veto_scale)
    old_temperature = float(model.normal_veto_temperature)
    model.normal_veto_scale = float(scale)
    model.normal_veto_temperature = float(temperature)
    try:
        return forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]
    finally:
        model.normal_veto_scale = old_scale
        model.normal_veto_temperature = old_temperature


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    eval_counts: dict[str, int],
    scale: float,
    temperature: float,
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# ST-TIS Normal Veto Gate",
        "",
        "This variant freezes the source model and trains a conservative normal-veto head over the base fused proposal.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- normal-veto all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.4f}`",
        f"- normal-veto affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.4f}`",
        f"- normal_veto_scale: {scale:.2f}",
        f"- normal_veto_temperature: {temperature:.2f}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- epochs: {args.epochs}",
        f"- batch_size: {args.batch_size}",
        f"- eval_batch_size: {args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size}",
        f"- lr: {args.lr}",
        f"- affected_weight: {args.affected_weight}",
        f"- veto_loss_weight: {args.veto_loss_weight}",
        f"- ranking_loss_weight: {args.ranking_loss_weight}",
        f"- ranking_margin: {args.ranking_margin}",
        f"- ranking_positive_margin: {args.ranking_positive_margin}",
        f"- ranking_negative_margin: {args.ranking_negative_margin}",
        f"- ranking_pairs_per_batch: {args.ranking_pairs_per_batch}",
        f"- ranking_affected_only: {args.ranking_affected_only}",
        f"- regret_loss_weight: {args.regret_loss_weight}",
        f"- sparsity_loss_weight: {args.sparsity_loss_weight}",
        f"- normal_better_margin: {args.normal_better_margin}",
        f"- normal_better_weight: {args.normal_better_weight}",
        f"- node_event_veto_loss_weight: {args.node_event_veto_loss_weight}",
        f"- node_event_veto_positive_fraction: {args.node_event_veto_positive_fraction}",
        f"- node_event_veto_positive_weight: {args.node_event_veto_positive_weight}",
        f"- node_event_veto_target_mode: {args.node_event_veto_target_mode}",
        f"- node_event_pretrain_epochs: {args.node_event_pretrain_epochs}",
        f"- node_event_pretrain_lr: {args.node_event_pretrain_lr}",
        f"- node_event_pretrain_affected_only: {args.node_event_pretrain_affected_only}",
        f"- node_event_pretrain_event_focus_multiplier: {args.node_event_pretrain_event_focus_multiplier}",
        f"- veto_negative_weight: {args.veto_negative_weight}",
        f"- veto_target_mode: {args.veto_target_mode}",
        f"- veto_loss_kind: {args.veto_loss_kind}",
        f"- normal_veto_granularity: {args.normal_veto_granularity}",
        f"- normal_veto_context: {args.normal_veto_context}",
        f"- impact_aux_weight: {args.impact_aux_weight}",
        f"- event_aux_weight: {args.event_aux_weight}",
        f"- node_aux_weight: {args.node_aux_weight}",
        f"- severity_focus_weight: {args.severity_focus_weight}",
        f"- recovery_focus_weight: {args.recovery_focus_weight}",
        f"- event_focus_temperature: {args.event_focus_temperature}",
        f"- event_focus_max: {args.event_focus_max}",
        f"- severity_target_boost: {args.severity_target_boost}",
        f"- recovery_target_boost: {args.recovery_target_boost}",
        f"- event_target_boost_max: {args.event_target_boost_max}",
        f"- severity_high_focus_weight: {args.severity_high_focus_weight}",
        f"- recovery_long_focus_weight: {args.recovery_long_focus_weight}",
        f"- high_long_focus_weight: {args.high_long_focus_weight}",
        f"- severity_high_target_boost: {args.severity_high_target_boost}",
        f"- recovery_long_target_boost: {args.recovery_long_target_boost}",
        f"- high_long_target_boost: {args.high_long_target_boost}",
        f"- tail_normal_better_margin_add: {args.tail_normal_better_margin_add}",
        f"- tail_veto_negative_weight: {args.tail_veto_negative_weight}",
        f"- tail_sparsity_weight: {args.tail_sparsity_weight}",
        f"- severity_high_z_threshold: {getattr(args, 'severity_high_z_threshold', float('nan'))}",
        f"- recovery_long_z_threshold: {getattr(args, 'recovery_long_z_threshold', float('nan'))}",
        f"- normal_veto_init_bias: {args.normal_veto_init_bias}",
        f"- trainable normal-veto parameters: {trainable_params}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": split_counts[key], "eval_samples": eval_counts[key]} for key in split_counts]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Training",
        "",
        f"- best_epoch: {int(log_df.loc[log_df['val_loss'].idxmin(), 'epoch']) if not log_df.empty else 'n/a'}",
        f"- best_val_loss: {float(log_df['val_loss'].min()) if not log_df.empty else float('nan'):.4f}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source model: {model_dir}", flush=True)

    model = build_model(ckpt, cache_path, args, device)

    source_args = ckpt.get("args", {})
    if not isinstance(source_args, dict):
        source_args = {}
    loss_args = argparse.Namespace(**vars(args))
    loss_args.use_dual_hist_residual = bool(source_args.get("use_dual_hist_residual", True))

    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)
    eval_indices = {
        split: cap_indices(idx, args.max_eval_samples, args.seed + offset)
        for offset, (split, idx) in enumerate(indices.items())
    }
    train_indices_full = indices["train"]
    if args.max_train_samples > 0 and train_indices_full.size > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(rng.choice(train_indices_full, size=args.max_train_samples, replace=False))
    else:
        train_indices = train_indices_full
    tail_thresholds = compute_tail_thresholds(cache_path, train_indices, stats)
    for key, value in tail_thresholds.items():
        setattr(args, key, value)
        setattr(loss_args, key, value)
    print(
        "tail thresholds: "
        f"severity_high_z={tail_thresholds['severity_high_z_threshold']:.4f}, "
        f"recovery_long_z={tail_thresholds['recovery_long_z_threshold']:.4f}",
        flush=True,
    )

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    train_loader = make_loader(cache_path, train_indices, stats, args.batch_size, shuffle=True)
    val_loader = make_loader(cache_path, eval_indices["val"], stats, eval_batch_size, shuffle=False)
    pretrain_params = run_node_event_pretrain(model, train_loader, val_loader, args, loss_args, device, output_dir)
    if pretrain_params > 0:
        print(f"finished node-event pretrain with {pretrain_params} trainable params", flush=True)

    trainable_params = freeze_except_veto(model, train_aux_heads=args.normal_veto_context == "impact_aux")
    print(f"trainable normal-veto parameters: {trainable_params}", flush=True)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _parts = compute_veto_loss(model, batch, loss_args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loss(model, val_loader, loss_args, device)
        val_loss = float(val_metrics["loss"])
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    scales = parse_float_list(args.sweep_scales)
    temperatures = parse_float_list(args.sweep_temperatures)
    betas = parse_float_list(args.sweep_betas)
    val_sweep = sweep_scale_temperature_beta(model, val_loader, scales, temperatures, betas, device)
    val_sweep.to_csv(output_dir / "normal_veto_scale_temperature_beta_sweep.csv", index=False)
    scale, temperature, residual_beta = select_scale_temperature_beta(val_sweep, args)
    model.normal_veto_scale = scale
    model.normal_veto_temperature = temperature

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        idx = eval_indices[split]
        loader = make_loader(cache_path, idx, stats, eval_batch_size, shuffle=False)
        metrics[split] = metrics_at(model, loader, scale, temperature, residual_beta, device)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    ckpt_args = dict(source_args)
    ckpt_args.update(json_safe_args(args))
    if args.normal_veto_context == "impact_aux" and args.normal_veto_granularity == "hierarchical":
        ckpt_args["model_class"] = "DualBranchSTTISHierarchicalImpactNormalVetoGate"
    elif args.normal_veto_context == "impact_aux":
        ckpt_args["model_class"] = "DualBranchSTTISImpactConditionedNormalVetoGate"
    elif args.normal_veto_granularity == "node_event":
        ckpt_args["model_class"] = "DualBranchSTTISNodeEventNormalVetoGate"
    else:
        ckpt_args["model_class"] = "DualBranchSTTISNormalVetoGate"
    ckpt_args["training_variant"] = "normal_veto_finetune"
    ckpt_args["normal_veto_scale"] = scale
    ckpt_args["normal_veto_temperature"] = temperature
    ckpt_args["normal_veto_init_bias"] = args.normal_veto_init_bias
    ckpt_args["residual_beta"] = residual_beta
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": ckpt_args,
            "residual_beta": residual_beta,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "samples": split_counts,
                "eval_samples": eval_counts,
                "normal_veto_scale": scale,
                "normal_veto_temperature": temperature,
                "normal_veto_init_bias": args.normal_veto_init_bias,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
                "source_model_dir": str(model_dir),
                "source_residual_beta": float(ckpt.get("residual_beta", 1.0)),
                "selection_metric": args.selection_metric,
                "all_val_tolerance": args.all_val_tolerance,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = ckpt_args["model_class"]
    config["training_variant"] = "normal_veto_finetune"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    config["normal_veto_scale"] = scale
    config["normal_veto_temperature"] = temperature
    config["residual_beta"] = residual_beta
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, eval_counts, scale, temperature, residual_beta, trainable_params, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
