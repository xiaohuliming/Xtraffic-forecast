#!/usr/bin/env python3
"""Train a frozen-source impact correction adapter for incident-aware forecasting."""

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

from analyze_dual_branch_gate import make_model, torch_load
from compare_dual_branch_group_metrics import read_event_groups, residual_beta, resolve_cache_path
from train_dual_branch_gate_baseline import cap_indices, infer_cache_shapes
from train_full_candidate_stgnn_heatmap_model import CHANNELS, compute_stats, forecast_metrics_for_loader, make_loader, split_indices
from train_impact_residual_model import choose_device, json_safe_args


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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-correction", type=float, default=0.35)
    parser.add_argument("--affected-weight", type=float, default=4.0)
    parser.add_argument("--severity-high-weight", type=float, default=0.75)
    parser.add_argument("--recovery-long-weight", type=float, default=0.75)
    parser.add_argument("--high-long-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight-max", type=float, default=2.0)
    parser.add_argument("--correction-l1-weight", type=float, default=0.02)
    parser.add_argument("--unaffected-correction-weight", type=float, default=0.05)
    parser.add_argument(
        "--correction-target-weight",
        type=float,
        default=0.0,
        help="Directly regress the correction toward clipped y - source_prediction during training.",
    )
    parser.add_argument("--correction-target-margin", type=float, default=0.0)
    parser.add_argument("--correction-target-tail-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--non-tail-affected-correction-weight", type=float, default=0.0)
    parser.add_argument(
        "--correction-regret-weight",
        type=float,
        default=0.0,
        help="Penalize correction elements that increase absolute error relative to the frozen source prediction.",
    )
    parser.add_argument("--correction-regret-tail-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--correction-node-gate-mode",
        choices=["none", "sigmoid"],
        default="none",
        help="Optionally gate the learned correction by the source model's predicted affected-node probability.",
    )
    parser.add_argument(
        "--correction-node-gate-floor",
        type=float,
        default=0.0,
        help="Minimum multiplicative gate for node-gated correction; 0 means fully gated, 1 disables the gate.",
    )
    parser.add_argument("--correction-node-gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--correction-anomaly-gate-mode",
        choices=["none", "branch_delta_abs"],
        default="none",
        help="Optionally gate correction by a residual/anomaly magnitude signal.",
    )
    parser.add_argument("--correction-anomaly-gate-threshold", type=float, default=0.5)
    parser.add_argument("--correction-anomaly-gate-temperature", type=float, default=0.25)
    parser.add_argument("--correction-anomaly-gate-floor", type=float, default=0.0)
    parser.add_argument("--base-beta", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument(
        "--selection-loss-key",
        choices=["loss", "final_loss"],
        default="loss",
        help="Validation loss field used for checkpoint selection.",
    )
    parser.add_argument("--write-group-metrics", action="store_true")
    parser.add_argument("--group-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
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


def tail_focus(event_aux: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    severity_high = event_aux[:, 0] >= float(getattr(args, "severity_high_z_threshold", float("inf")))
    recovery_long = event_aux[:, 1] >= float(getattr(args, "recovery_long_z_threshold", float("inf")))
    high_long = severity_high & recovery_long
    focus = (
        args.severity_high_weight * severity_high.to(event_aux.dtype)
        + args.recovery_long_weight * recovery_long.to(event_aux.dtype)
        + args.high_long_weight * high_long.to(event_aux.dtype)
    )
    if args.tail_weight_max > 0.0:
        focus = focus.clamp(max=args.tail_weight_max)
    return focus, severity_high, recovery_long, high_long


class ImpactCorrectionAdapter(nn.Module):
    """Frozen source model plus a small local residual correction head."""

    def __init__(
        self,
        base_model: nn.Module,
        base_beta: float,
        channels: int,
        horizon_steps: int,
        global_context_dim: int,
        hidden_dim: int,
        dropout: float,
        max_correction: float,
        correction_node_gate_mode: str = "none",
        correction_node_gate_floor: float = 0.0,
        correction_node_gate_temperature: float = 1.0,
        correction_anomaly_gate_mode: str = "none",
        correction_anomaly_gate_threshold: float = 0.5,
        correction_anomaly_gate_temperature: float = 0.25,
        correction_anomaly_gate_floor: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_beta = float(base_beta)
        self.channels = int(channels)
        self.horizon_steps = int(horizon_steps)
        self.hist_input_channels = int(getattr(base_model, "hist_input_channels", channels))
        self.max_correction = float(max_correction)
        self.correction_node_gate_mode = str(correction_node_gate_mode)
        self.correction_node_gate_floor = float(correction_node_gate_floor)
        self.correction_node_gate_temperature = float(correction_node_gate_temperature)
        self.correction_anomaly_gate_mode = str(correction_anomaly_gate_mode)
        self.correction_anomaly_gate_threshold = float(correction_anomaly_gate_threshold)
        self.correction_anomaly_gate_temperature = float(correction_anomaly_gate_temperature)
        self.correction_anomaly_gate_floor = float(correction_anomaly_gate_floor)

        scalar_feature_count = 12
        feature_dim = scalar_feature_count + 1 + 1 + 3 + int(global_context_dim) + 1 + self.channels
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.correction_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        horizon = torch.linspace(0.0, 1.0, steps=self.horizon_steps).reshape(1, self.horizon_steps, 1, 1, 1)
        channel_eye = torch.eye(self.channels).reshape(1, 1, 1, self.channels, self.channels)
        self.register_buffer("horizon_feature", horizon)
        self.register_buffer("channel_feature", channel_eye)

    def _scalar(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)

    def _expand_btn(self, x: torch.Tensor, batch: int, steps: int, nodes: int, channels: int) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != steps and x.shape[2] == steps:
            x = x.permute(0, 2, 1).contiguous()
        return x[:, :, :, None, None].expand(batch, steps, nodes, channels, 1)

    def _expand_bn(self, x: torch.Tensor, batch: int, steps: int, nodes: int, channels: int) -> torch.Tensor:
        return x[:, None, :, None, None].expand(batch, steps, nodes, channels, 1)

    def _expand_bf(self, x: torch.Tensor, batch: int, steps: int, nodes: int, channels: int) -> torch.Tensor:
        return x[:, None, None, None, :].expand(batch, steps, nodes, channels, x.shape[-1])

    def build_features(
        self,
        source_pred: torch.Tensor,
        pred_impact: torch.Tensor,
        pred_event_aux: torch.Tensor,
        pred_node_logits: torch.Tensor,
        details: dict[str, torch.Tensor],
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, steps, nodes, channels = source_pred.shape
        normal = self.base_beta * details["normal_residual"]
        incident = self.base_beta * details["incident_residual"]
        base_fused = self.base_beta * details.get("base_fused_residual", source_pred / max(self.base_beta, 1e-6))
        gate = details["gate"]
        base_gate = details.get("base_gate", gate)
        normal_veto = details.get("normal_veto_amount", torch.zeros_like(gate))
        delta = incident - normal
        nd = torch.zeros_like(source_pred) if normal_delta is None else normal_delta
        pieces = [
            self._scalar(source_pred),
            self._scalar(source_pred.abs()),
            self._scalar(normal),
            self._scalar(incident),
            self._scalar(delta),
            self._scalar(delta.abs()),
            self._scalar(base_fused),
            self._scalar(source_pred - base_fused),
            self._scalar(gate),
            self._scalar(base_gate),
            self._scalar(normal_veto),
            self._scalar(nd),
            self._expand_btn(pred_impact, batch, steps, nodes, channels),
            self._expand_bn(torch.tanh(pred_node_logits), batch, steps, nodes, channels),
            self._expand_bf(torch.tanh(pred_event_aux), batch, steps, nodes, channels),
            self._expand_bf(global_context, batch, steps, nodes, channels),
            self.horizon_feature.expand(batch, steps, nodes, channels, 1),
            self.channel_feature.expand(batch, steps, nodes, channels, channels),
        ]
        return torch.cat(pieces, dim=-1)

    def forward(
        self,
        hist_residual: torch.Tensor,
        node_context: torch.Tensor,
        global_context: torch.Tensor,
        normal_delta: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        self.base_model.eval()
        with torch.no_grad():
            base_pred, pred_impact, pred_event_aux, pred_node_logits, base_details = self.base_model(
                hist_residual,
                node_context,
                global_context,
                normal_delta,
                return_details=True,
            )
        source_pred = self.base_beta * base_pred
        normal_residual = self.base_beta * base_details["normal_residual"]
        incident_residual = self.base_beta * base_details["incident_residual"]
        features = self.build_features(source_pred, pred_impact, pred_event_aux, pred_node_logits, base_details, global_context, normal_delta)
        raw_correction = self.max_correction * torch.tanh(self.correction_head(self.feature_norm(features)).squeeze(-1))
        correction = raw_correction
        correction_node_gate = None
        correction_anomaly_gate = None
        if self.correction_node_gate_mode == "sigmoid":
            temperature = max(self.correction_node_gate_temperature, 1e-6)
            floor = min(max(self.correction_node_gate_floor, 0.0), 1.0)
            node_prob = torch.sigmoid(pred_node_logits / temperature)
            correction_node_gate = floor + (1.0 - floor) * node_prob[:, None, :, None].expand_as(raw_correction)
            correction = raw_correction * correction_node_gate
        if self.correction_anomaly_gate_mode == "branch_delta_abs":
            temperature = max(self.correction_anomaly_gate_temperature, 1e-6)
            floor = min(max(self.correction_anomaly_gate_floor, 0.0), 1.0)
            anomaly = torch.abs(incident_residual - normal_residual)
            correction_anomaly_gate = floor + (1.0 - floor) * torch.sigmoid(
                (anomaly - self.correction_anomaly_gate_threshold) / temperature
            )
            correction = correction * correction_anomaly_gate
        pred_y = source_pred + correction
        if return_details:
            details = dict(base_details)
            details["source_pred"] = source_pred
            details["raw_correction"] = raw_correction
            details["correction"] = correction
            if correction_node_gate is not None:
                details["correction_node_gate"] = correction_node_gate
            if correction_anomaly_gate is not None:
                details["correction_anomaly_gate"] = correction_anomaly_gate
            details["corrected_pred"] = pred_y
            return pred_y, pred_impact, pred_event_aux, pred_node_logits, details
        return pred_y, pred_impact, pred_event_aux, pred_node_logits


def compute_loss(
    model: ImpactCorrectionAdapter,
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
    if model.hist_input_channels > hist.shape[-1]:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    affected_mask = node_affected[:, None, :, None].bool() & y_mask.bool()
    valid_unaffected = y_mask.bool() & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
    weights = 1.0 + args.affected_weight * affected_mask.to(y_mask.dtype)
    tail_weight, severity_high, recovery_long, high_long = tail_focus(event_aux, args)
    tail_sample = severity_high | recovery_long
    tail_weight_map = tail_weight[:, None, None, None].expand_as(y)
    weights = weights * (1.0 + tail_weight_map * affected_mask.to(y_mask.dtype))

    final_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    source_raw = nn.functional.smooth_l1_loss(details["source_pred"].detach(), y, reduction="none")
    final_loss = masked_mean(final_raw, y_mask, weights)
    source_loss = masked_mean(source_raw, y_mask, weights)
    correction_abs = details["correction"].abs()
    correction_l1 = masked_mean(correction_abs, y_mask)
    unaffected_correction = masked_mean(correction_abs, valid_unaffected.to(y_mask.dtype)) if valid_unaffected.any() else final_loss * 0.0
    tail_mask = affected_mask & tail_sample[:, None, None, None]
    non_tail_affected = affected_mask & (~tail_sample[:, None, None, None])
    non_tail_affected_correction = (
        masked_mean(correction_abs, non_tail_affected.to(y_mask.dtype)) if non_tail_affected.any() else final_loss * 0.0
    )
    if args.correction_target_weight > 0.0:
        correction_target = (y - details["source_pred"].detach()).clamp(-args.max_correction, args.max_correction)
        target_mask = (tail_mask if args.correction_target_tail_only else y_mask.bool()) & y_mask.bool()
        if args.correction_target_margin > 0.0:
            target_mask = target_mask & (correction_target.abs() > args.correction_target_margin)
        target_raw = nn.functional.smooth_l1_loss(details["correction"], correction_target, reduction="none")
        target_loss = masked_mean(target_raw, target_mask.to(y_mask.dtype), weights) if target_mask.any() else final_loss * 0.0
    else:
        target_mask = torch.zeros_like(y_mask, dtype=torch.bool)
        target_loss = final_loss * 0.0
    if args.correction_regret_weight > 0.0:
        regret_target = y - details["source_pred"].detach()
        regret_raw = torch.relu(torch.abs(regret_target - details["correction"]) - torch.abs(regret_target))
        regret_mask = (tail_mask if args.correction_regret_tail_only else y_mask.bool()) & y_mask.bool()
        regret_loss = masked_mean(regret_raw, regret_mask.to(y_mask.dtype), weights) if regret_mask.any() else final_loss * 0.0
    else:
        regret_mask = torch.zeros_like(y_mask, dtype=torch.bool)
        regret_loss = final_loss * 0.0
    loss = (
        final_loss
        + args.correction_l1_weight * correction_l1
        + args.unaffected_correction_weight * unaffected_correction
        + args.non_tail_affected_correction_weight * non_tail_affected_correction
        + args.correction_target_weight * target_loss
        + args.correction_regret_weight * regret_loss
    )

    with torch.no_grad():
        affected_correction = masked_mean(correction_abs, affected_mask.to(y_mask.dtype)) if affected_mask.any() else correction_l1 * 0.0
        affected_rate = affected_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        target_rate = target_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        regret_rate = regret_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "source_loss": float(source_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "regret_loss": float(regret_loss.detach().cpu()),
        "correction_l1": float(correction_l1.detach().cpu()),
        "unaffected_correction": float(unaffected_correction.detach().cpu()),
        "non_tail_affected_correction": float(non_tail_affected_correction.detach().cpu()),
        "affected_correction": float(affected_correction.detach().cpu()),
        "affected_rate": float(affected_rate.detach().cpu()),
        "target_rate": float(target_rate.detach().cpu()),
        "regret_rate": float(regret_rate.detach().cpu()),
        "severity_high_rate": float(severity_high.to(torch.float32).mean().detach().cpu()),
        "recovery_long_rate": float(recovery_long.to(torch.float32).mean().detach().cpu()),
        "high_long_rate": float(high_long.to(torch.float32).mean().detach().cpu()),
    }


def evaluate_loss(
    model: ImpactCorrectionAdapter,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_loss(model, batch, args, device)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Impact correction adapter training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_group_metrics(
    output_dir: Path,
    model: ImpactCorrectionAdapter,
    cache_path: Path,
    stats: object,
    split_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    groups = read_event_groups(cache_path, split_idx)
    rows = []
    for group, (mask, label) in groups.items():
        idx = split_idx[mask]
        row: dict[str, float | str | int] = {"group": group, "label": label, "samples": int(idx.size)}
        if idx.size > 0:
            loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
            source_metrics = forecast_metrics_for_loader(model.base_model, loader, [model.base_beta], device)[model.base_beta]
            adapter_metrics = forecast_metrics_for_loader(model, loader, [1.0], device)[1.0]
            for prefix, metrics in [("source", source_metrics), ("adapter", adapter_metrics)]:
                row[f"{prefix}_all_mae"] = metrics["all_candidates_model_robust_mae"]
                row[f"{prefix}_affected_mae"] = metrics["affected_candidates_model_robust_mae"]
                row[f"{prefix}_unaffected_mae"] = metrics["unaffected_candidates_model_robust_mae"]
            for target in ["all", "affected", "unaffected"]:
                row[f"{target}_delta"] = float(row[f"adapter_{target}_mae"]) - float(row[f"source_{target}_mae"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "group_metrics.csv", index=False)
    cols = [
        "group",
        "samples",
        "source_all_mae",
        "adapter_all_mae",
        "all_delta",
        "source_affected_mae",
        "adapter_affected_mae",
        "affected_delta",
        "source_unaffected_mae",
        "adapter_unaffected_mae",
        "unaffected_delta",
    ]
    lines = [
        "# Impact Correction Group Metrics",
        "",
        "Negative delta means the adapter is better.",
        "",
        df[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "group_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    eval_counts: dict[str, int],
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    cols = ["all_candidates_model_robust_mae", "affected_candidates_model_robust_mae", "unaffected_candidates_model_robust_mae"]
    table = pd.DataFrame([{"split": split, **{col: values[col] for col in cols}} for split, values in metrics.items()])
    test = metrics["test"]
    selection_col = "selection_loss" if "selection_loss" in log_df.columns else "val_loss"
    best_idx = log_df[selection_col].idxmin() if not log_df.empty else None
    best_epoch = int(log_df.loc[best_idx, "epoch"]) if best_idx is not None else "n/a"
    best_selection_loss = float(log_df.loc[best_idx, selection_col]) if best_idx is not None else float("nan")
    best_val_loss = float(log_df["val_loss"].min()) if not log_df.empty else float("nan")
    lines = [
        "# Impact Correction Adapter",
        "",
        "This variant freezes the source dual-branch model and learns a small local correction for incident-impact magnitude.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.6f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.6f}`",
        f"- adapter all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.6f}`",
        f"- adapter affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.6f}`",
        f"- adapter unaffected candidates robust MAE: `{test['unaffected_candidates_model_robust_mae']:.6f}`",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- base_beta: {args.base_beta}",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- max_correction: {args.max_correction}",
        f"- affected_weight: {args.affected_weight}",
        f"- severity_high_weight: {args.severity_high_weight}",
        f"- recovery_long_weight: {args.recovery_long_weight}",
        f"- high_long_weight: {args.high_long_weight}",
        f"- correction_l1_weight: {args.correction_l1_weight}",
        f"- unaffected_correction_weight: {args.unaffected_correction_weight}",
        f"- correction_target_weight: {args.correction_target_weight}",
        f"- correction_target_tail_only: {args.correction_target_tail_only}",
        f"- non_tail_affected_correction_weight: {args.non_tail_affected_correction_weight}",
        f"- correction_regret_weight: {args.correction_regret_weight}",
        f"- correction_regret_tail_only: {args.correction_regret_tail_only}",
        f"- correction_node_gate_mode: {args.correction_node_gate_mode}",
        f"- correction_node_gate_floor: {args.correction_node_gate_floor}",
        f"- correction_node_gate_temperature: {args.correction_node_gate_temperature}",
        f"- correction_anomaly_gate_mode: {args.correction_anomaly_gate_mode}",
        f"- correction_anomaly_gate_threshold: {args.correction_anomaly_gate_threshold}",
        f"- correction_anomaly_gate_floor: {args.correction_anomaly_gate_floor}",
        f"- correction_anomaly_gate_temperature: {args.correction_anomaly_gate_temperature}",
        f"- selection_loss_key: {args.selection_loss_key}",
        f"- trainable parameters: {trainable_params}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": split_counts[key], "eval_samples": eval_counts[key]} for key in split_counts]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Training",
        "",
        f"- best_epoch: {best_epoch}",
        f"- best_selection_loss: {best_selection_loss:.6f}",
        f"- best_val_loss: {best_val_loss:.6f}",
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
    stats = compute_stats(cache_path)
    shapes = infer_cache_shapes(cache_path)
    base = make_model(ckpt, cache_path, device)
    if args.base_beta is None:
        args.base_beta = residual_beta(model_dir, ckpt)
    model = ImpactCorrectionAdapter(
        base_model=base,
        base_beta=float(args.base_beta),
        channels=int(shapes["channels"]),
        horizon_steps=int(shapes["horizon_steps"]),
        global_context_dim=int(shapes["global_context_dim"]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_correction=args.max_correction,
        correction_node_gate_mode=args.correction_node_gate_mode,
        correction_node_gate_floor=args.correction_node_gate_floor,
        correction_node_gate_temperature=args.correction_node_gate_temperature,
        correction_anomaly_gate_mode=args.correction_anomaly_gate_mode,
        correction_anomaly_gate_threshold=args.correction_anomaly_gate_threshold,
        correction_anomaly_gate_temperature=args.correction_anomaly_gate_temperature,
        correction_anomaly_gate_floor=args.correction_anomaly_gate_floor,
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source model: {model_dir}", flush=True)
    print(f"base_beta: {args.base_beta}", flush=True)
    print(f"trainable parameters: {trainable_params}", flush=True)

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
    print(
        "tail thresholds: "
        f"severity_high_z={tail_thresholds['severity_high_z_threshold']:.4f}, "
        f"recovery_long_z={tail_thresholds['recovery_long_z_threshold']:.4f}",
        flush=True,
    )

    train_loader = make_loader(cache_path, train_indices, stats, args.batch_size, shuffle=True)
    val_loader = make_loader(cache_path, eval_indices["val"], stats, args.batch_size, shuffle=False)
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
            loss, _parts = compute_loss(model, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loss(model, val_loader, args, device)
        val_loss = float(val_metrics["loss"])
        selection_loss = float(val_metrics[args.selection_loss_key])
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_loss": selection_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        print(f"epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} select={selection_loss:.6f}", flush=True)
        if selection_loss < best_val:
            best_val = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items() if not key.startswith("base_model.")}

    if best_state is not None:
        current = model.state_dict()
        current.update(best_state)
        model.load_state_dict(current, strict=True)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        loader = make_loader(cache_path, eval_indices[split], stats, args.batch_size, shuffle=False)
        metrics[split] = forecast_metrics_for_loader(model, loader, [1.0], device)[1.0]

    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    adapter_state = {key: value.detach().cpu() for key, value in model.state_dict().items() if not key.startswith("base_model.")}
    torch.save(
        {
            "adapter_state_dict": adapter_state,
            "args": json_safe_args(args),
            "source_model_dir": str(model_dir),
            "cache_path": str(cache_path),
            "base_beta": float(args.base_beta),
            "training_variant": "impact_correction_adapter",
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "samples": split_counts,
                "eval_samples": eval_counts,
                "residual_beta": 1.0,
                "source_beta": float(args.base_beta),
                "cache_path": str(cache_path),
                "source_model_dir": str(model_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, eval_counts, trainable_params, log_df)
    if args.write_group_metrics:
        write_group_metrics(output_dir, model, cache_path, stats, indices[args.group_split], args, device)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
