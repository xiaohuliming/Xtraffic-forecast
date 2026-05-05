#!/usr/bin/env python3
"""Train a frozen-model posthoc detector for normal-better veto positions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from compare_dual_branch_group_metrics import residual_beta
from train_dual_branch_gate_baseline import cap_indices
from train_full_candidate_stgnn_heatmap_model import CacheStats, compute_stats, split_indices
from train_impact_residual_model import choose_device


SUBSETS = ("all", "affected", "unaffected")
GROUPS = (
    "overall",
    "severity_low",
    "severity_mid",
    "severity_high",
    "recovery_short_lt30",
    "recovery_mid_30_90",
    "recovery_long_ge90",
    "severity_high_and_long",
)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "outputs/impact_guided_next_stage/"
            "dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/posthoc_normal_better_detector_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--positive-margin", type=float, default=0.10)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--affected-weight", type=float, default=3.0)
    parser.add_argument("--severity-focus-weight", type=float, default=0.5)
    parser.add_argument("--recovery-focus-weight", type=float, default=0.5)
    parser.add_argument("--event-focus-temperature", type=float, default=1.0)
    parser.add_argument("--event-focus-max", type=float, default=3.0)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-elements-per-batch", type=int, default=65536)
    parser.add_argument("--sweep-scales", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--sweep-temperatures", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
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
        payload = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        cache_path = Path(payload["cache_path"])
    return cache_path.resolve()


def model_uses_dual_hist(model: torch.nn.Module, hist_channels: int) -> bool:
    return int(getattr(model, "hist_input_channels", hist_channels)) > hist_channels


def read_event_groups(cache_path: Path, indices: np.ndarray) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    with h5py.File(cache_path, "r") as h5:
        event_aux = h5["event_aux"][indices].astype(np.float32)
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "overall": np.ones(indices.shape[0], dtype=bool),
        "severity_low": severity <= q33,
        "severity_mid": (severity > q33) & (severity <= q66),
        "severity_high": severity > q66,
        "recovery_short_lt30": recovery_min < 30.0,
        "recovery_mid_30_90": (recovery_min >= 30.0) & (recovery_min < 90.0),
        "recovery_long_ge90": recovery_min >= 90.0,
        "severity_high_and_long": (severity > q66) & (recovery_min >= 90.0),
    }
    return groups, (float(q33), float(q66))


class PosthocNormalBetterDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def channel_features(batch: int, horizon: int, nodes: int, channels: int, device: torch.device) -> torch.Tensor:
    eye = torch.eye(channels, dtype=torch.float32, device=device).reshape(1, 1, 1, channels, channels)
    return eye.expand(batch, horizon, nodes, channels, channels)


def build_element_features(
    details: dict[str, torch.Tensor],
    pred_impact: torch.Tensor,
    pred_event: torch.Tensor,
    pred_node_logits: torch.Tensor,
    normal_delta: torch.Tensor,
    node_context: torch.Tensor,
) -> torch.Tensor:
    normal = details["normal_residual"]
    incident = details["incident_residual"]
    base = details["base_fused_residual"]
    base_gate = details.get("base_gate", details.get("gate"))
    normal_veto = details.get("normal_veto")
    if normal_veto is None:
        normal_veto = torch.zeros_like(normal)

    batch, horizon, nodes, channels = normal.shape
    diff = incident - normal
    node_prob = torch.sigmoid(pred_node_logits).reshape(batch, 1, nodes, 1).expand(-1, horizon, -1, channels)
    impact = pred_impact.reshape(batch, horizon, nodes, 1).expand(-1, -1, -1, channels)
    event = pred_event.reshape(batch, 1, 1, 1, -1).expand(-1, horizon, nodes, channels, -1)
    node_ctx = node_context.reshape(batch, 1, nodes, 1, -1).expand(-1, horizon, -1, channels, -1)
    h = torch.linspace(0.0, 1.0, horizon, device=normal.device, dtype=normal.dtype).reshape(1, horizon, 1, 1)
    h = h.expand(batch, horizon, nodes, channels)
    channel = channel_features(batch, horizon, nodes, channels, normal.device)

    scalar_features = torch.stack(
        [
            normal,
            incident,
            base,
            diff,
            diff.abs(),
            normal.abs(),
            incident.abs(),
            base.abs(),
            base_gate,
            normal_veto,
            normal_delta,
            normal_delta.abs(),
            impact,
            node_prob,
            h,
        ],
        dim=-1,
    )
    features = torch.cat([scalar_features, event, node_ctx, channel], dim=-1)
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def prepare_batch(
    batch: tuple[torch.Tensor, ...],
    frozen_model: torch.nn.Module,
    dual_hist: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
        idx,
    ) = batch
    hist = hist.to(device)
    hist_normal = hist_normal.to(device)
    node = node.to(device)
    global_context = global_context.to(device)
    normal_delta = normal_delta.to(device)
    y = y.to(device)
    y_mask = y_mask.to(device)
    event_aux = event_aux.to(device)
    node_affected = node_affected.to(device)
    node_valid = node_valid.to(device)
    if dual_hist:
        hist = torch.cat([hist, hist_normal], dim=-1)

    with torch.no_grad():
        pred_y, pred_impact, pred_event, pred_node, details = frozen_model(
            hist,
            node,
            global_context,
            normal_delta,
            return_details=True,
        )
    features = build_element_features(
        details=details,
        pred_impact=pred_impact,
        pred_event=pred_event,
        pred_node_logits=pred_node,
        normal_delta=normal_delta,
        node_context=node,
    )
    base_abs = (details["base_fused_residual"] - y).abs()
    normal_abs = (details["normal_residual"] - y).abs()
    return {
        "features": features,
        "target": (base_abs - normal_abs > 0.0).to(features.dtype),
        "positive_margin_target": (base_abs - normal_abs > 0.10).to(features.dtype),
        "normal_advantage": base_abs - normal_abs,
        "source_pred": pred_y,
        "base_fused": details["base_fused_residual"],
        "normal": details["normal_residual"],
        "y": y,
        "y_mask": y_mask.bool(),
        "event_aux": event_aux,
        "node_affected": node_affected.bool(),
        "node_valid": node_valid.bool(),
        "idx": idx,
    }


def training_targets(prepared: dict[str, torch.Tensor], margin: float) -> torch.Tensor:
    return (prepared["normal_advantage"] > margin).to(prepared["features"].dtype)


def element_weights(prepared: dict[str, torch.Tensor], target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    affected = prepared["node_affected"][:, None, :, None].to(target.dtype)
    severity_focus = torch.relu(prepared["event_aux"][:, 0] / max(args.event_focus_temperature, 1e-6))
    recovery_focus = torch.relu(prepared["event_aux"][:, 1] / max(args.event_focus_temperature, 1e-6))
    event_focus = args.severity_focus_weight * severity_focus + args.recovery_focus_weight * recovery_focus
    if args.event_focus_max > 0.0:
        event_focus = event_focus.clamp(max=args.event_focus_max)
    weights = torch.ones_like(target)
    weights = weights + (args.affected_weight - 1.0) * affected
    weights = weights * (1.0 + event_focus[:, None, None, None] * affected)
    weights = weights * (1.0 + args.positive_weight * target)
    return weights


def sample_valid_elements(
    features: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    max_elements: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_valid = valid.flatten()
    valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    if max_elements > 0 and valid_idx.numel() > max_elements:
        choice = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_elements]
        valid_idx = valid_idx[choice]
    flat_features = features.reshape(-1, features.shape[-1])
    flat_target = target.flatten()
    flat_weights = weights.flatten()
    return flat_features[valid_idx], flat_target[valid_idx], flat_weights[valid_idx]


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weights).sum() / weights.sum().clamp_min(1.0)


def make_loader(cache_path: Path, indices: np.ndarray, stats: CacheStats, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_feature_dim(
    frozen_model: torch.nn.Module,
    cache_path: Path,
    indices: np.ndarray,
    stats: CacheStats,
    dual_hist: bool,
    device: torch.device,
) -> int:
    loader = make_loader(cache_path, indices[: min(indices.size, 4)], stats, batch_size=min(indices.size, 4), shuffle=False)
    prepared = prepare_batch(next(iter(loader)), frozen_model, dual_hist, device)
    return int(prepared["features"].shape[-1])


def train_detector(
    detector: PosthocNormalBetterDetector,
    frozen_model: torch.nn.Module,
    cache_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    stats: CacheStats,
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loader = make_loader(cache_path, train_idx, stats, args.batch_size, shuffle=True)
        detector.train()
        totals = {"loss": 0.0, "count": 0.0, "positive": 0.0}
        for batch_idx, batch in enumerate(train_loader, start=1):
            prepared = prepare_batch(batch, frozen_model, dual_hist, device)
            target = training_targets(prepared, args.positive_margin)
            weights = element_weights(prepared, target, args)
            features, sampled_target, sampled_weights = sample_valid_elements(
                prepared["features"],
                target,
                weights,
                prepared["y_mask"],
                args.max_elements_per_batch,
            )
            logits = detector(features)
            loss = weighted_bce_loss(logits, sampled_target, sampled_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = float(sampled_target.numel())
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["count"] += count
            totals["positive"] += float(sampled_target.sum().detach().cpu())
            if batch_idx % 20 == 0:
                print(
                    f"epoch {epoch}: trained {min(batch_idx * args.batch_size, train_idx.size)}/{train_idx.size}",
                    flush=True,
                )
        val_diag = detector_alignment(detector, frozen_model, cache_path, val_idx, stats, dual_hist, args.eval_batch_size, device)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": totals["loss"] / max(totals["count"], 1.0),
                "train_positive_rate": totals["positive"] / max(totals["count"], 1.0),
                **{f"val_{key}": value for key, value in val_diag.items()},
            }
        )
        print(
            f"epoch {epoch}: loss={rows[-1]['train_loss']:.6f}, "
            f"val_auc={rows[-1]['val_auc']:.6f}, val_score_gap={rows[-1]['val_score_pos_neg_gap']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def histogram_auc(scores: np.ndarray, labels: np.ndarray, bins: int = 400) -> float:
    if scores.size == 0:
        return float("nan")
    labels = labels.astype(bool)
    pos_total = float(labels.sum())
    neg_total = float((~labels).sum())
    if pos_total <= 0.0 or neg_total <= 0.0:
        return float("nan")
    hist_pos, edges = np.histogram(scores[labels], bins=bins, range=(0.0, 1.0))
    hist_neg, _ = np.histogram(scores[~labels], bins=edges)
    cum_neg_lower = np.cumsum(hist_neg) - hist_neg
    favorable = float((hist_pos * cum_neg_lower).sum() + 0.5 * (hist_pos * hist_neg).sum())
    return favorable / (pos_total * neg_total)


def detector_alignment(
    detector: PosthocNormalBetterDetector,
    frozen_model: torch.nn.Module,
    cache_path: Path,
    indices: np.ndarray,
    stats: CacheStats,
    dual_hist: bool,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(cache_path, indices, stats, batch_size, shuffle=False)
    score_sum = 0.0
    pos_score_sum = 0.0
    neg_score_sum = 0.0
    count = 0.0
    pos = 0.0
    hist_pos = np.zeros(400, dtype=np.float64)
    hist_neg = np.zeros(400, dtype=np.float64)
    detector.eval()
    with torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(batch, frozen_model, dual_hist, device)
            scores = torch.sigmoid(detector(prepared["features"].reshape(-1, prepared["features"].shape[-1]))).reshape(
                prepared["y"].shape
            )
            positive = training_targets(prepared, margin=0.10).bool()
            mask = prepared["y_mask"]
            s = scores[mask].detach().cpu().numpy()
            p = positive[mask].detach().cpu().numpy().astype(bool)
            if s.size == 0:
                continue
            score_sum += float(s.sum())
            pos_score_sum += float(s[p].sum()) if p.any() else 0.0
            neg_score_sum += float(s[~p].sum()) if (~p).any() else 0.0
            count += float(s.size)
            pos += float(p.sum())
            hp, edges = np.histogram(s[p], bins=400, range=(0.0, 1.0))
            hn, _ = np.histogram(s[~p], bins=edges)
            hist_pos += hp
            hist_neg += hn
    neg = max(count - pos, 1.0)
    pos_safe = max(pos, 1.0)
    cum_neg_lower = np.cumsum(hist_neg) - hist_neg
    auc = float(((hist_pos * cum_neg_lower).sum() + 0.5 * (hist_pos * hist_neg).sum()) / max(pos * (count - pos), 1.0))
    return {
        "score_mean": score_sum / max(count, 1.0),
        "positive_rate": pos / max(count, 1.0),
        "score_pos_mean": pos_score_sum / pos_safe,
        "score_neg_mean": neg_score_sum / neg,
        "score_pos_neg_gap": pos_score_sum / pos_safe - neg_score_sum / neg,
        "auc": auc,
    }


def empty_sums(scales: list[float], temperatures: list[float], betas: list[float]) -> dict[tuple[float, float, float], dict[str, float]]:
    return {
        (scale, temp, beta): {
            "all_model": 0.0,
            "all_base": 0.0,
            "all_source": 0.0,
            "all_count": 0.0,
            "affected_model": 0.0,
            "affected_base": 0.0,
            "affected_source": 0.0,
            "affected_count": 0.0,
            "unaffected_model": 0.0,
            "unaffected_base": 0.0,
            "unaffected_source": 0.0,
            "unaffected_count": 0.0,
            "amount_sum": 0.0,
            "affected_amount_sum": 0.0,
            "unaffected_amount_sum": 0.0,
        }
        for scale in scales
        for temp in temperatures
        for beta in betas
    }


def update_eval_sums(
    sums: dict[tuple[float, float, float], dict[str, float]],
    detector_scores: torch.Tensor,
    prepared: dict[str, torch.Tensor],
    source_beta: float,
    scales: list[float],
    temperatures: list[float],
    betas: list[float],
    group_mask: torch.Tensor,
) -> None:
    y = prepared["y"]
    y_mask = prepared["y_mask"] & group_mask[:, None, None, None]
    affected = prepared["node_affected"][:, None, :, None]
    valid = prepared["node_valid"][:, None, :, None]
    masks = {
        "all": y_mask,
        "affected": y_mask & affected,
        "unaffected": y_mask & (~affected) & valid,
    }
    base = prepared["base_fused"]
    normal = prepared["normal"]
    source_pred = prepared["source_pred"]
    base_abs = y.abs()
    source_abs = (source_beta * source_pred - y).abs()
    for scale in scales:
        for temp in temperatures:
            score = torch.sigmoid(torch.logit(detector_scores.clamp(1e-5, 1.0 - 1e-5)) / max(temp, 1e-6))
            amount = (scale * score).clamp(0.0, 1.0)
            residual = (1.0 - amount) * base + amount * normal
            for beta in betas:
                model_abs = (beta * residual - y).abs()
                row = sums[(scale, temp, beta)]
                for subset, mask in masks.items():
                    cnt = float(mask.sum().item())
                    if cnt <= 0.0:
                        continue
                    row[f"{subset}_model"] += float(model_abs[mask].sum().detach().cpu())
                    row[f"{subset}_base"] += float(base_abs[mask].sum().detach().cpu())
                    row[f"{subset}_source"] += float(source_abs[mask].sum().detach().cpu())
                    row[f"{subset}_count"] += cnt
                row["amount_sum"] += float(amount[masks["all"]].sum().detach().cpu())
                row["affected_amount_sum"] += float(amount[masks["affected"]].sum().detach().cpu())
                row["unaffected_amount_sum"] += float(amount[masks["unaffected"]].sum().detach().cpu())


def summarize_sums(
    split: str,
    group: str,
    sums: dict[tuple[float, float, float], dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for (scale, temp, beta), vals in sums.items():
        row: dict[str, float | str] = {
            "split": split,
            "group": group,
            "scale": scale,
            "temperature": temp,
            "beta": beta,
        }
        for subset in SUBSETS:
            count = max(vals[f"{subset}_count"], 1.0)
            row[f"{subset}_mae"] = vals[f"{subset}_model"] / count
            row[f"{subset}_base_mae"] = vals[f"{subset}_base"] / count
            row[f"{subset}_source_mae"] = vals[f"{subset}_source"] / count
        row["amount_mean"] = vals["amount_sum"] / max(vals["all_count"], 1.0)
        row["affected_amount_mean"] = vals["affected_amount_sum"] / max(vals["affected_count"], 1.0)
        row["unaffected_amount_mean"] = vals["unaffected_amount_sum"] / max(vals["unaffected_count"], 1.0)
        rows.append(row)
    return rows


def evaluate_sweep(
    detector: PosthocNormalBetterDetector,
    frozen_model: torch.nn.Module,
    cache_path: Path,
    indices: np.ndarray,
    split: str,
    source_beta: float,
    stats: CacheStats,
    dual_hist: bool,
    scales: list[float],
    temperatures: list[float],
    betas: list[float],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    groups, _severity_q = read_event_groups(cache_path, indices)
    loader = make_loader(cache_path, indices, stats, batch_size, shuffle=False)
    all_sums = {group: empty_sums(scales, temperatures, betas) for group in GROUPS}
    detector.eval()
    offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            prepared = prepare_batch(batch, frozen_model, dual_hist, device)
            flat_features = prepared["features"].reshape(-1, prepared["features"].shape[-1])
            detector_scores = torch.sigmoid(detector(flat_features)).reshape(prepared["y"].shape)
            batch_size_actual = prepared["y"].shape[0]
            for group in GROUPS:
                mask_np = groups[group][offset : offset + batch_size_actual]
                group_mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.bool)
                update_eval_sums(
                    all_sums[group],
                    detector_scores,
                    prepared,
                    source_beta,
                    scales,
                    temperatures,
                    betas,
                    group_mask,
                )
            offset += batch_size_actual
            if batch_idx % 20 == 0:
                print(f"{split}: evaluated {min(offset, indices.size)}/{indices.size}", flush=True)
    rows = []
    for group, sums in all_sums.items():
        rows.extend(summarize_sums(split, group, sums))
    return pd.DataFrame(rows)


def select_row(val_df: pd.DataFrame, metric: str, all_tolerance: float) -> pd.Series:
    overall = val_df[val_df["group"] == "overall"].copy()
    best_all = float(overall["all_mae"].min())
    eligible = overall[overall["all_mae"] <= best_all + all_tolerance]
    if eligible.empty:
        eligible = overall
    if metric not in eligible.columns:
        raise KeyError(f"selection metric not found: {metric}")
    return eligible.loc[eligible[metric].idxmin()]


def matching_row(df: pd.DataFrame, selected: pd.Series, group: str) -> pd.Series:
    sub = df[df["group"] == group].copy()
    mask = (
        np.isclose(sub["scale"].astype(float), float(selected["scale"]))
        & np.isclose(sub["temperature"].astype(float), float(selected["temperature"]))
        & np.isclose(sub["beta"].astype(float), float(selected["beta"]))
    )
    return sub[mask].iloc[0]


def write_summary(
    output_dir: Path,
    train_log: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected: pd.Series,
    alignment_val: dict[str, float],
    alignment_test: dict[str, float],
) -> None:
    test_row = matching_row(test_df, selected, "overall")
    group_rows = []
    for group in ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"]:
        row = matching_row(test_df, selected, group)
        group_rows.append(
            {
                "group": group,
                "all_mae": row["all_mae"],
                "affected_mae": row["affected_mae"],
                "unaffected_mae": row["unaffected_mae"],
                "source_affected_mae": row["affected_source_mae"],
                "affected_delta_vs_source": row["affected_mae"] - row["affected_source_mae"],
                "affected_amount_mean": row["affected_amount_mean"],
            }
        )
    group_df = pd.DataFrame(group_rows)
    show_cols = [
        "scale",
        "temperature",
        "beta",
        "all_mae",
        "affected_mae",
        "unaffected_mae",
        "affected_source_mae",
        "affected_amount_mean",
    ]
    lines = [
        "# Posthoc Normal-Better Detector",
        "",
        "Frozen source model; only a detached proposal-feature detector is trained.",
        "",
        "## Detector Alignment",
        "",
        pd.DataFrame(
            [
                {"split": "val", **alignment_val},
                {"split": "test", **alignment_test},
            ]
        ).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Validation-Selected Result",
        "",
        f"- scale: `{float(selected['scale']):.4g}`",
        f"- temperature: `{float(selected['temperature']):.4g}`",
        f"- beta: `{float(selected['beta']):.4g}`",
        f"- validation all / affected MAE: `{float(selected['all_mae']):.6f}` / `{float(selected['affected_mae']):.6f}`",
        f"- test all / affected / unaffected MAE: `{float(test_row['all_mae']):.6f}` / `{float(test_row['affected_mae']):.6f}` / `{float(test_row['unaffected_mae']):.6f}`",
        f"- test source all / affected / unaffected MAE: `{float(test_row['all_source_mae']):.6f}` / `{float(test_row['affected_source_mae']):.6f}` / `{float(test_row['unaffected_source_mae']):.6f}`",
        "",
        "## Test Groups At Selected Config",
        "",
        group_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top Validation Affected Rows",
        "",
        val_df[val_df["group"] == "overall"].sort_values("affected_mae")[show_cols].head(12).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Training Log",
        "",
        train_log.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    stats = compute_stats(cache_path)
    frozen_model = make_model(ckpt, cache_path, device)
    source_beta = residual_beta(model_dir, ckpt)
    dual_hist = model_uses_dual_hist(frozen_model, hist_channels=3)
    splits = split_indices(cache_path)
    train_idx = cap_indices(splits["train"], args.max_train_samples, args.seed)
    val_idx = cap_indices(splits["val"], args.max_eval_samples, args.seed + 1)
    test_idx = cap_indices(splits["test"], args.max_eval_samples, args.seed + 2)
    feature_dim = compute_feature_dim(frozen_model, cache_path, train_idx, stats, dual_hist, device)
    detector = PosthocNormalBetterDetector(feature_dim, args.hidden_dim, args.dropout).to(device)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source model: {model_dir}", flush=True)
    print(f"source beta: {source_beta}", flush=True)
    print(f"feature dim: {feature_dim}", flush=True)
    print(f"train/val/test samples: {train_idx.size}/{val_idx.size}/{test_idx.size}", flush=True)

    train_log = train_detector(detector, frozen_model, cache_path, train_idx, val_idx, stats, dual_hist, args, device)
    alignment_val = detector_alignment(detector, frozen_model, cache_path, val_idx, stats, dual_hist, args.eval_batch_size, device)
    alignment_test = detector_alignment(detector, frozen_model, cache_path, test_idx, stats, dual_hist, args.eval_batch_size, device)

    scales = parse_float_list(args.sweep_scales)
    temperatures = parse_float_list(args.sweep_temperatures)
    betas = parse_float_list(args.sweep_betas)
    val_df = evaluate_sweep(
        detector,
        frozen_model,
        cache_path,
        val_idx,
        "val",
        source_beta,
        stats,
        dual_hist,
        scales,
        temperatures,
        betas,
        args.eval_batch_size,
        device,
    )
    test_df = evaluate_sweep(
        detector,
        frozen_model,
        cache_path,
        test_idx,
        "test",
        source_beta,
        stats,
        dual_hist,
        scales,
        temperatures,
        betas,
        args.eval_batch_size,
        device,
    )
    selected = select_row(val_df, args.selection_metric, args.all_val_tolerance)

    train_log.to_csv(output_dir / "training_log.csv", index=False)
    val_df.to_csv(output_dir / "val_posthoc_detector_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_posthoc_detector_sweep.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "feature_dim": feature_dim,
                "source_beta": source_beta,
                "device": str(device),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "test_samples": int(test_idx.size),
                "selected": {
                    "scale": float(selected["scale"]),
                    "temperature": float(selected["temperature"]),
                    "beta": float(selected["beta"]),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    torch.save(
        {
            "detector_state_dict": detector.state_dict(),
            "feature_dim": feature_dim,
            "args": vars(args),
            "source_model_dir": str(model_dir),
            "source_beta": source_beta,
        },
        output_dir / "detector.pt",
    )
    write_summary(output_dir, train_log, val_df, test_df, selected, alignment_val, alignment_test)
    print(f"wrote posthoc detector outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
