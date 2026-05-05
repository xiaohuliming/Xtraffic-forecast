#!/usr/bin/env python3
"""Train a deployable selector between source veto and posthoc detector residuals."""

from __future__ import annotations

import argparse
import json
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
from train_posthoc_normal_better_detector import PosthocNormalBetterDetector, build_element_features


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
SUBSETS = ("all", "affected", "unaffected")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=Path(
            "outputs/impact_guided_next_stage/"
            "dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware"
        ),
    )
    parser.add_argument(
        "--detector-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/posthoc_normal_better_detector_seed_23_full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/source_posthoc_selector_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-elements-per-batch", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--posthoc-scale", type=float, default=0.20)
    parser.add_argument("--posthoc-temperature", type=float, default=2.0)
    parser.add_argument("--train-beta", type=float, default=1.10)
    parser.add_argument("--selector-target-margin", type=float, default=0.0)
    parser.add_argument("--selector-loss-weight", type=float, default=0.5)
    parser.add_argument("--final-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-prior-weight", type=float, default=0.01)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--affected-weight", type=float, default=6.0)
    parser.add_argument("--severity-focus-weight", type=float, default=2.0)
    parser.add_argument("--recovery-focus-weight", type=float, default=2.0)
    parser.add_argument("--event-focus-temperature", type=float, default=1.0)
    parser.add_argument("--event-focus-max", type=float, default=5.0)
    parser.add_argument("--sweep-selector-scales", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--sweep-selector-temperatures", default="0.75,1.0,1.5,2.0")
    parser.add_argument("--sweep-betas", default="1.05,1.1")
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


def make_loader(cache_path: Path, indices: np.ndarray, stats: CacheStats, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_detector(detector_dir: Path, device: torch.device) -> PosthocNormalBetterDetector:
    payload = torch_load(detector_dir / "detector.pt")
    feature_dim = int(payload["feature_dim"])
    detector_args = payload.get("args", {})
    hidden_dim = int(detector_args.get("hidden_dim", 96)) if isinstance(detector_args, dict) else 96
    dropout = float(detector_args.get("dropout", 0.10)) if isinstance(detector_args, dict) else 0.10
    detector = PosthocNormalBetterDetector(feature_dim, hidden_dim, dropout)
    state = payload["detector_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("detector_state_dict must be a dict")
    detector.load_state_dict(state, strict=True)
    detector.to(device)
    detector.eval()
    return detector


class SourcePosthocSelector(nn.Module):
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


def event_semantic(raw_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    severity = np.expm1(raw_event[:, 0])
    recovery = raw_event[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery[~np.isfinite(recovery)] = 0.0
    severity[severity < 0.0] = 0.0
    recovery[recovery < 0.0] = 0.0
    return severity, recovery


def severity_quantiles(cache_path: Path, indices: np.ndarray) -> tuple[float, float]:
    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][indices].astype(np.float32)
    severity, _recovery = event_semantic(raw_event)
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def group_masks(raw_event: np.ndarray, severity_q: tuple[float, float]) -> dict[str, np.ndarray]:
    severity, recovery = event_semantic(raw_event)
    q33, q66 = severity_q
    return {
        "overall": np.ones(raw_event.shape[0], dtype=bool),
        "severity_low": severity <= q33,
        "severity_mid": (severity > q33) & (severity <= q66),
        "severity_high": severity > q66,
        "recovery_short_lt30": recovery < 30.0,
        "recovery_mid_30_90": (recovery >= 30.0) & (recovery < 90.0),
        "recovery_long_ge90": recovery >= 90.0,
        "severity_high_and_long": (severity > q66) & (recovery >= 90.0),
    }


def prepare_batch(
    batch: tuple[torch.Tensor, ...],
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    dual_hist: bool,
    posthoc_scale: float,
    posthoc_temperature: float,
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
    y_mask = y_mask.to(device).bool()
    event_aux = event_aux.to(device)
    node_affected = node_affected.to(device).bool()
    node_valid = node_valid.to(device).bool()
    if dual_hist:
        hist = torch.cat([hist, hist_normal], dim=-1)
    with torch.no_grad():
        source_pred, pred_impact, pred_event, pred_node, details = source_model(
            hist,
            node,
            global_context,
            normal_delta,
            return_details=True,
        )
        posthoc_features = build_element_features(
            details=details,
            pred_impact=pred_impact,
            pred_event=pred_event,
            pred_node_logits=pred_node,
            normal_delta=normal_delta,
            node_context=node,
        )
        posthoc_logits = posthoc_detector(posthoc_features.reshape(-1, posthoc_features.shape[-1])).reshape(y.shape)
        posthoc_amount = (
            posthoc_scale * torch.sigmoid(posthoc_logits / max(posthoc_temperature, 1e-6))
        ).clamp(0.0, 1.0)
        source_amount = details["normal_veto_amount"]
        base = details["base_fused_residual"]
        normal = details["normal_residual"]
        source_residual = (1.0 - source_amount) * base + source_amount * normal
        posthoc_residual = (1.0 - posthoc_amount) * base + posthoc_amount * normal
        selector_features = torch.cat(
            [
                posthoc_features,
                source_amount.unsqueeze(-1),
                posthoc_amount.unsqueeze(-1),
                source_residual.unsqueeze(-1),
                posthoc_residual.unsqueeze(-1),
                (posthoc_residual - source_residual).unsqueeze(-1),
                (posthoc_amount - source_amount).unsqueeze(-1),
            ],
            dim=-1,
        )
        selector_features = torch.nan_to_num(selector_features, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "features": selector_features,
        "source_residual": source_residual,
        "posthoc_residual": posthoc_residual,
        "source_pred": source_pred,
        "y": y,
        "y_mask": y_mask,
        "event_aux": event_aux,
        "node_affected": node_affected,
        "node_valid": node_valid,
        "idx": idx,
    }


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


def sample_elements(
    features: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    max_elements: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_valid = valid.flatten()
    valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    if max_elements > 0 and valid_idx.numel() > max_elements:
        choice = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_elements]
        valid_idx = valid_idx[choice]
    return (
        features.reshape(-1, features.shape[-1])[valid_idx],
        target.flatten()[valid_idx],
        weights.flatten()[valid_idx],
        valid_idx,
    )


def train_selector(
    selector: SourcePosthocSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, args.epochs + 1):
        loader = make_loader(cache_path, train_idx, stats, args.batch_size, shuffle=True)
        selector.train()
        totals = {"loss": 0.0, "final_loss": 0.0, "selector_loss": 0.0, "source_prior": 0.0, "count": 0.0, "target": 0.0}
        for batch_idx, batch in enumerate(loader, start=1):
            prepared = prepare_batch(
                batch,
                source_model,
                posthoc_detector,
                dual_hist,
                args.posthoc_scale,
                args.posthoc_temperature,
                device,
            )
            source_abs = (args.train_beta * prepared["source_residual"] - prepared["y"]).abs()
            posthoc_abs = (args.train_beta * prepared["posthoc_residual"] - prepared["y"]).abs()
            target = (posthoc_abs + args.selector_target_margin < source_abs).to(prepared["y"].dtype)
            weights = element_weights(prepared, target, args)
            features, sampled_target, sampled_weights, flat_idx = sample_elements(
                prepared["features"],
                target,
                weights,
                prepared["y_mask"],
                args.max_elements_per_batch,
            )
            source_flat = prepared["source_residual"].reshape(-1)[flat_idx]
            posthoc_flat = prepared["posthoc_residual"].reshape(-1)[flat_idx]
            y_flat = prepared["y"].reshape(-1)[flat_idx]
            logits = selector(features)
            gate = torch.sigmoid(logits)
            residual = (1.0 - gate) * source_flat + gate * posthoc_flat
            final_raw = (args.train_beta * residual - y_flat).abs()
            final_loss = (final_raw * sampled_weights).sum() / sampled_weights.sum().clamp_min(1.0)
            selector_raw = nn.functional.binary_cross_entropy_with_logits(logits, sampled_target, reduction="none")
            selector_loss = (selector_raw * sampled_weights).sum() / sampled_weights.sum().clamp_min(1.0)
            source_prior = (gate * sampled_weights).sum() / sampled_weights.sum().clamp_min(1.0)
            loss = (
                args.final_loss_weight * final_loss
                + args.selector_loss_weight * selector_loss
                + args.source_prior_weight * source_prior
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = float(sampled_target.numel())
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["final_loss"] += float(final_loss.detach().cpu()) * count
            totals["selector_loss"] += float(selector_loss.detach().cpu()) * count
            totals["source_prior"] += float(source_prior.detach().cpu()) * count
            totals["count"] += count
            totals["target"] += float(sampled_target.sum().detach().cpu())
            if batch_idx % 20 == 0:
                print(f"epoch {epoch}: trained {min(batch_idx * args.batch_size, train_idx.size)}/{train_idx.size}", flush=True)
        val_diag = selector_alignment(
            selector,
            source_model,
            posthoc_detector,
            cache_path,
            stats,
            val_idx,
            dual_hist,
            args,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(totals["count"], 1.0),
            "train_final_loss": totals["final_loss"] / max(totals["count"], 1.0),
            "train_selector_loss": totals["selector_loss"] / max(totals["count"], 1.0),
            "train_source_prior": totals["source_prior"] / max(totals["count"], 1.0),
            "train_posthoc_better_rate": totals["target"] / max(totals["count"], 1.0),
            **{f"val_{key}": value for key, value in val_diag.items()},
        }
        rows.append(row)
        print(
            f"epoch {epoch}: val_gate={row['val_gate_mean']:.6f}, "
            f"val_target={row['val_posthoc_better_rate']:.6f}, val_auc={row['val_auc']:.6f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def selector_alignment(
    selector: SourcePosthocSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    indices: np.ndarray,
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(cache_path, indices, stats, args.eval_batch_size, shuffle=False)
    score_sum = 0.0
    pos_score_sum = 0.0
    neg_score_sum = 0.0
    count = 0.0
    pos = 0.0
    hist_pos = np.zeros(400, dtype=np.float64)
    hist_neg = np.zeros(400, dtype=np.float64)
    selector.eval()
    with torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(
                batch,
                source_model,
                posthoc_detector,
                dual_hist,
                args.posthoc_scale,
                args.posthoc_temperature,
                device,
            )
            source_abs = (args.train_beta * prepared["source_residual"] - prepared["y"]).abs()
            posthoc_abs = (args.train_beta * prepared["posthoc_residual"] - prepared["y"]).abs()
            target = (posthoc_abs + args.selector_target_margin < source_abs)
            logits = selector(prepared["features"].reshape(-1, prepared["features"].shape[-1])).reshape(prepared["y"].shape)
            gate = torch.sigmoid(logits)
            mask = prepared["y_mask"]
            s = gate[mask].detach().cpu().numpy()
            p = target[mask].detach().cpu().numpy().astype(bool)
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
        "gate_mean": score_sum / max(count, 1.0),
        "posthoc_better_rate": pos / max(count, 1.0),
        "gate_pos_mean": pos_score_sum / pos_safe,
        "gate_neg_mean": neg_score_sum / neg,
        "gate_pos_neg_gap": pos_score_sum / pos_safe - neg_score_sum / neg,
        "auc": auc,
    }


def empty_sums() -> dict[str, float]:
    out = {}
    for subset in SUBSETS:
        out[f"{subset}_model"] = 0.0
        out[f"{subset}_source"] = 0.0
        out[f"{subset}_posthoc"] = 0.0
        out[f"{subset}_base"] = 0.0
        out[f"{subset}_count"] = 0.0
        out[f"{subset}_gate"] = 0.0
    return out


def update_sums(
    sums: dict[str, float],
    residual: torch.Tensor,
    source_residual: torch.Tensor,
    posthoc_residual: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    source_beta: float,
    gate: torch.Tensor,
    y_mask: torch.Tensor,
    affected: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    masks = {
        "all": y_mask,
        "affected": y_mask & affected[:, None, :, None],
        "unaffected": y_mask & (~affected[:, None, :, None]) & valid[:, None, :, None],
    }
    model_abs = (beta * residual - y).abs()
    source_abs = (source_beta * source_residual - y).abs()
    posthoc_abs = (beta * posthoc_residual - y).abs()
    base_abs = y.abs()
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0.0:
            continue
        sums[f"{subset}_model"] += float(model_abs[mask].sum().detach().cpu())
        sums[f"{subset}_source"] += float(source_abs[mask].sum().detach().cpu())
        sums[f"{subset}_posthoc"] += float(posthoc_abs[mask].sum().detach().cpu())
        sums[f"{subset}_base"] += float(base_abs[mask].sum().detach().cpu())
        sums[f"{subset}_gate"] += float(gate[mask].sum().detach().cpu())
        sums[f"{subset}_count"] += count


def summarize(split: str, scale: float, temp: float, beta: float, group: str, sums: dict[str, float]) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "split": split,
        "selector_scale": scale,
        "selector_temperature": temp,
        "beta": beta,
        "group": group,
    }
    for subset in SUBSETS:
        count = max(sums[f"{subset}_count"], 1.0)
        row[f"{subset}_mae"] = sums[f"{subset}_model"] / count
        row[f"{subset}_source_mae"] = sums[f"{subset}_source"] / count
        row[f"{subset}_posthoc_mae"] = sums[f"{subset}_posthoc"] / count
        row[f"{subset}_baseline_mae"] = sums[f"{subset}_base"] / count
        row[f"{subset}_selector_gate_mean"] = sums[f"{subset}_gate"] / count
        row[f"{subset}_count"] = sums[f"{subset}_count"]
    return row


def evaluate_split(
    split: str,
    selector: SourcePosthocSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    indices: np.ndarray,
    severity_q: tuple[float, float],
    source_beta: float,
    dual_hist: bool,
    selector_scales: list[float],
    selector_temperatures: list[float],
    betas: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    loader = make_loader(cache_path, indices, stats, args.eval_batch_size, shuffle=False)
    keys = [(scale, temp, beta, group) for scale in selector_scales for temp in selector_temperatures for beta in betas for group in GROUPS]
    all_sums = {key: empty_sums() for key in keys}
    selector.eval()
    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            prepared = prepare_batch(
                batch,
                source_model,
                posthoc_detector,
                dual_hist,
                args.posthoc_scale,
                args.posthoc_temperature,
                device,
            )
            logits = selector(prepared["features"].reshape(-1, prepared["features"].shape[-1])).reshape(prepared["y"].shape)
            idx_np = prepared["idx"].numpy().astype(np.int64)
            raw_event = h5["event_aux"][idx_np].astype(np.float32)
            groups = group_masks(raw_event, severity_q)
            for scale in selector_scales:
                for temp in selector_temperatures:
                    gate = (scale * torch.sigmoid(logits / max(temp, 1e-6))).clamp(0.0, 1.0)
                    residual = (1.0 - gate) * prepared["source_residual"] + gate * prepared["posthoc_residual"]
                    for beta in betas:
                        for group in GROUPS:
                            sample_mask = torch.from_numpy(groups[group]).to(device=device, dtype=torch.bool)
                            y_mask = prepared["y_mask"] & sample_mask[:, None, None, None]
                            update_sums(
                                all_sums[(scale, temp, beta, group)],
                                residual,
                                prepared["source_residual"],
                                prepared["posthoc_residual"],
                                prepared["y"],
                                beta,
                                source_beta,
                                gate,
                                y_mask,
                                prepared["node_affected"],
                                prepared["node_valid"],
                            )
            if batch_idx % 20 == 0:
                print(f"{split}: evaluated {min(batch_idx * args.eval_batch_size, indices.size)}/{indices.size}", flush=True)
    return pd.DataFrame([summarize(split, scale, temp, beta, group, sums) for (scale, temp, beta, group), sums in all_sums.items()])


def select_group_aware(val_df: pd.DataFrame, all_tolerance: float) -> pd.Series:
    overall = val_df[val_df["group"] == "overall"].copy()
    best_all = float(overall["all_mae"].min())
    eligible = overall[overall["all_mae"] <= best_all + all_tolerance]
    if eligible.empty:
        eligible = overall
    scored = []
    for idx, row in eligible.iterrows():
        mask = (
            np.isclose(val_df["selector_scale"].astype(float), float(row["selector_scale"]))
            & np.isclose(val_df["selector_temperature"].astype(float), float(row["selector_temperature"]))
            & np.isclose(val_df["beta"].astype(float), float(row["beta"]))
        )
        sub = val_df[mask]
        sev = float(sub[sub["group"] == "severity_high"]["affected_mae"].iloc[0])
        rec = float(sub[sub["group"] == "recovery_long_ge90"]["affected_mae"].iloc[0])
        score = 0.5 * float(row["affected_mae"]) + sev + rec
        scored.append((score, idx))
    return eligible.loc[min(scored)[1]]


def matching_rows(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    return df[
        np.isclose(df["selector_scale"].astype(float), float(selected["selector_scale"]))
        & np.isclose(df["selector_temperature"].astype(float), float(selected["selector_temperature"]))
        & np.isclose(df["beta"].astype(float), float(selected["beta"]))
    ].copy()


def write_summary(
    output_dir: Path,
    train_log: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected: pd.Series,
) -> None:
    test_selected = matching_rows(test_df, selected)
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long", "severity_low", "recovery_short_lt30"]
    focus = test_selected[test_selected["group"].isin(focus_groups)][
        [
            "group",
            "all_mae",
            "affected_mae",
            "unaffected_mae",
            "affected_source_mae",
            "affected_posthoc_mae",
            "affected_selector_gate_mean",
        ]
    ].copy()
    focus["affected_delta_vs_source"] = focus["affected_mae"] - focus["affected_source_mae"]
    show_cols = [
        "selector_scale",
        "selector_temperature",
        "beta",
        "all_mae",
        "affected_mae",
        "unaffected_mae",
        "affected_source_mae",
        "affected_posthoc_mae",
        "affected_selector_gate_mean",
    ]
    lines = [
        "# Source/Posthoc Selector",
        "",
        "Trains a deployable selector gate between source group-aware residual and posthoc detector residual.",
        "",
        "## Validation-Selected Result",
        "",
        f"- selector_scale: `{float(selected['selector_scale']):.4g}`",
        f"- selector_temperature: `{float(selected['selector_temperature']):.4g}`",
        f"- beta: `{float(selected['beta']):.4g}`",
        f"- validation all / affected MAE: `{float(selected['all_mae']):.6f}` / `{float(selected['affected_mae']):.6f}`",
        "",
        "## Test Groups At Selected Config",
        "",
        focus.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top Validation Overall Affected",
        "",
        val_df[val_df["group"] == "overall"].sort_values("affected_mae")[show_cols].head(15).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Top Validation Severity-High Affected",
        "",
        val_df[val_df["group"] == "severity_high"].sort_values("affected_mae")[show_cols].head(15).to_markdown(
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


def compute_feature_dim(
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    indices: np.ndarray,
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    loader = make_loader(cache_path, indices[: min(4, indices.size)], stats, min(4, indices.size), shuffle=False)
    prepared = prepare_batch(
        next(iter(loader)),
        source_model,
        posthoc_detector,
        dual_hist,
        args.posthoc_scale,
        args.posthoc_temperature,
        device,
    )
    return int(prepared["features"].shape[-1])


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    source_dir = args.source_model_dir.resolve()
    detector_dir = args.detector_dir.resolve()
    ckpt = torch_load(source_dir / "model.pt")
    cache_path = resolve_cache_path(source_dir, ckpt)
    stats = compute_stats(cache_path)
    source_model = make_model(ckpt, cache_path, device)
    posthoc_detector = load_detector(detector_dir, device)
    source_beta = residual_beta(source_dir, ckpt)
    dual_hist = model_uses_dual_hist(source_model, hist_channels=3)
    splits = split_indices(cache_path)
    train_idx = cap_indices(splits["train"], args.max_train_samples, args.seed)
    val_idx = cap_indices(splits["val"], args.max_eval_samples, args.seed + 1)
    test_idx = cap_indices(splits["test"], args.max_eval_samples, args.seed + 2)
    val_severity_q = severity_quantiles(cache_path, val_idx)
    test_severity_q = severity_quantiles(cache_path, test_idx)
    feature_dim = compute_feature_dim(source_model, posthoc_detector, cache_path, stats, train_idx, dual_hist, args, device)
    selector = SourcePosthocSelector(feature_dim, args.hidden_dim, args.dropout).to(device)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source: {source_dir}", flush=True)
    print(f"posthoc detector: {detector_dir}", flush=True)
    print(f"source_beta: {source_beta}", flush=True)
    print(f"feature_dim: {feature_dim}", flush=True)
    print(f"train/val/test samples: {train_idx.size}/{val_idx.size}/{test_idx.size}", flush=True)

    train_log = train_selector(
        selector,
        source_model,
        posthoc_detector,
        cache_path,
        stats,
        train_idx,
        val_idx,
        dual_hist,
        args,
        device,
    )
    selector_scales = parse_float_list(args.sweep_selector_scales)
    selector_temperatures = parse_float_list(args.sweep_selector_temperatures)
    betas = parse_float_list(args.sweep_betas)
    val_df = evaluate_split(
        "val",
        selector,
        source_model,
        posthoc_detector,
        cache_path,
        stats,
        val_idx,
        val_severity_q,
        source_beta,
        dual_hist,
        selector_scales,
        selector_temperatures,
        betas,
        args,
        device,
    )
    test_df = evaluate_split(
        "test",
        selector,
        source_model,
        posthoc_detector,
        cache_path,
        stats,
        test_idx,
        test_severity_q,
        source_beta,
        dual_hist,
        selector_scales,
        selector_temperatures,
        betas,
        args,
        device,
    )
    selected = select_group_aware(val_df, args.all_val_tolerance)
    train_log.to_csv(output_dir / "training_log.csv", index=False)
    val_df.to_csv(output_dir / "val_selector_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_selector_sweep.csv", index=False)
    torch.save(
        {
            "selector_state_dict": selector.state_dict(),
            "feature_dim": feature_dim,
            "args": vars(args),
            "source_model_dir": str(source_dir),
            "detector_dir": str(detector_dir),
            "source_beta": source_beta,
        },
        output_dir / "selector.pt",
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "source_model_dir": str(source_dir),
                "detector_dir": str(detector_dir),
                "cache_path": str(cache_path),
                "source_beta": source_beta,
                "feature_dim": feature_dim,
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "test_samples": int(test_idx.size),
                "val_severity_quantiles": val_severity_q,
                "test_severity_quantiles": test_severity_q,
                "selected": {
                    "selector_scale": float(selected["selector_scale"]),
                    "selector_temperature": float(selected["selector_temperature"]),
                    "beta": float(selected["beta"]),
                },
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    write_summary(output_dir, train_log, val_df, test_df, selected)
    print(f"wrote selector outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
