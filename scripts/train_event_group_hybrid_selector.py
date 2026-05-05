#!/usr/bin/env python3
"""Train a sample-level event-group selector for source/posthoc hybrid inference."""

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
from sweep_hybrid_source_posthoc_detector import event_semantic, group_masks
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
        default=Path("outputs/impact_guided_next_stage/event_group_hybrid_selector_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--unsafe-weight", type=float, default=4.0)
    parser.add_argument("--posthoc-scales", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--posthoc-temperatures", default="1.0,2.0")
    parser.add_argument("--safe-thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--betas", default="1.05,1.1")
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


class EventGroupSelector(nn.Module):
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


def severity_quantiles(cache_path: Path, indices: np.ndarray) -> tuple[float, float]:
    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][indices].astype(np.float32)
    severity, _recovery, _spread = event_semantic(raw_event)
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def read_event_aux(h5: h5py.File, indices: np.ndarray) -> np.ndarray:
    order = np.argsort(indices)
    sorted_event = h5["event_aux"][indices[order]].astype(np.float32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_event[inverse]


def safe_targets(raw_event: np.ndarray, severity_q: tuple[float, float]) -> np.ndarray:
    severity, recovery, _spread = event_semantic(raw_event)
    _q33, q66 = severity_q
    return ((severity <= q66) & (recovery < 90.0)).astype(np.float32)


def empty_sums() -> dict[str, float]:
    sums = {}
    for subset in SUBSETS:
        sums[f"{subset}_model"] = 0.0
        sums[f"{subset}_source"] = 0.0
        sums[f"{subset}_posthoc"] = 0.0
        sums[f"{subset}_base"] = 0.0
        sums[f"{subset}_count"] = 0.0
        sums[f"{subset}_amount"] = 0.0
    return sums


def update_sums(
    sums: dict[str, float],
    residual: torch.Tensor,
    source_residual: torch.Tensor,
    posthoc_residual: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    source_beta: float,
    amount: torch.Tensor,
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
        sums[f"{subset}_count"] += count
        sums[f"{subset}_amount"] += float(amount[mask].sum().detach().cpu())


def summarize_sums(
    split: str,
    mode: str,
    scale: float,
    temperature: float,
    beta: float,
    group: str,
    sums: dict[str, float],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "split": split,
        "mode": mode,
        "posthoc_scale": scale,
        "posthoc_temperature": temperature,
        "beta": beta,
        "group": group,
    }
    for subset in SUBSETS:
        count = max(sums[f"{subset}_count"], 1.0)
        row[f"{subset}_mae"] = sums[f"{subset}_model"] / count
        row[f"{subset}_source_mae"] = sums[f"{subset}_source"] / count
        row[f"{subset}_posthoc_mae"] = sums[f"{subset}_posthoc"] / count
        row[f"{subset}_baseline_mae"] = sums[f"{subset}_base"] / count
        row[f"{subset}_amount_mean"] = sums[f"{subset}_amount"] / count
        row[f"{subset}_count"] = sums[f"{subset}_count"]
    return row


def masked_mean_max(values: torch.Tensor, mask: torch.Tensor, dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(values.dtype)
    while mask_f.dim() < values.dim():
        mask_f = mask_f.unsqueeze(-1)
    masked = values * mask_f
    denom = mask_f.sum(dim=dims).clamp_min(1.0)
    mean = masked.sum(dim=dims) / denom
    very_neg = torch.full_like(values, -1e6)
    maxv = torch.where(mask_f.bool(), values, very_neg).amax(dim=dims)
    maxv = torch.where(maxv < -1e5, torch.zeros_like(maxv), maxv)
    return mean, maxv


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
        _event_aux,
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

        valid_h = node_valid[:, None, :, None].expand_as(source_amount)
        source_mean, source_max = masked_mean_max(source_amount, valid_h, dims=(1, 2, 3))
        posthoc_mean, posthoc_max = masked_mean_max(posthoc_amount, valid_h, dims=(1, 2, 3))
        impact_valid = node_valid[:, None, :].expand_as(pred_impact)
        impact_mean, impact_max = masked_mean_max(pred_impact, impact_valid, dims=(1, 2))
        node_prob = torch.sigmoid(pred_node)
        node_mean, node_max = masked_mean_max(node_prob, node_valid, dims=(1,))
        residual_gap = (posthoc_residual - source_residual).abs()
        gap_mean, gap_max = masked_mean_max(residual_gap, valid_h, dims=(1, 2, 3))

        sample_features = torch.cat(
            [
                pred_event,
                torch.relu(pred_event),
                source_mean[:, None],
                source_max[:, None],
                posthoc_mean[:, None],
                posthoc_max[:, None],
                (posthoc_mean - source_mean)[:, None],
                gap_mean[:, None],
                gap_max[:, None],
                impact_mean[:, None],
                impact_max[:, None],
                node_mean[:, None],
                node_max[:, None],
                global_context,
            ],
            dim=-1,
        )
        sample_features = torch.nan_to_num(sample_features, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "features": sample_features,
        "source_residual": source_residual,
        "posthoc_residual": posthoc_residual,
        "source_pred": source_pred,
        "y": y,
        "y_mask": y_mask,
        "node_affected": node_affected,
        "node_valid": node_valid,
        "idx": idx,
    }


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    return float((ranks[labels_bool].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_selector(
    selector: EventGroupSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_severity_q: tuple[float, float],
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    with h5py.File(cache_path, "r") as h5:
        for epoch in range(1, args.epochs + 1):
            loader = make_loader(cache_path, train_idx, stats, args.batch_size, shuffle=True)
            selector.train()
            totals = {"loss": 0.0, "count": 0.0, "safe": 0.0}
            for batch_idx, batch in enumerate(loader, start=1):
                prepared = prepare_batch(
                    batch,
                    source_model,
                    posthoc_detector,
                    dual_hist,
                    args.posthoc_scales_for_train[0],
                    args.posthoc_temperatures_for_train[0],
                    device,
                )
                idx_np = prepared["idx"].numpy().astype(np.int64)
                target_np = safe_targets(read_event_aux(h5, idx_np), train_severity_q)
                target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
                logits = selector(prepared["features"])
                weights = torch.ones_like(target)
                weights = weights + (args.unsafe_weight - 1.0) * (1.0 - target)
                raw = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                loss = (raw * weights).sum() / weights.sum().clamp_min(1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = float(target.numel())
                totals["loss"] += float(loss.detach().cpu()) * count
                totals["count"] += count
                totals["safe"] += float(target.sum().detach().cpu())
                if batch_idx % 20 == 0:
                    print(f"epoch {epoch}: trained {min(batch_idx * args.batch_size, train_idx.size)}/{train_idx.size}", flush=True)
            val_diag = evaluate_classifier(
                selector,
                source_model,
                posthoc_detector,
                cache_path,
                stats,
                val_idx,
                train_severity_q,
                dual_hist,
                args,
                device,
            )
            row = {
                "epoch": epoch,
                "train_loss": totals["loss"] / max(totals["count"], 1.0),
                "train_safe_rate": totals["safe"] / max(totals["count"], 1.0),
                **{f"val_{key}": value for key, value in val_diag.items()},
            }
            rows.append(row)
            print(
                f"epoch {epoch}: val_auc={row['val_auc']:.6f}, val_safe_rate={row['val_safe_rate']:.6f}, "
                f"val_score_mean={row['val_score_mean']:.6f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def evaluate_classifier(
    selector: EventGroupSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    indices: np.ndarray,
    severity_q: tuple[float, float],
    dual_hist: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(cache_path, indices, stats, args.eval_batch_size, shuffle=False)
    scores = []
    labels = []
    selector.eval()
    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(
                batch,
                source_model,
                posthoc_detector,
                dual_hist,
                args.posthoc_scales_for_train[0],
                args.posthoc_temperatures_for_train[0],
                device,
            )
            score = torch.sigmoid(selector(prepared["features"])).detach().cpu().numpy()
            idx_np = prepared["idx"].numpy().astype(np.int64)
            target = safe_targets(read_event_aux(h5, idx_np), severity_q)
            scores.append(score)
            labels.append(target)
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    return {
        "auc": auc_score(s, y),
        "safe_rate": float(y.mean()),
        "score_mean": float(s.mean()),
        "score_safe_mean": float(s[y.astype(bool)].mean()) if y.astype(bool).any() else float("nan"),
        "score_unsafe_mean": float(s[~y.astype(bool)].mean()) if (~y.astype(bool)).any() else float("nan"),
    }


def evaluate_hybrid(
    split: str,
    selector: EventGroupSelector,
    source_model: torch.nn.Module,
    posthoc_detector: PosthocNormalBetterDetector,
    cache_path: Path,
    stats: CacheStats,
    indices: np.ndarray,
    severity_q: tuple[float, float],
    source_beta: float,
    dual_hist: bool,
    posthoc_scales: list[float],
    posthoc_temperatures: list[float],
    thresholds: list[float],
    betas: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    keys = [
        (scale, temp, threshold, beta, group)
        for scale in posthoc_scales
        for temp in posthoc_temperatures
        for threshold in thresholds
        for beta in betas
        for group in GROUPS
    ]
    all_sums = {key: empty_sums() for key in keys}
    loader = make_loader(cache_path, indices, stats, args.eval_batch_size, shuffle=False)
    selector.eval()
    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            # Run once for the sample selector, then reuse it when the swept
            # calibration matches the selector's feature calibration.
            selector_prepared = prepare_batch(
                batch,
                source_model,
                posthoc_detector,
                dual_hist,
                args.posthoc_scales_for_train[0],
                args.posthoc_temperatures_for_train[0],
                device,
            )
            safe_score = torch.sigmoid(selector(selector_prepared["features"]))
            idx_np = selector_prepared["idx"].numpy().astype(np.int64)
            raw_event = read_event_aux(h5, idx_np)
            groups = group_masks(raw_event, severity_q)
            for scale in posthoc_scales:
                for temp in posthoc_temperatures:
                    if np.isclose(scale, args.posthoc_scales_for_train[0]) and np.isclose(
                        temp,
                        args.posthoc_temperatures_for_train[0],
                    ):
                        prepared = selector_prepared
                    else:
                        prepared = prepare_batch(
                            batch,
                            source_model,
                            posthoc_detector,
                            dual_hist,
                            scale,
                            temp,
                            device,
                        )
                    for threshold in thresholds:
                        use_posthoc = safe_score >= threshold
                        residual = torch.where(
                            use_posthoc[:, None, None, None],
                            prepared["posthoc_residual"],
                            prepared["source_residual"],
                        )
                        gate = use_posthoc[:, None, None, None].to(prepared["y"].dtype).expand_as(prepared["y"])
                        for beta in betas:
                            for group in GROUPS:
                                sample_mask = torch.from_numpy(groups[group]).to(device=device, dtype=torch.bool)
                                y_mask = prepared["y_mask"] & sample_mask[:, None, None, None]
                                update_sums(
                                    all_sums[(scale, temp, threshold, beta, group)],
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
    rows = []
    for scale, temp, threshold, beta, group in keys:
        row = summarize_sums(split, "event_group_selector", scale, temp, beta, group, all_sums[(scale, temp, threshold, beta, group)])
        row["safe_threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def select_group_aware(val_df: pd.DataFrame, all_tolerance: float) -> pd.Series:
    overall = val_df[val_df["group"] == "overall"].copy()
    best_all = float(overall["all_mae"].min())
    eligible = overall[overall["all_mae"] <= best_all + all_tolerance]
    if eligible.empty:
        eligible = overall
    scores = []
    for idx, row in eligible.iterrows():
        mask = (
            np.isclose(val_df["posthoc_scale"].astype(float), float(row["posthoc_scale"]))
            & np.isclose(val_df["posthoc_temperature"].astype(float), float(row["posthoc_temperature"]))
            & np.isclose(val_df["safe_threshold"].astype(float), float(row["safe_threshold"]))
            & np.isclose(val_df["beta"].astype(float), float(row["beta"]))
        )
        sub = val_df[mask]
        sev = float(sub[sub["group"] == "severity_high"]["affected_mae"].iloc[0])
        rec = float(sub[sub["group"] == "recovery_long_ge90"]["affected_mae"].iloc[0])
        score = 0.5 * float(row["affected_mae"]) + sev + rec
        scores.append((score, idx))
    return eligible.loc[min(scores)[1]]


def matching_rows(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    return df[
        np.isclose(df["posthoc_scale"].astype(float), float(selected["posthoc_scale"]))
        & np.isclose(df["posthoc_temperature"].astype(float), float(selected["posthoc_temperature"]))
        & np.isclose(df["safe_threshold"].astype(float), float(selected["safe_threshold"]))
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
            "affected_amount_mean",
        ]
    ].copy()
    focus["affected_delta_vs_source"] = focus["affected_mae"] - focus["affected_source_mae"]
    show_cols = [
        "posthoc_scale",
        "posthoc_temperature",
        "safe_threshold",
        "beta",
        "all_mae",
        "affected_mae",
        "unaffected_mae",
        "affected_source_mae",
        "affected_posthoc_mae",
        "affected_amount_mean",
    ]
    lines = [
        "# Event-Group Hybrid Selector",
        "",
        "Trains a sample-level safe-to-posthoc classifier and uses it for source/posthoc hybrid inference.",
        "",
        "## Validation-Selected Result",
        "",
        f"- posthoc_scale: `{float(selected['posthoc_scale']):.4g}`",
        f"- posthoc_temperature: `{float(selected['posthoc_temperature']):.4g}`",
        f"- safe_threshold: `{float(selected['safe_threshold']):.4g}`",
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
        "## Training Log",
        "",
        train_log.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    posthoc_scales = parse_float_list(args.posthoc_scales)
    posthoc_temperatures = parse_float_list(args.posthoc_temperatures)
    # Store a nonzero feature calibration for selector training. If this uses
    # scale=0, the posthoc detector features become nearly invisible.
    args.posthoc_scales_for_train = [max(posthoc_scales)]
    args.posthoc_temperatures_for_train = [max(posthoc_temperatures)]
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
    source_model.eval()
    posthoc_detector = load_detector(detector_dir, device)
    source_beta = residual_beta(source_dir, ckpt)
    dual_hist = model_uses_dual_hist(source_model, hist_channels=3)
    splits = split_indices(cache_path)
    train_idx = cap_indices(splits["train"], args.max_train_samples, args.seed)
    val_idx = cap_indices(splits["val"], args.max_eval_samples, args.seed + 1)
    test_idx = cap_indices(splits["test"], args.max_eval_samples, args.seed + 2)
    train_severity_q = severity_quantiles(cache_path, train_idx)
    val_severity_q = severity_quantiles(cache_path, val_idx)
    test_severity_q = severity_quantiles(cache_path, test_idx)

    # Infer feature dimension.
    first_loader = make_loader(cache_path, train_idx[: min(4, train_idx.size)], stats, min(4, train_idx.size), False)
    first = prepare_batch(
        next(iter(first_loader)),
        source_model,
        posthoc_detector,
        dual_hist,
        args.posthoc_scales_for_train[0],
        args.posthoc_temperatures_for_train[0],
        device,
    )
    selector = EventGroupSelector(int(first["features"].shape[-1]), args.hidden_dim, args.dropout).to(device)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source: {source_dir}", flush=True)
    print(f"posthoc detector: {detector_dir}", flush=True)
    print(f"feature_dim: {first['features'].shape[-1]}", flush=True)
    print(f"train/val/test samples: {train_idx.size}/{val_idx.size}/{test_idx.size}", flush=True)

    train_log = train_selector(
        selector,
        source_model,
        posthoc_detector,
        cache_path,
        stats,
        train_idx,
        val_idx,
        train_severity_q,
        dual_hist,
        args,
        device,
    )
    thresholds = parse_float_list(args.safe_thresholds)
    betas = parse_float_list(args.betas)
    val_df = evaluate_hybrid(
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
        posthoc_scales,
        posthoc_temperatures,
        thresholds,
        betas,
        args,
        device,
    )
    test_df = evaluate_hybrid(
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
        posthoc_scales,
        posthoc_temperatures,
        thresholds,
        betas,
        args,
        device,
    )
    selected = select_group_aware(val_df, args.all_val_tolerance)
    train_log.to_csv(output_dir / "training_log.csv", index=False)
    val_df.to_csv(output_dir / "val_event_group_hybrid_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_event_group_hybrid_sweep.csv", index=False)
    torch.save(
        {
            "selector_state_dict": selector.state_dict(),
            "feature_dim": int(first["features"].shape[-1]),
            "args": vars(args),
            "source_model_dir": str(source_dir),
            "detector_dir": str(detector_dir),
            "source_beta": source_beta,
            "train_severity_quantiles": train_severity_q,
        },
        output_dir / "event_group_selector.pt",
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "source_model_dir": str(source_dir),
                "detector_dir": str(detector_dir),
                "cache_path": str(cache_path),
                "source_beta": source_beta,
                "feature_dim": int(first["features"].shape[-1]),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "test_samples": int(test_idx.size),
                "train_severity_quantiles": train_severity_q,
                "val_severity_quantiles": val_severity_q,
                "test_severity_quantiles": test_severity_q,
                "selected": {
                    "posthoc_scale": float(selected["posthoc_scale"]),
                    "posthoc_temperature": float(selected["posthoc_temperature"]),
                    "safe_threshold": float(selected["safe_threshold"]),
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
    print(f"wrote event-group hybrid selector outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
