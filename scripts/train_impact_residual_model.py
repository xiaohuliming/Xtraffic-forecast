#!/usr/bin/env python3
"""Train a first-pass incident impact residual forecaster.

The model is deliberately lightweight:

    final forecast = normal forecast + learned residual

The normal forecast is the same transparent baseline used in the previous
validation step: future normal slot baseline blended with the last observation.
The neural model only predicts normalized residuals on local incident nodes and
is auxiliary-supervised by event-level impact labels.

This is not the final paper model. It is a runnable scaffold to test whether
impact-aware residual learning is a promising direction before we invest in a
full spatiotemporal graph architecture.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_impact_labels import (
    build_baseline_valid_mask,
    build_matches,
    build_robust_baseline,
    load_incidents_2023,
    load_region_traffic,
    load_sensor_meta,
    region_specs,
)
from validate_forecast_error_against_impact import (
    fit_blend_alphas,
    parse_incident_ids,
    select_local_nodes,
)


CHANNELS = ("flow", "occupancy", "speed")


@dataclass
class SampleArrays:
    x: np.ndarray
    y_residual: np.ndarray
    y_mask: np.ndarray
    y_aux_raw: np.ndarray
    normal_pred: np.ndarray
    actual_future: np.ndarray
    future_scale: np.ndarray
    split: np.ndarray
    region: np.ndarray
    baseline_robust_mae: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("archive"))
    parser.add_argument(
        "--event-root",
        type=Path,
        default=Path("outputs/impact_labels_aggregated/region_area_sensor_window"),
    )
    parser.add_argument("--raw-label-dir", type=Path, default=Path("outputs/impact_labels"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_residual_model/first_pass"),
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["Alameda", "ContraCosta", "Orange"],
        help="Regions to include.",
    )
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--local-topk-nodes", type=int, default=5)
    parser.add_argument(
        "--sample-offsets",
        nargs="+",
        type=int,
        default=[0, 6, 12],
        help="Forecast starts at incident_start + each offset, in 5-minute steps.",
    )
    parser.add_argument("--candidate-pm-radius", type=float, default=5.0)
    parser.add_argument("--anchor-pm-radius", type=float, default=2.0)
    parser.add_argument("--baseline-mask-extra-steps", type=int, default=12)
    parser.add_argument("--min-baseline-count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap for faster smoke tests; 0 means no cap.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device.",
    )
    return parser.parse_args()


class ImpactResidualMLP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        aux_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Linear(hidden_dim, y_dim)
        self.aux_head = nn.Linear(hidden_dim, aux_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.residual_head(h), self.aux_head(h)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def split_name(sample_start: int, total_steps: int) -> str:
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)
    if sample_start < train_end:
        return "train"
    if sample_start < val_end:
        return "val"
    return "test"


def robust_mae(pred: np.ndarray, actual: np.ndarray, scale: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(pred - actual) / scale
    vals = err[np.isfinite(err)]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def make_time_features(times: pd.DatetimeIndex, sample_start: int) -> np.ndarray:
    ts = times[sample_start]
    slot = (ts.hour * 60 + ts.minute) / 1440.0
    dow = ts.dayofweek / 7.0
    return np.asarray(
        [
            math.sin(2 * math.pi * slot),
            math.cos(2 * math.pi * slot),
            math.sin(2 * math.pi * dow),
            math.cos(2 * math.pi * dow),
            float(ts.dayofweek >= 5),
        ],
        dtype=np.float32,
    )


def make_event_context(row: object, offset: int, horizon_steps: int) -> np.ndarray:
    direction = str(row.direction).upper()[:1]
    direction_flags = [
        float(direction == "N"),
        float(direction == "S"),
        float(direction == "E"),
        float(direction == "W"),
    ]
    type_text = f"{getattr(row, 'primary_type', '')} {getattr(row, 'primary_description', '')}".lower()
    type_flags = [
        float("hazard" in type_text or "1125" in type_text),
        float("noinj" in type_text or "no inj" in type_text),
        float("unkninj" in type_text or "unk inj" in type_text),
        float("1141" in type_text or "collision" in type_text),
    ]
    values = np.asarray(
        [
            float(getattr(row, "anchor_pm_dist", 0.0)),
            float(getattr(row, "candidate_nodes", 0.0)) / 30.0,
            float(getattr(row, "fwy", 0.0)) / 100.0,
            float(offset) / max(horizon_steps, 1),
            *direction_flags,
            *type_flags,
        ],
        dtype=np.float32,
    )
    values[~np.isfinite(values)] = 0.0
    return values


def build_node_lookup(raw_nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    cols = ["region_node_idx", "any_z_auc", "affected"]
    for incident_id, group in raw_nodes.groupby("incident_id", sort=False):
        lookup[str(incident_id)] = group[cols].to_numpy()
    return lookup


def pad_or_trim_nodes(node_idx: np.ndarray, anchor_idx: int, k: int) -> np.ndarray:
    unique = []
    seen = set()
    for value in list(node_idx) + [anchor_idx]:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            unique.append(ivalue)
    if not unique:
        unique = [int(anchor_idx)]
    while len(unique) < k:
        unique.append(unique[-1])
    return np.asarray(unique[:k], dtype=np.int32)


def build_region_samples(
    region_name: str,
    data_dir: Path,
    event_root: Path,
    raw_label_dir: Path,
    meta: pd.DataFrame,
    inc: pd.DataFrame,
    args: argparse.Namespace,
) -> SampleArrays:
    specs = region_specs()
    region = specs[region_name]
    region_meta = meta[(meta["County"] == region.county) & (meta["Type"] == "Mainline")].copy()
    region_meta = region_meta.reset_index(drop=True)
    region_node_idx = region_meta["node_idx"].to_numpy(dtype=np.int32)

    print(f"[{region_name}] loading traffic", flush=True)
    traffic, times = load_region_traffic(data_dir, region_node_idx)
    day_kind = (times.dayofweek.to_numpy() >= 5).astype(np.int8)
    tod = ((times.hour.to_numpy() * 60 + times.minute.to_numpy()) // 5).astype(np.int16)
    total_steps = traffic.shape[0]

    print(f"[{region_name}] fitting normal forecaster", flush=True)
    matches = build_matches(
        inc=inc,
        region_meta=region_meta,
        times=times,
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
        baseline_mask_extra_steps=args.baseline_mask_extra_steps,
    )
    baseline_valid = build_baseline_valid_mask(traffic.shape[:2], matches)
    train_valid = baseline_valid.copy()
    train_valid[int(total_steps * 0.70) :, :] = False
    baseline, scale, _ = build_robust_baseline(
        traffic=traffic,
        times=times,
        baseline_valid=train_valid,
        min_count=args.min_baseline_count,
    )
    alphas = fit_blend_alphas(
        traffic=traffic,
        times=times,
        train_valid=train_valid,
        baseline=baseline,
        input_steps=args.input_steps,
        horizon_steps=args.horizon_steps,
    )

    print(f"[{region_name}] building local incident samples", flush=True)
    events = pd.read_csv(event_root / region_name / "event_labels.csv")
    raw_nodes = pd.read_csv(raw_label_dir / region_name / "node_labels.csv")
    raw_nodes["incident_id"] = raw_nodes["incident_id"].astype(str)
    node_lookup = build_node_lookup(raw_nodes)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    y_mask_rows: list[np.ndarray] = []
    aux_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    actual_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    split_rows: list[str] = []
    region_rows: list[str] = []
    baseline_error_rows: list[float] = []

    for row in events.itertuples(index=False):
        incident_ids = parse_incident_ids(row.incident_ids)
        selected = select_local_nodes(
            incident_ids=incident_ids,
            node_lookup=node_lookup,
            anchor_region_idx=int(row.anchor_region_idx),
            topk=args.local_topk_nodes,
        )
        selected = pad_or_trim_nodes(selected, int(row.anchor_region_idx), args.local_topk_nodes)

        for offset in args.sample_offsets:
            sample_start = int(row.start_idx) + int(offset)
            input_start = sample_start - args.input_steps
            future_end = sample_start + args.horizon_steps
            if input_start < 0 or future_end > total_steps:
                continue

            input_idx = np.arange(input_start, sample_start, dtype=np.int32)
            future_idx = np.arange(sample_start, future_end, dtype=np.int32)

            hist = traffic[input_idx][:, selected, :]
            hist_base = baseline[day_kind[input_idx], tod[input_idx]][:, selected, :]
            hist_scale = scale[day_kind[input_idx], tod[input_idx]][:, selected, :]
            fut_base = baseline[day_kind[future_idx], tod[future_idx]][:, selected, :]
            fut_scale = scale[day_kind[future_idx], tod[future_idx]][:, selected, :]
            actual_future = traffic[future_idx][:, selected, :]
            last_obs = traffic[sample_start - 1, selected, :]

            normal_pred = np.empty_like(actual_future, dtype=np.float32)
            for h in range(args.horizon_steps):
                normal_pred[h] = fut_base[h] + alphas[h][None, :] * (last_obs - fut_base[h])

            with np.errstate(divide="ignore", invalid="ignore"):
                hist_res_z = (hist - hist_base) / hist_scale
                y_res_z = (actual_future - normal_pred) / fut_scale
            y_mask = np.isfinite(y_res_z)
            hist_res_z[~np.isfinite(hist_res_z)] = 0.0
            y_res_z[~np.isfinite(y_res_z)] = 0.0

            if not np.isfinite(actual_future).any() or not np.isfinite(normal_pred).any():
                continue
            if not y_mask.any():
                continue
            base_err = robust_mae(normal_pred, actual_future, fut_scale)
            if not np.isfinite(base_err):
                continue

            event_features = make_event_context(row, offset, args.horizon_steps)
            time_features = make_time_features(times, sample_start)

            x = np.concatenate(
                [
                    hist_res_z.reshape(-1),
                    event_features,
                    time_features,
                ]
            ).astype(np.float32)
            aux = np.asarray(
                [
                    float(np.log1p(max(row.severity_any_z_auc_topk, 0.0))),
                    float(row.recovery_time_min) / 180.0,
                    float(np.log1p(max(row.spread_nodes, 0.0))),
                ],
                dtype=np.float32,
            )
            aux[~np.isfinite(aux)] = 0.0

            x_rows.append(x)
            y_rows.append(y_res_z.reshape(-1).astype(np.float32))
            y_mask_rows.append(y_mask.reshape(-1).astype(np.float32))
            aux_rows.append(aux)
            normal_rows.append(normal_pred.astype(np.float32))
            actual_rows.append(actual_future.astype(np.float32))
            scale_rows.append(fut_scale.astype(np.float32))
            split_rows.append(split_name(sample_start, total_steps))
            region_rows.append(region_name)
            baseline_error_rows.append(base_err)

    return SampleArrays(
        x=np.asarray(x_rows, dtype=np.float32),
        y_residual=np.asarray(y_rows, dtype=np.float32),
        y_mask=np.asarray(y_mask_rows, dtype=np.float32),
        y_aux_raw=np.asarray(aux_rows, dtype=np.float32),
        normal_pred=np.asarray(normal_rows, dtype=np.float32),
        actual_future=np.asarray(actual_rows, dtype=np.float32),
        future_scale=np.asarray(scale_rows, dtype=np.float32),
        split=np.asarray(split_rows),
        region=np.asarray(region_rows),
        baseline_robust_mae=np.asarray(baseline_error_rows, dtype=np.float32),
    )


def concat_samples(parts: list[SampleArrays]) -> SampleArrays:
    return SampleArrays(
        x=np.concatenate([p.x for p in parts], axis=0),
        y_residual=np.concatenate([p.y_residual for p in parts], axis=0),
        y_mask=np.concatenate([p.y_mask for p in parts], axis=0),
        y_aux_raw=np.concatenate([p.y_aux_raw for p in parts], axis=0),
        normal_pred=np.concatenate([p.normal_pred for p in parts], axis=0),
        actual_future=np.concatenate([p.actual_future for p in parts], axis=0),
        future_scale=np.concatenate([p.future_scale for p in parts], axis=0),
        split=np.concatenate([p.split for p in parts], axis=0),
        region=np.concatenate([p.region for p in parts], axis=0),
        baseline_robust_mae=np.concatenate([p.baseline_robust_mae for p in parts], axis=0),
    )


def standardize_train_val_test(
    x: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    x_std = (x - mean) / std
    x_std[~np.isfinite(x_std)] = 0.0
    return x_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_aux(
    y: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y[train_mask].mean(axis=0)
    std = y[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    y_std = (y - mean) / std
    y_std[~np.isfinite(y_std)] = 0.0
    return y_std.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def cap_train_samples(mask: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0:
        return mask
    idx = np.flatnonzero(mask)
    if idx.size <= max_samples:
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(idx, size=max_samples, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[keep] = True
    return out


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    aux: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x[mask]),
        torch.from_numpy(y[mask]),
        torch.from_numpy(y_mask[mask]),
        torch.from_numpy(aux[mask]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_loader(
    model: ImpactResidualMLP,
    loader: DataLoader,
    device: torch.device,
    aux_weight: float,
) -> dict[str, float]:
    model.eval()
    total = 0
    residual_loss_sum = 0.0
    residual_mask_sum = 0.0
    aux_loss = 0.0
    aux_loss_fn = nn.SmoothL1Loss(reduction="sum")
    with torch.no_grad():
        for x, y, y_mask, aux in loader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            aux = aux.to(device)
            pred_y, pred_aux = model(x)
            batch = x.shape[0]
            res = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
            residual_loss_sum += float((res * y_mask).sum().item())
            residual_mask_sum += float(y_mask.sum().item())
            aux_loss += aux_loss_fn(pred_aux, aux).item() / aux.shape[1]
            total += batch
    if total == 0:
        return {"loss": float("nan"), "residual_loss": float("nan"), "aux_loss": float("nan")}
    residual_loss = residual_loss_sum / max(residual_mask_sum, 1.0)
    aux_loss /= total
    return {
        "loss": residual_loss + aux_weight * aux_loss,
        "residual_loss": residual_loss,
        "aux_loss": aux_loss,
    }


def predict_residuals(
    model: ImpactResidualMLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            pred_y, _ = model(xb)
            preds.append(pred_y.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_forecast_metrics(
    samples: SampleArrays,
    pred_residual_flat: np.ndarray,
    mask: np.ndarray,
    horizon_steps: int,
    local_topk_nodes: int,
    residual_beta: float,
) -> dict[str, float]:
    shape = (-1, horizon_steps, local_topk_nodes, len(CHANNELS))
    pred_residual = pred_residual_flat.reshape(shape)
    pred = samples.normal_pred[mask] + residual_beta * pred_residual * samples.future_scale[mask]
    actual = samples.actual_future[mask]
    scale = samples.future_scale[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        model_err = np.abs(pred - actual) / scale
        baseline_err = np.abs(samples.normal_pred[mask] - actual) / scale
    model_vals = model_err[np.isfinite(model_err)]
    baseline_vals = baseline_err[np.isfinite(baseline_err)]
    if model_vals.size == 0 or baseline_vals.size == 0:
        return {"model_robust_mae": float("nan"), "baseline_robust_mae": float("nan"), "improvement_pct": float("nan")}
    model_mae = float(model_vals.mean())
    baseline_mae = float(baseline_vals.mean())
    return {
        "model_robust_mae": model_mae,
        "baseline_robust_mae": baseline_mae,
        "improvement_pct": 100.0 * (baseline_mae - model_mae) / baseline_mae,
    }


def select_residual_beta(
    samples: SampleArrays,
    pred_residual_flat: np.ndarray,
    val_mask: np.ndarray,
    horizon_steps: int,
    local_topk_nodes: int,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for beta in np.linspace(0.0, 1.0, 21):
        metrics = compute_forecast_metrics(
            samples=samples,
            pred_residual_flat=pred_residual_flat[val_mask],
            mask=val_mask,
            horizon_steps=horizon_steps,
            local_topk_nodes=local_topk_nodes,
            residual_beta=float(beta),
        )
        rows.append({"residual_beta": float(beta), **metrics})
    df = pd.DataFrame(rows)
    idx = df["model_robust_mae"].idxmin()
    return float(df.loc[idx, "residual_beta"]), df


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Impact residual model training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    samples: SampleArrays,
    log_df: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
    region_metrics: pd.DataFrame,
    residual_beta: float,
) -> None:
    lines = ["# 第一版事故影响残差模型", ""]
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 区域: {', '.join(args.regions)}")
    lines.append(f"- 历史输入步数 input_steps: {args.input_steps}")
    lines.append(f"- 未来预测步数 horizon_steps: {args.horizon_steps}")
    lines.append(f"- 局部评估节点数 local_topk_nodes: {args.local_topk_nodes}")
    lines.append(f"- 事故内采样偏移 sample_offsets: {args.sample_offsets}")
    lines.append(f"- 辅助监督权重 aux_weight: {args.aux_weight}")
    lines.append(f"- 残差修正系数 residual_beta: {residual_beta:.2f}")
    lines.append("")

    split_counts = pd.Series(samples.split).value_counts().rename_axis("split").reset_index(name="samples")
    lines.append("## 样本数量")
    lines.append("")
    lines.append(split_counts.to_markdown(index=False))
    lines.append("")

    metric_df = pd.DataFrame(
        [
            {"split": split, **values}
            for split, values in metrics.items()
        ]
    )
    lines.append("## 预测指标")
    lines.append("")
    lines.append(metric_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if not region_metrics.empty:
        lines.append("## 各地区测试集指标")
        lines.append("")
        lines.append(region_metrics.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    best = log_df.loc[log_df["val_loss"].idxmin()]
    lines.append("## 训练情况")
    lines.append("")
    lines.append(f"- 最佳轮数 best_epoch: {int(best['epoch'])}")
    lines.append(f"- 最佳验证损失 best_val_loss: {best['val_loss']:.4f}")
    lines.append("")

    lines.append("## 重要说明")
    lines.append("")
    lines.append(
        "这一版实验是在 derived node labels 中选出的 top-k impacted local nodes 上评估。"
        "也就是说，对每个事故事件，我们先在同一路段、同方向、事故点附近的候选传感器中，"
        "根据已经构造好的节点级影响强度 `any_z_auc` 选出影响最明显的前 k 个传感器，"
        "然后只在这些局部受影响节点上计算预测误差。"
    )
    lines.append("")
    lines.append(
        "这样做适合第一阶段验证 residual learning 是否有潜力，因为它直接聚焦事故真正影响到的位置。"
        "但这不是最终论文模型应该采用的推理方式，因为测试时用标签来选节点会带来 "
        "label-informed node selection 的风险。最终版本应该改为对候选节点集合或 "
        "anchor-neighborhood 做统一监督，让模型自己学习哪些节点会受到影响。"
    )
    lines.append("")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    event_root = args.event_root.resolve()
    raw_label_dir = args.raw_label_dir.resolve()

    print(f"using device: {device}", flush=True)
    meta = load_sensor_meta(data_dir)
    inc = load_incidents_2023(data_dir)

    parts = []
    for region_name in args.regions:
        parts.append(
            build_region_samples(
                region_name=region_name,
                data_dir=data_dir,
                event_root=event_root,
                raw_label_dir=raw_label_dir,
                meta=meta,
                inc=inc,
                args=args,
            )
        )
    samples = concat_samples(parts)
    if samples.x.size == 0:
        raise RuntimeError("No training samples were built.")

    train_mask_full = samples.split == "train"
    val_mask = samples.split == "val"
    test_mask = samples.split == "test"
    train_mask = cap_train_samples(train_mask_full, args.max_train_samples, args.seed)

    x_std, x_mean, x_stddev = standardize_train_val_test(samples.x, train_mask)
    aux_std, aux_mean, aux_stddev = standardize_aux(samples.y_aux_raw, train_mask)

    train_loader = make_loader(
        x=x_std,
        y=samples.y_residual,
        y_mask=samples.y_mask,
        aux=aux_std,
        mask=train_mask,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        x=x_std,
        y=samples.y_residual,
        y_mask=samples.y_mask,
        aux=aux_std,
        mask=val_mask,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ImpactResidualMLP(
        x_dim=x_std.shape[1],
        y_dim=samples.y_residual.shape[1],
        aux_dim=aux_std.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    aux_loss_fn = nn.SmoothL1Loss()

    log_rows = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for x, y, y_mask, aux in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            aux = aux.to(device)
            optim.zero_grad(set_to_none=True)
            pred_y, pred_aux = model(x)
            res = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
            residual_loss = (res * y_mask).sum() / y_mask.sum().clamp_min(1.0)
            aux_loss = aux_loss_fn(pred_aux, aux)
            loss = residual_loss + args.aux_weight * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.item())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(model, val_loader, device, args.aux_weight)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_residual_loss": val_metrics["residual_loss"],
                "val_aux_loss": val_metrics["aux_loss"],
            }
        )
        print(
            f"epoch {epoch:03d} train={train_loss:.4f} val={val_metrics['loss']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    metrics: dict[str, dict[str, float]] = {}
    pred_all = predict_residuals(model, x_std, args.batch_size, device)
    residual_beta, beta_df = select_residual_beta(
        samples=samples,
        pred_residual_flat=pred_all,
        val_mask=val_mask,
        horizon_steps=args.horizon_steps,
        local_topk_nodes=args.local_topk_nodes,
    )
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)

    for name, mask in [("train", train_mask_full), ("val", val_mask), ("test", test_mask)]:
        metrics[name] = compute_forecast_metrics(
            samples=samples,
            pred_residual_flat=pred_all[mask],
            mask=mask,
            horizon_steps=args.horizon_steps,
            local_topk_nodes=args.local_topk_nodes,
            residual_beta=residual_beta,
        )

    region_rows = []
    for region_name in args.regions:
        mask = test_mask & (samples.region == region_name)
        if not np.any(mask):
            continue
        row = {"region": region_name, "samples": int(mask.sum())}
        row.update(
            compute_forecast_metrics(
                samples=samples,
                pred_residual_flat=pred_all[mask],
                mask=mask,
                horizon_steps=args.horizon_steps,
                local_topk_nodes=args.local_topk_nodes,
                residual_beta=residual_beta,
            )
        )
        region_rows.append(row)
    region_metrics = pd.DataFrame(region_rows)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_mean": x_mean,
            "x_std": x_stddev,
            "aux_mean": aux_mean,
            "aux_std": aux_stddev,
            "args": json_safe_args(args),
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "region_metrics": region_rows,
                "samples": pd.Series(samples.split).value_counts().to_dict(),
                "residual_beta": residual_beta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["device"] = str(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_report(
        output_dir=output_dir,
        args=args,
        samples=samples,
        log_df=log_df,
        metrics=metrics,
        region_metrics=region_metrics,
        residual_beta=residual_beta,
    )
    print(f"wrote model outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
