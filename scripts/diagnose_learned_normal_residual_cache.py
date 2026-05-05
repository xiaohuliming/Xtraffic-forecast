#!/usr/bin/env python3
"""Diagnose how a learned-normal residual cache differs from the old cache."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ("flow", "occupancy", "speed")
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}
REGION_NAMES = {0: "Alameda", 1: "ContraCosta", 2: "Orange"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-cache",
        type=Path,
        default=Path("outputs/full_candidate_stgnn_heatmap_model/first_pass/full_candidate_samples.h5"),
        help="Residual cache built with the statistical blend normal baseline.",
    )
    parser.add_argument(
        "--new-cache",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal/full_candidate_samples.h5"),
        help="Residual cache rebuilt with the learned normal STGNN baseline.",
    )
    parser.add_argument(
        "--old-metrics",
        type=Path,
        default=Path("outputs/full_candidate_stgnn_heatmap_model/ablation_sigma_3_00_undirected/metrics.json"),
    )
    parser.add_argument(
        "--new-metrics",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal/metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/learned_normal_residual_diagnostics"),
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--sign-eps",
        type=float,
        default=0.05,
        help="Ignore tiny residuals when estimating sign-flip rate.",
    )
    return parser.parse_args()


def finite_percent(num: float, denom: float) -> float:
    if denom <= 0 or not math.isfinite(num) or not math.isfinite(denom):
        return float("nan")
    return 100.0 * num / denom


def reduction_pct(old_value: float, new_value: float) -> float:
    if old_value <= 0 or not math.isfinite(old_value) or not math.isfinite(new_value):
        return float("nan")
    return 100.0 * (old_value - new_value) / old_value


def make_bins() -> np.ndarray:
    fine = np.linspace(0.0, 5.0, 101)
    tail = np.linspace(5.1, 20.0, 150)
    return np.concatenate([fine, tail, [np.inf]]).astype(np.float64)


@dataclass
class StreamingResidualStats:
    bins: np.ndarray = field(default_factory=make_bins)
    count: int = 0
    old_sum: float = 0.0
    new_sum: float = 0.0
    old_sq_sum: float = 0.0
    new_sq_sum: float = 0.0
    old_signed_sum: float = 0.0
    new_signed_sum: float = 0.0
    delta_abs_sum: float = 0.0
    new_lower_count: int = 0
    sign_flip_count: int = 0
    old_hist: np.ndarray = field(init=False)
    new_hist: np.ndarray = field(init=False)
    delta_hist: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        hist_len = len(self.bins) - 1
        self.old_hist = np.zeros(hist_len, dtype=np.int64)
        self.new_hist = np.zeros(hist_len, dtype=np.int64)
        self.delta_hist = np.zeros(hist_len, dtype=np.int64)

    def update(self, old_values: np.ndarray, new_values: np.ndarray, sign_eps: float) -> None:
        old = np.asarray(old_values, dtype=np.float64).reshape(-1)
        new = np.asarray(new_values, dtype=np.float64).reshape(-1)
        finite = np.isfinite(old) & np.isfinite(new)
        if not finite.any():
            return
        old = old[finite]
        new = new[finite]
        old_abs = np.abs(old)
        new_abs = np.abs(new)
        delta_abs = np.abs(old - new)
        n = int(old_abs.size)
        self.count += n
        self.old_sum += float(old_abs.sum())
        self.new_sum += float(new_abs.sum())
        self.old_sq_sum += float(np.square(old).sum())
        self.new_sq_sum += float(np.square(new).sum())
        self.old_signed_sum += float(old.sum())
        self.new_signed_sum += float(new.sum())
        self.delta_abs_sum += float(delta_abs.sum())
        self.new_lower_count += int((new_abs < old_abs).sum())
        active = (old_abs > sign_eps) & (new_abs > sign_eps)
        self.sign_flip_count += int(((np.sign(old) != np.sign(new)) & active).sum())
        self.old_hist += np.histogram(old_abs, bins=self.bins)[0]
        self.new_hist += np.histogram(new_abs, bins=self.bins)[0]
        self.delta_hist += np.histogram(delta_abs, bins=self.bins)[0]

    def _quantile(self, hist: np.ndarray, q: float) -> float:
        if self.count <= 0:
            return float("nan")
        target = self.count * q
        cdf = np.cumsum(hist)
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(idx, len(self.bins) - 2)
        edge = float(self.bins[idx + 1])
        if not math.isfinite(edge):
            edge = float(self.bins[-2])
        return edge

    def row(self, labels: dict[str, object]) -> dict[str, object]:
        c = max(self.count, 1)
        old_abs_mean = self.old_sum / c
        new_abs_mean = self.new_sum / c
        return {
            **labels,
            "count": self.count,
            "old_abs_mean": old_abs_mean,
            "new_abs_mean": new_abs_mean,
            "target_reduction_pct": reduction_pct(old_abs_mean, new_abs_mean),
            "old_rms": math.sqrt(self.old_sq_sum / c),
            "new_rms": math.sqrt(self.new_sq_sum / c),
            "old_signed_mean": self.old_signed_sum / c,
            "new_signed_mean": self.new_signed_sum / c,
            "normal_delta_abs_mean": self.delta_abs_sum / c,
            "normal_delta_to_old_abs_pct": finite_percent(self.delta_abs_sum, self.old_sum),
            "new_abs_lower_rate_pct": finite_percent(float(self.new_lower_count), float(self.count)),
            "sign_flip_rate_pct": finite_percent(float(self.sign_flip_count), float(self.count)),
            "old_abs_p50": self._quantile(self.old_hist, 0.50),
            "new_abs_p50": self._quantile(self.new_hist, 0.50),
            "old_abs_p90": self._quantile(self.old_hist, 0.90),
            "new_abs_p90": self._quantile(self.new_hist, 0.90),
            "old_abs_p95": self._quantile(self.old_hist, 0.95),
            "new_abs_p95": self._quantile(self.new_hist, 0.95),
            "old_abs_p99": self._quantile(self.old_hist, 0.99),
            "new_abs_p99": self._quantile(self.new_hist, 0.99),
            "normal_delta_abs_p90": self._quantile(self.delta_hist, 0.90),
        }


@dataclass
class MeanAccumulator:
    count: int = 0
    sum_value: float = 0.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum_value += float(arr.sum())

    @property
    def mean(self) -> float:
        return self.sum_value / max(self.count, 1)


def update_residual(
    accs: dict[tuple[object, ...], StreamingResidualStats],
    key: tuple[object, ...],
    old_y: np.ndarray,
    new_y: np.ndarray,
    mask: np.ndarray,
    sign_eps: float,
) -> None:
    if not mask.any():
        return
    accs[key].update(old_y[mask], new_y[mask], sign_eps=sign_eps)


def iter_slices(n: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n, chunk_size):
        yield slice(start, min(start + chunk_size, n))


def consistency_checks(old: h5py.File, new: h5py.File, chunk_size: int) -> dict[str, object]:
    n = int(old.attrs["samples"])
    out = {
        "old_samples": n,
        "new_samples": int(new.attrs["samples"]),
        "same_sample_count": n == int(new.attrs["samples"]),
        "split_mismatch_count": 0,
        "region_mismatch_count": 0,
        "node_valid_mismatch_count": 0,
        "node_affected_mismatch_count": 0,
        "hist_residual_max_abs_diff": 0.0,
        "hist_residual_mean_abs_diff": 0.0,
    }
    hist_diff_sum = 0.0
    hist_diff_count = 0
    for slc in iter_slices(n, chunk_size):
        old_split = old["split"][slc]
        new_split = new["split"][slc]
        out["split_mismatch_count"] += int((old_split != new_split).sum())
        out["region_mismatch_count"] += int((old["region_code"][slc] != new["region_code"][slc]).sum())
        out["node_valid_mismatch_count"] += int((old["node_valid"][slc] != new["node_valid"][slc]).sum())
        out["node_affected_mismatch_count"] += int((old["node_affected"][slc] != new["node_affected"][slc]).sum())
        diff = np.abs(old["hist_residual"][slc].astype(np.float32) - new["hist_residual"][slc].astype(np.float32))
        finite = np.isfinite(diff)
        if finite.any():
            out["hist_residual_max_abs_diff"] = max(out["hist_residual_max_abs_diff"], float(diff[finite].max()))
            hist_diff_sum += float(diff[finite].sum())
            hist_diff_count += int(finite.sum())
    out["hist_residual_mean_abs_diff"] = hist_diff_sum / max(hist_diff_count, 1)
    return out


def add_input_alignment(
    accs: dict[tuple[object, ...], MeanAccumulator],
    old_h5: h5py.File,
    new_h5: h5py.File,
    slc: slice,
    split: np.ndarray,
    node_valid: np.ndarray,
    node_affected: np.ndarray,
    old_y: np.ndarray,
    new_y: np.ndarray,
    valid: np.ndarray,
) -> None:
    old_hist_last = old_h5["hist_residual"][slc][:, -1, :, :].astype(np.float32)
    new_hist_last = new_h5["hist_residual"][slc][:, -1, :, :].astype(np.float32)
    hist_valid = np.isfinite(old_hist_last) & np.isfinite(new_hist_last) & node_valid[:, :, None].astype(bool)
    old_step1 = old_y[:, 0, :, :]
    new_step1 = new_y[:, 0, :, :]
    step_valid = valid[:, 0, :, :]

    for split_code, split_name in SPLIT_NAMES.items():
        sample_mask = split == split_code
        if not sample_mask.any():
            continue
        classes = {
            "all": node_valid.astype(bool),
            "affected": node_affected.astype(bool),
            "unaffected": node_valid.astype(bool) & (~node_affected.astype(bool)),
        }
        for node_class, class_mask in classes.items():
            node_mask = sample_mask[:, None, None] & class_mask[:, :, None]
            hist_mask = hist_valid & node_mask
            step_mask = step_valid & node_mask
            accs[(split_name, node_class, "hist_last_abs")].update(np.abs(old_hist_last[hist_mask]))
            accs[(split_name, node_class, "new_hist_last_abs")].update(np.abs(new_hist_last[hist_mask]))
            accs[(split_name, node_class, "old_y_step1_abs")].update(np.abs(old_step1[step_mask]))
            accs[(split_name, node_class, "new_y_step1_abs")].update(np.abs(new_step1[step_mask]))


def make_rows(
    accs: dict[tuple[object, ...], StreamingResidualStats],
    label_names: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for key, acc in sorted(accs.items()):
        labels = {name: value for name, value in zip(label_names, key)}
        rows.append(acc.row(labels))
    return pd.DataFrame(rows)


def load_metrics(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_comparison(old_metrics: dict[str, object] | None, new_metrics: dict[str, object] | None) -> pd.DataFrame:
    if not old_metrics or not new_metrics:
        return pd.DataFrame()
    rows = []
    metric_names = [
        "all_candidates_baseline_robust_mae",
        "all_candidates_model_robust_mae",
        "affected_candidates_baseline_robust_mae",
        "affected_candidates_model_robust_mae",
        "unaffected_candidates_baseline_robust_mae",
        "unaffected_candidates_model_robust_mae",
    ]
    for split in ("train", "val", "test"):
        old_split = old_metrics["metrics"][split]  # type: ignore[index]
        new_split = new_metrics["metrics"][split]  # type: ignore[index]
        for name in metric_names:
            old_value = float(old_split[name])
            new_value = float(new_split[name])
            rows.append(
                {
                    "split": split,
                    "metric": name,
                    "old_statistical_normal": old_value,
                    "new_learned_normal": new_value,
                    "new_vs_old_reduction_pct": reduction_pct(old_value, new_value),
                }
            )
    return pd.DataFrame(rows)


def save_plots(output_dir: Path, by_region: pd.DataFrame, by_horizon: pd.DataFrame, by_channel: pd.DataFrame) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    region_plot = by_region[(by_region["split"] == "test") & (by_region["node_class"] == "all")].copy()
    if not region_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(region_plot))
        width = 0.35
        ax.bar(x - width / 2, region_plot["old_abs_mean"], width, label="statistical normal")
        ax.bar(x + width / 2, region_plot["new_abs_mean"], width, label="learned normal")
        ax.set_xticks(x, region_plot["region"])
        ax.set_ylabel("mean |future residual|")
        ax.set_title("Test residual target magnitude by region")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "test_region_target_magnitude.png", dpi=180)
        plt.close(fig)

    horizon_plot = by_horizon[(by_horizon["split"] == "test") & (by_horizon["node_class"].isin(["all", "affected"]))].copy()
    if not horizon_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for node_class, sub in horizon_plot.groupby("node_class"):
            sub = sub.sort_values("horizon")
            ax.plot(sub["horizon"], sub["old_abs_mean"], marker="o", label=f"old {node_class}")
            ax.plot(sub["horizon"], sub["new_abs_mean"], marker="o", linestyle="--", label=f"new {node_class}")
        ax.set_xlabel("horizon step")
        ax.set_ylabel("mean |future residual|")
        ax.set_title("Residual target magnitude across horizons")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "test_horizon_target_magnitude.png", dpi=180)
        plt.close(fig)

    channel_plot = by_channel[(by_channel["split"] == "test") & (by_channel["node_class"] == "all")].copy()
    if not channel_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(channel_plot))
        width = 0.35
        ax.bar(x - width / 2, channel_plot["old_abs_mean"], width, label="statistical normal")
        ax.bar(x + width / 2, channel_plot["new_abs_mean"], width, label="learned normal")
        ax.set_xticks(x, channel_plot["channel"])
        ax.set_ylabel("mean |future residual|")
        ax.set_title("Test residual target magnitude by channel")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "test_channel_target_magnitude.png", dpi=180)
        plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    checks: dict[str, object],
    by_split: pd.DataFrame,
    by_region: pd.DataFrame,
    by_channel: pd.DataFrame,
    by_horizon: pd.DataFrame,
    input_alignment: pd.DataFrame,
    metric_cmp: pd.DataFrame,
) -> None:
    test_all = by_split[(by_split["split"] == "test") & (by_split["node_class"] == "all")].iloc[0]
    test_aff = by_split[(by_split["split"] == "test") & (by_split["node_class"] == "affected")].iloc[0]
    test_unaff = by_split[(by_split["split"] == "test") & (by_split["node_class"] == "unaffected")].iloc[0]
    metric_test = metric_cmp[metric_cmp["split"] == "test"] if not metric_cmp.empty else pd.DataFrame()

    lines = [
        "# Learned Normal Residual Cache Diagnostic",
        "",
        "## 一句话结论",
        "",
        (
            "learned normal cache 确实降低了事故窗口 residual target 的整体幅度，"
            f"test 全候选 mean |residual| 从 {test_all['old_abs_mean']:.4f} 降到 {test_all['new_abs_mean']:.4f}，"
            f"下降 {test_all['target_reduction_pct']:.2f}%。"
        ),
        "",
        (
            "但历史输入 `hist_residual` 与旧 cache 完全一致，说明当前 residual branch 的输入仍是统计 baseline residual，"
            "而预测目标已经变成 learned-normal residual；这支持“输入/目标不对齐”这个诊断。"
        ),
        "",
        "## Cache 一致性检查",
        "",
        pd.DataFrame([checks]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Test Split 关键分布",
        "",
        by_split[by_split["split"] == "test"][
            [
                "split",
                "node_class",
                "count",
                "old_abs_mean",
                "new_abs_mean",
                "target_reduction_pct",
                "normal_delta_abs_mean",
                "normal_delta_to_old_abs_pct",
                "new_abs_lower_rate_pct",
                "sign_flip_rate_pct",
                "old_abs_p95",
                "new_abs_p95",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Region 对比",
        "",
        by_region[(by_region["split"] == "test") & (by_region["node_class"].isin(["all", "affected"]))][
            [
                "region",
                "node_class",
                "old_abs_mean",
                "new_abs_mean",
                "target_reduction_pct",
                "normal_delta_abs_mean",
                "sign_flip_rate_pct",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Channel 对比",
        "",
        by_channel[(by_channel["split"] == "test") & (by_channel["node_class"].isin(["all", "affected"]))][
            [
                "channel",
                "node_class",
                "old_abs_mean",
                "new_abs_mean",
                "target_reduction_pct",
                "normal_delta_abs_mean",
                "sign_flip_rate_pct",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Horizon 对比",
        "",
        by_horizon[(by_horizon["split"] == "test") & (by_horizon["node_class"].isin(["all", "affected"]))][
            [
                "horizon",
                "node_class",
                "old_abs_mean",
                "new_abs_mean",
                "target_reduction_pct",
                "normal_delta_abs_mean",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 输入目标对齐",
        "",
        input_alignment[input_alignment["split"] == "test"].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if not metric_test.empty:
        lines.extend(
            [
                "## 模型指标对照",
                "",
                metric_test.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    lines.extend(
        [
            "## 解释",
            "",
            (
                f"- 全候选 target 下降 {test_all['target_reduction_pct']:.2f}%，"
                f"受影响节点下降 {test_aff['target_reduction_pct']:.2f}%，"
                f"非受影响节点下降 {test_unaff['target_reduction_pct']:.2f}%。"
            ),
            "- 这说明 learned normal 没有只修正常规节点；它也吸收了一部分事故相关偏差，使 residual target 更小。",
            "- 当前模型输入的历史 residual 仍然来自统计 baseline，因此模型看到的历史异常尺度和未来 target 的尺度不一致。",
            "- 下一步应把 learned-normal 信息显式加入 residual branch 输入，例如 `normal_delta = Y_normal_stgnn - Y_blend` 和 normal uncertainty proxy。",
            "",
            "## 输出文件",
            "",
            "- `by_split_nodeclass.csv`",
            "- `by_region_nodeclass.csv`",
            "- `by_channel_nodeclass.csv`",
            "- `by_horizon_nodeclass.csv`",
            "- `input_target_alignment.csv`",
            "- `model_metric_comparison.csv`",
            "- `plots/test_region_target_magnitude.png`",
            "- `plots/test_horizon_target_magnitude.png`",
            "- `plots/test_channel_target_magnitude.png`",
            "",
            "## Inputs",
            "",
            f"- old_cache: `{args.old_cache}`",
            f"- new_cache: `{args.new_cache}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[tuple[object, ...], StreamingResidualStats] = defaultdict(StreamingResidualStats)
    by_region: dict[tuple[object, ...], StreamingResidualStats] = defaultdict(StreamingResidualStats)
    by_channel: dict[tuple[object, ...], StreamingResidualStats] = defaultdict(StreamingResidualStats)
    by_horizon: dict[tuple[object, ...], StreamingResidualStats] = defaultdict(StreamingResidualStats)
    input_alignment_acc: dict[tuple[object, ...], MeanAccumulator] = defaultdict(MeanAccumulator)

    with h5py.File(args.old_cache, "r") as old_h5, h5py.File(args.new_cache, "r") as new_h5:
        n_old = int(old_h5.attrs["samples"])
        n_new = int(new_h5.attrs["samples"])
        if n_old != n_new:
            raise ValueError(f"cache sample count mismatch: {n_old} vs {n_new}")
        if old_h5["y_residual"].shape != new_h5["y_residual"].shape:
            raise ValueError(f"y_residual shape mismatch: {old_h5['y_residual'].shape} vs {new_h5['y_residual'].shape}")

        print("running consistency checks", flush=True)
        checks = consistency_checks(old_h5, new_h5, args.chunk_size)

        print("streaming residual diagnostics", flush=True)
        for idx, slc in enumerate(iter_slices(n_old, args.chunk_size), start=1):
            old_y = old_h5["y_residual"][slc].astype(np.float32)
            new_y = new_h5["y_residual"][slc].astype(np.float32)
            valid = old_h5["y_mask"][slc].astype(bool) & new_h5["y_mask"][slc].astype(bool)
            split = old_h5["split"][slc]
            region = old_h5["region_code"][slc]
            node_valid = old_h5["node_valid"][slc].astype(bool) & new_h5["node_valid"][slc].astype(bool)
            node_affected = old_h5["node_affected"][slc].astype(bool) & new_h5["node_affected"][slc].astype(bool)

            node_classes = {
                "all": node_valid,
                "affected": node_affected,
                "unaffected": node_valid & (~node_affected),
            }
            for split_code, split_name in SPLIT_NAMES.items():
                sample_mask = split == split_code
                if not sample_mask.any():
                    continue
                for node_class, class_mask in node_classes.items():
                    mask = valid & sample_mask[:, None, None, None] & class_mask[:, None, :, None]
                    update_residual(by_split, (split_name, node_class), old_y, new_y, mask, args.sign_eps)

            test_sample = split == 2
            if test_sample.any():
                for region_code, region_name in REGION_NAMES.items():
                    sample_mask = test_sample & (region == region_code)
                    if not sample_mask.any():
                        continue
                    for node_class, class_mask in node_classes.items():
                        mask = valid & sample_mask[:, None, None, None] & class_mask[:, None, :, None]
                        update_residual(by_region, ("test", region_name, node_class), old_y, new_y, mask, args.sign_eps)

                for c, channel in enumerate(CHANNELS):
                    for node_class, class_mask in node_classes.items():
                        mask = valid[:, :, :, c] & test_sample[:, None, None] & class_mask[:, None, :]
                        update_residual(
                            by_channel,
                            ("test", channel, node_class),
                            old_y[:, :, :, c],
                            new_y[:, :, :, c],
                            mask,
                            args.sign_eps,
                        )

                for h in range(old_y.shape[1]):
                    for node_class, class_mask in node_classes.items():
                        mask = valid[:, h, :, :] & test_sample[:, None, None] & class_mask[:, :, None]
                        update_residual(
                            by_horizon,
                            ("test", h + 1, node_class),
                            old_y[:, h, :, :],
                            new_y[:, h, :, :],
                            mask,
                            args.sign_eps,
                        )

            add_input_alignment(
                input_alignment_acc,
                old_h5,
                new_h5,
                slc,
                split,
                node_valid,
                node_affected,
                old_y,
                new_y,
                valid,
            )
            if idx % 25 == 0:
                print(f"processed {min(slc.stop, n_old)}/{n_old} samples", flush=True)

    by_split_df = make_rows(by_split, ("split", "node_class"))
    by_region_df = make_rows(by_region, ("split", "region", "node_class"))
    by_channel_df = make_rows(by_channel, ("split", "channel", "node_class"))
    by_horizon_df = make_rows(by_horizon, ("split", "horizon", "node_class"))

    alignment_rows = []
    for split_name in ("train", "val", "test"):
        for node_class in ("all", "affected", "unaffected"):
            hist_old = input_alignment_acc[(split_name, node_class, "hist_last_abs")]
            hist_new = input_alignment_acc[(split_name, node_class, "new_hist_last_abs")]
            old_step = input_alignment_acc[(split_name, node_class, "old_y_step1_abs")]
            new_step = input_alignment_acc[(split_name, node_class, "new_y_step1_abs")]
            alignment_rows.append(
                {
                    "split": split_name,
                    "node_class": node_class,
                    "hist_last_abs_mean": hist_old.mean,
                    "new_cache_hist_last_abs_mean": hist_new.mean,
                    "old_target_step1_abs_mean": old_step.mean,
                    "new_target_step1_abs_mean": new_step.mean,
                    "hist_to_new_step1_ratio": hist_old.mean / new_step.mean if new_step.mean > 0 else float("nan"),
                    "hist_cache_abs_diff": abs(hist_old.mean - hist_new.mean),
                }
            )
    input_alignment_df = pd.DataFrame(alignment_rows)

    metric_cmp_df = metrics_comparison(load_metrics(args.old_metrics), load_metrics(args.new_metrics))

    by_split_df.to_csv(args.output_dir / "by_split_nodeclass.csv", index=False)
    by_region_df.to_csv(args.output_dir / "by_region_nodeclass.csv", index=False)
    by_channel_df.to_csv(args.output_dir / "by_channel_nodeclass.csv", index=False)
    by_horizon_df.to_csv(args.output_dir / "by_horizon_nodeclass.csv", index=False)
    input_alignment_df.to_csv(args.output_dir / "input_target_alignment.csv", index=False)
    metric_cmp_df.to_csv(args.output_dir / "model_metric_comparison.csv", index=False)
    (args.output_dir / "consistency_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    save_plots(args.output_dir, by_region_df, by_horizon_df, by_channel_df)
    write_summary(
        output_dir=args.output_dir,
        args=args,
        checks=checks,
        by_split=by_split_df,
        by_region=by_region_df,
        by_channel=by_channel_df,
        by_horizon=by_horizon_df,
        input_alignment=input_alignment_df,
        metric_cmp=metric_cmp_df,
    )
    print(f"wrote diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
