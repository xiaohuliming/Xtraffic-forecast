#!/usr/bin/env python3
"""Diagnose whether normal-veto scores align with normal-better positions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from compare_dual_branch_group_metrics import residual_beta
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


SUBSETS = ("all", "affected", "unaffected")
EVENT_GROUPS = (
    "overall",
    "severity_low",
    "severity_mid",
    "severity_high",
    "recovery_short_lt30",
    "recovery_mid_30_90",
    "recovery_long_ge90",
    "severity_high_and_long",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid"),
            Path(
                "outputs/impact_guided_next_stage/"
                "dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid"
            ),
        ],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["element_normal_veto", "focused_impact_veto"],
        help="Optional display labels matching --model-dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--positive-margin", type=float, default=0.10)
    parser.add_argument("--score-bin-width", type=float, default=0.0025)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
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


def event_group_masks(event_aux: np.ndarray, severity_q: tuple[float, float]) -> dict[str, np.ndarray]:
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    q33, q66 = severity_q
    return {
        "overall": np.ones(event_aux.shape[0], dtype=bool),
        "severity_low": severity <= q33,
        "severity_mid": (severity > q33) & (severity <= q66),
        "severity_high": severity > q66,
        "recovery_short_lt30": recovery_min < 30.0,
        "recovery_mid_30_90": (recovery_min >= 30.0) & (recovery_min < 90.0),
        "recovery_long_ge90": recovery_min >= 90.0,
        "severity_high_and_long": (severity > q66) & (recovery_min >= 90.0),
    }


@dataclass
class RunningAlignment:
    bins: np.ndarray
    count: float = 0.0
    positive: float = 0.0
    score_sum: float = 0.0
    score_pos_sum: float = 0.0
    score_neg_sum: float = 0.0
    advantage_sum: float = 0.0
    final_gain_sum: float = 0.0
    final_better: float = 0.0
    hist_count: np.ndarray | None = None
    hist_positive: np.ndarray | None = None
    hist_final_gain: np.ndarray | None = None
    hist_final_better: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.bins.size - 1
        self.hist_count = np.zeros(n, dtype=np.float64)
        self.hist_positive = np.zeros(n, dtype=np.float64)
        self.hist_final_gain = np.zeros(n, dtype=np.float64)
        self.hist_final_better = np.zeros(n, dtype=np.float64)

    def update(self, score: np.ndarray, positive: np.ndarray, advantage: np.ndarray, final_gain: np.ndarray) -> None:
        if score.size == 0:
            return
        pos = positive.astype(bool)
        neg = ~pos
        self.count += float(score.size)
        self.positive += float(pos.sum())
        self.score_sum += float(score.sum())
        self.score_pos_sum += float(score[pos].sum()) if pos.any() else 0.0
        self.score_neg_sum += float(score[neg].sum()) if neg.any() else 0.0
        self.advantage_sum += float(advantage.sum())
        self.final_gain_sum += float(final_gain.sum())
        self.final_better += float((final_gain > 0.0).sum())
        bin_idx = np.searchsorted(self.bins, score, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.bins.size - 2)
        assert self.hist_count is not None
        assert self.hist_positive is not None
        assert self.hist_final_gain is not None
        assert self.hist_final_better is not None
        self.hist_count += np.bincount(bin_idx, minlength=self.hist_count.size)
        self.hist_positive += np.bincount(bin_idx, weights=pos.astype(np.float64), minlength=self.hist_positive.size)
        self.hist_final_gain += np.bincount(bin_idx, weights=final_gain.astype(np.float64), minlength=self.hist_final_gain.size)
        self.hist_final_better += np.bincount(
            bin_idx,
            weights=(final_gain > 0.0).astype(np.float64),
            minlength=self.hist_final_better.size,
        )

    def summary_row(self) -> dict[str, float]:
        count = max(self.count, 1.0)
        pos = max(self.positive, 1.0)
        neg = max(self.count - self.positive, 1.0)
        return {
            "count": self.count,
            "positive_rate": self.positive / count,
            "score_mean": self.score_sum / count,
            "score_pos_mean": self.score_pos_sum / pos,
            "score_neg_mean": self.score_neg_sum / neg,
            "score_pos_neg_gap": (self.score_pos_sum / pos) - (self.score_neg_sum / neg),
            "normal_advantage_mean": self.advantage_sum / count,
            "final_gain_mean": self.final_gain_sum / count,
            "final_better_rate": self.final_better / count,
            "hist_auc": self.hist_auc(),
        }

    def threshold_rows(self) -> list[dict[str, float]]:
        assert self.hist_count is not None
        assert self.hist_positive is not None
        assert self.hist_final_gain is not None
        assert self.hist_final_better is not None
        selected = np.cumsum(self.hist_count[::-1])[::-1]
        selected_pos = np.cumsum(self.hist_positive[::-1])[::-1]
        selected_gain = np.cumsum(self.hist_final_gain[::-1])[::-1]
        selected_better = np.cumsum(self.hist_final_better[::-1])[::-1]
        positives = max(self.positive, 1.0)
        negatives = max(self.count - self.positive, 1.0)
        rows = []
        for i, threshold in enumerate(self.bins[:-1]):
            sel = selected[i]
            if sel <= 0.0:
                continue
            tp = selected_pos[i]
            fp = sel - tp
            precision = tp / sel
            recall = tp / positives
            fpr = fp / negatives
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            rows.append(
                {
                    "threshold": float(threshold),
                    "selected_rate": sel / max(self.count, 1.0),
                    "precision": precision,
                    "recall": recall,
                    "fpr": fpr,
                    "f1": f1,
                    "selected_final_gain_mean": selected_gain[i] / sel,
                    "selected_final_better_rate": selected_better[i] / sel,
                    "selected_count": sel,
                }
            )
        return rows

    def hist_auc(self) -> float:
        assert self.hist_count is not None
        assert self.hist_positive is not None
        positives = self.hist_positive
        negatives = self.hist_count - self.hist_positive
        total_pos = positives.sum()
        total_neg = negatives.sum()
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan")
        # Pairwise AUC over histogram bins, with half credit for ties.
        cum_neg_lower = np.cumsum(negatives) - negatives
        favorable = (positives * cum_neg_lower).sum() + 0.5 * (positives * negatives).sum()
        return float(favorable / (total_pos * total_neg))


def make_store(bins: np.ndarray) -> dict[tuple[str, str], RunningAlignment]:
    return {(group, subset): RunningAlignment(bins=bins) for group in EVENT_GROUPS for subset in SUBSETS}


def update_store(
    store: dict[tuple[str, str], RunningAlignment],
    groups: dict[str, np.ndarray],
    subset_masks: dict[str, np.ndarray],
    score: np.ndarray,
    positive: np.ndarray,
    advantage: np.ndarray,
    final_gain: np.ndarray,
) -> None:
    for group, sample_mask in groups.items():
        if not sample_mask.any():
            continue
        group_mask = sample_mask[:, None, None, None]
        for subset, subset_mask in subset_masks.items():
            mask = subset_mask & group_mask
            if not mask.any():
                continue
            store[(group, subset)].update(
                score=score[mask].astype(np.float64, copy=False),
                positive=positive[mask],
                advantage=advantage[mask].astype(np.float64, copy=False),
                final_gain=final_gain[mask].astype(np.float64, copy=False),
            )


def summarize_model(
    model_dir: Path,
    label: str,
    cache_path: Path,
    indices: np.ndarray,
    severity_q: tuple[float, float],
    bins: np.ndarray,
    positive_margin: float,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ckpt = torch_load(model_dir / "model.pt")
    beta = residual_beta(model_dir, ckpt)
    model = make_model(ckpt, cache_path, device)
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    store = make_store(bins)
    dual_hist = model_uses_dual_hist(model, hist_channels=3)

    model.eval()
    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
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
            y_mask = y_mask.to(device)
            node_affected = node_affected.to(device)
            node_valid = node_valid.to(device)
            if dual_hist:
                hist = torch.cat([hist, hist_normal], dim=-1)
            pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal_abs = torch.abs(beta * details["normal_residual"] - y)
            base_abs = torch.abs(beta * details["base_fused_residual"] - y)
            final_abs = torch.abs(beta * pred_y - y)
            advantage = base_abs - normal_abs
            final_gain = base_abs - final_abs
            score = details["normal_veto_amount"]

            idx_np = idx.detach().cpu().numpy().astype(np.int64)
            event_aux = h5["event_aux"][idx_np].astype(np.float32)
            groups = event_group_masks(event_aux, severity_q)
            y_mask_np = y_mask.detach().cpu().numpy().astype(bool)
            affected_np = node_affected.detach().cpu().numpy().astype(bool)
            valid_np = node_valid.detach().cpu().numpy().astype(bool)
            subset_masks = {
                "all": y_mask_np,
                "affected": y_mask_np & affected_np[:, None, :, None],
                "unaffected": y_mask_np & (~affected_np[:, None, :, None]) & valid_np[:, None, :, None],
            }
            update_store(
                store=store,
                groups=groups,
                subset_masks=subset_masks,
                score=score.detach().cpu().numpy(),
                positive=(advantage.detach().cpu().numpy() > positive_margin),
                advantage=advantage.detach().cpu().numpy(),
                final_gain=final_gain.detach().cpu().numpy(),
            )
            if batch_idx % 20 == 0:
                print(f"{label}: scored {min(batch_idx * batch_size, indices.size)}/{indices.size}", flush=True)

    summary_rows = []
    threshold_rows = []
    for (group, subset), metric in store.items():
        summary_rows.append({"model": label, "group": group, "subset": subset, **metric.summary_row()})
        for row in metric.threshold_rows():
            threshold_rows.append({"model": label, "group": group, "subset": subset, **row})
    return pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def best_threshold_table(threshold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in threshold_df.groupby(["model", "group", "subset"], sort=False):
        model, group, subset = keys
        filtered = sub[sub["selected_rate"] > 0.0].copy()
        if filtered.empty:
            continue
        best_f1 = filtered.loc[filtered["f1"].idxmax()]
        high_precision_pool = filtered[filtered["precision"] >= 0.55]
        if high_precision_pool.empty:
            high_precision = filtered.loc[filtered["precision"].idxmax()]
        else:
            high_precision = high_precision_pool.sort_values(["recall", "selected_final_gain_mean"], ascending=False).iloc[0]
        rows.append(
            {
                "model": model,
                "group": group,
                "subset": subset,
                "best_f1_threshold": best_f1["threshold"],
                "best_f1": best_f1["f1"],
                "best_f1_precision": best_f1["precision"],
                "best_f1_recall": best_f1["recall"],
                "best_f1_selected_gain": best_f1["selected_final_gain_mean"],
                "hp_threshold": high_precision["threshold"],
                "hp_precision": high_precision["precision"],
                "hp_recall": high_precision["recall"],
                "hp_selected_gain": high_precision["selected_final_gain_mean"],
                "hp_selected_rate": high_precision["selected_rate"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long"]
    summary_focus = summary_df[
        (summary_df["group"].isin(focus_groups))
        & (summary_df["subset"].isin(["affected", "unaffected"]))
    ].copy()
    best_focus = best_df[
        (best_df["group"].isin(focus_groups))
        & (best_df["subset"].isin(["affected"]))
    ].copy()
    lines = [
        "# Normal-Veto Detector Alignment",
        "",
        f"- split: `{args.split}`",
        f"- positive target: `base_abs - normal_abs > {args.positive_margin}`",
        "",
        "## Summary",
        "",
        summary_focus.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Thresholds",
        "",
        best_focus.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation Notes",
        "",
        "- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.",
        "- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.",
        "- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.model_dirs):
        raise ValueError("--labels must match --model-dirs")
    labels = args.labels if args.labels else [path.name for path in args.model_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    first_ckpt = torch_load(args.model_dirs[0].resolve() / "model.pt")
    cache_path = resolve_cache_path(args.model_dirs[0].resolve(), first_ckpt)
    for model_dir in args.model_dirs[1:]:
        ckpt = torch_load(model_dir.resolve() / "model.pt")
        other_cache = resolve_cache_path(model_dir.resolve(), ckpt)
        if other_cache != cache_path:
            raise ValueError(f"model cache mismatch: {cache_path} vs {other_cache}")

    split_idx = split_indices(cache_path)[args.split]
    if args.max_samples > 0 and split_idx.size > args.max_samples:
        rng = np.random.default_rng(args.seed)
        split_idx = np.sort(rng.choice(split_idx, size=args.max_samples, replace=False))
    with h5py.File(cache_path, "r") as h5:
        event_aux = h5["event_aux"][split_idx].astype(np.float32)
    severity = np.expm1(event_aux[:, 0])
    severity[~np.isfinite(severity)] = 0.0
    severity_q = tuple(float(x) for x in np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0]))
    bin_width = max(float(args.score_bin_width), 1e-5)
    bins = np.arange(0.0, 1.0 + bin_width, bin_width, dtype=np.float64)
    if bins[-1] < 1.0:
        bins = np.append(bins, 1.0)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"samples: {split_idx.size}", flush=True)
    all_summary = []
    all_thresholds = []
    for model_dir, label in zip(args.model_dirs, labels):
        print(f"diagnosing {label}: {model_dir}", flush=True)
        summary_df, threshold_df = summarize_model(
            model_dir=model_dir.resolve(),
            label=label,
            cache_path=cache_path,
            indices=split_idx,
            severity_q=severity_q,
            bins=bins,
            positive_margin=args.positive_margin,
            batch_size=args.batch_size,
            device=device,
        )
        all_summary.append(summary_df)
        all_thresholds.append(threshold_df)

    summary_df = pd.concat(all_summary, ignore_index=True)
    threshold_df = pd.concat(all_thresholds, ignore_index=True)
    best_df = best_threshold_table(threshold_df)
    summary_df.to_csv(output_dir / "alignment_summary.csv", index=False)
    threshold_df.to_csv(output_dir / "threshold_sweep.csv", index=False)
    best_df.to_csv(output_dir / "best_thresholds.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dirs": [str(path.resolve()) for path in args.model_dirs],
                "labels": labels,
                "cache_path": str(cache_path),
                "split": args.split,
                "samples": int(split_idx.size),
                "positive_margin": args.positive_margin,
                "score_bin_width": args.score_bin_width,
                "severity_quantiles": severity_q,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_report(output_dir, summary_df, best_df, args)
    print(f"wrote detector alignment outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
