#!/usr/bin/env python3
"""Analyze gate behavior and branch specialization for the dual-branch model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_dual_branch_confidence_gate import DualBranchConfidenceGate
from train_dual_branch_gate_baseline import DualBranchGateBaseline, infer_cache_shapes
from train_dual_branch_sttis_gate import (
    DualBranchSTTISBranchConfidenceGate,
    DualBranchSTTISGate,
    DualBranchSTTISHierarchicalImpactNormalVetoGate,
    DualBranchSTTISDeltaGate,
    DualBranchSTTISImpactConditionedNormalVetoGate,
    DualBranchSTTISLocalSelectorGate,
    DualBranchSTTISNodeEventNormalVetoGate,
    DualBranchSTTISNormalVetoGate,
    DualBranchSTTISProposalUncertaintyGate,
    DualBranchSTTISProposalGate,
    DualBranchSTTISReliabilityGate,
    DualBranchSTTISUncertaintyGate,
    DualBranchSTTISVetoGate,
)
from train_full_candidate_stgnn_heatmap_model import CHANNELS, H5IncidentDataset, compute_stats, split_indices
from train_impact_residual_model import choose_device


BRANCH_LABELS = {
    "baseline": "Normal baseline",
    "normal_branch": "Normal-style residual",
    "incident_branch": "Incident-graph residual",
    "fixed_gate_05": "Fixed gate 0.5",
    "learned_gate": "Learned gate",
}
SUBSETS = ("all", "affected", "unaffected")


class IndexedH5IncidentDataset(H5IncidentDataset):
    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        sample = super().__getitem__(item)
        idx = torch.tensor(int(self.indices[item]), dtype=torch.long)
        return (*sample, idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_gate_full_no_aux"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_gate_interpretability"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def torch_load(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cap_indices(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def make_model(ckpt: dict[str, object], cache_path: Path, device: torch.device) -> DualBranchGateBaseline:
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError("checkpoint args must be a dict")
    shapes = infer_cache_shapes(cache_path)
    common_kwargs = {
        "channels": shapes["channels"],
        "hist_input_channels": len(CHANNELS) * (2 if bool(model_args.get("use_dual_hist_residual", False)) else 1),
        "node_context_dim": shapes["node_context_dim"],
        "global_context_dim": shapes["global_context_dim"],
        "horizon_steps": shapes["horizon_steps"],
        "hidden_dim": int(model_args.get("hidden_dim", 96)),
        "graph_layers": int(model_args.get("graph_layers", 2)),
        "dropout": float(model_args.get("dropout", 0.10)),
        "graph_sigma": float(model_args.get("graph_sigma", 3.0)),
        "graph_mode": str(model_args.get("graph_mode", "undirected")),
        "use_normal_delta": bool(model_args.get("use_normal_delta", True)),
        "use_normal_delta_abs": bool(model_args.get("use_normal_delta_abs", True)),
    }
    if str(model_args.get("model_class", "")) == "DualBranchConfidenceGate":
        model = DualBranchConfidenceGate(
            **common_kwargs,
            confidence_scale=float(model_args.get("confidence_scale", 1.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISGate":
        model = DualBranchSTTISGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISProposalGate":
        model = DualBranchSTTISProposalGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISReliabilityGate":
        model = DualBranchSTTISReliabilityGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            reliability_scale=float(model_args.get("reliability_scale", 1.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISVetoGate":
        model = DualBranchSTTISVetoGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            veto_scale=float(model_args.get("veto_scale", 1.0)),
            veto_max=float(model_args.get("veto_max", 2.0)),
            veto_init_bias=float(model_args.get("veto_init_bias", -6.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISDeltaGate":
        model = DualBranchSTTISDeltaGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            delta_scale=float(model_args.get("delta_scale", 1.0)),
            delta_max=float(model_args.get("delta_max", 2.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISBranchConfidenceGate":
        model = DualBranchSTTISBranchConfidenceGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            confidence_scale=float(model_args.get("confidence_scale", 1.0)),
            confidence_max=float(model_args.get("confidence_max", 2.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISUncertaintyGate":
        model = DualBranchSTTISUncertaintyGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            uncertainty_scale=float(model_args.get("uncertainty_scale", 1.0)),
            uncertainty_max=float(model_args.get("uncertainty_max", 2.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISProposalUncertaintyGate":
        model = DualBranchSTTISProposalUncertaintyGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            uncertainty_scale=float(model_args.get("uncertainty_scale", 1.0)),
            uncertainty_max=float(model_args.get("uncertainty_max", 2.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISLocalSelectorGate":
        model = DualBranchSTTISLocalSelectorGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            selector_temperature=float(model_args.get("selector_temperature", 1.0)),
            selector_init_base_bias=float(model_args.get("selector_init_base_bias", 2.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISNormalVetoGate":
        model = DualBranchSTTISNormalVetoGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            normal_veto_scale=float(model_args.get("normal_veto_scale", 1.0)),
            normal_veto_temperature=float(model_args.get("normal_veto_temperature", 1.0)),
            normal_veto_init_bias=float(model_args.get("normal_veto_init_bias", -4.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISNodeEventNormalVetoGate":
        model = DualBranchSTTISNodeEventNormalVetoGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            normal_veto_scale=float(model_args.get("normal_veto_scale", 1.0)),
            normal_veto_temperature=float(model_args.get("normal_veto_temperature", 1.0)),
            normal_veto_init_bias=float(model_args.get("normal_veto_init_bias", -4.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISImpactConditionedNormalVetoGate":
        model = DualBranchSTTISImpactConditionedNormalVetoGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            normal_veto_scale=float(model_args.get("normal_veto_scale", 1.0)),
            normal_veto_temperature=float(model_args.get("normal_veto_temperature", 1.0)),
            normal_veto_init_bias=float(model_args.get("normal_veto_init_bias", -4.0)),
        )
    elif str(model_args.get("model_class", "")) == "DualBranchSTTISHierarchicalImpactNormalVetoGate":
        model = DualBranchSTTISHierarchicalImpactNormalVetoGate(
            **common_kwargs,
            sttis_heads=int(model_args.get("sttis_heads", 4)),
            sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
            sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
            sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
            proposal_feature_count=int(model_args.get("proposal_feature_count", 5)),
            normal_veto_scale=float(model_args.get("normal_veto_scale", 1.0)),
            normal_veto_temperature=float(model_args.get("normal_veto_temperature", 1.0)),
            normal_veto_init_bias=float(model_args.get("normal_veto_init_bias", -4.0)),
        )
    else:
        model = DualBranchGateBaseline(**common_kwargs)
    state = ckpt["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def empty_metric_sums(horizon_steps: int) -> dict[str, dict[str, dict[str, object]]]:
    out: dict[str, dict[str, dict[str, object]]] = {}
    for branch in BRANCH_LABELS:
        out[branch] = {}
        for subset in SUBSETS:
            out[branch][subset] = {
                "sum": 0.0,
                "count": 0.0,
                "h_sum": np.zeros(horizon_steps, dtype=np.float64),
                "h_count": np.zeros(horizon_steps, dtype=np.float64),
            }
    return out


def empty_value_stats(horizon_steps: int) -> dict[str, dict[str, object]]:
    return {
        subset: {
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0.0,
            "h_sum": np.zeros(horizon_steps, dtype=np.float64),
            "h_sumsq": np.zeros(horizon_steps, dtype=np.float64),
            "h_count": np.zeros(horizon_steps, dtype=np.float64),
        }
        for subset in SUBSETS
    }


def empty_pair_stats() -> dict[str, dict[str, float]]:
    return {subset: {"sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0, "count": 0.0} for subset in SUBSETS}


def update_metric_sums(
    metric_sums: dict[str, dict[str, dict[str, object]]],
    branch: str,
    error: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> None:
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0:
            continue
        metric_sums[branch][subset]["sum"] = float(metric_sums[branch][subset]["sum"]) + float(error[mask].sum().detach().cpu())
        metric_sums[branch][subset]["count"] = float(metric_sums[branch][subset]["count"]) + count
        h_sum = metric_sums[branch][subset]["h_sum"]
        h_count = metric_sums[branch][subset]["h_count"]
        assert isinstance(h_sum, np.ndarray) and isinstance(h_count, np.ndarray)
        for h in range(error.shape[1]):
            h_mask = mask[:, h]
            h_c = float(h_mask.sum().item())
            if h_c <= 0:
                continue
            h_sum[h] += float(error[:, h][h_mask].sum().detach().cpu())
            h_count[h] += h_c


def update_value_stats(stats: dict[str, dict[str, object]], values: torch.Tensor, masks: dict[str, torch.Tensor]) -> None:
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0:
            continue
        selected = values[mask]
        stats[subset]["sum"] = float(stats[subset]["sum"]) + float(selected.sum().detach().cpu())
        stats[subset]["sumsq"] = float(stats[subset]["sumsq"]) + float((selected * selected).sum().detach().cpu())
        stats[subset]["count"] = float(stats[subset]["count"]) + count
        h_sum = stats[subset]["h_sum"]
        h_sumsq = stats[subset]["h_sumsq"]
        h_count = stats[subset]["h_count"]
        assert isinstance(h_sum, np.ndarray) and isinstance(h_sumsq, np.ndarray) and isinstance(h_count, np.ndarray)
        for h in range(values.shape[1]):
            h_mask = mask[:, h]
            h_c = float(h_mask.sum().item())
            if h_c <= 0:
                continue
            h_values = values[:, h][h_mask]
            h_sum[h] += float(h_values.sum().detach().cpu())
            h_sumsq[h] += float((h_values * h_values).sum().detach().cpu())
            h_count[h] += h_c


def update_pair_stats(
    stats: dict[str, dict[str, float]],
    x: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> None:
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0:
            continue
        xv = x[mask]
        yv = y[mask]
        row = stats[subset]
        row["sum_x"] += float(xv.sum().detach().cpu())
        row["sum_y"] += float(yv.sum().detach().cpu())
        row["sum_x2"] += float((xv * xv).sum().detach().cpu())
        row["sum_y2"] += float((yv * yv).sum().detach().cpu())
        row["sum_xy"] += float((xv * yv).sum().detach().cpu())
        row["count"] += count


def corr_from_stats(row: dict[str, float]) -> float:
    n = row["count"]
    if n <= 1:
        return float("nan")
    cov = row["sum_xy"] - row["sum_x"] * row["sum_y"] / n
    var_x = row["sum_x2"] - row["sum_x"] ** 2 / n
    var_y = row["sum_y2"] - row["sum_y"] ** 2 / n
    denom = np.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
    return float(cov / denom) if denom > 0 else float("nan")


def stats_mean_std(row: dict[str, object]) -> tuple[float, float]:
    count = float(row["count"])
    if count <= 0:
        return float("nan"), float("nan")
    mean = float(row["sum"]) / count
    var = float(row["sumsq"]) / count - mean * mean
    return mean, float(np.sqrt(max(var, 0.0)))


def event_group(sample_values: np.ndarray, low_q: float, high_q: float, low: str, mid: str, high: str) -> np.ndarray:
    labels = np.full(sample_values.shape, mid, dtype=object)
    labels[sample_values <= low_q] = low
    labels[sample_values >= high_q] = high
    return labels


def write_branch_metrics(metric_sums: dict[str, dict[str, dict[str, object]]], output_dir: Path) -> pd.DataFrame:
    rows = []
    horizon_rows = []
    for branch, subsets in metric_sums.items():
        for subset, vals in subsets.items():
            count = float(vals["count"])
            mae = float(vals["sum"]) / max(count, 1.0)
            rows.append({"branch": branch, "branch_label": BRANCH_LABELS[branch], "subset": subset, "mae": mae, "count": count})
            h_sum = vals["h_sum"]
            h_count = vals["h_count"]
            assert isinstance(h_sum, np.ndarray) and isinstance(h_count, np.ndarray)
            for h, (s, c) in enumerate(zip(h_sum, h_count), start=1):
                horizon_rows.append(
                    {
                        "branch": branch,
                        "branch_label": BRANCH_LABELS[branch],
                        "subset": subset,
                        "horizon": h,
                        "mae": float(s / max(c, 1.0)),
                        "count": float(c),
                    }
                )
    df = pd.DataFrame(rows)
    h_df = pd.DataFrame(horizon_rows)
    df.to_csv(output_dir / "branch_ablation_metrics.csv", index=False)
    h_df.to_csv(output_dir / "branch_ablation_by_horizon.csv", index=False)
    return df


def write_gate_stats(
    gate_stats: dict[str, dict[str, object]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    horizon_rows = []
    for subset, vals in gate_stats.items():
        mean, std = stats_mean_std(vals)
        rows.append({"subset": subset, "gate_mean": mean, "gate_std": std, "count": float(vals["count"])})
        h_sum = vals["h_sum"]
        h_sumsq = vals["h_sumsq"]
        h_count = vals["h_count"]
        assert isinstance(h_sum, np.ndarray) and isinstance(h_sumsq, np.ndarray) and isinstance(h_count, np.ndarray)
        for h, (s, ss, c) in enumerate(zip(h_sum, h_sumsq, h_count), start=1):
            mean_h = float(s / max(c, 1.0))
            var_h = float(ss / max(c, 1.0) - mean_h * mean_h)
            horizon_rows.append(
                {
                    "subset": subset,
                    "horizon": h,
                    "gate_mean": mean_h,
                    "gate_std": float(np.sqrt(max(var_h, 0.0))),
                    "count": float(c),
                }
            )
    df = pd.DataFrame(rows)
    h_df = pd.DataFrame(horizon_rows)
    df.to_csv(output_dir / "gate_summary.csv", index=False)
    h_df.to_csv(output_dir / "gate_by_horizon.csv", index=False)
    return df, h_df


def write_pair_stats(
    residual_corr: dict[str, dict[str, float]],
    normal_delta_corr: dict[str, dict[str, float]],
    selection_stats: dict[str, dict[str, dict[str, object]]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows = []
    for subset in SUBSETS:
        corr_rows.append(
            {
                "subset": subset,
                "gate_vs_abs_target_residual_corr": corr_from_stats(residual_corr[subset]),
                "gate_vs_abs_normal_delta_corr": corr_from_stats(normal_delta_corr[subset]),
                "count": residual_corr[subset]["count"],
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "gate_correlations.csv", index=False)

    align_rows = []
    for subset, buckets in selection_stats.items():
        for bucket, vals in buckets.items():
            mean, std = stats_mean_std(vals)
            align_rows.append(
                {
                    "subset": subset,
                    "case": bucket,
                    "gate_mean": mean,
                    "gate_std": std,
                    "count": float(vals["count"]),
                }
            )
    align_df = pd.DataFrame(align_rows)
    align_df.to_csv(output_dir / "gate_selection_alignment.csv", index=False)
    return corr_df, align_df


def plot_outputs(branch_df: pd.DataFrame, gate_horizon_df: pd.DataFrame, align_df: pd.DataFrame, event_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    plot_df = branch_df[branch_df["subset"].isin(["all", "affected"])]
    labels = [BRANCH_LABELS[key] for key in BRANCH_LABELS]
    x = np.arange(len(labels))
    width = 0.36
    for offset, subset in [(-width / 2, "all"), (width / 2, "affected")]:
        vals = [
            float(plot_df[(plot_df["branch"] == branch) & (plot_df["subset"] == subset)]["mae"].iloc[0])
            for branch in BRANCH_LABELS
        ]
        ax.bar(x + offset, vals, width=width, label=subset)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("robust MAE")
    ax.set_title("Branch ablation under the same residual beta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "branch_ablation_mae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for subset in ["affected", "unaffected"]:
        sdf = gate_horizon_df[gate_horizon_df["subset"] == subset]
        ax.plot(sdf["horizon"], sdf["gate_mean"], marker="o", label=subset)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel("mean incident-branch gate")
    ax.set_title("Gate weight by horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gate_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    align_plot = align_df[align_df["subset"].isin(["affected", "unaffected"])]
    cases = ["incident_branch_better", "normal_branch_better"]
    x = np.arange(len(cases))
    width = 0.36
    for offset, subset in [(-width / 2, "affected"), (width / 2, "unaffected")]:
        vals = [
            float(align_plot[(align_plot["subset"] == subset) & (align_plot["case"] == case)]["gate_mean"].iloc[0])
            for case in cases
        ]
        ax.bar(x + offset, vals, width=width, label=subset)
    ax.set_xticks(x)
    ax.set_xticklabels(["incident branch\nhas lower error", "normal branch\nhas lower error"])
    ax.set_ylabel("mean incident-branch gate")
    ax.set_title("Does the gate lean toward the locally better branch?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gate_selection_alignment.png", dpi=180)
    plt.close(fig)

    if not event_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
        for ax, dim in zip(axes, ["severity", "recovery"]):
            sdf = event_df[(event_df["dimension"] == dim) & (event_df["subset"] == "affected")]
            ax.bar(sdf["group"], sdf["gate_mean"], color="#3B6EA8")
            ax.set_title(dim)
            ax.set_xlabel("group")
            ax.tick_params(axis="x", rotation=20)
        axes[0].set_ylabel("affected mean incident-branch gate")
        fig.suptitle("Gate by incident group")
        fig.tight_layout()
        fig.savefig(output_dir / "gate_by_event_group.png", dpi=180)
        plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    branch_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    align_df: pd.DataFrame,
    event_df: pd.DataFrame,
    residual_beta: float,
    sample_count: int,
) -> None:
    def branch_mae(branch: str, subset: str) -> float:
        return float(branch_df[(branch_df["branch"] == branch) & (branch_df["subset"] == subset)]["mae"].iloc[0])

    def gate_mean(subset: str) -> float:
        return float(gate_df[gate_df["subset"] == subset]["gate_mean"].iloc[0])

    def align_mean(subset: str, case: str) -> float:
        return float(align_df[(align_df["subset"] == subset) & (align_df["case"] == case)]["gate_mean"].iloc[0])

    affected_learned = branch_mae("learned_gate", "affected")
    affected_fixed = branch_mae("fixed_gate_05", "affected")
    affected_normal = branch_mae("normal_branch", "affected")
    affected_incident = branch_mae("incident_branch", "affected")
    all_learned = branch_mae("learned_gate", "all")
    all_fixed = branch_mae("fixed_gate_05", "all")
    aff_gate = gate_mean("affected")
    unaff_gate = gate_mean("unaffected")
    aff_incident_better = align_mean("affected", "incident_branch_better")
    aff_normal_better = align_mean("affected", "normal_branch_better")
    all_corr = float(corr_df[corr_df["subset"] == "all"]["gate_vs_abs_target_residual_corr"].iloc[0])

    lines = [
        "# Dual-Branch Gate Interpretability",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- split: `{args.split}`",
        f"- evaluated samples: `{sample_count}`",
        f"- residual_beta: `{residual_beta:.2f}`",
        "",
        "## Branch ablation",
        "",
        f"- Learned gate all-candidate MAE: `{all_learned:.4f}`; fixed 0.5 gate: `{all_fixed:.4f}`.",
        f"- Learned gate affected-candidate MAE: `{affected_learned:.4f}`; fixed 0.5 gate: `{affected_fixed:.4f}`.",
        f"- Affected branch-only MAE: normal-style `{affected_normal:.4f}`, incident-graph `{affected_incident:.4f}`.",
        "",
        "## Gate behavior",
        "",
        f"- Mean gate on affected elements: `{aff_gate:.4f}`.",
        f"- Mean gate on unaffected elements: `{unaff_gate:.4f}`.",
        f"- On affected elements where the incident branch has lower local error, mean gate is `{aff_incident_better:.4f}`.",
        f"- On affected elements where the normal-style branch has lower local error, mean gate is `{aff_normal_better:.4f}`.",
        f"- Correlation between gate and absolute target residual on all valid elements: `{all_corr:.4f}`.",
        "",
        "## Files",
        "",
        "- `branch_ablation_metrics.csv`",
        "- `branch_ablation_by_horizon.csv`",
        "- `gate_summary.csv`",
        "- `gate_by_horizon.csv`",
        "- `gate_selection_alignment.csv`",
        "- `gate_correlations.csv`",
        "- `event_group_gate_metrics.csv`",
        "- `branch_ablation_mae.png`",
        "- `gate_by_horizon.png`",
        "- `gate_selection_alignment.png`",
        "- `gate_by_event_group.png`",
    ]
    if not event_df.empty:
        affected_event = event_df[event_df["subset"] == "affected"]
        severity_rows = affected_event[affected_event["dimension"] == "severity"][["group", "gate_mean"]]
        recovery_rows = affected_event[affected_event["dimension"] == "recovery"][["group", "gate_mean"]]
        lines.extend(
            [
                "",
                "## Event group gate means",
                "",
                "Severity:",
                severity_rows.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Recovery:",
                recovery_rows.to_markdown(index=False, floatfmt=".4f"),
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        metrics_path = model_dir / "metrics.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        cache_path = Path(data["cache_path"])
    cache_path = cache_path.resolve()
    residual_beta = float(ckpt.get("residual_beta", 1.0))
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        model_args = {}

    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)

    model = make_model(ckpt, cache_path, device)
    stats = compute_stats(cache_path)
    indices_by_split = split_indices(cache_path)
    selected_indices = cap_indices(indices_by_split[args.split], args.max_samples, args.seed)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=selected_indices, stats=stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    horizon_steps = int(model.horizon_steps)

    metric_sums = empty_metric_sums(horizon_steps)
    gate_stats = empty_value_stats(horizon_steps)
    residual_corr = empty_pair_stats()
    normal_delta_corr = empty_pair_stats()
    selection_stats: dict[str, dict[str, dict[str, object]]] = {
        subset: {
            "incident_branch_better": {"sum": 0.0, "sumsq": 0.0, "count": 0.0},
            "normal_branch_better": {"sum": 0.0, "sumsq": 0.0, "count": 0.0},
        }
        for subset in SUBSETS
    }
    event_stats: DefaultDict[tuple[str, str, str], dict[str, object]] = defaultdict(lambda: {"sum": 0.0, "sumsq": 0.0, "count": 0.0})

    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][selected_indices]
        severity_low, severity_high = np.quantile(raw_event[:, 0], [1 / 3, 2 / 3])
        recovery_low, recovery_high = np.quantile(raw_event[:, 1], [1 / 3, 2 / 3])

        with torch.no_grad():
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
                idx_np = idx.detach().cpu().numpy().astype(np.int64)

                if bool(model_args.get("use_dual_hist_residual", False)):
                    hist_in = torch.cat([hist, hist_normal], dim=-1)
                else:
                    hist_in = hist

                pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                    hist_in,
                    node,
                    global_context,
                    normal_delta,
                    return_details=True,
                )
                normal_residual = details["normal_residual"]
                incident_residual = details["incident_residual"]
                gate = details["gate"]
                fixed_residual = 0.5 * normal_residual + 0.5 * incident_residual

                all_mask = y_mask.bool()
                affected_mask = all_mask & node_affected[:, None, :, None].bool()
                unaffected_mask = all_mask & (~node_affected[:, None, :, None].bool()) & node_valid[:, None, :, None].bool()
                masks = {"all": all_mask, "affected": affected_mask, "unaffected": unaffected_mask}

                errors = {
                    "baseline": torch.abs(y),
                    "normal_branch": torch.abs(residual_beta * normal_residual - y),
                    "incident_branch": torch.abs(residual_beta * incident_residual - y),
                    "fixed_gate_05": torch.abs(residual_beta * fixed_residual - y),
                    "learned_gate": torch.abs(residual_beta * pred_y - y),
                }
                for branch, error in errors.items():
                    update_metric_sums(metric_sums, branch, error, masks)

                update_value_stats(gate_stats, gate, masks)
                update_pair_stats(residual_corr, gate, torch.abs(y), masks)
                update_pair_stats(normal_delta_corr, gate, torch.abs(normal_delta), masks)

                incident_better = errors["incident_branch"] < errors["normal_branch"]
                normal_better = errors["normal_branch"] < errors["incident_branch"]
                for subset, base_mask in masks.items():
                    for case, case_mask in [
                        ("incident_branch_better", base_mask & incident_better),
                        ("normal_branch_better", base_mask & normal_better),
                    ]:
                        count = float(case_mask.sum().item())
                        if count <= 0:
                            continue
                        selected_gate = gate[case_mask]
                        vals = selection_stats[subset][case]
                        vals["sum"] = float(vals["sum"]) + float(selected_gate.sum().detach().cpu())
                        vals["sumsq"] = float(vals["sumsq"]) + float((selected_gate * selected_gate).sum().detach().cpu())
                        vals["count"] = float(vals["count"]) + count

                batch_event = h5["event_aux"][idx_np]
                severity_labels = event_group(batch_event[:, 0], severity_low, severity_high, "low", "mid", "high")
                recovery_labels = event_group(batch_event[:, 1], recovery_low, recovery_high, "short", "mid", "long")
                for dimension, labels in [("severity", severity_labels), ("recovery", recovery_labels)]:
                    for group in sorted(set(labels.tolist())):
                        group_mask_np = labels == group
                        group_mask = torch.from_numpy(group_mask_np).to(device=device, dtype=torch.bool)
                        group_masks = {
                            "all": all_mask & group_mask[:, None, None, None],
                            "affected": affected_mask & group_mask[:, None, None, None],
                        }
                        for subset, mask in group_masks.items():
                            count = float(mask.sum().item())
                            if count <= 0:
                                continue
                            selected_gate = gate[mask]
                            vals = event_stats[(dimension, group, subset)]
                            vals["sum"] = float(vals["sum"]) + float(selected_gate.sum().detach().cpu())
                            vals["sumsq"] = float(vals["sumsq"]) + float((selected_gate * selected_gate).sum().detach().cpu())
                            vals["count"] = float(vals["count"]) + count

                if batch_idx % 25 == 0:
                    print(f"processed {min(batch_idx * args.batch_size, selected_indices.size)}/{selected_indices.size}", flush=True)

    branch_df = write_branch_metrics(metric_sums, output_dir)
    gate_df, gate_horizon_df = write_gate_stats(gate_stats, output_dir)
    corr_df, align_df = write_pair_stats(residual_corr, normal_delta_corr, selection_stats, output_dir)

    event_rows = []
    for (dimension, group, subset), vals in sorted(event_stats.items()):
        mean, std = stats_mean_std(vals)
        event_rows.append(
            {
                "dimension": dimension,
                "group": group,
                "subset": subset,
                "gate_mean": mean,
                "gate_std": std,
                "count": float(vals["count"]),
            }
        )
    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(output_dir / "event_group_gate_metrics.csv", index=False)

    plot_outputs(branch_df, gate_horizon_df, align_df, event_df, output_dir)
    write_summary(
        output_dir=output_dir,
        args=args,
        branch_df=branch_df,
        gate_df=gate_df,
        corr_df=corr_df,
        align_df=align_df,
        event_df=event_df,
        residual_beta=residual_beta,
        sample_count=int(selected_indices.size),
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "split": args.split,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "sample_count": int(selected_indices.size),
                "residual_beta": residual_beta,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote interpretability outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
