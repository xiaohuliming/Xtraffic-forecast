#!/usr/bin/env python3
"""Sweep hybrids between source group-aware veto and a posthoc normal-better detector."""

from __future__ import annotations

import argparse
import json
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
MODES = (
    "source",
    "posthoc",
    "true_non_high_non_long_posthoc",
    "true_low_or_short_posthoc",
    "pred_non_high_non_long_posthoc",
    "pred_low_or_short_posthoc",
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
        default=Path("outputs/impact_guided_next_stage/hybrid_source_posthoc_detector_seed_23"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--posthoc-scales", default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--posthoc-temperatures", default="1.0,2.0")
    parser.add_argument("--betas", default="1.05,1.1")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def cap_indices(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


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


def make_loader(cache_path: Path, indices: np.ndarray, batch_size: int) -> DataLoader:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def load_detector(detector_dir: Path, device: torch.device) -> PosthocNormalBetterDetector:
    payload = torch_load(detector_dir / "detector.pt")
    feature_dim = int(payload["feature_dim"])
    args = payload.get("args", {})
    hidden_dim = int(args.get("hidden_dim", 96)) if isinstance(args, dict) else 96
    dropout = float(args.get("dropout", 0.10)) if isinstance(args, dict) else 0.10
    detector = PosthocNormalBetterDetector(feature_dim, hidden_dim, dropout)
    state = payload["detector_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("detector_state_dict must be a dict")
    detector.load_state_dict(state, strict=True)
    detector.to(device)
    detector.eval()
    return detector


def event_semantic(raw_event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    severity = np.expm1(raw_event[:, 0])
    recovery = raw_event[:, 1] * 180.0
    spread = np.expm1(raw_event[:, 2])
    values = [severity, recovery, spread]
    for value in values:
        value[~np.isfinite(value)] = 0.0
        value[value < 0.0] = 0.0
    return severity, recovery, spread


def split_thresholds(cache_path: Path, indices: np.ndarray) -> tuple[float, float]:
    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][indices].astype(np.float32)
    severity, _recovery, _spread = event_semantic(raw_event)
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return float(q33), float(q66)


def group_masks(raw_event: np.ndarray, severity_q: tuple[float, float]) -> dict[str, np.ndarray]:
    severity, recovery, _spread = event_semantic(raw_event)
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


def predicted_group_masks(
    pred_event: torch.Tensor,
    event_aux_mean: np.ndarray,
    event_aux_std: np.ndarray,
    severity_q: tuple[float, float],
) -> dict[str, torch.Tensor]:
    mean = torch.as_tensor(event_aux_mean, dtype=pred_event.dtype, device=pred_event.device)
    std = torch.as_tensor(event_aux_std, dtype=pred_event.dtype, device=pred_event.device)
    pred_raw = pred_event * std.reshape(1, -1) + mean.reshape(1, -1)
    severity = torch.expm1(pred_raw[:, 0]).nan_to_num(0.0).clamp_min(0.0)
    recovery = (pred_raw[:, 1] * 180.0).nan_to_num(0.0).clamp_min(0.0)
    _q33, q66 = severity_q
    return {
        "non_high_non_long": (severity <= q66) & (recovery < 90.0),
        "low_or_short": (severity <= _q33) | (recovery < 30.0),
    }


def empty_sums() -> dict[str, float]:
    sums = {}
    for subset in SUBSETS:
        sums[f"{subset}_model"] = 0.0
        sums[f"{subset}_source"] = 0.0
        sums[f"{subset}_base"] = 0.0
        sums[f"{subset}_count"] = 0.0
        sums[f"{subset}_amount"] = 0.0
    return sums


def update_sums(
    sums: dict[str, float],
    residual: torch.Tensor,
    source_residual: torch.Tensor,
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
    base_abs = y.abs()
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0.0:
            continue
        sums[f"{subset}_model"] += float(model_abs[mask].sum().detach().cpu())
        sums[f"{subset}_source"] += float(source_abs[mask].sum().detach().cpu())
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
        row[f"{subset}_baseline_mae"] = sums[f"{subset}_base"] / count
        row[f"{subset}_amount_mean"] = sums[f"{subset}_amount"] / count
        row[f"{subset}_count"] = sums[f"{subset}_count"]
    return row


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    detector: PosthocNormalBetterDetector,
    cache_path: Path,
    indices: np.ndarray,
    source_beta: float,
    scales: list[float],
    temperatures: list[float],
    betas: list[float],
    batch_size: int,
    severity_q: tuple[float, float],
    dual_hist: bool,
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    loader = make_loader(cache_path, indices, batch_size)
    keys = [
        (mode, scale, temp, beta, group)
        for mode in MODES
        for scale in scales
        for temp in temperatures
        for beta in betas
        for group in GROUPS
    ]
    all_sums = {key: empty_sums() for key in keys}
    model.eval()
    detector.eval()
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
            y_mask = y_mask.to(device).bool()
            node_affected = node_affected.to(device).bool()
            node_valid = node_valid.to(device).bool()
            if dual_hist:
                hist = torch.cat([hist, hist_normal], dim=-1)
            source_pred, pred_impact, pred_event, pred_node, details = model(
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
            posthoc_logits = detector(features.reshape(-1, features.shape[-1])).reshape(y.shape)
            source_amount = details["normal_veto_amount"]
            base = details["base_fused_residual"]
            normal = details["normal_residual"]
            source_residual = (1.0 - source_amount) * base + source_amount * normal

            idx_np = idx.numpy().astype(np.int64)
            raw_event = h5["event_aux"][idx_np].astype(np.float32)
            true_groups = group_masks(raw_event, severity_q)
            severity, recovery, _spread = event_semantic(raw_event)
            _q33, q66 = severity_q
            true_masks = {
                "non_high_non_long": torch.from_numpy((severity <= q66) & (recovery < 90.0)).to(device),
                "low_or_short": torch.from_numpy((severity <= _q33) | (recovery < 30.0)).to(device),
            }
            pred_masks = predicted_group_masks(pred_event, stats.event_aux_mean, stats.event_aux_std, severity_q)

            for scale in scales:
                for temp in temperatures:
                    posthoc_score = torch.sigmoid(posthoc_logits / max(temp, 1e-6))
                    posthoc_amount = (scale * posthoc_score).clamp(0.0, 1.0)
                    mode_amounts = {
                        "source": source_amount,
                        "posthoc": posthoc_amount,
                        "true_non_high_non_long_posthoc": torch.where(
                            true_masks["non_high_non_long"][:, None, None, None],
                            posthoc_amount,
                            source_amount,
                        ),
                        "true_low_or_short_posthoc": torch.where(
                            true_masks["low_or_short"][:, None, None, None],
                            posthoc_amount,
                            source_amount,
                        ),
                        "pred_non_high_non_long_posthoc": torch.where(
                            pred_masks["non_high_non_long"][:, None, None, None],
                            posthoc_amount,
                            source_amount,
                        ),
                        "pred_low_or_short_posthoc": torch.where(
                            pred_masks["low_or_short"][:, None, None, None],
                            posthoc_amount,
                            source_amount,
                        ),
                    }
                    for mode, amount in mode_amounts.items():
                        residual = (1.0 - amount) * base + amount * normal
                        for beta in betas:
                            for group in GROUPS:
                                sample_mask = torch.from_numpy(true_groups[group]).to(device=device, dtype=torch.bool)
                                group_y_mask = y_mask & sample_mask[:, None, None, None]
                                update_sums(
                                    all_sums[(mode, scale, temp, beta, group)],
                                    residual,
                                    source_residual,
                                    y,
                                    beta,
                                    source_beta,
                                    amount,
                                    group_y_mask,
                                    node_affected,
                                    node_valid,
                                )
            if batch_idx % 20 == 0:
                print(f"{split}: evaluated {min(batch_idx * batch_size, indices.size)}/{indices.size}", flush=True)
    rows = []
    for mode, scale, temp, beta, group in keys:
        rows.append(summarize_sums(split, mode, scale, temp, beta, group, all_sums[(mode, scale, temp, beta, group)]))
    return pd.DataFrame(rows)


def config_key(row: pd.Series) -> tuple[str, float, float, float]:
    return (
        str(row["mode"]),
        float(row["posthoc_scale"]),
        float(row["posthoc_temperature"]),
        float(row["beta"]),
    )


def select_group_aware(val_df: pd.DataFrame, all_tolerance: float) -> pd.Series:
    overall = val_df[val_df["group"] == "overall"].copy()
    best_all = float(overall["all_mae"].min())
    eligible = overall[overall["all_mae"] <= best_all + all_tolerance].copy()
    if eligible.empty:
        eligible = overall
    scores = []
    for idx, row in eligible.iterrows():
        mode, scale, temp, beta = config_key(row)
        mask = (
            val_df["mode"].eq(mode)
            & np.isclose(val_df["posthoc_scale"].astype(float), scale)
            & np.isclose(val_df["posthoc_temperature"].astype(float), temp)
            & np.isclose(val_df["beta"].astype(float), beta)
        )
        sub = val_df[mask]
        sev = float(sub[sub["group"] == "severity_high"]["affected_mae"].iloc[0])
        rec = float(sub[sub["group"] == "recovery_long_ge90"]["affected_mae"].iloc[0])
        score = 0.5 * float(row["affected_mae"]) + sev + rec
        scores.append((score, idx))
    best_idx = min(scores)[1]
    return eligible.loc[best_idx]


def matching_rows(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    mode, scale, temp, beta = config_key(selected)
    return df[
        df["mode"].eq(mode)
        & np.isclose(df["posthoc_scale"].astype(float), scale)
        & np.isclose(df["posthoc_temperature"].astype(float), temp)
        & np.isclose(df["beta"].astype(float), beta)
    ].copy()


def write_summary(output_dir: Path, val_df: pd.DataFrame, test_df: pd.DataFrame, selected: pd.Series) -> None:
    test_selected = matching_rows(test_df, selected)
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long", "severity_low", "recovery_short_lt30"]
    selected_focus = test_selected[test_selected["group"].isin(focus_groups)][
        [
            "group",
            "all_mae",
            "affected_mae",
            "unaffected_mae",
            "affected_source_mae",
            "affected_amount_mean",
        ]
    ].copy()
    selected_focus["affected_delta_vs_source"] = selected_focus["affected_mae"] - selected_focus["affected_source_mae"]
    top_cols = [
        "mode",
        "posthoc_scale",
        "posthoc_temperature",
        "beta",
        "all_mae",
        "affected_mae",
        "unaffected_mae",
        "affected_source_mae",
        "affected_amount_mean",
    ]
    lines = [
        "# Hybrid Source/Posthoc Detector Sweep",
        "",
        "Source is the group-aware normal-veto model; posthoc is the detached normal-better detector.",
        "",
        "## Group-Aware Validation Selection",
        "",
        f"- mode: `{selected['mode']}`",
        f"- posthoc_scale: `{float(selected['posthoc_scale']):.4g}`",
        f"- posthoc_temperature: `{float(selected['posthoc_temperature']):.4g}`",
        f"- beta: `{float(selected['beta']):.4g}`",
        f"- validation all / affected MAE: `{float(selected['all_mae']):.6f}` / `{float(selected['affected_mae']):.6f}`",
        "",
        "## Test Groups At Selected Config",
        "",
        selected_focus.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top Validation Overall Affected",
        "",
        val_df[val_df["group"] == "overall"].sort_values("affected_mae")[top_cols].head(15).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Top Validation Severity-High Affected",
        "",
        val_df[val_df["group"] == "severity_high"].sort_values("affected_mae")[top_cols].head(15).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Top Test Severity-High Affected",
        "",
        test_df[test_df["group"] == "severity_high"].sort_values("affected_mae")[top_cols].head(15).to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    source_dir = args.source_model_dir.resolve()
    detector_dir = args.detector_dir.resolve()
    ckpt = torch_load(source_dir / "model.pt")
    cache_path = resolve_cache_path(source_dir, ckpt)
    model = make_model(ckpt, cache_path, device)
    detector = load_detector(detector_dir, device)
    source_beta = residual_beta(source_dir, ckpt)
    dual_hist = model_uses_dual_hist(model, hist_channels=3)
    splits = split_indices(cache_path)
    val_idx = cap_indices(splits["val"], args.max_samples, args.seed + 1)
    test_idx = cap_indices(splits["test"], args.max_samples, args.seed + 2)
    val_severity_q = split_thresholds(cache_path, val_idx)
    test_severity_q = split_thresholds(cache_path, test_idx)
    scales = parse_float_list(args.posthoc_scales)
    temperatures = parse_float_list(args.posthoc_temperatures)
    betas = parse_float_list(args.betas)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source: {source_dir}", flush=True)
    print(f"detector: {detector_dir}", flush=True)
    print(f"source_beta: {source_beta}", flush=True)
    print(f"val/test samples: {val_idx.size}/{test_idx.size}", flush=True)
    val_df = evaluate_split(
        "val",
        model,
        detector,
        cache_path,
        val_idx,
        source_beta,
        scales,
        temperatures,
        betas,
        args.batch_size,
        val_severity_q,
        dual_hist,
        device,
    )
    test_df = evaluate_split(
        "test",
        model,
        detector,
        cache_path,
        test_idx,
        source_beta,
        scales,
        temperatures,
        betas,
        args.batch_size,
        test_severity_q,
        dual_hist,
        device,
    )
    selected = select_group_aware(val_df, args.all_val_tolerance)
    val_df.to_csv(output_dir / "val_hybrid_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_hybrid_sweep.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "source_model_dir": str(source_dir),
                "detector_dir": str(detector_dir),
                "cache_path": str(cache_path),
                "source_beta": source_beta,
                "val_samples": int(val_idx.size),
                "test_samples": int(test_idx.size),
                "val_severity_quantiles": val_severity_q,
                "test_severity_quantiles": test_severity_q,
                "selected": {
                    "mode": str(selected["mode"]),
                    "posthoc_scale": float(selected["posthoc_scale"]),
                    "posthoc_temperature": float(selected["posthoc_temperature"]),
                    "beta": float(selected["beta"]),
                },
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    write_summary(output_dir, val_df, test_df, selected)
    print(f"wrote hybrid sweep outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
