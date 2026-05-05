#!/usr/bin/env python3
"""Select normal-veto posthoc scale/beta with severity/recovery-aware validation groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from analyze_dual_branch_gate import make_model, torch_load
from train_full_candidate_stgnn_heatmap_model import compute_stats, forecast_metrics_for_loader, make_loader, split_indices
from train_impact_residual_model import choose_device


METRIC_COLUMNS = (
    "all_candidates_model_robust_mae",
    "affected_candidates_model_robust_mae",
    "unaffected_candidates_model_robust_mae",
)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sweep-scales", default="0.0,0.25,0.5,1.0")
    parser.add_argument("--sweep-temperatures", default="0.75,1.0,1.5")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument(
        "--selection-groups",
        default="overall,severity_high,recovery_long_ge90",
        help="Comma-separated validation groups used in the group-aware score.",
    )
    parser.add_argument(
        "--selection-weights",
        default="0.5,1.0,1.0",
        help="Comma-separated weights matching --selection-groups.",
    )
    parser.add_argument(
        "--all-val-tolerance",
        type=float,
        default=0.002,
        help="Only consider configs within this overall-all validation MAE gap from the best overall-all config.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_cache_path(model_dir: Path, ckpt: dict[str, object]) -> Path:
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        metrics_path = model_dir / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cache_path = Path(payload["cache_path"])
    return cache_path.resolve()


def event_groups(cache_path: Path, indices: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    with h5py.File(cache_path, "r") as h5:
        event_aux = h5["event_aux"][indices].astype(np.float32)
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "overall": (np.ones(indices.shape[0], dtype=bool), "all events"),
        "severity_low": (severity <= q33, f"severity <= {q33:.3f}"),
        "severity_mid": ((severity > q33) & (severity <= q66), f"{q33:.3f} < severity <= {q66:.3f}"),
        "severity_high": (severity > q66, f"severity > {q66:.3f}"),
        "recovery_short_lt30": (recovery_min < 30.0, "recovery < 30 min"),
        "recovery_mid_30_90": ((recovery_min >= 30.0) & (recovery_min < 90.0), "30 <= recovery < 90 min"),
        "recovery_long_ge90": (recovery_min >= 90.0, "recovery >= 90 min"),
        "severity_high_and_long": ((severity > q66) & (recovery_min >= 90.0), "high severity and long recovery"),
    }


def evaluate_sweep(
    model: torch.nn.Module,
    cache_path: Path,
    stats: object,
    split_indices_np: np.ndarray,
    group_names: list[str],
    scales: list[float],
    temperatures: list[float],
    betas: list[float],
    batch_size: int,
    device: torch.device,
    split: str,
) -> pd.DataFrame:
    if not hasattr(model, "normal_veto_scale") or not hasattr(model, "normal_veto_temperature"):
        raise TypeError("model does not expose normal_veto_scale/normal_veto_temperature")
    groups = event_groups(cache_path, split_indices_np)
    missing = [name for name in group_names if name not in groups]
    if missing:
        raise KeyError(f"unknown group(s): {missing}")

    rows: list[dict[str, float | int | str]] = []
    old_scale = float(model.normal_veto_scale)
    old_temperature = float(model.normal_veto_temperature)
    try:
        for group_name in group_names:
            mask, label = groups[group_name]
            idx = split_indices_np[mask]
            loader = make_loader(cache_path, idx, stats, batch_size=batch_size, shuffle=False)
            print(f"{split}:{group_name} samples={idx.size}", flush=True)
            for scale in scales:
                model.normal_veto_scale = float(scale)
                for temperature in temperatures:
                    model.normal_veto_temperature = float(temperature)
                    metrics_by_beta = forecast_metrics_for_loader(model, loader, betas, device)
                    for beta, metrics in metrics_by_beta.items():
                        row = {
                            "split": split,
                            "group": group_name,
                            "label": label,
                            "samples": int(idx.size),
                            "normal_veto_scale": float(scale),
                            "normal_veto_temperature": float(temperature),
                            "residual_beta": float(beta),
                        }
                        for metric in METRIC_COLUMNS:
                            row[metric] = float(metrics[metric])
                        rows.append(row)
    finally:
        model.normal_veto_scale = old_scale
        model.normal_veto_temperature = old_temperature
    return pd.DataFrame(rows)


def select_config(
    val_df: pd.DataFrame,
    selection_groups: list[str],
    selection_weights: list[float],
    all_val_tolerance: float,
) -> tuple[pd.Series, pd.DataFrame]:
    if len(selection_groups) != len(selection_weights):
        raise ValueError("--selection-groups and --selection-weights must have the same length")
    key_cols = ["normal_veto_scale", "normal_veto_temperature", "residual_beta"]
    weight_df = pd.DataFrame({"group": selection_groups, "group_weight": selection_weights})
    score_df = val_df[val_df["group"].isin(selection_groups)].merge(weight_df, on="group", how="inner")
    score_rows = (
        score_df.assign(weighted_affected=lambda df: df["affected_candidates_model_robust_mae"] * df["group_weight"])
        .groupby(key_cols, as_index=False)
        .agg(
            group_score=("weighted_affected", "sum"),
            group_weight=("group_weight", "sum"),
            group_affected_max=("affected_candidates_model_robust_mae", "max"),
        )
    )
    score_rows["group_score"] = score_rows["group_score"] / score_rows["group_weight"].clip(lower=1e-12)

    overall = val_df[val_df["group"] == "overall"][key_cols + list(METRIC_COLUMNS)].copy()
    overall = overall.rename(
        columns={
            "all_candidates_model_robust_mae": "overall_all_mae",
            "affected_candidates_model_robust_mae": "overall_affected_mae",
            "unaffected_candidates_model_robust_mae": "overall_unaffected_mae",
        }
    )
    score_rows = score_rows.merge(overall, on=key_cols, how="left")
    best_all = float(score_rows["overall_all_mae"].min())
    score_rows["eligible"] = score_rows["overall_all_mae"] <= best_all + all_val_tolerance
    eligible = score_rows[score_rows["eligible"]].copy()
    if eligible.empty:
        eligible = score_rows.copy()
    eligible = eligible.sort_values(
        ["group_score", "group_affected_max", "overall_affected_mae", "overall_all_mae"],
        ascending=[True, True, True, True],
    )
    return eligible.iloc[0], score_rows.sort_values("group_score")


def evaluate_selected(
    model: torch.nn.Module,
    cache_path: Path,
    stats: object,
    split_indices_np: np.ndarray,
    selected: pd.Series,
    batch_size: int,
    device: torch.device,
    split: str,
) -> pd.DataFrame:
    group_names = list(event_groups(cache_path, split_indices_np).keys())
    return evaluate_sweep(
        model=model,
        cache_path=cache_path,
        stats=stats,
        split_indices_np=split_indices_np,
        group_names=group_names,
        scales=[float(selected["normal_veto_scale"])],
        temperatures=[float(selected["normal_veto_temperature"])],
        betas=[float(selected["residual_beta"])],
        batch_size=batch_size,
        device=device,
        split=split,
    )


def write_summary(
    output_dir: Path,
    model_dir: Path,
    selected: pd.Series,
    score_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    score_cols = [
        "normal_veto_scale",
        "normal_veto_temperature",
        "residual_beta",
        "group_score",
        "group_affected_max",
        "overall_all_mae",
        "overall_affected_mae",
        "overall_unaffected_mae",
        "eligible",
    ]
    test_cols = [
        "group",
        "samples",
        "all_candidates_model_robust_mae",
        "affected_candidates_model_robust_mae",
        "unaffected_candidates_model_robust_mae",
    ]
    lines = [
        "# Group-Aware Normal-Veto Posthoc Selection",
        "",
        f"- model_dir: `{model_dir}`",
        f"- selection_groups: `{args.selection_groups}`",
        f"- selection_weights: `{args.selection_weights}`",
        f"- all_val_tolerance: `{args.all_val_tolerance}`",
        "",
        "## Selected Config",
        "",
        pd.DataFrame([selected.to_dict()])[score_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Validation Top Configs",
        "",
        score_df[score_cols].head(12).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Test Groups At Selected Config",
        "",
        test_df[test_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    stats = compute_stats(cache_path)
    splits = split_indices(cache_path)
    device = choose_device(args.device)

    selection_groups = parse_str_list(args.selection_groups)
    selection_weights = parse_float_list(args.selection_weights)
    scales = parse_float_list(args.sweep_scales)
    temperatures = parse_float_list(args.sweep_temperatures)
    betas = parse_float_list(args.sweep_betas)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)
    model = make_model(ckpt, cache_path, device)

    val_group_names = list(dict.fromkeys(["overall", *selection_groups]))
    val_df = evaluate_sweep(
        model=model,
        cache_path=cache_path,
        stats=stats,
        split_indices_np=splits["val"],
        group_names=val_group_names,
        scales=scales,
        temperatures=temperatures,
        betas=betas,
        batch_size=args.batch_size,
        device=device,
        split="val",
    )
    val_df.to_csv(output_dir / "val_group_sweep.csv", index=False)
    selected, score_df = select_config(val_df, selection_groups, selection_weights, args.all_val_tolerance)
    score_df.to_csv(output_dir / "val_selection_scores.csv", index=False)

    test_df = evaluate_selected(
        model=model,
        cache_path=cache_path,
        stats=stats,
        split_indices_np=splits["test"],
        selected=selected,
        batch_size=args.batch_size,
        device=device,
        split="test",
    )
    test_df.to_csv(output_dir / "test_group_metrics.csv", index=False)
    with (output_dir / "selected_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "normal_veto_scale": float(selected["normal_veto_scale"]),
                "normal_veto_temperature": float(selected["normal_veto_temperature"]),
                "residual_beta": float(selected["residual_beta"]),
                "group_score": float(selected["group_score"]),
                "group_affected_max": float(selected["group_affected_max"]),
                "overall_all_mae": float(selected["overall_all_mae"]),
                "overall_affected_mae": float(selected["overall_affected_mae"]),
                "overall_unaffected_mae": float(selected["overall_unaffected_mae"]),
                "selection_groups": selection_groups,
                "selection_weights": selection_weights,
                "all_val_tolerance": args.all_val_tolerance,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, model_dir, selected, score_df, test_df, args)
    print(f"selected scale={selected['normal_veto_scale']} temp={selected['normal_veto_temperature']} beta={selected['residual_beta']}", flush=True)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
