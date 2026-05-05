#!/usr/bin/env python3
"""Compare two dual-branch checkpoints by incident severity/recovery groups."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid"),
    )
    parser.add_argument(
        "--candidate-model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_quickgrid"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/impact_aux_group_comparison_seed_23"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-beta", type=float, default=None, help="Override the base model residual beta.")
    parser.add_argument("--candidate-beta", type=float, default=None, help="Override the candidate model residual beta.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_cache_path(model_dir: Path, ckpt: dict[str, object]) -> Path:
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        cache_path = Path(data["cache_path"])
    return cache_path.resolve()


def residual_beta(model_dir: Path, ckpt: dict[str, object]) -> float:
    metrics_path = model_dir / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "residual_beta" in payload:
            return float(payload["residual_beta"])
    return float(ckpt.get("residual_beta", 1.0))


def read_event_groups(cache_path: Path, indices: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    with h5py.File(cache_path, "r") as h5:
        event_aux = h5["event_aux"][indices].astype(np.float32)
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "overall": (np.ones(indices.shape[0], dtype=bool), "all test events"),
        "severity_low": (severity <= q33, f"severity <= {q33:.3f}"),
        "severity_mid": ((severity > q33) & (severity <= q66), f"{q33:.3f} < severity <= {q66:.3f}"),
        "severity_high": (severity > q66, f"severity > {q66:.3f}"),
        "recovery_short_lt30": (recovery_min < 30.0, "recovery < 30 min"),
        "recovery_mid_30_90": ((recovery_min >= 30.0) & (recovery_min < 90.0), "30 <= recovery < 90 min"),
        "recovery_long_ge90": (recovery_min >= 90.0, "recovery >= 90 min"),
        "severity_high_and_long": ((severity > q66) & (recovery_min >= 90.0), "severity high and recovery >= 90 min"),
    }


def pick_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_all_mae": metrics["all_candidates_model_robust_mae"],
        f"{prefix}_affected_mae": metrics["affected_candidates_model_robust_mae"],
        f"{prefix}_unaffected_mae": metrics["unaffected_candidates_model_robust_mae"],
    }


def evaluate_model(
    model: torch.nn.Module,
    beta: float,
    cache_path: Path,
    stats: object,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = make_loader(cache_path, indices, stats, batch_size=batch_size, shuffle=False)
    return forecast_metrics_for_loader(model, loader, [beta], device)[beta]


def write_summary(output_dir: Path, df: pd.DataFrame, base_label: str, candidate_label: str) -> None:
    cols = [
        "group",
        "samples",
        "base_all_mae",
        "candidate_all_mae",
        "all_delta",
        "base_affected_mae",
        "candidate_affected_mae",
        "affected_delta",
        "base_unaffected_mae",
        "candidate_unaffected_mae",
        "unaffected_delta",
    ]
    lines = [
        "# Dual-Branch Group Comparison",
        "",
        f"Base: `{base_label}`",
        f"Candidate: `{candidate_label}`",
        "",
        "Negative delta means the candidate is better.",
        "",
        df[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    base_dir = args.base_model_dir.resolve()
    candidate_dir = args.candidate_model_dir.resolve()
    base_ckpt = torch_load(base_dir / "model.pt")
    candidate_ckpt = torch_load(candidate_dir / "model.pt")
    cache_path = resolve_cache_path(base_dir, base_ckpt)
    candidate_cache = resolve_cache_path(candidate_dir, candidate_ckpt)
    if candidate_cache != cache_path:
        raise ValueError(f"model cache mismatch: {cache_path} vs {candidate_cache}")

    stats = compute_stats(cache_path)
    splits = split_indices(cache_path)
    split_idx = splits[args.split]
    groups = read_event_groups(cache_path, split_idx)

    base_model = make_model(base_ckpt, cache_path, device)
    candidate_model = make_model(candidate_ckpt, cache_path, device)
    base_beta = residual_beta(base_dir, base_ckpt) if args.base_beta is None else float(args.base_beta)
    candidate_beta = residual_beta(candidate_dir, candidate_ckpt) if args.candidate_beta is None else float(args.candidate_beta)

    rows = []
    for group, (mask, label) in groups.items():
        idx = split_idx[mask]
        row: dict[str, float | str | int] = {"group": group, "label": label, "samples": int(idx.size)}
        if idx.size > 0:
            base_metrics = evaluate_model(base_model, base_beta, cache_path, stats, idx, args.batch_size, device)
            candidate_metrics = evaluate_model(candidate_model, candidate_beta, cache_path, stats, idx, args.batch_size, device)
            row.update(pick_metrics(base_metrics, "base"))
            row.update(pick_metrics(candidate_metrics, "candidate"))
            for target in ["all", "affected", "unaffected"]:
                row[f"{target}_delta"] = float(row[f"candidate_{target}_mae"]) - float(row[f"base_{target}_mae"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "group_metrics.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model_dir": str(base_dir),
                "candidate_model_dir": str(candidate_dir),
                "cache_path": str(cache_path),
                "split": args.split,
                "base_beta": base_beta,
                "candidate_beta": candidate_beta,
                "batch_size": args.batch_size,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, df, base_dir.name, candidate_dir.name)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
