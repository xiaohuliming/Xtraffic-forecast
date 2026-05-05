#!/usr/bin/env python3
"""Evaluate temporal-decay gains by severity, recovery, and forecast horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from train_full_candidate_stgnn_heatmap_model import (
    CHANNELS,
    FullCandidateSTGNNHeatmap,
    compute_stats,
    forecast_metrics_for_loader,
    make_loader,
    split_indices,
)
from train_impact_residual_model import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"),
    )
    parser.add_argument(
        "--no-decay-model",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty/model.pt"),
    )
    parser.add_argument(
        "--decay-model",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/model.pt"),
    )
    parser.add_argument(
        "--no-decay-metrics",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty/metrics.json"),
    )
    parser.add_argument(
        "--decay-metrics",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/metrics.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/impact_guided_next_stage/decay_group_analysis"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def _torch_load(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(model_path: Path, device: torch.device) -> tuple[FullCandidateSTGNNHeatmap, dict[str, object]]:
    ckpt = _torch_load(model_path)
    model_args = dict(ckpt["args"])  # type: ignore[arg-type]
    use_dual = bool(model_args.get("use_dual_hist_residual", False))
    model = FullCandidateSTGNNHeatmap(
        channels=len(CHANNELS),
        hist_input_channels=len(CHANNELS) * (2 if use_dual else 1),
        node_context_dim=8,
        global_context_dim=17,
        horizon_steps=int(model_args.get("horizon_steps", 12)),
        hidden_dim=int(model_args.get("hidden_dim", 96)),
        graph_layers=int(model_args.get("graph_layers", 2)),
        dropout=float(model_args.get("dropout", 0.10)),
        graph_sigma=float(model_args.get("graph_sigma", 3.0)),
        graph_mode=str(model_args.get("graph_mode", "undirected")),
        use_normal_delta=bool(model_args.get("use_normal_delta", False)),
        use_normal_delta_abs=bool(model_args.get("use_normal_delta_abs", False)),
        use_temporal_decay_head=bool(model_args.get("use_temporal_decay_head", False)),
    ).to(device)
    state = ckpt["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unexpected model state in {model_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_args


def residual_beta(metrics_path: Path) -> float:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(payload.get("residual_beta", 1.0))


def read_group_arrays(cache_path: Path, test_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(cache_path, "r") as h5:
        event_aux = h5["event_aux"][test_indices].astype(np.float32)
    severity = np.expm1(event_aux[:, 0])
    recovery_min = event_aux[:, 1] * 180.0
    severity[~np.isfinite(severity)] = 0.0
    recovery_min[~np.isfinite(recovery_min)] = 0.0
    return severity, recovery_min


def build_severity_groups(test_indices: np.ndarray, severity: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    specs = [
        ("severity_low", severity <= q33, f"severity <= {q33:.3f}"),
        ("severity_mid", (severity > q33) & (severity <= q66), f"{q33:.3f} < severity <= {q66:.3f}"),
        ("severity_high", severity > q66, f"severity > {q66:.3f}"),
    ]
    return [(name, test_indices[mask], label) for name, mask, label in specs]


def build_recovery_groups(test_indices: np.ndarray, recovery_min: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    specs = [
        ("recovery_short_lt30", recovery_min < 30.0, "recovery < 30 min"),
        ("recovery_mid_30_90", (recovery_min >= 30.0) & (recovery_min < 90.0), "30 <= recovery < 90 min"),
        ("recovery_long_ge90", recovery_min >= 90.0, "recovery >= 90 min"),
    ]
    return [(name, test_indices[mask], label) for name, mask, label in specs]


def select_metrics(row: dict[str, float], prefix: str = "") -> dict[str, float]:
    out = {
        f"{prefix}all_model_robust_mae": row["all_candidates_model_robust_mae"],
        f"{prefix}affected_model_robust_mae": row["affected_candidates_model_robust_mae"],
        f"{prefix}unaffected_model_robust_mae": row["unaffected_candidates_model_robust_mae"],
        f"{prefix}all_improvement_pct": row["all_candidates_improvement_pct"],
        f"{prefix}affected_improvement_pct": row["affected_candidates_improvement_pct"],
    }
    for step in range(1, 13):
        out[f"{prefix}h{step:02d}_all_model_robust_mae"] = row[f"horizon_{step:02d}_all_candidates_model_robust_mae"]
        out[f"{prefix}h{step:02d}_affected_model_robust_mae"] = row[
            f"horizon_{step:02d}_affected_candidates_model_robust_mae"
        ]
    return out


def evaluate_group(
    name: str,
    label: str,
    indices: np.ndarray,
    cache_path: Path,
    stats: object,
    batch_size: int,
    no_decay_model: FullCandidateSTGNNHeatmap,
    decay_model: FullCandidateSTGNNHeatmap,
    no_decay_beta: float,
    decay_beta: float,
    device: torch.device,
) -> dict[str, float | str | int]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return {"group": name, "label": label, "samples": 0}
    loader = make_loader(cache_path, indices, stats, batch_size=batch_size, shuffle=False)
    no_decay = forecast_metrics_for_loader(no_decay_model, loader, [no_decay_beta], device)[no_decay_beta]
    loader = make_loader(cache_path, indices, stats, batch_size=batch_size, shuffle=False)
    decay = forecast_metrics_for_loader(decay_model, loader, [decay_beta], device)[decay_beta]
    row: dict[str, float | str | int] = {"group": name, "label": label, "samples": int(indices.size)}
    row.update(select_metrics(no_decay, "no_decay_"))
    row.update(select_metrics(decay, "decay_"))
    for target in ["all", "affected", "unaffected"]:
        no_val = float(row[f"no_decay_{target}_model_robust_mae"])
        de_val = float(row[f"decay_{target}_model_robust_mae"])
        row[f"{target}_decay_delta"] = de_val - no_val
        row[f"{target}_decay_gain_pct"] = 100.0 * (no_val - de_val) / no_val if no_val > 0 else float("nan")
    for step in range(1, 13):
        no_val = float(row[f"no_decay_h{step:02d}_affected_model_robust_mae"])
        de_val = float(row[f"decay_h{step:02d}_affected_model_robust_mae"])
        row[f"h{step:02d}_affected_decay_delta"] = de_val - no_val
        row[f"h{step:02d}_affected_decay_gain_pct"] = 100.0 * (no_val - de_val) / no_val if no_val > 0 else float("nan")
    return row


def horizon_comparison(overall_row: dict[str, float | str | int]) -> pd.DataFrame:
    rows = []
    for step in range(1, 13):
        no_aff = float(overall_row[f"no_decay_h{step:02d}_affected_model_robust_mae"])
        de_aff = float(overall_row[f"decay_h{step:02d}_affected_model_robust_mae"])
        no_all = float(overall_row[f"no_decay_h{step:02d}_all_model_robust_mae"])
        de_all = float(overall_row[f"decay_h{step:02d}_all_model_robust_mae"])
        rows.append(
            {
                "horizon": step,
                "no_decay_all_model_robust_mae": no_all,
                "decay_all_model_robust_mae": de_all,
                "all_decay_delta": de_all - no_all,
                "all_decay_gain_pct": 100.0 * (no_all - de_all) / no_all if no_all > 0 else float("nan"),
                "no_decay_affected_model_robust_mae": no_aff,
                "decay_affected_model_robust_mae": de_aff,
                "affected_decay_delta": de_aff - no_aff,
                "affected_decay_gain_pct": 100.0 * (no_aff - de_aff) / no_aff if no_aff > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    severity_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    overall_row: dict[str, float | str | int],
) -> None:
    def compact(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "group",
            "samples",
            "no_decay_all_model_robust_mae",
            "decay_all_model_robust_mae",
            "all_decay_gain_pct",
            "no_decay_affected_model_robust_mae",
            "decay_affected_model_robust_mae",
            "affected_decay_gain_pct",
            "h06_affected_decay_gain_pct",
            "h12_affected_decay_gain_pct",
        ]
        return df[cols]

    horizon_cols = [
        "horizon",
        "no_decay_all_model_robust_mae",
        "decay_all_model_robust_mae",
        "all_decay_gain_pct",
        "no_decay_affected_model_robust_mae",
        "decay_affected_model_robust_mae",
        "affected_decay_gain_pct",
    ]
    lines = ["# Decay Group Analysis", ""]
    lines.append("This compares the no-decay disagreement-proxy model against the temporal-decay-head model on the shared test split.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    overall_df = pd.DataFrame([overall_row])
    lines.append(compact(overall_df).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Severity Groups")
    lines.append("")
    lines.append(compact(severity_df).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Recovery Groups")
    lines.append("")
    lines.append(compact(recovery_df).to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Horizon Comparison")
    lines.append("")
    lines.append(horizon_df[horizon_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Negative `decay_delta` means the temporal decay head has lower robust MAE.")
    lines.append("- Positive `decay_gain_pct` means the temporal decay head improves over no-decay.")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    cache_path = args.cache_path.resolve()
    stats = compute_stats(cache_path)
    split = split_indices(cache_path)
    test_indices = split["test"]
    severity, recovery_min = read_group_arrays(cache_path, test_indices)

    print("loading models", flush=True)
    no_decay_model, _ = load_model(args.no_decay_model.resolve(), device)
    decay_model, _ = load_model(args.decay_model.resolve(), device)
    no_decay_beta = residual_beta(args.no_decay_metrics.resolve())
    decay_beta = residual_beta(args.decay_metrics.resolve())

    overall_row = evaluate_group(
        name="overall_test",
        label="all test samples",
        indices=test_indices,
        cache_path=cache_path,
        stats=stats,
        batch_size=args.batch_size,
        no_decay_model=no_decay_model,
        decay_model=decay_model,
        no_decay_beta=no_decay_beta,
        decay_beta=decay_beta,
        device=device,
    )
    print(
        f"overall affected gain={float(overall_row['affected_decay_gain_pct']):.3f}%",
        flush=True,
    )

    severity_rows = []
    for name, idx, label in build_severity_groups(test_indices, severity):
        print(f"evaluating {name}: {idx.size} samples", flush=True)
        severity_rows.append(
            evaluate_group(
                name,
                label,
                idx,
                cache_path,
                stats,
                args.batch_size,
                no_decay_model,
                decay_model,
                no_decay_beta,
                decay_beta,
                device,
            )
        )
    recovery_rows = []
    for name, idx, label in build_recovery_groups(test_indices, recovery_min):
        print(f"evaluating {name}: {idx.size} samples", flush=True)
        recovery_rows.append(
            evaluate_group(
                name,
                label,
                idx,
                cache_path,
                stats,
                args.batch_size,
                no_decay_model,
                decay_model,
                no_decay_beta,
                decay_beta,
                device,
            )
        )

    severity_df = pd.DataFrame(severity_rows)
    recovery_df = pd.DataFrame(recovery_rows)
    horizon_df = horizon_comparison(overall_row)

    pd.DataFrame([overall_row]).to_csv(output_dir / "overall_metrics.csv", index=False)
    severity_df.to_csv(output_dir / "severity_group_metrics.csv", index=False)
    recovery_df.to_csv(output_dir / "recovery_group_metrics.csv", index=False)
    horizon_df.to_csv(output_dir / "horizon_comparison.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": overall_row,
                "severity": severity_rows,
                "recovery": recovery_rows,
                "horizon": horizon_df.to_dict(orient="records"),
                "no_decay_beta": no_decay_beta,
                "decay_beta": decay_beta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, severity_df, recovery_df, horizon_df, overall_row)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
