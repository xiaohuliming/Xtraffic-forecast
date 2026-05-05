#!/usr/bin/env python3
"""Evaluate a saved impact correction adapter checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from analyze_dual_branch_gate import make_model, torch_load
from compare_dual_branch_group_metrics import resolve_cache_path
from train_dual_branch_gate_baseline import cap_indices, infer_cache_shapes
from train_full_candidate_stgnn_heatmap_model import compute_stats, forecast_metrics_for_loader, make_loader, split_indices
from train_impact_correction_adapter import ImpactCorrectionAdapter, write_group_metrics
from train_impact_residual_model import choose_device


def parse_split_list(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val", "test"}
    bad = [item for item in splits if item not in allowed]
    if bad:
        raise ValueError(f"unsupported eval split(s): {bad}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--write-group-metrics", action="store_true")
    parser.add_argument("--group-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--override-correction-anomaly-gate-threshold", type=float, default=None)
    parser.add_argument("--override-correction-anomaly-gate-temperature", type=float, default=None)
    parser.add_argument("--override-correction-anomaly-gate-floor", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def load_adapter(
    adapter_dir: Path,
    device: torch.device,
    config_overrides: dict[str, float | str] | None = None,
) -> tuple[ImpactCorrectionAdapter, Path]:
    ckpt = torch_load(adapter_dir / "model.pt")
    config = json.loads((adapter_dir / "config.json").read_text(encoding="utf-8"))
    if config_overrides:
        config.update(config_overrides)
    source_dir = Path(str(ckpt["source_model_dir"]))
    source_ckpt = torch_load(source_dir / "model.pt")
    cache_path = resolve_cache_path(source_dir, source_ckpt)
    shapes = infer_cache_shapes(cache_path)
    base_model = make_model(source_ckpt, cache_path, device)
    model = ImpactCorrectionAdapter(
        base_model=base_model,
        base_beta=float(ckpt["base_beta"]),
        channels=int(shapes["channels"]),
        horizon_steps=int(shapes["horizon_steps"]),
        global_context_dim=int(shapes["global_context_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        max_correction=float(config["max_correction"]),
        correction_node_gate_mode=str(config.get("correction_node_gate_mode", "none")),
        correction_node_gate_floor=float(config.get("correction_node_gate_floor", 0.0)),
        correction_node_gate_temperature=float(config.get("correction_node_gate_temperature", 1.0)),
        correction_anomaly_gate_mode=str(config.get("correction_anomaly_gate_mode", "none")),
        correction_anomaly_gate_threshold=float(config.get("correction_anomaly_gate_threshold", 0.5)),
        correction_anomaly_gate_temperature=float(config.get("correction_anomaly_gate_temperature", 0.25)),
        correction_anomaly_gate_floor=float(config.get("correction_anomaly_gate_floor", 0.0)),
    ).to(device)
    state = model.state_dict()
    state.update(ckpt["adapter_state_dict"])
    model.load_state_dict(state, strict=True)
    return model, cache_path


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = (args.output_dir or (adapter_dir / "adapter_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    config_overrides: dict[str, float] = {}
    if args.override_correction_anomaly_gate_threshold is not None:
        config_overrides["correction_anomaly_gate_threshold"] = float(args.override_correction_anomaly_gate_threshold)
    if args.override_correction_anomaly_gate_temperature is not None:
        config_overrides["correction_anomaly_gate_temperature"] = float(args.override_correction_anomaly_gate_temperature)
    if args.override_correction_anomaly_gate_floor is not None:
        config_overrides["correction_anomaly_gate_floor"] = float(args.override_correction_anomaly_gate_floor)
    model, cache_path = load_adapter(adapter_dir, device, config_overrides or None)
    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)
    eval_indices = {
        split: cap_indices(idx, args.max_eval_samples, args.seed + offset)
        for offset, (split, idx) in enumerate(indices.items())
    }

    rows = []
    source_rows = []
    for split in parse_split_list(args.eval_splits):
        loader = make_loader(cache_path, eval_indices[split], stats, args.batch_size, shuffle=False)
        adapter_metrics = forecast_metrics_for_loader(model, loader, [1.0], device)[1.0]
        source_metrics = forecast_metrics_for_loader(model.base_model, loader, [model.base_beta], device)[model.base_beta]
        rows.append({"split": split, **adapter_metrics})
        source_rows.append({"split": split, **source_metrics})

    adapter_df = pd.DataFrame(rows)
    source_df = pd.DataFrame(source_rows)
    key_cols = [
        "all_candidates_model_robust_mae",
        "affected_candidates_model_robust_mae",
        "unaffected_candidates_model_robust_mae",
    ]
    delta_rows = []
    for _, adapter_row in adapter_df.iterrows():
        split = str(adapter_row["split"])
        source_row = source_df[source_df["split"] == split].iloc[0]
        delta_rows.append({"split": split, **{col: float(adapter_row[col]) - float(source_row[col]) for col in key_cols}})
    delta_df = pd.DataFrame(delta_rows)
    adapter_df.to_csv(output_dir / "adapter_metrics.csv", index=False)
    source_df.to_csv(output_dir / "source_metrics.csv", index=False)
    delta_df.to_csv(output_dir / "delta_metrics.csv", index=False)
    if args.write_group_metrics:
        write_group_metrics(output_dir, model, cache_path, stats, indices[args.group_split], args, device)

    cols = ["split", *key_cols]
    lines = [
        "# Impact Correction Adapter Evaluation",
        "",
        f"- adapter_dir: `{adapter_dir}`",
        f"- cache_path: `{cache_path}`",
        f"- config_overrides: `{config_overrides}`",
        "",
        "Negative delta means the adapter is better.",
        "",
        "## Delta",
        "",
        delta_df[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Adapter",
        "",
        adapter_df[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Source",
        "",
        source_df[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
