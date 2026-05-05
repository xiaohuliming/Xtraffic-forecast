#!/usr/bin/env python3
"""Fine-tune a conservative incident-branch veto on a proposal-aware ST-TIS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from analyze_dual_branch_gate import torch_load
from train_dual_branch_gate_baseline import cap_indices, infer_cache_shapes
from train_dual_branch_sttis_gate import DualBranchSTTISVetoGate
from train_full_candidate_stgnn_heatmap_model import (
    CHANNELS,
    compute_stats,
    forecast_metrics_for_loader,
    make_loader,
    region_codes,
    split_indices,
)
from train_impact_residual_model import choose_device, json_safe_args


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_split_list(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val", "test"}
    bad = [item for item in splits if item not in allowed]
    if bad:
        raise ValueError(f"unsupported eval split(s): {bad}")
    if "test" not in splits:
        splits.append("test")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_23"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_veto_gate_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--veto-loss-weight", type=float, default=0.05)
    parser.add_argument("--hard-residual-weight", type=float, default=0.10)
    parser.add_argument("--veto-sparsity-weight", type=float, default=0.002)
    parser.add_argument("--affected-weight", type=float, default=0.0)
    parser.add_argument("--preference-temperature", type=float, default=0.20)
    parser.add_argument("--preference-margin", type=float, default=0.10)
    parser.add_argument("--hard-gate-min", type=float, default=0.40)
    parser.add_argument("--hard-weight", type=float, default=4.0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--train-veto-scale", type=float, default=1.0)
    parser.add_argument("--veto-max", type=float, default=2.0)
    parser.add_argument("--veto-init-bias", type=float, default=-6.0)
    parser.add_argument("--proposal-feature-count", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--sweep-scales", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="all_candidates_model_robust_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--eval-splits", default="val,test")
    parser.add_argument("--write-region-metrics", action="store_true")
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
        data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        cache_path = Path(data["cache_path"])
    return cache_path.resolve()


def build_model(
    ckpt: dict[str, object],
    cache_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> DualBranchSTTISVetoGate:
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError("checkpoint args must be a dict")
    shapes = infer_cache_shapes(cache_path)
    model = DualBranchSTTISVetoGate(
        channels=shapes["channels"],
        hist_input_channels=len(CHANNELS) * (2 if bool(model_args.get("use_dual_hist_residual", True)) else 1),
        node_context_dim=shapes["node_context_dim"],
        global_context_dim=shapes["global_context_dim"],
        horizon_steps=shapes["horizon_steps"],
        hidden_dim=int(model_args.get("hidden_dim", 96)),
        graph_layers=int(model_args.get("graph_layers", 2)),
        dropout=float(model_args.get("dropout", 0.10)),
        graph_sigma=float(model_args.get("graph_sigma", 3.0)),
        graph_mode=str(model_args.get("graph_mode", "undirected")),
        use_normal_delta=bool(model_args.get("use_normal_delta", True)),
        use_normal_delta_abs=bool(model_args.get("use_normal_delta_abs", True)),
        sttis_heads=int(model_args.get("sttis_heads", 4)),
        sttis_temporal_layers=int(model_args.get("sttis_temporal_layers", 1)),
        sttis_spatial_topk=int(model_args.get("sttis_spatial_topk", 8)),
        sttis_adj_bias=float(model_args.get("sttis_adj_bias", 0.25)),
        proposal_feature_count=args.proposal_feature_count,
        veto_scale=args.train_veto_scale,
        veto_max=args.veto_max,
        veto_init_bias=args.veto_init_bias,
    )
    source_state = ckpt["model_state_dict"]
    if not isinstance(source_state, dict):
        raise TypeError("checkpoint model_state_dict must be a dict")
    state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in state and tuple(state[key].shape) == tuple(value.shape)
    }
    state.update(compatible)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def freeze_except_veto(model: nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("veto_head.") or name.startswith("base_gate_logit_norm.")
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no veto parameters were marked trainable")
    return trainable


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def compute_veto_loss(
    model: DualBranchSTTISVetoGate,
    batch: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
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
        _node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    affected_mask = node_affected[:, None, :, None].bool() & y_mask.bool()
    affected_weights = 1.0 + args.affected_weight * affected_mask.to(y_mask.dtype)
    residual_loss_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_loss_raw, y_mask, affected_weights)

    normal_error = nn.functional.smooth_l1_loss(details["normal_residual"], y, reduction="none").detach()
    incident_error = nn.functional.smooth_l1_loss(details["incident_residual"], y, reduction="none").detach()
    incident_worse_gap = incident_error - normal_error
    target = torch.sigmoid((incident_worse_gap - args.preference_margin) / max(args.preference_temperature, 1e-6))
    base_gate = details["base_gate"].detach()
    hard_negative = (incident_worse_gap > args.preference_margin) & (base_gate > args.hard_gate_min) & y_mask.bool()
    positive = (incident_worse_gap > args.preference_margin) & y_mask.bool()
    weights = (
        torch.ones_like(target)
        + args.hard_weight * hard_negative.to(target.dtype)
        + args.positive_weight * positive.to(target.dtype)
        + args.affected_weight * affected_mask.to(target.dtype)
    )
    veto_raw = nn.functional.binary_cross_entropy_with_logits(details["veto_logits"], target, reduction="none")
    veto_loss = masked_mean(veto_raw, y_mask, weights)
    hard_residual_loss = (
        masked_mean(residual_loss_raw, hard_negative.to(y_mask.dtype), affected_weights)
        if hard_negative.any()
        else residual_loss * 0.0
    )
    sparsity_loss = masked_mean(details["veto_amount"], y_mask)
    loss = (
        residual_loss
        + args.veto_loss_weight * veto_loss
        + args.hard_residual_weight * hard_residual_loss
        + args.veto_sparsity_weight * sparsity_loss
    )
    with torch.no_grad():
        gate = details["gate"]
        hard_rate = hard_negative.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        affected_rate = affected_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "veto_loss": float(veto_loss.detach().cpu()),
        "hard_residual_loss": float(hard_residual_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "hard_negative_rate": float(hard_rate.detach().cpu()),
        "affected_rate": float(affected_rate.detach().cpu()),
        "gate_mean": float(masked_mean(gate, y_mask).detach().cpu()),
        "base_gate_mean": float(masked_mean(base_gate, y_mask).detach().cpu()),
        "veto_mean": float(sparsity_loss.detach().cpu()),
    }


def evaluate_loss(
    model: DualBranchSTTISVetoGate,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "residual_loss": 0.0,
        "veto_loss": 0.0,
        "hard_residual_loss": 0.0,
        "sparsity_loss": 0.0,
        "hard_negative_rate": 0.0,
        "affected_rate": 0.0,
        "gate_mean": 0.0,
        "base_gate_mean": 0.0,
        "veto_mean": 0.0,
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_veto_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("ST-TIS veto adapter fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def sweep_scale_beta(
    model: DualBranchSTTISVetoGate,
    loader: torch.utils.data.DataLoader,
    scales: list[float],
    betas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    old_scale = float(model.veto_scale)
    rows = []
    try:
        for scale in scales:
            model.veto_scale = float(scale)
            metrics_by_beta = forecast_metrics_for_loader(model, loader, betas, device)
            rows.extend({"veto_scale": scale, "residual_beta": beta, **values} for beta, values in metrics_by_beta.items())
    finally:
        model.veto_scale = old_scale
    return pd.DataFrame(rows)


def metrics_at(
    model: DualBranchSTTISVetoGate,
    loader: torch.utils.data.DataLoader,
    veto_scale: float,
    residual_beta: float,
    device: torch.device,
) -> dict[str, float]:
    old_scale = float(model.veto_scale)
    model.veto_scale = float(veto_scale)
    try:
        return forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]
    finally:
        model.veto_scale = old_scale


def select_row(val_sweep: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.selection_metric not in val_sweep.columns:
        raise KeyError(f"selection metric not found in sweep: {args.selection_metric}")
    all_metric = "all_candidates_model_robust_mae"
    all_best = float(val_sweep[all_metric].min())
    eligible = val_sweep[val_sweep[all_metric] <= all_best + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_sweep
    return eligible.loc[eligible[args.selection_metric].idxmin()]


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    veto_scale: float,
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# ST-TIS Hard-Negative Veto Gate",
        "",
        "This variant freezes the proposal-aware model and trains only a conservative incident-branch veto adapter.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- veto-gate all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.4f}`",
        f"- veto-gate affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.4f}`",
        f"- veto_scale: {veto_scale:.2f}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- veto_loss_weight: {args.veto_loss_weight}",
        f"- hard_residual_weight: {args.hard_residual_weight}",
        f"- veto_sparsity_weight: {args.veto_sparsity_weight}",
        f"- affected_weight: {args.affected_weight}",
        f"- preference_margin: {args.preference_margin}",
        f"- hard_gate_min: {args.hard_gate_min}",
        f"- veto_max: {args.veto_max}",
        f"- selection_metric: `{args.selection_metric}`",
        f"- all_val_tolerance: `{args.all_val_tolerance}`",
        f"- trainable veto parameters: {trainable_params}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": value} for key, value in split_counts.items()]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        pd.DataFrame([{"split": split, **values} for split, values in metrics.items()]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Training",
        "",
        f"- best_epoch: {int(log_df.loc[log_df['val_loss'].idxmin(), 'epoch']) if not log_df.empty else 'n/a'}",
        f"- best_val_loss: {float(log_df['val_loss'].min()) if not log_df.empty else float('nan'):.4f}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"source model: {model_dir}", flush=True)

    model = build_model(ckpt, cache_path, args, device)
    trainable_params = freeze_except_veto(model)
    print(f"trainable veto parameters: {trainable_params}", flush=True)

    source_args = ckpt.get("args", {})
    if not isinstance(source_args, dict):
        source_args = {}
    loss_args = argparse.Namespace(**vars(args))
    loss_args.use_dual_hist_residual = bool(source_args.get("use_dual_hist_residual", True))

    stats = compute_stats(cache_path)
    indices = split_indices(cache_path)
    eval_indices = {
        split: cap_indices(idx, args.max_eval_samples, args.seed + offset)
        for offset, (split, idx) in enumerate(indices.items())
    }
    train_indices_full = indices["train"]
    if args.max_train_samples > 0 and train_indices_full.size > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        train_indices = np.sort(rng.choice(train_indices_full, size=args.max_train_samples, replace=False))
    else:
        train_indices = train_indices_full

    train_loader = make_loader(cache_path, train_indices, stats, args.batch_size, shuffle=True)
    val_loader = make_loader(cache_path, eval_indices["val"], stats, args.batch_size, shuffle=False)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _parts = compute_veto_loss(model, batch, loss_args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loss(model, val_loader, loss_args, device)
        val_loss = float(val_metrics["loss"])
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    scales = parse_float_list(args.sweep_scales)
    beta_candidates = parse_float_list(args.sweep_betas)
    val_sweep = sweep_scale_beta(model, val_loader, scales, beta_candidates, device)
    val_sweep.to_csv(output_dir / "veto_scale_beta_sweep.csv", index=False)
    best_row = select_row(val_sweep, args)
    veto_scale = float(best_row["veto_scale"])
    residual_beta = float(best_row["residual_beta"])
    model.veto_scale = veto_scale

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        idx = eval_indices[split]
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = metrics_at(model, loader, veto_scale, residual_beta, device)

    region_metrics = []
    if args.write_region_metrics:
        region_code_arr = region_codes(cache_path)
        for code in sorted(np.unique(region_code_arr[eval_indices["test"]]).tolist()):
            mask_idx = indices["test"][region_code_arr[indices["test"]] == code]
            mask_idx = cap_indices(mask_idx, args.max_eval_samples, args.seed + 100 + int(code))
            loader = make_loader(cache_path, mask_idx, stats, args.batch_size, shuffle=False)
            row = {"region_code": int(code), "samples": int(mask_idx.size)}
            row.update(metrics_at(model, loader, veto_scale, residual_beta, device))
            region_metrics.append(row)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    ckpt_args = dict(source_args)
    ckpt_args.update(json_safe_args(args))
    ckpt_args["model_class"] = "DualBranchSTTISVetoGate"
    ckpt_args["training_variant"] = "hard_negative_veto_gate"
    ckpt_args["veto_scale"] = veto_scale
    ckpt_args["residual_beta"] = residual_beta
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": ckpt_args,
            "residual_beta": residual_beta,
        },
        output_dir / "model.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "region_metrics": region_metrics,
                "samples": split_counts,
                "eval_samples": eval_counts,
                "veto_scale": veto_scale,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
                "source_model_dir": str(model_dir),
                "source_residual_beta": float(ckpt.get("residual_beta", 1.0)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = "DualBranchSTTISVetoGate"
    config["training_variant"] = "hard_negative_veto_gate"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    config["veto_scale"] = veto_scale
    config["residual_beta"] = residual_beta
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, veto_scale, residual_beta, trainable_params, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
