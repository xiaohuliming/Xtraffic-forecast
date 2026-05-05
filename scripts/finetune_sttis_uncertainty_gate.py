#!/usr/bin/env python3
"""Fine-tune branch uncertainty heads on a proposal-aware ST-TIS gate."""

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
from train_dual_branch_sttis_gate import DualBranchSTTISProposalUncertaintyGate, DualBranchSTTISUncertaintyGate
from train_full_candidate_stgnn_heatmap_model import CHANNELS, compute_stats, forecast_metrics_for_loader, make_loader, split_indices
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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_convexgate"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_uncertainty_gate"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--affected-weight", type=float, default=4.0)
    parser.add_argument("--risk-loss-weight", type=float, default=0.10)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.05)
    parser.add_argument("--convex-gate-loss-weight", type=float, default=0.05)
    parser.add_argument("--preference-temperature", type=float, default=0.20)
    parser.add_argument("--preference-margin", type=float, default=0.10)
    parser.add_argument("--convex-gate-min-gap", type=float, default=0.05)
    parser.add_argument("--train-uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-max", type=float, default=3.0)
    parser.add_argument(
        "--risk-input",
        choices=["branch", "proposal"],
        default="branch",
        help="Use branch representations or proposal-aware gate features for risk prediction.",
    )
    parser.add_argument("--proposal-feature-count", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--sweep-scales", default="0.0,0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="all_candidates_model_robust_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--eval-splits", default="val,test")
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
) -> DualBranchSTTISUncertaintyGate | DualBranchSTTISProposalUncertaintyGate:
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError("checkpoint args must be a dict")
    shapes = infer_cache_shapes(cache_path)
    model_class = DualBranchSTTISProposalUncertaintyGate if args.risk_input == "proposal" else DualBranchSTTISUncertaintyGate
    model = model_class(
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
        uncertainty_scale=args.train_uncertainty_scale,
        uncertainty_max=args.uncertainty_max,
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


def freeze_except_uncertainty(model: nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith("normal_uncertainty_head.")
            or name.startswith("incident_uncertainty_head.")
            or name.startswith("base_gate_logit_norm.")
        )
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no uncertainty parameters were marked trainable")
    return trainable


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def compute_uncertainty_loss(
    model: DualBranchSTTISUncertaintyGate,
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
    weights = 1.0 + args.affected_weight * affected_mask.to(y_mask.dtype)

    residual_loss_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_loss_raw, y_mask, weights)

    normal_abs = torch.abs(details["normal_residual"].detach() - y)
    incident_abs = torch.abs(details["incident_residual"].detach() - y)
    normal_target = torch.log1p(normal_abs)
    incident_target = torch.log1p(incident_abs)
    risk_raw = (
        nn.functional.smooth_l1_loss(details["normal_risk"], normal_target, reduction="none")
        + nn.functional.smooth_l1_loss(details["incident_risk"], incident_target, reduction="none")
    )
    risk_loss = masked_mean(risk_raw, y_mask, weights)

    incident_worse_gap = incident_abs - normal_abs
    ranking_target = torch.sigmoid(incident_worse_gap / max(args.preference_temperature, 1e-6))
    risk_diff_logit = details["incident_risk"] - details["normal_risk"]
    ranking_raw = nn.functional.binary_cross_entropy_with_logits(risk_diff_logit, ranking_target, reduction="none")
    if args.preference_margin > 0:
        ranking_weight = (incident_worse_gap.abs() / args.preference_margin).clamp(0.0, 1.0)
    else:
        ranking_weight = torch.ones_like(y_mask)
    ranking_weights = ranking_weight + args.affected_weight * affected_mask.to(ranking_weight.dtype)
    ranking_loss = masked_mean(ranking_raw, y_mask, ranking_weights)

    normal_detached = details["normal_residual"].detach()
    incident_detached = details["incident_residual"].detach()
    proposal_gap = incident_detached - normal_detached
    convex_target_mask = (proposal_gap.abs() > args.convex_gate_min_gap) & y_mask.bool()
    safe_gap = torch.where(convex_target_mask, proposal_gap, torch.ones_like(proposal_gap))
    convex_target = ((y - normal_detached) / safe_gap).clamp(0.0, 1.0)
    convex_gate_raw = nn.functional.smooth_l1_loss(details["gate"], convex_target, reduction="none")
    convex_gate_loss = masked_mean(convex_gate_raw, convex_target_mask.to(y_mask.dtype), weights)

    loss = (
        residual_loss
        + args.risk_loss_weight * risk_loss
        + args.ranking_loss_weight * ranking_loss
        + args.convex_gate_loss_weight * convex_gate_loss
    )
    with torch.no_grad():
        hard_rate = ((incident_worse_gap.abs() > args.preference_margin) & y_mask.bool()).to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        incident_worse_rate = ((incident_worse_gap > 0) & y_mask.bool()).to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "convex_gate_loss": float(convex_gate_loss.detach().cpu()),
        "hard_rate": float(hard_rate.detach().cpu()),
        "incident_worse_rate": float(incident_worse_rate.detach().cpu()),
        "gate_mean": float(masked_mean(details["gate"], y_mask).detach().cpu()),
        "base_gate_mean": float(masked_mean(details["base_gate"], y_mask).detach().cpu()),
        "risk_delta_mean": float(masked_mean(details["risk_delta"], y_mask).detach().cpu()),
    }


def evaluate_loss(
    model: DualBranchSTTISUncertaintyGate,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    keys = [
        "loss",
        "residual_loss",
        "risk_loss",
        "ranking_loss",
        "convex_gate_loss",
        "hard_rate",
        "incident_worse_rate",
        "gate_mean",
        "base_gate_mean",
        "risk_delta_mean",
    ]
    totals = {key: 0.0 for key in keys}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_uncertainty_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("ST-TIS uncertainty gate fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def sweep_scale_beta(
    model: DualBranchSTTISUncertaintyGate,
    loader: torch.utils.data.DataLoader,
    scales: list[float],
    betas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    original_scale = float(model.uncertainty_scale)
    rows = []
    try:
        for scale in scales:
            model.uncertainty_scale = float(scale)
            metrics_by_beta = forecast_metrics_for_loader(model, loader, betas, device)
            rows.extend({"uncertainty_scale": scale, "residual_beta": beta, **values} for beta, values in metrics_by_beta.items())
    finally:
        model.uncertainty_scale = original_scale
    return pd.DataFrame(rows)


def select_scale_beta(val_sweep: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float]:
    if args.selection_metric not in val_sweep.columns:
        raise KeyError(f"selection metric not found in sweep: {args.selection_metric}")
    all_metric = "all_candidates_model_robust_mae"
    best_all = float(val_sweep[all_metric].min())
    eligible = val_sweep[val_sweep[all_metric] <= best_all + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_sweep
    row = eligible.loc[eligible[args.selection_metric].idxmin()]
    return float(row["uncertainty_scale"]), float(row["residual_beta"])


def metrics_at(
    model: DualBranchSTTISUncertaintyGate,
    loader: torch.utils.data.DataLoader,
    uncertainty_scale: float,
    residual_beta: float,
    device: torch.device,
) -> dict[str, float]:
    old_scale = float(model.uncertainty_scale)
    model.uncertainty_scale = float(uncertainty_scale)
    try:
        return forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]
    finally:
        model.uncertainty_scale = old_scale


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    eval_counts: dict[str, int],
    uncertainty_scale: float,
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# ST-TIS Branch Uncertainty Gate",
        "",
        "This variant freezes the source model and trains branch error-risk heads that shift the gate logit.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- uncertainty-gate all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.4f}`",
        f"- uncertainty-gate affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.4f}`",
        f"- uncertainty_scale: {uncertainty_scale:.2f}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- affected_weight: {args.affected_weight}",
        f"- risk_loss_weight: {args.risk_loss_weight}",
        f"- ranking_loss_weight: {args.ranking_loss_weight}",
        f"- convex_gate_loss_weight: {args.convex_gate_loss_weight}",
        f"- uncertainty_max: {args.uncertainty_max}",
        f"- trainable uncertainty parameters: {trainable_params}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": split_counts[key], "eval_samples": eval_counts[key]} for key in split_counts]).to_markdown(index=False),
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
    trainable_params = freeze_except_uncertainty(model)
    print(f"trainable uncertainty parameters: {trainable_params}", flush=True)

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
            loss, _parts = compute_uncertainty_loss(model, batch, loss_args, device)
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
    betas = parse_float_list(args.sweep_betas)
    val_sweep = sweep_scale_beta(model, val_loader, scales, betas, device)
    val_sweep.to_csv(output_dir / "uncertainty_scale_beta_sweep.csv", index=False)
    uncertainty_scale, residual_beta = select_scale_beta(val_sweep, args)
    model.uncertainty_scale = uncertainty_scale

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        idx = eval_indices[split]
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = metrics_at(model, loader, uncertainty_scale, residual_beta, device)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    ckpt_args = dict(source_args)
    ckpt_args.update(json_safe_args(args))
    ckpt_args["model_class"] = "DualBranchSTTISProposalUncertaintyGate" if args.risk_input == "proposal" else "DualBranchSTTISUncertaintyGate"
    ckpt_args["training_variant"] = "uncertainty_gate_finetune"
    ckpt_args["risk_input"] = args.risk_input
    ckpt_args["uncertainty_scale"] = uncertainty_scale
    ckpt_args["uncertainty_max"] = args.uncertainty_max
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
                "samples": split_counts,
                "eval_samples": eval_counts,
                "uncertainty_scale": uncertainty_scale,
                "uncertainty_max": args.uncertainty_max,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
                "source_model_dir": str(model_dir),
                "source_residual_beta": float(ckpt.get("residual_beta", 1.0)),
                "selection_metric": args.selection_metric,
                "all_val_tolerance": args.all_val_tolerance,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = "DualBranchSTTISProposalUncertaintyGate" if args.risk_input == "proposal" else "DualBranchSTTISUncertaintyGate"
    config["training_variant"] = "uncertainty_gate_finetune"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    config["uncertainty_scale"] = uncertainty_scale
    config["residual_beta"] = residual_beta
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, eval_counts, uncertainty_scale, residual_beta, trainable_params, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
