#!/usr/bin/env python3
"""Fine-tune the ST-TIS incident branch with affected-node emphasis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from analyze_dual_branch_gate import make_model, torch_load
from train_dual_branch_gate_baseline import cap_indices
from train_full_candidate_stgnn_heatmap_model import compute_stats, forecast_metrics_for_loader, make_loader, split_indices
from train_impact_residual_model import choose_device, json_safe_args


INCIDENT_PREFIXES = (
    "incident_temporal_encoder.",
    "incident_spatial_layers.",
    "incident_context_proj.",
    "incident_input_norm.",
    "incident_decoder.",
)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--affected-weight", type=float, default=4.0)
    parser.add_argument("--incident-loss-weight", type=float, default=0.35)
    parser.add_argument("--gate-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--normal-better-gate-loss-weight",
        type=float,
        default=0.0,
        help="Extra conservative gate loss when the normal branch is locally better than the incident branch.",
    )
    parser.add_argument("--preference-temperature", type=float, default=0.20)
    parser.add_argument("--hard-margin", type=float, default=0.10)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--normal-better-margin", type=float, default=0.10)
    parser.add_argument(
        "--normal-better-min-gate",
        type=float,
        default=0.0,
        help="Only apply the normal-better gate loss where the current gate is above this threshold.",
    )
    parser.add_argument(
        "--convex-gate-loss-weight",
        type=float,
        default=0.0,
        help="Distill the gate toward the elementwise optimal convex fusion coefficient.",
    )
    parser.add_argument(
        "--convex-gate-min-gap",
        type=float,
        default=0.05,
        help="Ignore convex gate targets when normal/incident residual proposals are nearly identical.",
    )
    parser.add_argument(
        "--severity-high-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples above the training severity 66th percentile.",
    )
    parser.add_argument(
        "--recovery-long-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples whose recovery duration is at least 90 minutes.",
    )
    parser.add_argument(
        "--high-long-weight",
        type=float,
        default=0.0,
        help="Extra affected loss weight for samples that are both severity-high and recovery-long.",
    )
    parser.add_argument(
        "--tail-weight-max",
        type=float,
        default=3.0,
        help="Maximum additive high-risk loss weight; set <= 0 to disable clipping.",
    )
    parser.add_argument("--train-incident", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument(
        "--eval-splits",
        default="train,val,test",
        help="Comma-separated splits to report after training. Use val,test for faster ablations.",
    )
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_candidates_model_robust_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.003)
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


def freeze_for_incident_finetune(model: nn.Module, train_incident: bool, train_gate: bool) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        is_incident = train_incident and name.startswith(INCIDENT_PREFIXES)
        is_gate = train_gate and (name.startswith("gate_head.") or name.startswith("proposal_norm."))
        param.requires_grad = is_incident or is_gate
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no incident/gate parameters were marked trainable")
    return trainable


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def compute_tail_thresholds(cache_path: Path, indices: np.ndarray, stats: object) -> dict[str, float]:
    with h5py.File(cache_path, "r") as h5:
        raw_event = h5["event_aux"][np.sort(indices)].astype(np.float32)
    severity_raw_q66 = float(np.quantile(raw_event[:, 0], 2.0 / 3.0))
    recovery_raw_90min = 90.0 / 180.0
    event_mean = np.asarray(getattr(stats, "event_aux_mean"), dtype=np.float32)
    event_std = np.asarray(getattr(stats, "event_aux_std"), dtype=np.float32)
    return {
        "severity_high_z_threshold": float((severity_raw_q66 - event_mean[0]) / max(float(event_std[0]), 1e-6)),
        "recovery_long_z_threshold": float((recovery_raw_90min - event_mean[1]) / max(float(event_std[1]), 1e-6)),
        "severity_high_raw_threshold": severity_raw_q66,
        "recovery_long_raw_threshold": recovery_raw_90min,
    }


def tail_focus(event_aux: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    severity_high = event_aux[:, 0] >= float(getattr(args, "severity_high_z_threshold", float("inf")))
    recovery_long = event_aux[:, 1] >= float(getattr(args, "recovery_long_z_threshold", float("inf")))
    high_long = severity_high & recovery_long
    focus = (
        args.severity_high_weight * severity_high.to(event_aux.dtype)
        + args.recovery_long_weight * recovery_long.to(event_aux.dtype)
        + args.high_long_weight * high_long.to(event_aux.dtype)
    )
    if args.tail_weight_max > 0.0:
        focus = focus.clamp(max=args.tail_weight_max)
    return focus, severity_high, recovery_long, high_long


def compute_loss(
    model: nn.Module,
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
        event_aux,
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
    tail_weight, severity_high, recovery_long, high_long = tail_focus(event_aux, args)
    tail_weight_map = tail_weight[:, None, None, None].expand_as(y)
    weights = weights * (1.0 + tail_weight_map * affected_mask.to(y_mask.dtype))

    final_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    final_loss = masked_mean(final_raw, y_mask, weights)

    normal_error = nn.functional.smooth_l1_loss(details["normal_residual"], y, reduction="none").detach()
    incident_error_raw = nn.functional.smooth_l1_loss(details["incident_residual"], y, reduction="none")
    incident_loss = masked_mean(incident_error_raw, y_mask, weights)

    incident_advantage = normal_error - incident_error_raw.detach()
    hard = incident_advantage.abs() > args.hard_margin
    hard_weights = torch.ones_like(y_mask) + args.hard_weight * (hard & y_mask.bool()).to(y_mask.dtype)
    branch_weights = weights * hard_weights
    branch_loss = masked_mean(incident_error_raw, y_mask, branch_weights)

    target_gate = torch.sigmoid(incident_advantage / max(args.preference_temperature, 1e-6))
    gate = details["gate"].clamp(1e-5, 1.0 - 1e-5)
    gate_loss_raw = nn.functional.binary_cross_entropy(gate, target_gate, reduction="none")
    gate_loss = masked_mean(gate_loss_raw, y_mask, branch_weights)

    normal_better = (incident_error_raw.detach() - normal_error) > args.normal_better_margin
    normal_better = normal_better & y_mask.bool() & (gate.detach() > args.normal_better_min_gate)
    if normal_better.any():
        normal_better_gate_raw = nn.functional.binary_cross_entropy(gate, torch.zeros_like(gate), reduction="none")
        normal_better_gate_loss = masked_mean(normal_better_gate_raw, normal_better.to(y_mask.dtype), weights)
    else:
        normal_better_gate_loss = final_loss * 0.0

    normal_detached = details["normal_residual"].detach()
    incident_detached = details["incident_residual"].detach()
    proposal_gap = incident_detached - normal_detached
    convex_target_mask = (proposal_gap.abs() > args.convex_gate_min_gap) & y_mask.bool()
    safe_gap = torch.where(convex_target_mask, proposal_gap, torch.ones_like(proposal_gap))
    convex_target = ((y - normal_detached) / safe_gap).clamp(0.0, 1.0)
    convex_gate_raw = nn.functional.smooth_l1_loss(gate, convex_target, reduction="none")
    convex_gate_loss = masked_mean(convex_gate_raw, convex_target_mask.to(y_mask.dtype), branch_weights)

    loss = (
        final_loss
        + args.incident_loss_weight * branch_loss
        + args.gate_loss_weight * gate_loss
        + args.normal_better_gate_loss_weight * normal_better_gate_loss
        + args.convex_gate_loss_weight * convex_gate_loss
    )
    with torch.no_grad():
        hard_rate = (hard & y_mask.bool()).to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        affected_rate = affected_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        normal_better_rate = normal_better.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        convex_target_rate = convex_target_mask.to(torch.float32).sum() / y_mask.sum().clamp_min(1.0)
        affected_tail_weight_mean = masked_mean(tail_weight_map, affected_mask.to(y_mask.dtype))
    return loss, {
        "final_loss": float(final_loss.detach().cpu()),
        "incident_loss": float(incident_loss.detach().cpu()),
        "branch_loss": float(branch_loss.detach().cpu()),
        "gate_loss": float(gate_loss.detach().cpu()),
        "normal_better_gate_loss": float(normal_better_gate_loss.detach().cpu()),
        "convex_gate_loss": float(convex_gate_loss.detach().cpu()),
        "hard_rate": float(hard_rate.detach().cpu()),
        "affected_rate": float(affected_rate.detach().cpu()),
        "normal_better_rate": float(normal_better_rate.detach().cpu()),
        "convex_target_rate": float(convex_target_rate.detach().cpu()),
        "severity_high_rate": float(severity_high.to(torch.float32).mean().detach().cpu()),
        "recovery_long_rate": float(recovery_long.to(torch.float32).mean().detach().cpu()),
        "high_long_rate": float(high_long.to(torch.float32).mean().detach().cpu()),
        "affected_tail_weight_mean": float(affected_tail_weight_mean.detach().cpu()),
        "gate_mean": float(masked_mean(details["gate"], y_mask).detach().cpu()),
    }


def evaluate_loss(model: nn.Module, loader: torch.utils.data.DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {
        key: 0.0
        for key in [
            "loss",
            "final_loss",
            "incident_loss",
            "branch_loss",
            "gate_loss",
            "normal_better_gate_loss",
            "convex_gate_loss",
            "hard_rate",
            "affected_rate",
            "normal_better_rate",
            "convex_target_rate",
            "severity_high_rate",
            "recovery_long_rate",
            "high_long_rate",
            "affected_tail_weight_mean",
            "gate_mean",
        ]
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def select_beta(val_sweep: pd.DataFrame, args: argparse.Namespace) -> float:
    if args.selection_metric not in val_sweep.columns:
        raise KeyError(f"selection metric not found in sweep: {args.selection_metric}")
    all_metric = "all_candidates_model_robust_mae"
    best_all = float(val_sweep[all_metric].min())
    eligible = val_sweep[val_sweep[all_metric] <= best_all + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_sweep
    row = eligible.loc[eligible[args.selection_metric].idxmin()]
    return float(row["residual_beta"])


def parse_split_list(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"train", "val", "test"}
    bad = [item for item in splits if item not in allowed]
    if bad:
        raise ValueError(f"unsupported eval split(s): {bad}")
    if "test" not in splits:
        splits.append("test")
    return splits


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("ST-TIS incident branch fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    eval_counts: dict[str, int],
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    cols = ["all_candidates_model_robust_mae", "affected_candidates_model_robust_mae", "unaffected_candidates_model_robust_mae"]
    table = pd.DataFrame([{"split": split, **{col: values[col] for col in cols}} for split, values in metrics.items()])
    lines = [
        "# ST-TIS Incident Branch Fine-Tune",
        "",
        "This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- incident-ft all candidates robust MAE: `{metrics['test']['all_candidates_model_robust_mae']:.4f}`",
        f"- incident-ft affected candidates robust MAE: `{metrics['test']['affected_candidates_model_robust_mae']:.4f}`",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- train_incident: {args.train_incident}",
        f"- train_gate: {args.train_gate}",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- affected_weight: {args.affected_weight}",
        f"- incident_loss_weight: {args.incident_loss_weight}",
        f"- gate_loss_weight: {args.gate_loss_weight}",
        f"- normal_better_gate_loss_weight: {args.normal_better_gate_loss_weight}",
        f"- normal_better_margin: {args.normal_better_margin}",
        f"- normal_better_min_gate: {args.normal_better_min_gate}",
        f"- convex_gate_loss_weight: {args.convex_gate_loss_weight}",
        f"- convex_gate_min_gap: {args.convex_gate_min_gap}",
        f"- severity_high_weight: {args.severity_high_weight}",
        f"- recovery_long_weight: {args.recovery_long_weight}",
        f"- high_long_weight: {args.high_long_weight}",
        f"- tail_weight_max: {args.tail_weight_max}",
        f"- severity_high_z_threshold: {getattr(args, 'severity_high_z_threshold', float('nan'))}",
        f"- recovery_long_z_threshold: {getattr(args, 'recovery_long_z_threshold', float('nan'))}",
        f"- trainable parameters: {trainable_params}",
        "",
        "## Split Counts",
        "",
        pd.DataFrame([{"split": key, "samples": split_counts[key], "eval_samples": eval_counts[key]} for key in split_counts]).to_markdown(index=False),
        "",
        "## Metrics",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
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

    model = make_model(ckpt, cache_path, device)
    trainable_params = freeze_for_incident_finetune(model, args.train_incident, args.train_gate)
    print(f"trainable parameters: {trainable_params}", flush=True)

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
    tail_thresholds = compute_tail_thresholds(cache_path, train_indices, stats)
    for key, value in tail_thresholds.items():
        setattr(args, key, value)
        setattr(loss_args, key, value)
    print(
        "tail thresholds: "
        f"severity_high_z={tail_thresholds['severity_high_z_threshold']:.4f}, "
        f"recovery_long_z={tail_thresholds['recovery_long_z_threshold']:.4f}",
        flush=True,
    )

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
            loss, _parts = compute_loss(model, batch, loss_args, device)
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

    betas = parse_float_list(args.sweep_betas)
    val_sweep = pd.DataFrame(
        [{"residual_beta": beta, **values} for beta, values in forecast_metrics_for_loader(model, val_loader, betas, device).items()]
    )
    val_sweep.to_csv(output_dir / "beta_sweep.csv", index=False)
    residual_beta = select_beta(val_sweep, args)

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        idx = eval_indices[split]
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]

    ckpt_args = dict(source_args)
    ckpt_args.update(json_safe_args(args))
    ckpt_args["training_variant"] = "incident_branch_affected_finetune"
    ckpt_args["residual_beta"] = residual_beta
    torch.save({"model_state_dict": model.state_dict(), "args": ckpt_args, "residual_beta": residual_beta}, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "samples": split_counts,
                "eval_samples": eval_counts,
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
                "source_model_dir": str(model_dir),
                "selection_metric": args.selection_metric,
                "all_val_tolerance": args.all_val_tolerance,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    config["residual_beta"] = residual_beta
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, eval_counts, residual_beta, trainable_params, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
