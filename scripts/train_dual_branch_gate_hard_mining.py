#!/usr/bin/env python3
"""Train the dual-branch gate with hard-example residual reweighting.

This keeps the original dual-branch gate architecture unchanged. The only
change is the training objective: in addition to the ordinary masked residual
loss, it gives extra weight to the hardest valid residual elements in each
batch. No affected-node labels are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_dual_branch_gate_baseline import DualBranchGateBaseline, cap_indices, infer_cache_shapes, save_training_plot
from train_full_candidate_stgnn_heatmap_model import (
    CHANNELS,
    compute_stats,
    forecast_metrics_for_loader,
    make_loader,
    region_codes,
    split_indices,
)
from train_impact_residual_model import choose_device, json_safe_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_gate_hard_mining_no_aux"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-mode", choices=["directional", "undirected"], default="undirected")
    parser.add_argument("--graph-sigma", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--heatmap-aux-weight", type=float, default=0.0)
    parser.add_argument("--event-aux-weight", type=float, default=0.0)
    parser.add_argument("--node-aux-weight", type=float, default=0.0)
    parser.add_argument("--hard-weight", type=float, default=0.20)
    parser.add_argument("--hard-fraction", type=float, default=0.20)
    parser.add_argument(
        "--hard-signal",
        choices=["prediction_error", "target_residual", "branch_disagreement"],
        default="prediction_error",
    )
    parser.add_argument("--use-normal-delta", action="store_true")
    parser.add_argument("--use-normal-delta-abs", action="store_true")
    parser.add_argument("--use-dual-hist-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Optional per-split cap for quick smoke evaluation; 0 evaluates each full split.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def top_fraction_loss(loss_raw: torch.Tensor, score: torch.Tensor, mask: torch.Tensor, fraction: float) -> torch.Tensor:
    valid = mask.bool()
    selected_loss = loss_raw[valid].reshape(-1)
    selected_score = score[valid].reshape(-1)
    if selected_loss.numel() == 0:
        return loss_raw.sum() * 0.0
    k = max(1, int(round(selected_loss.numel() * max(min(fraction, 1.0), 0.0))))
    k = min(k, selected_loss.numel())
    top_idx = torch.topk(selected_score, k=k, largest=True, sorted=False).indices
    return selected_loss[top_idx].mean()


def compute_hard_loss(
    model: DualBranchGateBaseline,
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
        impact,
        impact_mask,
        event_aux,
        node_affected,
        node_valid,
    ) = [item.to(device) for item in batch]
    if args.use_dual_hist_residual:
        hist = torch.cat([hist, hist_normal], dim=-1)

    pred_y, pred_impact, pred_event_aux, pred_node_logits, details = model(
        hist,
        node,
        global_context,
        normal_delta,
        return_details=True,
    )
    residual_loss_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_loss_raw, y_mask)

    if args.hard_signal == "prediction_error":
        hard_score = residual_loss_raw.detach()
    elif args.hard_signal == "target_residual":
        hard_score = y.abs().detach()
    elif args.hard_signal == "branch_disagreement":
        hard_score = (details["normal_residual"] - details["incident_residual"]).abs().detach()
    else:
        raise ValueError(f"unsupported hard signal: {args.hard_signal}")
    hard_loss = top_fraction_loss(residual_loss_raw, hard_score, y_mask, args.hard_fraction)

    impact_loss_raw = nn.functional.smooth_l1_loss(pred_impact, impact, reduction="none")
    impact_loss = masked_mean(impact_loss_raw, impact_mask)
    event_aux_loss = nn.functional.smooth_l1_loss(pred_event_aux, event_aux)
    node_bce = nn.functional.binary_cross_entropy_with_logits(pred_node_logits, node_affected, reduction="none")
    node_aux_loss = masked_mean(node_bce, node_valid)

    loss = (
        residual_loss
        + args.hard_weight * hard_loss
        + args.heatmap_aux_weight * impact_loss
        + args.event_aux_weight * event_aux_loss
        + args.node_aux_weight * node_aux_loss
    )
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "hard_loss": float(hard_loss.detach().cpu()),
        "impact_loss": float(impact_loss.detach().cpu()),
        "event_aux_loss": float(event_aux_loss.detach().cpu()),
        "node_aux_loss": float(node_aux_loss.detach().cpu()),
    }


def evaluate_loader(model: DualBranchGateBaseline, loader: torch.utils.data.DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "residual_loss": 0.0, "hard_loss": 0.0, "impact_loss": 0.0, "event_aux_loss": 0.0, "node_aux_loss": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            loss, parts = compute_hard_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    residual_beta: float,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# Dual-Branch Gate Hard Mining",
        "",
        "This variant keeps the original model architecture and adds top-fraction hard residual reweighting during training.",
        "",
        "## Test Result",
        "",
        f"- all candidates robust MAE: `{test['all_candidates_baseline_robust_mae']:.4f} -> {test['all_candidates_model_robust_mae']:.4f}` ({test['all_candidates_improvement_pct']:.2f}%)",
        f"- affected candidates robust MAE: `{test['affected_candidates_baseline_robust_mae']:.4f} -> {test['affected_candidates_model_robust_mae']:.4f}` ({test['affected_candidates_improvement_pct']:.2f}%)",
        "",
        "## Hard Mining Settings",
        "",
        f"- hard_weight: {args.hard_weight}",
        f"- hard_fraction: {args.hard_fraction}",
        f"- hard_signal: {args.hard_signal}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Data Settings",
        "",
        f"- cache_path: `{args.cache_path}`",
        f"- epochs: {args.epochs}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- graph_layers: {args.graph_layers}",
        f"- graph_mode: {args.graph_mode}",
        f"- use_normal_delta: {args.use_normal_delta}",
        f"- use_normal_delta_abs: {args.use_normal_delta_abs}",
        f"- use_dual_hist_residual: {args.use_dual_hist_residual}",
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
    cache_path = args.cache_path.resolve()
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)

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
    shapes = infer_cache_shapes(cache_path)
    model = DualBranchGateBaseline(
        channels=shapes["channels"],
        hist_input_channels=len(CHANNELS) * (2 if args.use_dual_hist_residual else 1),
        node_context_dim=shapes["node_context_dim"],
        global_context_dim=shapes["global_context_dim"],
        horizon_steps=shapes["horizon_steps"],
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        dropout=args.dropout,
        graph_sigma=args.graph_sigma,
        graph_mode=args.graph_mode,
        use_normal_delta=args.use_normal_delta,
        use_normal_delta_abs=args.use_normal_delta_abs,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            optim.zero_grad(set_to_none=True)
            loss, _parts = compute_hard_loss(model, batch, args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(model, val_loader, args, device)
        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"}}
        )
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_metrics['loss']:.4f}", flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "training_log.csv", index=False)
    save_training_plot(log_df, output_dir)

    beta_candidates = [round(x, 2) for x in np.arange(0.0, 1.51, 0.05)]
    val_metrics_by_beta = forecast_metrics_for_loader(model, val_loader, beta_candidates, device)
    beta_df = pd.DataFrame([{"residual_beta": beta, **values} for beta, values in val_metrics_by_beta.items()])
    beta_df.to_csv(output_dir / "residual_beta_sweep.csv", index=False)
    residual_beta = float(beta_df.loc[beta_df["all_candidates_model_robust_mae"].idxmin(), "residual_beta"])

    metrics: dict[str, dict[str, float]] = {}
    for split, idx in eval_indices.items():
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]

    region_code_arr = region_codes(cache_path)
    region_metrics = []
    for code in sorted(np.unique(region_code_arr[eval_indices["test"]]).tolist()):
        mask_idx = indices["test"][region_code_arr[indices["test"]] == code]
        mask_idx = cap_indices(mask_idx, args.max_eval_samples, args.seed + 100 + int(code))
        loader = make_loader(cache_path, mask_idx, stats, args.batch_size, shuffle=False)
        row = {"region_code": int(code), "samples": int(mask_idx.size)}
        row.update(forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta])
        region_metrics.append(row)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    ckpt_args = json_safe_args(args)
    ckpt_args["model_class"] = "DualBranchGateBaseline"
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
                "residual_beta": residual_beta,
                "cache_path": str(cache_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = "DualBranchGateBaseline"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, metrics, split_counts, residual_beta, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
