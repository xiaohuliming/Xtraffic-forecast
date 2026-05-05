#!/usr/bin/env python3
"""Fine-tune a local selector over normal, incident, and fused ST-TIS proposals."""

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
from train_dual_branch_sttis_gate import DualBranchSTTISLocalSelectorGate
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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_local_selector"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--affected-weight", type=float, default=4.0)
    parser.add_argument("--selection-loss-weight", type=float, default=0.20)
    parser.add_argument("--regret-loss-weight", type=float, default=0.10)
    parser.add_argument("--oracle-gap-weight", type=float, default=2.0)
    parser.add_argument("--selection-margin", type=float, default=0.10)
    parser.add_argument("--normal-label-weight", type=float, default=1.5)
    parser.add_argument("--incident-label-weight", type=float, default=1.0)
    parser.add_argument("--base-label-weight", type=float, default=1.0)
    parser.add_argument(
        "--hard-replay-samples",
        type=int,
        default=0,
        help="Add top training samples where the normal branch clearly beats incident/base fused proposals.",
    )
    parser.add_argument("--hard-replay-repeat", type=int, default=1)
    parser.add_argument(
        "--hard-replay-score-samples",
        type=int,
        default=0,
        help="Cap the training samples scanned for hard replay; 0 scans the full train split.",
    )
    parser.add_argument("--train-selector-temperature", type=float, default=1.0)
    parser.add_argument("--selector-init-base-bias", type=float, default=2.0)
    parser.add_argument("--proposal-feature-count", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--sweep-temperatures", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--sweep-betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_candidates_model_robust_mae")
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
) -> DualBranchSTTISLocalSelectorGate:
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        raise TypeError("checkpoint args must be a dict")
    shapes = infer_cache_shapes(cache_path)
    model = DualBranchSTTISLocalSelectorGate(
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
        selector_temperature=args.train_selector_temperature,
        selector_init_base_bias=args.selector_init_base_bias,
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


def freeze_except_selector(model: nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("selector_head.") or name.startswith("base_gate_logit_norm.")
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no selector parameters were marked trainable")
    return trainable


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    weighted_mask = mask * weights
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def label_weights(target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weights = torch.ones_like(target, dtype=torch.float32)
    weights = torch.where(target == 0, weights * args.normal_label_weight, weights)
    weights = torch.where(target == 1, weights * args.incident_label_weight, weights)
    weights = torch.where(target == 2, weights * args.base_label_weight, weights)
    return weights


def compute_selector_loss(
    model: DualBranchSTTISLocalSelectorGate,
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
    base_weights = 1.0 + args.affected_weight * affected_mask.to(y_mask.dtype)

    residual_raw = nn.functional.smooth_l1_loss(pred_y, y, reduction="none")
    residual_loss = masked_mean(residual_raw, y_mask, base_weights)

    normal_abs = torch.abs(details["normal_residual"].detach() - y)
    incident_abs = torch.abs(details["incident_residual"].detach() - y)
    fused_abs = torch.abs(details["base_fused_residual"].detach() - y)
    option_errors = torch.stack([normal_abs, incident_abs, fused_abs], dim=-1)
    sorted_errors, _order = torch.sort(option_errors, dim=-1)
    best_error = sorted_errors[..., 0]
    second_error = sorted_errors[..., 1]
    oracle_gap = second_error - best_error
    target = torch.argmin(option_errors, dim=-1)

    gap_weight = (oracle_gap / max(args.selection_margin, 1e-6)).clamp(0.0, 1.0)
    selector_weights = base_weights * (1.0 + args.oracle_gap_weight * gap_weight)
    selector_weights = selector_weights * label_weights(target, args).to(device=device, dtype=selector_weights.dtype)

    selector_logits = details["selector_logits"]
    ce_raw = nn.functional.cross_entropy(
        selector_logits.reshape(-1, 3),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(y)
    selection_loss = masked_mean(ce_raw, y_mask, selector_weights)

    expected_error = (details["selector_weights"] * option_errors).sum(dim=-1)
    regret_raw = (expected_error - best_error).clamp_min(0.0)
    regret_loss = masked_mean(regret_raw, y_mask, selector_weights)

    loss = residual_loss + args.selection_loss_weight * selection_loss + args.regret_loss_weight * regret_loss
    with torch.no_grad():
        valid = y_mask.bool()
        normal_rate = ((target == 0) & valid).to(torch.float32).sum() / valid.to(torch.float32).sum().clamp_min(1.0)
        incident_rate = ((target == 1) & valid).to(torch.float32).sum() / valid.to(torch.float32).sum().clamp_min(1.0)
        fused_rate = ((target == 2) & valid).to(torch.float32).sum() / valid.to(torch.float32).sum().clamp_min(1.0)
        hard_rate = ((oracle_gap > args.selection_margin) & valid).to(torch.float32).sum() / valid.to(torch.float32).sum().clamp_min(1.0)
        selector_mean = details["selector_weights"][valid].mean(dim=0)
    return loss, {
        "residual_loss": float(residual_loss.detach().cpu()),
        "selection_loss": float(selection_loss.detach().cpu()),
        "regret_loss": float(regret_loss.detach().cpu()),
        "oracle_gap_mean": float(masked_mean(oracle_gap, y_mask).detach().cpu()),
        "target_normal_rate": float(normal_rate.detach().cpu()),
        "target_incident_rate": float(incident_rate.detach().cpu()),
        "target_fused_rate": float(fused_rate.detach().cpu()),
        "hard_rate": float(hard_rate.detach().cpu()),
        "selector_normal_mean": float(selector_mean[0].detach().cpu()),
        "selector_incident_mean": float(selector_mean[1].detach().cpu()),
        "selector_fused_mean": float(selector_mean[2].detach().cpu()),
        "effective_gate_mean": float(masked_mean(details["gate"], y_mask).detach().cpu()),
        "base_gate_mean": float(masked_mean(details["base_gate"], y_mask).detach().cpu()),
    }


def evaluate_loss(
    model: DualBranchSTTISLocalSelectorGate,
    loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    keys = [
        "loss",
        "residual_loss",
        "selection_loss",
        "regret_loss",
        "oracle_gap_mean",
        "target_normal_rate",
        "target_incident_rate",
        "target_fused_rate",
        "hard_rate",
        "selector_normal_mean",
        "selector_incident_mean",
        "selector_fused_mean",
        "effective_gate_mean",
        "base_gate_mean",
    ]
    totals = {key: 0.0 for key in keys}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
            loss, parts = compute_selector_loss(model, batch, args, device)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                totals[key] += float(value) * batch_size
            count += batch_size
    return {key: totals[key] / max(count, 1) for key in keys}


def score_hard_replay_indices(
    model: DualBranchSTTISLocalSelectorGate,
    cache_path: Path,
    train_indices_full: np.ndarray,
    stats: object,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if args.hard_replay_samples <= 0:
        return np.asarray([], dtype=np.int64)
    score_indices = cap_indices(train_indices_full, args.hard_replay_score_samples, args.seed + 101)
    if score_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    loader = make_loader(cache_path, score_indices, stats, args.batch_size, shuffle=False)
    model.eval()
    rows: list[tuple[float, int]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch[0].shape[0])
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
            _pred_y, _pred_impact, _pred_event_aux, _pred_node_logits, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            affected_mask = node_affected[:, None, :, None].bool() & y_mask.bool()
            normal_abs = torch.abs(details["normal_residual"].detach() - y)
            incident_abs = torch.abs(details["incident_residual"].detach() - y)
            fused_abs = torch.abs(details["base_fused_residual"].detach() - y)
            next_best_abs = torch.minimum(incident_abs, fused_abs)
            advantage = (next_best_abs - normal_abs).clamp_min(0.0)
            per_sample_score = (advantage * affected_mask).flatten(start_dim=1).sum(dim=1)
            per_sample_count = affected_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1)
            per_sample_score = per_sample_score / per_sample_count
            batch_indices = score_indices[offset : offset + batch_size]
            rows.extend((float(score.detach().cpu()), int(idx)) for score, idx in zip(per_sample_score, batch_indices))
            offset += batch_size
    rows.sort(reverse=True)
    selected = [idx for score, idx in rows[: args.hard_replay_samples] if score > 0.0]
    return np.asarray(selected, dtype=np.int64)


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    if log_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


def sweep_temperature_beta(
    model: DualBranchSTTISLocalSelectorGate,
    loader: torch.utils.data.DataLoader,
    temperatures: list[float],
    betas: list[float],
    device: torch.device,
) -> pd.DataFrame:
    original_temperature = float(model.selector_temperature)
    rows = []
    try:
        for temperature in temperatures:
            model.selector_temperature = float(temperature)
            metrics_by_beta = forecast_metrics_for_loader(model, loader, betas, device)
            rows.extend({"selector_temperature": temperature, "residual_beta": beta, **values} for beta, values in metrics_by_beta.items())
    finally:
        model.selector_temperature = original_temperature
    return pd.DataFrame(rows)


def select_temperature_beta(val_sweep: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float]:
    if args.selection_metric not in val_sweep.columns:
        raise KeyError(f"selection metric not found in sweep: {args.selection_metric}")
    all_metric = "all_candidates_model_robust_mae"
    best_all = float(val_sweep[all_metric].min())
    eligible = val_sweep[val_sweep[all_metric] <= best_all + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_sweep
    row = eligible.loc[eligible[args.selection_metric].idxmin()]
    return float(row["selector_temperature"]), float(row["residual_beta"])


def metrics_at(
    model: DualBranchSTTISLocalSelectorGate,
    loader: torch.utils.data.DataLoader,
    selector_temperature: float,
    residual_beta: float,
    device: torch.device,
) -> dict[str, float]:
    old_temperature = float(model.selector_temperature)
    model.selector_temperature = float(selector_temperature)
    try:
        return forecast_metrics_for_loader(model, loader, [residual_beta], device)[residual_beta]
    finally:
        model.selector_temperature = old_temperature


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    source_metrics: dict[str, float],
    metrics: dict[str, dict[str, float]],
    split_counts: dict[str, int],
    eval_counts: dict[str, int],
    selector_temperature: float,
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
    hard_replay_count: int,
) -> None:
    test = metrics["test"]
    lines = [
        "# ST-TIS Local Selector Gate",
        "",
        "This variant freezes the source model and trains a local 3-way selector over normal, incident, and base fused proposals.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- selector all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.4f}`",
        f"- selector affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.4f}`",
        f"- selector_temperature: {selector_temperature:.2f}",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- affected_weight: {args.affected_weight}",
        f"- selection_loss_weight: {args.selection_loss_weight}",
        f"- regret_loss_weight: {args.regret_loss_weight}",
        f"- oracle_gap_weight: {args.oracle_gap_weight}",
        f"- selection_margin: {args.selection_margin}",
        f"- normal_label_weight: {args.normal_label_weight}",
        f"- hard_replay_samples: {hard_replay_count}",
        f"- hard_replay_repeat: {args.hard_replay_repeat}",
        f"- selector_init_base_bias: {args.selector_init_base_bias}",
        f"- trainable selector parameters: {trainable_params}",
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
    trainable_params = freeze_except_selector(model)
    print(f"trainable selector parameters: {trainable_params}", flush=True)

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
    hard_replay_indices = score_hard_replay_indices(model, cache_path, train_indices_full, stats, loss_args, device)
    if hard_replay_indices.size > 0 and args.hard_replay_repeat > 0:
        replay = np.tile(hard_replay_indices, max(args.hard_replay_repeat, 1))
        train_indices = np.concatenate([train_indices, replay])
        print(
            f"hard replay: selected {hard_replay_indices.size} samples, "
            f"repeat={args.hard_replay_repeat}, train rows={train_indices.size}",
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
            loss, _parts = compute_selector_loss(model, batch, loss_args, device)
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

    temperatures = parse_float_list(args.sweep_temperatures)
    betas = parse_float_list(args.sweep_betas)
    val_sweep = sweep_temperature_beta(model, val_loader, temperatures, betas, device)
    val_sweep.to_csv(output_dir / "selector_temperature_beta_sweep.csv", index=False)
    selector_temperature, residual_beta = select_temperature_beta(val_sweep, args)
    model.selector_temperature = selector_temperature

    metrics: dict[str, dict[str, float]] = {}
    for split in parse_split_list(args.eval_splits):
        idx = eval_indices[split]
        loader = make_loader(cache_path, idx, stats, args.batch_size, shuffle=False)
        metrics[split] = metrics_at(model, loader, selector_temperature, residual_beta, device)

    split_counts = {split: int(idx.size) for split, idx in indices.items()}
    eval_counts = {split: int(idx.size) for split, idx in eval_indices.items()}
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    ckpt_args = dict(source_args)
    ckpt_args.update(json_safe_args(args))
    ckpt_args["model_class"] = "DualBranchSTTISLocalSelectorGate"
    ckpt_args["training_variant"] = "local_selector_finetune"
    ckpt_args["selector_temperature"] = selector_temperature
    ckpt_args["selector_init_base_bias"] = args.selector_init_base_bias
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
                "train_rows": int(train_indices.size),
                "hard_replay_samples": int(hard_replay_indices.size),
                "hard_replay_repeat": int(args.hard_replay_repeat),
                "selector_temperature": selector_temperature,
                "selector_init_base_bias": args.selector_init_base_bias,
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
    config["model_class"] = "DualBranchSTTISLocalSelectorGate"
    config["training_variant"] = "local_selector_finetune"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    config["selector_temperature"] = selector_temperature
    config["residual_beta"] = residual_beta
    config["train_rows"] = int(train_indices.size)
    config["hard_replay_samples"] = int(hard_replay_indices.size)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(
        output_dir,
        args,
        source_metrics,
        metrics,
        split_counts,
        eval_counts,
        selector_temperature,
        residual_beta,
        trainable_params,
        log_df,
        int(hard_replay_indices.size),
    )
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
