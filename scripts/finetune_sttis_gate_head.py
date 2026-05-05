#!/usr/bin/env python3
"""Fine-tune only the gate head of a trained ST-TIS dual-branch model."""

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

from analyze_dual_branch_gate import make_model, torch_load
from train_dual_branch_gate_baseline import cap_indices
from train_full_candidate_stgnn_heatmap_model import (
    compute_loss,
    compute_stats,
    evaluate_loader,
    forecast_metrics_for_loader,
    make_loader,
    region_codes,
    split_indices,
)
from train_impact_residual_model import choose_device, json_safe_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux_seed_23"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_23"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--heatmap-aux-weight", type=float, default=0.0)
    parser.add_argument("--event-aux-weight", type=float, default=0.0)
    parser.add_argument("--node-aux-weight", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=0)
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


def freeze_except_gate(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gate_head.")
        if param.requires_grad:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError("no gate_head parameters were marked trainable")
    return trainable


def save_training_plot(log_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("residual loss")
    ax.set_title("ST-TIS gate-head fine-tuning")
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
    residual_beta: float,
    trainable_params: int,
    log_df: pd.DataFrame,
) -> None:
    test = metrics["test"]
    lines = [
        "# ST-TIS Gate-Head Fine-tune",
        "",
        "This variant freezes the trained normal/incident residual branches and fine-tunes only the gate head using the no-aux residual forecasting loss.",
        "",
        "## Test Result",
        "",
        f"- source all candidates robust MAE: `{source_metrics['all_candidates_model_robust_mae']:.4f}`",
        f"- source affected candidates robust MAE: `{source_metrics['affected_candidates_model_robust_mae']:.4f}`",
        f"- fine-tuned all candidates robust MAE: `{test['all_candidates_model_robust_mae']:.4f}`",
        f"- fine-tuned affected candidates robust MAE: `{test['affected_candidates_model_robust_mae']:.4f}`",
        f"- residual_beta: {residual_beta:.2f}",
        "",
        "## Settings",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- epochs: {args.epochs}",
        f"- lr: {args.lr}",
        f"- trainable gate parameters: {trainable_params}",
        f"- max_train_samples: {args.max_train_samples}",
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

    model = make_model(ckpt, cache_path, device)
    trainable_params = freeze_except_gate(model)
    print(f"trainable gate parameters: {trainable_params}", flush=True)

    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        model_args = {}
    loss_args = argparse.Namespace(
        use_dual_hist_residual=bool(model_args.get("use_dual_hist_residual", True)),
        heatmap_aux_weight=args.heatmap_aux_weight,
        event_aux_weight=args.event_aux_weight,
        node_aux_weight=args.node_aux_weight,
    )

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
            loss, _parts = compute_loss(model, batch, loss_args, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()
            running += float(loss.detach().cpu())
            batches += 1
        train_loss = running / max(batches, 1)
        val_metrics = evaluate_loader(model, val_loader, loss_args, device)
        val_loss = float(val_metrics["loss"])
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
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
    source_data = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    source_metrics = source_data["metrics"]["test"]
    ckpt_args = dict(model_args)
    ckpt_args.update(json_safe_args(args))
    ckpt_args["model_class"] = "DualBranchSTTISGate"
    ckpt_args["training_variant"] = "gate_head_finetune"
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
                "source_model_dir": str(model_dir),
                "source_residual_beta": float(ckpt.get("residual_beta", 1.0)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    config = json_safe_args(args)
    config["model_class"] = "DualBranchSTTISGate"
    config["training_variant"] = "gate_head_finetune"
    config["device"] = str(device)
    config["cache_path"] = str(cache_path)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    write_summary(output_dir, args, source_metrics, metrics, split_counts, residual_beta, trainable_params, log_df)
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
