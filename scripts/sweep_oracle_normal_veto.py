#!/usr/bin/env python3
"""Oracle sweep for normal-veto triggers based on true normal-vs-fused error."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from sweep_sttis_gate_posthoc import resolve_cache_path
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_oracle_normal_veto"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--margins", default="0.0,0.05,0.1,0.2,0.4,0.8,1.2")
    parser.add_argument("--amounts", default="0.25,0.5,1.0")
    parser.add_argument("--betas", default="0.95,1.0,1.05,1.1")
    parser.add_argument("--selection-metric", default="affected_mae")
    parser.add_argument("--all-val-tolerance", type=float, default=0.002)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def cap_indices(indices: torch.Tensor, max_samples: int, seed: int) -> torch.Tensor:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(indices.size, generator=generator)[:max_samples]
    return torch.sort(torch.as_tensor(indices)[perm]).values.numpy()


def empty_sums(betas: list[float]) -> dict[float, dict[str, float]]:
    return {
        beta: {
            "all_model": 0.0,
            "all_base": 0.0,
            "all_normal": 0.0,
            "all_count": 0.0,
            "aff_model": 0.0,
            "aff_base": 0.0,
            "aff_normal": 0.0,
            "aff_count": 0.0,
            "unaff_model": 0.0,
            "unaff_base": 0.0,
            "unaff_normal": 0.0,
            "unaff_count": 0.0,
            "trigger_sum": 0.0,
            "trigger_aff_sum": 0.0,
            "trigger_unaff_sum": 0.0,
        }
        for beta in betas
    }


def update_sums(
    sums: dict[float, dict[str, float]],
    residual: torch.Tensor,
    base_fused: torch.Tensor,
    normal: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
    trigger: torch.Tensor,
    betas: list[float],
) -> None:
    for beta in betas:
        model_abs = (beta * residual - y).abs()
        base_abs = (beta * base_fused - y).abs()
        normal_abs = (beta * normal - y).abs()
        for prefix, label in [("all", "all"), ("aff", "affected"), ("unaff", "unaffected")]:
            mask = masks[label]
            count = mask.sum().item()
            if count <= 0:
                continue
            sums[beta][f"{prefix}_model"] += float(model_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_base"] += float(base_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_normal"] += float(normal_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_count"] += float(count)
        sums[beta]["trigger_sum"] += float(trigger[masks["all"]].sum().detach().cpu())
        sums[beta]["trigger_aff_sum"] += float(trigger[masks["affected"]].sum().detach().cpu())
        sums[beta]["trigger_unaff_sum"] += float(trigger[masks["unaffected"]].sum().detach().cpu())


def summarize(split: str, margin: float, amount: float, sums: dict[float, dict[str, float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for beta, vals in sums.items():
        row: dict[str, float | str] = {"split": split, "margin": margin, "amount": amount, "beta": beta}
        for prefix, label in [("all", "all"), ("aff", "affected"), ("unaff", "unaffected")]:
            count = max(vals[f"{prefix}_count"], 1.0)
            model_mae = vals[f"{prefix}_model"] / count
            base_mae = vals[f"{prefix}_base"] / count
            normal_mae = vals[f"{prefix}_normal"] / count
            row[f"{label}_mae"] = model_mae
            row[f"{label}_base_mae"] = base_mae
            row[f"{label}_normal_mae"] = normal_mae
            row[f"{label}_gain_vs_base"] = base_mae - model_mae
            row[f"{label}_gain_pct_vs_base"] = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else math.nan
        row["trigger_rate"] = vals["trigger_sum"] / max(vals["all_count"], 1.0)
        row["affected_trigger_rate"] = vals["trigger_aff_sum"] / max(vals["aff_count"], 1.0)
        row["unaffected_trigger_rate"] = vals["trigger_unaff_sum"] / max(vals["unaff_count"], 1.0)
        rows.append(row)
    return rows


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    cache_path: Path,
    indices: torch.Tensor,
    margins: list[float],
    amounts: list[float],
    betas: list[float],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    all_sums = {(margin, amount): empty_sums(betas) for margin in margins for amount in amounts}
    model.eval()
    with torch.no_grad():
        for batch in loader:
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
                node_valid,
                _idx,
            ) = [item.to(device) for item in batch]
            if int(getattr(model, "hist_input_channels", hist.shape[-1])) > hist.shape[-1]:
                hist = torch.cat([hist, hist_normal], dim=-1)
            pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal = details["normal_residual"]
            base_fused = details.get("base_fused_residual", pred_y)
            base_abs = (base_fused - y).abs()
            normal_abs = (normal - y).abs()
            normal_advantage = base_abs - normal_abs
            masks = {
                "all": y_mask.bool(),
                "affected": y_mask.bool() & node_affected[:, None, :, None].bool(),
                "unaffected": y_mask.bool()
                & (~node_affected[:, None, :, None].bool())
                & node_valid[:, None, :, None].bool(),
            }
            for margin in margins:
                trigger = (normal_advantage >= margin).to(y.dtype)
                for amount in amounts:
                    residual = (1.0 - amount * trigger) * base_fused + (amount * trigger) * normal
                    update_sums(all_sums[(margin, amount)], residual, base_fused, normal, y, masks, trigger, betas)
    rows = []
    for margin in margins:
        for amount in amounts:
            rows.extend(summarize(split, margin, amount, all_sums[(margin, amount)]))
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, val_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> None:
    best_all = float(val_df["all_mae"].min())
    eligible = val_df[val_df["all_mae"] <= best_all + args.all_val_tolerance]
    if eligible.empty:
        eligible = val_df
    best_val = eligible.loc[eligible[args.selection_metric].idxmin()]
    test_row = test_df[
        (test_df["margin"] == best_val["margin"])
        & (test_df["amount"] == best_val["amount"])
        & (test_df["beta"] == best_val["beta"])
    ].iloc[0]
    lines = [
        "# Oracle Normal-veto Sweep",
        "",
        "This sweep uses true future residual errors to decide when normal is better than the base fused proposal.",
        "It is an upper-bound diagnostic, not a deployable inference method.",
        "",
        "## Validation-selected result",
        "",
        f"- margin: `{best_val['margin']:.4f}`",
        f"- amount: `{best_val['amount']:.2f}`",
        f"- beta: `{best_val['beta']:.2f}`",
        f"- validation all / affected MAE: `{best_val['all_mae']:.4f}` / `{best_val['affected_mae']:.4f}`",
        f"- validation trigger affected / unaffected rate: `{best_val['affected_trigger_rate']:.4f}` / `{best_val['unaffected_trigger_rate']:.4f}`",
        f"- test all / affected / unaffected MAE: `{test_row['all_mae']:.4f}` / `{test_row['affected_mae']:.4f}` / `{test_row['unaffected_mae']:.4f}`",
        f"- test trigger affected / unaffected rate: `{test_row['affected_trigger_rate']:.4f}` / `{test_row['unaffected_trigger_rate']:.4f}`",
        "",
        "## Top validation all rows",
        "",
        val_df.sort_values("all_mae").head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top validation affected rows",
        "",
        val_df.sort_values("affected_mae").head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top test affected rows",
        "",
        test_df.sort_values("affected_mae").head(12).to_markdown(index=False, floatfmt=".4f"),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    device = choose_device(args.device)
    model = make_model(ckpt, cache_path, device)
    splits = split_indices(cache_path)
    margins = parse_float_list(args.margins)
    amounts = parse_float_list(args.amounts)
    betas = parse_float_list(args.betas)
    val_idx = cap_indices(splits["val"], args.max_samples, args.seed)
    test_idx = cap_indices(splits["test"], args.max_samples, args.seed + 1)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)
    val_df = evaluate_split("val", model, cache_path, val_idx, margins, amounts, betas, args.batch_size, device)
    test_df = evaluate_split("test", model, cache_path, test_idx, margins, amounts, betas, args.batch_size, device)
    val_df.to_csv(output_dir / "val_oracle_normal_veto_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_oracle_normal_veto_sweep.csv", index=False)
    write_summary(output_dir, val_df, test_df, args)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "margins": margins,
                "amounts": amounts,
                "betas": betas,
                "max_samples": args.max_samples,
                "selection_metric": args.selection_metric,
                "all_val_tolerance": args.all_val_tolerance,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
