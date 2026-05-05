#!/usr/bin/env python3
"""Diagnose branch/oracle MAE by severity and recovery groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from compare_dual_branch_group_metrics import residual_beta
from train_dual_branch_gate_baseline import cap_indices
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


BRANCHES = (
    "baseline",
    "normal_branch",
    "incident_branch",
    "fixed_gate_05",
    "learned_gate",
    "oracle_branch_min",
    "oracle_convex",
)
SUBSETS = ("all", "affected", "unaffected")
GROUPS = (
    "overall",
    "severity_low",
    "severity_mid",
    "severity_high",
    "recovery_short_lt30",
    "recovery_mid_30_90",
    "recovery_long_ge90",
    "severity_high_and_long",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "outputs/impact_guided_next_stage/"
            "dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/branch_group_oracle_diagnostics_seed_23"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
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
        payload = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        cache_path = Path(payload["cache_path"])
    return cache_path.resolve()


def model_uses_dual_hist(model: torch.nn.Module, hist_channels: int) -> bool:
    return int(getattr(model, "hist_input_channels", hist_channels)) > hist_channels


def event_semantic(raw_event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    severity = np.expm1(raw_event[:, 0])
    recovery = raw_event[:, 1] * 180.0
    for value in (severity, recovery):
        value[~np.isfinite(value)] = 0.0
        value[value < 0.0] = 0.0
    return severity, recovery


def group_masks(raw_event: np.ndarray) -> dict[str, np.ndarray]:
    severity, recovery = event_semantic(raw_event)
    q33, q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "overall": np.ones(raw_event.shape[0], dtype=bool),
        "severity_low": severity <= q33,
        "severity_mid": (severity > q33) & (severity <= q66),
        "severity_high": severity > q66,
        "recovery_short_lt30": recovery < 30.0,
        "recovery_mid_30_90": (recovery >= 30.0) & (recovery < 90.0),
        "recovery_long_ge90": recovery >= 90.0,
        "severity_high_and_long": (severity > q66) & (recovery >= 90.0),
    }


def empty_sums() -> dict[tuple[str, str, str], dict[str, float]]:
    return {
        (group, subset, branch): {"abs_sum": 0.0, "count": 0.0}
        for group in GROUPS
        for subset in SUBSETS
        for branch in BRANCHES
    }


def empty_value_sums() -> dict[tuple[str, str], dict[str, float]]:
    return {
        (group, subset): {"gate_sum": 0.0, "alpha_sum": 0.0, "count": 0.0}
        for group in GROUPS
        for subset in SUBSETS
    }


def update_sums(
    sums: dict[tuple[str, str, str], dict[str, float]],
    group: str,
    errors: dict[str, torch.Tensor],
    y_mask: torch.Tensor,
    affected: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    masks = {
        "all": y_mask,
        "affected": y_mask & affected[:, None, :, None],
        "unaffected": y_mask & (~affected[:, None, :, None]) & valid[:, None, :, None],
    }
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0.0:
            continue
        for branch, error in errors.items():
            row = sums[(group, subset, branch)]
            row["abs_sum"] += float(error[mask].sum().detach().cpu())
            row["count"] += count


def update_value_sums(
    sums: dict[tuple[str, str], dict[str, float]],
    group: str,
    gate: torch.Tensor,
    alpha: torch.Tensor,
    y_mask: torch.Tensor,
    affected: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    masks = {
        "all": y_mask,
        "affected": y_mask & affected[:, None, :, None],
        "unaffected": y_mask & (~affected[:, None, :, None]) & valid[:, None, :, None],
    }
    for subset, mask in masks.items():
        count = float(mask.sum().item())
        if count <= 0.0:
            continue
        row = sums[(group, subset)]
        row["gate_sum"] += float(gate[mask].sum().detach().cpu())
        row["alpha_sum"] += float(alpha[mask].sum().detach().cpu())
        row["count"] += count


def summarize(sums: dict[tuple[str, str, str], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for subset in SUBSETS:
            for branch in BRANCHES:
                row = sums[(group, subset, branch)]
                count = max(row["count"], 1.0)
                rows.append(
                    {
                        "group": group,
                        "subset": subset,
                        "branch": branch,
                        "mae": row["abs_sum"] / count,
                        "count": row["count"],
                    }
                )
    return pd.DataFrame(rows)


def summarize_values(sums: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for subset in SUBSETS:
            row = sums[(group, subset)]
            count = max(row["count"], 1.0)
            rows.append(
                {
                    "group": group,
                    "subset": subset,
                    "gate_mean": row["gate_sum"] / count,
                    "oracle_alpha_mean": row["alpha_sum"] / count,
                    "alpha_minus_gate": row["alpha_sum"] / count - row["gate_sum"] / count,
                    "count": row["count"],
                }
            )
    return pd.DataFrame(rows)


def pivot_focus(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    table = df[df["subset"].eq(subset)].pivot(index="group", columns="branch", values="mae").reset_index()
    table["learned_minus_normal"] = table["learned_gate"] - table["normal_branch"]
    table["learned_minus_incident"] = table["learned_gate"] - table["incident_branch"]
    table["learned_minus_oracle_min"] = table["learned_gate"] - table["oracle_branch_min"]
    table["learned_minus_oracle_convex"] = table["learned_gate"] - table["oracle_convex"]
    return table


def write_summary(output_dir: Path, df: pd.DataFrame, value_df: pd.DataFrame, args: argparse.Namespace, beta: float) -> None:
    focus_groups = ["overall", "severity_high", "recovery_long_ge90", "severity_high_and_long", "severity_low", "recovery_short_lt30"]
    affected = pivot_focus(df, "affected")
    affected_focus = affected[affected["group"].isin(focus_groups)]
    affected_values = value_df[value_df["subset"].eq("affected")]
    value_focus = affected_values[affected_values["group"].isin(focus_groups)]
    cols = [
        "group",
        "normal_branch",
        "incident_branch",
        "fixed_gate_05",
        "learned_gate",
        "oracle_branch_min",
        "oracle_convex",
        "learned_minus_oracle_min",
        "learned_minus_oracle_convex",
    ]
    lines = [
        "# Branch Group Oracle Diagnostics",
        "",
        f"- model_dir: `{args.model_dir}`",
        f"- split: `{args.split}`",
        f"- residual_beta: `{beta:.4f}`",
        "",
        "## Affected Groups",
        "",
        affected_focus[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Affected Gate Vs Oracle Alpha",
        "",
        value_focus[["group", "gate_mean", "oracle_alpha_mean", "alpha_minus_gate"]].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Overall Affected Table",
        "",
        affected[cols].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    stats = compute_stats(cache_path)
    splits = split_indices(cache_path)
    indices = cap_indices(splits[args.split], args.max_samples, args.seed)
    device = choose_device(args.device)
    model = make_model(ckpt, cache_path, device)
    beta = residual_beta(model_dir, ckpt)
    dual_hist = model_uses_dual_hist(model, hist_channels=3)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    sums = empty_sums()
    value_sums = empty_value_sums()

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)
    print(f"samples: {indices.size}", flush=True)
    with h5py.File(cache_path, "r") as h5, torch.no_grad():
        raw_all = h5["event_aux"][indices].astype(np.float32)
        split_groups = group_masks(raw_all)
        offset = 0
        for batch_idx, batch in enumerate(loader, start=1):
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
            ) = batch
            batch_size = int(hist.shape[0])
            hist = hist.to(device)
            hist_normal = hist_normal.to(device)
            node = node.to(device)
            global_context = global_context.to(device)
            normal_delta = normal_delta.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device).bool()
            node_affected = node_affected.to(device).bool()
            node_valid = node_valid.to(device).bool()
            if dual_hist:
                hist = torch.cat([hist, hist_normal], dim=-1)
            pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal_pred = beta * details["normal_residual"]
            incident_pred = beta * details["incident_residual"]
            fixed_pred = 0.5 * normal_pred + 0.5 * incident_pred
            learned_pred = beta * pred_y
            gap = incident_pred - normal_pred
            usable = gap.abs() > 1e-6
            alpha = torch.where(usable, ((y - normal_pred) / torch.where(usable, gap, torch.ones_like(gap))).clamp(0.0, 1.0), torch.zeros_like(gap))
            oracle_convex_pred = normal_pred + alpha * gap
            normal_error = (normal_pred - y).abs()
            incident_error = (incident_pred - y).abs()
            errors = {
                "baseline": y.abs(),
                "normal_branch": normal_error,
                "incident_branch": incident_error,
                "fixed_gate_05": (fixed_pred - y).abs(),
                "learned_gate": (learned_pred - y).abs(),
                "oracle_branch_min": torch.minimum(normal_error, incident_error),
                "oracle_convex": (oracle_convex_pred - y).abs(),
            }
            for group in GROUPS:
                group_mask_np = split_groups[group][offset : offset + batch_size]
                group_mask = torch.from_numpy(group_mask_np).to(device=device, dtype=torch.bool)
                update_sums(
                    sums,
                    group,
                    errors,
                    y_mask & group_mask[:, None, None, None],
                    node_affected,
                    node_valid,
                )
                update_value_sums(
                    value_sums,
                    group,
                    details["gate"],
                    alpha,
                    y_mask & group_mask[:, None, None, None],
                    node_affected,
                    node_valid,
                )
            offset += batch_size
            if batch_idx % 25 == 0:
                print(f"processed {min(offset, indices.size)}/{indices.size}", flush=True)

    df = summarize(sums)
    value_df = summarize_values(value_sums)
    df.to_csv(output_dir / "branch_group_oracle_metrics.csv", index=False)
    value_df.to_csv(output_dir / "branch_group_oracle_alpha_metrics.csv", index=False)
    write_summary(output_dir, df, value_df, args, beta)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "split": args.split,
                "samples": int(indices.size),
                "residual_beta": beta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
