#!/usr/bin/env python3
"""Create case-study visualizations for the dual-branch gated residual model."""

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
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, cap_indices, make_model, torch_load
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_gate_full_no_aux"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/dual_branch_gate_case_studies"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument(
        "--selection",
        choices=["success", "neutral", "failure", "mixed"],
        default="success",
        help="Case selection rule based on learned-vs-fixed affected MAE gain.",
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=2,
        help="Number of success/neutral/failure cases used when --selection=mixed.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def resolve_cache_path(model_dir: Path, ckpt: dict[str, object]) -> Path:
    cache_path = Path(str(ckpt.get("cache_path", "")))
    if not cache_path.is_file():
        model_args = ckpt.get("args", {})
        if isinstance(model_args, dict):
            cache_path = Path(str(model_args.get("cache_path", "")))
    if not cache_path.is_file():
        metrics_path = model_dir / "metrics.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        cache_path = Path(data["cache_path"])
    return cache_path.resolve()


def masked_mae_by_sample(error: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = mask.sum(axis=(1, 2, 3)).astype(np.float64)
    summed = np.where(mask, error, 0.0).sum(axis=(1, 2, 3)).astype(np.float64)
    mae = np.divide(summed, np.maximum(count, 1.0), out=np.full_like(summed, np.nan), where=count > 0)
    return mae, count


def run_inference_table(
    model: torch.nn.Module,
    cache_path: Path,
    selected_indices: np.ndarray,
    residual_beta: float,
    model_args: dict[str, object],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=selected_indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    rows: list[dict[str, float | int]] = []

    with h5py.File(cache_path, "r") as h5, torch.no_grad():
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
                idx,
            ) = batch
            hist = hist.to(device)
            hist_normal = hist_normal.to(device)
            node = node.to(device)
            global_context = global_context.to(device)
            normal_delta = normal_delta.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            node_affected = node_affected.to(device)
            node_valid = node_valid.to(device)
            idx_np = idx.detach().cpu().numpy().astype(np.int64)

            if bool(model_args.get("use_dual_hist_residual", False)):
                hist_in = torch.cat([hist, hist_normal], dim=-1)
            else:
                hist_in = hist
            pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist_in,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal_residual = details["normal_residual"]
            incident_residual = details["incident_residual"]
            gate = details["gate"]
            fixed_residual = 0.5 * normal_residual + 0.5 * incident_residual

            y_np = y.detach().cpu().numpy()
            y_mask_np = y_mask.detach().cpu().numpy().astype(bool)
            affected_np = node_affected.detach().cpu().numpy().astype(bool)
            valid_np = node_valid.detach().cpu().numpy().astype(bool)
            gate_np = gate.detach().cpu().numpy()
            errors = {
                "baseline": np.abs(y_np),
                "learned": np.abs(residual_beta * pred_y.detach().cpu().numpy() - y_np),
                "fixed": np.abs(residual_beta * fixed_residual.detach().cpu().numpy() - y_np),
                "normal": np.abs(residual_beta * normal_residual.detach().cpu().numpy() - y_np),
                "incident": np.abs(residual_beta * incident_residual.detach().cpu().numpy() - y_np),
            }
            all_mask = y_mask_np
            affected_mask = y_mask_np & affected_np[:, None, :, None]
            unaffected_mask = y_mask_np & (~affected_np[:, None, :, None]) & valid_np[:, None, :, None]
            masks = {"all": all_mask, "affected": affected_mask, "unaffected": unaffected_mask}

            mae_by_branch: dict[tuple[str, str], np.ndarray] = {}
            counts_by_subset: dict[str, np.ndarray] = {}
            for subset, mask in masks.items():
                counts_by_subset[subset] = mask.sum(axis=(1, 2, 3)).astype(np.float64)
                for branch, error in errors.items():
                    mae_by_branch[(branch, subset)], _ = masked_mae_by_sample(error, mask)

            gate_mean_affected = np.full(idx_np.shape, np.nan, dtype=np.float64)
            gate_mean_unaffected = np.full(idx_np.shape, np.nan, dtype=np.float64)
            for i in range(idx_np.size):
                aff_mask = affected_mask[i]
                unaff_mask = unaffected_mask[i]
                if aff_mask.any():
                    gate_mean_affected[i] = float(gate_np[i][aff_mask].mean())
                if unaff_mask.any():
                    gate_mean_unaffected[i] = float(gate_np[i][unaff_mask].mean())

            region_code = h5["region_code"][idx_np]
            raw_event = h5["event_aux"][idx_np]
            affected_nodes = h5["node_affected"][idx_np].sum(axis=1)
            valid_nodes = h5["node_valid"][idx_np].sum(axis=1)
            for i, sample_idx in enumerate(idx_np):
                learned_aff = mae_by_branch[("learned", "affected")][i]
                fixed_aff = mae_by_branch[("fixed", "affected")][i]
                baseline_aff = mae_by_branch[("baseline", "affected")][i]
                rows.append(
                    {
                        "sample_idx": int(sample_idx),
                        "region_code": int(region_code[i]),
                        "affected_nodes": float(affected_nodes[i]),
                        "valid_nodes": float(valid_nodes[i]),
                        "affected_elements": float(counts_by_subset["affected"][i]),
                        "severity_log_auc": float(raw_event[i, 0]),
                        "recovery_scaled": float(raw_event[i, 1]),
                        "recovery_min": float(raw_event[i, 1] * 180.0),
                        "spread_log_nodes": float(raw_event[i, 2]),
                        "gate_mean_affected": float(gate_mean_affected[i]),
                        "gate_mean_unaffected": float(gate_mean_unaffected[i]),
                        "learned_vs_fixed_affected_gain": float(fixed_aff - learned_aff),
                        "learned_vs_baseline_affected_gain": float(baseline_aff - learned_aff),
                        "learned_vs_fixed_affected_gain_pct": float(100.0 * (fixed_aff - learned_aff) / fixed_aff) if fixed_aff > 0 else np.nan,
                        "learned_vs_baseline_affected_gain_pct": float(100.0 * (baseline_aff - learned_aff) / baseline_aff) if baseline_aff > 0 else np.nan,
                        **{
                            f"{branch}_{subset}_mae": float(mae_by_branch[(branch, subset)][i])
                            for branch in errors
                            for subset in masks
                        },
                    }
                )

            if batch_idx % 20 == 0:
                print(f"scored {min(batch_idx * batch_size, selected_indices.size)}/{selected_indices.size}", flush=True)

    return pd.DataFrame(rows)


def pick_single_category(df: pd.DataFrame, category: str, num_cases: int) -> pd.DataFrame:
    candidates = df[(df["affected_elements"] > 0) & np.isfinite(df["learned_vs_fixed_affected_gain"])].copy()
    if category == "success":
        candidates = candidates.sort_values(
            ["learned_vs_fixed_affected_gain", "learned_vs_baseline_affected_gain", "affected_elements"],
            ascending=[False, False, False],
        )
    elif category == "failure":
        candidates = candidates.sort_values(
            ["learned_vs_fixed_affected_gain", "learned_vs_baseline_affected_gain", "affected_elements"],
            ascending=[True, True, False],
        )
    elif category == "neutral":
        candidates = candidates.assign(abs_learned_vs_fixed_gain=np.abs(candidates["learned_vs_fixed_affected_gain"]))
        candidates = candidates.sort_values(
            ["abs_learned_vs_fixed_gain", "affected_elements", "learned_affected_mae"],
            ascending=[True, False, False],
        )
        candidates = candidates.drop(columns=["abs_learned_vs_fixed_gain"])
    else:
        raise ValueError(f"unsupported category: {category}")

    out = candidates.head(num_cases).copy()
    out["category"] = category
    return out.reset_index(drop=True)


def pick_cases(df: pd.DataFrame, num_cases: int, selection: str, cases_per_category: int) -> pd.DataFrame:
    if selection != "mixed":
        return pick_single_category(df, selection, num_cases)

    parts = [
        pick_single_category(df, "success", cases_per_category),
        pick_single_category(df, "neutral", cases_per_category),
        pick_single_category(df, "failure", cases_per_category),
    ]
    selected = []
    seen: set[int] = set()
    for part in parts:
        for _, row in part.iterrows():
            sample_idx = int(row["sample_idx"])
            if sample_idx in seen:
                continue
            selected.append(row)
            seen.add(sample_idx)
    if not selected:
        return pd.DataFrame(columns=list(df.columns) + ["category"])
    return pd.DataFrame(selected).reset_index(drop=True)


def sort_nodes_by_position(position: np.ndarray, valid: np.ndarray) -> np.ndarray:
    order = np.argsort(np.where(valid.astype(bool), position, np.inf))
    return order


def masked_channel_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool)
    summed = np.where(valid, values, 0.0).sum(axis=-1)
    count = valid.sum(axis=-1)
    out = np.divide(summed, np.maximum(count, 1), out=np.full_like(summed, np.nan, dtype=np.float64), where=count > 0)
    return out.astype(np.float32)


def visualize_case(
    model: torch.nn.Module,
    cache_path: Path,
    sample_idx: int,
    residual_beta: float,
    model_args: dict[str, object],
    stats: object,
    device: torch.device,
    output_dir: Path,
    rank: int,
    metrics_row: pd.Series,
) -> Path:
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=np.asarray([sample_idx], dtype=np.int64), stats=stats)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
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
    hist = hist.to(device)
    hist_normal = hist_normal.to(device)
    node = node.to(device)
    global_context = global_context.to(device)
    normal_delta = normal_delta.to(device)
    y = y.to(device)
    y_mask = y_mask.to(device)

    if bool(model_args.get("use_dual_hist_residual", False)):
        hist_in = torch.cat([hist, hist_normal], dim=-1)
    else:
        hist_in = hist
    with torch.no_grad():
        pred_y, _pred_impact, _pred_event, _pred_node, details = model(
            hist_in,
            node,
            global_context,
            normal_delta,
            return_details=True,
        )
    normal_residual = details["normal_residual"]
    incident_residual = details["incident_residual"]
    gate = details["gate"]
    fixed_residual = 0.5 * normal_residual + 0.5 * incident_residual

    y_np = y.detach().cpu().numpy()[0]
    mask_np = y_mask.detach().cpu().numpy()[0].astype(bool)
    gate_np = gate.detach().cpu().numpy()[0]
    learned_error = np.abs((residual_beta * pred_y).detach().cpu().numpy()[0] - y_np)
    fixed_error = np.abs((residual_beta * fixed_residual).detach().cpu().numpy()[0] - y_np)
    normal_error = np.abs((residual_beta * normal_residual).detach().cpu().numpy()[0] - y_np)
    incident_error = np.abs((residual_beta * incident_residual).detach().cpu().numpy()[0] - y_np)
    gain_vs_fixed = fixed_error - learned_error
    branch_diff = normal_error - incident_error

    with h5py.File(cache_path, "r") as h5:
        affected = h5["node_affected"][sample_idx].astype(bool)
        valid = h5["node_valid"][sample_idx].astype(bool)
        position = h5["node_signed_pm_raw"][sample_idx].astype(np.float32)
        node_idx = h5["node_idx"][sample_idx].astype(np.int32)
        region_code = int(h5["region_code"][sample_idx])
        raw_event = h5["event_aux"][sample_idx].astype(np.float32)

    order = sort_nodes_by_position(position, valid)
    valid_order = valid[order]
    affected_order = affected[order]
    labels = [str(int(n)) if v else "" for n, v in zip(node_idx[order], valid_order)]

    gate_map = masked_channel_mean(gate_np, mask_np)[:, order]
    residual_map = masked_channel_mean(np.abs(y_np), mask_np)[:, order]
    gain_map = masked_channel_mean(gain_vs_fixed, mask_np)[:, order]
    branch_diff_map = masked_channel_mean(branch_diff, mask_np)[:, order]
    for arr in [gate_map, residual_map, gain_map, branch_diff_map]:
        arr[:, ~valid_order] = np.nan

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13.0, 10.2),
        gridspec_kw={"height_ratios": [0.28, 1.0, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    affected_strip = np.where(affected_order[None, :], 1.0, 0.0)
    affected_strip[:, ~valid_order] = np.nan
    axes[0].imshow(affected_strip, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    axes[0].set_yticks([])
    axes[0].set_title("Affected candidate nodes")

    im1 = axes[1].imshow(gate_map, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_ylabel("Horizon")
    axes[1].set_title("Learned incident-branch gate")
    fig.colorbar(im1, ax=axes[1], fraction=0.018, pad=0.01)

    vmax_res = np.nanpercentile(residual_map, 95) if np.isfinite(residual_map).any() else 1.0
    im2 = axes[2].imshow(residual_map, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=max(vmax_res, 1e-6))
    axes[2].set_ylabel("Horizon")
    axes[2].set_title("Absolute target residual")
    fig.colorbar(im2, ax=axes[2], fraction=0.018, pad=0.01)

    vmax_gain = np.nanpercentile(np.abs(gain_map), 95) if np.isfinite(gain_map).any() else 1.0
    im3 = axes[3].imshow(gain_map, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-max(vmax_gain, 1e-6), vmax=max(vmax_gain, 1e-6))
    axes[3].set_ylabel("Horizon")
    axes[3].set_title("Fixed-gate error minus learned-gate error")
    fig.colorbar(im3, ax=axes[3], fraction=0.018, pad=0.01)

    vmax_branch = np.nanpercentile(np.abs(branch_diff_map), 95) if np.isfinite(branch_diff_map).any() else 1.0
    im4 = axes[4].imshow(branch_diff_map, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-max(vmax_branch, 1e-6), vmax=max(vmax_branch, 1e-6))
    axes[4].set_ylabel("Horizon")
    axes[4].set_title("Normal-branch error minus incident-branch error")
    fig.colorbar(im4, ax=axes[4], fraction=0.018, pad=0.01)

    for ax in axes[1:]:
        ax.set_yticks(np.arange(0, y_np.shape[0], 2))
        ax.set_yticklabels([str(i + 1) for i in range(0, y_np.shape[0], 2)])
    for ax in axes:
        ax.set_xticks(np.arange(0, len(labels), 4))
        ax.set_xticklabels([labels[i] for i in range(0, len(labels), 4)], rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Candidate sensors sorted by signed postmile")

    category = str(metrics_row.get("category", "success"))
    title = (
        f"Case {rank} ({category}): sample {sample_idx}, region {region_code}, "
        f"affected MAE learned={metrics_row['learned_affected_mae']:.3f}, fixed={metrics_row['fixed_affected_mae']:.3f}"
    )
    subtitle = (
        f"severity_log_auc={raw_event[0]:.3f}, recovery={raw_event[1] * 180.0:.1f} min, "
        f"spread_log_nodes={raw_event[2]:.3f}, affected_nodes={int(affected.sum())}"
    )
    fig.suptitle(title + "\n" + subtitle, fontsize=12)
    path = output_dir / f"case_{rank:02d}_{category}_sample_{sample_idx}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(cases: pd.DataFrame, paths: list[Path], output_dir: Path, args: argparse.Namespace) -> None:
    table_columns = [
        "rank",
        "category",
        "sample_idx",
        "region_code",
        "affected_nodes",
        "recovery_min",
        "learned_affected_mae",
        "fixed_affected_mae",
        "learned_vs_fixed_affected_gain",
        "gate_mean_affected",
    ]
    case_table = cases[[col for col in table_columns if col in cases.columns]].copy()
    selection_notes = {
        "success": f"top `{len(cases)}` samples by affected MAE improvement over fixed gate=0.5",
        "neutral": f"`{len(cases)}` samples with learned gate closest to fixed gate=0.5 on affected MAE",
        "failure": f"`{len(cases)}` samples where learned gate underperforms fixed gate=0.5 most on affected MAE",
        "mixed": (
            f"mixed cases with up to `{args.cases_per_category}` success, neutral, and failure samples "
            "based on affected MAE gain over fixed gate=0.5"
        ),
    }
    lines = [
        "# Dual-Branch Gate Case Studies",
        "",
        f"- split: `{args.split}`",
        f"- selection: {selection_notes[args.selection]}",
        "",
        "## Selected Cases",
        "",
        case_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Reading the Heatmaps",
            "",
            "- The first strip marks affected candidate sensors.",
            "- The gate heatmap shows the learned incident-branch weight. Larger values mean stronger reliance on the incident-graph residual branch.",
            "- The target residual heatmap shows where the normal counterfactual forecast is most wrong.",
            "- The fixed-minus-learned error heatmap is positive where learned gate improves over a fixed 0.5 fusion.",
            "- The normal-minus-incident branch error heatmap is positive where the incident branch is locally more accurate.",
        ]
    )
    (output_dir / "case_study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    ckpt = torch_load(model_dir / "model.pt")
    cache_path = resolve_cache_path(model_dir, ckpt)
    residual_beta = float(ckpt.get("residual_beta", 1.0))
    model_args = ckpt.get("args", {})
    if not isinstance(model_args, dict):
        model_args = {}
    device = choose_device(args.device)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)

    model = make_model(ckpt, cache_path, device)
    indices = split_indices(cache_path)[args.split]
    selected_indices = cap_indices(indices, args.max_samples, args.seed)
    metrics_df = run_inference_table(
        model=model,
        cache_path=cache_path,
        selected_indices=selected_indices,
        residual_beta=residual_beta,
        model_args=model_args,
        batch_size=args.batch_size,
        device=device,
    )
    metrics_df.to_csv(output_dir / "candidate_case_metrics.csv", index=False)

    cases = pick_cases(metrics_df, args.num_cases, args.selection, args.cases_per_category)
    cases.insert(0, "rank", np.arange(1, len(cases) + 1))
    cases.to_csv(output_dir / "selected_cases.csv", index=False)

    stats = compute_stats(cache_path)
    paths = []
    for _, row in cases.iterrows():
        path = visualize_case(
            model=model,
            cache_path=cache_path,
            sample_idx=int(row["sample_idx"]),
            residual_beta=residual_beta,
            model_args=model_args,
            stats=stats,
            device=device,
            output_dir=output_dir,
            rank=int(row["rank"]),
            metrics_row=row,
        )
        paths.append(path)
        print(f"wrote {path.name}", flush=True)

    write_report(cases, paths, output_dir, args)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "split": args.split,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "num_cases": args.num_cases,
                "selection": args.selection,
                "cases_per_category": args.cases_per_category,
                "residual_beta": residual_beta,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote case studies to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
