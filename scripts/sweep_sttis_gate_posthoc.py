#!/usr/bin/env python3
"""Post-hoc gate transform sweep for a trained dual-branch ST-TIS model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from analyze_dual_branch_gate import IndexedH5IncidentDataset, make_model, torch_load
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


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
        default=Path("outputs/impact_guided_next_stage/dual_branch_sttis_gate_posthoc_sweep_seed_23"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--transforms",
        default="",
        help="Comma-separated gate transforms. Defaults to the full built-in sweep.",
    )
    parser.add_argument(
        "--betas",
        default="",
        help="Comma-separated residual beta values. Defaults to 0.9,0.95,1.0,1.05,1.1.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def cap_indices(indices: torch.Tensor, max_samples: int, seed: int) -> torch.Tensor:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(indices.size, generator=generator)[:max_samples]
    return torch.sort(torch.as_tensor(indices)[perm]).values.numpy()


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


def transform_gate(
    gate: torch.Tensor,
    name: str,
    normal_residual: torch.Tensor | None = None,
    incident_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    if name == "original":
        return gate
    if name == "fixed_05":
        return torch.full_like(gate, 0.5)
    if name.startswith("shrink_"):
        scale = float(name.split("_", 1)[1])
        return (0.5 + scale * (gate - 0.5)).clamp(0.0, 1.0)
    if name.startswith("bias_"):
        bias = float(name.split("_", 1)[1])
        logits = torch.logit(gate.clamp(1e-5, 1.0 - 1e-5))
        return torch.sigmoid(logits + bias)
    if name.startswith("temp_"):
        temp = float(name.split("_", 1)[1])
        logits = torch.logit(gate.clamp(1e-5, 1.0 - 1e-5))
        return torch.sigmoid(logits * temp)
    if name.startswith("disagree_down_"):
        if normal_residual is None or incident_residual is None:
            raise ValueError(f"{name} requires branch residual proposals")
        scale = float(name.split("_", 2)[2])
        logits = torch.logit(gate.clamp(1e-5, 1.0 - 1e-5))
        disagreement = (incident_residual - normal_residual).abs()
        return torch.sigmoid(logits - scale * disagreement)
    if name.startswith("disagree_cap_"):
        if normal_residual is None or incident_residual is None:
            raise ValueError(f"{name} requires branch residual proposals")
        _prefix, _kind, threshold, cap = name.split("_", 3)
        disagreement = (incident_residual - normal_residual).abs()
        return torch.where(disagreement > float(threshold), torch.minimum(gate, torch.full_like(gate, float(cap))), gate)
    if name.startswith("magdiff_cap_"):
        if normal_residual is None or incident_residual is None:
            raise ValueError(f"{name} requires branch residual proposals")
        _prefix, _kind, threshold, cap = name.split("_", 3)
        mag_diff = incident_residual.abs() - normal_residual.abs()
        return torch.where(mag_diff > float(threshold), torch.minimum(gate, torch.full_like(gate, float(cap))), gate)
    if name.startswith("magdiff_"):
        if normal_residual is None or incident_residual is None:
            raise ValueError(f"{name} requires branch residual proposals")
        scale = float(name.split("_", 1)[1])
        logits = torch.logit(gate.clamp(1e-5, 1.0 - 1e-5))
        mag_diff = incident_residual.abs() - normal_residual.abs()
        return torch.sigmoid(logits - scale * mag_diff)
    raise ValueError(f"unknown gate transform: {name}")


def empty_sums(betas: list[float]) -> dict[float, dict[str, float]]:
    return {
        beta: {
            "all_model": 0.0,
            "all_base": 0.0,
            "all_count": 0.0,
            "aff_model": 0.0,
            "aff_base": 0.0,
            "aff_count": 0.0,
            "unaff_model": 0.0,
            "unaff_base": 0.0,
            "unaff_count": 0.0,
        }
        for beta in betas
    }


def update_sums(
    sums: dict[float, dict[str, float]],
    residual: torch.Tensor,
    y: torch.Tensor,
    masks: dict[str, torch.Tensor],
    betas: list[float],
) -> None:
    base_abs = y.abs()
    for beta in betas:
        model_abs = (beta * residual - y).abs()
        for prefix, mask in [("all", masks["all"]), ("aff", masks["affected"]), ("unaff", masks["unaffected"])]:
            count = mask.sum().item()
            if count <= 0:
                continue
            sums[beta][f"{prefix}_model"] += float(model_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_base"] += float(base_abs[mask].sum().detach().cpu())
            sums[beta][f"{prefix}_count"] += float(count)


def summarize(split: str, transform: str, sums: dict[float, dict[str, float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for beta, vals in sums.items():
        row: dict[str, float | str] = {"split": split, "transform": transform, "beta": beta}
        for prefix, label in [("all", "all"), ("aff", "affected"), ("unaff", "unaffected")]:
            model_mae = vals[f"{prefix}_model"] / max(vals[f"{prefix}_count"], 1.0)
            base_mae = vals[f"{prefix}_base"] / max(vals[f"{prefix}_count"], 1.0)
            row[f"{label}_mae"] = model_mae
            row[f"{label}_baseline_mae"] = base_mae
            row[f"{label}_gain_pct"] = 100.0 * (base_mae - model_mae) / base_mae if base_mae > 0 else math.nan
        rows.append(row)
    return rows


def evaluate_split(
    split: str,
    model: torch.nn.Module,
    cache_path: Path,
    indices: torch.Tensor,
    transforms: list[str],
    betas: list[float],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    all_sums = {name: empty_sums(betas) for name in transforms}
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
            model_args = getattr(model, "hist_input_channels", hist.shape[-1])
            if int(model_args) > hist.shape[-1]:
                hist = torch.cat([hist, hist_normal], dim=-1)
            _pred_y, _pred_impact, _pred_event, _pred_node, details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            normal_residual = details["normal_residual"]
            incident_residual = details["incident_residual"]
            gate = details["gate"]
            masks = {
                "all": y_mask.bool(),
                "affected": y_mask.bool() & node_affected[:, None, :, None].bool(),
                "unaffected": y_mask.bool()
                & (~node_affected[:, None, :, None].bool())
                & node_valid[:, None, :, None].bool(),
            }
            for name in transforms:
                transformed_gate = transform_gate(gate, name, normal_residual, incident_residual)
                residual = (1.0 - transformed_gate) * normal_residual + transformed_gate * incident_residual
                update_sums(all_sums[name], residual, y, masks, betas)
    rows = []
    for name in transforms:
        rows.extend(summarize(split, name, all_sums[name]))
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    best_val_all = val_df.loc[val_df["all_mae"].idxmin()]
    best_val_aff = val_df.loc[val_df["affected_mae"].idxmin()]

    def matching_test(row: pd.Series) -> pd.Series:
        match = test_df[(test_df["transform"] == row["transform"]) & (test_df["beta"] == row["beta"])]
        return match.iloc[0]

    test_all = matching_test(best_val_all)
    test_aff = matching_test(best_val_aff)
    lines = [
        "# ST-TIS Gate Post-hoc Sweep",
        "",
        "This sweep changes only the trained gate at inference time. The model weights are unchanged.",
        "",
        "## Best by validation all MAE",
        "",
        f"- transform: `{best_val_all['transform']}`",
        f"- beta: `{best_val_all['beta']:.2f}`",
        f"- validation all / affected MAE: `{best_val_all['all_mae']:.4f}` / `{best_val_all['affected_mae']:.4f}`",
        f"- test all / affected MAE: `{test_all['all_mae']:.4f}` / `{test_all['affected_mae']:.4f}`",
        "",
        "## Best by validation affected MAE",
        "",
        f"- transform: `{best_val_aff['transform']}`",
        f"- beta: `{best_val_aff['beta']:.2f}`",
        f"- validation all / affected MAE: `{best_val_aff['all_mae']:.4f}` / `{best_val_aff['affected_mae']:.4f}`",
        f"- test all / affected MAE: `{test_aff['all_mae']:.4f}` / `{test_aff['affected_mae']:.4f}`",
        "",
        "## Top validation all rows",
        "",
        val_df.sort_values("all_mae").head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top validation affected rows",
        "",
        val_df.sort_values("affected_mae").head(12).to_markdown(index=False, floatfmt=".4f"),
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
    default_transforms = [
        "original",
        "fixed_05",
        "shrink_0.0",
        "shrink_0.25",
        "shrink_0.5",
        "shrink_0.75",
        "shrink_1.25",
        "bias_-0.4",
        "bias_-0.2",
        "bias_0.2",
        "bias_0.4",
        "temp_0.75",
        "temp_1.25",
        "temp_1.5",
        "magdiff_0.05",
        "magdiff_0.10",
        "magdiff_0.20",
        "magdiff_0.40",
        "disagree_down_0.05",
        "disagree_down_0.10",
        "disagree_down_0.20",
        "disagree_cap_2.0_0.1",
        "disagree_cap_2.0_0.2",
        "disagree_cap_2.0_0.3",
        "disagree_cap_4.0_0.1",
        "disagree_cap_4.0_0.2",
        "disagree_cap_4.0_0.3",
        "disagree_cap_6.0_0.1",
        "disagree_cap_6.0_0.2",
        "disagree_cap_6.0_0.3",
        "disagree_cap_8.0_0.1",
        "disagree_cap_8.0_0.2",
        "disagree_cap_8.0_0.3",
        "magdiff_cap_2.0_0.1",
        "magdiff_cap_2.0_0.2",
        "magdiff_cap_2.0_0.3",
        "magdiff_cap_4.0_0.1",
        "magdiff_cap_4.0_0.2",
        "magdiff_cap_4.0_0.3",
        "magdiff_cap_6.0_0.1",
        "magdiff_cap_6.0_0.2",
        "magdiff_cap_6.0_0.3",
        "magdiff_cap_8.0_0.1",
        "magdiff_cap_8.0_0.2",
        "magdiff_cap_8.0_0.3",
    ]
    transforms = parse_csv_list(args.transforms) if args.transforms else default_transforms
    betas = parse_float_list(args.betas) if args.betas else [round(x, 2) for x in [0.9, 0.95, 1.0, 1.05, 1.1]]
    val_idx = cap_indices(splits["val"], args.max_samples, args.seed)
    test_idx = cap_indices(splits["test"], args.max_samples, args.seed + 1)
    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"model: {model_dir}", flush=True)
    val_df = evaluate_split("val", model, cache_path, val_idx, transforms, betas, args.batch_size, device)
    test_df = evaluate_split("test", model, cache_path, test_idx, transforms, betas, args.batch_size, device)
    val_df.to_csv(output_dir / "val_gate_posthoc_sweep.csv", index=False)
    test_df.to_csv(output_dir / "test_gate_posthoc_sweep.csv", index=False)
    write_summary(output_dir, val_df, test_df)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dir": str(model_dir),
                "cache_path": str(cache_path),
                "transforms": transforms,
                "betas": betas,
                "max_samples": args.max_samples,
            },
            f,
            indent=2,
        )
    print(f"wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
