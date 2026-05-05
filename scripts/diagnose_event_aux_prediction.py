#!/usr/bin/env python3
"""Diagnose whether predicted event auxiliary signals align with true impact labels."""

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
from train_full_candidate_stgnn_heatmap_model import compute_stats, split_indices
from train_impact_residual_model import choose_device


FEATURES = (
    ("severity", 0),
    ("recovery", 1),
    ("spread", 2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=[
            Path(
                "outputs/impact_guided_next_stage/"
                "dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_conservative_quickgrid"
            ),
            Path(
                "outputs/impact_guided_next_stage/"
                "dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware"
            ),
        ],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["hierarchical_conservative", "afffocus3_groupaware"],
        help="Optional display labels matching --model-dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/impact_guided_next_stage/event_aux_prediction_diagnostics_seed_23"),
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


def cap_indices(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or indices.size <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_samples, replace=False))


def model_uses_dual_hist(model: torch.nn.Module, hist_channels: int) -> bool:
    return int(getattr(model, "hist_input_channels", hist_channels)) > hist_channels


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() <= 1:
        return float("nan")
    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = np.sqrt((xv * xv).sum() * (yv * yv).sum())
    return float((xv * yv).sum() / denom) if denom > 0.0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() <= 1:
        return float("nan")
    return float(pd.Series(x[mask]).rank(method="average").corr(pd.Series(y[mask]).rank(method="average")))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() <= 1:
        return float("nan")
    labels_bool = labels[mask].astype(bool)
    n_pos = int(labels_bool.sum())
    n_neg = int((~labels_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores[mask]).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(ranks[labels_bool].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def to_semantic(raw_event: np.ndarray) -> pd.DataFrame:
    severity = np.expm1(raw_event[:, 0])
    recovery_min = raw_event[:, 1] * 180.0
    spread_nodes = np.expm1(raw_event[:, 2])
    values = np.stack([severity, recovery_min, spread_nodes], axis=1)
    values[~np.isfinite(values)] = 0.0
    values = np.maximum(values, 0.0)
    return pd.DataFrame(values, columns=["severity", "recovery", "spread"])


def event_group_masks(raw_event: np.ndarray) -> dict[str, np.ndarray]:
    semantic = to_semantic(raw_event)
    severity = semantic["severity"].to_numpy(dtype=np.float64)
    recovery = semantic["recovery"].to_numpy(dtype=np.float64)
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


def collect_predictions(
    model_dir: Path,
    label: str,
    cache_path: Path,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    ckpt = torch_load(model_dir / "model.pt")
    model = make_model(ckpt, cache_path, device)
    stats = compute_stats(cache_path)
    dataset = IndexedH5IncidentDataset(cache_path=cache_path, indices=indices, stats=stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    dual_hist = model_uses_dual_hist(model, hist_channels=3)

    pred_std_parts: list[np.ndarray] = []
    true_std_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            (
                hist,
                hist_normal,
                node,
                global_context,
                normal_delta,
                _y,
                _y_mask,
                _impact,
                _impact_mask,
                event_aux,
                _node_affected,
                _node_valid,
                idx,
            ) = batch
            hist = hist.to(device)
            hist_normal = hist_normal.to(device)
            node = node.to(device)
            global_context = global_context.to(device)
            normal_delta = normal_delta.to(device)
            if dual_hist:
                hist = torch.cat([hist, hist_normal], dim=-1)
            _pred_y, _pred_impact, pred_event, _pred_node, _details = model(
                hist,
                node,
                global_context,
                normal_delta,
                return_details=True,
            )
            pred_std_parts.append(pred_event.detach().cpu().numpy().astype(np.float32))
            true_std_parts.append(event_aux.numpy().astype(np.float32))
            idx_parts.append(idx.numpy().astype(np.int64))
            if batch_idx % 20 == 0:
                print(f"{label}: predicted {min(batch_idx * batch_size, indices.size)}/{indices.size}", flush=True)

    pred_std = np.concatenate(pred_std_parts, axis=0)
    true_std = np.concatenate(true_std_parts, axis=0)
    idx_all = np.concatenate(idx_parts, axis=0)
    with h5py.File(cache_path, "r") as h5:
        raw_true = h5["event_aux"][idx_all].astype(np.float32)
    raw_pred = pred_std * stats.event_aux_std.reshape(1, -1) + stats.event_aux_mean.reshape(1, -1)
    return {
        "model": label,
        "pred_std": pred_std,
        "true_std": true_std,
        "raw_pred": raw_pred,
        "raw_true": raw_true,
        "indices": idx_all,
    }


def feature_rows(payload: dict[str, object]) -> list[dict[str, float | str]]:
    label = str(payload["model"])
    pred_std = payload["pred_std"]
    true_std = payload["true_std"]
    raw_pred = payload["raw_pred"]
    raw_true = payload["raw_true"]
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(true_std, np.ndarray)
    assert isinstance(raw_pred, np.ndarray)
    assert isinstance(raw_true, np.ndarray)

    pred_sem = to_semantic(raw_pred)
    true_sem = to_semantic(raw_true)
    rows: list[dict[str, float | str]] = []
    for feature, col in FEATURES:
        ps = pred_std[:, col]
        ts = true_std[:, col]
        pr = raw_pred[:, col]
        tr = raw_true[:, col]
        psem = pred_sem[feature].to_numpy(dtype=np.float64)
        tsem = true_sem[feature].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": label,
                "feature": feature,
                "std_mae": float(np.mean(np.abs(ps - ts))),
                "std_rmse": float(np.sqrt(np.mean((ps - ts) ** 2))),
                "std_bias": float(np.mean(ps - ts)),
                "std_pearson": pearson(ps, ts),
                "std_spearman": spearman(ps, ts),
                "raw_mae": float(np.mean(np.abs(pr - tr))),
                "raw_rmse": float(np.sqrt(np.mean((pr - tr) ** 2))),
                "raw_bias": float(np.mean(pr - tr)),
                "raw_pearson": pearson(pr, tr),
                "raw_spearman": spearman(pr, tr),
                "semantic_mae": float(np.mean(np.abs(psem - tsem))),
                "semantic_rmse": float(np.sqrt(np.mean((psem - tsem) ** 2))),
                "semantic_bias": float(np.mean(psem - tsem)),
                "semantic_pearson": pearson(psem, tsem),
                "semantic_spearman": spearman(psem, tsem),
                "true_std_mean": float(np.mean(ts)),
                "pred_std_mean": float(np.mean(ps)),
                "true_semantic_mean": float(np.mean(tsem)),
                "pred_semantic_mean": float(np.mean(psem)),
                "relu_pred_std_mean": float(np.mean(np.maximum(ps, 0.0))),
            }
        )
    return rows


def classification_rows(payload: dict[str, object]) -> list[dict[str, float | str]]:
    label = str(payload["model"])
    pred_std = payload["pred_std"]
    raw_true = payload["raw_true"]
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(raw_true, np.ndarray)
    semantic = to_semantic(raw_true)
    severity = semantic["severity"].to_numpy(dtype=np.float64)
    recovery = semantic["recovery"].to_numpy(dtype=np.float64)
    spread = semantic["spread"].to_numpy(dtype=np.float64)
    sev_q33, sev_q66 = np.quantile(severity, [1.0 / 3.0, 2.0 / 3.0])
    spread_q66 = np.quantile(spread, 2.0 / 3.0)
    targets = [
        ("severity_high", pred_std[:, 0], severity > sev_q66, sev_q66),
        ("severity_low", -pred_std[:, 0], severity <= sev_q33, sev_q33),
        ("recovery_long_ge90", pred_std[:, 1], recovery >= 90.0, 90.0),
        ("spread_high", pred_std[:, 2], spread > spread_q66, spread_q66),
    ]
    rows: list[dict[str, float | str]] = []
    for target, score, labels, threshold in targets:
        rows.append(
            {
                "model": label,
                "target": target,
                "positive_rate": float(np.mean(labels)),
                "threshold": float(threshold),
                "auc": auc_score(score, labels.astype(np.float32)),
                "score_pos_mean": float(np.mean(score[labels])) if labels.any() else float("nan"),
                "score_neg_mean": float(np.mean(score[~labels])) if (~labels).any() else float("nan"),
                "score_gap": (
                    float(np.mean(score[labels]) - np.mean(score[~labels]))
                    if labels.any() and (~labels).any()
                    else float("nan")
                ),
            }
        )
    return rows


def group_rows(payload: dict[str, object]) -> list[dict[str, float | str]]:
    label = str(payload["model"])
    pred_std = payload["pred_std"]
    true_std = payload["true_std"]
    raw_pred = payload["raw_pred"]
    raw_true = payload["raw_true"]
    assert isinstance(pred_std, np.ndarray)
    assert isinstance(true_std, np.ndarray)
    assert isinstance(raw_pred, np.ndarray)
    assert isinstance(raw_true, np.ndarray)
    pred_sem = to_semantic(raw_pred)
    true_sem = to_semantic(raw_true)
    masks = event_group_masks(raw_true)
    rows: list[dict[str, float | str]] = []
    for group, mask in masks.items():
        if not mask.any():
            continue
        row: dict[str, float | str] = {
            "model": label,
            "group": group,
            "samples": int(mask.sum()),
        }
        for feature, col in FEATURES:
            row[f"true_{feature}_mean"] = float(true_sem.loc[mask, feature].mean())
            row[f"pred_{feature}_mean"] = float(pred_sem.loc[mask, feature].mean())
            row[f"true_{feature}_std_mean"] = float(np.mean(true_std[mask, col]))
            row[f"pred_{feature}_std_mean"] = float(np.mean(pred_std[mask, col]))
            row[f"relu_pred_{feature}_std_mean"] = float(np.mean(np.maximum(pred_std[mask, col], 0.0)))
        rows.append(row)
    return rows


def write_summary(
    output_dir: Path,
    feature_df: pd.DataFrame,
    class_df: pd.DataFrame,
    group_df: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    focus_feature = feature_df[
        feature_df["feature"].isin(["severity", "recovery"])
    ][
        [
            "model",
            "feature",
            "std_mae",
            "std_pearson",
            "std_spearman",
            "semantic_mae",
            "relu_pred_std_mean",
        ]
    ]
    focus_class = class_df[
        class_df["target"].isin(["severity_high", "recovery_long_ge90"])
    ][["model", "target", "positive_rate", "auc", "score_gap"]]
    focus_groups = group_df[
        group_df["group"].isin(["severity_low", "severity_high", "recovery_short_lt30", "recovery_long_ge90"])
    ][
        [
            "model",
            "group",
            "samples",
            "true_severity_mean",
            "pred_severity_std_mean",
            "relu_pred_severity_std_mean",
            "true_recovery_mean",
            "pred_recovery_std_mean",
            "relu_pred_recovery_std_mean",
        ]
    ]
    lines = [
        "# Event-Aux Prediction Diagnostics",
        "",
        f"- split: `{args.split}`",
        "- `pred_event_aux` is the event-level auxiliary output used by event-conditioned veto probes.",
        "- `std_*` compares standardized targets; `semantic_*` compares severity/recovery/spread after undoing normalization.",
        "",
        "## Continuous Alignment",
        "",
        focus_feature.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Group Discrimination",
        "",
        focus_class.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Predicted Signal By True Group",
        "",
        focus_groups.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation Notes",
        "",
        "- A useful dynamic event-conditioned gate should give clearly larger predicted severity/recovery scores on the high/long groups.",
        "- AUC close to 0.5 means the predicted event signal is weak for ranking high-impact events.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.model_dirs):
        raise ValueError("--labels must match --model-dirs")
    labels = args.labels if args.labels else [path.name for path in args.model_dirs]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    first_dir = args.model_dirs[0].resolve()
    first_ckpt = torch_load(first_dir / "model.pt")
    cache_path = resolve_cache_path(first_dir, first_ckpt)
    for model_dir in args.model_dirs[1:]:
        ckpt = torch_load(model_dir.resolve() / "model.pt")
        other_cache = resolve_cache_path(model_dir.resolve(), ckpt)
        if other_cache != cache_path:
            raise ValueError(f"model cache mismatch: {cache_path} vs {other_cache}")

    indices = split_indices(cache_path)[args.split]
    indices = cap_indices(indices, args.max_samples, args.seed)

    print(f"device: {device}", flush=True)
    print(f"cache: {cache_path}", flush=True)
    print(f"samples: {indices.size}", flush=True)

    payloads = []
    for model_dir, label in zip(args.model_dirs, labels):
        print(f"diagnosing event aux for {label}: {model_dir.resolve()}", flush=True)
        payloads.append(
            collect_predictions(
                model_dir=model_dir.resolve(),
                label=label,
                cache_path=cache_path,
                indices=indices,
                batch_size=args.batch_size,
                device=device,
            )
        )

    feature_df = pd.DataFrame([row for payload in payloads for row in feature_rows(payload)])
    class_df = pd.DataFrame([row for payload in payloads for row in classification_rows(payload)])
    group_df = pd.DataFrame([row for payload in payloads for row in group_rows(payload)])
    feature_df.to_csv(output_dir / "event_aux_prediction_summary.csv", index=False)
    class_df.to_csv(output_dir / "event_aux_group_auc.csv", index=False)
    group_df.to_csv(output_dir / "event_group_prediction_means.csv", index=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_dirs": [str(path.resolve()) for path in args.model_dirs],
                "labels": labels,
                "cache_path": str(cache_path),
                "split": args.split,
                "samples": int(indices.size),
                "batch_size": args.batch_size,
                "device": str(device),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    write_summary(output_dir, feature_df, class_df, group_df, args)
    print(f"wrote event-aux diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
