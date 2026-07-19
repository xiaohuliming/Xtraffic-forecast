#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--baseline-root", type=Path,
                   default=Path("/Users/xhlm/Desktop/Study/科研实习/baselines/IGSTGNN"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_prediction_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        if "y_mask" not in npz.files:
            raise RuntimeError(
                f"{path} is missing required y_mask; rerun IGSTGNN with "
                "observation-mask-aware data and evaluation"
            )
        preds = np.asarray(npz["preds"])
        labels = np.asarray(npz["labels"])
        y_mask = np.asarray(npz["y_mask"]).astype(bool, copy=False)
    if preds.shape != labels.shape or y_mask.shape != labels.shape:
        raise RuntimeError(
            f"prediction bundle shapes must match, got preds={preds.shape}, "
            f"labels={labels.shape}, y_mask={y_mask.shape}"
        )
    return preds, labels, y_mask


def masked_mae(pred, label, mask):
    err = np.abs(pred - label) * mask
    return err.sum() / np.maximum(mask.sum(), 1)


def masked_rmse(pred, label, mask):
    sq = ((pred - label) ** 2) * mask
    return np.sqrt(sq.sum() / np.maximum(mask.sum(), 1))


def masked_mape(pred, label, mask):
    err = np.abs((pred - label) / np.maximum(np.abs(label), 1e-6)) * mask
    return err.sum() / np.maximum(mask.sum(), 1) * 100.0


def evaluate(preds, labels, y_mask):
    horizons = preds.shape[1]
    valid_count = int(y_mask.sum())
    if valid_count <= 0:
        raise RuntimeError("evaluation has no observed target values")
    out = {
        "aggregation": "all_observed_targets",
        "valid_count": valid_count,
        "per_horizon": [],
    }
    for h in range(horizons):
        p = preds[:, h, :]
        y = labels[:, h, :]
        mask = y_mask[:, h, :]
        out["per_horizon"].append({
            "h": h + 1,
            "MAE": float(masked_mae(p, y, mask)),
            "RMSE": float(masked_rmse(p, y, mask)),
            "MAPE": float(masked_mape(p, y, mask)),
        })
    out["average"] = {
        "MAE": float(masked_mae(preds, labels, y_mask)),
        "RMSE": float(masked_rmse(preds, labels, y_mask)),
        "MAPE": float(masked_mape(preds, labels, y_mask)),
    }
    return out


def evaluate_subset(preds, labels, y_mask, subset_mask, name):
    if subset_mask.sum() == 0:
        return {"name": name, "n": 0}
    sub_preds = preds[subset_mask]
    sub_labels = labels[subset_mask]
    res = evaluate(sub_preds, sub_labels, y_mask[subset_mask])
    res["name"] = name
    res["n"] = int(subset_mask.sum())
    return res


def main():
    args = parse_args()
    save_dir = args.baseline_root / "experiments" / "igstgnn" / f"{args.region}_{args.seed}"
    pred_path = save_dir / "test_predictions.npz"
    test_data_path = args.baseline_root / "data" / "xtraffic" / args.region / "incident_data_test.npy"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}\n"
                                "Run training first (it auto-runs test at the end).")

    preds, labels, y_mask = load_prediction_bundle(pred_path)
    print(f"Loaded preds {preds.shape} labels {labels.shape}")

    print("\n=== Overall test metrics (raw flow units) ===")
    overall = evaluate(preds, labels, y_mask)
    for h in overall["per_horizon"]:
        print(f"  Horizon {h['h']:>2}: MAE={h['MAE']:.4f}  RMSE={h['RMSE']:.4f}  MAPE={h['MAPE']:.4f}")
    print(f"  Average  : MAE={overall['average']['MAE']:.4f}  RMSE={overall['average']['RMSE']:.4f}  MAPE={overall['average']['MAPE']:.4f}")

    # Subset analysis using incident metadata
    test_samples = np.load(test_data_path, allow_pickle=True)
    print(f"\nLoaded {len(test_samples)} test sample metadata entries")

    if len(test_samples) != preds.shape[0]:
        raise RuntimeError(
            "prediction/sample count mismatch: "
            f"{preds.shape[0]} predictions for {len(test_samples)} samples"
        )

    types = np.array([str(s["event_features"].get("Type", 0)) for s in test_samples])
    descs = np.array([str(s["event_features"].get("Description", 0)) for s in test_samples])
    holidays = np.array([int(s["event_features"].get("Holiday", 0)) for s in test_samples])
    durations = np.array([float(s.get("durations", 0.0)) for s in test_samples])

    print("\n=== By incident Type ===")
    type_results = []
    for t in np.unique(types):
        mask = types == t
        res = evaluate_subset(preds, labels, y_mask, mask, f"type_{t}")
        if res.get("n", 0) > 0:
            print(f"  Type={t:<3} n={res['n']:>5}  MAE_avg={res['average']['MAE']:.4f}")
            type_results.append(res)

    print("\n=== By duration bucket ===")
    duration_results = []
    for label, lo, hi in [("short_<30min", 0, 30), ("medium_30-120min", 30, 120), ("long_>=120min", 120, 1e9)]:
        mask = (durations >= lo) & (durations < hi)
        res = evaluate_subset(preds, labels, y_mask, mask, label)
        if res.get("n", 0) > 0:
            print(f"  {label:<22} n={res['n']:>5}  MAE_avg={res['average']['MAE']:.4f}")
            duration_results.append(res)

    print("\n=== Holiday vs non-holiday ===")
    holiday_results = []
    for h_val, name in [(0, "non_holiday"), (1, "holiday")]:
        mask = holidays == h_val
        res = evaluate_subset(preds, labels, y_mask, mask, name)
        if res.get("n", 0) > 0:
            print(f"  {name:<12} n={res['n']:>5}  MAE_avg={res['average']['MAE']:.4f}")
            holiday_results.append(res)

    out_path = args.out or save_dir / "evaluation_breakdown.json"
    with open(out_path, "w") as f:
        json.dump({
            "region": args.region,
            "seed": args.seed,
            "n_test": int(preds.shape[0]),
            "mask_semantics": "explicit source flow observation mask",
            "overall": overall,
            "by_type": type_results,
            "by_duration": duration_results,
            "by_holiday": holiday_results,
        }, f, indent=2)
    print(f"\nSaved breakdown to {out_path}")


if __name__ == "__main__":
    main()
