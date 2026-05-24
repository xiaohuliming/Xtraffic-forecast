#!/usr/bin/env python3
"""Apply our (incident, sensor)->affected labels to IGSTGNN's per-county
predictions to compute MAE/RMSE on affected vs unaffected nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGION_TO_LABEL_DIR = {
    "Alameda": "Alameda",
    "Contra_Costa": "ContraCosta",
    "Orange": "Orange",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=list(REGION_TO_LABEL_DIR.keys()))
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--baseline-root", type=Path,
                   default=Path("/Users/xhlm/Desktop/Study/科研实习/baselines/IGSTGNN"))
    p.add_argument("--label-root", type=Path,
                   default=Path("/Users/xhlm/Desktop/Study/科研实习/outputs/impact_labels"))
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def masked_mae(pred, label, mask):
    diff = np.abs(pred - label) * mask
    n = max(mask.sum(), 1)
    return float(diff.sum() / n)


def masked_rmse(pred, label, mask):
    diff = ((pred - label) ** 2) * mask
    n = max(mask.sum(), 1)
    return float(np.sqrt(diff.sum() / n))


def main():
    args = parse_args()
    label_dir = args.label_root / REGION_TO_LABEL_DIR[args.region]
    region_data = args.baseline_root / "data" / "xtraffic" / args.region
    save_dir = args.baseline_root / "experiments" / "igstgnn" / f"{args.region}_{args.seed}"

    print(f"[1/5] Load test predictions: {save_dir / 'test_predictions.npz'}")
    npz = np.load(save_dir / "test_predictions.npz")
    preds = npz["preds"]    # (N_test, 12, N_county)
    labels = npz["labels"]  # (N_test, 12, N_county)
    print(f"  preds {preds.shape}  labels {labels.shape}")

    print(f"[2/5] Load test sample metadata: {region_data / 'incident_data_test.npy'}")
    samples = np.load(region_data / "incident_data_test.npy", allow_pickle=True)
    print(f"  {len(samples)} samples")
    n_test = min(len(samples), preds.shape[0])
    samples = samples[:n_test]
    preds = preds[:n_test]
    labels = labels[:n_test]

    print("[3/5] Load impact labels (event + node)")
    event_df = pd.read_csv(label_dir / "event_labels.csv")
    node_df = pd.read_csv(label_dir / "node_labels.csv")
    print(f"  events: {len(event_df)}  node-rows: {len(node_df)}")

    # Index event_df by start_idx for quick lookup; multiple incidents may share same start_idx
    event_lookup = event_df.groupby("start_idx").apply(
        lambda g: g[["incident_id"]].to_dict("records")
    ).to_dict()
    # Map incident_id -> set of (region_node_idx, affected)
    node_lookup = node_df.groupby("incident_id")[["region_node_idx", "affected"]].apply(
        lambda g: g.to_numpy()
    ).to_dict()

    print("[4/5] Build per-sample affected mask matrix")
    n_county = preds.shape[2]
    aff_mask_all = np.zeros((n_test, n_county), dtype=bool)
    sample_has_label = np.zeros(n_test, dtype=bool)
    sample_n_affected = np.zeros(n_test, dtype=np.int32)

    for s_idx, sample in enumerate(samples):
        t_idx = int(sample.get("_t_idx", -1))
        if t_idx < 0 or t_idx not in event_lookup:
            continue
        # Pick first incident at this t_idx (rare collisions)
        incident_id = event_lookup[t_idx][0]["incident_id"]
        if incident_id not in node_lookup:
            continue
        rows = node_lookup[incident_id]  # array of [region_node_idx, affected]
        for region_node_idx, affected in rows:
            ridx = int(region_node_idx)
            if 0 <= ridx < n_county and int(affected) == 1:
                aff_mask_all[s_idx, ridx] = True
        sample_has_label[s_idx] = True
        sample_n_affected[s_idx] = aff_mask_all[s_idx].sum()

    n_with_label = int(sample_has_label.sum())
    print(f"  matched {n_with_label}/{n_test} samples to incident labels")
    print(f"  affected nodes per sample: mean={sample_n_affected[sample_has_label].mean():.2f}  "
          f"max={sample_n_affected[sample_has_label].max()}  "
          f"zero-affected: {int((sample_n_affected[sample_has_label]==0).sum())}")

    print("[5/5] Compute per-horizon metrics")
    # Build masks for valid label values (mask out missing/zero)
    mask_valid = labels > 0  # IGSTGNN convention: label > 0 means valid
    # affected mask broadcast across horizon
    aff_mask = aff_mask_all[:, None, :] & mask_valid  # (N, 12, N_county)
    unaff_mask = (~aff_mask_all[:, None, :]) & mask_valid

    # Restrict to samples that have labels (otherwise affected mask is all-zero, biases stats)
    label_only = sample_has_label[:, None, None]
    aff_mask = aff_mask & label_only
    unaff_mask = unaff_mask & label_only
    all_mask = mask_valid & label_only

    out = {
        "region": args.region,
        "seed": args.seed,
        "n_test_total": n_test,
        "n_test_matched": n_with_label,
        "mean_affected_nodes_per_incident": float(sample_n_affected[sample_has_label].mean()),
        "by_horizon": [],
    }
    for h in range(preds.shape[1]):
        p = preds[:, h, :]
        y = labels[:, h, :]
        a = aff_mask[:, h, :]
        u = unaff_mask[:, h, :]
        all_h = all_mask[:, h, :]
        row = {
            "horizon": h + 1,
            "MAE_all": masked_mae(p, y, all_h),
            "MAE_affected": masked_mae(p, y, a),
            "MAE_unaffected": masked_mae(p, y, u),
            "RMSE_all": masked_rmse(p, y, all_h),
            "RMSE_affected": masked_rmse(p, y, a),
            "RMSE_unaffected": masked_rmse(p, y, u),
            "n_affected_pixels": int(a.sum()),
            "n_unaffected_pixels": int(u.sum()),
            "n_all_pixels": int(all_h.sum()),
        }
        out["by_horizon"].append(row)

    out["average"] = {
        "MAE_all": float(np.mean([r["MAE_all"] for r in out["by_horizon"]])),
        "MAE_affected": float(np.mean([r["MAE_affected"] for r in out["by_horizon"]])),
        "MAE_unaffected": float(np.mean([r["MAE_unaffected"] for r in out["by_horizon"]])),
        "RMSE_all": float(np.mean([r["RMSE_all"] for r in out["by_horizon"]])),
        "RMSE_affected": float(np.mean([r["RMSE_affected"] for r in out["by_horizon"]])),
        "RMSE_unaffected": float(np.mean([r["RMSE_unaffected"] for r in out["by_horizon"]])),
    }
    out["affected_vs_unaffected_gap_avg_MAE"] = (
        out["average"]["MAE_affected"] - out["average"]["MAE_unaffected"]
    )

    print()
    print(f"=== {args.region} affected-vs-unaffected (raw flow) ===")
    print(f"  Avg MAE all:        {out['average']['MAE_all']:.4f}")
    print(f"  Avg MAE affected:   {out['average']['MAE_affected']:.4f}")
    print(f"  Avg MAE unaffected: {out['average']['MAE_unaffected']:.4f}")
    print(f"  Gap (aff - unaff):  {out['affected_vs_unaffected_gap_avg_MAE']:+.4f}  ({out['affected_vs_unaffected_gap_avg_MAE']/out['average']['MAE_unaffected']*100:+.1f}%)")
    print()
    print("  Per-horizon affected MAE:")
    for r in out["by_horizon"]:
        print(f"    H{r['horizon']:>2}  all={r['MAE_all']:6.3f}  aff={r['MAE_affected']:6.3f}  unaff={r['MAE_unaffected']:6.3f}  n_aff_pix={r['n_affected_pixels']:>9}")

    out_path = args.out or save_dir / "affected_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
