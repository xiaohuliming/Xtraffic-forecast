"""Apples-to-apples MAE comparison: FourierDualNet vs IGSTGNN on overlapping test timesteps.

Both models predict the SAME 12-step windows (matched via _t_idx == sample_start).
This avoids retraining and avoids data-pipeline mismatch.

IGSTGNN's test_predictions.npz contains (S, 12, N) preds and labels for their 4528 samples.
Our test_predictions.npz contains (S, 12, N) preds + affected_mask + sample_start.

Strategy:
  for each unique anchor timestep that appears in BOTH test sets:
    - their pred = mean over all their samples with that _t_idx
    - our pred = any of our samples with that sample_start (predictions are identical
      across our samples at the same sample_start since FourierDualNet uses only x_hist
      which is determined by sample_start)
    - y_true is identical for both models (same future window)
    - affected_mask = union over all our samples at this timestep (because different
      events at the same timestep can mark different nodes as affected)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def load_our_predictions(region: str):
    """Load our FourierDualNet predictions for a region (both modes)."""
    out = {}
    for mode in ["fixed_k_K3", "learnable_K3"]:
        p = Path(f"outputs/fourier_dual_net/{mode}/{region}/test_predictions.npz")
        if not p.exists():
            continue
        d = np.load(p)
        out[mode] = {
            "pred": d["pred_raw_flow"],          # (S, T_p, N)
            "actual": d["actual_future_flow"],   # (S, T_p, N)
            "mask": d["y_mask_flow"],            # (S, T_p, N)
            "aff": d["affected_mask"],           # (S, N)
            "sample_start": d["sample_start"],   # (S,)
        }
    # also load GWN baseline
    gp = Path(f"outputs/baselines/graphwavenet/{region}/test_predictions.npz")
    if gp.exists():
        d = np.load(gp)
        out["GraphWaveNet"] = {
            "pred": d["pred_raw_flow"],
            "actual": d["actual_future_flow"],
            "mask": d["y_mask_flow"],
            "aff": d["affected_mask"],
            "sample_start": d["sample_start"],
        }
    return out


def load_their_predictions(region_dir: str):
    """Load IGSTGNN's predictions for a region. They store preds + labels only.

    Need to also load their test samples to get _t_idx per prediction row.
    """
    pred_p = Path(f"baselines/IGSTGNN/experiments/IGSTGNN/{region_dir}/test_predictions.npz")
    sample_p = Path(f"baselines/IGSTGNN/data/xtraffic/{region_dir.replace('_23','')}/incident_data_test.npy")
    d = np.load(pred_p)
    samples = np.load(sample_p, allow_pickle=True)
    t_idx_full = np.array([s["_t_idx"] for s in samples], dtype=np.int64)
    n_pred = d["preds"].shape[0]
    # drop_last truncation: predictions cover first n_pred samples
    return {
        "pred": d["preds"],
        "label": d["labels"],
        "t_idx": t_idx_full[:n_pred],
        "n_full": len(t_idx_full),
        "n_pred": n_pred,
    }


def aggregate_their_preds_by_tidx(their):
    """Average their predictions across samples sharing the same _t_idx."""
    t_idx = their["t_idx"]
    unique_tidx = np.unique(t_idx)
    pred = their["pred"]
    label = their["label"]

    agg_pred = np.empty((len(unique_tidx), pred.shape[1], pred.shape[2]), dtype=pred.dtype)
    agg_label = np.empty((len(unique_tidx), label.shape[1], label.shape[2]), dtype=label.dtype)
    for i, ti in enumerate(unique_tidx):
        idx = np.where(t_idx == ti)[0]
        agg_pred[i] = pred[idx].mean(axis=0)
        agg_label[i] = label[idx].mean(axis=0)
    return unique_tidx, agg_pred, agg_label


def aggregate_our(data):
    """Take one prediction per unique sample_start. Union affected_masks."""
    ss = data["sample_start"]
    unique_ss = np.unique(ss)
    pred = data["pred"]
    actual = data["actual"]
    mask = data["mask"]
    aff = data["aff"]

    agg_pred = np.empty((len(unique_ss), pred.shape[1], pred.shape[2]), dtype=pred.dtype)
    agg_actual = np.empty_like(agg_pred)
    agg_mask = np.empty((len(unique_ss), mask.shape[1], mask.shape[2]), dtype=bool)
    agg_aff = np.empty((len(unique_ss), aff.shape[1]), dtype=bool)
    for i, ss_i in enumerate(unique_ss):
        idx = np.where(ss == ss_i)[0]
        agg_pred[i] = pred[idx[0]]
        agg_actual[i] = actual[idx[0]]
        agg_mask[i] = mask[idx[0]]
        # union affected across all our samples at this sample_start
        agg_aff[i] = aff[idx].any(axis=0)
    return unique_ss, agg_pred, agg_actual, agg_mask, agg_aff


def mae_breakdown(pred, actual, mask, aff):
    diff = np.abs(pred - actual)
    S, T_p, N = pred.shape
    aff_TpN = np.broadcast_to(aff[:, None, :], (S, T_p, N))
    return {
        "all": float(diff[mask].mean()),
        "affected": float(diff[mask & aff_TpN].mean()),
        "unaffected": float(diff[mask & ~aff_TpN].mean()),
    }


def main():
    regions = [("Alameda", "Alameda_23"), ("ContraCosta", "Contra_Costa_23")]
    all_results = {}

    for region_name, region_dir in regions:
        print(f"\n{'='*78}")
        print(f"=== {region_name}: overlap comparison ===")
        print(f"{'='*78}")

        ours_all = load_our_predictions(region_name)
        their = load_their_predictions(region_dir)
        if not ours_all or their is None:
            print(f"  missing data, skipping")
            continue

        # Aggregate their preds by _t_idx
        their_tidx, their_pred, their_label = aggregate_their_preds_by_tidx(their)
        print(f"  IGSTGNN: {len(their['t_idx'])} raw samples → {len(their_tidx)} unique t_idx")

        # Aggregate ours per unique sample_start (predictions are same anyway)
        ours_key = next(iter(ours_all.keys()))
        ss_unique, _, _, _, _ = aggregate_our(ours_all[ours_key])
        print(f"  Ours: {len(ours_all[ours_key]['sample_start'])} raw → {len(ss_unique)} unique sample_start")

        overlap = np.intersect1d(their_tidx, ss_unique)
        print(f"  Overlap timesteps: {len(overlap)}")
        if len(overlap) == 0:
            continue

        # Build aligned arrays — use overlap order
        # For each overlap timestep, get IGSTGNN pred and our pred
        their_t_to_idx = {t: i for i, t in enumerate(their_tidx)}
        their_aligned = np.empty((len(overlap), their_pred.shape[1], their_pred.shape[2]), dtype=np.float32)
        for i, t in enumerate(overlap):
            their_aligned[i] = their_pred[their_t_to_idx[t]]

        results = {}
        for model_name, data in ours_all.items():
            ss_u, our_pred_u, our_act_u, our_mask_u, our_aff_u = aggregate_our(data)
            ss_to_idx = {s: i for i, s in enumerate(ss_u)}
            our_pred_aligned = np.empty_like(their_aligned)
            our_act_aligned = np.empty_like(their_aligned)
            our_mask_aligned = np.empty(their_aligned.shape, dtype=bool)
            our_aff_aligned = np.empty((len(overlap), our_aff_u.shape[1]), dtype=bool)
            for i, t in enumerate(overlap):
                idx = ss_to_idx[t]
                our_pred_aligned[i] = our_pred_u[idx]
                our_act_aligned[i] = our_act_u[idx]
                our_mask_aligned[i] = our_mask_u[idx]
                our_aff_aligned[i] = our_aff_u[idx]
            results[model_name] = mae_breakdown(our_pred_aligned, our_act_aligned, our_mask_aligned, our_aff_aligned)

            # IGSTGNN: use OUR y_true + OUR mask + OUR affected_mask for fair comparison
            if model_name == "GraphWaveNet":  # only compute once
                igs_result = mae_breakdown(their_aligned, our_act_aligned, our_mask_aligned, our_aff_aligned)
                results["IGSTGNN"] = igs_result

        all_results[region_name] = {"overlap_n": int(len(overlap)), **results}

        print(f"\n  {'Model':<22}{'all':<10}{'affected':<12}{'unaffected':<12}")
        print(f"  {'-'*56}")
        order = ["GraphWaveNet", "IGSTGNN", "fixed_k_K3", "learnable_K3"]
        for model_name in order:
            if model_name not in results:
                continue
            r = results[model_name]
            label = "FourierDualNet " + model_name if "K3" in model_name else model_name
            print(f"  {label:<22}{r['all']:<10.3f}{r['affected']:<12.3f}{r['unaffected']:<12.3f}")

    out = Path("outputs/fourier_dual_net/overlap_comparison.json")
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
