"""Compute IGSTGNN affected/unaffected MAE breakdown from V100 predictions.

V100 IGSTGNN output: (S_subset, 12, N) with their own sample ordering.
Our FourierDualNet ground truth has affected_mask aligned with sample_start.

Approach:
1. Load IGSTGNN preds + labels (their order, S_subset samples)
2. Load our test samples (with affected_mask)
3. Match by ground-truth y_data: find which of our test samples has the same actual_future_flow
   as IGSTGNN's label[j] — that gives us the affected_mask for IGSTGNN's sample j
4. Compute affected/unaffected MAE
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def load_igstgnn(region_dir: str):
    pred_p = Path(f"outputs/igstgnn_ours_pipeline_fixed/{region_dir}/test_predictions.npz")
    d = np.load(pred_p)
    return d["preds"], d["labels"]


def match_and_breakdown(region: str, region_dir: str):
    igs_pred, igs_label = load_igstgnn(region_dir)
    S_igs, T_p, N_igs = igs_pred.shape
    print(f"\n=== {region} ===")
    print(f"IGSTGNN preds: {igs_pred.shape}, labels: {igs_label.shape}")

    # Load our test_predictions (we use it as reference — has affected_mask + matching y data)
    fdn = np.load(f"outputs/fourier_dual_net/learnable_K3/{region}/test_predictions.npz")
    our_actual = fdn["actual_future_flow"]    # (S_ours, T_p, N)
    our_aff = fdn["affected_mask"]             # (S_ours, N)
    our_mask = fdn["y_mask_flow"]              # (S_ours, T_p, N)
    our_ss = fdn["sample_start"]               # (S_ours,)
    S_ours, _, N_ours = our_actual.shape
    print(f"Our test:   actual {our_actual.shape}, affected {our_aff.shape}")

    assert N_igs == N_ours, f"node count mismatch: IGS={N_igs} vs ours={N_ours}"

    # IGSTGNN test samples are a subset (dropped due to batch_size truncation + filtering).
    # We need to match each IGSTGNN sample to one of OUR rows.
    # IGSTGNN samples are in same ORDER as the dict samples list which is BY sample_start
    # in our data (after filtering for n_active >= 1).
    # Since ALL our test samples have n_active >= 1, IGSTGNN's order should match the unique
    # sample_starts in our test set (first occurrence of each).

    # Build map: unique sample_start -> first row in our_actual
    unique_ss, first_idx = np.unique(our_ss, return_index=True)
    # Sort by row index (preserve our original order)
    order = np.argsort(first_idx)
    ss_in_order = unique_ss[order]
    row_in_order = first_idx[order]

    print(f"Unique sample_starts in our test: {len(unique_ss)}")
    print(f"IGSTGNN has {S_igs} samples — truncated from {len(unique_ss)} (drop_last on bs=48)")

    # We need a robust matching. Strategy: match each IGSTGNN sample to OUR row by
    # nearest-MAE of label vs actual_future_flow.
    # But this is O(S_igs × S_ours × T·N) = too slow if S_igs ~ 9000.
    # Faster: use the first non-zero label value at horizon 0, node 0 as fingerprint.

    # Adapter stores sample_start = first history step (s0).
    # FourierDualNet uses sample_start = LAST history step (t).
    # Relationship: t = s0 + (T_h - 1), so our_t[j] = adapter_t_idx[j] + 11.
    # Load adapter's _t_idx values for IGSTGNN samples (first S_igs after drop_last).
    adapter_samples = np.load(f"outputs/igstgnn_data_from_ours/{region}/incident_data_test.npy",
                              allow_pickle=True)
    adapter_t_idx = np.array([s["_t_idx"] for s in adapter_samples[:S_igs]], dtype=np.int64)
    target_t = adapter_t_idx + 11   # convert to FourierDualNet convention

    # Build map: our sample_start -> first row index (since duplicates share y)
    our_ss = fdn["sample_start"]
    unique_ss, first_idx = np.unique(our_ss, return_index=True)
    ss_to_row = dict(zip(unique_ss.tolist(), first_idx.tolist()))

    matches = -np.ones(S_igs, dtype=np.int64)
    for j in range(S_igs):
        t = int(target_t[j])
        if t in ss_to_row:
            matches[j] = ss_to_row[t]

    valid = matches >= 0
    print(f"Matched IGSTGNN samples: {valid.sum()} / {S_igs}")

    # Verify match quality
    if valid.any():
        sample_j = np.where(valid)[0][0]
        our_row = matches[sample_j]
        match_mae = float(np.abs(igs_label[sample_j] - our_actual[our_row]).mean())
        print(f"Sample-0 match: IGS[{sample_j}] vs OUR[{our_row}], label MAE = {match_mae:.4f}")

    # Compute MAE breakdown using matched samples (apples-to-apples: SAME 12-step prediction window)
    matched_idx = np.where(valid)[0]
    our_rows = matches[matched_idx]

    igs_p = igs_pred[matched_idx]
    igs_l = igs_label[matched_idx]
    our_aff_match = our_aff[our_rows]
    our_mask_match = our_mask[our_rows]

    # Also compute FourierDualNet MAE on the SAME windows for direct comparison
    fdn_pred_match = fdn["pred_raw_flow"][our_rows]
    fdn_actual_match = fdn["actual_future_flow"][our_rows]
    # IGSTGNN's labels should equal our actual at matched rows (since same window)
    label_self_diff = float(np.abs(igs_l - fdn_actual_match).mean())
    print(f"  Label agreement (IGSTGNN labels vs our actual on matched rows): mean diff = {label_self_diff:.4f}")

    # Use IGSTGNN's labels as ground truth (since they trained against them)
    diff = np.abs(igs_p - igs_l)
    fdn_diff = np.abs(fdn_pred_match - fdn_actual_match)

    S, T_p, N = diff.shape
    aff_TpN = np.broadcast_to(our_aff_match[:, None, :], (S, T_p, N))

    # Per-horizon MAE
    per_h_all = []; per_h_aff = []; per_h_un = []
    for h in range(T_p):
        m = our_mask_match[:, h]
        a = aff_TpN[:, h]
        d = diff[:, h]
        per_h_all.append(float(d[m].mean()))
        per_h_aff.append(float(d[m & a].mean()) if (m & a).any() else float("nan"))
        per_h_un.append(float(d[m & ~a].mean()) if (m & ~a).any() else float("nan"))

    mae_all = float(diff[our_mask_match].mean())
    mae_aff = float(diff[our_mask_match & aff_TpN].mean())
    mae_un  = float(diff[our_mask_match & ~aff_TpN].mean())

    fdn_mae_all = float(fdn_diff[our_mask_match].mean())
    fdn_mae_aff = float(fdn_diff[our_mask_match & aff_TpN].mean())
    fdn_mae_un  = float(fdn_diff[our_mask_match & ~aff_TpN].mean())

    result = {
        "region": region,
        "n_matched": int(valid.sum()),
        "n_igs": int(S_igs),
        "igs_mae_all": mae_all,
        "igs_mae_affected": mae_aff,
        "igs_mae_unaffected": mae_un,
        "fdn_mae_all_same_windows": fdn_mae_all,
        "fdn_mae_affected_same_windows": fdn_mae_aff,
        "fdn_mae_unaffected_same_windows": fdn_mae_un,
        "delta_fdn_minus_igs_all": fdn_mae_all - mae_all,
        "delta_fdn_minus_igs_affected": fdn_mae_aff - mae_aff,
        "delta_fdn_minus_igs_unaffected": fdn_mae_un - mae_un,
        "per_horizon_all": per_h_all,
        "per_horizon_affected": per_h_aff,
        "per_horizon_unaffected": per_h_un,
    }
    print(f"IGSTGNN (matched n={valid.sum()}):  all={mae_all:.3f}  affected={mae_aff:.3f}  unaffected={mae_un:.3f}")
    print(f"FourierDualNet on same windows: all={fdn_mae_all:.3f}  affected={fdn_mae_aff:.3f}  unaffected={fdn_mae_un:.3f}")
    print(f"Δ (FDN - IGS):                  all={fdn_mae_all - mae_all:+.3f}  affected={fdn_mae_aff - mae_aff:+.3f}  unaffected={fdn_mae_un - mae_un:+.3f}")
    return result


def main():
    regions = []
    for region, region_dir in [("Alameda", "Alameda"), ("ContraCosta", "Contra_Costa"), ("Orange", "Orange")]:
        if Path(f"outputs/igstgnn_ours_pipeline_fixed/{region_dir}/test_predictions.npz").exists():
            regions.append((region, region_dir))

    all_results = {}
    for region, region_dir in regions:
        result = match_and_breakdown(region, region_dir)
        all_results[region] = result

    out_path = Path("outputs/igstgnn_ours_pipeline_fixed/setting_a_breakdown.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
