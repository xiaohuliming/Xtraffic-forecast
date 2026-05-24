#!/usr/bin/env python3
"""Head-to-head comparison: our impact-correction adapter vs IGSTGNN.

For each region, match samples by anchor timestep (ours.sample_start ==
IGSTGNN._t_idx). Both predict 12 5-min steps from their anchor T:
    ours future window  = [T,   T+11]   (ours h=k → time T+k-1)
    IGSTGNN future window = [T+1, T+12] (IGSTGNN h=k → time T+k)
We compare them at the 11 overlapping time points T+1..T+11
    -> ours horizons 2..12  vs  IGSTGNN horizons 1..11
on the user's 36 candidate-node subset, with affected/unaffected breakdown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REGIONS = [
    (0, "Alameda",      "Alameda",      "Alameda"),
    (1, "ContraCosta",  "Contra_Costa", "Contra Costa"),
    (2, "Orange",       "Orange",       "Orange"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ours-pred", type=Path,
                   default=Path("outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_anomgate05_magq090_w005_seed_23/test_raw_flow_predictions.npz"))
    p.add_argument("--cache-h5", type=Path,
                   default=Path("outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5"))
    p.add_argument("--igstgnn-root", type=Path,
                   default=Path("baselines/IGSTGNN"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/impact_guided_next_stage/headtohead_igstgnn"))
    return p.parse_args()


def masked_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.abs(pred - true)[mask].sum() / n)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ours = np.load(args.ours_pred)
    n_total = ours["region_code"].size
    print(f"loaded ours: {n_total} samples")

    # Pull node_affected from H5 in test order
    with h5py.File(args.cache_h5, "r") as h5:
        split = h5["split"][:]
        test_idx = np.sort(np.flatnonzero(split == 2))
        node_affected = h5["node_affected"][test_idx]  # (N, K)
    print(f"node_affected shape: {node_affected.shape}")

    rows = []
    horizon_rows = []

    for region_code, region_name, igs_dirname, display in REGIONS:
        mask_r = ours["region_code"] == region_code
        n_r = int(mask_r.sum())
        ours_S = ours["sample_start"][mask_r]
        ours_node_idx = ours["node_idx"][mask_r]            # (n_r, K)
        ours_node_valid = ours["node_valid"][mask_r] > 0.5  # (n_r, K)
        ours_y_mask = ours["y_mask_flow"][mask_r]           # (n_r, H, K)
        ours_actual = ours["actual_future_flow"][mask_r]    # (n_r, H, K)
        ours_pred = ours["pred_raw_flow"][mask_r]
        ours_source = ours["source_raw_flow"][mask_r]
        ours_aff = node_affected[mask_r] > 0.5              # (n_r, K)

        # Load IGSTGNN
        igs_pred_path = args.igstgnn_root / "experiments" / "igstgnn" / f"{igs_dirname}_23" / "test_predictions.npz"
        igs_samples_path = args.igstgnn_root / "data" / "xtraffic" / igs_dirname / "incident_data_test.npy"
        if not igs_pred_path.exists():
            print(f"[{display}] missing IGSTGNN predictions {igs_pred_path} — skipping")
            continue
        igs_npz = np.load(igs_pred_path)
        igs_preds = igs_npz["preds"]   # (N_igs, 12, N_county)
        igs_labels = igs_npz["labels"]
        igs_samples = np.load(igs_samples_path, allow_pickle=True)
        n_igs = min(len(igs_samples), igs_preds.shape[0])
        igs_t_idx = np.array([int(igs_samples[i].get("_t_idx", -1)) for i in range(n_igs)])
        igs_lookup: dict[int, int] = {}
        for i, t in enumerate(igs_t_idx):
            igs_lookup.setdefault(int(t), i)
        n_county = igs_preds.shape[2]

        # Match
        matched_pairs = []
        for j, s in enumerate(ours_S):
            i = igs_lookup.get(int(s))
            if i is not None:
                matched_pairs.append((j, i))
        if not matched_pairs:
            print(f"[{display}] no matched samples — skipping")
            continue
        ours_idx = np.asarray([p[0] for p in matched_pairs], dtype=np.int64)
        igs_idx = np.asarray([p[1] for p in matched_pairs], dtype=np.int64)
        m = ours_idx.size
        print(f"[{display}] matched {m}/{n_r} ours samples with IGSTGNN ({m / n_r * 100:.1f}%)")

        m_node_idx = ours_node_idx[ours_idx]      # (m, K)
        m_node_valid = ours_node_valid[ours_idx]  # (m, K)
        m_y_mask = ours_y_mask[ours_idx]          # (m, H, K)
        m_actual = ours_actual[ours_idx]          # (m, H, K)
        m_pred = ours_pred[ours_idx]
        m_source = ours_source[ours_idx]
        m_aff = ours_aff[ours_idx]                # (m, K)

        # Slice IGSTGNN per-county to the user's K-node subset, per matched sample.
        K = m_node_idx.shape[1]
        igs_pred_subset = np.empty((m, 12, K), dtype=np.float32)
        igs_label_subset = np.empty((m, 12, K), dtype=np.float32)
        for k, i in enumerate(igs_idx):
            cols = m_node_idx[k]
            igs_pred_subset[k] = igs_preds[i][:, cols]
            igs_label_subset[k] = igs_labels[i][:, cols]

        # Time-aligned overlap: ours horizons 2..12 (idx 1..11) vs IGSTGNN horizons 1..11 (idx 0..10)
        ours_align_pred = m_pred[:, 1:12, :]          # (m, 11, K) — predicts T+1..T+11
        ours_align_source = m_source[:, 1:12, :]
        ours_align_actual = m_actual[:, 1:12, :]
        ours_align_mask = m_y_mask[:, 1:12, :]
        igs_align_pred = igs_pred_subset[:, 0:11, :]  # (m, 11, K)
        igs_align_label = igs_label_subset[:, 0:11, :]
        # IGSTGNN's own valid-label mask: label > 0
        igs_label_valid = igs_align_label > 0

        # Joint mask: both methods have valid targets, both nodes valid
        node_valid_b = m_node_valid[:, None, :]                     # (m, 1, K)
        joint_mask = ours_align_mask & igs_label_valid & node_valid_b
        # Also keep ours-only and igs-only masks for sanity
        # Affected-only joint mask
        aff_b = m_aff[:, None, :]                                   # (m, 1, K)
        unaff_b = ~m_aff[:, None, :] & node_valid_b
        joint_aff = joint_mask & aff_b
        joint_unaff = joint_mask & unaff_b

        # Per-region averaged metrics over the 11-horizon overlap
        for label_mask, scope_name in [(joint_mask, "all_candidates"),
                                       (joint_aff, "affected_only"),
                                       (joint_unaff, "unaffected_only")]:
            pixels = int(label_mask.sum())
            if pixels == 0:
                continue
            mae_ours_ad = masked_mae(ours_align_pred, ours_align_actual, label_mask)
            mae_ours_src = masked_mae(ours_align_source, ours_align_actual, label_mask)
            mae_igs = masked_mae(igs_align_pred, igs_align_label, label_mask)
            rows.append({
                "region": display,
                "scope": scope_name,
                "matched_samples": m,
                "pixels": pixels,
                "MAE_igstgnn": mae_igs,
                "MAE_ours_source": mae_ours_src,
                "MAE_ours_adapter": mae_ours_ad,
                "delta_adapter_minus_igs": mae_ours_ad - mae_igs,
                "delta_source_minus_igs": mae_ours_src - mae_igs,
            })

        # Per-horizon (time-aligned) breakdown
        for h_align in range(11):
            t_offset = h_align + 1  # forecasted timepoint = T + (h_align+1)
            for scope_name, scope_mask in [("all_candidates", joint_mask[:, h_align, :]),
                                           ("affected_only", joint_aff[:, h_align, :]),
                                           ("unaffected_only", joint_unaff[:, h_align, :])]:
                pixels = int(scope_mask.sum())
                if pixels == 0:
                    continue
                mae_ours_ad = masked_mae(ours_align_pred[:, h_align, :], ours_align_actual[:, h_align, :], scope_mask)
                mae_ours_src = masked_mae(ours_align_source[:, h_align, :], ours_align_actual[:, h_align, :], scope_mask)
                mae_igs = masked_mae(igs_align_pred[:, h_align, :], igs_align_label[:, h_align, :], scope_mask)
                horizon_rows.append({
                    "region": display,
                    "scope": scope_name,
                    "minutes_ahead": (t_offset) * 5,
                    "ours_horizon_index": h_align + 2,    # ours h_idx in 2..12
                    "igstgnn_horizon_index": h_align + 1, # igs h_idx in 1..11
                    "matched_samples": m,
                    "pixels": pixels,
                    "MAE_igstgnn": mae_igs,
                    "MAE_ours_source": mae_ours_src,
                    "MAE_ours_adapter": mae_ours_ad,
                })

    summary_df = pd.DataFrame(rows)
    horizon_df = pd.DataFrame(horizon_rows)
    summary_df.to_csv(args.out_dir / "headtohead_summary.csv", index=False)
    horizon_df.to_csv(args.out_dir / "headtohead_per_horizon.csv", index=False)
    print(f"\nsaved → {args.out_dir / 'headtohead_summary.csv'}")
    print(f"saved → {args.out_dir / 'headtohead_per_horizon.csv'}")
    print()
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
