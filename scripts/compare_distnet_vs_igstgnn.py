#!/usr/bin/env python3
"""Head-to-head comparison: DIST-Net vs IGSTGNN on raw-flow test predictions.

NEW FORMAT (vs scripts/compare_headtohead_igstgnn.py for the OLD Codex
pipeline): both methods now predict at the same horizon convention
[t+1, t+12] — no off-by-one slicing needed. Both also produce full-county
shape (S, 12, N) rather than the OLD 36-candidate-subset format.

For each region:
  1. Load DIST-Net predictions from --eval-dir/{region}_test_predictions.npz
  2. Load IGSTGNN predictions from --igstgnn-root/experiments/igstgnn/{r}_23/test_predictions.npz
  3. Match samples by sample_start ↔ _t_idx
  4. Compute raw-flow MAE on all / affected / unaffected node scopes
  5. Per-horizon breakdown (12 timesteps)
  6. Write summary CSVs

Affected_mask comes from the DIST-Net npz (built from node_labels.csv during
cache construction).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = [
    ("Alameda",     "Alameda"),
    ("ContraCosta", "Contra_Costa"),
    ("Orange",      "Orange"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="Directory containing {region}_test_predictions.npz from run_distnet_test_inference.py")
    p.add_argument("--igstgnn-root", type=Path,
                   default=Path("baselines/IGSTGNN"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <eval-dir>/headtohead")
    return p.parse_args()


def masked_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.abs(pred - true)[mask].sum() / n)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.eval_dir / "headtohead")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    horizon_rows = []

    for region_name, igs_dirname in REGIONS:
        ours_path = args.eval_dir / f"{region_name}_test_predictions.npz"
        igs_pred_path = args.igstgnn_root / "experiments" / "igstgnn" / f"{igs_dirname}_23" / "test_predictions.npz"
        igs_samples_path = args.igstgnn_root / "data" / "xtraffic" / igs_dirname / "incident_data_test.npy"

        if not ours_path.exists():
            print(f"[{region_name}] no DIST-Net predictions at {ours_path} — skipping")
            continue
        if not igs_pred_path.exists():
            print(f"[{region_name}] no IGSTGNN predictions at {igs_pred_path} — skipping")
            continue

        ours = np.load(ours_path)
        n_ours = ours["sample_start"].size
        print(f"\n[{region_name}] ours: {n_ours} test samples")

        igs_npz = np.load(igs_pred_path)
        igs_preds = igs_npz["preds"]                          # (N_igs, 12, N_county)
        igs_labels = igs_npz["labels"]
        igs_samples = np.load(igs_samples_path, allow_pickle=True)
        n_igs = min(len(igs_samples), igs_preds.shape[0])
        igs_t_idx = np.array([int(igs_samples[i].get("_t_idx", -1)) for i in range(n_igs)])
        print(f"          igstgnn: {n_igs} test samples, preds shape={igs_preds.shape}")

        # Sanity: same N?
        N_ours = ours["pred_raw_flow"].shape[2]
        N_igs = igs_preds.shape[2]
        if N_ours != N_igs:
            print(f"  WARNING: N differs (ours={N_ours} vs igs={N_igs}); proceed anyway")

        igs_lookup: dict[int, int] = {}
        for i, t in enumerate(igs_t_idx):
            if t >= 0:
                igs_lookup.setdefault(int(t), i)

        # Match
        ours_S = ours["sample_start"]
        matched: list[tuple[int, int]] = []
        for j, s in enumerate(ours_S):
            i = igs_lookup.get(int(s))
            if i is not None:
                matched.append((j, i))
        if not matched:
            print(f"  no matched (region, sample_start) — skipping")
            continue
        ours_idx = np.array([p[0] for p in matched], dtype=np.int64)
        igs_idx = np.array([p[1] for p in matched], dtype=np.int64)
        m = len(matched)
        print(f"  matched: {m}/{n_ours} ({m / n_ours * 100:.1f}%) ours samples")

        ours_pred   = ours["pred_raw_flow"][ours_idx]         # (m, 12, N)
        ours_actual = ours["actual_future_flow"][ours_idx]    # (m, 12, N)
        ours_mask   = ours["y_mask_flow"][ours_idx]           # (m, 12, N)
        ours_aff    = ours["affected_mask"][ours_idx]         # (m, N)

        # IGSTGNN at matched indices, only first N columns (defensive)
        igs_pred_m  = igs_preds[igs_idx][:, :, :N_ours]       # (m, 12, N)
        igs_label_m = igs_labels[igs_idx][:, :, :N_ours]

        # IGSTGNN's own validity mask: label > 0
        igs_label_valid = igs_label_m > 0

        # Joint mask: both methods have valid targets
        joint_mask = ours_mask & igs_label_valid                              # (m, 12, N)
        aff_full = np.broadcast_to(ours_aff[:, None, :], joint_mask.shape)
        joint_aff = joint_mask & aff_full
        joint_unaff = joint_mask & (~aff_full)

        # Per-region aggregate (all 12 horizons)
        for scope_name, mask in [("all_candidates", joint_mask),
                                 ("affected_only", joint_aff),
                                 ("unaffected_only", joint_unaff)]:
            mae_ours = masked_mae(ours_pred, ours_actual, mask)
            mae_igs  = masked_mae(igs_pred_m, igs_label_m, mask)
            summary_rows.append({
                "region": region_name,
                "scope": scope_name,
                "matched_samples": m,
                "pixels": int(mask.sum()),
                "MAE_distnet": mae_ours,
                "MAE_igstgnn": mae_igs,
                "delta_distnet_minus_igs": mae_ours - mae_igs,
            })
            print(f"    {scope_name:18s}  DIST={mae_ours:7.3f}  IGS={mae_igs:7.3f}  "
                  f"Δ={(mae_ours - mae_igs):+.3f}")

        # Per-horizon (12 timesteps)
        for h in range(12):
            ph = ours_pred[:, h, :]
            ah = ours_actual[:, h, :]
            ih = igs_pred_m[:, h, :]
            lh = igs_label_m[:, h, :]
            for scope_name, msk in [("all_candidates", joint_mask[:, h, :]),
                                    ("affected_only", joint_aff[:, h, :]),
                                    ("unaffected_only", joint_unaff[:, h, :])]:
                pix = int(msk.sum())
                if pix == 0:
                    continue
                horizon_rows.append({
                    "region": region_name,
                    "scope": scope_name,
                    "minutes_ahead": (h + 1) * 5,
                    "horizon_step": h + 1,
                    "matched_samples": m,
                    "pixels": pix,
                    "MAE_distnet": masked_mae(ph, ah, msk),
                    "MAE_igstgnn": masked_mae(ih, lh, msk),
                })

    summary_df = pd.DataFrame(summary_rows)
    horizon_df = pd.DataFrame(horizon_rows)
    summary_path = out_dir / "headtohead_summary.csv"
    horizon_path = out_dir / "headtohead_per_horizon.csv"
    summary_df.to_csv(summary_path, index=False)
    horizon_df.to_csv(horizon_path, index=False)
    print(f"\nsaved:\n  {summary_path}\n  {horizon_path}")

    # Pretty-print summary
    if not summary_df.empty:
        print("\n=== Per-region aggregate (12-horizon mean) ===")
        for region in summary_df["region"].unique():
            sub = summary_df[summary_df["region"] == region]
            print(f"\n{region}:")
            for _, row in sub.iterrows():
                print(f"  {row['scope']:18s}  DIST={row['MAE_distnet']:7.3f}  "
                      f"IGS={row['MAE_igstgnn']:7.3f}  Δ={row['delta_distnet_minus_igs']:+.3f}  "
                      f"({row['pixels']:,} pixels, {row['matched_samples']} samples)")


if __name__ == "__main__":
    main()
