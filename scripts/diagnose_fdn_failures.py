"""Diagnose FDN baseline failure modes across regions.

For each region (Alameda / ContraCosta / Orange), compare:
  - FDN learnable_K3 (current SOTA)
  - GraphWaveNet baseline

Breakdown:
  1) Per-horizon MAE (overall / affected / unaffected)
  2) Per-node MAE distribution — find the tail of worst nodes
  3) Affected-vs-unaffected gap per horizon
  4) FDN-minus-GWN improvement breakdown
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ["Alameda", "ContraCosta", "Orange"]
H = 12  # horizons


def load(path):
    d = np.load(path)
    sample_start = d["sample_start"] if "sample_start" in d.files else None
    return d["pred_raw_flow"], d["actual_future_flow"], d["y_mask_flow"], d["affected_mask"], sample_start


def per_horizon(pred, actual, mask, aff):
    S, T_p, N = pred.shape
    aff_TpN = np.broadcast_to(aff[:, None, :], (S, T_p, N))
    diff = np.abs(pred - actual)
    out = {"all": [], "affected": [], "unaffected": []}
    for h in range(T_p):
        m_all = mask[:, h, :]
        out["all"].append(float(diff[:, h, :][m_all].mean()))
        m_aff = m_all & aff_TpN[:, h, :]
        m_un = m_all & ~aff_TpN[:, h, :]
        out["affected"].append(float(diff[:, h, :][m_aff].mean()) if m_aff.any() else float("nan"))
        out["unaffected"].append(float(diff[:, h, :][m_un].mean()) if m_un.any() else float("nan"))
    return out


def per_node_mae(pred, actual, mask):
    # Average MAE per node, averaged over (samples, horizons) where mask is true
    # Use safe division: skip nodes with no valid observations
    diff = np.abs(pred - actual) * mask        # zeroed where mask=False
    sum_diff = diff.sum(axis=(0, 1))           # (N,)
    cnt = mask.sum(axis=(0, 1)).astype(np.float64)  # (N,)
    out = np.full_like(sum_diff, np.nan, dtype=np.float64)
    valid = cnt > 0
    out[valid] = sum_diff[valid] / cnt[valid]
    return out                                  # (N,)


def per_tod_mae(pred, actual, mask, sample_start, slot_per_day=288):
    """Bucket by time-of-day of the FIRST predicted step (sample_start+1).
    Returns dict horizon-of-day-band -> MAE."""
    # sample_start is the last hist step's slot; first pred step is at sample_start+1
    tod = (sample_start + 1) % slot_per_day                    # (S,)
    bands = {
        "night_22-06":   ((tod >= 22 * 12) | (tod < 6 * 12)),  # 264-288 + 0-72
        "amrush_06-10":  ((tod >= 6 * 12) & (tod < 10 * 12)),
        "midday_10-15":  ((tod >= 10 * 12) & (tod < 15 * 12)),
        "pmrush_15-19":  ((tod >= 15 * 12) & (tod < 19 * 12)),
        "evening_19-22": ((tod >= 19 * 12) & (tod < 22 * 12)),
    }
    diff = np.abs(pred - actual)
    out = {}
    for name, sel in bands.items():
        if sel.sum() == 0:
            out[name] = (float("nan"), 0)
            continue
        sub_d = diff[sel]
        sub_m = mask[sel]
        out[name] = (float(sub_d[sub_m].mean()), int(sel.sum()))
    return out


def main():
    Path("outputs/diagnostics").mkdir(parents=True, exist_ok=True)
    rows = []
    horizon_rows = []
    node_summary = []
    for region in REGIONS:
        fdn_path = Path(f"outputs/fourier_dual_net/learnable_K3/{region}/test_predictions.npz")
        gwn_path = Path(f"outputs/baselines/graphwavenet/{region}/test_predictions.npz")
        if not fdn_path.exists() or not gwn_path.exists():
            print(f"skip {region} (missing files)")
            continue

        p_fdn, y_fdn, m_fdn, aff_fdn, ss_fdn = load(fdn_path)
        p_gwn, y_gwn, m_gwn, aff_gwn, _ = load(gwn_path)
        S, T_p, N = p_fdn.shape

        # Sanity: same shapes, same masks expected
        assert p_fdn.shape == p_gwn.shape, f"shape mismatch {region}: {p_fdn.shape} vs {p_gwn.shape}"

        # ----- Per-horizon -----
        h_fdn = per_horizon(p_fdn, y_fdn, m_fdn, aff_fdn)
        h_gwn = per_horizon(p_gwn, y_gwn, m_gwn, aff_gwn)
        for h in range(T_p):
            horizon_rows.append({
                "region": region, "h_step": h + 1, "h_min": (h + 1) * 5,
                "FDN_all": h_fdn["all"][h], "FDN_aff": h_fdn["affected"][h], "FDN_un": h_fdn["unaffected"][h],
                "GWN_all": h_gwn["all"][h], "GWN_aff": h_gwn["affected"][h], "GWN_un": h_gwn["unaffected"][h],
                "delta_all": h_fdn["all"][h] - h_gwn["all"][h],
                "delta_aff": h_fdn["affected"][h] - h_gwn["affected"][h],
                "delta_un": h_fdn["unaffected"][h] - h_gwn["unaffected"][h],
            })

        # ----- Per-node MAE distribution -----
        node_fdn = per_node_mae(p_fdn, y_fdn, m_fdn)   # (N,)
        node_gwn = per_node_mae(p_gwn, y_gwn, m_gwn)
        node_delta = node_fdn - node_gwn               # negative = FDN better

        # affected node frequency: fraction of samples where this node is flagged affected
        aff_freq = aff_fdn.mean(axis=0)                # (N,)

        # quantiles of node-level FDN MAE (drop NaN nodes)
        node_fdn_valid = node_fdn[~np.isnan(node_fdn)]
        node_delta_valid = node_delta[~np.isnan(node_delta)]
        aff_freq_valid = aff_freq[~np.isnan(node_fdn)]
        node_fdn_for_corr = node_fdn_valid
        qs = np.quantile(node_fdn_valid, [0.5, 0.75, 0.9, 0.95, 0.99])
        node_summary.append({
            "region": region, "N": N, "S": S, "N_valid": int(len(node_fdn_valid)),
            "node_MAE_median": float(qs[0]),
            "node_MAE_p75": float(qs[1]),
            "node_MAE_p90": float(qs[2]),
            "node_MAE_p95": float(qs[3]),
            "node_MAE_p99": float(qs[4]),
            "node_MAE_max": float(node_fdn_valid.max()),
            "frac_FDN_worse_than_GWN": float((node_delta_valid > 0).mean()),
            "frac_FDN_better_by_1MAE": float((node_delta_valid < -1.0).mean()),
            "frac_FDN_worse_by_1MAE": float((node_delta_valid > 1.0).mean()),
            "corr_aff_freq_vs_node_MAE": float(np.corrcoef(aff_freq_valid, node_fdn_for_corr)[0, 1]),
        })

        # ----- Time-of-day breakdown -----
        if ss_fdn is not None:
            tod_fdn = per_tod_mae(p_fdn, y_fdn, m_fdn, ss_fdn)
            tod_gwn = per_tod_mae(p_gwn, y_gwn, m_gwn, ss_fdn)
            print(f"\n[{region}] Time-of-day MAE (first pred step's hour bucket):")
            print(f"  {'band':18s} {'n_samples':>10s}  {'FDN':>8s}  {'GWN':>8s}  {'delta':>8s}")
            for band in tod_fdn:
                fdn_v, n = tod_fdn[band]
                gwn_v, _ = tod_gwn[band]
                print(f"  {band:18s} {n:>10d}  {fdn_v:>8.3f}  {gwn_v:>8.3f}  {fdn_v - gwn_v:>+8.3f}")

        # Save per-node arrays for later visualization if needed
        np.savez(
            f"outputs/diagnostics/per_node_{region}.npz",
            node_fdn=node_fdn, node_gwn=node_gwn,
            node_delta=node_delta, aff_freq=aff_freq,
        )

    hdf = pd.DataFrame(horizon_rows)
    ndf = pd.DataFrame(node_summary)
    hdf.to_csv("outputs/diagnostics/per_horizon.csv", index=False, float_format="%.3f")
    ndf.to_csv("outputs/diagnostics/per_node_summary.csv", index=False, float_format="%.3f")

    # ----- Print key findings -----
    print("=" * 80)
    print("Per-horizon MAE (FDN_learnable_K3 vs GraphWaveNet)")
    print("=" * 80)
    for region in REGIONS:
        sub = hdf[hdf["region"] == region]
        if sub.empty:
            continue
        print(f"\n[{region}]")
        print(sub[["h_min", "FDN_all", "GWN_all", "delta_all",
                   "FDN_aff", "GWN_aff", "delta_aff"]].to_string(index=False))
        # Where does FDN degrade fastest as horizon grows?
        slope_fdn = sub["FDN_all"].iloc[-1] - sub["FDN_all"].iloc[0]
        slope_gwn = sub["GWN_all"].iloc[-1] - sub["GWN_all"].iloc[0]
        print(f"  growth: FDN h1→h12 = +{slope_fdn:.2f} MAE | GWN = +{slope_gwn:.2f} MAE")

    print("\n" + "=" * 80)
    print("Per-node distribution")
    print("=" * 80)
    print(ndf.to_string(index=False))


if __name__ == "__main__":
    main()
