"""Setting B comparison: FourierDualNet (no incident labels) vs GraphWaveNet baseline.

Both models trained + evaluated on the same data pipeline (h5 cache, event-anchored
samples). No comparison to IGSTGNN here — IGSTGNN's eval uses a different sample set
and requires Option D (retrain on our data) for fair comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mae_breakdown(npz_path: Path) -> dict:
    d = np.load(npz_path)
    pred, actual, mask, aff = (
        d["pred_raw_flow"], d["actual_future_flow"], d["y_mask_flow"], d["affected_mask"]
    )
    S, T_p, N = pred.shape
    diff = np.abs(pred - actual)
    aff_TpN = np.broadcast_to(aff[:, None, :], (S, T_p, N))
    return {
        "all": float(diff[mask].mean()),
        "affected": float(diff[mask & aff_TpN].mean()),
        "unaffected": float(diff[mask & ~aff_TpN].mean()),
        "S": int(S), "N": int(N),
    }


def main():
    regions = ["Alameda", "ContraCosta", "Orange"]
    rows = []

    for region in regions:
        gwn_p = Path(f"outputs/baselines/graphwavenet/{region}/test_predictions.npz")
        if gwn_p.exists():
            r = mae_breakdown(gwn_p); r["model"] = "GraphWaveNet"; r["region"] = region; r["mode"] = "-"; rows.append(r)

        for mode in ["fixed_k_K3", "learnable_K3"]:
            p = Path(f"outputs/fourier_dual_net/{mode}/{region}/test_predictions.npz")
            if p.exists():
                r = mae_breakdown(p); r["model"] = "FourierDualNet"; r["region"] = region; r["mode"] = mode; rows.append(r)

    # group by region, then print table + Δ vs GWN baseline
    print(f"\n{'='*92}")
    print(f"{'Setting B: no-incident-label models on our data pipeline':^92}")
    print(f"{'='*92}\n")
    print(f"{'Region':<13}{'Model':<18}{'Mode':<14}{'MAE all':<10}{'affected':<11}{'unaffected':<12}{'Δ all':<10}")
    print("-" * 92)

    for region in regions:
        region_rows = [r for r in rows if r["region"] == region]
        gwn_row = next((r for r in region_rows if r["model"] == "GraphWaveNet"), None)
        for r in region_rows:
            delta = ""
            if r["model"] == "FourierDualNet" and gwn_row:
                delta = f"{r['all'] - gwn_row['all']:+.3f}"
            print(f"{r['region']:<13}{r['model']:<18}{r['mode']:<14}"
                  f"{r['all']:<10.3f}{r['affected']:<11.3f}{r['unaffected']:<12.3f}{delta:<10}")
        print()

    # save JSON for later use
    out = Path("outputs/fourier_dual_net/setting_b_comparison.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved JSON: {out}")


if __name__ == "__main__":
    main()
