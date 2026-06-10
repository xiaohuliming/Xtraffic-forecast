"""Paired significance tests on per-window MAE (no scipy needed).

Comparisons:
  1. Setting B (full test set): FDN learnable vs GraphWaveNet, per region.
  2. Setting A (matched windows): FDN learnable vs IGSTGNN (bug-fixed), per region.
  3. Alameda only: D3 gated fusion vs FDN baseline (full test set).

Unit of analysis: one prediction window (sample) -> masked MAE over (T_p x N) cells.
Tests on paired diffs d = MAE_A - MAE_B:
  - paired t (normal approx, n is large)
  - Wilcoxon signed-rank (normal approx, no tie correction; MAE floats rarely tie)
  - sign-flip permutation test on mean (20k resamples)

Output: outputs/diagnostics/significance_tests.txt
"""
from __future__ import annotations
from math import erfc, sqrt
from pathlib import Path
import numpy as np

REGIONS = ["Alameda", "ContraCosta", "Orange"]
IGS_DIR = {"Alameda": "Alameda", "ContraCosta": "Contra_Costa", "Orange": "Orange"}


def per_window_mae(pred, actual, mask):
    diff = np.abs(pred - actual) * mask
    cnt = mask.sum(axis=(1, 2)).astype(np.float64)
    s = diff.sum(axis=(1, 2))
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def paired_tests(a, b, n_perm=20000, seed=0):
    """a, b: per-window MAE for model A and B on the SAME windows. Negative diff = A better."""
    d = (a - b)
    d = d[np.isfinite(d)]
    n = len(d)
    mean_d = float(d.mean())
    med_d = float(np.median(d))
    frac_better = float((d < 0).mean())

    se = d.std(ddof=1) / sqrt(n)
    t = mean_d / se
    p_t = erfc(abs(t) / sqrt(2))

    dz = d[d != 0]
    nz = len(dz)
    ranks = np.argsort(np.argsort(np.abs(dz))) + 1.0
    w_pos = float(ranks[dz > 0].sum())
    mu = nz * (nz + 1) / 4.0
    sigma = sqrt(nz * (nz + 1) * (2 * nz + 1) / 24.0)
    z = (w_pos - mu) / sigma
    p_w = erfc(abs(z) / sqrt(2))

    rng = np.random.default_rng(seed)
    obs = abs(mean_d)
    d32 = d.astype(np.float32)
    count, chunk = 0, 2000
    for _ in range(n_perm // chunk):
        signs = (rng.integers(0, 2, size=(chunk, n)).astype(np.float32) * 2 - 1)
        pm = signs @ d32 / n
        count += int((np.abs(pm) >= obs).sum())
    p_perm = (count + 1) / (n_perm + 1)

    return {"n": n, "mean_diff": mean_d, "median_diff": med_d,
            "frac_A_better": frac_better, "t": t, "p_t": p_t,
            "z_wilcoxon": z, "p_wilcoxon": p_w, "p_perm": p_perm}


def fmt(r, label_a, label_b):
    sig = "SIGNIFICANT" if max(r["p_t"], r["p_wilcoxon"], r["p_perm"]) < 0.01 else (
        "significant@0.05" if max(r["p_t"], r["p_wilcoxon"], r["p_perm"]) < 0.05 else "NOT significant")
    return (f"  n={r['n']}  mean diff={r['mean_diff']:+.4f}  median={r['median_diff']:+.4f}  "
            f"{label_a} better on {r['frac_A_better']*100:.1f}% of windows\n"
            f"  paired-t p={r['p_t']:.2e}   wilcoxon p={r['p_wilcoxon']:.2e}   "
            f"perm p={r['p_perm']:.2e}   -> {sig}")


def main():
    out_lines = []

    def emit(s):
        print(s, flush=True)
        out_lines.append(s)

    emit("=" * 78)
    emit("1) Setting B (full test): FDN learnable vs GraphWaveNet  [diff = FDN - GWN]")
    emit("=" * 78)
    for region in REGIONS:
        fdn = np.load(f"outputs/fourier_dual_net/learnable_K3/{region}/test_predictions.npz")
        gwn = np.load(f"outputs/baselines/graphwavenet/{region}/test_predictions.npz")
        assert np.array_equal(fdn["sample_start"], gwn["sample_start"]), f"{region}: window mismatch"
        a = per_window_mae(fdn["pred_raw_flow"], fdn["actual_future_flow"], fdn["y_mask_flow"])
        b = per_window_mae(gwn["pred_raw_flow"], gwn["actual_future_flow"], gwn["y_mask_flow"])
        emit(f"\n[{region}]")
        emit(fmt(paired_tests(a, b), "FDN", "GWN"))

    emit("")
    emit("=" * 78)
    emit("2) Setting A (matched windows): FDN learnable vs IGSTGNN  [diff = FDN - IGS]")
    emit("=" * 78)
    for region in REGIONS:
        igs = np.load(f"outputs/igstgnn_ours_pipeline_fixed/{IGS_DIR[region]}/test_predictions.npz")
        igs_pred, igs_label = igs["preds"], igs["labels"]
        s_igs = igs_pred.shape[0]
        adapter = np.load(f"outputs/igstgnn_data_from_ours/{region}/incident_data_test.npy",
                          allow_pickle=True)
        t_idx = np.array([s["_t_idx"] for s in adapter[:s_igs]], dtype=np.int64) + 11

        fdn = np.load(f"outputs/fourier_dual_net/learnable_K3/{region}/test_predictions.npz")
        ss = fdn["sample_start"]
        uniq, first = np.unique(ss, return_index=True)
        ss_to_row = dict(zip(uniq.tolist(), first.tolist()))
        matches = np.array([ss_to_row.get(int(t), -1) for t in t_idx], dtype=np.int64)
        ok = matches >= 0
        rows = matches[ok]

        mask = fdn["y_mask_flow"][rows]
        igs_w = per_window_mae(igs_pred[ok], igs_label[ok], mask)
        fdn_w = per_window_mae(fdn["pred_raw_flow"][rows], fdn["actual_future_flow"][rows], mask)
        emit(f"\n[{region}]  matched {ok.sum()}/{s_igs}")
        emit(fmt(paired_tests(fdn_w, igs_w), "FDN", "IGS"))

    emit("")
    emit("=" * 78)
    emit("3) Alameda full test: D3 gated fusion vs FDN baseline  [diff = D3 - baseline]")
    emit("=" * 78)
    d3_p = Path("outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz")
    if d3_p.exists():
        d3 = np.load(d3_p)
        fdn = np.load("outputs/fourier_dual_net/learnable_K3/Alameda/test_predictions.npz")
        assert np.array_equal(d3["sample_start"], fdn["sample_start"])
        a = per_window_mae(d3["pred_raw_flow"], d3["actual_future_flow"], d3["y_mask_flow"])
        b = per_window_mae(fdn["pred_raw_flow"], fdn["actual_future_flow"], fdn["y_mask_flow"])
        emit("\n[Alameda]")
        emit(fmt(paired_tests(a, b), "D3", "baseline"))
    else:
        emit("D3 npz not found, skipped")

    out = Path("outputs/diagnostics/significance_tests.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
