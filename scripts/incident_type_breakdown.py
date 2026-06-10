"""Per-incident-type MAE breakdown on matched windows (3-way: IGSTGNN / FDN / GWN).

Each adapter test sample stores the anchor event's Type code (IGSTGNN type_mapping).
On the matched-window subset (same 12-step prediction window for all 3 models),
bucket windows by anchor incident type and compare masked MAE per bucket.

This answers: does incident-label information help on SEVERE events (1141 = injury
collision) even if it does not help on average?

Output: outputs/diagnostics/incident_type_breakdown.txt (+ csv)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REGIONS = ["Alameda", "ContraCosta", "Orange"]
IGS_DIR = {"Alameda": "Alameda", "ContraCosta": "Contra_Costa", "Orange": "Orange"}
TYPE_NAME = {0: "1141_injury", 1: "AnimalHazard", 2: "CarFire", 3: "Fire",
             4: "Hazard", 5: "NoInj_collision", 6: "Other", 7: "UnknInj_collision"}
# Coarse severity groups for stable counts
GROUP_OF = {0: "collision_injury", 5: "collision_noinj", 7: "collision_unkinj",
            4: "hazard", 1: "hazard", 2: "fire", 3: "fire", 6: "other"}


def masked_mae(pred, actual, mask):
    d = np.abs(pred - actual) * mask
    c = mask.sum()
    return float(d.sum() / max(c, 1)), int(c)


def main():
    rows = []
    lines = []

    def emit(s):
        print(s, flush=True)
        lines.append(s)

    for region in REGIONS:
        igs = np.load(f"outputs/igstgnn_ours_pipeline_fixed/{IGS_DIR[region]}/test_predictions.npz")
        igs_pred, igs_label = igs["preds"], igs["labels"]
        s_igs = igs_pred.shape[0]
        adapter = np.load(f"outputs/igstgnn_data_from_ours/{region}/incident_data_test.npy",
                          allow_pickle=True)
        t_idx = np.array([s["_t_idx"] for s in adapter[:s_igs]], dtype=np.int64) + 11
        types = np.array([int(s["event_features"]["Type"]) for s in adapter[:s_igs]], dtype=np.int64)

        fdn = np.load(f"outputs/fourier_dual_net/learnable_K3/{region}/test_predictions.npz")
        gwn = np.load(f"outputs/baselines/graphwavenet/{region}/test_predictions.npz")
        assert np.array_equal(fdn["sample_start"], gwn["sample_start"])
        ss = fdn["sample_start"]
        uniq, first = np.unique(ss, return_index=True)
        ss_to_row = dict(zip(uniq.tolist(), first.tolist()))
        matches = np.array([ss_to_row.get(int(t), -1) for t in t_idx], dtype=np.int64)
        ok = matches >= 0
        rows_f = matches[ok]
        types_m = types[ok]

        mask = fdn["y_mask_flow"][rows_f]
        aff = fdn["affected_mask"][rows_f]
        aff3 = np.broadcast_to(aff[:, None, :], mask.shape)

        models = {
            "IGSTGNN": (igs_pred[ok], igs_label[ok]),
            "FDN": (fdn["pred_raw_flow"][rows_f], fdn["actual_future_flow"][rows_f]),
            "GWN": (gwn["pred_raw_flow"][rows_f], gwn["actual_future_flow"][rows_f]),
        }

        emit(f"\n{'='*78}\n[{region}]  matched windows n={int(ok.sum())}\n{'='*78}")
        emit(f"{'type':18s} {'n':>5s} | {'IGS_all':>8s} {'FDN_all':>8s} {'GWN_all':>8s} "
             f"{'FDNvIGS':>8s} | {'IGS_aff':>8s} {'FDN_aff':>8s} {'FDNvIGS':>8s}")

        # per fine type, then coarse groups, then total
        buckets = [(f"T{t}_{TYPE_NAME[t]}", types_m == t) for t in sorted(set(types_m.tolist()))]
        groups = {}
        for t in set(types_m.tolist()):
            groups.setdefault(GROUP_OF[t], []).append(t)
        buckets += [(f"G_{g}", np.isin(types_m, ts)) for g, ts in sorted(groups.items())]
        buckets += [("ALL", np.ones_like(types_m, dtype=bool))]

        for name, sel in buckets:
            n = int(sel.sum())
            if n == 0:
                continue
            m_all = mask[sel]
            m_aff = mask[sel] & aff3[sel]
            vals = {}
            for mod, (p, a) in models.items():
                vals[f"{mod}_all"], _ = masked_mae(p[sel], a[sel], m_all)
                vals[f"{mod}_aff"], _ = masked_mae(p[sel], a[sel], m_aff)
            d_all = vals["FDN_all"] - vals["IGSTGNN_all"]
            d_aff = vals["FDN_aff"] - vals["IGSTGNN_aff"]
            emit(f"{name:18s} {n:>5d} | {vals['IGSTGNN_all']:8.3f} {vals['FDN_all']:8.3f} "
                 f"{vals['GWN_all']:8.3f} {d_all:+8.3f} | {vals['IGSTGNN_aff']:8.3f} "
                 f"{vals['FDN_aff']:8.3f} {d_aff:+8.3f}")
            rows.append({"region": region, "bucket": name, "n": n, **vals,
                         "FDN_minus_IGS_all": d_all, "FDN_minus_IGS_aff": d_aff})

    out = Path("outputs/diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    (out / "incident_type_breakdown.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "incident_type_breakdown.csv", index=False, float_format="%.3f")
    print(f"\nsaved: {out}/incident_type_breakdown.txt + .csv")


if __name__ == "__main__":
    main()
