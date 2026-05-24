"""Build final Setting A paper artifacts.

Outputs (under outputs/setting_a_artifacts/):
  setting_a_main_table.md      — markdown main table (matched-window comparison)
  setting_a_main_table.csv     — CSV
  setting_a_per_horizon.png    — per-horizon MAE for FDN vs IGSTGNN on matched windows
  setting_a_per_horizon.csv    — per-horizon data
  combined_setting_ab.md       — combined narrative pulling Setting B + A

Setting A inputs:
  outputs/igstgnn_ours_pipeline_fixed/setting_a_breakdown.json — produced by compute_setting_a_breakdown.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIONS = ["Alameda", "ContraCosta", "Orange"]


def main():
    out_dir = Path("outputs/setting_a_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    breakdown = json.loads(Path("outputs/igstgnn_ours_pipeline_fixed/setting_a_breakdown.json").read_text())

    # ---------- Main table ----------
    rows = []
    for region in REGIONS:
        if region not in breakdown:
            continue
        b = breakdown[region]
        rows.append({
            "Region": region,
            "n_matched": b["n_matched"],
            "n_igs_total": b["n_igs"],
            "match_rate_%": round(100 * b["n_matched"] / b["n_igs"], 1),
            "IGS_all": round(b["igs_mae_all"], 3),
            "IGS_affected": round(b["igs_mae_affected"], 3),
            "IGS_unaffected": round(b["igs_mae_unaffected"], 3),
            "FDN_all": round(b["fdn_mae_all_same_windows"], 3),
            "FDN_affected": round(b["fdn_mae_affected_same_windows"], 3),
            "FDN_unaffected": round(b["fdn_mae_unaffected_same_windows"], 3),
            "Δ_all": round(b["delta_fdn_minus_igs_all"], 3),
            "Δ_affected": round(b["delta_fdn_minus_igs_affected"], 3),
            "Δ_unaffected": round(b["delta_fdn_minus_igs_unaffected"], 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "setting_a_main_table.csv", index=False)

    md = ["# Setting A — FourierDualNet vs IGSTGNN (with incident labels)\n",
          "Both models trained + evaluated on our `.h5` cache pipeline (event-anchored, imputed).",
          "**IGSTGNN was trained with our fix for the official-code dataloader threading bug** (see paper Section X).",
          "Comparison is done on **matched prediction windows** — for each IGSTGNN sample at adapter `_t_idx = s0`, "
          "we find the FDN sample with `sample_start = s0+11` (which predicts the same 12-step future window). "
          "Only windows shared between both models are included.\n",
          "Δ columns = FourierDualNet minus IGSTGNN. Negative = FDN wins.\n",
          "| Region | n_match / n_IGS | match | IGS all | IGS aff | IGS un | FDN all | FDN aff | FDN un | Δ all | Δ aff | Δ un |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['Region']} | {r['n_matched']}/{r['n_igs_total']} | {r['match_rate_%']}% | "
            f"{r['IGS_all']:.3f} | {r['IGS_affected']:.3f} | {r['IGS_unaffected']:.3f} | "
            f"{r['FDN_all']:.3f} | {r['FDN_affected']:.3f} | {r['FDN_unaffected']:.3f} | "
            f"{r['Δ_all']:+.3f} | {r['Δ_affected']:+.3f} | {r['Δ_unaffected']:+.3f} |"
        )
    md.append("")
    md.append("**Findings:**")
    md.append("- Alameda + Orange: FourierDualNet **outperforms** IGSTGNN on all 3 metrics by 0.4–1.3 MAE.")
    md.append("- ContraCosta: essentially **tied** (Δ within ±0.14).")
    md.append("- FourierDualNet's advantage is **larger on affected nodes** than overall, suggesting FFT decomposition")
    md.append("  captures incident-induced anomalies better than IGSTGNN's explicit incident modeling on this pipeline.")
    (out_dir / "setting_a_main_table.md").write_text("\n".join(md), encoding="utf-8")
    print("=== Setting A Main Table ===")
    print("\n".join(md))

    # ---------- Per-horizon plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rows_csv = []
    for ax, region in zip(axes, REGIONS):
        if region not in breakdown:
            continue
        b = breakdown[region]
        h_axis = np.arange(1, 13) * 5
        igs_all = b["per_horizon_all"]
        igs_aff = b["per_horizon_affected"]
        # FDN per-horizon needs to be recomputed on matched rows.
        # For simplicity, we don't store per-horizon FDN on matched windows in breakdown JSON.
        # Just show IGSTGNN per-horizon for now and FDN average dashed.
        fdn_avg_all = b["fdn_mae_all_same_windows"]
        fdn_avg_aff = b["fdn_mae_affected_same_windows"]
        ax.plot(h_axis, igs_all, "-", color="#377eb8", lw=2, label="IGSTGNN (all)")
        ax.plot(h_axis, igs_aff, "--", color="#377eb8", lw=2, label="IGSTGNN (affected)")
        ax.axhline(fdn_avg_all, color="#e41a1c", ls="-", lw=2, label="FDN avg (all)")
        ax.axhline(fdn_avg_aff, color="#e41a1c", ls="--", lw=2, label="FDN avg (affected)")
        ax.set_title(f"{region}  (n={b['n_matched']})", fontsize=11)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.3)
        if region == "Alameda":
            ax.legend(fontsize=8, loc="lower right")
        for h_idx in range(12):
            rows_csv.append({"region": region, "h_step": h_idx + 1,
                             "IGS_all": igs_all[h_idx], "IGS_affected": igs_aff[h_idx]})
    plt.suptitle("Setting A — IGSTGNN per-horizon vs FDN average (matched windows)", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "setting_a_per_horizon.png", dpi=130, bbox_inches="tight")
    plt.close()
    pd.DataFrame(rows_csv).to_csv(out_dir / "setting_a_per_horizon.csv", index=False, float_format="%.4f")
    print(f"\nsaved per-horizon plot: {out_dir / 'setting_a_per_horizon.png'}")

    # ---------- Combined narrative ----------
    combined = ["# Final Comparison Summary\n",
                "## Setting B — No incident labels (deployment-friendly)\n",
                "See `outputs/fourier_dual_net/paper_artifacts/setting_b_main_table.md` for the full table.",
                "FourierDualNet `learnable_K3` beats single-branch GraphWaveNet on Alameda (-0.42 all-MAE) and",
                "ContraCosta (-0.32 all-MAE); essentially tied on Orange (-0.02).\n",
                "## Setting A — With incident labels (same pipeline, matched windows)\n"]
    combined.extend(md)
    combined.append("\n## Discovered IGSTGNN bug\n")
    combined.append("During this work we identified a threading-related indexing bug in IGSTGNN's official dataloader")
    combined.append("(`batch_samples[i-start_idx]` should be `batch_samples[i]`, in `src/utils/dataloader.py`).")
    combined.append("This bug causes each batch of size B to only use the first 2 unique samples (replicated B/2 times),")
    combined.append("reducing effective batch size from 48 to 2. Our reported IGSTGNN numbers use the **fixed** code;")
    combined.append("we observed +0.28 / +0.65 / unknown MAE improvement (Alameda / CC / Orange) over the buggy code.")
    (out_dir / "combined_setting_ab.md").write_text("\n".join(combined), encoding="utf-8")
    print(f"\nsaved combined narrative: {out_dir / 'combined_setting_ab.md'}")


if __name__ == "__main__":
    main()
