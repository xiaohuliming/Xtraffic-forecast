#!/usr/bin/env python3
"""Plot per-horizon affected vs unaffected MAE for IGSTGNN across 3 regions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REGIONS = ["Alameda", "Contra_Costa", "Orange"]
DISPLAY = {"Alameda": "Alameda", "Contra_Costa": "Contra Costa", "Orange": "Orange"}
COLORS = {
    "Alameda": "#1f77b4",
    "Contra_Costa": "#ff7f0e",
    "Orange": "#2ca02c",
}

baseline_root = Path("/Users/xhlm/Desktop/Study/科研实习/baselines/IGSTGNN/experiments/igstgnn")
out_path = Path("/Users/xhlm/Desktop/Study/科研实习/outputs/paper_artifacts/figures/igstgnn_affected_per_horizon.png")
out_path.parent.mkdir(parents=True, exist_ok=True)


def load(region: str) -> dict:
    p = baseline_root / f"{region}_23" / "affected_breakdown.json"
    with open(p) as f:
        return json.load(f)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
ax_lines, ax_gap = axes

# Left: per-horizon MAE lines (solid = affected, dashed = unaffected)
for region in REGIONS:
    d = load(region)
    horizons = [r["horizon"] for r in d["by_horizon"]]
    aff = [r["MAE_affected"] for r in d["by_horizon"]]
    unaff = [r["MAE_unaffected"] for r in d["by_horizon"]]
    color = COLORS[region]
    ax_lines.plot(horizons, aff, "-o", color=color, label=f"{DISPLAY[region]} affected", markersize=5, linewidth=2)
    ax_lines.plot(horizons, unaff, "--", color=color, label=f"{DISPLAY[region]} unaffected", linewidth=1.4, alpha=0.7)

ax_lines.set_xlabel("Forecast horizon (5-min steps)")
ax_lines.set_ylabel("IGSTGNN test MAE (raw flow)")
ax_lines.set_title("Affected vs unaffected MAE per horizon")
ax_lines.set_xticks(range(1, 13))
ax_lines.legend(fontsize=8, loc="upper left", ncol=1)
ax_lines.grid(True, alpha=0.3)

# Right: relative gap (aff - unaff) / unaff per horizon
for region in REGIONS:
    d = load(region)
    horizons = [r["horizon"] for r in d["by_horizon"]]
    rel_gap = [
        (r["MAE_affected"] - r["MAE_unaffected"]) / r["MAE_unaffected"] * 100
        for r in d["by_horizon"]
    ]
    ax_gap.plot(horizons, rel_gap, "-o", color=COLORS[region], label=DISPLAY[region], markersize=5, linewidth=2)

ax_gap.set_xlabel("Forecast horizon (5-min steps)")
ax_gap.set_ylabel("Affected vs unaffected MAE gap (%)")
ax_gap.set_title("Affected-node degradation grows with horizon")
ax_gap.set_xticks(range(1, 13))
ax_gap.legend(fontsize=9, loc="lower right")
ax_gap.grid(True, alpha=0.3)
ax_gap.axhline(0, color="black", linewidth=0.5, alpha=0.4)

fig.suptitle("IGSTGNN (KDD'26) breakdown on incident-affected sensors",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"saved: {out_path}")
