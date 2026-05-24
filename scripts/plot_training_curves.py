#!/usr/bin/env python3
"""Plot training curves from a DIST-Net run.

Reads outputs/dist_net/runs/<run>/log.jsonl and summary.json, produces a
multi-panel PNG figure with:
  1. Per-batch train losses (L_main / L_normal / L_incident, smoothed)
  2. Per-epoch val losses
  3. Mode-collapse monitors (branch_cosine, os_z_normal, os_z_incident)
  4. Gate dynamics (gate.mean, gate.aff_over_un)
  5. Delta dynamics + LR
  6. Multi-scale α weights (per branch)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np


def smooth(x: np.ndarray, k: int = 25) -> np.ndarray:
    if len(x) < 2 * k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Defaults to <run_dir>/training_curves.png")
    p.add_argument("--smooth", type=int, default=25, help="Smoothing window for train curves")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.run_dir / "log.jsonl"
    summary_path = args.run_dir / "summary.json"
    out_path = args.out or (args.run_dir / "training_curves.png")

    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"loaded {len(records)} log entries from {log_path}")
    summary = json.loads(summary_path.read_text())

    steps = np.array([r["step"] for r in records])
    epochs = np.array([r["epoch"] for r in records])

    def col(key, default=np.nan):
        return np.array([r.get(key, default) for r in records], dtype=float)

    L_main = col("L_main")
    L_normal = col("L_normal")
    L_incident = col("L_incident")

    branch_cos = col("monitor.branch_cosine")
    os_zn = col("monitor.os_z_normal")
    os_zi = col("monitor.os_z_incident")

    g_mean = col("gate.mean")
    g_aff = col("gate.affected_mean")
    g_un = col("gate.unaffected_mean")
    g_ratio = col("gate.aff_over_un")

    delta_mag = col("delta_pred.abs_mean")
    lr = col("lr")

    a_n_long = col("alpha.normal.long")
    a_n_short = col("alpha.normal.short")
    a_n_mid = col("alpha.normal.mid")
    a_i_long = col("alpha.incident.long")
    a_i_short = col("alpha.incident.short")
    a_i_mid = col("alpha.incident.mid")

    history = summary.get("history", [])
    val_epochs = [h["epoch"] for h in history if "val" in h]
    val_L_main = [h["val"]["L_main"] for h in history if "val" in h]
    val_L_normal = [h["val"]["L_normal"] for h in history if "val" in h]
    val_L_incident = [h["val"]["L_incident"] for h in history if "val" in h]
    train_avg_main = [h["train_avg"]["L_main"] for h in history]
    train_avg_normal = [h["train_avg"]["L_normal"] for h in history]
    train_avg_incident = [h["train_avg"]["L_incident"] for h in history]

    k = args.smooth
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 1: train losses (smoothed)
    ax = axes[0, 0]
    ax.plot(steps[k - 1:] if k > 1 else steps, smooth(L_main, k), label="L_main", color="C0")
    ax.plot(steps[k - 1:] if k > 1 else steps, smooth(L_normal, k), label="L_normal", color="C1")
    ax.plot(steps[k - 1:] if k > 1 else steps, smooth(L_incident, k), label="L_incident", color="C2")
    ax.set_title(f"Train losses (smooth window={k})")
    ax.set_xlabel("step"); ax.set_ylabel("MAE")
    ax.legend(); ax.grid(alpha=0.3)

    # 2: per-epoch train/val
    ax = axes[0, 1]
    eps = np.arange(1, len(train_avg_main) + 1)
    ax.plot(eps, train_avg_main, "o-", label="train L_main", color="C0", alpha=0.5)
    ax.plot(val_epochs, val_L_main, "s-", label="val L_main", color="C0")
    ax.plot(eps, train_avg_incident, "o-", label="train L_incident", color="C2", alpha=0.5)
    ax.plot(val_epochs, val_L_incident, "s-", label="val L_incident", color="C2")
    ax.set_title("Per-epoch losses")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3: mode-collapse monitors
    ax = axes[1, 0]
    ax.plot(steps, branch_cos, label="branch_cosine (healthy <0.6)", color="C3", linewidth=1)
    ax.plot(steps, os_zn, label="os_z_normal", color="C0", linewidth=1)
    ax.plot(steps, os_zi, label="os_z_incident", color="C2", linewidth=1)
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.4, label="alert >0.85")
    ax.set_title("Over-smoothing / branch divergence")
    ax.set_xlabel("step"); ax.set_ylabel("cosine sim")
    ax.set_ylim(-0.2, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4: gate dynamics
    ax = axes[1, 1]
    ax.plot(steps, g_mean, label="g.mean", color="C0")
    ax.plot(steps, g_aff, label="g.affected_mean", color="C3")
    ax.plot(steps, g_un, label="g.unaffected_mean", color="C9")
    ax.set_title("Gate dynamics  (aff/un ratio annotated)")
    ax.set_xlabel("step"); ax.set_ylabel("gate")
    ax.set_ylim(0, 1.05)
    ax2 = ax.twinx()
    ax2.plot(steps, g_ratio, "--", color="C4", alpha=0.7, label="gate.aff_over_un")
    ax2.set_ylabel("aff/un ratio")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 5: delta + LR
    ax = axes[2, 0]
    ax.plot(steps, delta_mag, label="delta_pred.abs_mean", color="C1")
    ax.set_xlabel("step"); ax.set_ylabel("|delta_pred|")
    ax.set_title("Delta magnitude + LR")
    ax2 = ax.twinx()
    ax2.plot(steps, lr, "--", color="C0", alpha=0.7, label="lr")
    ax2.set_ylabel("lr")
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # 6: multi-scale α weights
    ax = axes[2, 1]
    if np.any(np.isfinite(a_n_long)):
        ax.plot(steps, a_n_long, label="normal long", color="C0", linewidth=1)
        ax.plot(steps, a_n_mid, label="normal mid", color="C0", linestyle="--", linewidth=1)
        ax.plot(steps, a_n_short, label="normal short", color="C0", linestyle=":", linewidth=1)
        ax.plot(steps, a_i_long, label="incident long", color="C2", linewidth=1)
        ax.plot(steps, a_i_mid, label="incident mid", color="C2", linestyle="--", linewidth=1)
        ax.plot(steps, a_i_short, label="incident short", color="C2", linestyle=":", linewidth=1)
    ax.set_title("Multi-scale patching weights (α)")
    ax.set_xlabel("step"); ax.set_ylabel("α (softmax)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"DIST-Net training: {args.run_dir.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
