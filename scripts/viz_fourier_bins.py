import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def reconstruct_single_bin(F, k, T):
    F_k = np.zeros_like(F)
    F_k[k] = F[k]
    return np.fft.irfft(F_k, n=T)


def plot_decomp(x, ax_grid, title, color_bin0=("steelblue", "firebrick")):
    T = x.shape[0]
    F = np.fft.rfft(x)
    n_bins = F.shape[0]
    mag = np.abs(F)
    phase = np.angle(F)

    bin_signals = [reconstruct_single_bin(F, k, T) for k in range(n_bins)]
    low_sum = sum(bin_signals[k] for k in range(3))
    high_sum = sum(bin_signals[k] for k in range(3, n_bins))
    full_sum = sum(bin_signals)

    t = np.arange(T)

    # Row 1: original signal
    ax = ax_grid[0]
    ax.plot(t, x, "o-", color=color_bin0[1], lw=2, label="original x[t]")
    ax.plot(t, full_sum, "x--", color="gray", lw=1, label="sum of all 7 bins")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Rows 2..8: each bin alone
    period_label = {0: "DC (const)", 1: "1 cyc/h (60 min)", 2: "2 cyc/h (30 min)",
                    3: "3 cyc/h (20 min)", 4: "4 cyc/h (15 min)",
                    5: "5 cyc/h (12 min)", 6: "6 cyc/h (10 min, Nyquist)"}
    for k in range(n_bins):
        ax = ax_grid[k + 1]
        is_low = k < 3
        color = "steelblue" if is_low else "purple"
        ax.plot(t, bin_signals[k], "o-", color=color, lw=1.8)
        ax.axhline(0, color="gray", lw=0.5)
        amp = (np.abs(bin_signals[k]).max())
        ax.set_ylabel(f"bin{k}\nA={mag[k]:.1f}\nϕ={np.degrees(phase[k]):.0f}°",
                      fontsize=8, rotation=0, labelpad=30, va="center")
        tag = "low → Normal" if is_low else "high → Residual"
        ax.set_title(f"{period_label[k]}   [{tag}]   max|·|={amp:.2f}",
                     fontsize=8.5, loc="left")
        ax.grid(alpha=0.3)

    # Row 9: low+high reconstruction
    ax = ax_grid[-1]
    ax.plot(t, x, "k-", lw=1, alpha=0.4, label="original")
    ax.plot(t, low_sum, "-", color="steelblue", lw=2.2, label="low sum (bin 0-2) → Normal branch")
    ax.plot(t, high_sum, "-", color="purple", lw=2.2, label="high sum (bin 3-6) → Residual branch")
    ax.plot(t, low_sum + high_sum, "x--", color="gray", lw=0.8, label="low+high check")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Reconstruction split (K=3 cutoff)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlabel("step (5 min/step)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--out_dir", default="outputs/fourier_test")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--aff_idx", type=int, default=57489)
    p.add_argument("--aff_node", type=int, default=8)
    p.add_argument("--nor_idx", type=int, default=-1, help="-1 = auto pick from test split")
    p.add_argument("--nor_node", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_dir / f"{args.region}_samples.h5", "r") as f:
        split = f["split"][:]
        aff_mask = f["affected_mask"][:]
        sample_start = f["sample_start"][:]
        T_h = int(f.attrs["T_h"])
        T_p = int(f.attrs["T_p"])
    with h5py.File(data_dir / f"{args.region}_traffic.h5", "r") as f:
        flow = f["flow_series_imputed"][:]

    if args.nor_idx < 0:
        rng = np.random.default_rng(args.seed)
        test_idx = np.where(split == 2)[0]
        aff_count = aff_mask.sum(axis=1)
        nor_pool = test_idx[aff_count[test_idx] == 0]
        args.nor_idx = int(rng.choice(nor_pool))

    s_aff = int(sample_start[args.aff_idx])
    s_nor = int(sample_start[args.nor_idx])
    x_aff = flow[s_aff:s_aff + T_h, args.aff_node, args.channel].astype(float)
    x_nor = flow[s_nor:s_nor + T_h, args.nor_node, args.channel].astype(float)

    ch_names = ["flow", "occupancy", "speed"]
    ch_name = ch_names[args.channel]

    n_rows = 1 + 7 + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 1.5 * n_rows + 1),
                             sharex=True)
    plot_decomp(
        x_aff, axes[:, 0],
        title=f"AFFECTED  idx={args.aff_idx} node={args.aff_node}  ({args.region}, {ch_name})",
    )
    plot_decomp(
        x_nor, axes[:, 1],
        title=f"NORMAL  idx={args.nor_idx} node={args.nor_node}  ({args.region}, {ch_name})",
    )

    plt.suptitle(
        f"7 bin signals (time-domain) + sum reconstruction + low/high split\n"
        f"(T_h={T_h} steps, 5 min/step = 1 hour; A=amplitude, phi=phase)",
        fontsize=12, y=0.998,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.985))

    out = out_dir / f"fourier_bins_{args.region}_ch{args.channel}_aff{args.aff_idx}_nor{args.nor_idx}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")
    print(f"  affected: idx={args.aff_idx} node={args.aff_node}  values={x_aff.round(1).tolist()}")
    print(f"  normal  : idx={args.nor_idx} node={args.nor_node}  values={x_nor.round(1).tolist()}")


if __name__ == "__main__":
    main()
