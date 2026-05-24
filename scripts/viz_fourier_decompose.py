import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def rfft_split(x, k_low):
    """x: (T,). Returns (low_recon, high_recon, freq, mag)."""
    T = x.shape[0]
    F = np.fft.rfft(x)
    mag = np.abs(F)
    freq = np.fft.rfftfreq(T, d=1.0)

    F_low = F.copy()
    F_low[k_low:] = 0
    low = np.fft.irfft(F_low, n=T)

    F_high = F.copy()
    F_high[:k_low] = 0
    high = np.fft.irfft(F_high, n=T)
    return low, high, freq, mag


def pick_samples(samples_h5, n_affected, n_normal, seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(samples_h5, "r") as f:
        split = f["split"][:]
        n_active = f["n_active_incidents"][:]
        aff_mask = f["affected_mask"][:]
        sample_start = f["sample_start"][:]

    test_idx = np.where(split == 2)[0]
    aff_count = aff_mask.sum(axis=1)

    aff_pool = test_idx[aff_count[test_idx] >= 5]
    nor_pool = test_idx[aff_count[test_idx] == 0]

    aff_pick = rng.choice(aff_pool, size=n_affected, replace=False)
    nor_pick = rng.choice(nor_pool, size=n_normal, replace=False)
    return aff_pick, nor_pick, aff_mask, sample_start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--out_dir", default="outputs/fourier_test")
    p.add_argument("--channel", type=int, default=0, help="0=flow, 1=occ, 2=speed (likely)")
    p.add_argument("--n_affected", type=int, default=3)
    p.add_argument("--n_normal", type=int, default=3)
    p.add_argument("--k_list", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_h5 = data_dir / f"{args.region}_samples.h5"
    traffic_h5 = data_dir / f"{args.region}_traffic.h5"

    aff_pick, nor_pick, aff_mask, sample_start = pick_samples(
        samples_h5, args.n_affected, args.n_normal, seed=args.seed
    )

    with h5py.File(traffic_h5, "r") as f:
        flow = f["flow_series_imputed"][:]
    with h5py.File(samples_h5, "r") as f:
        T_h = int(f.attrs["T_h"])
        T_p = int(f.attrs["T_p"])

    print(f"flow shape: {flow.shape}  T_h={T_h}  T_p={T_p}")
    print(f"affected samples: {aff_pick}")
    print(f"normal samples:   {nor_pick}")

    # ---- panel 1: affected vs normal full window (history + future) ----
    fig, axes = plt.subplots(
        args.n_affected + args.n_normal, 4, figsize=(16, 2.6 * (args.n_affected + args.n_normal))
    )
    if axes.ndim == 1:
        axes = axes[None, :]

    K_main = args.k_list[len(args.k_list) // 2]  # middle K for main viz
    for row, idx in enumerate(list(aff_pick) + list(nor_pick)):
        is_aff = row < args.n_affected
        s0 = int(sample_start[idx])
        # pick a node: for affected, pick one affected node; for normal, pick a random one
        aff_nodes = np.where(aff_mask[idx])[0]
        if is_aff and aff_nodes.size > 0:
            node = int(aff_nodes[0])
        else:
            node = 100  # arbitrary normal node

        # full T_h+T_p window for context
        x_full = flow[s0:s0 + T_h + T_p, node, args.channel]  # (24,)
        x_hist = x_full[:T_h]
        x_fut = x_full[T_h:]

        low, high, freq, mag = rfft_split(x_hist, K_main)

        # col 0: full signal
        ax = axes[row, 0]
        t_full = np.arange(T_h + T_p)
        ax.plot(t_full[:T_h], x_hist, "b-", label="history", lw=1.5)
        ax.plot(t_full[T_h:], x_fut, "g--", label="future", lw=1.5)
        ax.axvline(T_h - 0.5, color="gray", ls=":", alpha=0.5)
        ax.set_title(f"{'AFFECTED' if is_aff else 'NORMAL'} idx={idx} node={node}", fontsize=9)
        ax.set_ylabel("flow")
        if row == 0:
            ax.legend(fontsize=7)

        # col 1: low-freq reconstruction
        ax = axes[row, 1]
        ax.plot(x_hist, "b-", label="hist", lw=1.5, alpha=0.5)
        ax.plot(low, "r-", label=f"low (K={K_main})", lw=2)
        ax.set_title(f"low-freq → NormalBranch  (K={K_main})", fontsize=9)
        if row == 0:
            ax.legend(fontsize=7)

        # col 2: high-freq reconstruction (residual)
        ax = axes[row, 2]
        ax.plot(high, "purple", label="high freq", lw=2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"high-freq → IncidentBranch", fontsize=9)
        amp = np.abs(high).max()
        ax.set_ylabel(f"max|·|={amp:.1f}", fontsize=8)

        # col 3: magnitude spectrum
        ax = axes[row, 3]
        ax.bar(freq, mag, width=freq[1] * 0.6, color="teal")
        ax.axvline(freq[K_main - 1] + freq[1] * 0.5, color="red", ls="--",
                   label=f"K={K_main} cutoff")
        ax.set_xlabel("freq (1/step)")
        ax.set_title(f"|FFT|  (T_h={T_h})", fontsize=9)
        if row == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    main_path = out_dir / f"fourier_decompose_{args.region}_ch{args.channel}_K{K_main}.png"
    plt.savefig(main_path, dpi=130)
    print(f"saved: {main_path}")
    plt.close()

    # ---- panel 2: K sweep on one affected sample ----
    if len(aff_pick) > 0:
        idx = int(aff_pick[0])
        s0 = int(sample_start[idx])
        aff_nodes = np.where(aff_mask[idx])[0]
        node = int(aff_nodes[0]) if aff_nodes.size > 0 else 100
        x_hist = flow[s0:s0 + T_h, node, args.channel]

        fig, axes = plt.subplots(1, len(args.k_list), figsize=(4.5 * len(args.k_list), 3.5))
        if len(args.k_list) == 1:
            axes = [axes]
        for ax, K in zip(axes, args.k_list):
            low, high, _, _ = rfft_split(x_hist, K)
            ax.plot(x_hist, "k-", label="orig", lw=1.5)
            ax.plot(low, "r-", label=f"low (K={K})", lw=2)
            ax.plot(high + x_hist.mean(), "purple", label="high+mean", lw=1.5, alpha=0.7)
            ax.set_title(f"K={K}  (keep {K}/{len(x_hist)//2+1} bins)", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlabel("step")
        plt.suptitle(f"K sweep — affected idx={idx} node={node}", fontsize=11)
        plt.tight_layout()
        ksweep_path = out_dir / f"fourier_Ksweep_{args.region}_ch{args.channel}.png"
        plt.savefig(ksweep_path, dpi=130)
        print(f"saved: {ksweep_path}")
        plt.close()

    # ---- panel 3: aggregate stats — energy in high-freq for affected vs normal ----
    n_agg = 200
    rng = np.random.default_rng(args.seed + 1)
    with h5py.File(samples_h5, "r") as f:
        split = f["split"][:]
        n_active = f["n_active_incidents"][:]
        aff_full = f["affected_mask"][:]
        starts = f["sample_start"][:]
        N = int(f.attrs["N"])
    test_idx = np.where(split == 2)[0]
    aff_count_arr = aff_full.sum(axis=1)
    aff_test = test_idx[aff_count_arr[test_idx] >= 5]
    nor_test = test_idx[aff_count_arr[test_idx] == 0]
    aff_samp = rng.choice(aff_test, size=min(n_agg, aff_test.size), replace=False)
    nor_samp = rng.choice(nor_test, size=min(n_agg, nor_test.size), replace=False)

    def hf_ratio(idx_set, only_aff_node=True):
        ratios = []
        for idx in idx_set:
            s0 = int(starts[idx])
            if only_aff_node and aff_full[idx].any():
                nodes = np.where(aff_full[idx])[0][:3]
            else:
                nodes = rng.choice(N, size=3, replace=False)
            for node in nodes:
                x = flow[s0:s0 + T_h, int(node), args.channel]
                F = np.fft.rfft(x)
                e_total = (np.abs(F) ** 2).sum()
                if e_total < 1e-9:
                    continue
                e_high = (np.abs(F[K_main:]) ** 2).sum()
                ratios.append(e_high / e_total)
        return np.array(ratios)

    r_aff = hf_ratio(aff_samp, only_aff_node=True)
    r_nor = hf_ratio(nor_samp, only_aff_node=False)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bins = np.linspace(0, max(r_aff.max(), r_nor.max()) * 1.02, 40)
    ax.hist(r_nor, bins=bins, alpha=0.5, label=f"normal (n={r_nor.size}, μ={r_nor.mean():.3f})",
            color="steelblue", density=True)
    ax.hist(r_aff, bins=bins, alpha=0.5, label=f"affected (n={r_aff.size}, μ={r_aff.mean():.3f})",
            color="firebrick", density=True)
    ax.axvline(r_nor.mean(), color="steelblue", ls="--", lw=1.5)
    ax.axvline(r_aff.mean(), color="firebrick", ls="--", lw=1.5)
    ax.set_xlabel(f"high-freq energy ratio (bins≥{K_main} / total)")
    ax.set_ylabel("density")
    ax.set_title(f"High-freq energy share — affected vs normal  (K={K_main}, T_h={T_h})")
    ax.legend()
    plt.tight_layout()
    hist_path = out_dir / f"fourier_energy_hist_{args.region}_ch{args.channel}_K{K_main}.png"
    plt.savefig(hist_path, dpi=130)
    print(f"saved: {hist_path}")
    plt.close()

    # ---- console summary ----
    print()
    print(f"=== high-freq energy ratio (K={K_main}, channel={args.channel}) ===")
    print(f"  normal  : mean={r_nor.mean():.4f}  median={np.median(r_nor):.4f}  std={r_nor.std():.4f}")
    print(f"  affected: mean={r_aff.mean():.4f}  median={np.median(r_aff):.4f}  std={r_aff.std():.4f}")
    print(f"  ratio aff/nor = {r_aff.mean() / max(r_nor.mean(), 1e-9):.2f}x")

    # Welch t-test-ish
    from scipy import stats
    t, pv = stats.ttest_ind(r_aff, r_nor, equal_var=False)
    print(f"  Welch t={t:.3f}  p={pv:.2e}")


if __name__ == "__main__":
    main()
