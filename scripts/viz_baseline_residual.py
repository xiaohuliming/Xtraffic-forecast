import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


SLOTS_PER_DAY = 288  # 5 min/step
DAYS_PER_WEEK = 7


def build_baseline(flow, train_end_step, channel):
    """flow: (T_total, N, C). Returns (7, 288, N) baseline means over train period only."""
    T_total, N, _ = flow.shape
    steps = np.arange(min(train_end_step, T_total))
    dow = (steps // SLOTS_PER_DAY) % DAYS_PER_WEEK
    slot = steps % SLOTS_PER_DAY

    base = np.zeros((DAYS_PER_WEEK, SLOTS_PER_DAY, N), dtype=np.float32)
    cnt = np.zeros((DAYS_PER_WEEK, SLOTS_PER_DAY, 1), dtype=np.float32)
    for d in range(DAYS_PER_WEEK):
        for s in range(SLOTS_PER_DAY):
            mask = (dow == d) & (slot == s)
            if mask.any():
                idx = steps[mask]
                base[d, s] = flow[idx, :, channel].mean(axis=0)
                cnt[d, s] = len(idx)
    return base, cnt


def lookup_baseline(base, start_step, T_h):
    """Return (T_h, N) baseline for given history window."""
    steps = start_step + np.arange(T_h)
    dow = (steps // SLOTS_PER_DAY) % DAYS_PER_WEEK
    slot = steps % SLOTS_PER_DAY
    return base[dow, slot]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--out_dir", default="outputs/fourier_test")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--n_agg", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_names = ["flow", "occupancy", "speed"]
    ch_name = ch_names[args.channel]

    with h5py.File(data_dir / f"{args.region}_samples.h5", "r") as f:
        split = f["split"][:]
        aff_mask = f["affected_mask"][:]
        starts = f["sample_start"][:]
        T_h = int(f.attrs["T_h"])
        T_p = int(f.attrs["T_p"])
        N = int(f.attrs["N"])
        train_end_frac = float(f.attrs["split_train_end"])
    with h5py.File(data_dir / f"{args.region}_traffic.h5", "r") as f:
        flow = f["flow_series_imputed"][:]

    T_total = flow.shape[0]
    train_end_step = int(T_total * train_end_frac)
    print(f"{args.region} ch={args.channel} ({ch_name})  T_total={T_total}  train_end={train_end_step}")
    print("building baseline ...")
    base, cnt = build_baseline(flow, train_end_step, args.channel)
    print(f"  baseline shape={base.shape}  min count per (dow, slot)={cnt.min():.0f}  max={cnt.max():.0f}")

    rng = np.random.default_rng(args.seed)
    test_idx = np.where(split == 2)[0]
    aff_count = aff_mask.sum(axis=1)
    aff_test = test_idx[aff_count[test_idx] >= 5]
    nor_test = test_idx[aff_count[test_idx] == 0]
    aff_samp = rng.choice(aff_test, size=min(args.n_agg, aff_test.size), replace=False)
    nor_samp = rng.choice(nor_test, size=min(args.n_agg, nor_test.size), replace=False)

    def collect_stats(idx_set, only_aff_node):
        """For each (sample, node), return (resid_energy_ratio, mean_drop_pct)."""
        ratios, drops = [], []
        for s_idx in idx_set:
            s0 = int(starts[s_idx])
            if only_aff_node and aff_mask[s_idx].any():
                nodes = np.where(aff_mask[s_idx])[0][:3]
            else:
                nodes = rng.choice(N, size=3, replace=False)
            obs_all = flow[s0:s0 + T_h, :, args.channel].astype(np.float32)  # (T_h, N)
            base_all = lookup_baseline(base, s0, T_h)  # (T_h, N)
            for nd in nodes:
                obs = obs_all[:, int(nd)]
                bs = base_all[:, int(nd)]
                if (obs**2).sum() < 1e-6 or (bs**2).sum() < 1e-6:
                    continue
                resid = obs - bs
                ratios.append((resid**2).sum() / (obs**2).sum())
                drops.append((obs.mean() - bs.mean()) / max(bs.mean(), 1e-6))
        return np.array(ratios), np.array(drops)

    print(f"collecting stats: aff n={len(aff_samp)}  nor n={len(nor_samp)} ...")
    r_aff, d_aff = collect_stats(aff_samp, only_aff_node=True)
    r_nor, d_nor = collect_stats(nor_samp, only_aff_node=False)

    print()
    print(f"=== residual energy ratio  ||obs-base||^2 / ||obs||^2 ===")
    print(f"  normal  : n={len(r_nor)}  mean={r_nor.mean():.4f}  median={np.median(r_nor):.4f}")
    print(f"  affected: n={len(r_aff)}  mean={r_aff.mean():.4f}  median={np.median(r_aff):.4f}")
    t, pv = stats.ttest_ind(r_aff, r_nor, equal_var=False)
    print(f"  ratio aff/nor = {r_aff.mean()/max(r_nor.mean(),1e-9):.2f}x   Welch t={t:.2f}  p={pv:.2e}")

    print()
    print(f"=== mean drop pct  (obs.mean - base.mean) / base.mean ===")
    print(f"  normal  : mean={d_nor.mean():+.4f}  median={np.median(d_nor):+.4f}")
    print(f"  affected: mean={d_aff.mean():+.4f}  median={np.median(d_aff):+.4f}")
    t2, pv2 = stats.ttest_ind(d_aff, d_nor, equal_var=False)
    print(f"  Welch t={t2:.2f}  p={pv2:.2e}")

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # row 1 col 1: histogram of residual energy ratio
    ax = axes[0, 0]
    maxv = max(np.percentile(r_aff, 99), np.percentile(r_nor, 99)) * 1.05
    bins = np.linspace(0, maxv, 50)
    ax.hist(r_nor, bins=bins, alpha=0.55, color="steelblue", density=True,
            label=f"normal  μ={r_nor.mean():.3f}  med={np.median(r_nor):.3f}  n={r_nor.size}")
    ax.hist(r_aff, bins=bins, alpha=0.55, color="firebrick", density=True,
            label=f"affected μ={r_aff.mean():.3f}  med={np.median(r_aff):.3f}  n={r_aff.size}")
    ax.axvline(r_nor.mean(), color="steelblue", ls="--", lw=1.5)
    ax.axvline(r_aff.mean(), color="firebrick", ls="--", lw=1.5)
    ax.set_xlabel("residual energy ratio  ||obs-base||² / ||obs||²")
    ax.set_ylabel("density")
    ax.set_title(f"{args.region} {ch_name}:  daily baseline residual energy share\n"
                 f"affected/normal = {r_aff.mean()/max(r_nor.mean(),1e-9):.2f}x   p={pv:.2e}",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # row 1 col 2: histogram of mean drop pct
    ax = axes[0, 1]
    rng_v = max(abs(np.percentile(d_aff, 1)), abs(np.percentile(d_aff, 99)),
                abs(np.percentile(d_nor, 1)), abs(np.percentile(d_nor, 99))) * 1.05
    bins = np.linspace(-rng_v, rng_v, 50)
    ax.hist(d_nor, bins=bins, alpha=0.55, color="steelblue", density=True,
            label=f"normal  μ={d_nor.mean():+.3f}")
    ax.hist(d_aff, bins=bins, alpha=0.55, color="firebrick", density=True,
            label=f"affected μ={d_aff.mean():+.3f}")
    ax.axvline(0, color="gray", lw=0.5)
    ax.axvline(d_nor.mean(), color="steelblue", ls="--", lw=1.5)
    ax.axvline(d_aff.mean(), color="firebrick", ls="--", lw=1.5)
    ax.set_xlabel("(obs.mean - base.mean) / base.mean   (negative = flow below baseline)")
    ax.set_ylabel("density")
    ax.set_title(f"mean drop vs baseline   p={pv2:.2e}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # rows 2: 2 example time-domain plots (1 affected, 1 normal)
    for col, idx_set, label, color in [(0, aff_samp, "AFFECTED", "firebrick"),
                                        (1, nor_samp, "NORMAL", "steelblue")]:
        ax = axes[1, col]
        # pick one example
        ex_idx = int(idx_set[0])
        s0 = int(starts[ex_idx])
        if label == "AFFECTED" and aff_mask[ex_idx].any():
            nd = int(np.where(aff_mask[ex_idx])[0][0])
        else:
            nd = 8
        obs = flow[s0:s0 + T_h, nd, args.channel]
        bs = lookup_baseline(base, s0, T_h)[:, nd]
        resid = obs - bs
        t = np.arange(T_h)
        ax.plot(t, obs, "o-", color=color, lw=2, label=f"obs  (sample {ex_idx} node {nd})")
        ax.plot(t, bs, "s--", color="green", lw=1.8, label="baseline (same dow, slot)")
        ax.bar(t, resid, color="purple", alpha=0.5, label=f"residual μ={resid.mean():.1f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("step (5 min/step)")
        ax.set_ylabel(ch_name)
        ax.set_title(f"{label} example  resid energy ratio={(resid**2).sum()/(obs**2).sum():.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Daily-baseline residual decomposition  ({args.region}, {ch_name}, K=3 not used)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = out_dir / f"baseline_residual_{args.region}_ch{args.channel}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
