import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


SLOTS_PER_DAY = 288
DAYS_PER_WEEK = 7


def build_baseline(flow, train_end_step, channel):
    T_total, N, _ = flow.shape
    steps = np.arange(min(train_end_step, T_total))
    dow = (steps // SLOTS_PER_DAY) % DAYS_PER_WEEK
    slot = steps % SLOTS_PER_DAY
    base = np.zeros((DAYS_PER_WEEK, SLOTS_PER_DAY, N), dtype=np.float32)
    for d in range(DAYS_PER_WEEK):
        for s in range(SLOTS_PER_DAY):
            mask = (dow == d) & (slot == s)
            if mask.any():
                base[d, s] = flow[steps[mask], :, channel].mean(axis=0)
    return base


def lookup(base, start_step, T):
    steps = start_step + np.arange(T)
    return base[(steps // SLOTS_PER_DAY) % DAYS_PER_WEEK, steps % SLOTS_PER_DAY]


def resid_energy_ratio(obs, base_vec):
    if (obs ** 2).sum() < 1e-6:
        return None
    return ((obs - base_vec) ** 2).sum() / (obs ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--label_dir", default="outputs/impact_labels")
    p.add_argument("--out_dir", default="outputs/fourier_test")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--T_h", type=int, default=12, help="history window length in 5-min steps")
    p.add_argument("--max_per_def", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(Path(args.data_dir) / f"{args.region}_traffic.h5", "r") as f:
        flow = f["flow_series_imputed"][:]
    with h5py.File(Path(args.data_dir) / f"{args.region}_samples.h5", "r") as f:
        train_end_frac = float(f.attrs["split_train_end"])
    T_total = flow.shape[0]
    train_end = int(T_total * train_end_frac)
    print(f"{args.region} ch{args.channel}  building baseline ...")
    base = build_baseline(flow, train_end, args.channel)

    nodes_df = pd.read_csv(Path(args.label_dir) / args.region / "node_labels.csv")
    ev_df = pd.read_csv(
        Path(args.label_dir) / args.region / "event_labels.csv",
        usecols=["incident_id", "start_idx"],
    )
    df = nodes_df.merge(ev_df, on="incident_id", how="inner")
    df = df.dropna(subset=["any_z_peak"]).reset_index(drop=True)
    print(f"  loaded {len(df)} (incident, node) labels;  affected==1: {(df.affected==1).sum()}")

    # Restrict to test split period to avoid using train baselines on train events
    df = df[df.start_idx >= train_end].reset_index(drop=True)
    print(f"  in test period: {len(df)}")

    # Restrict to events where the history window fits inside data and the node is non-trivial
    df = df[(df.start_idx + args.T_h) <= T_total].reset_index(drop=True)

    # ============ 4 LABEL DEFINITIONS ============
    defs = {
        "orig (any_z_peak>=3)": df.any_z_peak >= 3.0,
        "S1 flow_only_z_auc>=2": (df.flow_z_auc >= 2.0),
        "S2 flow_peak>=0.5 + pm<1.5": (df.flow_peak >= 0.5) & (df.pm_dist < 1.5),
        "S3 flow_peak>=0.5 + pm<0.5": (df.flow_peak >= 0.5) & (df.pm_dist < 0.5),
    }

    rng = np.random.default_rng(args.seed)
    # Build a NORMAL reference once: pick (incident, node) pairs where ALL criteria say "not affected"
    nor_mask = (df.any_z_peak < 1.0) & (df.flow_z_auc.fillna(0) < 0.5) & (df.flow_peak.fillna(0) < 0.10)
    nor_pool = df[nor_mask].reset_index(drop=True)
    nor_pick = nor_pool.sample(n=min(args.max_per_def, len(nor_pool)), random_state=args.seed)
    print(f"  normal reference pool: {len(nor_pool)}  picked: {len(nor_pick)}")

    def collect(rows):
        ratios = []
        for _, r in rows.iterrows():
            s0 = int(r.start_idx)
            nd = int(r.region_node_idx)
            obs = flow[s0:s0 + args.T_h, nd, args.channel].astype(np.float32)
            bs = lookup(base, s0, args.T_h)[:, nd]
            v = resid_energy_ratio(obs, bs)
            if v is not None:
                ratios.append(v)
        return np.array(ratios)

    nor_ratios = collect(nor_pick)

    results = {}
    for name, mask in defs.items():
        pool = df[mask].reset_index(drop=True)
        if len(pool) == 0:
            print(f"  [{name}] pool empty, skipping")
            continue
        pick = pool.sample(n=min(args.max_per_def, len(pool)), random_state=args.seed)
        r = collect(pick)
        if r.size == 0:
            continue
        t, pv = stats.ttest_ind(r, nor_ratios, equal_var=False)
        results[name] = dict(ratios=r, n_pool=len(pool), n_used=len(r),
                             mean=r.mean(), median=float(np.median(r)),
                             t=t, p=pv,
                             ratio_vs_nor=r.mean() / max(nor_ratios.mean(), 1e-9))
        print(f"  [{name:36s}] pool={len(pool):6d}  used={len(r):4d}  "
              f"μ={r.mean():.4f}  med={np.median(r):.4f}  vs_nor={results[name]['ratio_vs_nor']:5.2f}x  p={pv:.2e}")

    print(f"  [{'NORMAL ref':36s}] used={len(nor_ratios):4d}  μ={nor_ratios.mean():.4f}  med={np.median(nor_ratios):.4f}")

    # ============ PLOT ============
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5), sharey=True)
    if len(results) == 1:
        axes = [axes]

    all_max = 1.0  # clip to [0, 1] for readability; long tail summarised in title

    for ax, (name, d) in zip(axes, results.items()):
        bins = np.linspace(0, all_max, 40)
        n_clip = (d["ratios"] > all_max).sum()
        n_clip_nor = (nor_ratios > all_max).sum()
        ax.hist(nor_ratios, bins=bins, alpha=0.55, color="steelblue", density=True,
                label=f"NORMAL μ={nor_ratios.mean():.3f}\nmed={np.median(nor_ratios):.3f}  n={nor_ratios.size}")
        ax.hist(d["ratios"], bins=bins, alpha=0.55, color="firebrick", density=True,
                label=f"AFFECTED μ={d['mean']:.3f}\nmed={d['median']:.3f}  n={d['n_used']}")
        ax.axvline(nor_ratios.mean(), color="steelblue", ls="--", lw=1.5)
        ax.axvline(d["mean"], color="firebrick", ls="--", lw=1.5)
        ax.set_title(
            f"{name}\n"
            f"μ_ratio={d['ratio_vs_nor']:.2f}x   median_ratio={d['median']/max(np.median(nor_ratios),1e-9):.1f}x   p={d['p']:.2e}\n"
            f"(clipped: {n_clip} aff + {n_clip_nor} nor > 1.0)",
            fontsize=9.5)
        ax.set_xlabel("resid energy ratio ||obs-base||²/||obs||²  (x clipped at 1.0)")
        ax.set_xlim(0, all_max)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("density")
    plt.suptitle(f"{args.region} flow channel: baseline-residual energy under different 'affected' definitions",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = Path(args.out_dir) / f"strict_affected_{args.region}_ch{args.channel}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
