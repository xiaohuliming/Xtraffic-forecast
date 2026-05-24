import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


def rfft_hf_ratio(x, K):
    F = np.fft.rfft(x)
    e_total = (np.abs(F) ** 2).sum()
    if e_total < 1e-9:
        return None
    return (np.abs(F[K:]) ** 2).sum() / e_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--n_agg", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    with h5py.File(data_dir / f"{args.region}_samples.h5", "r") as f:
        split = f["split"][:]
        aff_mask = f["affected_mask"][:]
        sample_start = f["sample_start"][:]
        T_h = int(f.attrs["T_h"])
        T_p = int(f.attrs["T_p"])
        N = int(f.attrs["N"])
    with h5py.File(data_dir / f"{args.region}_traffic.h5", "r") as f:
        flow = f["flow_series_imputed"][:]

    rng = np.random.default_rng(args.seed)
    test_idx = np.where(split == 2)[0]
    aff_count = aff_mask.sum(axis=1)
    aff_test = test_idx[aff_count[test_idx] >= 5]
    nor_test = test_idx[aff_count[test_idx] == 0]
    aff_samp = rng.choice(aff_test, size=min(args.n_agg, aff_test.size), replace=False)
    nor_samp = rng.choice(nor_test, size=min(args.n_agg, nor_test.size), replace=False)

    # Three window definitions
    windows = {
        "history (1h before)": (0, T_h),
        "future (1h forecast)": (T_h, T_h + T_p),
        "full (2h around incident)": (0, T_h + T_p),
    }

    print(f"\nRegion={args.region}  Channel={args.channel}  K={args.K}")
    print(f"affected sample N={len(aff_samp)}  normal sample N={len(nor_samp)}\n")
    print(f"{'Window':30s}  {'Norm μ':>10s}  {'Aff μ':>10s}  {'Ratio':>8s}  {'p-value':>10s}")
    print("-" * 78)

    for win_name, (a, b) in windows.items():
        r_aff = []
        r_nor = []
        for s_idx in aff_samp:
            s0 = int(sample_start[s_idx])
            nodes = np.where(aff_mask[s_idx])[0][:3]
            for nd in nodes:
                x = flow[s0 + a:s0 + b, int(nd), args.channel].astype(float)
                v = rfft_hf_ratio(x, args.K)
                if v is not None:
                    r_aff.append(v)
        for s_idx in nor_samp:
            s0 = int(sample_start[s_idx])
            nodes = rng.choice(N, size=3, replace=False)
            for nd in nodes:
                x = flow[s0 + a:s0 + b, int(nd), args.channel].astype(float)
                v = rfft_hf_ratio(x, args.K)
                if v is not None:
                    r_nor.append(v)
        r_aff = np.array(r_aff)
        r_nor = np.array(r_nor)
        t, pv = stats.ttest_ind(r_aff, r_nor, equal_var=False)
        ratio = r_aff.mean() / max(r_nor.mean(), 1e-9)
        print(f"{win_name:30s}  {r_nor.mean():10.5f}  {r_aff.mean():10.5f}  "
              f"{ratio:7.2f}x  {pv:10.2e}")


if __name__ == "__main__":
    main()
