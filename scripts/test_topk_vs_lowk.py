import h5py
import numpy as np
from collections import Counter


def main():
    with h5py.File("outputs/dist_net/region_data/Alameda_samples.h5", "r") as f:
        split = f["split"][:]
        aff_mask = f["affected_mask"][:]
        sample_start = f["sample_start"][:]
        T_h = int(f.attrs["T_h"])
        N = int(f.attrs["N"])
    with h5py.File("outputs/dist_net/region_data/Alameda_traffic.h5", "r") as f:
        flow = f["flow_series_imputed"][:]

    rng = np.random.default_rng(42)
    test_idx = np.where(split == 2)[0]
    aff_count = aff_mask.sum(axis=1)
    aff_test = test_idx[aff_count[test_idx] >= 5]
    nor_test = test_idx[aff_count[test_idx] == 0]

    aff_samp = rng.choice(aff_test, size=min(500, aff_test.size), replace=False)
    nor_samp = rng.choice(nor_test, size=min(500, nor_test.size), replace=False)

    K = 3
    n_bins = T_h // 2 + 1  # 7

    print(f"\nFor each sample, compare:\n"
          f"  low-K = bins (0, 1, ..., K-1)\n"
          f"  top-K = K bins with largest |F[k]|\n"
          f"K={K}, total bins={n_bins}\n")

    print("=" * 75)
    print("CHANNEL 0 (flow)")
    print("=" * 75)

    for ch in [0]:
        for label, samp_set, only_aff in [("AFFECTED + aff nodes", aff_samp, True),
                                          ("NORMAL + random nodes", nor_samp, False)]:
            topk_picks = []
            overlap_counts = []
            energy_ratios_lowk = []
            energy_ratios_topk = []

            for s_idx in samp_set:
                s0 = int(sample_start[s_idx])
                if only_aff and aff_mask[s_idx].any():
                    nodes = np.where(aff_mask[s_idx])[0][:3]
                else:
                    nodes = rng.choice(N, size=3, replace=False)
                for nd in nodes:
                    x = flow[s0:s0 + T_h, int(nd), ch].astype(float)
                    F = np.fft.rfft(x)
                    mag = np.abs(F)
                    total_e = (mag ** 2).sum()
                    if total_e < 1e-9:
                        continue

                    lowk_set = set(range(K))
                    topk_set = set(np.argsort(mag)[-K:].tolist())
                    topk_picks.append(tuple(sorted(topk_set)))
                    overlap_counts.append(len(lowk_set & topk_set))

                    e_lowk = (mag[list(lowk_set)] ** 2).sum() / total_e
                    e_topk = (mag[list(topk_set)] ** 2).sum() / total_e
                    energy_ratios_lowk.append(e_lowk)
                    energy_ratios_topk.append(e_topk)

            overlap_counts = np.array(overlap_counts)
            energy_ratios_lowk = np.array(energy_ratios_lowk)
            energy_ratios_topk = np.array(energy_ratios_topk)

            print(f"\n>>> {label}  (n={len(overlap_counts)})")
            print(f"  How often does top-K equal low-K?")
            for k_overlap in [3, 2, 1, 0]:
                pct = (overlap_counts == k_overlap).mean() * 100
                print(f"    overlap={k_overlap}/{K}: {pct:5.1f}%")

            print(f"  Energy captured by main branch:")
            print(f"    low-K bins:  mean={energy_ratios_lowk.mean():.4f}  med={np.median(energy_ratios_lowk):.4f}")
            print(f"    top-K bins:  mean={energy_ratios_topk.mean():.4f}  med={np.median(energy_ratios_topk):.4f}")
            print(f"  Residual energy (sent to perturbation branch):")
            print(f"    low-K residual:  mean={1 - energy_ratios_lowk.mean():.4f}")
            print(f"    top-K residual:  mean={1 - energy_ratios_topk.mean():.4f}")

            print(f"  Most common top-K bin sets:")
            for pat, cnt in Counter(topk_picks).most_common(5):
                print(f"    {pat}: {cnt} ({cnt/len(topk_picks)*100:.1f}%)")

    print("\n" + "=" * 75)
    print("KEY QUESTION: does residual differ between affected and normal?")
    print("=" * 75)

    for method in ["low-K", "top-K"]:
        r_aff, r_nor = [], []
        for label, samp_set, only_aff, out in [("aff", aff_samp, True, r_aff),
                                               ("nor", nor_samp, False, r_nor)]:
            for s_idx in samp_set:
                s0 = int(sample_start[s_idx])
                if only_aff and aff_mask[s_idx].any():
                    nodes = np.where(aff_mask[s_idx])[0][:3]
                else:
                    nodes = rng.choice(N, size=3, replace=False)
                for nd in nodes:
                    x = flow[s0:s0 + T_h, int(nd), 0].astype(float)
                    F = np.fft.rfft(x)
                    mag = np.abs(F)
                    total_e = (mag ** 2).sum()
                    if total_e < 1e-9:
                        continue
                    if method == "low-K":
                        keep = set(range(K))
                    else:
                        keep = set(np.argsort(mag)[-K:].tolist())
                    drop = set(range(len(mag))) - keep
                    e_resid = (mag[list(drop)] ** 2).sum() / total_e
                    out.append(e_resid)
        r_aff = np.array(r_aff)
        r_nor = np.array(r_nor)
        from scipy import stats
        t, pv = stats.ttest_ind(r_aff, r_nor, equal_var=False)
        ratio = r_aff.mean() / max(r_nor.mean(), 1e-9)
        print(f"\n{method} residual energy (送 perturbation branch):")
        print(f"  normal:   {r_nor.mean():.5f}")
        print(f"  affected: {r_aff.mean():.5f}")
        print(f"  ratio aff/nor = {ratio:.2f}x  Welch t={t:.3f}  p={pv:.3e}")


if __name__ == "__main__":
    main()
