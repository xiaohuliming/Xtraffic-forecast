#!/usr/bin/env python3
"""Local sanity test for the v2 target-disjoint full-window loader."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from dist_net.data import FullWindowRegionData

REGION = sys.argv[1] if len(sys.argv) > 1 else "ContraCosta"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def main():
    rd = FullWindowRegionData(REGION, "outputs/dist_net/region_data_v2_noleak",
                              "outputs/region_graphs", lazy=True,
                              stride=STRIDE, train_frac=0.7, val_frac=0.1)
    T, T_h, T_p, N = rd.T, rd.T_h, rd.T_p, rd.N
    starts = rd.sample_start
    n = len(starts)

    # 1. window range valid: every anchor has full history + future in-bounds
    assert starts.min() == T_h - 1, (starts.min(), T_h - 1)
    assert starts.max() <= T - T_p - 1, (starts.max(), T - T_p - 1)
    for code in (0, 1, 2):
        split_starts = starts[rd.split == code]
        assert (np.diff(split_starts) == STRIDE).all(), "non-uniform stride"

    # 2. chronological split and target sets are pairwise disjoint.
    tr = starts[rd.split == 0]; va = starts[rd.split == 1]; te = starts[rd.split == 2]
    assert tr.max() < va.min() < te.min(), "splits overlap in time"
    fr = (len(tr) / n, len(va) / n, len(te) / n)
    targets = {
        code: {
            target
            for anchor in starts[rd.split == code]
            for target in range(int(anchor) + 1, int(anchor) + T_p + 1)
        }
        for code in (0, 1, 2)
    }
    assert targets[0].isdisjoint(targets[1])
    assert targets[0].isdisjoint(targets[2])
    assert targets[1].isdisjoint(targets[2])

    # 3. event fields zeroed and sized to n
    assert rd.analysis_labels(0)["affected_mask"].shape == (N,)
    assert not rd.analysis_labels(0)["affected_mask"].any()
    assert rd.n_active_incidents.shape == (n,) and not rd.n_active_incidents.any()
    assert not rd.has_rel_feat

    # 4. get_sample shapes + x_hist matches a direct traffic slice
    idx = n // 2
    s = rd.get_sample(idx)
    assert s["x_hist"].shape == (N, T_h, 3), s["x_hist"].shape
    assert s["y_true"].shape == (N, T_p, 3), s["y_true"].shape
    assert s["time_feat"].shape == (T_h, 2)
    assert s["affected_mask"].shape == (N,) and not s["affected_mask"].any()
    assert int(s["sample_start"]) == int(starts[idx])

    t = int(starts[idx])
    import h5py
    with h5py.File(rd.traffic_path, "r") as f:
        raw_hist = f["flow_series_imputed"][t - T_h + 1: t + 1]   # (T_h, N, 3)
        raw_fut = f["flow_series_imputed"][t + 1: t + 1 + T_p]    # (T_p, N, 3)
    assert np.allclose(s["x_hist"], np.transpose(raw_hist, (1, 0, 2))), "x_hist mismatch"
    assert np.allclose(s["y_true"], np.transpose(raw_fut, (1, 0, 2))), "y_true mismatch"

    # 5. scaler/climatology fitting uses the explicit raw timeline boundary.
    assert rd.train_end == int(T * 0.7)
    train_stat_hi = rd.train_end

    print(f"[OK] {REGION} stride={STRIDE}: T={T} N={N} windows={n} "
          f"train/val/test={len(tr)}/{len(va)}/{len(te)} "
          f"fracs={fr[0]:.3f}/{fr[1]:.3f}/{fr[2]:.3f}")
    print(f"     anchor range [{starts.min()},{starts.max()}], "
          f"train_stat_hi={train_stat_hi} val_start={int(va.min())}")
    print("[OK] all assertions passed")


if __name__ == "__main__":
    main()
