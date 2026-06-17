import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fourier_dual_net.deseason import lookup_baseline, train_residual_std


def test_lookup_baseline_picks_right_bins():
    N, C = 3, 2
    bm = np.arange(2 * 288 * N * C, dtype=np.float32).reshape(2, 288, N, C)
    T = 10
    day_kind = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    tod = np.arange(T, dtype=np.int64)
    out = lookup_baseline(bm, day_kind, tod, 2, 6)        # steps 2..5
    assert out.shape == (4, N, C)
    assert np.allclose(out[0], bm[1, 2])                  # step2: dk=1 tod=2
    assert np.allclose(out[3], bm[1, 5])                  # step5: dk=1 tod=5


def test_train_residual_std_train_only_and_masked():
    N, T, hi = 4, 20, 10
    bm = np.zeros((2, 288, N, 2), dtype=np.float32)       # baseline 0 -> res == flow
    day_kind = np.zeros(T, dtype=np.int64)
    tod = np.zeros(T, dtype=np.int64)
    flow = np.ones((T, N, 2), dtype=np.float32)
    mask = np.ones((T, N, 2), dtype=bool)
    flow[hi:] = 1e6                                       # post-train poison, must be ignored
    flow[0, 0, 0] = 1e6; mask[0, 0, 0] = False            # masked-out poison, must be ignored
    sd = train_residual_std(flow, mask, bm, day_kind, tod, hi, ch=0)
    assert sd < 1e-3, sd                                  # all valid train residuals == 1.0 -> std ~ 0
