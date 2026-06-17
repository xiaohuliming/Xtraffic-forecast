"""Numpy-only de-seasonalization helpers for RGDN. No torch import so it runs locally."""
from __future__ import annotations

import numpy as np


def lookup_baseline(baseline_median: np.ndarray, day_kind: np.ndarray,
                    tod: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """baseline_median (2,288,N,C) -> (hi-lo, N, C) for time steps [lo, hi)."""
    return baseline_median[day_kind[lo:hi], tod[lo:hi]]


def train_residual_std(flow_series: np.ndarray, flow_mask: np.ndarray,
                       baseline_median: np.ndarray, day_kind: np.ndarray,
                       tod: np.ndarray, hi: int, ch: int = 0,
                       floor: float = 1e-6) -> float:
    """Masked std of (flow - baseline) over train steps [0, hi), channel ch."""
    flow = flow_series[:hi, :, ch]                                  # (hi, N)
    mask = flow_mask[:hi, :, ch].astype(bool)
    base = baseline_median[day_kind[:hi], tod[:hi]][:, :, ch]       # (hi, N)
    res = flow - base
    vals = res[mask]
    if vals.size == 0:
        return 1.0
    return float(vals.std() + floor)
