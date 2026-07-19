import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_full_county_cache import (
    CACHE_SCHEMA_VERSION,
    build_samples_h5,
    impute_flow,
    replace_negative_missing_inplace,
    require_causal_traffic_cache,
    split_code,
    validate_preprocessing_args,
)


def traffic_meta(region="Synthetic", n=2, t=100, frac=0.70):
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_id": "traffic-cache-id",
        "build_id": "traffic-build-id",
        "traffic_cache_id": "traffic-cache-id",
        "imputation_method": "causal_forward_fill",
        "baseline_train_frac": frac,
        "baseline_train_cutoff": int(t * frac),
        "candidate_pm_radius": 5.0,
        "anchor_pm_radius": 2.0,
        "baseline_mask_extra_steps": 12,
        "min_baseline_count": 8,
        "builder_source_sha256": "a" * 64,
        "region_name": region,
        "N": n,
        "T": t,
    }


def write_traffic_meta(path, **overrides):
    meta = traffic_meta(**{
        key: overrides.pop(key)
        for key in list(overrides)
        if key in {"region", "n", "t", "frac"}
    })
    meta.update(overrides)
    n = int(meta["N"])
    t = int(meta["T"])
    with h5py.File(path, "w") as f:
        for key, value in meta.items():
            f.attrs[key] = value
        f.create_dataset("flow_series_imputed", shape=(t, n, 3), dtype="f4")
        f.create_dataset("flow_mask", shape=(t, n, 3), dtype="?")
        f.create_dataset("time_enc", shape=(t, 5), dtype="f4")
        f.create_dataset("static_meta", shape=(n, 1), dtype="f4")
        f.create_dataset("baseline_median", shape=(2, 288, n, 3), dtype="f4")
        f.create_dataset("baseline_scale", shape=(2, 288, n, 3), dtype="f4")
        f.create_dataset("region_idx", shape=(n,), dtype="i8")
        f.create_dataset("times_ns", shape=(t,), dtype="i8")
    return meta


def test_impute_flow_is_causal_forward_fill():
    traffic = np.full((4, 2, 2), np.nan, dtype=np.float32)
    traffic[:, 0, 0] = [1.0, np.nan, np.nan, 9.0]
    traffic[:, 0, 1] = [np.nan, np.nan, 5.0, np.nan]
    traffic[:, 1, 1] = [2.0, -1.0, 4.0, np.nan]

    expected = np.array([
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 5.0], [0.0, 4.0]],
        [[9.0, 5.0], [0.0, 4.0]],
    ], dtype=np.float32)

    imputed, mask = impute_flow(traffic)

    assert imputed.dtype == np.float32
    assert np.array_equal(mask, np.isfinite(traffic) & (traffic >= 0.0))
    assert np.array_equal(imputed, expected)


def test_replace_negative_missing_inplace_preserves_nonnegative_values():
    traffic = np.array(
        [[[-1.0, 0.0], [np.nan, 2.0]], [[3.0, -9.0], [4.0, np.inf]]],
        dtype=np.float32,
    )
    assert replace_negative_missing_inplace(traffic, chunk_steps=1) == 2
    assert np.isnan(traffic[0, 0, 0])
    assert np.isnan(traffic[1, 0, 1])
    assert traffic[0, 0, 1] == 0.0
    assert traffic[1, 1, 0] == 4.0
    assert np.isnan(traffic[1, 1, 1])


def test_split_code_requires_complete_target_window():
    expected = {
        57: 0,     # targets 58..69: train
        58: None,  # targets 59..70: crosses train/val
        69: 1,     # targets 70..81: val
        70: 1,
        72: 1,     # targets 73..84: val
        73: None,  # targets 74..85: crosses val/test
        84: 2,     # targets 85..96: test
        85: 2,
        87: 2,     # targets 88..99: test
        88: None,  # targets 89..100: outside the series
    }

    for sample_start, want in expected.items():
        assert split_code(sample_start, 100, 0.70, 0.85, 12) == want


def test_preprocessing_guards_reject_leaky_inputs(tmp_path):
    validate_preprocessing_args(SimpleNamespace(
        baseline_train_frac=0.70, split_train_end=0.70, split_val_end=0.85,
    ))
    with pytest.raises(ValueError, match="baseline statistics"):
        validate_preprocessing_args(SimpleNamespace(
            baseline_train_frac=0.71, split_train_end=0.70,
            split_val_end=0.85,
        ))

    cache_path = tmp_path / "traffic.h5"
    with h5py.File(cache_path, "w") as f:
        f.attrs["T"] = 100
    with pytest.raises(RuntimeError, match="missing metadata"):
        require_causal_traffic_cache(cache_path)

    write_traffic_meta(cache_path)
    meta = require_causal_traffic_cache(
        cache_path,
        expected_region="Synthetic",
        expected_n=2,
        expected_baseline_train_frac=0.70,
        expected_split_train_end=0.70,
    )
    assert meta["T"] == 100
    assert meta["traffic_cache_id"] == "traffic-cache-id"

    with pytest.raises(RuntimeError, match="baseline fraction mismatch"):
        require_causal_traffic_cache(
            cache_path, expected_baseline_train_frac=0.60,
        )
    with pytest.raises(RuntimeError, match="region mismatch"):
        require_causal_traffic_cache(cache_path, expected_region="Orange")


def test_traffic_cache_guard_rejects_inconsistent_cutoff_and_pair_id(tmp_path):
    cache_path = tmp_path / "traffic.h5"
    write_traffic_meta(cache_path, baseline_train_cutoff=69)
    with pytest.raises(RuntimeError, match="inconsistent baseline cutoff"):
        require_causal_traffic_cache(cache_path)

    write_traffic_meta(cache_path, traffic_cache_id="different-id")
    with pytest.raises(RuntimeError, match="inconsistent cache_id"):
        require_causal_traffic_cache(cache_path)


def test_build_samples_h5_drops_boundaries_and_duplicate_starts(tmp_path):
    starts = [10, 11, 57, 58, 69, 69, 70, 72, 73, 84, 84, 85, 87, 88]
    events = np.array(starts, dtype=np.int64)

    import pandas as pd

    event_frame = pd.DataFrame({
        "start_idx": events,
        "anchor_region_idx": np.zeros(len(events), dtype=np.int64),
        "incident_ids": [""] * len(events),
    })
    args = SimpleNamespace(
        smoke_events=0,
        sample_offsets=[0],
        t_h=12,
        t_p=12,
        active_window=24,
        m_max=32,
        split_train_end=0.70,
        split_val_end=0.85,
    )
    out_path = tmp_path / "samples.h5"

    build_samples_h5(
        region_name="Synthetic",
        region_node_idx=np.array([0, 1], dtype=np.int64),
        events=event_frame,
        affected_lookup={},
        total_steps=100,
        args=args,
        out_path=out_path,
        traffic_cache_meta=traffic_meta(),
        samples_cache_id="samples-cache-id",
        build_id="samples-build-id",
    )

    with h5py.File(out_path, "r") as f:
        assert np.array_equal(
            f["sample_start"][:],
            np.array([11, 57, 69, 70, 72, 84, 85, 87], dtype=np.int64),
        )
        assert np.array_equal(
            f["split"][:],
            np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int8),
        )
        assert int(f.attrs["n_split_boundary_dropped"]) == 2
        assert int(f.attrs["n_duplicate_sample_start_dropped"]) == 2
        assert np.unique(f["sample_start"][:]).size == f["sample_start"].size
        assert f.attrs["cache_id"] == "samples-cache-id"
        assert f.attrs["build_id"] == "samples-build-id"
        assert f.attrs["traffic_cache_id"] == "traffic-cache-id"
        assert int(f.attrs["baseline_train_cutoff"]) == 70
        assert float(f.attrs["baseline_train_frac"]) == pytest.approx(0.70)
        assert int(f.attrs["T"]) == 100
