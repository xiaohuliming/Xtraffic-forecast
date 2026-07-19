from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from dist_net.data import RegionBucketedSampler, RegionData


def _write_region_cache(tmp_path):
    data_dir = tmp_path / "data"
    graph_dir = tmp_path / "graphs"
    data_dir.mkdir()
    graph_dir.mkdir()

    region = "Alameda"
    n_nodes, total_steps, t_h, t_p = 2, 100, 12, 12
    starts = np.array([11, 20, 69, 84], dtype=np.int64)
    splits = np.array([0, 0, 1, 2], dtype=np.int8)
    traffic_id = "traffic-cache-1"

    traffic_path = data_dir / f"{region}_traffic.h5"
    with h5py.File(traffic_path, "w") as f:
        f.create_dataset(
            "flow_series_imputed",
            data=np.zeros((total_steps, n_nodes, 3), dtype=np.float32),
        )
        f.create_dataset(
            "flow_mask",
            data=np.ones((total_steps, n_nodes, 3), dtype=np.bool_),
        )
        f.create_dataset("time_enc", data=np.zeros((total_steps, 5), dtype=np.float32))
        f.create_dataset("static_meta", data=np.zeros((n_nodes, 1), dtype=np.float32))
        f.create_dataset("region_idx", data=np.arange(n_nodes, dtype=np.int64))
        f.create_dataset(
            "baseline_median", data=np.zeros((2, 288, n_nodes, 3), dtype=np.float32)
        )
        f.create_dataset(
            "baseline_scale", data=np.ones((2, 288, n_nodes, 3), dtype=np.float32)
        )
        times = pd.date_range("2023-01-01", periods=total_steps, freq="5min")
        f.create_dataset("times_ns", data=times.asi8.astype(np.int64))
        f.attrs.update({
            "cache_schema_version": 2,
            "cache_id": traffic_id,
            "build_id": "build-1",
            "traffic_cache_id": traffic_id,
            "imputation_method": "causal_forward_fill",
            "baseline_train_frac": 0.70,
            "baseline_train_cutoff": 70,
            "candidate_pm_radius": 5.0,
            "anchor_pm_radius": 2.0,
            "baseline_mask_extra_steps": 12,
            "min_baseline_count": 8,
            "builder_source_sha256": "a" * 64,
            "region_name": region,
            "N": n_nodes,
            "T": total_steps,
        })

    samples_path = data_dir / f"{region}_samples.h5"
    n_samples, m_max, c_e = len(starts), 32, 13
    with h5py.File(samples_path, "w") as f:
        f.create_dataset("sample_start", data=starts)
        f.create_dataset("split", data=splits)
        f.create_dataset("event_idx", data=np.arange(n_samples, dtype=np.int32))
        f.create_dataset("anchor_region_idx", data=np.zeros(n_samples, dtype=np.int32))
        f.create_dataset("sample_offset", data=np.zeros(n_samples, dtype=np.int8))
        f.create_dataset(
            "incident_feat", data=np.zeros((n_samples, m_max, c_e), dtype=np.float32)
        )
        f.create_dataset(
            "incident_mask", data=np.zeros((n_samples, m_max), dtype=np.bool_)
        )
        f.create_dataset(
            "affected_mask", data=np.zeros((n_samples, n_nodes), dtype=np.bool_)
        )
        f.create_dataset("n_active_incidents", data=np.zeros(n_samples, dtype=np.int32))
        f.attrs.update({
            "cache_schema_version": 2,
            "cache_id": "samples-cache-1",
            "build_id": "build-1",
            "traffic_cache_id": traffic_id,
            "imputation_method": "causal_forward_fill",
            "baseline_train_frac": 0.70,
            "baseline_train_cutoff": 70,
            "region_name": region,
            "N": n_nodes,
            "T": total_steps,
            "M_max": m_max,
            "C_e": c_e,
            "T_h": t_h,
            "T_p": t_p,
            "active_window": 24,
            "split_train_end": 0.70,
            "split_val_end": 0.85,
            "n_duplicate_sample_start_dropped": 1,
            "sample_offsets": np.asarray([0, 6, 12], dtype=np.int32),
            "smoke_events": 0,
            "traffic_builder_source_sha256": "a" * 64,
            "samples_builder_source_sha256": "b" * 64,
        })

    np.savez(
        graph_dir / "alameda_sparse_adj.npz",
        edge_index=np.empty((2, 0), dtype=np.int64),
        core_idx=np.arange(n_nodes, dtype=np.int64),
        region_idx=np.arange(n_nodes, dtype=np.int64),
    )
    return data_dir, graph_dir, samples_path


def test_region_data_strict_causal_cache_accepts_matched_v2_pair(tmp_path):
    data_dir, graph_dir, _ = _write_region_cache(tmp_path)

    data = RegionData(
        "Alameda", data_dir, graph_dir, lazy=True, strict_causal_cache=True
    )

    assert data.cache_provenance["traffic_cache_id"] == "traffic-cache-1"
    assert data.cache_provenance["baseline_train_cutoff"] == 70
    assert data.cache_provenance["sample_start_unique"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("pair_id", "pairing ID mismatch"),
        ("duplicate", "duplicate rows"),
        ("cutoff", "baseline_train_cutoff does not match"),
        ("offsets", "sample_offsets mismatch"),
        ("build_id", "one complete build"),
    ],
)
def test_region_data_strict_causal_cache_rejects_bad_cache(
    tmp_path, mutation, message
):
    data_dir, graph_dir, samples_path = _write_region_cache(tmp_path)
    with h5py.File(samples_path, "a") as f:
        if mutation == "pair_id":
            f.attrs["traffic_cache_id"] = "other-traffic-cache"
        elif mutation == "duplicate":
            f["sample_start"][1] = f["sample_start"][0]
        elif mutation == "offsets":
            f.attrs["sample_offsets"] = np.asarray([0, 12], dtype=np.int32)
        elif mutation == "build_id":
            f.attrs["build_id"] = "samples-only-build"
        else:
            f.attrs["baseline_train_cutoff"] = 69

    with pytest.raises(RuntimeError, match=message):
        RegionData(
            "Alameda", data_dir, graph_dir, lazy=True, strict_causal_cache=True
        )


def test_region_data_non_strict_keeps_legacy_compatibility(tmp_path):
    data_dir, graph_dir, samples_path = _write_region_cache(tmp_path)
    traffic_path = data_dir / "Alameda_traffic.h5"
    with h5py.File(traffic_path, "a") as f:
        for key in ("cache_schema_version", "cache_id", "build_id",
                    "traffic_cache_id", "baseline_train_frac",
                    "baseline_train_cutoff"):
            del f.attrs[key]
    with h5py.File(samples_path, "a") as f:
        for key in ("cache_schema_version", "cache_id", "build_id",
                    "traffic_cache_id", "baseline_train_frac",
                    "baseline_train_cutoff", "T"):
            del f.attrs[key]

    data = RegionData("Alameda", data_dir, graph_dir, lazy=True)
    assert data.N == 2
    assert data.cache_provenance["strict_causal_cache"] is False

    with pytest.raises(RuntimeError, match="missing required attr"):
        RegionData(
            "Alameda", data_dir, graph_dir, lazy=True, strict_causal_cache=True
        )


def test_region_bucketed_sampler_changes_each_epoch_reproducibly():
    index_table = [
        (region, local_idx)
        for region in ("Alameda", "ContraCosta", "Orange")
        for local_idx in range(12)
    ]
    dataset = SimpleNamespace(index_table=index_table)

    sampler_a = RegionBucketedSampler(dataset, batch_size=3, seed=42)
    epoch_0_a = list(sampler_a)
    epoch_1_a = list(sampler_a)
    assert epoch_0_a != epoch_1_a

    sampler_b = RegionBucketedSampler(dataset, batch_size=3, seed=42)
    assert list(sampler_b) == epoch_0_a
    assert list(sampler_b) == epoch_1_a

    sampler_a.set_epoch(7)
    sampler_b.set_epoch(7)
    assert list(sampler_a) == list(sampler_b)
