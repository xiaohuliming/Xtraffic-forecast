import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from dist_net.data import FullWindowRegionData, RegionData
from dist_net.protocol import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION


def _write_v2_fixture(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    graph_dir = root / "graphs"
    data_dir.mkdir()
    graph_dir.mkdir()
    region = "Alameda"
    total_steps, n_nodes, history, horizon, max_events = 100, 2, 12, 12, 3
    event_starts = np.array([20, 70, 85], dtype=np.int64)

    with h5py.File(data_dir / f"{region}_samples.h5", "w") as h5:
        h5.create_dataset("sample_start", data=event_starts)
        h5.create_dataset("split", data=np.array([0, 1, 2], dtype=np.int8))
        h5.create_dataset("event_idx", data=np.arange(3, dtype=np.int32))
        h5.create_dataset("event_start_idx", data=event_starts)
        h5.create_dataset("anchor_region_idx", data=np.zeros(3, dtype=np.int32))
        h5.create_dataset("sample_offset", data=np.zeros(3, dtype=np.int8))
        feature = np.zeros((3, max_events, 12), dtype=np.float32)
        feature[:, 1, 0] = 1.0
        h5.create_dataset("incident_feat", data=feature)
        mask = np.zeros((3, max_events), dtype=bool)
        mask[:, 1] = True
        h5.create_dataset("incident_mask", data=mask)
        h5.create_dataset("n_active_incidents", data=np.ones(3, dtype=np.int32))
        active = np.full((3, max_events), -1, dtype=np.int32)
        active[:, 1] = np.arange(3)
        h5.create_dataset("active_event_idx", data=active)
        h5.create_dataset("primary_incident_slot", data=np.ones(3, dtype=np.int16))
        analysis = h5.create_group("analysis_labels")
        affected = np.zeros((3, n_nodes), dtype=bool)
        affected[:, 0] = True
        analysis.create_dataset("affected_mask", data=affected)
        h5.attrs["cache_schema_version"] = CACHE_SCHEMA_VERSION
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.attrs["M_max"] = max_events
        h5.attrs["C_e"] = 12
        h5.attrs["N"] = n_nodes
        h5.attrs["T_h"] = history
        h5.attrs["T_p"] = horizon
        h5.attrs["train_end"] = 70
        h5.attrs["val_end"] = 85

    flow = np.arange(total_steps * n_nodes * 3, dtype=np.float32).reshape(
        total_steps, n_nodes, 3
    )
    times = pd.date_range("2023-01-01", periods=total_steps, freq="5min")
    with h5py.File(data_dir / f"{region}_traffic.h5", "w") as h5:
        h5.create_dataset("flow_series_imputed", data=flow)
        h5.create_dataset("flow_mask", data=np.ones_like(flow, dtype=bool))
        h5.create_dataset("time_enc", data=np.zeros((total_steps, 5), dtype=np.float32))
        h5.create_dataset("static_meta", data=np.zeros((n_nodes, 8), dtype=np.float32))
        h5.create_dataset("baseline_median", data=np.zeros((2, 288, n_nodes, 3)))
        h5.create_dataset("baseline_scale", data=np.ones((2, 288, n_nodes, 3)))
        h5.create_dataset("region_idx", data=np.arange(n_nodes, dtype=np.int64))
        h5.create_dataset("times_ns", data=times.asi8)
        h5.attrs["cache_schema_version"] = CACHE_SCHEMA_VERSION
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.attrs["T"] = total_steps
        h5.attrs["meta_feature_names"] = np.array([], dtype="S1")

    np.savez(
        data_dir / f"{region}_event_rel_feat.npz",
        rel_feat=np.zeros((3, n_nodes, 4), dtype=np.float32),
        event_start_idx=event_starts,
        protocol_version=np.array(PROTOCOL_VERSION),
    )
    np.savez(
        graph_dir / "alameda_sparse_adj.npz",
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        core_idx=np.arange(n_nodes, dtype=np.int64),
        region_idx=np.arange(n_nodes, dtype=np.int64),
    )
    return data_dir, graph_dir


def test_region_data_reads_v2_analysis_labels_and_exact_window(tmp_path):
    data_dir, graph_dir = _write_v2_fixture(tmp_path)
    region = RegionData("Alameda", data_dir, graph_dir)

    sample = region.get_sample(0)

    assert region.train_end == 70
    assert region.val_end == 85
    assert sample["primary_incident_slot"] == 1
    np.testing.assert_array_equal(sample["affected_mask"], [True, False])
    assert sample["x_hist"][0, 0, 0] == region.flow_series[9, 0, 0]
    assert sample["y_true"][0, 0, 0] == region.flow_series[21, 0, 0]
    metadata = region.result_metadata()
    assert metadata["protocol_version"] == PROTOCOL_VERSION
    assert metadata["split"] == {"train_end": 70, "val_end": 85}
    assert "training_git" in metadata
    json.dumps(metadata)


def test_full_window_loader_purges_overlapping_targets(tmp_path):
    data_dir, graph_dir = _write_v2_fixture(tmp_path)
    region = FullWindowRegionData(
        "Alameda", data_dir, graph_dir, train_frac=0.5, val_frac=0.25
    )

    target_sets = {
        split: {
            target
            for anchor in region.sample_start[region.split == split]
            for target in range(int(anchor) + 1, int(anchor) + region.T_p + 1)
        }
        for split in (0, 1, 2)
    }
    assert all(target_sets.values())
    assert target_sets[0].isdisjoint(target_sets[1])
    assert target_sets[1].isdisjoint(target_sets[2])


def test_region_data_rejects_legacy_cache_by_default(tmp_path):
    data_dir, graph_dir = _write_v2_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as h5:
        h5.attrs["protocol_version"] = "legacy"

    with pytest.raises(RuntimeError, match="required 'protocol_v2_noleak'"):
        RegionData("Alameda", data_dir, graph_dir)
