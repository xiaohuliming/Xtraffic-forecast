import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from dist_net.data import FullWindowRegionData, RegionData
from dist_net.protocol import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION


def _write_v3_fixture(root: Path) -> tuple[Path, Path]:
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
        h5.create_dataset(
            "event_id",
            data=np.array(["agg:a", "agg:b", "agg:c"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        h5.create_dataset("anchor_region_idx", data=np.zeros(3, dtype=np.int32))
        h5.create_dataset("sample_offset", data=np.zeros(3, dtype=np.int8))
        feature = np.zeros((3, max_events, 12), dtype=np.float32)
        feature[:, 1, 0] = 1.0
        h5.create_dataset("incident_feat", data=feature)
        mask = np.zeros((3, max_events), dtype=bool)
        mask[:, 1] = True
        h5.create_dataset("incident_mask", data=mask)
        h5.create_dataset("n_active_incidents", data=np.ones(3, dtype=np.int32))
        h5.create_dataset("n_packed_incidents", data=np.ones(3, dtype=np.int32))
        active = np.full((3, max_events), -1, dtype=np.int32)
        active[:, 1] = np.arange(3)
        h5.create_dataset("active_event_idx", data=active)
        h5.create_dataset("primary_incident_slot", data=np.ones(3, dtype=np.int16))
        anchor_mask = np.zeros((3, max_events), dtype=bool)
        anchor_mask[:, 1] = True
        h5.create_dataset("anchor_event_mask", data=anchor_mask)
        h5.create_dataset("n_origin_events", data=np.ones(3, dtype=np.int16))
        analysis = h5.create_group("analysis_labels")
        affected = np.zeros((3, n_nodes), dtype=bool)
        affected[:, 0] = True
        analysis.create_dataset("affected_mask", data=affected)
        h5.attrs["cache_schema_version"] = CACHE_SCHEMA_VERSION
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.attrs["M_max"] = max_events
        h5.attrs["C_e"] = 12
        h5.attrs["N"] = n_nodes
        h5.attrs["T"] = total_steps
        h5.attrs["T_h"] = history
        h5.attrs["T_p"] = horizon
        h5.attrs["train_end"] = 70
        h5.attrs["val_end"] = 85
        h5.attrs["bundle_id"] = "bundle-test"
        h5.attrs["traffic_cache_id"] = "traffic-test"
        h5.attrs["samples_cache_id"] = "samples-test"
        h5.attrs["artifact_id"] = "samples-test"
        h5.attrs["cache_role"] = "samples"
        h5.attrs["cache_mode"] = "formal"
        h5.attrs["git_sha"] = "a" * 40
        h5.attrs["git_branch"] = "main"
        h5.attrs["git_dirty"] = False
        h5.attrs["baseline_fit_start"] = 0
        h5.attrs["baseline_fit_end"] = 70
        h5.attrs["n_duplicate_sample_start_dropped"] = 0
        h5.attrs["n_mixed_active_event_samples_dropped"] = 0
        h5.attrs["n_mixed_events_dropped"] = 0
        h5.attrs["max_active_seen"] = 1
        h5.attrs["n_truncated_active"] = 0

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
        h5.attrs["N"] = n_nodes
        h5.attrs["train_end"] = 70
        h5.attrs["val_end"] = 85
        h5.attrs["baseline_fit_start"] = 0
        h5.attrs["baseline_fit_end"] = 70
        h5.attrs["imputation_method"] = "causal_forward_fill_zero_prefix"
        h5.attrs["baseline_method"] = "label_free_train_only_robust_median"
        h5.attrs["bundle_id"] = "bundle-test"
        h5.attrs["traffic_cache_id"] = "traffic-test"
        h5.attrs["artifact_id"] = "traffic-test"
        h5.attrs["cache_role"] = "traffic"
        h5.attrs["cache_mode"] = "formal"
        h5.attrs["git_sha"] = "a" * 40
        h5.attrs["git_branch"] = "main"
        h5.attrs["git_dirty"] = False
        h5.attrs["meta_feature_names"] = np.array([], dtype="S1")

    np.savez(
        data_dir / f"{region}_event_rel_feat.npz",
        rel_feat=np.zeros((3, n_nodes, 4), dtype=np.float32),
        event_start_idx=event_starts,
        event_id=np.array(["agg:a", "agg:b", "agg:c"]),
        protocol_version=np.array(PROTOCOL_VERSION),
        cache_schema_version=np.int32(CACHE_SCHEMA_VERSION),
        bundle_id=np.array("bundle-test"),
        traffic_cache_id=np.array("traffic-test"),
        samples_cache_id=np.array("samples-test"),
        relation_cache_id=np.array("relation-test"),
        artifact_id=np.array("relation-test"),
        cache_role=np.array("relation"),
        cache_mode=np.array("formal"),
        git_sha=np.array("a" * 40),
        git_branch=np.array("main"),
        git_dirty=np.bool_(False),
    )
    np.savez(
        graph_dir / "alameda_sparse_adj.npz",
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        core_idx=np.arange(n_nodes, dtype=np.int64),
        region_idx=np.arange(n_nodes, dtype=np.int64),
    )
    return data_dir, graph_dir


def test_region_data_reads_v3_analysis_labels_and_exact_window(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    region = RegionData(
        "Alameda",
        data_dir,
        graph_dir,
        include_analysis_labels=True,
        require_rel_feat=True,
    )

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
    assert metadata["cache_provenance"]["bundle_id"] == "bundle-test"
    assert metadata["cache_provenance"]["samples_cache_id"] == "samples-test"
    assert metadata["cache_provenance"]["relation_cache_id"] == "relation-test"
    assert len(metadata["cache_provenance"]["artifact_sha256"]) == 4


def test_analysis_labels_are_opt_in(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    region = RegionData("Alameda", data_dir, graph_dir)

    assert "affected_mask" not in region.get_sample(0)
    assert not hasattr(region, "affected_mask")
    np.testing.assert_array_equal(
        region.analysis_labels(0)["affected_mask"], [True, False]
    )


def test_full_window_loader_purges_overlapping_targets(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    region = FullWindowRegionData(
        "Alameda", data_dir, graph_dir, train_frac=0.7, val_frac=0.15
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
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as h5:
        h5.attrs["protocol_version"] = "legacy"

    with pytest.raises(RuntimeError, match=PROTOCOL_VERSION):
        RegionData("Alameda", data_dir, graph_dir)


def test_region_data_rejects_duplicate_forecast_origins(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as h5:
        h5["sample_start"][1] = h5["sample_start"][0]

    with pytest.raises(RuntimeError, match="sample_start.*unique"):
        RegionData("Alameda", data_dir, graph_dir)


def test_region_data_rejects_missing_v3_required_field(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as h5:
        del h5["anchor_event_mask"]

    with pytest.raises(RuntimeError, match="anchor_event_mask"):
        RegionData("Alameda", data_dir, graph_dir)


def test_region_data_rejects_mismatched_bundle_identity(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_traffic.h5", "r+") as h5:
        h5.attrs["bundle_id"] = "wrong-bundle"

    with pytest.raises(RuntimeError, match="bundle_id"):
        RegionData("Alameda", data_dir, graph_dir)


@pytest.mark.parametrize(
    ("artifact", "expected_message"),
    [
        ("samples", "samples.*artifact_id"),
        ("traffic", "traffic.*artifact_id"),
        ("relation", "relation.*artifact_id"),
    ],
)
def test_region_data_rejects_mismatched_artifact_identity(
    tmp_path, artifact, expected_message
):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    if artifact in {"samples", "traffic"}:
        path = data_dir / f"Alameda_{artifact}.h5"
        with h5py.File(path, "r+") as h5:
            h5.attrs["artifact_id"] = "wrong-artifact"
    else:
        path = data_dir / "Alameda_event_rel_feat.npz"
        with np.load(path, allow_pickle=False) as relation:
            payload = {name: relation[name] for name in relation.files}
        payload["artifact_id"] = np.array("wrong-artifact")
        np.savez(path, **payload)

    with pytest.raises(RuntimeError, match=expected_message):
        RegionData("Alameda", data_dir, graph_dir, require_rel_feat=True)


def test_region_data_rejects_dirty_formal_cache(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    with h5py.File(data_dir / "Alameda_samples.h5", "r+") as h5:
        h5.attrs["git_dirty"] = True

    with pytest.raises(RuntimeError, match="dirty.*formal|formal.*dirty"):
        RegionData("Alameda", data_dir, graph_dir)


def test_region_data_requires_relation_when_requested(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)
    (data_dir / "Alameda_event_rel_feat.npz").unlink()

    with pytest.raises(RuntimeError, match="relation"):
        RegionData(
            "Alameda", data_dir, graph_dir, require_rel_feat=True
        )


def test_full_window_rejects_baseline_fitted_past_new_train_boundary(tmp_path):
    data_dir, graph_dir = _write_v3_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="baseline_fit_end"):
        FullWindowRegionData(
            "Alameda",
            data_dir,
            graph_dir,
            train_frac=0.5,
            val_frac=0.25,
        )
