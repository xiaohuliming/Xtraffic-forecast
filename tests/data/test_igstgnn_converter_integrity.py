from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from dist_net.protocol import PROTOCOL_VERSION
from scripts import build_igstgnn_data_from_h5 as converter


def _write_converter_fixture(root: Path) -> tuple[str, Path]:
    region = "Alameda"
    total_steps = 32
    n_nodes = 2
    anchor = 10
    flow = np.arange(
        total_steps * n_nodes * 3, dtype=np.float32
    ).reshape(total_steps, n_nodes, 3)

    with h5py.File(root / f"{region}_traffic.h5", "w") as h5:
        h5.attrs["protocol_version"] = PROTOCOL_VERSION
        h5.create_dataset("flow_series_imputed", data=flow)
        h5.create_dataset("flow_mask", data=np.ones_like(flow, dtype=bool))
        h5.create_dataset(
            "time_enc", data=np.zeros((total_steps, 5), dtype=np.float32)
        )
        h5.create_dataset(
            "times_ns",
            data=pd.date_range(
                "2023-01-01", periods=total_steps, freq="5min"
            ).asi8,
        )

    incident_feat = np.zeros((1, 1, 12), dtype=np.float32)
    incident_feat[0, 0, 0] = 1.0
    incident_feat[0, 0, 8] = 0.5
    with h5py.File(root / f"{region}_samples.h5", "w") as h5:
        h5.attrs.update(
            {
                "protocol_version": PROTOCOL_VERSION,
                "T_h": 3,
                "T_p": 2,
                "N": n_nodes,
                "M_max": 1,
                "train_end": 20,
            }
        )
        h5.create_dataset("split", data=np.array([0], dtype=np.int8))
        h5.create_dataset(
            "sample_start", data=np.array([anchor], dtype=np.int64)
        )
        h5.create_dataset("incident_feat", data=incident_feat)
        h5.create_dataset(
            "n_active_incidents", data=np.array([1], dtype=np.int16)
        )
        h5.create_dataset(
            "active_event_idx", data=np.array([[0]], dtype=np.int32)
        )
        h5.create_dataset(
            "primary_incident_slot", data=np.array([0], dtype=np.int16)
        )
        h5.create_dataset("event_idx", data=np.array([0], dtype=np.int32))
        labels = h5.create_group("analysis_labels")
        labels.create_dataset(
            "affected_mask", data=np.zeros((1, n_nodes), dtype=bool)
        )

    rel_feat = np.zeros((1, n_nodes, 4), dtype=np.float32)
    rel_feat[0, 1, 0:2] = 1.0
    rel_path = root / f"{region}_event_rel_feat.npz"
    np.savez(rel_path, rel_feat=rel_feat)
    return region, rel_path


def _convert_unverified_fixture(source: Path, output: Path) -> str:
    region, _ = _write_converter_fixture(source)
    converter.build_split(
        region,
        source,
        output,
        sigma_euc_log=1.5,
        sigma_road_log=1.5,
        cutoff_euc_log=4.0,
        cutoff_road_log=5.0,
        normalization={
            "mean_per_ch": np.zeros(3, dtype=np.float32),
            "std_per_ch": np.ones(3, dtype=np.float32),
            "flow_mean": 0.0,
            "flow_std": 1.0,
        },
        validate_source=False,
    )
    return region


def test_converter_feeds_elapsed_time_to_the_feature_used_by_model(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    region = _convert_unverified_fixture(source, output)

    samples = np.load(
        output / region / "incident_data_train.npy", allow_pickle=True
    )
    sample = samples[0]
    assert sample["event_features"]["Event Time"] == pytest.approx(6.0)
    assert sample["durations"] == pytest.approx(6.0)


def test_converter_does_not_zero_the_second_nearest_sensor_for_exact_hit(
    tmp_path,
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    region = _convert_unverified_fixture(source, output)

    samples = np.load(
        output / region / "incident_data_train.npy", allow_pickle=True
    )
    distances = samples[0]["event_distances"]
    np.testing.assert_array_equal(distances[0], np.zeros(3))
    assert distances[1, 0] > 0
    assert distances[1, 1] > 0


def test_converter_uses_strict_region_bundle_validation_by_default(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    graph = tmp_path / "graph"
    source.mkdir()
    graph.mkdir()
    region, _ = _write_converter_fixture(source)

    class ValidationReached(RuntimeError):
        pass

    def reject_unverified_bundle(*_args, **_kwargs):
        raise ValidationReached("strict RegionData validation reached")

    monkeypatch.setattr(converter, "RegionData", reject_unverified_bundle)
    with pytest.raises(ValidationReached, match="strict RegionData"):
        converter.build_split(
            region,
            source,
            output,
            sigma_euc_log=1.5,
            sigma_road_log=1.5,
            cutoff_euc_log=4.0,
            cutoff_road_log=5.0,
            graph_dir=graph,
        )

