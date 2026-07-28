import hashlib
import os
from types import SimpleNamespace

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.runtime_data import (
    IncidentStandardScaler,
    TupleLoaderView,
    load_official_adjacency,
    load_official_incident_stats,
    load_official_sensor_info,
)


class _FakeLoader:
    def __init__(self):
        self.size = 3
        self.num_batch = 2
        self.current_ind = 0
        self.shuffle_calls = 0

    def shuffle(self):
        self.shuffle_calls += 1

    def get_iterator(self):
        self.current_ind = 0

        def iterator():
            self.current_ind += 1
            yield {
                "x_data": np.ones((2, 12, 3, 1), dtype=np.float32),
                "y_data": np.zeros((2, 12, 3, 1), dtype=np.float32),
                "sample_indices": np.array([0, 1]),
            }

        return iterator()


def test_tuple_loader_view_matches_the_largest_engine_contract():
    source = _FakeLoader()
    view = TupleLoaderView(source)

    view.shuffle()
    batch = next(view.get_iterator())

    assert source.shuffle_calls == 1
    assert len(batch) == 2
    assert batch[0].shape == (2, 12, 3, 1)
    assert batch[1].shape == (2, 12, 3, 1)
    assert view.size == 3
    assert view.num_batch == 2


def test_runtime_support_loaders_preserve_the_published_sensor_fallbacks(
    tmp_path,
):
    (tmp_path / "sensors.csv").write_text(
        "order,Sensor Type,Surface,Roadway Use,Road Width,"
        "Design Speed Limit\n"
        "42,loops,Concrete,None,80 ft,70 mph\n"
        "7,,Bridge Deck,,24 ft,65 mph\n",
        encoding="utf-8",
    )
    np.savez(
        tmp_path / "incident_stats.npz",
        mean=np.float32(147.5),
        std=np.float32(173.25),
        normalized=np.array(True),
    )

    sensor_info, dimensions = load_official_sensor_info(
        tmp_path / "sensors.csv",
        node_count=2,
    )
    stats = load_official_incident_stats(tmp_path / "incident_stats.npz")

    assert sensor_info["sensor_type"].tolist() == [0, 1]
    assert sensor_info["surface"].tolist() == [0, 1]
    assert sensor_info["roadway_use"].tolist() == [0, 1]
    assert sensor_info["road_width"].tolist() == [0.0, 0.0]
    assert sensor_info["speed_limit"].tolist() == [50.0, 50.0]
    assert dimensions == SimpleNamespace(
        sensor_type_size=2,
        surface_size=2,
        roadway_use_size=2,
    )
    assert stats == {
        "mean": np.float32(147.5),
        "std": np.float32(173.25),
        "normalized": True,
    }


def test_runtime_support_loaders_use_the_windows_compatibility_boundary(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sensors.csv"
    path.write_text(
        "order,Sensor Type,Surface,Roadway Use,Road Width,"
        "Design Speed Limit\n"
        "0,loops,Concrete,None,80,70\n",
        encoding="utf-8",
    )
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    sensor_info, _ = load_official_sensor_info(path, node_count=1)

    assert sensor_info["road_width"].tolist() == [80.0]


def test_runtime_support_loader_binds_the_bytes_to_the_split_manifest(
    tmp_path,
):
    path = tmp_path / "sensors.csv"
    original = (
        "order,Sensor Type,Surface,Roadway Use,Road Width,"
        "Design Speed Limit\n"
        "0,loops,Concrete,None,80,70\n"
    ).encode("utf-8")
    path.write_bytes(original)
    expected_sha256 = hashlib.sha256(original).hexdigest()
    path.write_bytes(original.replace(b"loops", b"radar"))

    try:
        load_official_sensor_info(
            path,
            node_count=1,
            expected_bytes=len(original),
            expected_sha256=expected_sha256,
        )
    except ValueError as error:
        assert "split manifest" in str(error)
    else:
        raise AssertionError("Unbound runtime support bytes were accepted")


def test_runtime_adjacency_and_scaler_match_the_upstream_contract(tmp_path):
    adjacency = np.asarray(
        [[1.0, 0.25], [0.5, 1.0]],
        dtype=np.float32,
    )
    np.save(tmp_path / "adj_matrix.npy", adjacency, allow_pickle=False)

    loaded = load_official_adjacency(
        tmp_path / "adj_matrix.npy",
        node_count=2,
    )
    scaler = IncidentStandardScaler(
        mean=np.float32(100),
        std=np.float32(20),
    )
    normalized = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32)

    np.testing.assert_array_equal(loaded, adjacency)
    restored = scaler.inverse_transform(normalized)
    torch.testing.assert_close(
        restored,
        torch.tensor([[[[100.0], [120.0]]]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        scaler.transform(restored),
        normalized,
    )
