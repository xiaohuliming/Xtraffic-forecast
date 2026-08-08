from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.protocol_v3_event_extension_data import (
    V3EventBatchLoader,
    WeeklyClimatology,
    fit_weekly_climatology,
    load_aligned_sensor_support,
    prepare_ndr_batch,
)
from scripts.train_protocol_v3_event_extension import (
    _backward_standard_batch,
    _forward_standard,
)


class _FakeRegionData:
    region_name = "Alameda"
    N = 2
    T_h = 12
    T_p = 12
    region_idx = np.asarray([1, 3], dtype=np.int64)
    time_enc = np.zeros((64, 5), dtype=np.float32)

    def __init__(self) -> None:
        self.split = np.asarray([0, 0, 1, 2], dtype=np.int8)
        self.sample_start = np.asarray([20, 21, 40, 52], dtype=np.int64)
        self.primary_incident_slot = np.asarray([1, 1, 1, 1], dtype=np.int16)
        self.incident_feat = np.zeros((4, 2, 12), dtype=np.float32)
        self.incident_feat[:, 1, 4] = 1
        self.incident_feat[:, 1, 8] = 0.5
        self.time_enc[self.sample_start, 4] = 1

    def get_sample(self, index: int) -> dict[str, np.ndarray]:
        x = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
        y = x + np.float32(index + 1)
        x_mask = np.ones_like(x, dtype=np.bool_)
        y_mask = np.ones_like(y, dtype=np.bool_)
        y_mask[0, 0, 0] = False
        rel = np.zeros((2, 2, 4), dtype=np.float32)
        rel[1, :, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
        rel[1, :, 1] = np.asarray([0.0, 2.0], dtype=np.float32)
        rel[1, :, 2] = np.asarray([0.0, 1.0], dtype=np.float32)
        return {
            "x_hist": x,
            "x_hist_mask": x_mask,
            "y_true": y,
            "y_mask": y_mask,
            "time_feat": np.stack(
                [
                    np.arange(12, dtype=np.float32) / 288,
                    np.ones(12, dtype=np.float32) / 7,
                ],
                axis=-1,
            ),
            "incident_feat": self.incident_feat[index],
            "incident_mask": np.asarray([True, True]),
            "rel_feat": rel,
            "primary_incident_slot": np.int16(1),
            "sample_start": np.int64(self.sample_start[index]),
            "region_code": np.int64(0),
        }


def _write_sensors(path: Path, *, station_ids: list[int], orders: list[int]) -> None:
    pd.DataFrame(
        {
            "station_id": station_ids,
            "order": orders,
            "Sensor Type": ["loops", "loops"],
            "Surface": ["Concrete", "Bridge Deck"],
            "Roadway Use": ["Normal", "HOV"],
            "Road Width": ["24 ft", "36 ft"],
            "Design Speed Limit": ["65 mph", "70 mph"],
        }
    ).to_csv(path, index=False)


def test_sensor_support_requires_exact_node_identity_and_order(tmp_path: Path) -> None:
    sensors = tmp_path / "sensors.csv"
    node_order = tmp_path / "node_order.npy"
    _write_sensors(sensors, station_ids=[101, 303], orders=[1, 3])
    np.save(node_order, np.asarray([9, 101, 8, 303], dtype=np.int64))

    support = load_aligned_sensor_support(
        sensors_path=sensors,
        node_order_path=node_order,
        region_idx=np.asarray([1, 3], dtype=np.int64),
    )
    assert support.node_count == 2
    assert support.station_id_sha256
    assert support.sensor_info["sensor_type"].shape == (2,)

    _write_sensors(sensors, station_ids=[303, 101], orders=[1, 3])
    with pytest.raises(ValueError, match="station_id"):
        load_aligned_sensor_support(
            sensors_path=sensors,
            node_order_path=node_order,
            region_idx=np.asarray([1, 3], dtype=np.int64),
        )


def test_v3_loader_keeps_masks_and_excludes_analysis_labels(tmp_path: Path) -> None:
    rdata = _FakeRegionData()
    sensors = tmp_path / "sensors.csv"
    node_order = tmp_path / "node_order.npy"
    _write_sensors(sensors, station_ids=[101, 303], orders=[1, 3])
    np.save(node_order, np.asarray([9, 101, 8, 303], dtype=np.int64))
    support = load_aligned_sensor_support(
        sensors_path=sensors,
        node_order_path=node_order,
        region_idx=rdata.region_idx,
    )
    loader = V3EventBatchLoader(
        region_data=rdata,
        indices=np.asarray([0, 1]),
        batch_size=2,
        flow_mean=10.0,
        flow_std=2.0,
        sensor_support=support,
        include_incident=True,
        shuffle=False,
        drop_last=False,
        seed=42,
    )

    batch = next(loader.get_iterator())
    assert batch["x_data"].shape == (2, 12, 2, 3)
    assert batch["y_data"].shape == (2, 12, 2, 1)
    assert batch["y_mask"].shape == (2, 12, 2, 1)
    assert batch["y_mask"][:, 0, 0, 0].tolist() == [False, False]
    assert batch["incident_features"].shape == (2, 4)
    assert batch["incident_features"][0].tolist() == [6.0, 1.0, 4.0, 1.0]
    assert batch["incident_position"].tolist() == [0, 0]
    assert batch["incident_distances"].shape == (2, 2, 3)
    assert batch["sample_start"].tolist() == [20, 21]
    assert "affected_mask" not in batch
    assert all(not key.startswith("analysis") for key in batch)


def test_ndr_batch_uses_explicit_observation_mask() -> None:
    rdata = _FakeRegionData()
    loader = V3EventBatchLoader(
        region_data=rdata,
        indices=np.asarray([0]),
        batch_size=1,
        flow_mean=10.0,
        flow_std=2.0,
        sensor_support=None,
        include_incident=False,
        shuffle=False,
        drop_last=False,
        seed=42,
    )
    batch = next(loader.get_iterator())
    climatology = WeeklyClimatology(
        baseline=np.zeros((7, 288, 2), dtype=np.float32),
        flow_mean=10.0,
        flow_std=2.0,
        residual_std=3.0,
        residual_std_by_node=np.ones(2, dtype=np.float32),
        valid_training_points=100,
    )
    prepared = prepare_ndr_batch(batch, climatology)
    assert prepared["target_mask"].shape == (1, 2, 12)
    assert not bool(prepared["target_mask"][0, 0, 0])
    assert bool(prepared["target_mask"][0, 1, 0])
    assert prepared["target"].shape == (1, 2, 12)


def test_weekly_climatology_fits_raw_train_timestamps_once() -> None:
    class Region:
        N = 2
        train_end = 4
        dow = np.asarray([0, 0, 0, 0, 0], dtype=np.int8)
        tod = np.asarray([1, 1, 2, 2, 1], dtype=np.int16)
        flow_series = np.zeros((5, 2, 3), dtype=np.float32)
        flow_mask = np.ones((5, 2, 3), dtype=np.bool_)

    region = Region()
    region.flow_series[:, :, 0] = np.asarray(
        [[1, 10], [3, 30], [5, 50], [7, 70], [999, 999]],
        dtype=np.float32,
    )
    region.flow_mask[1, 1, 0] = False
    stats = fit_weekly_climatology(region)

    np.testing.assert_allclose(stats.baseline[0, 1], [2, 10])
    np.testing.assert_allclose(stats.baseline[0, 2], [6, 60])
    assert stats.valid_training_points == 7
    assert stats.flow_mean < 100
    assert np.isfinite(stats.residual_std)


def test_standard_models_never_receive_future_targets() -> None:
    class BiST(torch.nn.Module):
        def forward(self, x, label=None):
            assert label is None
            return torch.zeros(x.shape[0], 12, x.shape[2], 1)

    class IGSTGNN(torch.nn.Module):
        def forward(self, x, label=None, **kwargs):
            assert label is None
            assert set(kwargs) == {"incident_data", "sensor_data"}
            return torch.zeros(x.shape[0], 12, x.shape[2], 1)

    tensors = {
        "x_data": torch.zeros(2, 12, 3, 3),
        "y_data": torch.ones(2, 12, 3, 1),
        "incident_features": torch.zeros(2, 4),
        "incident_position": torch.zeros(2, dtype=torch.long),
        "incident_distances": torch.zeros(2, 3, 3),
        "sensor_data": {},
    }
    assert _forward_standard(BiST(), "bist", tensors).shape == (2, 12, 3)
    assert _forward_standard(IGSTGNN(), "igstgnn", tensors).shape == (2, 12, 3)


def test_curriculum_loss_uses_only_the_active_horizons() -> None:
    class BiST(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return self.value.expand(x.shape[0], 12, x.shape[2], 1)

    model = BiST()
    batch = {
        "x_data": np.zeros((2, 12, 3, 3), dtype=np.float32),
        "y_data": np.ones((2, 12, 3, 1), dtype=np.float32),
        "y_mask": np.ones((2, 12, 3, 1), dtype=np.bool_),
    }
    error_sum, valid_count = _backward_standard_batch(
        model=model,
        method="bist",
        batch=batch,
        device=torch.device("cpu"),
        mean=0.0,
        std=1.0,
        max_microbatch_size=1,
        curriculum_horizon=3,
    )

    assert error_sum == 18.0
    assert valid_count == 18
    assert model.value.grad is not None
    assert float(model.value.grad) == pytest.approx(-1.0)
