#!/usr/bin/env python3
"""Strict lazy data adapter for v3 event BiST, IGSTGNN, and NDR-Net runs."""

from __future__ import annotations

import csv
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from scripts.build_igstgnn_data_from_h5 import (
    DESC_CODE_BY_OUR_TYPE,
    IGS_TYPE_CODE,
    OUR_TYPES,
    gaussian_kernel,
)
from scripts.reproduction.igstgnn_table4.runtime_data import (
    load_official_sensor_info,
)


@dataclass(frozen=True)
class AlignedSensorSupport:
    node_count: int
    sensor_info: Mapping[str, np.ndarray]
    sensor_dimensions: Any
    station_id_sha256: str
    region_idx_sha256: str


@dataclass(frozen=True)
class WeeklyClimatology:
    baseline: np.ndarray
    flow_mean: float
    flow_std: float
    residual_std: float
    residual_std_by_node: np.ndarray
    valid_training_points: int


def fit_weekly_climatology(region_data: Any) -> WeeklyClimatology:
    """Fit normalization and weekly medians from raw train timestamps only."""
    train_end = int(region_data.train_end)
    node_count = int(region_data.N)
    if train_end <= 0 or node_count <= 0:
        raise ValueError("region training boundary is invalid")
    if region_data.flow_series is None:
        flow_all, mask_all = region_data._slice_traffic(0, train_end)
    else:
        flow_all = region_data.flow_series[:train_end]
        mask_all = region_data.flow_mask[:train_end]
    flow = np.asarray(flow_all[..., 0], dtype=np.float32)
    mask = np.asarray(mask_all[..., 0], dtype=np.bool_)
    if flow.shape != (train_end, node_count) or mask.shape != flow.shape:
        raise ValueError("training flow tensors have an invalid shape")
    observed = np.asarray(flow[mask], dtype=np.float64)
    if observed.size == 0 or not np.isfinite(observed).all():
        raise ValueError("training flow has no finite observed values")
    flow_mean = float(observed.mean(dtype=np.float64))
    flow_std = float(observed.std(dtype=np.float64))
    if not np.isfinite(flow_std) or flow_std <= 1e-6:
        flow_std = 1.0

    dow = np.asarray(region_data.dow[:train_end], dtype=np.int64)
    tod = np.asarray(region_data.tod[:train_end], dtype=np.int64)
    baseline = np.zeros((7, 288, node_count), dtype=np.float32)
    for day in range(7):
        for slot in range(288):
            selected = np.flatnonzero((dow == day) & (tod == slot))
            if selected.size == 0:
                continue
            values = np.where(mask[selected], flow[selected], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(values, axis=0)
            baseline[day, slot] = np.where(
                np.isfinite(medians), medians, np.float32(0)
            )

    expected = baseline[dow, tod]
    residual = np.asarray(flow - expected, dtype=np.float64)
    selected_residual = residual[mask]
    residual_std = float(selected_residual.std(dtype=np.float64))
    if not np.isfinite(residual_std) or residual_std <= 1e-6:
        residual_std = 1.0
    residual_std_by_node = np.ones(node_count, dtype=np.float32)
    for node in range(node_count):
        values = residual[:, node][mask[:, node]]
        if values.size:
            value = float(values.std(dtype=np.float64))
            if np.isfinite(value) and value > 1e-6:
                residual_std_by_node[node] = value
    baseline.setflags(write=False)
    residual_std_by_node.setflags(write=False)
    return WeeklyClimatology(
        baseline=baseline,
        flow_mean=flow_mean,
        flow_std=flow_std,
        residual_std=residual_std,
        residual_std_by_node=residual_std_by_node,
        valid_training_points=int(observed.size),
    )


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _strict_sensor_rows(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"sensor metadata is missing or a symlink: {path}")
    payload = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("sensors.csv contains no rows")
    required = {"station_id", "order"}
    if not required.issubset(rows[0]):
        raise ValueError("sensors.csv must contain station_id and order")
    return rows


def load_aligned_sensor_support(
    *,
    sensors_path: Path | str,
    node_order_path: Path | str,
    region_idx: np.ndarray,
) -> AlignedSensorSupport:
    """Load official metadata only after exact global-node identity alignment."""
    sensor_path = Path(sensors_path)
    rows = _strict_sensor_rows(sensor_path)
    indices = np.asarray(region_idx)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or indices.size == 0:
        raise ValueError("region_idx must be a nonempty integer vector")
    if np.unique(indices).size != indices.size:
        raise ValueError("region_idx must be unique")
    node_path = Path(node_order_path)
    if node_path.is_symlink() or not node_path.is_file():
        raise FileNotFoundError(f"node_order is missing or a symlink: {node_path}")
    node_order = np.load(node_path, allow_pickle=False)
    if (
        node_order.ndim != 1
        or node_order.dtype.kind not in "iu"
        or indices.min() < 0
        or indices.max() >= node_order.size
    ):
        raise ValueError("node_order and region_idx are incompatible")
    try:
        sensor_orders = np.asarray([int(row["order"]) for row in rows], dtype=np.int64)
        station_ids = np.asarray(
            [int(float(row["station_id"])) for row in rows], dtype=np.int64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("sensors.csv node identity columns must be integers") from exc
    if not np.array_equal(sensor_orders, indices.astype(np.int64, copy=False)):
        raise ValueError("sensors.csv order does not exactly match v3 region_idx")
    expected_station_ids = node_order[indices].astype(np.int64, copy=False)
    if not np.array_equal(station_ids, expected_station_ids):
        raise ValueError("sensors.csv station_id does not match v3 node identity")
    sensor_info, dimensions = load_official_sensor_info(
        sensor_path,
        node_count=int(indices.size),
    )
    return AlignedSensorSupport(
        node_count=int(indices.size),
        sensor_info=sensor_info,
        sensor_dimensions=dimensions,
        station_id_sha256=_sha256_array(station_ids),
        region_idx_sha256=_sha256_array(indices.astype(np.int64, copy=False)),
    )


def _weekly_slots_from_history(time_feat: np.ndarray) -> tuple[np.ndarray, ...]:
    tod_scaled = np.asarray(time_feat[..., 0], dtype=np.float64) * 288
    dow_scaled = np.asarray(time_feat[..., 1], dtype=np.float64) * 7
    tod = np.rint(tod_scaled).astype(np.int64)
    dow = np.rint(dow_scaled).astype(np.int64)
    if (
        not np.allclose(tod_scaled, tod, rtol=0, atol=1e-4)
        or not np.allclose(dow_scaled, dow, rtol=0, atol=1e-4)
        or np.any(tod < 0)
        or np.any(tod >= 288)
        or np.any(dow < 0)
        or np.any(dow >= 7)
    ):
        raise ValueError("time features are outside the five-minute weekly grid")
    horizon_offsets = np.arange(1, 13, dtype=np.int64)
    future_absolute = tod[:, -1, None] + horizon_offsets
    future_tod = future_absolute % 288
    future_dow = (dow[:, -1, None] + future_absolute // 288) % 7
    return dow, tod, future_dow, future_tod


class V3EventBatchLoader:
    """Yield numeric batches without materializing object-pickle split files."""

    def __init__(
        self,
        *,
        region_data: Any,
        indices: np.ndarray,
        batch_size: int,
        flow_mean: float,
        flow_std: float,
        sensor_support: AlignedSensorSupport | None,
        include_incident: bool,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.region_data = region_data
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("loader indices must be a nonempty vector")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("loader indices must be unique")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(flow_mean) or not np.isfinite(flow_std) or flow_std <= 0:
            raise ValueError("flow normalization is invalid")
        if sensor_support is not None and sensor_support.node_count != region_data.N:
            raise ValueError("sensor support node count does not match region data")
        self.bs = batch_size
        self.size = int(self.indices.size)
        self.num_batch = (
            self.size // self.bs
            if drop_last
            else (self.size + self.bs - 1) // self.bs
        )
        if self.num_batch <= 0:
            raise ValueError("drop_last removed every loader batch")
        self.flow_mean = np.float32(flow_mean)
        self.flow_std = np.float32(flow_std)
        self.sensor_support = sensor_support
        self.include_incident = bool(include_incident)
        self.should_shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.current_ind = 0
        self._rng = np.random.default_rng(seed)
        self._order = self.indices.copy()

    def shuffle(self) -> None:
        if self.should_shuffle:
            self._order = self.indices.copy()
            self._rng.shuffle(self._order)

    def _incident(self, samples: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        features = np.empty((len(samples), 4), dtype=np.float32)
        positions = np.zeros(len(samples), dtype=np.int64)
        distances = np.empty(
            (len(samples), int(self.region_data.N), 3),
            dtype=np.float32,
        )
        for row, sample in enumerate(samples):
            slot = int(sample["primary_incident_slot"])
            mask = np.asarray(sample["incident_mask"], dtype=np.bool_)
            if slot < 0 or slot >= mask.size or not bool(mask[slot]):
                raise RuntimeError("primary incident slot is not active")
            incident = np.asarray(sample["incident_feat"], dtype=np.float32)[slot]
            one_hot = incident[:8]
            type_index = int(one_hot.argmax()) if float(one_hot.sum()) > 0 else 7
            type_name = OUR_TYPES[type_index]
            elapsed_steps = float(incident[8]) * 12.0
            start = int(sample["sample_start"])
            holiday = int(float(self.region_data.time_enc[start, 4]) > 0.5)
            features[row] = (
                elapsed_steps,
                float(DESC_CODE_BY_OUR_TYPE[type_name]),
                float(IGS_TYPE_CODE[type_name]),
                float(holiday),
            )
            relation = np.asarray(sample["rel_feat"], dtype=np.float32)[slot]
            euclidean = gaussian_kernel(relation[:, 0], 1.5, 4.0)
            road = gaussian_kernel(relation[:, 1], 1.5, 5.0)
            downstream = np.asarray(relation[:, 2] > 0.5, dtype=np.float32)
            value = np.stack((euclidean, road, downstream), axis=-1)
            value[int(np.argmin(relation[:, 0]))] = 0
            distances[row] = value
        return {
            "incident_features": features,
            "incident_position": positions,
            "incident_distances": distances,
        }

    def _build_batch(self, local_indices: np.ndarray) -> dict[str, Any]:
        samples = [self.region_data.get_sample(int(index)) for index in local_indices]
        x_raw = np.stack(
            [np.asarray(sample["x_hist"], dtype=np.float32)[..., 0] for sample in samples]
        )
        y_raw = np.stack(
            [np.asarray(sample["y_true"], dtype=np.float32)[..., 0] for sample in samples]
        )
        x_mask = np.stack(
            [np.asarray(sample["x_hist_mask"], dtype=np.bool_)[..., 0] for sample in samples]
        )
        y_mask = np.stack(
            [np.asarray(sample["y_mask"], dtype=np.bool_)[..., 0] for sample in samples]
        )
        time_feat = np.stack(
            [np.asarray(sample["time_feat"], dtype=np.float32) for sample in samples]
        )
        flow = ((x_raw - self.flow_mean) / self.flow_std).transpose(0, 2, 1)
        nodes = int(self.region_data.N)
        temporal = np.broadcast_to(
            time_feat[:, :, None, :],
            (len(samples), int(self.region_data.T_h), nodes, 2),
        )
        x_data = np.concatenate((flow[..., None], temporal), axis=-1).astype(
            np.float32,
            copy=False,
        )
        y_data = (
            ((y_raw - self.flow_mean) / self.flow_std)
            .transpose(0, 2, 1)[..., None]
            .astype(np.float32, copy=False)
        )
        batch: dict[str, Any] = {
            "x_data": x_data,
            "y_data": y_data,
            "x_mask": x_mask.transpose(0, 2, 1)[..., None],
            "y_mask": y_mask.transpose(0, 2, 1)[..., None],
            "x_raw_flow": x_raw.transpose(0, 2, 1),
            "y_raw_flow": y_raw.transpose(0, 2, 1),
            "time_feat": time_feat,
            "sample_start": np.asarray(
                [int(sample["sample_start"]) for sample in samples], dtype=np.int64
            ),
        }
        if self.include_incident:
            batch.update(self._incident(samples))
        if self.sensor_support is not None:
            batch["sensor_data"] = {
                key: np.broadcast_to(value, (len(samples), *value.shape)).copy()
                for key, value in self.sensor_support.sensor_info.items()
            }
        forbidden = {"affected_mask", "analysis_labels"}.intersection(batch)
        if forbidden:
            raise RuntimeError(f"analysis-only fields entered training batch: {forbidden}")
        return batch

    def get_iterator(self) -> Iterator[dict[str, Any]]:
        self.current_ind = 0

        def batches() -> Iterator[dict[str, Any]]:
            while self.current_ind < self.num_batch:
                start = self.current_ind * self.bs
                stop = min(start + self.bs, self.size)
                selected = self._order[start:stop]
                self.current_ind += 1
                if selected.size < self.bs and self.drop_last:
                    continue
                yield self._build_batch(selected)

        return batches()


def prepare_ndr_batch(
    batch: Mapping[str, Any],
    statistics: WeeklyClimatology,
) -> dict[str, np.ndarray]:
    """Prepare raw-unit NDR inputs while preserving the explicit v3 mask."""
    x_raw = np.asarray(batch["x_raw_flow"], dtype=np.float32)
    y_raw = np.asarray(batch["y_raw_flow"], dtype=np.float32)
    time_feat = np.asarray(batch["time_feat"], dtype=np.float32)
    x_dow, x_tod, y_dow, y_tod = _weekly_slots_from_history(time_feat)
    x_baseline = statistics.baseline[x_dow, x_tod]
    y_baseline = statistics.baseline[y_dow, y_tod]
    target_mask = np.asarray(batch["y_mask"], dtype=np.bool_)[..., 0].transpose(0, 2, 1)
    return {
        "flow_hist": x_raw.transpose(0, 2, 1),
        "x_baseline": x_baseline.transpose(0, 2, 1),
        "y_baseline": y_baseline.transpose(0, 2, 1),
        "time_feat": time_feat,
        "target": y_raw.transpose(0, 2, 1),
        "target_mask": target_mask,
    }
