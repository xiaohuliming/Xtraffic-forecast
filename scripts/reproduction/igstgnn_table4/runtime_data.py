from __future__ import annotations

import csv
import hashlib
import hmac
import io
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .secure_io import read_verified_regular_file


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    if (expected_bytes is None) != (expected_sha256 is None):
        raise ValueError("Runtime support binding requires both bytes and SHA-256")
    payload = read_verified_regular_file(
        path,
        purpose="Runtime support file",
        max_bytes=max_bytes,
        require_nonempty=True,
    )
    if expected_bytes is not None:
        if (
            type(expected_bytes) is not int
            or expected_bytes <= 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("Runtime support split manifest binding is invalid")
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if len(payload) != expected_bytes or not hmac.compare_digest(
            observed_sha256,
            expected_sha256,
        ):
            raise ValueError("Runtime support file does not match the split manifest")
    return payload


def _clean_text(value: Any) -> str:
    if value is None:
        return "Unknown"
    cleaned = str(value).strip()
    return cleaned if cleaned else "Unknown"


def _first_column(
    row: Mapping[str, Any],
    candidates: tuple[str, ...],
    default: Any,
) -> Any:
    for name in candidates:
        if name in row:
            return row[name]
    return default


def _float_value(value: Any, default: float) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except ValueError:
        return default


def _encode_categories(values: list[Any]) -> tuple[np.ndarray, int]:
    mapping: dict[str, int] = {}
    encoded = []
    for value in values:
        key = _clean_text(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded.append(mapping[key])
    return np.asarray(encoded, dtype=np.int64), max(len(mapping), 1)


def load_official_sensor_info(
    path: Path | str,
    *,
    node_count: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], SimpleNamespace]:
    payload = _read_regular_file(
        Path(path),
        max_bytes=4 * 1024 * 1024,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    text = payload.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if len(rows) != node_count:
        raise ValueError(f"sensors.csv has {len(rows)} rows, expected {node_count}")
    sensor_type, sensor_type_size = _encode_categories(
        [_first_column(row, ("Sensor Type", "Type"), "Unknown") for row in rows]
    )
    surface, surface_size = _encode_categories(
        [_first_column(row, ("Surface",), "Unknown") for row in rows]
    )
    roadway_use, roadway_use_size = _encode_categories(
        [_first_column(row, ("Roadway Use",), "Unknown") for row in rows]
    )
    road_width = np.asarray(
        [
            _float_value(
                _first_column(row, ("Road Width",), 0.0),
                0.0,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    speed_limit = np.asarray(
        [
            _float_value(
                _first_column(
                    row,
                    ("Design Speed Limit", "Speed Limit"),
                    50.0,
                ),
                50.0,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    return (
        {
            "sensor_type": sensor_type,
            "surface": surface,
            "roadway_use": roadway_use,
            "road_width": road_width,
            "speed_limit": speed_limit,
        },
        SimpleNamespace(
            sensor_type_size=sensor_type_size,
            surface_size=surface_size,
            roadway_use_size=roadway_use_size,
        ),
    )


def load_official_incident_stats(
    path: Path | str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _read_regular_file(
        Path(path),
        max_bytes=1024 * 1024,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "std", "normalized"}:
            raise ValueError("incident_stats.npz keys are invalid")
        mean = np.asarray(archive["mean"])
        std = np.asarray(archive["std"])
        normalized = np.asarray(archive["normalized"])
    if (
        mean.shape != ()
        or std.shape != ()
        or normalized.shape != ()
        or mean.dtype.kind not in "f"
        or std.dtype.kind not in "f"
        or normalized.dtype.kind != "b"
        or not np.isfinite(mean)
        or not np.isfinite(std)
        or float(std) <= 0
    ):
        raise ValueError("incident_stats.npz values are invalid")
    return {
        "mean": mean[()],
        "std": std[()],
        "normalized": bool(normalized),
    }


def load_official_adjacency(
    path: Path | str,
    *,
    node_count: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> np.ndarray:
    if isinstance(node_count, bool) or not isinstance(node_count, int):
        raise TypeError("node_count must be an integer")
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    payload = _read_regular_file(
        Path(path),
        max_bytes=16 * 1024 * 1024,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (node_count, node_count)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
    ):
        raise ValueError("adj_matrix.npy is invalid")
    return np.array(value, copy=True)


class IncidentStandardScaler:
    def __init__(self, *, mean: Any, std: Any) -> None:
        mean_value = np.asarray(mean)
        std_value = np.asarray(std)
        if (
            mean_value.shape != ()
            or std_value.shape != ()
            or mean_value.dtype.kind not in "fiu"
            or std_value.dtype.kind not in "fiu"
            or not np.isfinite(mean_value)
            or not np.isfinite(std_value)
            or float(std_value) <= 0
        ):
            raise ValueError("Incident scaler statistics are invalid")
        self.mean = torch.as_tensor(mean_value, dtype=torch.float32)
        self.std = torch.as_tensor(std_value, dtype=torch.float32)

    def _for(self, value: Any) -> tuple[Any, Any]:
        if torch.is_tensor(value):
            return (
                self.mean.to(device=value.device),
                self.std.to(device=value.device),
            )
        return self.mean.numpy(), self.std.numpy()

    def transform(self, value: Any) -> Any:
        mean, std = self._for(value)
        return (value - mean) / std

    def inverse_transform(self, value: Any) -> Any:
        mean, std = self._for(value)
        return value * std + mean


class TupleLoaderView:
    def __init__(self, source: Any) -> None:
        self._source = source

    @property
    def size(self) -> int:
        return int(self._source.size)

    @property
    def num_batch(self) -> int:
        return int(self._source.num_batch)

    @property
    def current_ind(self) -> int:
        return int(self._source.current_ind)

    def shuffle(self) -> None:
        self._source.shuffle()

    def get_iterator(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        iterator = self._source.get_iterator()

        def batches() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for batch in iterator:
                yield batch["x_data"], batch["y_data"]

        return batches()
