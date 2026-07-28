from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np


TOP_LEVEL_KEYS = frozenset(
    {
        "x_data",
        "y_data",
        "incident_features",
        "incident_position",
        "incident_distances",
        "durations",
    }
)
INCIDENT_FEATURE_KEYS = frozenset(
    {
        "Incident Time",
        "Description",
        "Type",
        "Holiday",
    }
)


def _hash_token(digest: Any, token: bytes) -> None:
    digest.update(len(token).to_bytes(8, "big"))
    digest.update(token)


def _update_value_digest(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Nested object arrays are not permitted")
        _hash_token(digest, b"ndarray")
        _hash_token(digest, value.dtype.str.encode("ascii"))
        _hash_token(
            digest,
            ",".join(str(item) for item in value.shape).encode("ascii"),
        )
        contiguous = np.ascontiguousarray(value)
        _hash_token(digest, memoryview(contiguous).cast("B"))
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("All record dictionary keys must be strings")
        _hash_token(digest, b"dict")
        for key in sorted(value):
            _hash_token(digest, key.encode("utf-8"))
            _update_value_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        _hash_token(digest, type(value).__name__.encode("ascii"))
        for item in value:
            _update_value_digest(digest, item)
        return
    if isinstance(value, (np.bool_, bool)):
        _hash_token(digest, b"bool")
        _hash_token(digest, b"1" if bool(value) else b"0")
        return
    if isinstance(value, (np.integer, int)):
        _hash_token(digest, b"int")
        _hash_token(digest, str(int(value)).encode("ascii"))
        return
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite scalar values are not permitted")
        _hash_token(digest, b"float")
        _hash_token(digest, number.hex().encode("ascii"))
        return
    if isinstance(value, str):
        _hash_token(digest, b"str")
        _hash_token(digest, value.encode("utf-8"))
        return
    if value is None:
        _hash_token(digest, b"none")
        return
    raise ValueError(
        f"Unsupported object type in incident record: {type(value).__name__}"
    )


def _value_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _update_value_digest(digest, value)
    return digest.hexdigest()


def sequence_content_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        _hash_token(digest, bytes.fromhex(_value_digest(value)))
    return digest.hexdigest()


def _require_numeric_array(
    value: Any,
    *,
    field: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(
            f"{field} shape mismatch: expected {shape!r}, got {value.shape!r}"
        )
    if value.dtype.kind not in "biuf":
        raise ValueError(f"{field} must have a real numeric dtype")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field} contains non-finite values")
    if not bool(np.isfinite(value.astype(np.float32)).all()):
        raise ValueError(f"{field} overflows the official float32 conversion")
    return value


def _require_finite_numeric_value(value: Any, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{field} must contain real numeric values")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{field} contains non-finite values")
    if not bool(np.isfinite(array.astype(np.float32)).all()):
        raise ValueError(f"{field} overflows the official float32 conversion")
    return array


def _finite_scalar(value: Any, *, field: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "biuf":
        raise ValueError(f"{field} must be a real numeric scalar")
    number = float(array)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if not math.isfinite(float(np.float32(number))):
        raise ValueError(f"{field} overflows the official float32 conversion")
    return number


def _integer_scalar(value: Any, *, field: str) -> int:
    number = _finite_scalar(value, field=field)
    integer = int(number)
    if float(integer) != number:
        raise ValueError(f"{field} must be exactly integral")
    return integer


def _validate_time_channels(data: np.ndarray, *, field: str) -> np.ndarray:
    time_of_day = np.asarray(data[..., 1], dtype=np.float64)
    day_of_week = np.asarray(data[..., 2], dtype=np.float64)
    for name, values, periods in (
        ("time-of-day", time_of_day, 288),
        ("day-of-week", day_of_week, 7),
    ):
        if not bool(((values >= 0) & (values < 1)).all()):
            raise ValueError(f"{field} {name} channel must be in [0, 1)")
        if not bool(
            np.allclose(
                values * periods,
                np.rint(values * periods),
                rtol=0,
                atol=2e-5,
            )
        ):
            raise ValueError(f"{field} {name} channel is not quantized to 1/{periods}")
        if not bool(
            np.allclose(
                values,
                values[:, :1],
                rtol=0,
                atol=1e-7,
            )
        ):
            raise ValueError(f"{field} {name} channel differs across nodes")
    time_indices = np.rint(time_of_day[:, 0] * 288).astype(np.int64)
    day_indices = np.rint(day_of_week[:, 0] * 7).astype(np.int64)
    ticks = day_indices * 288 + time_indices
    if not bool(np.all(np.diff(ticks) % (7 * 288) == 1)):
        raise ValueError(f"{field} time channels are not consecutive")
    return ticks


def _official_split_name(index: int, sample_count: int) -> str:
    train_end = int(sample_count * 0.70)
    val_end = train_end + int(sample_count * 0.15)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def _intersection_counts(
    values: dict[str, set[str]],
) -> dict[str, int]:
    return {
        "train_val": len(values["train"] & values["val"]),
        "train_test": len(values["train"] & values["test"]),
        "val_test": len(values["val"] & values["test"]),
    }


def audit_loaded_samples(
    samples: np.ndarray,
    *,
    region: str,
    node_count: int,
    expected_sample_count: int,
    description_count: int,
    type_count: int,
) -> dict[str, Any]:
    if not isinstance(samples, np.ndarray):
        raise ValueError("incident_all.npy must load as a NumPy array")
    if samples.dtype != np.dtype("object") or samples.ndim != 1:
        raise ValueError("incident_all.npy must be a one-dimensional object array")
    if len(samples) != expected_sample_count:
        raise ValueError(
            f"Sample count mismatch: expected {expected_sample_count}, "
            f"got {len(samples)}"
        )
    if expected_sample_count <= 0:
        raise ValueError("The incident release must not be empty")

    x_dtypes: Counter[str] = Counter()
    y_dtypes: Counter[str] = Counter()
    distance_dtypes: Counter[str] = Counter()
    duration_dtypes: Counter[str] = Counter()
    duration_shapes: Counter[str] = Counter()
    feature_layouts: Counter[str] = Counter()
    description_values = []
    type_values = []
    holiday_values = []
    incident_positions = []
    full_counts: Counter[str] = Counter()
    xy_counts: Counter[str] = Counter()
    model_consumed_counts: Counter[str] = Counter()
    full_by_split = {"train": set(), "val": set(), "test": set()}
    xy_by_split = {"train": set(), "val": set(), "test": set()}
    model_consumed_by_split = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    duplicate_full_records = 0
    duplicate_xy_windows = 0
    duplicate_model_consumed_records = 0
    content_digest = hashlib.sha256()
    model_consumed_digest = hashlib.sha256()
    all_zero_distance_records = 0

    for index, sample in enumerate(samples):
        if type(sample) is not dict:
            raise ValueError(f"Record {index} must be an exact dict")
        if set(sample) != TOP_LEVEL_KEYS:
            raise ValueError(
                f"Record {index} top-level keys mismatch: {sorted(sample)!r}"
            )

        x_data = _require_numeric_array(
            sample["x_data"],
            field=f"record {index} x_data",
            shape=(12, node_count, 3),
        )
        y_data = _require_numeric_array(
            sample["y_data"],
            field=f"record {index} y_data",
            shape=(12, node_count, 3),
        )
        distances = _require_numeric_array(
            sample["incident_distances"],
            field=f"record {index} incident_distances",
            shape=(node_count, 3),
        )
        x_ticks = _validate_time_channels(x_data, field="x_data")
        y_ticks = _validate_time_channels(y_data, field="y_data")
        if int((y_ticks[0] - x_ticks[-1]) % (7 * 288)) != 1:
            raise ValueError("x_data to y_data time discontinuity")
        x_dtypes[str(x_data.dtype)] += 1
        y_dtypes[str(y_data.dtype)] += 1
        distance_dtypes[str(distances.dtype)] += 1
        if not bool(np.any(distances)):
            all_zero_distance_records += 1
        durations = _require_finite_numeric_value(
            sample["durations"],
            field=f"record {index} durations",
        )
        if durations.shape != (12,):
            raise ValueError(
                f"record {index} durations shape mismatch: "
                f"expected {(12,)!r}, got {durations.shape!r}"
            )
        duration_dtypes[str(durations.dtype)] += 1
        duration_shape = json.dumps(
            list(durations.shape),
            separators=(",", ":"),
        )
        duration_shapes[duration_shape] += 1

        features = sample["incident_features"]
        if type(features) is not dict:
            raise ValueError(f"Record {index} incident_features must be an exact dict")
        if set(features) != INCIDENT_FEATURE_KEYS:
            raise ValueError(
                f"Record {index} incident feature keys mismatch: {sorted(features)!r}"
            )
        feature_layouts["dict"] += 1
        _finite_scalar(
            features["Incident Time"],
            field=f"record {index} Incident Time",
        )
        description = _integer_scalar(
            features["Description"],
            field=f"record {index} Description",
        )
        incident_type = _integer_scalar(
            features["Type"],
            field=f"record {index} Type",
        )
        holiday = _integer_scalar(
            features["Holiday"],
            field=f"record {index} Holiday",
        )
        if not 0 <= description < description_count:
            raise ValueError(f"Record {index} Description is out of range")
        if not 0 <= incident_type < type_count:
            raise ValueError(f"Record {index} Type is out of range")
        if holiday not in {0, 1}:
            raise ValueError(f"Record {index} Holiday must be 0 or 1")
        description_values.append(description)
        type_values.append(incident_type)
        holiday_values.append(holiday)

        position = _integer_scalar(
            sample["incident_position"],
            field=f"record {index} incident_position",
        )
        if not 0 <= position < 12:
            raise ValueError(f"Record {index} incident_position is out of range")
        incident_positions.append(position)

        full_record_digest = _value_digest(sample)
        xy_digest = _value_digest([x_data, y_data])
        model_record_digest = _value_digest(
            {
                "x_data": np.asarray(x_data, dtype=np.float32),
                "y_data": np.asarray(y_data, dtype=np.float32)[..., :1],
                "incident_features": np.asarray(
                    [
                        features["Incident Time"],
                        features["Description"],
                        features["Type"],
                        features["Holiday"],
                    ],
                    dtype=np.float32,
                ),
                "incident_position": np.asarray(position, dtype=np.int64),
                "incident_distances": np.asarray(
                    distances,
                    dtype=np.float32,
                ),
            }
        )
        if full_counts[full_record_digest]:
            duplicate_full_records += 1
        if xy_counts[xy_digest]:
            duplicate_xy_windows += 1
        if model_consumed_counts[model_record_digest]:
            duplicate_model_consumed_records += 1
        full_counts[full_record_digest] += 1
        xy_counts[xy_digest] += 1
        model_consumed_counts[model_record_digest] += 1
        split_name = _official_split_name(index, expected_sample_count)
        full_by_split[split_name].add(full_record_digest)
        xy_by_split[split_name].add(xy_digest)
        model_consumed_by_split[split_name].add(model_record_digest)
        _hash_token(content_digest, bytes.fromhex(full_record_digest))
        _hash_token(
            model_consumed_digest,
            bytes.fromhex(model_record_digest),
        )

    cross_split_full = _intersection_counts(full_by_split)
    cross_split_xy = _intersection_counts(xy_by_split)
    cross_split_model_consumed = _intersection_counts(model_consumed_by_split)
    has_cross_split_overlap = any(
        count > 0
        for counts in (
            cross_split_full,
            cross_split_xy,
            cross_split_model_consumed,
        )
        for count in counts.values()
    )
    return {
        "status": "records_structurally_verified",
        "split_integrity_status": (
            "cross_split_duplicates_detected"
            if has_cross_split_overlap
            else "no_cross_split_duplicates_detected"
        ),
        "validity_label": (
            "official_protocol_with_known_overlap"
            if has_cross_split_overlap
            else "official_protocol_without_detected_overlap"
        ),
        "region": region,
        "sample_count": len(samples),
        "records_validated": len(samples),
        "top_level_keys": sorted(TOP_LEVEL_KEYS),
        "x_data": {
            "shape": [12, node_count, 3],
            "dtypes": dict(sorted(x_dtypes.items())),
            "finite": True,
        },
        "y_data": {
            "stored_shape": [12, node_count, 3],
            "consumed_shape": [12, node_count, 1],
            "discarded_feature_count": 2,
            "dtypes": dict(sorted(y_dtypes.items())),
            "finite": True,
        },
        "incident_features": {
            "layouts": dict(sorted(feature_layouts.items())),
            "description_range": [
                min(description_values),
                max(description_values),
            ],
            "type_range": [min(type_values), max(type_values)],
            "holiday_range": [min(holiday_values), max(holiday_values)],
        },
        "incident_position_range": [
            min(incident_positions),
            max(incident_positions),
        ],
        "incident_distances": {
            "shape": [node_count, 3],
            "dtypes": dict(sorted(distance_dtypes.items())),
            "finite": True,
        },
        "durations": {
            "consumed_by_official_model": False,
            "shapes": dict(sorted(duration_shapes.items())),
            "dtypes": dict(sorted(duration_dtypes.items())),
            "finite": True,
        },
        "all_zero_distance_record_count": all_zero_distance_records,
        "duplicate_full_record_count": duplicate_full_records,
        "duplicate_xy_window_count": duplicate_xy_windows,
        "duplicate_model_consumed_record_count": (duplicate_model_consumed_records),
        "full_record_multiplicity": {
            str(multiplicity): count
            for multiplicity, count in sorted(Counter(full_counts.values()).items())
        },
        "xy_window_multiplicity": {
            str(multiplicity): count
            for multiplicity, count in sorted(Counter(xy_counts.values()).items())
        },
        "model_consumed_record_multiplicity": {
            str(multiplicity): count
            for multiplicity, count in sorted(
                Counter(model_consumed_counts.values()).items()
            )
        },
        "cross_split_full_record_intersections": cross_split_full,
        "cross_split_xy_window_intersections": cross_split_xy,
        "cross_split_model_consumed_record_intersections": (cross_split_model_consumed),
        "content_digest": content_digest.hexdigest(),
        "model_consumed_digest": model_consumed_digest.hexdigest(),
    }
