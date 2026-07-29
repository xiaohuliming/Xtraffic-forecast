from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .restricted_npy import load_restricted_object_npy_handle
from .secure_io import (
    open_verified_regular_file,
    verify_secure_directory,
)


_SPLITS = ("train", "val", "test")
_SPLIT_FILENAMES = {
    "train": "incident_train.npy",
    "val": "incident_val.npy",
    "test": "incident_test.npy",
}
_SUPPORT_FILENAMES = frozenset(
    {
        "adj_matrix.npy",
        "incident_stats.npz",
        "sensors.csv",
        "desc_mapping.json",
        "type_mapping.json",
    }
)
_BUNDLE_FILENAMES = frozenset(
    {*_SPLIT_FILENAMES.values(), *_SUPPORT_FILENAMES, "split_manifest.json"}
)
_RECORD_KEYS = frozenset(
    {
        "x_data",
        "y_data",
        "incident_features",
        "incident_position",
        "incident_distances",
        "durations",
    }
)
_INCIDENT_FEATURE_KEYS = (
    "Incident Time",
    "Description",
    "Type",
    "Holiday",
)
_OVERLAP_KEYS = frozenset({"train_val", "train_test", "val_test"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXPECTED_SPLIT_SEMANTICS = {
    "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "ordered": True,
    "shuffled": False,
    "renormalized": False,
    "deduplicated": False,
}
_ALLOWED_VALIDITY_LABELS = frozenset({"official_protocol_with_known_overlap"})
_SENSOR_KEYS = frozenset(
    {
        "sensor_type",
        "surface",
        "roadway_use",
        "road_width",
        "speed_limit",
    }
)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key in split manifest: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value in split manifest: {value}")


def _read_file_and_hash(path: Path) -> tuple[bytes, str]:
    with open_verified_regular_file(
        path,
        purpose="Bundle file",
        require_nonempty=True,
    ) as opened:
        payload = opened.handle.read()
    return payload, hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_plain_integer(
    value: Any,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_overlap_counts(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _OVERLAP_KEYS:
        raise ValueError(f"{field} has an invalid shape")
    return {
        key: _require_plain_integer(value[key], field=f"{field}.{key}")
        for key in sorted(_OVERLAP_KEYS)
    }


def _validate_manifest(
    manifest: Any,
) -> tuple[dict[str, int], int, int, int]:
    if not isinstance(manifest, dict):
        raise ValueError("split_manifest.json must contain a JSON object")
    exact_values = {
        "schema_version": 2,
        "protocol_id": "igstgnn_table4_official_v1",
        "status": "official_splits_prepared",
        "record_structure_status": "records_structurally_verified",
        "split_integrity_status": "cross_split_duplicates_detected",
        "support_structure_status": "safe_structure_verified",
    }
    for field, expected in exact_values.items():
        if manifest.get(field) != expected:
            raise ValueError(f"split manifest {field} must be {expected!r}")
    validity_label = manifest.get("validity_label")
    if validity_label not in _ALLOWED_VALIDITY_LABELS:
        raise ValueError(
            f"Unsupported split manifest validity_label: {validity_label!r}"
        )
    if manifest.get("split_semantics") != _EXPECTED_SPLIT_SEMANTICS:
        raise ValueError("split manifest semantics do not match Table 4")
    if manifest.get("artifact_allowlist") != sorted(_BUNDLE_FILENAMES):
        raise ValueError("split manifest artifact_allowlist is invalid")
    if manifest.get("recombined_matches_input") is not True:
        raise ValueError("split manifest does not preserve the source sequence")

    region = manifest.get("region")
    if not isinstance(region, str) or not region:
        raise ValueError("split manifest region must be a non-empty string")

    counts_value = manifest.get("counts")
    if not isinstance(counts_value, dict) or set(counts_value) != set(_SPLITS):
        raise ValueError("split manifest counts must contain train, val, and test")
    counts = {
        split: _require_plain_integer(
            counts_value[split],
            field=f"counts.{split}",
            minimum=1,
        )
        for split in _SPLITS
    }
    total = sum(counts.values())
    expected_counts = {
        "train": int(total * 0.70),
        "val": int(total * 0.15),
    }
    expected_counts["test"] = total - expected_counts["train"] - expected_counts["val"]
    if counts != expected_counts:
        raise ValueError("split manifest counts do not implement ordered 70/15/15")

    full_overlap = _validate_overlap_counts(
        manifest.get("cross_split_full_record_intersections"),
        field="cross_split_full_record_intersections",
    )
    xy_overlap = _validate_overlap_counts(
        manifest.get("cross_split_xy_window_intersections"),
        field="cross_split_xy_window_intersections",
    )
    model_overlap = _validate_overlap_counts(
        manifest.get("cross_split_model_consumed_record_intersections"),
        field="cross_split_model_consumed_record_intersections",
    )
    if not any(
        (
            *full_overlap.values(),
            *xy_overlap.values(),
            *model_overlap.values(),
        )
    ):
        raise ValueError(
            "known-overlap validity label has no recorded overlap evidence"
        )
    model_consumed_digest = _validate_sha256(
        manifest.get("model_consumed_digest"),
        field="model_consumed_digest",
    )
    duplicate_model_consumed = _require_plain_integer(
        manifest.get("duplicate_model_consumed_record_count"),
        field="duplicate_model_consumed_record_count",
    )
    multiplicity = manifest.get("model_consumed_record_multiplicity")
    if not isinstance(multiplicity, dict) or not multiplicity:
        raise ValueError("model-consumed multiplicity is missing")
    multiplicity_counts: dict[int, int] = {}
    for key, item in multiplicity.items():
        if not isinstance(key, str) or not key.isdecimal() or int(key) < 1:
            raise ValueError("model-consumed multiplicity key is invalid")
        multiplicity_counts[int(key)] = _require_plain_integer(
            item,
            field=f"model_consumed_record_multiplicity.{key}",
            minimum=1,
        )
    if (
        sum(
            multiplicity_value * count
            for multiplicity_value, count in multiplicity_counts.items()
        )
        != total
    ):
        raise ValueError("model-consumed multiplicity does not cover all records")
    if (
        sum(
            (multiplicity_value - 1) * count
            for multiplicity_value, count in multiplicity_counts.items()
        )
        != duplicate_model_consumed
    ):
        raise ValueError("model-consumed duplicate count is inconsistent")

    postflight = manifest.get("independent_postflight")
    if not isinstance(postflight, dict):
        raise ValueError("split manifest independent_postflight is missing")
    if (
        postflight.get("status") != "official_splits_independently_reverified"
        or postflight.get("counts") != counts
        or postflight.get("records_revalidated") != total
        or postflight.get("cross_split_full_record_intersections") != full_overlap
        or postflight.get("cross_split_xy_window_intersections") != xy_overlap
        or postflight.get("cross_split_model_consumed_record_intersections")
        != model_overlap
        or postflight.get("duplicate_model_consumed_record_count")
        != duplicate_model_consumed
        or postflight.get("model_consumed_record_multiplicity") != multiplicity
        or postflight.get("model_consumed_digest") != model_consumed_digest
    ):
        raise ValueError("split manifest independent postflight is inconsistent")

    safe_structure = manifest.get("safe_structure")
    if not isinstance(safe_structure, dict):
        raise ValueError("split manifest safe_structure is missing")
    mappings = safe_structure.get("mappings")
    sensors = safe_structure.get("sensors")
    if not isinstance(mappings, dict) or not isinstance(sensors, dict):
        raise ValueError("split manifest safe structure is incomplete")
    description_count = _require_plain_integer(
        mappings.get("description_count"),
        field="safe_structure.mappings.description_count",
        minimum=1,
    )
    type_count = _require_plain_integer(
        mappings.get("type_count"),
        field="safe_structure.mappings.type_count",
        minimum=1,
    )
    node_count = _require_plain_integer(
        sensors.get("row_count"),
        field="safe_structure.sensors.row_count",
        minimum=1,
    )
    if sensors.get("order_is_unique") is not True:
        raise ValueError("split manifest sensor order was not verified")

    release_binding = manifest.get("release_manifest_binding")
    if not isinstance(release_binding, dict) or release_binding.get("region") != region:
        raise ValueError("split manifest release binding is invalid")
    _validate_sha256(
        release_binding.get("sha256"),
        field="release_manifest_binding.sha256",
    )

    security = manifest.get("security_boundary")
    negative_controls = (
        security.get("negative_controls") if isinstance(security, dict) else None
    )
    if not isinstance(negative_controls, dict) or set(negative_controls) != {
        "writer",
        "postflight",
    }:
        raise ValueError("split manifest sandbox boundary was not verified")
    required_probe_flags = (
        "allowed_input_read",
        "credential_read_denied",
        "neighbor_read_denied",
        "network_denied",
        "fork_denied",
        "spawn_denied",
    )
    for profile in ("writer", "postflight"):
        probe = negative_controls[profile]
        if (
            not isinstance(probe, dict)
            or probe.get("status") != "sandbox_boundary_verified"
            or probe.get("profile") != profile
            or any(probe.get(flag) is not True for flag in required_probe_flags)
        ):
            raise ValueError(
                f"split manifest {profile} sandbox boundary was not verified"
            )
        _validate_sha256(
            probe.get("sandbox_profile_sha256"),
            field=f"security_boundary.negative_controls.{profile}."
            "sandbox_profile_sha256",
        )
        _validate_sha256(
            probe.get("probe_worker_sha256"),
            field=f"security_boundary.negative_controls.{profile}.probe_worker_sha256",
        )
    return counts, node_count, description_count, type_count


def _manifest_file_entry(
    manifest: Mapping[str, Any],
    *,
    collection: str,
    key: str,
    expected_filename: str,
) -> tuple[int, str]:
    entries = manifest.get(collection)
    if not isinstance(entries, dict):
        raise ValueError(f"split manifest {collection} is missing")
    entry = entries.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"split manifest entry is missing: {collection}.{key}")
    if Path(str(entry.get("path", ""))).name != expected_filename:
        raise ValueError(f"split manifest path does not name {expected_filename}")
    expected_size = _require_plain_integer(
        entry.get("bytes"),
        field=f"{collection}.{key}.bytes",
        minimum=1,
    )
    expected_sha256 = _validate_sha256(
        entry.get("sha256"),
        field=f"{collection}.{key}.sha256",
    )
    return expected_size, expected_sha256


def _verify_hashed_support_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    payload, observed_sha256 = _read_file_and_hash(path)
    if len(payload) != expected_size:
        raise ValueError(f"Bundle file size mismatch: {path}")
    if not hmac.compare_digest(observed_sha256, expected_sha256):
        raise ValueError(f"Bundle file SHA-256 mismatch: {path}")


def _load_hashed_split_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> np.ndarray:
    with open_verified_regular_file(
        path,
        purpose="Bundle split",
        require_nonempty=True,
    ) as opened:
        before = opened.stat_result
        handle = opened.handle
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        if before.st_size != expected_size:
            raise ValueError(f"Bundle split size mismatch: {path}")
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ValueError(f"Bundle split SHA-256 mismatch: {path}")
        handle.seek(0)
        records = load_restricted_object_npy_handle(
            handle,
            source_size=expected_size,
        )
    return records


def _numeric_array(
    value: Any,
    *,
    field: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype.kind not in "biuf"
    ):
        raise ValueError(f"{field} must be a real NumPy array of shape {shape}")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field} contains non-finite values")
    converted = np.asarray(value, dtype=np.float32)
    if not bool(np.isfinite(converted).all()):
        raise ValueError(f"{field} overflows float32")
    return value


def _validate_time_channels(
    value: np.ndarray,
    *,
    field: str,
) -> np.ndarray:
    encoded = {}
    for name, channel, period in (
        ("time-of-day", 1, 288),
        ("day-of-week", 2, 7),
    ):
        values = np.asarray(value[..., channel], dtype=np.float64)
        if not bool(((values >= 0) & (values < 1)).all()):
            raise ValueError(f"{field} {name} channel must be in [0, 1)")
        if not bool(
            np.allclose(
                values * period,
                np.rint(values * period),
                rtol=0,
                atol=2e-5,
            )
        ):
            raise ValueError(f"{field} {name} channel is not quantized to 1/{period}")
        if not bool(
            np.allclose(
                values,
                values[:, :1],
                rtol=0,
                atol=1e-7,
            )
        ):
            raise ValueError(f"{field} {name} channel differs across nodes")
        encoded[name] = np.rint(values[:, 0] * period).astype(np.int64)
    ticks = encoded["day-of-week"] * 288 + encoded["time-of-day"]
    if not bool(np.all(np.diff(ticks) % (7 * 288) == 1)):
        raise ValueError(f"{field} time channels are not consecutive")
    return ticks


def _finite_scalar(value: Any, *, field: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "biuf":
        raise ValueError(f"{field} must be a real numeric scalar")
    number = float(array)
    if not math.isfinite(number) or not math.isfinite(float(np.float32(number))):
        raise ValueError(f"{field} must be finite and float32-representable")
    return number


def _integer_scalar(value: Any, *, field: str) -> int:
    number = _finite_scalar(value, field=field)
    integer = int(number)
    if float(integer) != number:
        raise ValueError(f"{field} must be exactly integral")
    return integer


def _validate_records(
    records: np.ndarray,
    *,
    split: str,
    expected_count: int,
    node_count: int,
    description_count: int,
    type_count: int,
) -> None:
    if (
        not isinstance(records, np.ndarray)
        or records.dtype != np.dtype("object")
        or records.ndim != 1
        or len(records) != expected_count
    ):
        raise ValueError(f"{split} records do not match the split manifest")
    for index, record in enumerate(records):
        label = f"{split} record {index}"
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise ValueError(f"{label} has an invalid top-level structure")
        x_data = _numeric_array(
            record["x_data"],
            field=f"{label} x_data",
            shape=(12, node_count, 3),
        )
        y_data = _numeric_array(
            record["y_data"],
            field=f"{label} y_data",
            shape=(12, node_count, 3),
        )
        x_ticks = _validate_time_channels(
            x_data,
            field=f"{label} x_data",
        )
        y_ticks = _validate_time_channels(
            y_data,
            field=f"{label} y_data",
        )
        if int((y_ticks[0] - x_ticks[-1]) % (7 * 288)) != 1:
            raise ValueError(f"{label} x_data to y_data time discontinuity")
        _numeric_array(
            record["incident_distances"],
            field=f"{label} incident_distances",
            shape=(node_count, 3),
        )
        _numeric_array(
            record["durations"],
            field=f"{label} durations",
            shape=(12,),
        )
        features = record["incident_features"]
        if type(features) is not dict or set(features) != set(_INCIDENT_FEATURE_KEYS):
            raise ValueError(f"{label} incident_features is invalid")
        _finite_scalar(
            features["Incident Time"],
            field=f"{label} Incident Time",
        )
        description = _integer_scalar(
            features["Description"],
            field=f"{label} Description",
        )
        incident_type = _integer_scalar(
            features["Type"],
            field=f"{label} Type",
        )
        holiday = _integer_scalar(
            features["Holiday"],
            field=f"{label} Holiday",
        )
        if not 0 <= description < description_count:
            raise ValueError(f"{label} Description is out of range")
        if not 0 <= incident_type < type_count:
            raise ValueError(f"{label} Type is out of range")
        if holiday not in {0, 1}:
            raise ValueError(f"{label} Holiday must be 0 or 1")
        position = _integer_scalar(
            record["incident_position"],
            field=f"{label} incident_position",
        )
        if not 0 <= position < 12:
            raise ValueError(f"{label} incident_position is out of range")


def _validate_sensor_info(
    sensor_info: Mapping[str, Any] | None,
    *,
    node_count: int,
) -> dict[str, np.ndarray] | None:
    if sensor_info is None:
        return None
    if not isinstance(sensor_info, Mapping):
        raise ValueError("sensor_info must be a mapping or None")
    unknown = set(sensor_info) - _SENSOR_KEYS
    if unknown:
        raise ValueError(f"sensor_info has unsupported keys: {sorted(unknown)!r}")
    prepared = {}
    for key, value in sensor_info.items():
        array = np.asarray(value)
        if array.shape != (node_count,) or array.dtype.kind not in "biuf":
            raise ValueError(
                f"sensor_info.{key} must be a numeric vector of length {node_count}"
            )
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"sensor_info.{key} contains non-finite values")
        prepared[key] = np.array(array, copy=True)
    return prepared


class VerifiedSplitLoader:
    def __init__(
        self,
        records: np.ndarray,
        *,
        split: str,
        batch_size: int,
        input_dim: int,
        target_dim: int,
        include_incident: bool,
        sensor_info: dict[str, np.ndarray] | None,
        seed: int,
        loader_contract: Mapping[str, str],
    ) -> None:
        self._records = records
        self._split = split
        self._batch_size = batch_size
        self._input_dim = input_dim
        self._target_dim = target_dim
        self._include_incident = include_incident
        self._sensor_info = sensor_info
        self._seed = seed
        self.loader_contract = dict(loader_contract)
        self._indices = np.arange(len(records), dtype=np.int64)
        self.size = len(records)
        if self.loader_contract["partial_batch"] == "drop":
            self.num_batch = self.size // self._batch_size
        else:
            self.num_batch = math.ceil(self.size / self._batch_size)
        self.current_ind = 0

    def shuffle(self) -> None:
        if self._split != "train":
            raise RuntimeError("Evaluation split order is fixed and cannot shuffle")
        permutation = np.random.permutation(self.size)
        self._indices = self._indices[permutation]

    def _build_batch(self, indices: np.ndarray) -> dict[str, np.ndarray | dict]:
        records = [self._records[int(index)] for index in indices]
        batch: dict[str, Any] = {
            "x_data": np.stack(
                [
                    np.asarray(record["x_data"], dtype=np.float32)[
                        ..., : self._input_dim
                    ]
                    for record in records
                ],
                axis=0,
            ),
            "y_data": np.stack(
                [
                    np.asarray(record["y_data"], dtype=np.float32)[
                        ..., : self._target_dim
                    ]
                    for record in records
                ],
                axis=0,
            ),
            "sample_indices": np.asarray(indices, dtype=np.int64),
        }
        if self._include_incident:
            batch.update(
                {
                    "incident_features": np.asarray(
                        [
                            [
                                record["incident_features"][key]
                                for key in _INCIDENT_FEATURE_KEYS
                            ]
                            for record in records
                        ],
                        dtype=np.float32,
                    ),
                    "incident_position": np.asarray(
                        [record["incident_position"] for record in records],
                        dtype=np.int64,
                    ),
                    "incident_distances": np.stack(
                        [
                            np.asarray(
                                record["incident_distances"],
                                dtype=np.float64,
                            )
                            for record in records
                        ],
                        axis=0,
                    ),
                    "durations": np.stack(
                        [
                            np.asarray(
                                record["durations"],
                                dtype=np.float64,
                            )
                            for record in records
                        ],
                        axis=0,
                    ),
                }
            )
        if self._sensor_info is not None:
            batch_size = len(indices)
            batch["sensor_data"] = {
                key: np.broadcast_to(
                    value,
                    (batch_size, value.shape[0]),
                )
                for key, value in self._sensor_info.items()
            }
        return batch

    def get_iterator(self) -> Iterator[dict[str, Any]]:
        order = self._indices.copy()
        self.current_ind = 0

        def iterator() -> Iterator[dict[str, Any]]:
            while self.current_ind < self.num_batch:
                start = self.current_ind * self._batch_size
                end = min(start + self._batch_size, self.size)
                self.current_ind += 1
                yield self._build_batch(order[start:end])

        return iterator()


@dataclass(frozen=True)
class VerifiedSplitBundle:
    train_loader: VerifiedSplitLoader
    val_loader: VerifiedSplitLoader
    test_loader: VerifiedSplitLoader
    manifest: Mapping[str, Any]
    manifest_sha256: str
    node_count: int

    def __getitem__(self, key: str) -> VerifiedSplitLoader:
        loaders = {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "test_loader": self.test_loader,
        }
        try:
            return loaders[key]
        except KeyError as error:
            raise KeyError(key) from error


def load_verified_split_bundle(
    bundle_dir: Path,
    *,
    expected_manifest_sha256: str,
    batch_size: int,
    input_dim: int,
    include_incident: bool,
    sensor_info: Mapping[str, Any] | None,
    seed: int,
    loader_contract: Mapping[str, str],
    target_dim: int = 1,
) -> VerifiedSplitBundle:
    expected_manifest_sha256 = _validate_sha256(
        expected_manifest_sha256,
        field="expected_manifest_sha256",
    )
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if input_dim not in {1, 3}:
        raise ValueError("Table 4 incident input_dim must be 1 or 3")
    if target_dim not in {1, 3}:
        raise ValueError("Table 4 incident target_dim must be 1 or 3")
    if type(include_incident) is not bool:
        raise ValueError("include_incident must be a boolean")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(loader_contract, Mapping):
        raise TypeError("loader_contract must be a mapping")
    locked_loader_contract = dict(loader_contract)
    if set(locked_loader_contract) != {
        "training_shuffle",
        "partial_batch",
    }:
        raise ValueError("loader_contract has an invalid schema")
    if locked_loader_contract["training_shuffle"] != "global_numpy_permutation":
        raise ValueError("loader_contract training_shuffle is unsupported")
    if locked_loader_contract["partial_batch"] not in {"drop", "retain"}:
        raise ValueError("loader_contract partial_batch is unsupported")

    directory = verify_secure_directory(
        bundle_dir,
        purpose="Verified split bundle",
    )
    observed_names = {item.name for item in directory.iterdir()}
    if observed_names != _BUNDLE_FILENAMES:
        raise ValueError(
            "Verified split bundle contains missing or extra files: "
            f"{sorted(observed_names ^ _BUNDLE_FILENAMES)!r}"
        )

    manifest_payload, observed_manifest_sha256 = _read_file_and_hash(
        directory / "split_manifest.json"
    )
    if not hmac.compare_digest(
        observed_manifest_sha256,
        expected_manifest_sha256,
    ):
        raise ValueError("split_manifest.json SHA-256 mismatch")
    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("split_manifest.json is not strict UTF-8 JSON") from error
    counts, node_count, description_count, type_count = _validate_manifest(manifest)
    if set(manifest.get("files", {})) != set(_SPLITS):
        raise ValueError("split manifest files must contain exactly three splits")
    if set(manifest.get("support_files", {})) != _SUPPORT_FILENAMES:
        raise ValueError("split manifest support_files has an invalid allowlist")

    for name in sorted(_SUPPORT_FILENAMES):
        expected_size, expected_sha256 = _manifest_file_entry(
            manifest,
            collection="support_files",
            key=name,
            expected_filename=name,
        )
        _verify_hashed_support_file(
            directory / name,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    records_by_split = {}
    for split in _SPLITS:
        filename = _SPLIT_FILENAMES[split]
        expected_size, expected_sha256 = _manifest_file_entry(
            manifest,
            collection="files",
            key=split,
            expected_filename=filename,
        )
        records = _load_hashed_split_file(
            directory / filename,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        _validate_records(
            records,
            split=split,
            expected_count=counts[split],
            node_count=node_count,
            description_count=description_count,
            type_count=type_count,
        )
        records_by_split[split] = records

    if {item.name for item in directory.iterdir()} != _BUNDLE_FILENAMES:
        raise RuntimeError("Verified split bundle changed while being loaded")
    prepared_sensor_info = _validate_sensor_info(
        sensor_info,
        node_count=node_count,
    )
    loaders = {
        split: VerifiedSplitLoader(
            records_by_split[split],
            split=split,
            batch_size=batch_size,
            input_dim=input_dim,
            target_dim=target_dim,
            include_incident=include_incident,
            sensor_info=prepared_sensor_info,
            seed=seed,
            loader_contract=locked_loader_contract,
        )
        for split in _SPLITS
    }
    return VerifiedSplitBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        manifest=manifest,
        manifest_sha256=observed_manifest_sha256,
        node_count=node_count,
    )
