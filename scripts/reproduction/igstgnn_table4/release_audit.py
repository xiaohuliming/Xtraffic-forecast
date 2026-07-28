from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format


REQUIRED_RELEASE_FILES = (
    "adj_matrix.npy",
    "desc_mapping.json",
    "incident_all.npy",
    "incident_stats.npz",
    "sensors.csv",
    "type_mapping.json",
)

RELEASE_ID = "kaggle_lixiangfan_data4igstgnn_v1"
PROTOCOL_ID = "igstgnn_table4_official_v1"
OFFICIAL_REGION_SPECS = {
    "Alameda": {
        "nodes": 521,
        "samples": 14_687,
        "description_count": 30,
        "type_count": 6,
    },
    "Contra_Costa": {
        "nodes": 496,
        "samples": 5_585,
        "description_count": 29,
        "type_count": 6,
    },
    "Orange": {
        "nodes": 990,
        "samples": 18_700,
        "description_count": 32,
        "type_count": 6,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_npy_header(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY format version: {version}")
    return {
        "shape": list(shape),
        "fortran_order": bool(fortran_order),
        "dtype": str(dtype),
    }


def build_release_file_manifest(
    data_root: Path,
    *,
    regions: Iterable[str],
) -> dict[str, Any]:
    root = Path(data_root)
    if root.is_symlink():
        raise ValueError(f"Release root must not be a symlink: {root}")
    region_manifests = {}
    for region in regions:
        region_dir = root / region
        if region_dir.is_symlink():
            raise ValueError(f"Release region must not be a symlink: {region_dir}")
        files = {}
        for filename in REQUIRED_RELEASE_FILES:
            path = region_dir / filename
            if path.is_symlink():
                raise ValueError(f"Required release file must not be a symlink: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Required release file not found: {path}")
            files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        region_manifests[region] = {"files": files}
    return {"regions": region_manifests}


def _split_counts(total_samples: int) -> dict[str, int]:
    train_count = int(total_samples * 0.70)
    val_count = int(total_samples * 0.15)
    return {
        "train": train_count,
        "val": val_count,
        "test": total_samples - train_count - val_count,
    }


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_adjacency(path: Path, *, node_count: int) -> dict[str, Any]:
    adjacency = np.load(path, allow_pickle=False)
    if adjacency.shape != (node_count, node_count):
        raise ValueError(f"Adjacency shape mismatch in {path}: {adjacency.shape!r}")
    if adjacency.dtype != np.dtype("float64"):
        raise ValueError(f"Adjacency dtype mismatch in {path}: {adjacency.dtype!s}")

    finite = bool(np.isfinite(adjacency).all())
    nonnegative = bool((adjacency >= 0).all())
    diagonal_zero = bool(np.equal(np.diag(adjacency), 0).all())
    if not finite:
        raise ValueError(f"Adjacency contains non-finite values: {path}")
    if not nonnegative:
        raise ValueError(f"Adjacency contains negative values: {path}")
    if not diagonal_zero:
        raise ValueError(f"Adjacency diagonal must be zero: {path}")

    return {
        "dtype": str(adjacency.dtype),
        "fortran_order": bool(adjacency.flags.f_contiguous),
        "finite": finite,
        "nonnegative": nonnegative,
        "diagonal_zero": diagonal_zero,
        "symmetric": bool(np.array_equal(adjacency, adjacency.T)),
        "minimum": float(adjacency.min()),
        "maximum": float(adjacency.max()),
    }


def _audit_incident_stats(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stats:
        expected_keys = {"mean", "std", "normalized"}
        if set(stats.files) != expected_keys:
            raise ValueError(
                f"incident_stats.npz keys mismatch in {path}: {sorted(stats.files)!r}"
            )
        if any(stats[key].shape != () for key in expected_keys):
            raise ValueError(f"incident_stats.npz values must be scalars: {path}")
        mean = float(stats["mean"])
        std = float(stats["std"])
        normalized = bool(stats["normalized"])

    if not np.isfinite(mean):
        raise ValueError(f"Incident mean must be finite: {path}")
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"Incident std must be finite and positive: {path}")
    if not normalized:
        raise ValueError(f"Official incident release must be normalized: {path}")
    return {"mean": mean, "std": std, "normalized": normalized}


def _load_mapping(
    path: Path,
    *,
    expected_count: int,
) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"Mapping must be a JSON object: {path}")
    if len(mapping) != expected_count:
        raise ValueError(
            f"Mapping size mismatch in {path}: "
            f"expected {expected_count}, got {len(mapping)}"
        )
    if not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in mapping.items()
    ):
        raise ValueError(f"Mapping entries must be string-to-integer: {path}")
    if sorted(mapping.values()) != list(range(expected_count)):
        raise ValueError(
            f"Mapping indices must be unique and contiguous from zero: {path}"
        )
    return mapping


def _uses_float_fallback(value: str | None) -> bool:
    if value is None or value.strip() == "":
        return True
    try:
        float(value)
    except ValueError:
        return True
    return False


def _audit_sensors(path: Path, *, node_count: int) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            "order",
            "Surface",
            "Roadway Use",
            "Road Width",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"sensors.csv missing required columns in {path}: {missing!r}"
            )
        if not ({"Sensor Type", "Type"} & fieldnames):
            raise ValueError(f"sensors.csv needs Sensor Type or Type in {path}")
        speed_columns = [
            name for name in ("Design Speed Limit", "Speed Limit") if name in fieldnames
        ]
        if not speed_columns:
            raise ValueError(f"sensors.csv needs a speed-limit column in {path}")
        rows = list(reader)

    if len(rows) != node_count:
        raise ValueError(
            f"sensors.csv row count mismatch in {path}: "
            f"expected {node_count}, got {len(rows)}"
        )
    try:
        orders = [int(row["order"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sensors.csv order values must be integers: {path}") from exc
    order_is_unique = len(set(orders)) == node_count
    if not order_is_unique:
        raise ValueError(f"sensors.csv order values must be unique: {path}")

    speed_column = speed_columns[0]
    return {
        "row_count": len(rows),
        "order_is_unique": order_is_unique,
        "road_width_fallback_count": sum(
            _uses_float_fallback(row.get("Road Width")) for row in rows
        ),
        "speed_limit_fallback_count": sum(
            _uses_float_fallback(row.get(speed_column)) for row in rows
        ),
    }


def _audit_safe_region_structure(
    region_dir: Path,
    *,
    spec: dict[str, int],
) -> dict[str, Any]:
    description_mapping = _load_mapping(
        region_dir / "desc_mapping.json",
        expected_count=spec["description_count"],
    )
    type_mapping = _load_mapping(
        region_dir / "type_mapping.json",
        expected_count=spec["type_count"],
    )
    return {
        "adjacency": _audit_adjacency(
            region_dir / "adj_matrix.npy",
            node_count=spec["nodes"],
        ),
        "incident_stats": _audit_incident_stats(region_dir / "incident_stats.npz"),
        "mappings": {
            "description_count": len(description_mapping),
            "type_count": len(type_mapping),
        },
        "sensors": _audit_sensors(
            region_dir / "sensors.csv",
            node_count=spec["nodes"],
        ),
    }


def audit_official_release(
    data_root: Path,
    *,
    write_manifest: Path | None = None,
) -> dict[str, Any]:
    file_manifest = build_release_file_manifest(
        data_root,
        regions=OFFICIAL_REGION_SPECS,
    )
    regions = {}
    for region, spec in OFFICIAL_REGION_SPECS.items():
        region_dir = Path(data_root) / region
        incident_header = read_npy_header(region_dir / "incident_all.npy")
        expected_incident_header = {
            "shape": [spec["samples"]],
            "fortran_order": False,
            "dtype": "object",
        }
        if incident_header != expected_incident_header:
            raise ValueError(
                f"{region} incident_all.npy header mismatch: {incident_header!r}"
            )

        adjacency_header = read_npy_header(region_dir / "adj_matrix.npy")
        if adjacency_header["shape"] != [spec["nodes"], spec["nodes"]]:
            raise ValueError(
                f"{region} adj_matrix.npy shape mismatch: {adjacency_header['shape']!r}"
            )

        regions[region] = {
            **file_manifest["regions"][region],
            "sample_count": spec["samples"],
            "node_count": spec["nodes"],
            "split_counts": _split_counts(spec["samples"]),
            "incident_all_header": incident_header,
            "adjacency_header": adjacency_header,
            "safe_structure": _audit_safe_region_structure(
                region_dir,
                spec=spec,
            ),
        }

    digest_payload = {
        "protocol_id": PROTOCOL_ID,
        "release_id": RELEASE_ID,
        "regions": regions,
    }
    manifest = {
        "status": "safe_structure_verified",
        **digest_payload,
        "release_digest": _canonical_sha256(digest_payload),
        "hash_authority": "locally_observed_not_publisher_signed",
    }
    if write_manifest is not None:
        output = Path(write_manifest)
        if output.exists():
            raise FileExistsError(f"Release manifest already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output.with_name(f".{output.name}.tmp")
        payload = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
    return manifest
