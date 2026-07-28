from __future__ import annotations

import csv
import hashlib
import hmac
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .manifest import METHODS, REGIONS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "paper_table4.reference_v1.csv"
)
DEFAULT_SOURCE_LOCK_PATH = DEFAULT_REFERENCE_PATH.with_name("paper_source.lock.json")
METRIC_COLUMNS = (
    "h3_mae",
    "h3_rmse",
    "h3_mape_percent",
    "h6_mae",
    "h6_rmse",
    "h6_mape_percent",
    "h12_mae",
    "h12_rmse",
    "h12_mape_percent",
    "avg_mae",
    "avg_rmse",
    "avg_mape_percent",
)
EXPECTED_HEADER = ("region", "method", *METRIC_COLUMNS)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate paper source lock key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite paper source lock value: {value}")


def load_paper_table4_reference(
    path: Path | str = DEFAULT_REFERENCE_PATH,
    *,
    source_lock_path: Path | str = DEFAULT_SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Paper Table 4 reference not found: {source}")
    raw_bytes = source.read_bytes()
    source_lock_file = Path(source_lock_path)
    if source_lock_file.is_symlink() or not source_lock_file.is_file():
        raise FileNotFoundError(f"Paper source lock not found: {source_lock_file}")
    source_lock_bytes = source_lock_file.read_bytes()
    source_lock = json.loads(
        source_lock_bytes.decode("utf-8"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    if (
        source_lock.get("schema_version") != 1
        or source_lock.get("arxiv_id") != "2602.02528"
        or source_lock.get("arxiv_version") != 2
        or source_lock.get("table_number") != 4
        or source_lock.get("reference_csv") != source.name
        or source_lock.get("reference_csv_bytes") != len(raw_bytes)
        or not hmac.compare_digest(
            str(source_lock.get("reference_csv_sha256")),
            _sha256_bytes(raw_bytes),
        )
    ):
        raise ValueError("Paper source lock does not bind the reference CSV")
    text = raw_bytes.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
        raise ValueError("Paper Table 4 reference header mismatch")

    rows: list[dict[str, Any]] = []
    for row_number, raw_row in enumerate(reader, start=2):
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise ValueError(f"Malformed Paper Table 4 reference row {row_number}")
        parsed: dict[str, Any] = {
            "region": raw_row["region"],
            "method": raw_row["method"],
        }
        for column in METRIC_COLUMNS:
            token = raw_row[column]
            try:
                value = Decimal(token)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid {column} at row {row_number}") from exc
            if value.as_tuple().exponent != -2 or not value.is_finite():
                raise ValueError(f"{column} must be a finite two-decimal value")
            if value <= 0:
                raise ValueError(f"{column} must be positive")
            parsed[column] = float(value)
        rows.append(parsed)

    expected_order = [(region, method) for region in REGIONS for method in METHODS]
    observed_order = [(row["region"], row["method"]) for row in rows]
    if observed_order != expected_order:
        raise ValueError(
            "Paper Table 4 reference rows must match the ordered 3 by 13 matrix"
        )

    return {
        "schema_version": 1,
        "status": "external_paper_reference_verified",
        "title": "Incident-Guided Spatiotemporal Traffic Forecasting",
        "table": 4,
        "arxiv_id": "2602.02528",
        "doi": "10.1145/3770854.3780215",
        "source_file": str(source.resolve()),
        "source_file_sha256": _sha256_bytes(raw_bytes),
        "source_lock_file": str(source_lock_file.resolve()),
        "source_lock_sha256": _sha256_bytes(source_lock_bytes),
        "source_lock": source_lock,
        "row_count": len(rows),
        "rows": rows,
    }
