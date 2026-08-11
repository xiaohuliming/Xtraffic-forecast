#!/usr/bin/env python3
"""Append verified v3 BiST-NDR results to the immutable 42-row table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import uuid
from typing import Mapping, Sequence

from scripts.protocol_v3_event_baseline_report import COLUMNS
from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (
    CANONICAL_MANIFEST,
    load_bist_ndr_manifest,
)


EXPECTED_PROTOCOL = "protocol_v3_noleak_dedup"
EXPECTED_BASE_STATUS = "verified_no_leak_comparison_generated"
EXPECTED_BASE_VALIDITY = "target_disjoint_event_no_leak_code_stratified"
EXPECTED_BASE_MANIFEST_SHA256 = (
    "03796d1c63fcc571e80fc0bda794b6ee146090af3ef1d256e35bdd5bd02bb5ea"
)
EXPECTED_BASE_CSV_SHA256 = (
    "9817f5086f836f17265d87071271cc3b724b62420a2db2f2f4f5401e2f37651c"
)
REGION_LABELS = {
    "Alameda": "Alameda",
    "ContraCosta": "Contra Costa",
    "Orange": "Orange",
}


class BistNdrSupplementError(RuntimeError):
    """Raised when the BiST-NDR supplement cannot be proven read-only."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise BistNdrSupplementError(f"{label} is missing or a symlink")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise BistNdrSupplementError(
                    f"duplicate JSON key in {label}: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                BistNdrSupplementError(
                    f"non-finite JSON value in {label}: {item}"
                )
            ),
        )
    except BistNdrSupplementError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BistNdrSupplementError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise BistNdrSupplementError(f"{label} must contain an object")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise BistNdrSupplementError(f"cannot hash non-regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BistNdrSupplementError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise BistNdrSupplementError(f"{name} must be a nonempty string")
    return value


def _required_sha(payload: Mapping[str, object], name: str, length: int) -> str:
    value = _required_string(payload, name).lower()
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise BistNdrSupplementError(f"{name} is not a valid SHA")
    return value


def _number(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BistNdrSupplementError(f"summary metric {name} is missing")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise BistNdrSupplementError(f"summary metric {name} is invalid")
    return result


def _metric_array(payload: Mapping[str, object], name: str) -> list[float]:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != 12:
        raise BistNdrSupplementError(f"summary metric {name} must have 12 values")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise BistNdrSupplementError(f"summary metric {name} is invalid")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise BistNdrSupplementError(f"summary metric {name} is invalid")
        result.append(number)
    return result


def _safe_published_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise BistNdrSupplementError("published file path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise BistNdrSupplementError(f"unsafe published file path: {value}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise BistNdrSupplementError(f"published file is missing: {path}")
    return path


def _load_base_table(
    csv_path: Path, manifest_path: Path
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if _sha256(manifest_path) != EXPECTED_BASE_MANIFEST_SHA256:
        raise BistNdrSupplementError("base manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path, "base comparison manifest")
    expected_identity = {
        "status": EXPECTED_BASE_STATUS,
        "validity_label": EXPECTED_BASE_VALIDITY,
        "protocol_version": EXPECTED_PROTOCOL,
        "row_count": 42,
        "task_count": 42,
        "seed_policy": "single_seed_42",
        "epoch_cap": 30,
    }
    for name, expected in expected_identity.items():
        if manifest.get(name) != expected:
            raise BistNdrSupplementError(f"base manifest {name} mismatch")
    if _sha256(csv_path) != EXPECTED_BASE_CSV_SHA256:
        raise BistNdrSupplementError("base table.csv SHA-256 mismatch")
    outputs = manifest.get("outputs")
    csv_proof = outputs.get("table.csv") if isinstance(outputs, dict) else None
    if not isinstance(csv_proof, dict) or csv_proof.get("sha256") != EXPECTED_BASE_CSV_SHA256:
        raise BistNdrSupplementError("base manifest does not bind table.csv")
    try:
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise BistNdrSupplementError("base table columns mismatch")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise BistNdrSupplementError("cannot read base table.csv") from exc
    if len(rows) != 42:
        raise BistNdrSupplementError("base table must contain exactly 42 rows")
    return rows, manifest


def _verify_export(
    *,
    report_path: Path,
    expected_report_sha256: str,
    published_root: Path,
    base_manifest: Mapping[str, object],
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if _sha256(report_path) != expected_report_sha256.lower():
        raise BistNdrSupplementError("postflight SHA-256 mismatch")
    report = _load_json(report_path, "BiST-NDR postflight")
    expected = {
        "status": "verified",
        "runner_kind": "event_bist_ndr",
        "protocol_version": EXPECTED_PROTOCOL,
        "cache_schema_version": 3,
        "validated_tasks": 3,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise BistNdrSupplementError(f"postflight {name} mismatch")
    for name in ("cache_bundle_id", "cache_bundle_digest"):
        if report.get(name) != base_manifest.get(name):
            raise BistNdrSupplementError(f"postflight {name} mismatch")
    if report.get("canonical_manifest_sha256") != _sha256(CANONICAL_MANIFEST):
        raise BistNdrSupplementError("BiST-NDR manifest identity mismatch")
    if report.get("task_state_counts") != {"completed": 3}:
        raise BistNdrSupplementError("all BiST-NDR tasks must be newly completed")

    raw_files = report.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BistNdrSupplementError("postflight file list is invalid")
    if report.get("file_count") != len(raw_files):
        raise BistNdrSupplementError("postflight file_count mismatch")
    if report.get("files_digest") != hashlib.sha256(_canonical_json(raw_files)).hexdigest():
        raise BistNdrSupplementError("postflight files_digest mismatch")
    records: dict[str, Mapping[str, object]] = {}
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise BistNdrSupplementError("postflight file record is invalid")
        relative = _required_string(raw, "path")
        if relative in records:
            raise BistNdrSupplementError(f"duplicate postflight file: {relative}")
        path = _safe_published_path(published_root, relative)
        expected_size = raw.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise BistNdrSupplementError(f"invalid file size: {relative}")
        if path.stat().st_size != expected_size:
            raise BistNdrSupplementError(f"published file size mismatch: {relative}")
        if _sha256(path) != _required_sha(raw, "sha256", 64):
            raise BistNdrSupplementError(f"published file SHA-256 mismatch: {relative}")
        records[relative] = raw
        total_bytes += expected_size
    if report.get("total_bytes") != total_bytes:
        raise BistNdrSupplementError("postflight total_bytes mismatch")

    manifest = load_bist_ndr_manifest(CANONICAL_MANIFEST)
    raw_tasks = report.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 3:
        raise BistNdrSupplementError("postflight task list is invalid")
    task_proofs: dict[str, Mapping[str, object]] = {}
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise BistNdrSupplementError("postflight task proof is invalid")
        task_id = _required_string(item, "task_id")
        if task_id in task_proofs:
            raise BistNdrSupplementError(f"duplicate postflight task: {task_id}")
        if item.get("state") != "completed":
            raise BistNdrSupplementError(f"task was not completed: {task_id}")
        task_proofs[task_id] = item
    if set(task_proofs) != {task.task_id for task in manifest.tasks}:
        raise BistNdrSupplementError("postflight tasks differ from the manifest")

    rows: dict[str, dict[str, str]] = {}
    for task in manifest.tasks:
        proof = task_proofs[task.task_id]
        source_id = _required_string(proof, "source_id")
        code_sha = _required_sha(proof, "code_sha", 40)
        config_hash = _required_sha(proof, "config_hash", 64)
        relative = (
            f"sources/{source_id}/results/{task.output_dir.as_posix()}/summary.json"
        )
        record = records.get(relative)
        if record is None:
            raise BistNdrSupplementError(f"summary is absent from allowlist: {task.task_id}")
        summary = _load_json(
            _safe_published_path(published_root, relative),
            f"{task.task_id} summary",
        )
        identities = {
            "task_id": task.task_id,
            "region": task.region,
            "model": "bist_ndr",
            "variant": "v3_event_bist_ndr",
            "seed": 42,
            "protocol_version": EXPECTED_PROTOCOL,
            "sampling_protocol": "event",
            "cache_schema_version": 3,
            "cache_bundle_id": report["cache_bundle_id"],
            "cache_bundle_digest": report["cache_bundle_digest"],
            "code_sha": code_sha,
            "config_hash": config_hash,
        }
        for name, value in identities.items():
            if summary.get(name) != value:
                raise BistNdrSupplementError(
                    f"summary identity mismatch for {task.task_id}: {name}"
                )
        mae_h = _metric_array(summary, "mae_h")
        rmse_h = _metric_array(summary, "rmse_h")
        mape_h = _metric_array(summary, "mape_h")
        for name, values, index in (
            ("h3", mae_h, 2),
            ("h6", mae_h, 5),
            ("h12", mae_h, 11),
        ):
            if not math.isclose(_number(summary, name), values[index], rel_tol=0, abs_tol=1e-8):
                raise BistNdrSupplementError(
                    f"summary horizon metric mismatch: {task.task_id}/{name}"
                )
        rows[task.region] = {
            "region": REGION_LABELS[task.region],
            "method": "BiST-NDR",
            "h3_mae": f"{mae_h[2]:.6f}",
            "h3_rmse": f"{rmse_h[2]:.6f}",
            "h3_mape_percent": f"{mape_h[2]:.6f}",
            "h6_mae": f"{mae_h[5]:.6f}",
            "h6_rmse": f"{rmse_h[5]:.6f}",
            "h6_mape_percent": f"{mape_h[5]:.6f}",
            "h12_mae": f"{mae_h[11]:.6f}",
            "h12_rmse": f"{rmse_h[11]:.6f}",
            "h12_mape_percent": f"{mape_h[11]:.6f}",
            "avg_mae": f"{_number(summary, 'all'):.6f}",
            "avg_rmse": f"{_number(summary, 'rmse'):.6f}",
            "avg_mape_percent": f"{_number(summary, 'mape'):.6f}",
        }
    return rows, report


def _csv_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _markdown_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    labels = [
        "Data", "Method", "H3 MAE", "H3 RMSE", "H3 MAPE(%)",
        "H6 MAE", "H6 RMSE", "H6 MAPE(%)", "H12 MAE", "H12 RMSE",
        "H12 MAPE(%)", "Avg MAE", "Avg RMSE", "Avg MAPE(%)",
    ]
    lines = [
        "# Protocol v3 no-leak event comparison with BiST-NDR",
        "",
        "This 45-row supplement preserves the verified 42-row table and adds three seed-42 BiST-NDR runs under the same no-leak data, split, mask, metric and 30-epoch cap.",
        "",
        "The BiST-NDR rows are host- and code-stratified. Treat this as a single-seed descriptive comparison.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    lines.extend("| " + " | ".join(row[column] for column in COLUMNS) + " |" for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _latex_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        r"\begin{tabular}{llrrrrrrrrrrrr}",
        r"\toprule",
        r"Data & Method & \multicolumn{3}{c}{Horizon 3} & \multicolumn{3}{c}{Horizon 6} & \multicolumn{3}{c}{Horizon 12} & \multicolumn{3}{c}{Average} \\",
        r" & & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        region = row["region"] if row["region"] != previous else ""
        values = [region, row["method"], *(row[column] for column in COLUMNS[2:])]
        lines.append(" & ".join(value.replace("&", r"\&") for value in values) + r" \\")
        previous = row["region"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BistNdrSupplementError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise BistNdrSupplementError(f"output appeared during write: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_bist_ndr_supplement(
    *,
    base_csv_path: Path | str,
    base_manifest_path: Path | str,
    postflight_path: Path | str,
    postflight_sha256: str,
    published_root: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    base_csv = Path(base_csv_path).expanduser().resolve(strict=True)
    base_manifest_path = Path(base_manifest_path).expanduser().resolve(strict=True)
    base_rows, base_manifest = _load_base_table(base_csv, base_manifest_path)
    report_path = Path(postflight_path).expanduser().resolve(strict=True)
    root = Path(published_root).expanduser().resolve(strict=True)
    additions, postflight = _verify_export(
        report_path=report_path,
        expected_report_sha256=postflight_sha256,
        published_root=root,
        base_manifest=base_manifest,
    )
    rows: list[dict[str, str]] = []
    for region_key, region_label in REGION_LABELS.items():
        regional = [row for row in base_rows if row["region"] == region_label]
        if len(regional) != 14:
            raise BistNdrSupplementError(f"base region row count mismatch: {region_key}")
        rows.extend(regional)
        rows.append(additions[region_key])
    if len(rows) != 45:
        raise BistNdrSupplementError("supplement must contain exactly 45 rows")
    outputs = {
        "table.csv": _csv_payload(rows),
        "table.md": _markdown_payload(rows),
        "table.tex": _latex_payload(rows),
    }
    output = Path(output_dir).expanduser().resolve(strict=False)
    for name, payload in outputs.items():
        _atomic_create(output / name, payload)
    source_code_shas = sorted(
        {
            str(value["code_sha"])
            for value in base_manifest["source_indexes"].values()  # type: ignore[union-attr]
        }
        | {_required_sha(item, "code_sha", 40) for item in postflight["tasks"]}  # type: ignore[index]
    )
    manifest: dict[str, object] = {
        "report_schema_version": 1,
        "status": "verified_no_leak_bist_ndr_supplement_generated",
        "protocol_version": EXPECTED_PROTOCOL,
        "validity_label": "target_disjoint_event_no_leak_host_and_code_stratified",
        "execution_layout": "verified_42_plus_rtx5080_bist_ndr_supplement",
        "seed_policy": "single_seed_42",
        "epoch_cap": 30,
        "training_control_scope": base_manifest["training_control_scope"],
        "row_count": 45,
        "task_count": 45,
        "cache_bundle_id": base_manifest["cache_bundle_id"],
        "cache_bundle_digest": base_manifest["cache_bundle_digest"],
        "base_table": {
            "manifest_path": str(base_manifest_path),
            "manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
            "csv_path": str(base_csv),
            "csv_sha256": EXPECTED_BASE_CSV_SHA256,
            "row_count": 42,
        },
        "bist_ndr_source": {
            "matrix_id": postflight["matrix_id"],
            "host_id": postflight["host_id"],
            "postflight_path": str(report_path),
            "postflight_sha256": postflight_sha256.lower(),
            "published_root": str(root),
            "canonical_manifest_sha256": postflight["canonical_manifest_sha256"],
            "files_digest": postflight["files_digest"],
            "code_shas": source_code_shas,
        },
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in outputs.items()
        },
    }
    _atomic_create(
        output / "manifest.json",
        (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--postflight", type=Path, required=True)
    parser.add_argument("--postflight-sha256", required=True)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_bist_ndr_supplement(
            base_csv_path=args.base_csv,
            base_manifest_path=args.base_manifest,
            postflight_path=args.postflight,
            postflight_sha256=args.postflight_sha256,
            published_root=args.published_root,
            output_dir=args.output_dir,
        )
    except (BistNdrSupplementError, OSError) as exc:
        print(f"BIST_NDR_SUPPLEMENT_FAILED {exc}", file=sys.stderr)
        return 1
    print(f"BIST_NDR_SUPPLEMENT_OK rows={report['row_count']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
