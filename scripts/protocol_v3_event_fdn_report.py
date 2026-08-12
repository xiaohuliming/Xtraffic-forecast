#!/usr/bin/env python3
"""Append verified v3 FDN results to the immutable 42-row event table."""

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
from typing import Mapping, Sequence
import uuid

import numpy as np

from scripts.protocol_v3_event_baseline_report import COLUMNS
from scripts.remote.windows.protocol_v3_event_fdn_runner import (
    DEFAULT_MANIFEST,
    load_fdn_manifest,
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


class FdnSupplementError(RuntimeError):
    """Raised when the FDN supplement cannot be proven read-only."""


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
        raise FdnSupplementError(f"{label} is missing or a symlink")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FdnSupplementError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                FdnSupplementError(
                    f"non-finite JSON value in {label}: {item}"
                )
            ),
        )
    except FdnSupplementError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FdnSupplementError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise FdnSupplementError(f"{label} must contain an object")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FdnSupplementError(f"cannot hash non-regular file: {path}")
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
        raise FdnSupplementError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise FdnSupplementError(f"{name} must be a nonempty string")
    return value


def _required_sha(payload: Mapping[str, object], name: str, length: int) -> str:
    value = _required_string(payload, name).lower()
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FdnSupplementError(f"{name} is not a valid SHA")
    return value


def _safe_published_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise FdnSupplementError("published file path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise FdnSupplementError(f"unsafe published file path: {value}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise FdnSupplementError(f"published file is missing: {path}")
    return path


def _load_base_table(
    csv_path: Path, manifest_path: Path
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if _sha256(manifest_path) != EXPECTED_BASE_MANIFEST_SHA256:
        raise FdnSupplementError("base manifest SHA-256 mismatch")
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
            raise FdnSupplementError(f"base manifest {name} mismatch")
    if _sha256(csv_path) != EXPECTED_BASE_CSV_SHA256:
        raise FdnSupplementError("base table.csv SHA-256 mismatch")
    outputs = manifest.get("outputs")
    csv_proof = outputs.get("table.csv") if isinstance(outputs, dict) else None
    if (
        not isinstance(csv_proof, dict)
        or csv_proof.get("sha256") != EXPECTED_BASE_CSV_SHA256
    ):
        raise FdnSupplementError("base manifest does not bind table.csv")
    try:
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != COLUMNS:
                raise FdnSupplementError("base table columns mismatch")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FdnSupplementError("cannot read base table.csv") from exc
    if len(rows) != 42:
        raise FdnSupplementError("base table must contain exactly 42 rows")
    return rows, manifest


def _prediction_metrics(path: Path) -> dict[str, float]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            required = {
                "pred_raw_flow",
                "actual_future_flow",
                "y_mask_flow",
            }
            if not required.issubset(arrays.files):
                raise FdnSupplementError(
                    "prediction artifact lacks table metric arrays"
                )
            prediction = arrays["pred_raw_flow"]
            actual = arrays["actual_future_flow"]
            mask = arrays["y_mask_flow"]
    except FdnSupplementError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise FdnSupplementError("cannot read prediction artifact") from exc
    if (
        prediction.ndim != 3
        or prediction.shape != actual.shape
        or prediction.shape != mask.shape
        or prediction.shape[1] != 12
    ):
        raise FdnSupplementError("prediction artifact shape mismatch")
    if mask.dtype != np.bool_:
        raise FdnSupplementError("prediction mask must be boolean")
    if not np.isfinite(prediction).all() or not np.isfinite(actual[mask]).all():
        raise FdnSupplementError("prediction artifact contains non-finite values")
    prediction64 = prediction.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    absolute_error = np.abs(prediction64 - actual64)
    actual_abs = np.abs(actual64)

    def metrics(
        errors_source: np.ndarray,
        actual_source: np.ndarray,
        selected: np.ndarray,
    ) -> tuple[float, float, float]:
        errors = errors_source[selected]
        percentage = selected & (actual_source > 1e-4)
        if errors.size == 0 or not percentage.any():
            raise FdnSupplementError("prediction artifact has empty metric support")
        mae = float(errors.mean())
        rmse = float(np.sqrt(np.square(errors).mean()))
        mape = float(
            (errors_source[percentage] / actual_source[percentage]).mean()
            * 100.0
        )
        return mae, rmse, mape

    result: dict[str, float] = {}
    for label, index in (("h3", 2), ("h6", 5), ("h12", 11)):
        mae, rmse, mape = metrics(
            absolute_error[:, index, :],
            actual_abs[:, index, :],
            mask[:, index, :],
        )
        result[f"{label}_mae"] = mae
        result[f"{label}_rmse"] = rmse
        result[f"{label}_mape_percent"] = mape
    mae, rmse, mape = metrics(absolute_error, actual_abs, mask)
    result["avg_mae"] = mae
    result["avg_rmse"] = rmse
    result["avg_mape_percent"] = mape
    return result


def _verify_export(
    *,
    report_path: Path,
    expected_report_sha256: str,
    published_root: Path,
    base_manifest: Mapping[str, object],
) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    if _sha256(report_path) != expected_report_sha256.lower():
        raise FdnSupplementError("postflight SHA-256 mismatch")
    report = _load_json(report_path, "FDN postflight")
    expected = {
        "status": "verified",
        "runner_kind": "event_fdn",
        "protocol_version": EXPECTED_PROTOCOL,
        "cache_schema_version": 3,
        "validated_tasks": 3,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise FdnSupplementError(f"postflight {name} mismatch")
    for name in ("cache_bundle_id", "cache_bundle_digest"):
        if report.get(name) != base_manifest.get(name):
            raise FdnSupplementError(f"postflight {name} mismatch")
    if report.get("canonical_manifest_sha256") != _sha256(DEFAULT_MANIFEST):
        raise FdnSupplementError("FDN manifest identity mismatch")
    if report.get("task_state_counts") != {"completed": 3}:
        raise FdnSupplementError("all FDN tasks must be newly completed")

    raw_files = report.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FdnSupplementError("postflight file list is invalid")
    if report.get("file_count") != len(raw_files):
        raise FdnSupplementError("postflight file_count mismatch")
    if report.get("files_digest") != hashlib.sha256(
        _canonical_json(raw_files)
    ).hexdigest():
        raise FdnSupplementError("postflight files_digest mismatch")
    records: dict[str, Mapping[str, object]] = {}
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise FdnSupplementError("postflight file record is invalid")
        relative = _required_string(raw, "path")
        if relative in records:
            raise FdnSupplementError(f"duplicate postflight file: {relative}")
        path = _safe_published_path(published_root, relative)
        expected_size = raw.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise FdnSupplementError(f"invalid file size: {relative}")
        if path.stat().st_size != expected_size:
            raise FdnSupplementError(f"published file size mismatch: {relative}")
        if _sha256(path) != _required_sha(raw, "sha256", 64):
            raise FdnSupplementError(f"published file SHA-256 mismatch: {relative}")
        records[relative] = raw
        total_bytes += expected_size
    if report.get("total_bytes") != total_bytes:
        raise FdnSupplementError("postflight total_bytes mismatch")

    manifest = load_fdn_manifest(DEFAULT_MANIFEST)
    raw_tasks = report.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 3:
        raise FdnSupplementError("postflight task list is invalid")
    task_proofs: dict[str, Mapping[str, object]] = {}
    for item in raw_tasks:
        if not isinstance(item, dict):
            raise FdnSupplementError("postflight task proof is invalid")
        task_id = _required_string(item, "task_id")
        if task_id in task_proofs:
            raise FdnSupplementError(f"duplicate postflight task: {task_id}")
        if item.get("state") != "completed":
            raise FdnSupplementError(f"task was not completed: {task_id}")
        config = item.get("invocation_config")
        if not isinstance(config, dict):
            raise FdnSupplementError(f"task invocation config is invalid: {task_id}")
        expected_config = {
            "sampling_protocol": "event",
            "decomp_mode": "learnable",
            "K": 3,
            "epochs": 30,
            "patience": 6,
            "batch_size": 32,
        }
        for name, value in expected_config.items():
            if config.get(name) != value:
                raise FdnSupplementError(
                    f"task training contract mismatch: {task_id}/{name}"
                )
        task_proofs[task_id] = item
    if set(task_proofs) != {task.task_id for task in manifest.tasks}:
        raise FdnSupplementError("postflight tasks differ from the manifest")

    rows: dict[str, dict[str, str]] = {}
    for task in manifest.tasks:
        proof = task_proofs[task.task_id]
        source_id = _required_string(proof, "source_id")
        code_sha = _required_sha(proof, "code_sha", 40)
        config_hash = _required_sha(proof, "config_hash", 64)
        prefix = f"sources/{source_id}/results/{task.output_dir.as_posix()}"
        summary_relative = f"{prefix}/summary.json"
        prediction_relative = f"{prefix}/test_predictions.npz"
        if summary_relative not in records or prediction_relative not in records:
            raise FdnSupplementError(
                f"FDN metric artifacts are absent from allowlist: {task.task_id}"
            )
        summary = _load_json(
            _safe_published_path(published_root, summary_relative),
            f"{task.task_id} summary",
        )
        identities = {
            "task_id": task.task_id,
            "region": task.region,
            "model": "fourier_dual_net",
            "variant": "learnable_k3",
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
                raise FdnSupplementError(
                    f"summary identity mismatch for {task.task_id}: {name}"
                )
        metric_values = _prediction_metrics(
            _safe_published_path(published_root, prediction_relative)
        )
        reported_all = summary.get("all")
        if (
            isinstance(reported_all, bool)
            or not isinstance(reported_all, (int, float))
            or not math.isclose(
                float(reported_all),
                metric_values["avg_mae"],
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise FdnSupplementError(
                f"summary all does not match prediction artifact: {task.task_id}"
            )
        rows[task.region] = {
            "region": REGION_LABELS[task.region],
            "method": "FDN",
            **{
                name: f"{metric_values[name]:.6f}"
                for name in COLUMNS[2:]
            },
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
        "# Protocol v3 no-leak event comparison with FDN",
        "",
        "This 45-row supplement preserves the verified 42-row table and adds three seed-42 FDN learnable-K3 runs under the same no-leak event split, mask, metric and 30-epoch cap.",
        "",
        "The FDN rows are host- and code-stratified. Treat this as a single-seed descriptive comparison.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[column] for column in COLUMNS) + " |"
        for row in rows
    )
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
        lines.append(
            " & ".join(value.replace("&", r"\&") for value in values) + r" \\"
        )
        previous = row["region"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FdnSupplementError(f"output already exists: {path}")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FdnSupplementError(f"output appeared during write: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_fdn_supplement(
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
            raise FdnSupplementError(
                f"base region row count mismatch: {region_key}"
            )
        bist_index = next(
            (index for index, row in enumerate(regional) if row["method"] == "BiST"),
            None,
        )
        if bist_index is None:
            raise FdnSupplementError(f"base BiST row is missing: {region_key}")
        rows.extend(regional[:bist_index])
        rows.append(additions[region_key])
        rows.extend(regional[bist_index:])
    if len(rows) != 45:
        raise FdnSupplementError("supplement must contain exactly 45 rows")
    outputs = {
        "table.csv": _csv_payload(rows),
        "table.md": _markdown_payload(rows),
        "table.tex": _latex_payload(rows),
    }
    output = Path(output_dir).expanduser().resolve(strict=False)
    for name, payload in outputs.items():
        _atomic_create(output / name, payload)
    base_indexes = base_manifest.get("source_indexes")
    if not isinstance(base_indexes, dict):
        raise FdnSupplementError("base source indexes are invalid")
    source_code_shas = sorted(
        {
            _required_sha(value, "code_sha", 40)
            for value in base_indexes.values()
            if isinstance(value, dict)
        }
        | {
            _required_sha(item, "code_sha", 40)
            for item in postflight["tasks"]
            if isinstance(item, dict)
        }
    )
    manifest: dict[str, object] = {
        "report_schema_version": 1,
        "status": "verified_no_leak_fdn_supplement_generated",
        "protocol_version": EXPECTED_PROTOCOL,
        "validity_label": "target_disjoint_event_no_leak_host_and_code_stratified",
        "execution_layout": "verified_42_plus_rtx5080_fdn_supplement",
        "seed_policy": "single_seed_42",
        "epoch_cap": 30,
        "early_stopping_patience": 6,
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
        "fdn_source": {
            "matrix_id": postflight["matrix_id"],
            "host_id": postflight["host_id"],
            "postflight_path": str(report_path),
            "postflight_sha256": postflight_sha256.lower(),
            "published_root": str(root),
            "canonical_manifest_sha256": postflight[
                "canonical_manifest_sha256"
            ],
            "files_digest": postflight["files_digest"],
            "code_shas": source_code_shas,
            "method": "FDN learnable-K3",
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
        report = build_fdn_supplement(
            base_csv_path=args.base_csv,
            base_manifest_path=args.base_manifest,
            postflight_path=args.postflight,
            postflight_sha256=args.postflight_sha256,
            published_root=args.published_root,
            output_dir=args.output_dir,
        )
    except (FdnSupplementError, OSError) as exc:
        print(f"FDN_SUPPLEMENT_FAILED {exc}", file=sys.stderr)
        return 1
    print(f"FDN_SUPPLEMENT_OK rows={report['row_count']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
