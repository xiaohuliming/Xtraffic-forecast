#!/usr/bin/env python3
"""Merge verified v3 baseline and extension indexes into the 42-row paper table."""

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

from scripts.protocol_v3_event_baseline_report import (
    COLUMNS,
    METHOD_LABELS,
    METRIC_PATHS,
    REGION_LABELS,
)
from scripts.remote.linux.protocol_v3_event_full33_composite import (
    EXPECTED_PROTOCOL,
    EXPECTED_STATUS,
)
from scripts.remote.linux.protocol_v3_event_extension_composite import (
    EXPECTED_STATUS as EXTENSION_STATUS,
)
from scripts.remote.windows.protocol_v3_baseline_runner import load_baseline_manifest
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST as EXTENSION_MANIFEST,
    load_extension_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MANIFEST = REPO_ROOT / "config/protocol_v3/table4_event_baselines_phase1_v1.json"
EXTENSION_LABELS = {"bist": "BiST", "igstgnn": "IGSTGNN", "ndr_net": "NDR-Net"}
METHOD_ORDER = tuple(METHOD_LABELS) + tuple(EXTENSION_LABELS)


class EventComparisonReportError(RuntimeError):
    """Raised when verified sources cannot be merged into one table."""


def _load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise EventComparisonReportError(f"{label} is missing or a symlink")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EventComparisonReportError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EventComparisonReportError(f"non-finite JSON value in {label}: {value}")
            ),
        )
    except EventComparisonReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EventComparisonReportError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise EventComparisonReportError(f"{label} must contain an object")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EventComparisonReportError(f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _published_file(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise EventComparisonReportError("summary_path must be a nonempty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise EventComparisonReportError(f"unsafe summary path: {value}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise EventComparisonReportError(f"summary is missing: {path}")
    return path


def _metric(metrics: Mapping[str, object], group: str, name: str) -> float:
    group_value = metrics.get(group)
    value = group_value.get(name) if isinstance(group_value, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventComparisonReportError(f"metric {group}.{name} is missing")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EventComparisonReportError(f"metric {group}.{name} is invalid")
    return result


def _validate_index(
    *,
    path: Path | str,
    digest: str,
    published_root: Path | str,
    status: str,
    task_count: int,
    runner_kind: str,
    manifest_path: Path,
) -> tuple[dict[str, object], Path]:
    index_path = Path(path).expanduser().resolve(strict=True)
    if _sha256(index_path) != digest.lower():
        raise EventComparisonReportError("composite index SHA-256 mismatch")
    index = _load_json(index_path, "composite index")
    expected = {
        "status": status,
        "protocol_version": EXPECTED_PROTOCOL,
        "task_count": task_count,
        "runner_kind": runner_kind,
        "cache_schema_version": 3,
    }
    for name, value in expected.items():
        observed = index.get(name, "baseline") if name == "runner_kind" else index.get(name)
        if observed != value:
            raise EventComparisonReportError(f"composite index {name} mismatch")
    root = Path(published_root).expanduser().resolve(strict=True)
    if Path(str(index.get("published_root"))).resolve(strict=True) != root:
        raise EventComparisonReportError("published root does not match its index")
    if index.get("canonical_manifest_sha256") != _sha256(manifest_path):
        raise EventComparisonReportError("canonical manifest identity mismatch")
    tasks = index.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != task_count:
        raise EventComparisonReportError("composite task list is invalid")
    return index, root


def _rows_from_index(
    *,
    index: Mapping[str, object],
    root: Path,
    manifest,
    labels: Mapping[str, str],
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, str]]]:
    raw_tasks = index.get("tasks")
    assert isinstance(raw_tasks, list)
    by_id: dict[str, Mapping[str, object]] = {}
    for item in raw_tasks:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise EventComparisonReportError("invalid composite task proof")
        task_id = str(item["task_id"])
        if task_id in by_id:
            raise EventComparisonReportError(f"duplicate task proof: {task_id}")
        by_id[task_id] = item
    rows: dict[tuple[str, str], dict[str, str]] = {}
    proofs: list[dict[str, str]] = []
    for task in manifest.tasks:
        item = by_id.get(task.task_id)
        if item is None:
            raise EventComparisonReportError(f"missing task proof: {task.task_id}")
        for name, expected in (
            ("region", task.region), ("model", task.model),
            ("variant", task.variant), ("seed", task.seed),
        ):
            if item.get(name) != expected:
                raise EventComparisonReportError(f"task identity mismatch: {task.task_id}")
        if task.seed != 42 or task.sampling_protocol != "event":
            raise EventComparisonReportError(f"task is outside the paper contract: {task.task_id}")
        summary_path = _published_file(root, item.get("summary_path"))
        summary_sha = _sha256(summary_path)
        if summary_sha != item.get("summary_sha256"):
            raise EventComparisonReportError(f"summary SHA-256 mismatch: {task.task_id}")
        summary = _load_json(summary_path, f"{task.task_id} summary")
        if (
            summary.get("task_id") != task.task_id
            or summary.get("protocol_version") != EXPECTED_PROTOCOL
            or summary.get("sampling_protocol") != "event"
            or summary.get("seed") != 42
        ):
            raise EventComparisonReportError(f"summary identity mismatch: {task.task_id}")
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            raise EventComparisonReportError(f"metrics are missing: {task.task_id}")
        row = {"region": REGION_LABELS[task.region], "method": labels[task.model]}
        for column, (group, name) in METRIC_PATHS.items():
            row[column] = f"{_metric(metrics, group, name):.6f}"
        key = (task.region, task.model)
        if key in rows:
            raise EventComparisonReportError(f"duplicate table row: {key}")
        rows[key] = row
        proofs.append(
            {
                "task_id": task.task_id,
                "summary_path": str(item["summary_path"]),
                "summary_sha256": summary_sha,
            }
        )
    if set(by_id) != {task.task_id for task in manifest.tasks}:
        raise EventComparisonReportError("composite task proofs exceed the manifest")
    return rows, proofs


def _csv(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _markdown(rows: Sequence[Mapping[str, str]]) -> bytes:
    header = [
        "Data", "Method", "H3 MAE", "H3 RMSE", "H3 MAPE(%)",
        "H6 MAE", "H6 RMSE", "H6 MAPE(%)", "H12 MAE", "H12 RMSE",
        "H12 MAPE(%)", "Avg MAE", "Avg RMSE", "Avg MAPE(%)",
    ]
    lines = [
        "# Protocol v3 no-leak event comparison",
        "",
        "All 42 rows use seed 42, 30 epochs at most, the same target-disjoint event split and explicit observation masks.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |" for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _latex(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        r"\begin{tabular}{llrrrrrrrrrrrr}", r"\toprule",
        r"Data & Method & \multicolumn{3}{c}{Horizon 3} & \multicolumn{3}{c}{Horizon 6} & \multicolumn{3}{c}{Horizon 12} & \multicolumn{3}{c}{Average} \\",
        r" & & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        region = row["region"] if row["region"] != previous else ""
        values = [region, row["method"], *(row[column] for column in COLUMNS[2:])]
        lines.append(" & ".join(str(value).replace("&", r"\&") for value in values) + r" \\")
        previous = row["region"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EventComparisonReportError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise EventComparisonReportError(f"output appeared during write: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def build_event_comparison_report(
    *,
    baseline_index_path: Path | str,
    baseline_index_sha256: str,
    baseline_published_root: Path | str,
    extension_index_path: Path | str,
    extension_index_sha256: str,
    extension_published_root: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    baseline_index, baseline_root = _validate_index(
        path=baseline_index_path, digest=baseline_index_sha256,
        published_root=baseline_published_root, status=EXPECTED_STATUS,
        task_count=33, runner_kind="baseline", manifest_path=BASELINE_MANIFEST,
    )
    extension_index, extension_root = _validate_index(
        path=extension_index_path, digest=extension_index_sha256,
        published_root=extension_published_root, status=EXTENSION_STATUS,
        task_count=9, runner_kind="event_extension", manifest_path=EXTENSION_MANIFEST,
    )
    for name in ("cache_bundle_id", "cache_bundle_digest"):
        if baseline_index.get(name) != extension_index.get(name):
            raise EventComparisonReportError(f"cache bundle {name} mismatch")
    baseline_manifest = load_baseline_manifest(BASELINE_MANIFEST)
    extension_manifest = load_extension_manifest(EXTENSION_MANIFEST)
    baseline_rows, baseline_proofs = _rows_from_index(
        index=baseline_index, root=baseline_root, manifest=baseline_manifest,
        labels=METHOD_LABELS,
    )
    extension_rows, extension_proofs = _rows_from_index(
        index=extension_index, root=extension_root, manifest=extension_manifest,
        labels=EXTENSION_LABELS,
    )
    rows: list[dict[str, str]] = []
    for region in ("Alameda", "ContraCosta", "Orange"):
        for method in METHOD_ORDER:
            source = baseline_rows if method in METHOD_LABELS else extension_rows
            row = source.get((region, method))
            if row is None:
                raise EventComparisonReportError(f"missing table row: {region}/{method}")
            rows.append(row)
    if len(rows) != 42:
        raise EventComparisonReportError("final table must contain exactly 42 rows")
    outputs = {"table.csv": _csv(rows), "table.md": _markdown(rows), "table.tex": _latex(rows)}
    output = Path(output_dir).expanduser().resolve(strict=False)
    for name, data in outputs.items():
        _atomic_create(output / name, data)
    manifest: dict[str, object] = {
        "report_schema_version": 1,
        "status": "verified_no_leak_comparison_generated",
        "protocol_version": EXPECTED_PROTOCOL,
        "validity_label": "target_disjoint_event_no_leak",
        "seed_policy": "single_seed_42",
        "epoch_cap": 30,
        "training_control_scope": (
            "shared_data_split_mask_metric_seed_and_epoch_cap_with_method_native_"
            "batch_optimizer_scheduler_and_early_stopping"
        ),
        "row_count": len(rows),
        "task_count": 42,
        "cache_bundle_id": baseline_index["cache_bundle_id"],
        "cache_bundle_digest": baseline_index["cache_bundle_digest"],
        "source_indexes": {
            "baseline33": {"path": str(Path(baseline_index_path).resolve()), "sha256": baseline_index_sha256.lower()},
            "extension9": {"path": str(Path(extension_index_path).resolve()), "sha256": extension_index_sha256.lower()},
        },
        "summary_proofs": baseline_proofs + extension_proofs,
        "outputs": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in outputs.items()
        },
    }
    _atomic_create(
        output / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--baseline-index-sha256", required=True)
    parser.add_argument("--baseline-published-root", type=Path, required=True)
    parser.add_argument("--extension-index", type=Path, required=True)
    parser.add_argument("--extension-index-sha256", required=True)
    parser.add_argument("--extension-published-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_event_comparison_report(
            baseline_index_path=args.baseline_index,
            baseline_index_sha256=args.baseline_index_sha256,
            baseline_published_root=args.baseline_published_root,
            extension_index_path=args.extension_index,
            extension_index_sha256=args.extension_index_sha256,
            extension_published_root=args.extension_published_root,
            output_dir=args.output_dir,
        )
    except (EventComparisonReportError, OSError) as exc:
        print(f"EVENT_COMPARISON_REPORT_FAILED {exc}", file=sys.stderr)
        return 1
    print(f"EVENT_COMPARISON_REPORT_OK rows={report['row_count']} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
