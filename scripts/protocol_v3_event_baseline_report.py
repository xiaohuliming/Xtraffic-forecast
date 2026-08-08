#!/usr/bin/env python3
"""Generate a paper-shaped table from a verified v3 event full-33 index."""

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

from scripts.remote.linux.protocol_v3_event_full33_composite import (
    EXPECTED_PROTOCOL,
    EXPECTED_STATUS,
    EXPECTED_TASKS,
)
from scripts.remote.windows import protocol_v3_runner as runner
from scripts.remote.windows.protocol_v3_baseline_runner import load_baseline_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = (
    REPO_ROOT / "config/protocol_v3/table4_event_baselines_phase1_v1.json"
)
COLUMNS = (
    "region", "method",
    "h3_mae", "h3_rmse", "h3_mape_percent",
    "h6_mae", "h6_rmse", "h6_mape_percent",
    "h12_mae", "h12_rmse", "h12_mape_percent",
    "avg_mae", "avg_rmse", "avg_mape_percent",
)
METRIC_PATHS = {
    "h3_mae": ("h3", "mae"),
    "h3_rmse": ("h3", "rmse"),
    "h3_mape_percent": ("h3", "mape_percent"),
    "h6_mae": ("h6", "mae"),
    "h6_rmse": ("h6", "rmse"),
    "h6_mape_percent": ("h6", "mape_percent"),
    "h12_mae": ("h12", "mae"),
    "h12_rmse": ("h12", "rmse"),
    "h12_mape_percent": ("h12", "mape_percent"),
    "avg_mae": ("average", "mae"),
    "avg_rmse": ("average", "rmse"),
    "avg_mape_percent": ("average", "mape_percent"),
}
METHOD_LABELS = {
    "hl": "HL",
    "lstm": "LSTM",
    "dcrnn": "DCRNN",
    "agcrn": "AGCRN",
    "stgcn": "STGCN",
    "graphwavenet": "GWNET",
    "astgcn": "ASTGCN",
    "sttn": "STTN",
    "dstagnn": "DSTAGNN",
    "dgcrn": "DGCRN",
    "d2stgnn": "D²STGNN",
}
REGION_LABELS = {
    "Alameda": "Alameda",
    "ContraCosta": "Contra Costa",
    "Orange": "Orange",
}


class EventBaselineReportError(RuntimeError):
    """Raised when the verified index cannot produce a faithful table."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EventBaselineReportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise EventBaselineReportError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise EventBaselineReportError(f"{label} is missing or a symlink")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except EventBaselineReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EventBaselineReportError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EventBaselineReportError(f"{label} must contain an object")
    return value


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EventBaselineReportError(f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise EventBaselineReportError("summary_path must be a nonempty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise EventBaselineReportError(f"unsafe summary path: {value}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise EventBaselineReportError(f"summary is missing or a symlink: {path}")
    return path


def _metric(metrics: Mapping[str, object], group: str, name: str) -> float:
    group_value = metrics.get(group)
    value = group_value.get(name) if isinstance(group_value, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventBaselineReportError(f"metric {group}.{name} is missing")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EventBaselineReportError(f"metric {group}.{name} is invalid")
    return number


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EventBaselineReportError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise EventBaselineReportError(f"output appeared during write: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _csv_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _markdown_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    header = ["Data", "Method", "H3 MAE", "H3 RMSE", "H3 MAPE(%)",
              "H6 MAE", "H6 RMSE", "H6 MAPE(%)", "H12 MAE", "H12 RMSE",
              "H12 MAPE(%)", "Avg MAE", "Avg RMSE", "Avg MAPE(%)"]
    lines = [
        "# Protocol v3 no-leak event baseline comparison",
        "",
        "Protocol: `protocol_v3_noleak_dedup`. All rows use seed 42 and the same target-disjoint event split.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _latex_payload(rows: Sequence[Mapping[str, str]]) -> bytes:
    lines = [
        r"\begin{tabular}{llrrrrrrrrrrrr}",
        r"\toprule",
        r"Data & Method & \multicolumn{3}{c}{Horizon 3} & \multicolumn{3}{c}{Horizon 6} & \multicolumn{3}{c}{Horizon 12} & \multicolumn{3}{c}{Average} \\",
        r" & & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) \\",
        r"\midrule",
    ]
    previous_region = None
    for row in rows:
        region = row["region"] if row["region"] != previous_region else ""
        values = [region, row["method"], *(row[column] for column in COLUMNS[2:])]
        lines.append(" & ".join(_latex_escape(str(value)) for value in values) + r" \\")
        previous_region = row["region"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_event_baseline_report(
    *,
    composite_index_path: Path | str,
    expected_index_sha256: str,
    published_root: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    index_path = Path(composite_index_path).expanduser().resolve(strict=True)
    if _sha256(index_path) != expected_index_sha256.lower():
        raise EventBaselineReportError("composite index SHA-256 mismatch")
    index = _load_json(index_path, label="full-33 composite index")
    expected_identity = {
        "status": EXPECTED_STATUS,
        "protocol_version": EXPECTED_PROTOCOL,
        "task_count": EXPECTED_TASKS,
        "execution_layout": "single_host_four_gpu_interleaved",
    }
    for name, expected in expected_identity.items():
        if index.get(name) != expected:
            raise EventBaselineReportError(f"composite index {name} mismatch")
    root = Path(published_root).expanduser().resolve(strict=True)
    if Path(str(index.get("published_root"))).resolve(strict=True) != root:
        raise EventBaselineReportError("published root does not match the composite index")
    manifest = load_baseline_manifest(CANONICAL_MANIFEST)
    if index.get("canonical_manifest_sha256") != _sha256(CANONICAL_MANIFEST):
        raise EventBaselineReportError("canonical manifest identity mismatch")
    task_items = index.get("tasks")
    if not isinstance(task_items, list) or len(task_items) != EXPECTED_TASKS:
        raise EventBaselineReportError("composite index must contain 33 tasks")
    by_id: dict[str, Mapping[str, object]] = {}
    for item in task_items:
        if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
            raise EventBaselineReportError("invalid composite task record")
        by_id[str(item["task_id"])] = item
    if len(by_id) != EXPECTED_TASKS:
        raise EventBaselineReportError("duplicate composite task record")

    rows: list[dict[str, str]] = []
    summary_proofs: list[dict[str, str]] = []
    for task in manifest.tasks:
        item = by_id.get(task.task_id)
        if item is None:
            raise EventBaselineReportError(f"missing task: {task.task_id}")
        expected_task_fields = {
            "region": task.region,
            "model": task.model,
            "variant": task.variant,
            "seed": task.seed,
        }
        for name, expected in expected_task_fields.items():
            if item.get(name) != expected:
                raise EventBaselineReportError(f"task identity mismatch: {task.task_id}")
        summary_path = _safe_path(root, item.get("summary_path"))
        summary_sha = _sha256(summary_path)
        if summary_sha != item.get("summary_sha256"):
            raise EventBaselineReportError(f"summary SHA-256 mismatch: {task.task_id}")
        summary = _load_json(summary_path, label=f"{task.task_id} summary")
        if (
            summary.get("task_id") != task.task_id
            or summary.get("protocol_version") != EXPECTED_PROTOCOL
            or summary.get("sampling_protocol") != "event"
            or summary.get("seed") != 42
        ):
            raise EventBaselineReportError(f"summary identity mismatch: {task.task_id}")
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            raise EventBaselineReportError(f"task metrics are missing: {task.task_id}")
        row = {
            "region": REGION_LABELS[task.region],
            "method": METHOD_LABELS[task.model],
        }
        for column, (group, name) in METRIC_PATHS.items():
            row[column] = f"{_metric(metrics, group, name):.6f}"
        rows.append(row)
        summary_proofs.append(
            {
                "task_id": task.task_id,
                "summary_path": str(item["summary_path"]),
                "summary_sha256": summary_sha,
            }
        )

    output = Path(output_dir).expanduser().resolve(strict=False)
    csv_payload = _csv_payload(rows)
    markdown_payload = _markdown_payload(rows)
    latex_payload = _latex_payload(rows)
    outputs = {
        "table.csv": csv_payload,
        "table.md": markdown_payload,
        "table.tex": latex_payload,
    }
    for name, payload in outputs.items():
        _atomic_create(output / name, payload)
    manifest_payload: dict[str, object] = {
        "report_schema_version": 1,
        "status": "verified_table_generated",
        "protocol_version": EXPECTED_PROTOCOL,
        "validity_label": "target_disjoint_event_no_leak",
        "seed_policy": "single_seed_42",
        "task_count": EXPECTED_TASKS,
        "row_count": len(rows),
        "composite_index_path": str(index_path),
        "composite_index_sha256": expected_index_sha256.lower(),
        "summary_proofs": summary_proofs,
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in outputs.items()
        },
    }
    manifest_bytes = (
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_create(output / "manifest.json", manifest_bytes)
    return manifest_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite-index", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_event_baseline_report(
            composite_index_path=args.composite_index,
            expected_index_sha256=args.expected_index_sha256,
            published_root=args.published_root,
            output_dir=args.output_dir,
        )
    except (EventBaselineReportError, OSError, runner.ManifestError) as exc:
        print(f"EVENT_BASELINE_REPORT_FAILED {exc}", file=sys.stderr)
        return 1
    print(
        "EVENT_BASELINE_REPORT_OK "
        f"rows={report['row_count']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
