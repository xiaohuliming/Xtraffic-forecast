from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .manifest import METHODS, REGIONS
from .paper_reference import (
    DEFAULT_REFERENCE_PATH,
    DEFAULT_SOURCE_LOCK_PATH,
    METRIC_COLUMNS,
    load_paper_table4_reference,
)
from .secure_io import read_verified_regular_file


_MAX_INDEX_BYTES = 32 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_METRIC_PATHS = {
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
_METHOD_LABELS = {
    "hl": "HL",
    "lstm": "LSTM",
    "dcrnn": "DCRNN",
    "agcrn": "AGCRN",
    "stgcn": "STGCN",
    "gwnet": "GWNET",
    "astgcn": "ASTGCN",
    "sttn": "STTN",
    "dstagnn": "DSTAGNN",
    "dgcrn": "DGCRN",
    "d2stgnn": r"D$^2$STGNN",
    "bist": "BiST",
    "igstgnn": "IGSTGNN",
}
_REGION_LABELS = {
    "Alameda": "Alameda",
    "Contra_Costa": "Contra Costa",
    "Orange": "Orange",
}


class CompositeReportError(ValueError):
    """Raised when a verified composite cannot produce a trustworthy table."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CompositeReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise CompositeReportError(f"non-finite JSON constant: {value}")


def _ensure_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise CompositeReportError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _ensure_finite(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_finite(item, label=label)


def _read_index(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise CompositeReportError(
            "expected composite index SHA-256 is invalid"
        )
    try:
        payload = read_verified_regular_file(
            path,
            purpose="Table 4 composite index",
            max_bytes=_MAX_INDEX_BYTES,
            require_nonempty=True,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CompositeReportError(
            f"cannot read composite index: {path}"
        ) from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise CompositeReportError("composite index SHA-256 mismatch")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except CompositeReportError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CompositeReportError("composite index is not strict JSON") from error
    if not isinstance(value, dict):
        raise CompositeReportError("composite index must contain an object")
    _ensure_finite(value, label="composite index")
    return value


def _metric(value: Any, *, task_id: str, column: str) -> float:
    first, second = _METRIC_PATHS[column]
    if not isinstance(value, dict) or not isinstance(value.get(first), dict):
        raise CompositeReportError(
            f"{task_id} metrics do not contain {first}.{second}"
        )
    number = value[first].get(second)
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(float(number))
        or float(number) <= 0
    ):
        raise CompositeReportError(
            f"{task_id} {first}.{second} must be finite and positive"
        )
    return float(number)


def _load_reproduced_rows(index: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        index.get("schema_version") != 1
        or index.get("status") != "verified"
        or index.get("matrix_validity_status")
        != "verified_official_protocol_with_known_overlap_39"
        or index.get("protocol_id") != "igstgnn_table4_official_v1"
        or index.get("matrix_id") != "igstgnn_table4_public_repro_39_v2"
        or index.get("claim_scope")
        != "public_material_maximum_fidelity_reproduction"
        or index.get("validity_label")
        != "official_protocol_with_known_overlap"
        or index.get("source_count") != 3
        or index.get("task_count") != 39
    ):
        raise CompositeReportError(
            "composite index identity is not the verified Table 4 matrix"
        )
    tasks = index.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 39:
        raise CompositeReportError("composite index must contain 39 tasks")
    expected_order = [
        (number, region, method)
        for number, (region, method) in enumerate(
            (
                (region, method)
                for region in REGIONS
                for method in METHODS
            ),
            start=1,
        )
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task, (number, region, method) in zip(
        tasks,
        expected_order,
        strict=True,
    ):
        if not isinstance(task, dict):
            raise CompositeReportError("composite task must be an object")
        expected_task_id = f"{number:03d}_{region.lower()}_{method}"
        task_id = task.get("task_id")
        if (
            task_id != expected_task_id
            or task.get("region") != region
            or task.get("method") != method
            or task_id in seen
        ):
            raise CompositeReportError(
                "composite task order or identity is not canonical"
            )
        seen.add(task_id)
        metrics = task.get("metrics")
        row: dict[str, Any] = {"region": region, "method": method}
        for column in METRIC_COLUMNS:
            row[column] = _metric(
                metrics,
                task_id=task_id,
                column=column,
            )
        rows.append(row)
    return rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("region", "method", *METRIC_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "region": row["region"],
                "method": row["method"],
                **{
                    column: f"{row[column]:.6f}"
                    for column in METRIC_COLUMNS
                },
            }
        )
    return buffer.getvalue().encode("utf-8")


def _comparison_csv_bytes(
    reproduced_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> bytes:
    reference_by_key = {
        (row["region"], row["method"]): row for row in reference_rows
    }
    fieldnames = ["region", "method"]
    for column in METRIC_COLUMNS:
        fieldnames.extend(
            [
                f"paper_{column}",
                f"reproduced_{column}",
                f"delta_{column}",
            ]
        )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    for reproduced in reproduced_rows:
        paper = reference_by_key[
            (reproduced["region"], reproduced["method"])
        ]
        row: dict[str, str] = {
            "region": reproduced["region"],
            "method": reproduced["method"],
        }
        for column in METRIC_COLUMNS:
            row[f"paper_{column}"] = f"{paper[column]:.6f}"
            row[f"reproduced_{column}"] = f"{reproduced[column]:.6f}"
            row[f"delta_{column}"] = (
                f"{reproduced[column] - paper[column]:.6f}"
            )
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _markdown_bytes(
    rows: list[dict[str, Any]],
    *,
    composite_index_sha256: str,
) -> bytes:
    lines = [
        "# IGSTGNN Table 4 公开材料最大忠实复现结果",
        "",
        "协议：`igstgnn_table4_official_v1`。",
        "有效性标签：`official_protocol_with_known_overlap`。",
        "本表不是无泄露 v3 结果，也不是作者原表数值。",
        f"Composite index SHA-256：`{composite_index_sha256}`。",
        "",
    ]
    headers = [
        "Method",
        "H3 MAE",
        "H3 RMSE",
        "H3 MAPE(%)",
        "H6 MAE",
        "H6 RMSE",
        "H6 MAPE(%)",
        "H12 MAE",
        "H12 RMSE",
        "H12 MAPE(%)",
        "Avg MAE",
        "Avg RMSE",
        "Avg MAPE(%)",
    ]
    for region in REGIONS:
        lines.extend(
            [
                f"## {_REGION_LABELS[region]}",
                "",
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] + ["---:"] * 12) + "|",
            ]
        )
        for row in rows:
            if row["region"] != region:
                continue
            values = [
                _METHOD_LABELS[row["method"]].replace(r"$^2$", "²"),
                *[f"{row[column]:.2f}" for column in METRIC_COLUMNS],
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _latex_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = [
        r"\begin{sidewaystable*}[t]",
        r"\centering",
        r"\caption{IGSTGNN Table 4 public-material maximum-fidelity reproduction. "
        r"The protocol retains known overlap and is not the no-leak v3 protocol.}",
        r"\label{tab:igstgnn-table4-public-reproduction}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrrrrrrrr}",
        r"\toprule",
        r"Data & Method & \multicolumn{3}{c}{Horizon 3} & "
        r"\multicolumn{3}{c}{Horizon 6} & "
        r"\multicolumn{3}{c}{Horizon 12} & "
        r"\multicolumn{3}{c}{Average} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}"
        r"\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
        r"& & MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) "
        r"& MAE & RMSE & MAPE(\%) & MAE & RMSE & MAPE(\%) \\",
        r"\midrule",
    ]
    for region_index, region in enumerate(REGIONS):
        region_rows = [row for row in rows if row["region"] == region]
        for row_index, row in enumerate(region_rows):
            data_cell = (
                rf"\multirow{{13}}{{*}}{{{_REGION_LABELS[region]}}}"
                if row_index == 0
                else ""
            )
            values = " & ".join(
                f"{row[column]:.2f}" for column in METRIC_COLUMNS
            )
            lines.append(
                f"{data_cell} & {_METHOD_LABELS[row['method']]} & "
                f"{values} \\\\"
            )
        if region_index < len(REGIONS) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{sidewaystable*}",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _publish_directory(
    output_dir: Path,
    files: dict[str, bytes],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise CompositeReportError(
            f"report output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        output_dir.parent
        / f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir()
    try:
        for name, payload in files.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(staging, output_dir)
    except Exception:
        if staging.is_dir():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise


def build_table4_reproduction_report(
    *,
    composite_index_path: Path | str,
    expected_composite_index_sha256: str,
    output_dir: Path | str,
    reference_path: Path | str = DEFAULT_REFERENCE_PATH,
    source_lock_path: Path | str = DEFAULT_SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    index_path = Path(composite_index_path)
    index = _read_index(
        index_path,
        expected_sha256=expected_composite_index_sha256,
    )
    reproduced_rows = _load_reproduced_rows(index)
    try:
        reference = load_paper_table4_reference(
            reference_path,
            source_lock_path=source_lock_path,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise CompositeReportError(
            "paper Table 4 reference verification failed"
        ) from error
    reference_rows = reference["rows"]
    reproduced_csv = _csv_bytes(reproduced_rows)
    comparison_csv = _comparison_csv_bytes(
        reproduced_rows,
        reference_rows,
    )
    markdown = _markdown_bytes(
        reproduced_rows,
        composite_index_sha256=expected_composite_index_sha256,
    )
    latex = _latex_bytes(reproduced_rows)
    files = {
        "table4_reproduced.csv": reproduced_csv,
        "table4_paper_comparison.csv": comparison_csv,
        "table4_reproduced.md": markdown,
        "table4_reproduced.tex": latex,
    }
    report = {
        "schema_version": 1,
        "status": "verified_reproduction_table",
        "protocol_id": index["protocol_id"],
        "matrix_id": index["matrix_id"],
        "composite_id": index["composite_id"],
        "composite_index_sha256": expected_composite_index_sha256,
        "paper_reference_sha256": reference["source_file_sha256"],
        "paper_source_lock_sha256": reference["source_lock_sha256"],
        "claim_scope": index["claim_scope"],
        "validity_label": index["validity_label"],
        "row_count": len(reproduced_rows),
        "region_count": len(REGIONS),
        "method_count": len(METHODS),
        "outputs": {
            name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in files.items()
        },
        "caveats": [
            "official_protocol_with_known_overlap",
            "public_material_maximum_fidelity_reproduction",
            "not_protocol_v3_noleak_dedup",
            "not_author_reported_values",
        ],
    }
    report_payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    files["table4_report_manifest.json"] = report_payload
    _publish_directory(Path(output_dir), files)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the verified IGSTGNN Table 4 reproduction report",
    )
    parser.add_argument("--composite-index", required=True, type=Path)
    parser.add_argument(
        "--composite-index-sha256",
        required=True,
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_SOURCE_LOCK_PATH,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_table4_reproduction_report(
            composite_index_path=args.composite_index,
            expected_composite_index_sha256=args.composite_index_sha256,
            reference_path=args.reference,
            source_lock_path=args.source_lock,
            output_dir=args.output_dir,
        )
    except CompositeReportError as error:
        print(f"REPORT_BLOCKED {error}")
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "row_count": report["row_count"],
                "composite_index_sha256": report[
                    "composite_index_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
