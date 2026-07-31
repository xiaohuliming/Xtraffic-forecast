from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .manifest import METHODS, REGIONS
from .native100_compare_manifest import (
    NATIVE100_CLAIM_SCOPE,
    NATIVE100_METHODS,
    NATIVE100_SEEDS,
    load_native100_compare_manifest,
)
from .paper_reference import METRIC_COLUMNS
from .result_artifacts import validate_completed_task_artifacts
from .secure_io import read_verified_regular_file


_STAT_COLUMNS = tuple(
    f"{column}_sample_std" for column in METRIC_COLUMNS
)
_HEADER = ("region", "method", *METRIC_COLUMNS, *_STAT_COLUMNS)
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
    "d2stgnn": "D²STGNN",
    "bist": "BiST",
    "igstgnn": "IGSTGNN",
    "ndr_net": "NDR-Net*",
    "bist_native100": "BiST 100ep*",
    "ndr_net_native100": "NDR-Net 100ep*",
}
_REGION_LABELS = {
    "Alameda": "Alameda",
    "Contra_Costa": "Contra Costa",
    "Orange": "Orange",
}
_SUPPLEMENT_METHOD_IDS = {
    "bist": "bist_native100",
    "ndr_net": "ndr_net_native100",
}


class Native100ReportError(ValueError):
    """Raised when the native-recipe 100 epoch matrix is not reportable."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(
    path: Path,
    *,
    purpose: str,
    expected_sha256: str,
    max_bytes: int,
) -> bytes:
    try:
        payload = bytes(
            read_verified_regular_file(
                path,
                purpose=purpose,
                max_bytes=max_bytes,
                require_nonempty=True,
            )
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise Native100ReportError(f"cannot read {purpose}: {path}") from error
    if _sha256(payload) != expected_sha256:
        raise Native100ReportError(f"{purpose} SHA-256 mismatch")
    return payload


def _finite_number(value: Any, *, label: str, positive: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise Native100ReportError(f"{label} is not numeric") from error
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise Native100ReportError(f"{label} must be finite and {qualifier}")
    return number


def _load_base_table(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, str]], bytes]:
    payload = _read_bytes(
        path,
        purpose="verified Table 4 with NDR CSV",
        expected_sha256=expected_sha256,
        max_bytes=4 * 1024 * 1024,
    )
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    except UnicodeError as error:
        raise Native100ReportError("base table is not UTF-8") from error
    if tuple(reader.fieldnames or ()) != _HEADER:
        raise Native100ReportError("base table header is not canonical")
    rows = list(reader)
    expected_keys = [
        (region, method) for region in REGIONS for method in METHODS
    ] + [(region, "ndr_net") for region in REGIONS]
    if (
        len(rows) != 42
        or [(row.get("region"), row.get("method")) for row in rows]
        != expected_keys
    ):
        raise Native100ReportError("base table must contain canonical 42 rows")
    for index, row in enumerate(rows):
        for column in METRIC_COLUMNS:
            _finite_number(
                row.get(column),
                label=f"base row {index} {column}",
                positive=True,
            )
        for column in _STAT_COLUMNS:
            value = row.get(column)
            if index < 39:
                if value != "":
                    raise Native100ReportError(
                        "single-run base rows must not contain sample statistics"
                    )
            else:
                _finite_number(
                    value,
                    label=f"base row {index} {column}",
                    positive=False,
                )
    return rows, payload


def _task_metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for column, (group, name) in _METRIC_PATHS.items():
        group_value = metrics.get(group)
        value = group_value.get(name) if isinstance(group_value, dict) else None
        values[column] = _finite_number(
            value,
            label=f"task metric {column}",
            positive=True,
        )
    return values


def _load_supplement_rows(
    *,
    manifest: dict[str, Any],
    results_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, float]]]] = {}
    task_records: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        task_dir = results_root / task["output_dir"]
        try:
            validation = validate_completed_task_artifacts(
                task_dir,
                expected_task=task,
                expected_claim_scope=NATIVE100_CLAIM_SCOPE,
            )
            summary_payload = bytes(
                read_verified_regular_file(
                    task_dir / "summary.json",
                    purpose="native 100 task summary",
                    max_bytes=1024 * 1024,
                    require_nonempty=True,
                )
            )
        except (OSError, ValueError, RuntimeError) as error:
            raise Native100ReportError(
                "native 100 report requires a complete 18-task matrix"
            ) from error
        key = (task["region"], task["method"])
        grouped.setdefault(key, []).append(
            (task["seed"], _task_metric_row(validation["metrics"]))
        )
        task_records.append(
            {
                "task_id": task["task_id"],
                "summary_sha256": _sha256(summary_payload),
            }
        )
    rows: list[dict[str, str]] = []
    for region in REGIONS:
        for method in NATIVE100_METHODS:
            samples = grouped.get((region, method), [])
            if [seed for seed, _ in samples] != list(NATIVE100_SEEDS):
                raise Native100ReportError(
                    "native 100 report requires a complete 18-task matrix"
                )
            row: dict[str, str] = {
                "region": region,
                "method": _SUPPLEMENT_METHOD_IDS[method],
            }
            for column in METRIC_COLUMNS:
                values = [sample[column] for _, sample in samples]
                row[column] = f"{statistics.mean(values):.6f}"
                row[f"{column}_sample_std"] = (
                    f"{statistics.stdev(values):.6f}"
                )
            rows.append(row)
    if len(task_records) != 18 or len(rows) != 6:
        raise Native100ReportError(
            "native 100 report requires a complete 18-task matrix"
        )
    return rows, task_records


def _combined_csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _display_value(row: dict[str, str], column: str) -> str:
    mean = float(row[column])
    sample_std = row[f"{column}_sample_std"]
    if sample_std == "":
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {float(sample_std):.2f}"


def _markdown_bytes(
    rows: list[dict[str, str]],
    *,
    base_sha256: str,
    manifest_sha256: str,
) -> bytes:
    lines = [
        "# IGSTGNN Table 4 复现与原生配方 100 epoch 补充对比",
        "",
        "BiST 100ep* 和 NDR-Net 100ep* 报告种子 42、1、2 的均值和样本标准差。",
        "新行只覆盖 max_epochs 为 100，保留两个方法各自的原生训练配方。",
        "全部结果保留官方协议中的已知预测时间重合，不是无泄露 v3 结果。",
        "",
        f"基础 42 行表 SHA-256：`{base_sha256}`。",
        f"100 epoch 矩阵 SHA-256：`{manifest_sha256}`。",
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
                _METHOD_LABELS[row["method"]],
                *[_display_value(row, column) for column in METRIC_COLUMNS],
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _latex_bytes(rows: list[dict[str, str]]) -> bytes:
    lines = [
        r"\begin{sidewaystable*}[t]",
        r"\centering",
        r"\caption{IGSTGNN Table 4 reproduction with native-recipe "
        r"100-epoch BiST and NDR-Net supplements. Starred 100-epoch "
        r"rows report mean $\pm$ sample standard deviation over seeds "
        r"42, 1, and 2.}",
        r"\label{tab:igstgnn-table4-native100}",
        r"\setlength{\tabcolsep}{3.0pt}",
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
                rf"\multirow{{{len(region_rows)}}}{{*}}{{{_REGION_LABELS[region]}}}"
                if row_index == 0
                else ""
            )
            method = _METHOD_LABELS[row["method"]].replace(
                "D²STGNN", r"D$^2$STGNN"
            )
            values = []
            for column in METRIC_COLUMNS:
                value = _display_value(row, column)
                values.append(f"${value.replace('±', r'\pm')}$" if "±" in value else value)
            lines.append(
                f"{data_cell} & {method} & {' & '.join(values)} \\\\"
            )
        if region_index < len(REGIONS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{sidewaystable*}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _publish(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise Native100ReportError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        for name, payload in files.items():
            with (staging / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise


def build_native100_table_report(
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
    base_manifest_path: Path | str,
    expected_base_manifest_sha256: str,
    results_root: Path | str,
    base_table_csv: Path | str,
    expected_base_table_sha256: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    manifest = load_native100_compare_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=expected_base_manifest_sha256,
    )
    base_rows, _ = _load_base_table(
        Path(base_table_csv),
        expected_sha256=expected_base_table_sha256,
    )
    supplement_rows, task_records = _load_supplement_rows(
        manifest=manifest,
        results_root=Path(results_root),
    )
    rows = [*base_rows, *supplement_rows]
    outputs = {
        "table4_with_native100.csv": _combined_csv_bytes(rows),
        "table4_with_native100.md": _markdown_bytes(
            rows,
            base_sha256=expected_base_table_sha256,
            manifest_sha256=expected_manifest_sha256,
        ),
        "table4_with_native100.tex": _latex_bytes(rows),
    }
    report = {
        "schema_version": 1,
        "status": "verified_native100_table_supplement",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "base_matrix_id": manifest["base_matrix_id"],
        "claim_scope": NATIVE100_CLAIM_SCOPE,
        "validity_label": manifest["validity_label"],
        "manifest_sha256": expected_manifest_sha256,
        "base_manifest_sha256": expected_base_manifest_sha256,
        "base_table_sha256": expected_base_table_sha256,
        "base_row_count": len(base_rows),
        "supplement_row_count": len(supplement_rows),
        "display_row_count": len(rows),
        "task_count": len(task_records),
        "seeds": list(NATIVE100_SEEDS),
        "statistic": "mean_and_sample_standard_deviation",
        "task_records": task_records,
        "outputs": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in outputs.items()
        },
        "caveats": [
            "official_protocol_with_known_overlap",
            "not_protocol_v3_noleak_dedup",
            "ndr_net_is_not_an_original_igstgnn_table4_method",
            "native_recipes_differ_except_for_the_shared_100_epoch_cap",
            "existing_bist_500_epoch_and_ndr_30_epoch_rows_are_preserved",
        ],
    }
    outputs["table4_with_native100_manifest.json"] = _json_bytes(report)
    _publish(Path(output_dir), outputs)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the verified native-recipe 100 epoch Table 4 supplement."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--base-table-csv", required=True, type=Path)
    parser.add_argument("--base-table-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_native100_table_report(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        base_manifest_path=args.base_manifest,
        expected_base_manifest_sha256=args.base_manifest_sha256,
        results_root=args.results_root,
        base_table_csv=args.base_table_csv,
        expected_base_table_sha256=args.base_table_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
