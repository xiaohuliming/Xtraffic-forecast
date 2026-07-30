from __future__ import annotations

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
from .paper_reference import METRIC_COLUMNS
from .secure_io import read_verified_regular_file


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASE_HEADER = ("region", "method", *METRIC_COLUMNS)
_STAT_COLUMNS = tuple(
    f"{column}_sample_std" for column in METRIC_COLUMNS
)
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
}
_REGION_LABELS = {
    "Alameda": "Alameda",
    "Contra_Costa": "Contra Costa",
    "Orange": "Orange",
}


class NDRExtensionMergeError(ValueError):
    """Raised when the verified base and extension cannot be merged."""


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NDRExtensionMergeError(f"{label} must be a SHA-256 digest")
    return value


def _read_bytes(
    path: Path,
    *,
    purpose: str,
    expected_sha256: str,
    max_bytes: int,
) -> bytes:
    _require_sha256(expected_sha256, label=f"{purpose} SHA-256")
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
        raise NDRExtensionMergeError(
            f"cannot read {purpose}: {path}"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise NDRExtensionMergeError(f"{purpose} SHA-256 mismatch")
    return payload


def _positive_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise NDRExtensionMergeError(f"{label} is not numeric") from error
    if not math.isfinite(number) or number <= 0:
        raise NDRExtensionMergeError(
            f"{label} must be finite and positive"
        )
    return number


def _nonnegative_float(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise NDRExtensionMergeError(
            f"{label} must be finite and nonnegative"
        )
    return float(value)


def _load_base_rows(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], bytes]:
    payload = _read_bytes(
        path,
        purpose="verified Table 4 base CSV",
        expected_sha256=expected_sha256,
        max_bytes=2 * 1024 * 1024,
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise NDRExtensionMergeError(
            "verified Table 4 base CSV is not UTF-8"
        ) from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != _BASE_HEADER:
        raise NDRExtensionMergeError(
            "verified Table 4 base CSV header is invalid"
        )
    rows: list[dict[str, Any]] = []
    expected_keys = [
        (region, method) for region in REGIONS for method in METHODS
    ]
    for raw, expected_key in zip(reader, expected_keys, strict=False):
        if (raw.get("region"), raw.get("method")) != expected_key:
            raise NDRExtensionMergeError(
                "verified Table 4 base CSV order is not canonical"
            )
        rows.append(
            {
                "region": raw["region"],
                "method": raw["method"],
                **{
                    column: _positive_float(
                        raw[column],
                        label=f"{raw['region']} {raw['method']} {column}",
                    )
                    for column in METRIC_COLUMNS
                },
            }
        )
    if len(rows) != 39 or next(reader, None) is not None:
        raise NDRExtensionMergeError(
            "verified Table 4 base CSV must contain 39 rows"
        )
    return rows, payload


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NDRExtensionMergeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise NDRExtensionMergeError(f"non-finite JSON constant: {value}")


def _load_ndr_rows(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    payload = _read_bytes(
        path,
        purpose="verified NDR extension index",
        expected_sha256=expected_sha256,
        max_bytes=8 * 1024 * 1024,
    )
    try:
        index = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except NDRExtensionMergeError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise NDRExtensionMergeError(
            "verified NDR extension index is not strict JSON"
        ) from error
    if not isinstance(index, dict):
        raise NDRExtensionMergeError(
            "verified NDR extension index must contain an object"
        )
    expected_identity = {
        "schema_version": 1,
        "status": "verified_ndr_extension_report",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "claim_scope": "same_protocol_project_method_extension",
        "validity_label": "official_protocol_with_known_overlap",
        "task_count": 9,
        "region_count": 3,
        "seed_count": 3,
        "seeds": [42, 1, 2],
    }
    for key, expected in expected_identity.items():
        if index.get(key) != expected:
            raise NDRExtensionMergeError(
                f"unexpected NDR extension index field: {key}"
            )
    summaries = index.get("region_summaries")
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise NDRExtensionMergeError(
            "NDR extension index must contain three region summaries"
        )
    rows = []
    for summary, region in zip(summaries, REGIONS, strict=True):
        if (
            not isinstance(summary, dict)
            or summary.get("region") != region
            or summary.get("method") != "ndr_net"
            or summary.get("seeds") != [42, 1, 2]
        ):
            raise NDRExtensionMergeError(
                "NDR region summary order or identity is invalid"
            )
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise NDRExtensionMergeError(
                f"{region} NDR metrics are missing"
            )
        row: dict[str, Any] = {
            "region": region,
            "method": "ndr_net",
        }
        for column, (group, name) in _METRIC_PATHS.items():
            group_value = metrics.get(group)
            statistic = (
                group_value.get(name)
                if isinstance(group_value, dict)
                else None
            )
            if not isinstance(statistic, dict):
                raise NDRExtensionMergeError(
                    f"{region} NDR metric {column} is missing"
                )
            row[column] = _positive_float(
                statistic.get("mean"),
                label=f"{region} NDR {column} mean",
            )
            row[f"{column}_sample_std"] = _nonnegative_float(
                statistic.get("sample_std"),
                label=f"{region} NDR {column} sample standard deviation",
            )
        rows.append(row)
    return rows, index, payload


def _combined_csv_bytes(
    base_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(*_BASE_HEADER, *_STAT_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in base_rows:
        writer.writerow(
            {
                "region": row["region"],
                "method": row["method"],
                **{
                    column: f"{row[column]:.6f}"
                    for column in METRIC_COLUMNS
                },
                **{column: "" for column in _STAT_COLUMNS},
            }
        )
    for row in extension_rows:
        writer.writerow(
            {
                "region": row["region"],
                "method": row["method"],
                **{
                    column: f"{row[column]:.6f}"
                    for column in METRIC_COLUMNS
                },
                **{
                    column: f"{row[column]:.6f}"
                    for column in _STAT_COLUMNS
                },
            }
        )
    return buffer.getvalue().encode("utf-8")


def _markdown_bytes(
    base_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
    *,
    base_sha256: str,
    extension_sha256: str,
) -> bytes:
    lines = [
        "# IGSTGNN Table 4 复现与 NDR-Net 同协议扩展",
        "",
        "前 13 个方法来自公开材料最大忠实复现，每个方法为论文协议单次运行。",
        "NDR-Net* 是项目方法扩展，报告种子 42、1、2 的均值和样本标准差。",
        "全部结果保留官方协议中的已知预测时间重叠，不是无泄露 v3 结果。",
        "",
        f"39 项复现表 SHA-256：`{base_sha256}`。",
        f"NDR 扩展索引 SHA-256：`{extension_sha256}`。",
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
    extension_by_region = {
        row["region"]: row for row in extension_rows
    }
    for region in REGIONS:
        lines.extend(
            [
                f"## {_REGION_LABELS[region]}",
                "",
                "| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] + ["---:"] * 12) + "|",
            ]
        )
        for row in base_rows:
            if row["region"] != region:
                continue
            values = [
                _METHOD_LABELS[row["method"]],
                *[f"{row[column]:.2f}" for column in METRIC_COLUMNS],
            ]
            lines.append("| " + " | ".join(values) + " |")
        extension = extension_by_region[region]
        values = ["NDR-Net*"]
        for column in METRIC_COLUMNS:
            values.append(
                f"{extension[column]:.2f} ± "
                f"{extension[f'{column}_sample_std']:.2f}"
            )
        lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _latex_bytes(
    base_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
) -> bytes:
    lines = [
        r"\begin{sidewaystable*}[t]",
        r"\centering",
        r"\caption{IGSTGNN Table 4 public-material reproduction with "
        r"the NDR-Net project-method extension. NDR-Net reports mean "
        r"$\pm$ sample standard deviation over seeds 42, 1, and 2.}",
        r"\label{tab:igstgnn-table4-with-ndr-extension}",
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
    extension_by_region = {
        row["region"]: row for row in extension_rows
    }
    for region_index, region in enumerate(REGIONS):
        region_rows = [row for row in base_rows if row["region"] == region]
        for row_index, row in enumerate(region_rows):
            data_cell = (
                rf"\multirow{{14}}{{*}}{{{_REGION_LABELS[region]}}}"
                if row_index == 0
                else ""
            )
            values = " & ".join(
                f"{row[column]:.2f}" for column in METRIC_COLUMNS
            )
            method = _METHOD_LABELS[row["method"]].replace(
                "D²STGNN",
                r"D$^2$STGNN",
            )
            lines.append(f"{data_cell} & {method} & {values} \\\\")
        extension = extension_by_region[region]
        values = " & ".join(
            f"${extension[column]:.2f} \\pm "
            f"{extension[f'{column}_sample_std']:.2f}$"
            for column in METRIC_COLUMNS
        )
        lines.append(f" & NDR-Net$^*$ & {values} \\\\")
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
        raise NDRExtensionMergeError(
            f"combined output directory already exists: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (
        f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        staging.mkdir()
        for name, payload in files.items():
            path = staging / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise


def build_table4_with_ndr_extension(
    *,
    base_table_csv: Path | str,
    expected_base_table_sha256: str,
    ndr_extension_index: Path | str,
    expected_ndr_index_sha256: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Append three NDR mean and sample-standard-deviation rows."""

    base_rows, _ = _load_base_rows(
        Path(base_table_csv),
        expected_sha256=expected_base_table_sha256,
    )
    extension_rows, extension_index, _ = _load_ndr_rows(
        Path(ndr_extension_index),
        expected_sha256=expected_ndr_index_sha256,
    )
    combined_csv = _combined_csv_bytes(base_rows, extension_rows)
    markdown = _markdown_bytes(
        base_rows,
        extension_rows,
        base_sha256=expected_base_table_sha256,
        extension_sha256=expected_ndr_index_sha256,
    )
    latex = _latex_bytes(base_rows, extension_rows)
    outputs = {
        "table4_with_ndr_extension.csv": combined_csv,
        "table4_with_ndr_extension.md": markdown,
        "table4_with_ndr_extension.tex": latex,
    }
    report = {
        "schema_version": 1,
        "status": "verified_table4_with_ndr_extension",
        "protocol_id": "igstgnn_table4_official_v1",
        "base_matrix_id": "igstgnn_table4_public_repro_39_v2",
        "extension_matrix_id": "igstgnn_table4_ndr_extension_9_v1",
        "base_claim_scope": "public_material_maximum_fidelity_reproduction",
        "extension_claim_scope": (
            "same_protocol_project_method_extension"
        ),
        "validity_label": "official_protocol_with_known_overlap",
        "base_table_sha256": expected_base_table_sha256,
        "ndr_extension_index_sha256": expected_ndr_index_sha256,
        "ndr_receipt_sha256": extension_index["receipt_sha256"],
        "base_row_count": len(base_rows),
        "extension_row_count": len(extension_rows),
        "display_row_count": len(base_rows) + len(extension_rows),
        "ndr_seeds": [42, 1, 2],
        "ndr_statistic": "mean_and_sample_standard_deviation",
        "outputs": {
            name: {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in outputs.items()
        },
        "caveats": [
            "official_protocol_with_known_overlap",
            "not_protocol_v3_noleak_dedup",
            "ndr_net_is_not_an_original_igstgnn_table4_method",
            "base_rows_are_single_run_values",
            "ndr_rows_are_three_seed_mean_and_sample_standard_deviation",
        ],
    }
    outputs["table4_with_ndr_extension_manifest.json"] = _json_bytes(report)
    _publish(Path(output_dir), outputs)
    return report
