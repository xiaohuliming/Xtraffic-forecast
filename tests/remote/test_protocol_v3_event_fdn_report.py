from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = (
    PROJECT_ROOT
    / "outputs/protocol_v3_noleak_dedup/event_full42_20260810/verified_table"
)
BASE_CSV = BASE_DIR / "table.csv"
BASE_MANIFEST = BASE_DIR / "manifest.json"
CODE_SHA = "c" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _make_verified_export(tmp_path: Path) -> tuple[Path, Path, str]:
    from scripts.remote.windows.protocol_v3_event_fdn_runner import (
        DEFAULT_MANIFEST,
        load_fdn_manifest,
    )

    manifest = load_fdn_manifest(DEFAULT_MANIFEST)
    base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    bundle_id = base_manifest["cache_bundle_id"]
    bundle_digest = base_manifest["cache_bundle_digest"]
    export_root = tmp_path / "published"
    files: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for task in manifest.tasks:
        result_root = Path("sources/fdn5080/results") / task.output_dir
        result_dir = export_root / result_root
        result_dir.mkdir(parents=True, exist_ok=True)
        error_h = np.arange(1, 13, dtype=np.float32).reshape(1, 12, 1)
        actual = np.full((1, 12, 2), 10.0, dtype=np.float32)
        pred = actual + error_h
        prediction = result_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction,
            region_code=np.asarray(
                [{"Alameda": 0, "ContraCosta": 1, "Orange": 2}[task.region]],
                dtype=np.int64,
            ),
            sample_start=np.asarray([100], dtype=np.int64),
            region_node_idx=np.asarray([0, 1], dtype=np.int64),
            pred_raw_flow=pred,
            actual_future_flow=actual,
            y_mask_flow=np.ones_like(actual, dtype=np.bool_),
            affected_mask=np.asarray([[True, False]], dtype=np.bool_),
        )
        summary = {
            "task_id": task.task_id,
            "region": task.region,
            "model": task.model,
            "variant": task.variant,
            "seed": task.seed,
            "protocol_version": manifest.protocol_version,
            "sampling_protocol": task.sampling_protocol,
            "cache_schema_version": manifest.cache_schema_version,
            "cache_bundle_id": bundle_id,
            "cache_bundle_digest": bundle_digest,
            "code_sha": CODE_SHA,
            "config_hash": f"{len(tasks) + 1:064x}",
            "all": 6.5,
            "affected": 6.5,
            "unaffected": 6.5,
            "valid_count": 24,
        }
        summary_path = result_dir / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        for path in (summary_path, prediction):
            relative = path.relative_to(export_root).as_posix()
            files.append(
                {
                    "path": relative,
                    "source_path": f"D:/results/{task.output_dir.as_posix()}/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "source_id": "fdn5080",
                }
            )
        tasks.append(
            {
                "task_id": task.task_id,
                "source_id": "fdn5080",
                "state": "completed",
                "code_sha": CODE_SHA,
                "config_hash": summary["config_hash"],
                "invocation_config": {
                    "sampling_protocol": "event",
                    "decomp_mode": "learnable",
                    "K": 3,
                    "epochs": 30,
                    "patience": 6,
                    "batch_size": 32,
                },
            }
        )
    report = {
        "status": "verified",
        "runner_kind": "event_fdn",
        "matrix_id": "protocol_v3_event_fdn_5080_v1",
        "host_id": "event_rtx5080_fdn",
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": bundle_id,
        "cache_bundle_digest": bundle_digest,
        "canonical_manifest_sha256": _sha256(DEFAULT_MANIFEST),
        "validated_tasks": 3,
        "task_state_counts": {"completed": 3},
        "tasks": tasks,
        "sources": [
            {
                "source_id": "fdn5080",
                "code_sha": CODE_SHA,
                "task_ids": [task.task_id for task in manifest.tasks],
                "validated_tasks": 3,
            }
        ],
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files_digest": _canonical_digest(files),
        "files": files,
    }
    report_path = tmp_path / "postflight.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, export_root, _sha256(report_path)


def test_verified_fdn_supplement_appends_one_row_per_region(
    tmp_path: Path,
) -> None:
    from scripts.protocol_v3_event_fdn_report import build_fdn_supplement

    postflight, export_root, digest = _make_verified_export(tmp_path)
    output = tmp_path / "table45"

    manifest = build_fdn_supplement(
        base_csv_path=BASE_CSV,
        base_manifest_path=BASE_MANIFEST,
        postflight_path=postflight,
        postflight_sha256=digest,
        published_root=export_root,
        output_dir=output,
    )

    with (output / "table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert manifest["status"] == "verified_no_leak_fdn_supplement_generated"
    assert manifest["validity_label"] == (
        "target_disjoint_event_no_leak_host_and_code_stratified"
    )
    assert len(rows) == 45
    for region in ("Alameda", "Contra Costa", "Orange"):
        methods = [row["method"] for row in rows if row["region"] == region]
        assert methods[-4:] == ["FDN", "BiST", "IGSTGNN", "NDR-Net"]
    alameda = next(
        row
        for row in rows
        if row["region"] == "Alameda" and row["method"] == "FDN"
    )
    assert alameda["h3_mae"] == "3.000000"
    assert alameda["h6_rmse"] == "6.000000"
    assert alameda["h12_mape_percent"] == "120.000000"
    assert alameda["avg_mae"] == "6.500000"


def test_fdn_supplement_rejects_changed_prediction_artifact(
    tmp_path: Path,
) -> None:
    import pytest

    from scripts.protocol_v3_event_fdn_report import (
        FdnSupplementError,
        build_fdn_supplement,
    )

    postflight, export_root, digest = _make_verified_export(tmp_path)
    prediction = next(export_root.rglob("test_predictions.npz"))
    prediction.write_bytes(b"changed")

    with pytest.raises(FdnSupplementError, match="published file .* mismatch"):
        build_fdn_supplement(
            base_csv_path=BASE_CSV,
            base_manifest_path=BASE_MANIFEST,
            postflight_path=postflight,
            postflight_sha256=digest,
            published_root=export_root,
            output_dir=tmp_path / "rejected",
        )


def test_verified_fdn_supplement_builds_a_pdf(tmp_path: Path) -> None:
    from scripts.protocol_v3_event_fdn_pdf import build_pdf
    from scripts.protocol_v3_event_fdn_report import build_fdn_supplement

    postflight, export_root, digest = _make_verified_export(tmp_path)
    output = tmp_path / "table45"
    build_fdn_supplement(
        base_csv_path=BASE_CSV,
        base_manifest_path=BASE_MANIFEST,
        postflight_path=postflight,
        postflight_sha256=digest,
        published_root=export_root,
        output_dir=output,
    )

    pdf = output / "protocol_v3_no_leak_comparison_45_fdn.pdf"
    build_pdf(output / "table.csv", output / "manifest.json", pdf)

    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 10_000
