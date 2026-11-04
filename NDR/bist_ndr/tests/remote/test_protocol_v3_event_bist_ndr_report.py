from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


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
    from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (
        CANONICAL_MANIFEST,
        load_bist_ndr_manifest,
    )

    manifest = load_bist_ndr_manifest(CANONICAL_MANIFEST)
    base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    bundle_id = base_manifest["cache_bundle_id"]
    bundle_digest = base_manifest["cache_bundle_digest"]
    export_root = tmp_path / "published"
    files: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for offset, task in enumerate(manifest.tasks):
        relative = (
            Path("sources/gpu00/results") / task.output_dir / "summary.json"
        )
        path = export_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = 9.0 + offset
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
            "config_hash": f"{offset + 1:064x}",
            "all": value + 0.5,
            "h3": value + 0.2,
            "h6": value + 0.5,
            "h12": value + 1.1,
            "mae_h": [value + index / 10 for index in range(12)],
            "rmse": value + 10.5,
            "rmse_h": [value + 10 + index / 10 for index in range(12)],
            "mape": value + 5.5,
            "mape_h": [value + 5 + index / 10 for index in range(12)],
            "valid_count": 100,
        }
        path.write_text(json.dumps(summary), encoding="utf-8")
        files.append(
            {
                "path": relative.as_posix(),
                "source_path": f"D:/results/{task.output_dir.as_posix()}/summary.json",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_id": "gpu00",
            }
        )
        tasks.append(
            {
                "task_id": task.task_id,
                "source_id": "gpu00",
                "state": "completed",
                "code_sha": CODE_SHA,
                "config_hash": summary["config_hash"],
                "invocation_config": {"claim_scope": "same_protocol_project_method_extension"},
            }
        )
    report = {
        "status": "verified",
        "runner_kind": "event_bist_ndr",
        "matrix_id": "protocol_v3_event_bist_ndr_5080_v1",
        "host_id": "event_rtx5080_bist_ndr_v1",
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": bundle_id,
        "cache_bundle_digest": bundle_digest,
        "canonical_manifest_sha256": _sha256(CANONICAL_MANIFEST),
        "validated_tasks": 3,
        "task_state_counts": {"completed": 3},
        "tasks": tasks,
        "sources": [
            {
                "source_id": "gpu00",
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


def test_verified_bist_ndr_supplement_appends_one_row_per_region(
    tmp_path: Path,
) -> None:
    from scripts.protocol_v3_event_bist_ndr_report import (
        build_bist_ndr_supplement,
    )

    postflight, export_root, digest = _make_verified_export(tmp_path)
    output = tmp_path / "table45"

    manifest = build_bist_ndr_supplement(
        base_csv_path=BASE_CSV,
        base_manifest_path=BASE_MANIFEST,
        postflight_path=postflight,
        postflight_sha256=digest,
        published_root=export_root,
        output_dir=output,
    )

    with (output / "table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert manifest["status"] == "verified_no_leak_bist_ndr_supplement_generated"
    assert manifest["validity_label"] == (
        "target_disjoint_event_no_leak_host_and_code_stratified"
    )
    assert len(rows) == 45
    for region in ("Alameda", "Contra Costa", "Orange"):
        methods = [row["method"] for row in rows if row["region"] == region]
        assert methods[-2:] == ["NDR-Net", "BiST-NDR"]
    alameda = next(
        row
        for row in rows
        if row["region"] == "Alameda" and row["method"] == "BiST-NDR"
    )
    assert alameda["h3_mae"] == "9.200000"
    assert alameda["h6_rmse"] == "19.500000"
    assert alameda["h12_mape_percent"] == "15.100000"
    assert alameda["avg_mae"] == "9.500000"


def test_bist_ndr_supplement_rejects_changed_published_summary(
    tmp_path: Path,
) -> None:
    import pytest

    from scripts.protocol_v3_event_bist_ndr_report import (
        BistNdrSupplementError,
        build_bist_ndr_supplement,
    )

    postflight, export_root, digest = _make_verified_export(tmp_path)
    summary = next(export_root.rglob("summary.json"))
    summary.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BistNdrSupplementError, match="published file .* mismatch"):
        build_bist_ndr_supplement(
            base_csv_path=BASE_CSV,
            base_manifest_path=BASE_MANIFEST,
            postflight_path=postflight,
            postflight_sha256=digest,
            published_root=export_root,
            output_dir=tmp_path / "rejected",
        )


def test_verified_45_row_supplement_builds_a_pdf(tmp_path: Path) -> None:
    from scripts.protocol_v3_event_bist_ndr_pdf import build_pdf
    from scripts.protocol_v3_event_bist_ndr_report import (
        build_bist_ndr_supplement,
    )

    postflight, export_root, digest = _make_verified_export(tmp_path)
    output = tmp_path / "table45"
    build_bist_ndr_supplement(
        base_csv_path=BASE_CSV,
        base_manifest_path=BASE_MANIFEST,
        postflight_path=postflight,
        postflight_sha256=digest,
        published_root=export_root,
        output_dir=output,
    )

    pdf = output / "protocol_v3_no_leak_comparison_45_bist_ndr.pdf"
    build_pdf(output / "table.csv", output / "manifest.json", pdf)

    assert pdf.read_bytes().startswith(b"%PDF-")
    assert pdf.stat().st_size > 10_000
