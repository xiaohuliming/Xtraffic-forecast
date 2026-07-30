from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.reproduction.igstgnn_table4.composite_report import (
    build_table4_reproduction_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
REFERENCE = MANIFEST.with_name("paper_table4.reference_v1.csv")
SOURCE_LOCK = MANIFEST.with_name("paper_source.lock.json")


def _write_verified_index(tmp_path: Path) -> tuple[Path, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = []
    for number, task in enumerate(manifest["tasks"], start=1):
        base = float(number)
        tasks.append(
            {
                "task_id": task["task_id"],
                "method": task["method"],
                "region": task["region"],
                "seed": task["seed"],
                "source_id": "fixture",
                "source_host_id": "fixture_host",
                "source_execution_layout": "fixture",
                "relative_task_root": f"tasks/{task['task_id']}",
                "training_code_sha256": "1" * 64,
                "summary_sha256": "2" * 64,
                "predictions_sha256": "3" * 64,
                "provenance_sha256": "4" * 64,
                "valid_count": 12,
                "prediction_shape": [1, 12, 1],
                "metrics": {
                    "h3": {
                        "mae": base + 0.01,
                        "rmse": base + 0.02,
                        "mape_percent": base + 0.03,
                    },
                    "h6": {
                        "mae": base + 0.11,
                        "rmse": base + 0.12,
                        "mape_percent": base + 0.13,
                    },
                    "h12": {
                        "mae": base + 0.21,
                        "rmse": base + 0.22,
                        "mape_percent": base + 0.23,
                    },
                    "average": {
                        "mae": base + 0.31,
                        "rmse": base + 0.32,
                        "mape_percent": base + 0.33,
                    },
                },
            }
        )
    value = {
        "schema_version": 1,
        "status": "verified",
        "matrix_validity_status": (
            "verified_official_protocol_with_known_overlap_39"
        ),
        "composite_id": "igstgnn_table4_public_repro_39_composite_v1",
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "igstgnn_table4_public_repro_39_v2",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_spec_sha256": "5" * 64,
        "control_git_sha": "6" * 40,
        "validator_sha256": "7" * 64,
        "claim_scope": "public_material_maximum_fidelity_reproduction",
        "validity_label": "official_protocol_with_known_overlap",
        "execution_layout": "three_source_published_composite",
        "source_count": 3,
        "task_count": 39,
        "file_count": 199,
        "source_bytes": 123,
        "sources": [],
        "tasks": tasks,
        "verified_at": "2026-07-30T00:00:00Z",
        "caveats": [],
    }
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path = tmp_path / "composite_index.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_report_generates_same_shape_table_and_paper_deltas(
    tmp_path: Path,
) -> None:
    index_path, index_sha256 = _write_verified_index(tmp_path)
    output_dir = tmp_path / "report"

    report = build_table4_reproduction_report(
        composite_index_path=index_path,
        expected_composite_index_sha256=index_sha256,
        reference_path=REFERENCE,
        source_lock_path=SOURCE_LOCK,
        output_dir=output_dir,
    )

    assert report["status"] == "verified_reproduction_table"
    assert report["row_count"] == 39
    assert report["composite_index_sha256"] == index_sha256
    reproduced_rows = list(
        csv.DictReader(
            (output_dir / "table4_reproduced.csv").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    )
    comparison_rows = list(
        csv.DictReader(
            (output_dir / "table4_paper_comparison.csv").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    )
    assert len(reproduced_rows) == 39
    assert len(comparison_rows) == 39
    orange_sttn = next(
        row
        for row in comparison_rows
        if row["region"] == "Orange" and row["method"] == "sttn"
    )
    assert orange_sttn["reproduced_avg_mae"] == "34.310000"
    assert orange_sttn["paper_avg_mae"] == "13.820000"
    assert orange_sttn["delta_avg_mae"] == "20.490000"
    assert (output_dir / "table4_reproduced.md").is_file()
    assert (output_dir / "table4_reproduced.tex").is_file()
    assert (output_dir / "table4_report_manifest.json").is_file()
