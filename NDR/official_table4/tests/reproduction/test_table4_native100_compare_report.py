from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.native100_compare_manifest import (
    load_native100_compare_manifest,
)
from scripts.reproduction.igstgnn_table4.native100_compare_report import (
    Native100ReportError,
    build_native100_table_report,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
MANIFEST = CONFIG_ROOT / "matrix.native100_compare_v1.json"
BASE_MANIFEST = CONFIG_ROOT / "matrix.public_material_v2.json"
BASE_MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
)
BASE_TABLE = (
    PROJECT_ROOT
    / "outputs"
    / "reproduction"
    / "igstgnn_table4_official_v1"
    / "table4_with_ndr_verified_v1"
    / "table4_with_ndr_extension.csv"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_results(root: Path, *, count: int = 18) -> None:
    manifest = load_native100_compare_manifest(
        MANIFEST,
        expected_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
    )
    checkpoint = root / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint\n")
    train_log = root / "train.log"
    train_log.write_text("complete\n", encoding="utf-8")
    seed_offsets = {42: 1.0, 1: 2.0, 2: 3.0}
    method_offsets = {"bist": 0.0, "ndr_net": 3.0}
    region_offsets = {"Alameda": 0.0, "Contra_Costa": 1.0, "Orange": 2.0}
    labels = np.full((2, 12, 3), 10.0, dtype=np.float32)
    for task in manifest["tasks"][:count]:
        error = (
            seed_offsets[task["seed"]]
            + method_offsets[task["method"]]
            + region_offsets[task["region"]]
        )
        predictions = labels + np.float32(error)
        write_completed_task_artifacts(
            target_dir=root / task["output_dir"],
            task=task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance={
                "protocol_id": "igstgnn_table4_official_v1",
                "training_code_sha256": "3" * 64,
            },
            claim_scope="same_table_native_recipe_100_epoch_supplement",
        )


def _build(results_root: Path, output_dir: Path):
    return build_native100_table_report(
        manifest_path=MANIFEST,
        expected_manifest_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
        results_root=results_root,
        base_table_csv=BASE_TABLE,
        expected_base_table_sha256=_sha256(BASE_TABLE),
        output_dir=output_dir,
    )


def test_native100_report_appends_six_same_table_rows(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_results(results_root)

    report = _build(results_root, tmp_path / "report")

    assert report["status"] == "verified_native100_table_supplement"
    assert report["base_row_count"] == 42
    assert report["supplement_row_count"] == 6
    assert report["display_row_count"] == 48
    assert report["seeds"] == [42, 1, 2]
    with (tmp_path / "report" / "table4_with_native100.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert rows[-6]["region"] == "Alameda"
    assert rows[-6]["method"] == "bist_native100"
    assert rows[-6]["avg_mae"] == "2.000000"
    assert rows[-6]["avg_mae_sample_std"] == "1.000000"
    assert rows[-5]["method"] == "ndr_net_native100"
    markdown = (tmp_path / "report" / "table4_with_native100.md").read_text(
        encoding="utf-8"
    )
    assert "BiST 100ep*" in markdown
    assert "NDR-Net 100ep*" in markdown
    assert "2.00 ± 1.00" in markdown
    assert (tmp_path / "report" / "table4_with_native100.tex").is_file()


def test_native100_report_refuses_an_incomplete_matrix(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_results(results_root, count=17)

    with pytest.raises(Native100ReportError, match="complete 18-task matrix"):
        _build(results_root, tmp_path / "report")


def test_native100_report_reads_verified_composite_index(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    _write_results(combined)
    manifest = load_native100_compare_manifest(
        MANIFEST,
        expected_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
    )
    composite_root = tmp_path / "hosts"
    records = []
    for task in manifest["tasks"]:
        host_id = "orange" if task["region"] == "Orange" else "non_orange"
        target = composite_root / host_id / "results" / task["output_dir"]
        target.parent.mkdir(parents=True, exist_ok=True)
        (combined / task["output_dir"]).rename(target)
        records.append(
            {
                "task_id": task["task_id"],
                "host_id": host_id,
                "source_root_relative": host_id,
                "task_relative": f"results/{task['output_dir']}",
                "summary_sha256": _sha256(target / "summary.json"),
            }
        )
    index = {
        "schema_version": 1,
        "status": "verified_native100_dual_host_18",
        "protocol_id": manifest["protocol_id"],
        "matrix_id": manifest["matrix_id"],
        "manifest_sha256": _sha256(MANIFEST),
        "claim_scope": "same_table_native_recipe_100_epoch_supplement",
        "task_count": 18,
        "tasks": records,
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    report = build_native100_table_report(
        manifest_path=MANIFEST,
        expected_manifest_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        expected_base_manifest_sha256=BASE_MANIFEST_SHA256,
        results_root=None,
        base_table_csv=BASE_TABLE,
        expected_base_table_sha256=_sha256(BASE_TABLE),
        output_dir=tmp_path / "report",
        composite_index_path=index_path,
        expected_composite_index_sha256=_sha256(index_path),
        composite_root=composite_root,
    )

    assert report["task_count"] == 18
    assert report["composite_index_sha256"] == _sha256(index_path)
