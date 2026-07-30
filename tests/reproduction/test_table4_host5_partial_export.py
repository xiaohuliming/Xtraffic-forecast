from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.host5_partial_export_postflight import (
    build_host5_partial_export_postflight,
    load_host5_partial_export_plan,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
PLAN = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "host5_partial_export_12_v1.json"
)
CONTROLLER = (
    PROJECT_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "igstgnn_table4_export_host5_partial12.sh"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)


def _write_host5_results(tmp_path: Path) -> Path:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    results_base = tmp_path / "results"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    train_log = tmp_path / "train.log"
    train_log.write_text("complete\n", encoding="utf-8")
    labels = np.arange(12, dtype=np.float32).reshape(1, 12, 1)
    predictions = labels + np.float32(0.25)
    for assignment in plan["tasks"]:
        task = task_by_id[assignment["task_id"]]
        write_completed_task_artifacts(
            target_dir=(
                results_base
                / assignment["results_root"]
                / task["output_dir"]
            ),
            task=task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance={
                "protocol_id": "igstgnn_table4_official_v1",
                "training_code_sha256": "3" * 64,
                "host": {
                    "hostname": plan["hostname"],
                    "system": "Linux",
                    "release": "test",
                    "machine": "x86_64",
                },
                "device": {
                    "requested": "cuda:0",
                    "type": "cuda",
                    "resolved_index": 0,
                    "name": plan["gpu_sku"],
                },
                "runtime": {
                    "python": "3.11.15",
                    "torch": "2.3.1+cu121",
                    "numpy": "1.24.4",
                },
            },
        )
    return results_base


def test_host5_partial_export_plan_is_exactly_the_twelve_completed_tasks() -> None:
    plan = load_host5_partial_export_plan(
        PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
    )

    task_ids = [assignment["task_id"] for assignment in plan["tasks"]]
    assert task_ids == [
        "027_orange_hl",
        "028_orange_lstm",
        "029_orange_dcrnn",
        "030_orange_agcrn",
        "031_orange_stgcn",
        "032_orange_gwnet",
        "033_orange_astgcn",
        "035_orange_dstagnn",
        "036_orange_dgcrn",
        "037_orange_d2stgnn",
        "038_orange_bist",
        "039_orange_igstgnn",
    ]
    assert "034_orange_sttn" not in task_ids
    assert plan["execution_layout"] == "single_host_partial_original_export"


def test_host5_partial_postflight_exports_only_sixty_task_files(tmp_path) -> None:
    results_base = _write_host5_results(tmp_path)
    report_path = tmp_path / "postflight.json"

    report = build_host5_partial_export_postflight(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
        results_base=results_base,
        report_path=report_path,
    )

    assert report["status"] == "verified"
    assert report["validated_tasks"] == 12
    assert report["validated_shards"] == 5
    assert report["file_count"] == 60
    assert "034_orange_sttn" not in {
        task["task_id"] for task in report["tasks"]
    }


def test_host5_partial_postflight_rejects_an_extra_task_file(tmp_path) -> None:
    results_base = _write_host5_results(tmp_path)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    first = plan["tasks"][0]
    task_dir = (
        results_base
        / first["results_root"]
        / "tasks"
        / first["task_id"]
    )
    (task_dir / "unexpected.txt").write_text("bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match="allowlist"):
        build_host5_partial_export_postflight(
            plan_path=PLAN,
            manifest_path=MANIFEST,
            expected_manifest_sha256=MANIFEST_SHA256,
            results_base=results_base,
            report_path=tmp_path / "postflight.json",
        )


def test_host5_partial_controller_can_only_export_and_shutdown_via_finalizer() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert (
        "PLAN_SHA256=cd81652493690d1d329f5eecd45fe541"
        "c9de7db26c93650fc490a2507616845d"
    ) in source
    assert "034_orange_sttn" in source
    assert "task034 must remain absent" in source
    assert "recovery_task_runner" in source
    assert "EXPECTED_PLAN_ID=\"$PLAN_ID\"" in source
    assert (
        'EXPECTED_EXECUTION_LAYOUT="single_host_partial_original_export"'
        in source
    )
    assert 'SHUTDOWN_BIN="/usr/bin/shutdown"' in source
    assert "bash \"$FINALIZER\"" in source
    assert '"$SHUTDOWN_BIN" -h now' not in source
