from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST,
    SUPPORT_MANIFEST,
    ExtensionInputs,
    build_extension_invocation,
    load_extension_manifest,
    verify_extension_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_extension_trainer_bootstraps_project_imports_from_script_path() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "scripts/train_protocol_v3_event_extension.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    assert "Train BiST, IGSTGNN, or NDR-Net" in completed.stdout


def test_extension_manifest_is_exact_seed42_nine_task_matrix() -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    assert len(manifest.tasks) == 9
    assert {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in manifest.tasks
    } == {
        (region, method, 42, "event")
        for region in ("Alameda", "ContraCosta", "Orange")
        for method in ("bist", "igstgnn", "ndr_net")
    }


def test_checked_support_and_invocation_bind_the_v3_adapter(tmp_path: Path) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(task for task in manifest.tasks if task.task_id == "v3x002_event_igstgnn_alameda_seed42")
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=True,
            max_microbatch_size=8,
        ),
        tmp_path,
    )
    command = list(invocation.command)
    assert "scripts/train_protocol_v3_event_extension.py" in command
    assert "--igstgnn-data-parallel" in command
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.required_artifacts == (
        "summary.json",
        "ckpt_best.pt",
        "test_predictions.npz",
        "train.log",
    )
    assert invocation.config["support_manifest_sha256"] == support.manifest_sha256
    assert invocation.config["upstream_curriculum_applied"] is True
    assert command[command.index("--traffic-loading") + 1] == "eager"
    assert invocation.config["traffic_loading"] == "eager"


def test_orange_ndr_invocation_records_its_effective_microbatch(tmp_path: Path) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(task for task in manifest.tasks if task.task_id == "v3x009_event_ndr_net_orange_seed42")
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
        ),
        tmp_path,
    )

    command = list(invocation.command)
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.config["max_microbatch_size"] == 8
    assert invocation.config["early_stopping_patience"] is None


def test_orange_igstgnn_invocation_records_its_effective_microbatch(
    tmp_path: Path,
) -> None:
    manifest = load_extension_manifest(CANONICAL_MANIFEST)
    support = verify_extension_support(
        support_manifest_path=SUPPORT_MANIFEST,
        support_root=(
            PROJECT_ROOT
            / "outputs/reproduction/igstgnn_table4_official_v1/data/official_splits"
        ),
        node_order=PROJECT_ROOT / "archive/node_order.npy",
    )
    task = next(
        task
        for task in manifest.tasks
        if task.task_id == "v3x008_event_igstgnn_orange_seed42"
    )
    invocation = build_extension_invocation(
        task,
        manifest,
        ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=False,
            max_microbatch_size=0,
        ),
        tmp_path,
    )

    command = list(invocation.command)
    assert command[command.index("--max-microbatch-size") + 1] == "8"
    assert invocation.config["max_microbatch_size"] == 8
    assert invocation.config["upstream_curriculum_applied"] is True
