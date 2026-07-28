from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4 import matrix_runner
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    write_completed_task_artifacts,
)


def _task(number: int, method: str = "hl") -> dict[str, object]:
    task_id = f"{number:03d}_alameda_{method}"
    return {
        "task_id": task_id,
        "method": method,
        "region": "Alameda",
        "seed": 2023,
        "source_id": "largest_b7c6ff5",
        "profile_id": f"{method}_public_material_v2",
        "config_sha256": format(number, "x") * 64,
        "split_bundle_manifest_sha256": "b" * 64,
        "output_dir": f"tasks/{task_id}",
    }


def _manifest(tasks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": "test_table4_matrix",
        "manifest_sha256": "a" * 64,
        "tasks": tasks,
    }


def _write_valid_result(
    *,
    tmp_path: Path,
    results_root: Path,
    task: dict[str, object],
) -> None:
    checkpoint = tmp_path / f"{task['task_id']}.checkpoint.pt"
    train_log = tmp_path / f"{task['task_id']}.train.log"
    checkpoint.write_bytes(b"checkpoint\n")
    train_log.write_text("finished\n", encoding="utf-8")
    write_completed_task_artifacts(
        target_dir=results_root / task["output_dir"],
        task=task,
        predictions=np.ones((2, 12, 1), dtype=np.float32),
        labels=np.full((2, 12, 1), 2, dtype=np.float32),
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "f" * 64,
        },
    )


def test_allowlisted_tasks_run_in_manifest_order_and_write_completed_status(
    tmp_path,
    monkeypatch,
):
    tasks = [_task(1), _task(2, "lstm"), _task(3, "dcrnn")]
    manifest = _manifest(tasks)
    results_root = tmp_path / "results"
    manifest_path = tmp_path / "matrix manifest.json"
    manifest_path.write_text("locked", encoding="utf-8")
    observed_commands: list[list[str]] = []
    observed_process_options: list[dict[str, object]] = []
    validated: list[str] = []

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )

    def fake_run(command, **kwargs):
        command = list(command)
        observed_commands.append(command)
        observed_process_options.append(kwargs)
        task_id = command[command.index("--task-id") + 1]
        (results_root / "tasks" / task_id).mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        matrix_runner,
        "validate_completed_task_artifacts",
        lambda path, expected_task: validated.append(expected_task["task_id"])
        or {"status": "completed_task_artifacts_verified"},
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=manifest_path,
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "split bundles",
        results_root=results_root,
        device="cuda:0",
        task_ids=[tasks[2]["task_id"], tasks[0]["task_id"]],
    )

    assert returncode == 0
    assert [
        command[command.index("--task-id") + 1] for command in observed_commands
    ] == [tasks[0]["task_id"], tasks[2]["task_id"]]
    assert validated == [tasks[0]["task_id"], tasks[2]["task_id"]]
    assert all(
        command[1:3]
        == [
            "-m",
            "scripts.reproduction.igstgnn_table4.task_runner",
        ]
        and command[command.index("--manifest") + 1] == str(manifest_path.resolve())
        and command[command.index("--splits-root") + 1]
        == str((tmp_path / "split bundles").resolve())
        and command[command.index("--results-root") + 1] == str(results_root.resolve())
        and command[command.index("--device") + 1] == "cuda:0"
        for command in observed_commands
    )
    assert observed_process_options == [
        {
            "cwd": str(matrix_runner.PROJECT_ROOT),
            "check": False,
            "shell": False,
        },
        {
            "cwd": str(matrix_runner.PROJECT_ROOT),
            "check": False,
            "shell": False,
        },
    ]
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert status["selected_task_ids"] == [
        tasks[0]["task_id"],
        tasks[2]["task_id"],
    ]
    assert [entry["task_id"] for entry in status["completed"]] == [
        tasks[0]["task_id"],
        tasks[2]["task_id"],
    ]
    assert status["skipped"] == []
    assert status["failed"] == []
    assert status["current"] is None
    assert status["command"] == observed_commands[-1]
    assert {
        key: status[key]
        for key in (
            "selected_task_count",
            "completed_count",
            "skipped_count",
            "failed_count",
            "verified_task_count",
        )
    } == {
        "selected_task_count": 2,
        "completed_count": 2,
        "skipped_count": 0,
        "failed_count": 0,
        "verified_task_count": 2,
    }
    assert status["manifest_path"] == str(manifest_path.resolve())
    assert status["splits_root"] == str((tmp_path / "split bundles").resolve())
    assert status["results_root"] == str(results_root.resolve())
    assert status["device"] == "cuda:0"
    assert status["started_at"].endswith("Z")
    assert status["updated_at"].endswith("Z")
    assert status["finished_at"].endswith("Z")
    assert all(
        entry["command"] == command
        and entry["started_at"].endswith("Z")
        and entry["finished_at"].endswith("Z")
        for entry, command in zip(
            status["completed"],
            observed_commands,
            strict=True,
        )
    )


def test_invalid_existing_task_fails_closed_without_launch_or_overwrite(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    target = results_root / task["output_dir"]
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an invalid existing task must not launch")
        ),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix_manifest.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 4
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["completed"] == []
    assert status["skipped"] == []
    assert status["current"]["task_id"] == task["task_id"]
    assert status["current"]["command"] is None
    assert status["failed"] == [
        {
            "task_id": task["task_id"],
            "phase": "existing_result_validation",
            "command": None,
            "started_at": status["current"]["started_at"],
            "failed_at": status["finished_at"],
            "returncode": None,
            "error": status["failed"][0]["error"],
        }
    ]
    assert "allowlist" in status["failed"][0]["error"]
    assert not list(results_root.glob(".q.*.tmp"))


def test_symlinked_tasks_parent_is_rejected_before_external_result_validation(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    outside = tmp_path / "outside_tasks"
    results_root.mkdir()
    outside.mkdir()
    (results_root / "tasks").symlink_to(
        outside,
        target_is_directory=True,
    )
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_completed_task_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("external result path reached the validator")
        ),
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe task path must not launch")
        ),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 4
    assert list(outside.iterdir()) == []
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["failed"][0]["phase"] == "task_path_validation"
    assert "tasks directory" in status["failed"][0]["error"]


def test_restart_strictly_revalidates_existing_tasks_and_skips_them(
    tmp_path,
    monkeypatch,
):
    tasks = [_task(1), _task(2, "lstm")]
    manifest = _manifest(tasks)
    results_root = tmp_path / "results"
    for task in tasks:
        _write_valid_result(
            tmp_path=tmp_path,
            results_root=results_root,
            task=task,
        )

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("valid completed tasks must not relaunch")
        ),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix_manifest.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 0
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert status["completed"] == []
    assert [entry["task_id"] for entry in status["skipped"]] == [
        task["task_id"] for task in tasks
    ]
    assert status["failed"] == []
    assert status["current"] is None
    assert status["command"] is None
    assert status["verified_task_count"] == 2
    assert status["completed_count"] == 0
    assert status["skipped_count"] == 2


def test_nonzero_task_exit_stops_the_queue_and_records_the_failed_command(
    tmp_path,
    monkeypatch,
):
    tasks = [_task(1), _task(2, "lstm"), _task(3, "dcrnn")]
    manifest = _manifest(tasks)
    results_root = tmp_path / "results"
    launched: list[str] = []

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )

    def fake_run(command, **kwargs):
        command = list(command)
        task_id = command[command.index("--task-id") + 1]
        launched.append(task_id)
        if task_id == tasks[0]["task_id"]:
            _write_valid_result(
                tmp_path=tmp_path,
                results_root=results_root,
                task=tasks[0],
            )
            return subprocess.CompletedProcess(command, 0)
        if task_id == tasks[1]["task_id"]:
            return subprocess.CompletedProcess(command, 7)
        raise AssertionError("the queue continued after a failed task")

    monkeypatch.setattr(matrix_runner.subprocess, "run", fake_run)

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix_manifest.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 3
    assert launched == [tasks[0]["task_id"], tasks[1]["task_id"]]
    assert not (results_root / tasks[2]["output_dir"]).exists()
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert [entry["task_id"] for entry in status["completed"]] == [tasks[0]["task_id"]]
    assert status["skipped"] == []
    assert status["current"]["task_id"] == tasks[1]["task_id"]
    assert status["command"] == status["current"]["command"]
    assert status["failed"] == [
        {
            "task_id": tasks[1]["task_id"],
            "phase": "task_process",
            "command": status["current"]["command"],
            "started_at": status["current"]["started_at"],
            "failed_at": status["finished_at"],
            "returncode": 7,
            "error": "task runner exited with code 7",
        }
    ]
    assert not list(results_root.glob(".q.*.tmp"))


def test_zero_exit_without_valid_artifacts_fails_postflight_and_stops(
    tmp_path,
    monkeypatch,
):
    tasks = [_task(1), _task(2, "lstm")]
    manifest = _manifest(tasks)
    results_root = tmp_path / "results"
    launched: list[str] = []

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )

    def fake_run(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        launched.append(task_id)
        if len(launched) > 1:
            raise AssertionError("postflight failure did not stop the queue")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix_runner.subprocess, "run", fake_run)

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix_manifest.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 4
    assert launched == [tasks[0]["task_id"]]
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["completed"] == []
    assert status["skipped"] == []
    assert status["current"]["task_id"] == tasks[0]["task_id"]
    assert status["failed"][0]["phase"] == "completed_result_validation"
    assert status["failed"][0]["returncode"] == 0
    assert status["failed"][0]["command"] == status["command"]
    assert status["failed"][0]["failed_at"] == status["finished_at"]


def test_restart_after_failure_skips_verified_prefix_and_continues_in_order(
    tmp_path,
    monkeypatch,
):
    tasks = [_task(1), _task(2, "lstm"), _task(3, "dcrnn")]
    manifest = _manifest(tasks)
    results_root = tmp_path / "results"

    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )

    first_launch: list[str] = []

    def first_run(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        first_launch.append(task_id)
        if task_id == tasks[0]["task_id"]:
            _write_valid_result(
                tmp_path=tmp_path,
                results_root=results_root,
                task=tasks[0],
            )
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(matrix_runner.subprocess, "run", first_run)
    assert (
        matrix_runner.run_table4_matrix(
            manifest_path=tmp_path / "matrix_manifest.json",
            manifest_sha256="a" * 64,
            splits_root=tmp_path / "splits",
            results_root=results_root,
            device="cuda:0",
        )
        == 3
    )
    assert first_launch == [tasks[0]["task_id"], tasks[1]["task_id"]]

    resumed_launch: list[str] = []

    def resumed_run(command, **kwargs):
        task_id = command[command.index("--task-id") + 1]
        resumed_launch.append(task_id)
        task = next(task for task in tasks if task["task_id"] == task_id)
        _write_valid_result(
            tmp_path=tmp_path,
            results_root=results_root,
            task=task,
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix_runner.subprocess, "run", resumed_run)
    assert (
        matrix_runner.run_table4_matrix(
            manifest_path=tmp_path / "matrix_manifest.json",
            manifest_sha256="a" * 64,
            splits_root=tmp_path / "splits",
            results_root=results_root,
            device="cuda:0",
        )
        == 0
    )

    assert resumed_launch == [tasks[1]["task_id"], tasks[2]["task_id"]]
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert [entry["task_id"] for entry in status["skipped"]] == [tasks[0]["task_id"]]
    assert [entry["task_id"] for entry in status["completed"]] == [
        tasks[1]["task_id"],
        tasks[2]["task_id"],
    ]
    assert status["failed"] == []


def test_cli_requires_external_manifest_sha_and_preserves_task_allowlist(
    tmp_path,
    monkeypatch,
    capsys,
):
    with pytest.raises(SystemExit) as error:
        matrix_runner.main(
            [
                "--manifest",
                str(tmp_path / "matrix.json"),
                "--splits-root",
                str(tmp_path / "splits"),
                "--results-root",
                str(tmp_path / "results"),
                "--device",
                "cuda:0",
            ]
        )
    assert error.value.code == 2
    assert "--manifest-sha256" in capsys.readouterr().err

    observed: dict[str, object] = {}

    def fake_matrix(**kwargs):
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(matrix_runner, "run_table4_matrix", fake_matrix)
    returncode = matrix_runner.main(
        [
            "--manifest",
            str(tmp_path / "matrix with spaces.json"),
            "--manifest-sha256",
            "a" * 64,
            "--splits-root",
            str(tmp_path / "split bundles"),
            "--results-root",
            str(tmp_path / "results root"),
            "--device",
            "cuda:0",
            "--task-id",
            "003_alameda_dcrnn",
            "--task-id",
            "001_alameda_hl",
        ]
    )

    assert returncode == 0
    assert observed == {
        "manifest_path": tmp_path / "matrix with spaces.json",
        "manifest_sha256": "a" * 64,
        "splits_root": tmp_path / "split bundles",
        "results_root": tmp_path / "results root",
        "device": "cuda:0",
        "task_ids": ["003_alameda_dcrnn", "001_alameda_hl"],
    }


def test_subprocess_launch_error_is_recorded_before_the_runner_returns(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot launch Python")),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 3
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["current"]["task_id"] == task["task_id"]
    assert status["failed"][0]["phase"] == "task_launch"
    assert status["failed"][0]["returncode"] is None
    assert status["failed"][0]["error"] == "cannot launch Python"
    assert status["finished_at"].endswith("Z")


def test_keyboard_interrupt_is_atomically_recorded_for_safe_resume(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 130
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "interrupted"
    assert status["current"]["task_id"] == task["task_id"]
    assert status["failed"][0]["phase"] == "interrupted"
    assert status["failed"][0]["returncode"] == 130
    assert status["failed"][0]["failed_at"] == status["finished_at"]
    assert not list(results_root.glob(".q.*.tmp"))


def test_interrupt_during_existing_result_validation_is_also_recorded(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    (results_root / task["output_dir"]).mkdir(parents=True)
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_completed_task_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 130
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "interrupted"
    assert status["current"]["task_id"] == task["task_id"]
    assert status["failed"][0]["phase"] == ("existing_result_validation_interrupted")
    assert status["failed"][0]["returncode"] == 130


def test_system_exit_is_atomically_recorded_instead_of_leaving_running_state(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(23)),
    )

    returncode = matrix_runner.run_table4_matrix(
        manifest_path=tmp_path / "matrix.json",
        manifest_sha256="a" * 64,
        splits_root=tmp_path / "splits",
        results_root=results_root,
        device="cuda:0",
    )

    assert returncode == 23
    status = json.loads(
        (results_root / "queue_status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "interrupted"
    assert status["current"]["task_id"] == task["task_id"]
    assert status["failed"][0]["phase"] == "system_exit"
    assert status["failed"][0]["returncode"] == 23
    assert status["failed"][0]["error"] == (
        "matrix runner received SystemExit with code 23"
    )
    assert status["failed"][0]["failed_at"] == status["finished_at"]


def test_existing_queue_status_symlink_is_rejected_without_touching_target(
    tmp_path,
    monkeypatch,
):
    task = _task(1)
    manifest = _manifest([task])
    results_root = tmp_path / "results"
    results_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"owner":"outside"}\n', encoding="utf-8")
    (results_root / "queue_status.json").symlink_to(outside)
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe status path must fail before launch")
        ),
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        matrix_runner.run_table4_matrix(
            manifest_path=tmp_path / "matrix.json",
            manifest_sha256="a" * 64,
            splits_root=tmp_path / "splits",
            results_root=results_root,
            device="cuda:0",
        )

    assert outside.read_text(encoding="utf-8") == '{"owner":"outside"}\n'
    assert not list(results_root.glob(".q.*.tmp"))


def test_atomic_status_collision_does_not_delete_an_unowned_temporary_file(
    tmp_path,
    monkeypatch,
):
    manifest = _manifest([_task(1)])
    results_root = tmp_path / "results"
    results_root.mkdir()
    collision = results_root / ".q.0000000000000000.tmp"
    collision.write_text("belongs to another writer", encoding="utf-8")
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )
    monkeypatch.setattr(
        matrix_runner.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0" * 32),
    )

    with pytest.raises(FileExistsError):
        matrix_runner.run_table4_matrix(
            manifest_path=tmp_path / "matrix.json",
            manifest_sha256="a" * 64,
            splits_root=tmp_path / "splits",
            results_root=results_root,
            device="cuda:0",
        )

    assert collision.read_text(encoding="utf-8") == ("belongs to another writer")
    assert not (results_root / "queue_status.json").exists()


@pytest.mark.parametrize(
    ("task_ids", "message"),
    [
        (
            ["001_alameda_hl", "001_alameda_hl"],
            "must not contain duplicates",
        ),
        (["999_unknown"], "contains unknown tasks"),
    ],
)
def test_invalid_task_allowlist_is_rejected_before_results_are_touched(
    tmp_path,
    monkeypatch,
    task_ids,
    message,
):
    manifest = _manifest([_task(1)])
    results_root = tmp_path / "results"
    monkeypatch.setattr(
        matrix_runner,
        "load_table4_manifest",
        lambda path, expected_sha256: manifest,
    )

    with pytest.raises(ValueError, match=message):
        matrix_runner.run_table4_matrix(
            manifest_path=tmp_path / "matrix.json",
            manifest_sha256="a" * 64,
            splits_root=tmp_path / "splits",
            results_root=results_root,
            device="cuda:0",
            task_ids=task_ids,
        )

    assert not results_root.exists()
