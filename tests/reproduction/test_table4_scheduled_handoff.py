from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from scripts.reproduction.igstgnn_table4.scheduled_handoff import run_handoff


def _kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": tmp_path / "repo",
        "manifest_path": tmp_path / "matrix.json",
        "manifest_sha256": "a" * 64,
        "splits_root": tmp_path / "splits",
        "results_root": tmp_path / "results",
        "device": "cuda:0",
        "pilot_task_ids": ("003_alameda_dcrnn", "013_alameda_igstgnn"),
        "status_path": tmp_path / "handoff_status.json",
    }


def test_handoff_runs_pilot_gate_before_full_matrix(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path) -> CompletedProcess[str]:
        assert cwd == tmp_path / "repo"
        commands.append(command)
        return CompletedProcess(command, 0)

    assert run_handoff(**_kwargs(tmp_path), runner=runner) == 0

    assert len(commands) == 2
    assert commands[0][-4:] == [
        "--task-id",
        "003_alameda_dcrnn",
        "--task-id",
        "013_alameda_igstgnn",
    ]
    assert "--task-id" not in commands[1]
    status = json.loads((tmp_path / "handoff_status.json").read_text())
    assert status["state"] == "completed"
    assert status["completed_phases"] == ["pilot_gate", "full_matrix"]
    assert status["failed_phase"] is None


def test_handoff_fails_closed_before_full_matrix(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path) -> CompletedProcess[str]:
        commands.append(command)
        return CompletedProcess(command, 7)

    assert run_handoff(**_kwargs(tmp_path), runner=runner) == 7

    assert len(commands) == 1
    status = json.loads((tmp_path / "handoff_status.json").read_text())
    assert status["state"] == "failed"
    assert status["completed_phases"] == []
    assert status["failed_phase"] == "pilot_gate"
    assert status["returncode"] == 7
