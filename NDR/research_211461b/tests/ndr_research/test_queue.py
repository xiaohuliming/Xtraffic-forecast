"""Execution-boundary tests for the isolated research controller."""

import hashlib
import json
import subprocess
import sys
import time

import pytest

from scripts.remote.windows.ndr_research_queue import (
    digest,
    execute,
    gpu_ready,
    validate_plan,
)


def test_deadline_stops_owned_child_without_stopping_another_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    state = {}
    try:
        with pytest.raises(TimeoutError):
            execute(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                log=tmp_path / "worker.log",
                repo=tmp_path,
                state=state,
                status_path=tmp_path / "status.json",
                deadline=time.time() - 1,
            )
        assert unrelated.poll() is None
        assert json.loads((tmp_path / "status.json").read_text())["worker_pid"] is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_snapshot_mismatch_blocks_worker_before_execution(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "model.py"
    source.write_text("original\n")
    root = tmp_path / "run"
    root.mkdir()
    plan = {
        "protocol_version": "protocol_v3_noleak_dedup",
        "cache_schema_version": 3,
        "test_evaluated": False,
        "automatic_transfer": False,
        "automatic_shutdown": False,
        "python": sys.executable,
        "run_root": str(root),
        "source_files": {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()},
        "tasks": [
            {
                "task_id": "t1",
                "output_dir": str(root / "t1"),
                "train_command": [sys.executable, "train.py"],
                "verify_command": [sys.executable, "verify.py"],
            }
        ],
    }
    plan["plan_digest"] = digest(plan)
    assert validate_plan(plan, repo) is plan
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="snapshot changed"):
        validate_plan(plan, repo)


def test_windows_desktop_usage_is_not_mistaken_for_cuda_training(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: "NVIDIA GeForce RTX 5080, 14500, 6\n",
    )
    assert gpu_ready(6000)[0]
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: "NVIDIA GeForce RTX 5080, 1000, 99\n",
    )
    assert not gpu_ready(6000)[0]
