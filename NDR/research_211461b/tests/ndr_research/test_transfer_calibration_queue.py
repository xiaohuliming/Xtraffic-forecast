"""Predecessor gates prevent overlapping or silently restarting training."""


import pytest

from scripts.remote.windows.ndr_transfer_calibration import predecessor_admission
from tests.ndr_research.test_development_admission import fixture_matrix, write


def test_live_partial_predecessor_does_not_require_or_invent_postflight(tmp_path):
    write(
        tmp_path / "results/queue_status.json",
        {
            "status": "running",
            "plan_digest": "fixed",
            "completed": [],
            "pid": 11,
            "worker_pid": 12,
        },
    )
    ready, state = predecessor_admission(tmp_path, "fixed")
    assert not ready and state["worker_pid"] == 12
    assert not (tmp_path / "results/postflight.json").exists()


def test_terminal_incomplete_predecessor_never_auto_restarts(tmp_path):
    write(
        tmp_path / "results/queue_status.json",
        {"status": "failed", "plan_digest": "fixed", "completed": ["t1"]},
    )
    with pytest.raises(RuntimeError, match="terminal incomplete"):
        predecessor_admission(tmp_path, "fixed")


def test_complete_predecessor_requires_exact_verified_checkpoint_tree(tmp_path):
    plan = fixture_matrix(tmp_path)
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    ready, _ = predecessor_admission(tmp_path, plan["plan_digest"], plan_path)
    assert ready
    checkpoint = next((tmp_path / "tasks").glob("*/best.pt"))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="postflight"):
        predecessor_admission(tmp_path, plan["plan_digest"], plan_path)
