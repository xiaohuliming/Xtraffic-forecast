from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RELAY = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "linux"
    / "protocol_v3_event_extension_relay_host.sh"
)


def test_relay_prioritizes_orange_igstgnn_shard() -> None:
    completed = subprocess.run(
        ["bash", str(RELAY), "--print-shard-priority"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines() == ["gpu01", "gpu02", "gpu03"]


def test_relay_rejects_an_incomplete_shard_priority() -> None:
    completed = subprocess.run(
        ["bash", str(RELAY), "--print-shard-priority"],
        cwd=REPO_ROOT,
        env={**os.environ, "SHARD_PRIORITY": "1,2"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "invalid SHARD_PRIORITY" in completed.stderr
