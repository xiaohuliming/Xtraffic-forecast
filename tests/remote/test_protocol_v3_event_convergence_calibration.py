from __future__ import annotations

import pytest

from scripts.remote.linux.protocol_v3_event_convergence_calibration import (
    build_task_calibration,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    load_convergence_manifest,
)


def test_convergence_calibration_requires_one_100_epoch_probe_per_task() -> None:
    manifest = load_convergence_manifest(CANONICAL_MANIFEST)
    probes = [
        {
            "status": "PROBE_OK",
            "task_id": task.task_id,
            "method": task.model,
            "region": task.region,
            "epochs": 100,
            "estimated_task_seconds": float(index + 1),
            "peak_allocated_bytes": 1_000 + index,
        }
        for index, task in enumerate(manifest.tasks)
    ]

    calibration = build_task_calibration(manifest, probes, host_id="probe-host")

    assert calibration["schema_version"] == 1
    assert calibration["matrix_id"] == manifest.matrix_id
    assert calibration["host_id"] == "probe-host"
    assert calibration["task_seconds"] == {
        task.task_id: float(index + 1)
        for index, task in enumerate(manifest.tasks)
    }
    assert len(calibration["probes"]) == 39

    with pytest.raises(ValueError, match="exactly once"):
        build_task_calibration(manifest, probes[:-1], host_id="probe-host")
