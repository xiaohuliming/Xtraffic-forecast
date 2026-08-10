#!/usr/bin/env python3
"""Build the exact 39-task convergence calibration from CUDA probe records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    ConvergenceManifest,
    load_convergence_manifest,
)


def build_task_calibration(
    manifest: ConvergenceManifest,
    probes: Sequence[Mapping[str, object]],
    *,
    host_id: str,
) -> dict[str, object]:
    expected = {task.task_id: task for task in manifest.tasks}
    actual_ids = [probe.get("task_id") for probe in probes]
    if (
        len(probes) != len(expected)
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected)
    ):
        raise ValueError("every canonical convergence task must be probed exactly once")
    normalized: list[dict[str, object]] = []
    task_seconds: dict[str, float] = {}
    for probe in probes:
        task_id = str(probe["task_id"])
        task = expected[task_id]
        seconds = probe.get("estimated_task_seconds")
        peak = probe.get("peak_allocated_bytes")
        if (
            probe.get("status") != "PROBE_OK"
            or probe.get("method") != task.model
            or probe.get("region") != task.region
            or probe.get("epochs") != 100
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak <= 0
        ):
            raise ValueError(f"invalid convergence probe record: {task_id}")
        task_seconds[task_id] = float(seconds)
        normalized.append(dict(probe))
    return {
        "schema_version": 1,
        "matrix_id": manifest.matrix_id,
        "host_id": host_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "loader_plus_complete_optimization_step_extrapolated_to_100_epochs",
        "task_seconds": {
            task_id: task_seconds[task_id]
            for task_id in sorted(task_seconds)
        },
        "probes": sorted(normalized, key=lambda row: str(row["task_id"])),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--probes-jsonl", type=Path, required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_convergence_manifest(args.manifest)
    probes = [
        json.loads(line)
        for line in args.probes_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = build_task_calibration(manifest, probes, host_id=args.host_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"CALIBRATION_OK tasks={len(payload['task_seconds'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
