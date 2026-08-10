#!/usr/bin/env python3
"""Probe all 39 convergence tasks and emit a task-bound calibration JSON."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
from typing import Mapping, Sequence

from scripts.remote.linux.protocol_v3_event_convergence_calibration import (
    build_task_calibration,
)
from scripts.remote.linux.protocol_v3_event_convergence_multi_gpu import (
    default_microbatch_sizes,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    BASELINE_METHODS,
    CANONICAL_MANIFEST,
    EXTENSION_METHODS,
    load_convergence_manifest,
)
from scripts.remote.windows.protocol_v3_runner import Task


@dataclass(frozen=True)
class ProbeInputs:
    python_executable: str
    data_dir: Path
    graph_dir: Path
    support_root: Path
    node_order: Path
    output_dir: Path
    warmup_batches: int = 2
    timed_batches: int = 5
    timed_val_batches: int = 3
    loader_batches: int = 10


def build_probe_command(
    task: Task,
    inputs: ProbeInputs,
    microbatch_sizes: Mapping[str, int],
) -> list[str]:
    common = [inputs.python_executable, "-X", "utf8", "-m"]
    if task.model in BASELINE_METHODS:
        command = common + [
            "scripts.remote.linux.protocol_v3_event_gpu_probe",
            "--region",
            task.region,
            "--models",
            task.model,
            "--data-dir",
            str(inputs.data_dir),
            "--graph-dir",
            str(inputs.graph_dir),
            "--epochs",
            "100",
            "--recipe-profile",
            "convergence100",
            "--task-id",
            task.task_id,
            "--warmup-batches",
            str(inputs.warmup_batches),
            "--timed-batches",
            str(inputs.timed_batches),
            "--timed-val-batches",
            str(inputs.timed_val_batches),
            "--output",
            str(inputs.output_dir / f"{task.task_id}.json"),
        ]
        size = microbatch_sizes.get(task.task_id)
        if size is not None:
            command += ["--microbatch-size", str(size)]
        return command
    if task.model in EXTENSION_METHODS:
        command = common + [
            "scripts.remote.linux.protocol_v3_event_extension_gpu_probe",
            "--method",
            task.model,
            "--region",
            task.region,
            "--data-dir",
            str(inputs.data_dir),
            "--graph-dir",
            str(inputs.graph_dir),
            "--support-root",
            str(inputs.support_root),
            "--node-order",
            str(inputs.node_order),
            "--traffic-loading",
            "eager",
            "--epochs",
            "100",
            "--task-id",
            task.task_id,
            "--loader-batches",
            str(inputs.loader_batches),
        ]
        size = microbatch_sizes.get(task.task_id)
        if size is not None:
            command += ["--max-microbatch-size", str(size)]
        return command
    raise ValueError(f"unsupported convergence probe method: {task.model}")


def _last_json_object(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("probe output contains no JSON object")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=5)
    parser.add_argument("--timed-val-batches", type=int, default=3)
    parser.add_argument("--loader-batches", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_convergence_manifest(args.manifest)
    if not 1 <= args.gpu_count <= len(manifest.tasks):
        raise SystemExit("gpu-count must be between one and 39")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = ProbeInputs(
        python_executable=args.python_executable,
        data_dir=args.data_dir.resolve(),
        graph_dir=args.graph_dir.resolve(),
        support_root=args.support_root.resolve(),
        node_order=args.node_order.resolve(),
        output_dir=output_dir / "raw",
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
        timed_val_batches=args.timed_val_batches,
        loader_batches=args.loader_batches,
    )
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    microbatch_sizes = default_microbatch_sizes(manifest.tasks)
    queues = {
        gpu: list(manifest.tasks[gpu :: args.gpu_count])
        for gpu in range(args.gpu_count)
    }
    records: list[dict[str, object]] = []
    failures: list[str] = []
    lock = Lock()

    def worker(gpu: int) -> None:
        for task in queues[gpu]:
            command = build_probe_command(task, inputs, microbatch_sizes)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            )
            completed = subprocess.run(
                command,
                cwd=args.repo_root.resolve(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path = output_dir / "logs" / f"{task.task_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode:
                with lock:
                    failures.append(task.task_id)
                continue
            try:
                if task.model in BASELINE_METHODS:
                    payload = json.loads(
                        (inputs.output_dir / f"{task.task_id}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    record = dict(payload["measurements"][0])
                else:
                    record = _last_json_object(completed.stdout)
            except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError, RuntimeError):
                with lock:
                    failures.append(task.task_id)
                continue
            with lock:
                records.append(record)

    threads = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in queues]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print(f"PROBE_MATRIX_FAILED tasks={sorted(failures)}", file=sys.stderr)
        return 1
    calibration = build_task_calibration(
        manifest,
        records,
        host_id=args.host_id,
    )
    calibration_path = output_dir / "task_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    probes_path = output_dir / "probes.jsonl"
    probes_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in sorted(records, key=lambda row: str(row["task_id"]))
        ),
        encoding="utf-8",
    )
    print(
        f"PROBE_MATRIX_OK tasks={len(records)} calibration={calibration_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
