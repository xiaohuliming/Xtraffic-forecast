#!/usr/bin/env python3
# ruff: noqa: E402
"""Launch isolated FDN benchmark v1 shards across visible CUDA devices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.fdn_benchmark_v1 import sha256_file, strict_json_write
from scripts.remote.windows.fdn_benchmark_v1_runner import load_manifest


DEFAULT_MANIFEST = ROOT / "config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json"
RUNNER = ROOT / "scripts/remote/windows/fdn_benchmark_v1_runner.py"


def build_shard_command(args: argparse.Namespace, shard_index: int) -> list[str]:
    return [
        args.python_executable,
        "-X",
        "utf8",
        str(RUNNER),
        "--manifest",
        str(args.manifest),
        "--repo-root",
        str(args.repo_root),
        "--data-dir",
        str(args.data_dir),
        "--results-root",
        str(args.results_root),
        "--code-sha",
        args.code_sha,
        "--python",
        args.python_executable,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(args.gpu_count),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--gpu-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.repo_root = args.repo_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.results_root = args.results_root.resolve()
    args.control_root = args.control_root.resolve()
    args.code_sha = args.code_sha.lower()
    if len(args.code_sha) != 40 or any(
        character not in "0123456789abcdef" for character in args.code_sha
    ):
        raise ValueError("code SHA must be a full lowercase Git SHA")
    if args.gpu_count <= 0 or args.gpu_count > 8:
        raise ValueError("gpu count must be between 1 and 8")
    manifest = load_manifest(args.manifest)
    if args.gpu_count > len(manifest.tasks):
        raise ValueError("gpu count exceeds the number of tasks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    visible_devices = torch.cuda.device_count()
    if visible_devices < args.gpu_count:
        raise RuntimeError(
            f"requested {args.gpu_count} GPUs but only {visible_devices} are visible"
        )
    if args.control_root.exists():
        raise FileExistsError(f"control root must be fresh: {args.control_root}")
    args.control_root.mkdir(parents=True, exist_ok=False)
    args.results_root.mkdir(parents=True, exist_ok=True)
    aggregate_queue_path = args.results_root / "queue_status.json"
    if aggregate_queue_path.exists():
        raise FileExistsError(f"aggregate queue already exists: {aggregate_queue_path}")
    started_at = time.time()
    strict_json_write(
        args.control_root / "launch.json",
        {
            "state": "running",
            "matrix_id": manifest.matrix_id,
            "manifest_sha256": sha256_file(args.manifest),
            "code_sha": args.code_sha,
            "gpu_count": args.gpu_count,
            "task_count": len(manifest.tasks),
            "started_at_unix": started_at,
        },
    )
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for shard_index in range(args.gpu_count):
        shard_control = args.control_root / f"shard_{shard_index:02d}"
        shard_control.mkdir()
        log_handle = (shard_control / "runner.log").open("xb")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(shard_index)
        process = subprocess.Popen(
            build_shard_command(args, shard_index),
            cwd=args.repo_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard_index, process, log_handle))
    exitcodes: dict[str, int] = {}
    for shard_index, process, log_handle in processes:
        returncode = int(process.wait())
        log_handle.close()
        exitcodes[f"shard_{shard_index:02d}"] = returncode
        strict_json_write(
            args.control_root / f"shard_{shard_index:02d}" / "exit.json",
            {"shard_index": shard_index, "returncode": returncode},
        )
    success = all(returncode == 0 for returncode in exitcodes.values())
    shard_statuses: dict[str, object] = {}
    completed_tasks = 0
    if success:
        for shard_index in range(args.gpu_count):
            shard_name = f"shard_{shard_index:02d}"
            status_path = args.results_root / "state/shards" / f"{shard_name}.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                status.get("state") != "completed"
                or status.get("code_sha") != args.code_sha
                or status.get("shard_index") != shard_index
                or status.get("shard_count") != args.gpu_count
                or status.get("failed_count") != 0
            ):
                success = False
                break
            completed_tasks += int(status.get("completed_tasks", -1))
            shard_statuses[shard_name] = {
                "queue_status_sha256": sha256_file(status_path),
                "completed_tasks": status["completed_tasks"],
                "skipped_valid_tasks": status["skipped_valid_tasks"],
            }
        if completed_tasks != len(manifest.tasks):
            success = False
    if success:
        strict_json_write(
            aggregate_queue_path,
            {
                "state": "completed",
                "matrix_id": manifest.matrix_id,
                "total_tasks": len(manifest.tasks),
                "completed_tasks": completed_tasks,
                "skipped_valid_tasks": sum(
                    int(status["skipped_valid_tasks"])
                    for status in shard_statuses.values()
                ),
                "failed_count": 0,
                "current_task": None,
                "started_at_unix": started_at,
                "completed_at_unix": time.time(),
                "manifest_sha256": sha256_file(args.manifest),
                "code_sha": args.code_sha,
                "gpu_count": args.gpu_count,
                "shards": shard_statuses,
            },
        )
    strict_json_write(
        args.control_root / "final.json",
        {
            "state": "completed" if success else "failed",
            "matrix_id": manifest.matrix_id,
            "code_sha": args.code_sha,
            "gpu_count": args.gpu_count,
            "task_count": len(manifest.tasks),
            "exitcodes": exitcodes,
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
        },
    )
    return 0 if success else 3


if __name__ == "__main__":
    raise SystemExit(main())
