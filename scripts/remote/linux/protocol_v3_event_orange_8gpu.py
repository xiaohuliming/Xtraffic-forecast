#!/usr/bin/env python3
"""Run the Orange v3 event baseline shard across eight GPUs on one host."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from scripts.remote.windows.protocol_v3_runner import Manifest, Task
from scripts.remote.windows.protocol_v3_baseline_runner import (
    load_baseline_manifest,
)


GPU_METHODS = OrderedDict(
    {
        0: ("dcrnn",),
        1: ("dgcrn",),
        2: ("d2stgnn",),
        3: ("dstagnn",),
        4: ("sttn",),
        5: ("astgcn", "stgcn"),
        6: ("graphwavenet", "agcrn"),
        7: ("lstm", "hl"),
    }
)
OOM_RECOVERY_GPU_METHODS = OrderedDict(
    {
        1: ("dgcrn",),
        2: ("d2stgnn",),
        4: ("sttn",),
    }
)
GWN_RECOVERY_GPU_METHODS = OrderedDict(
    {
        6: ("graphwavenet", "agcrn"),
    }
)
OOM_RECOVERY_MICROBATCH_SIZES = {
    "dgcrn": 8,
    "d2stgnn": 8,
    "sttn": 16,
}

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "protocol_v3"
    / "table4_event_baselines_phase1_v1.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_orange_gpu_shards(
    manifest: Manifest,
    *,
    assignments: OrderedDict[int, tuple[str, ...]] = GPU_METHODS,
) -> dict[int, tuple[Task, ...]]:
    orange_by_method = {
        task.model: task for task in manifest.tasks if task.region == "Orange"
    }
    expected_methods = {
        method for methods in assignments.values() for method in methods
    }
    if not expected_methods <= set(orange_by_method):
        raise ValueError(
            "event manifest is missing an Orange task required by the GPU plan"
        )
    return {
        gpu: tuple(orange_by_method[method] for method in methods)
        for gpu, methods in assignments.items()
    }


def write_orange_shard_manifests(
    source_path: Path,
    manifest: Manifest,
    output_dir: Path,
    *,
    assignments: OrderedDict[int, tuple[str, ...]] = GPU_METHODS,
    microbatch_sizes: dict[str, int] | None = None,
) -> dict[int, Path]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_tasks = {
        task["task_id"]: task for task in payload["tasks"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for gpu, tasks in build_orange_gpu_shards(
        manifest,
        assignments=assignments,
    ).items():
        shard_payload = dict(payload)
        shard_payload["training"] = dict(payload["training"])
        if microbatch_sizes is not None:
            if len(tasks) != 1 or tasks[0].model not in microbatch_sizes:
                raise ValueError(
                    "microbatch recovery shards must bind exactly one configured model"
                )
            shard_payload["training"]["microbatch_size"] = (
                microbatch_sizes[tasks[0].model]
            )
        shard_payload["default_results_root"] = (
            "outputs/protocol_v3_noleak_dedup/remote_runs/"
            f"table4_event_orange_4090d8_gpu{gpu}"
        )
        shard_payload["tasks"] = [
            source_tasks[task.task_id] for task in tasks
        ]
        path = output_dir / f"orange_gpu{gpu}.json"
        path.write_text(
            json.dumps(
                shard_payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths[gpu] = path
    return paths


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "oom-recovery", "gwn-recovery"),
        default="full",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    source_path = args.manifest.resolve()
    control_root = args.control_root.resolve()
    results_root = args.results_root.resolve()
    cache_dir = args.cache_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    pid_path = control_root / "controller.pid"
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing_pid = -1
        if existing_pid != os.getpid() and _pid_is_running(existing_pid):
            raise SystemExit(
                f"another Orange controller is already running: {existing_pid}"
            )
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")

    manifest = load_baseline_manifest(source_path)
    if {task.sampling_protocol for task in manifest.tasks} != {"event"}:
        raise SystemExit("Orange eight-GPU controller requires the event manifest")

    import torch

    assignments = {
        "full": GPU_METHODS,
        "oom-recovery": OOM_RECOVERY_GPU_METHODS,
        "gwn-recovery": GWN_RECOVERY_GPU_METHODS,
    }[args.mode]
    microbatch_sizes = (
        OOM_RECOVERY_MICROBATCH_SIZES
        if args.mode == "oom-recovery"
        else None
    )
    device_count = torch.cuda.device_count()
    required_device_count = max(assignments) + 1
    if device_count < required_device_count:
        raise SystemExit(
            f"GPU plan requires at least {required_device_count} CUDA devices, "
            f"got {device_count}"
        )

    shards = build_orange_gpu_shards(
        manifest,
        assignments=assignments,
    )
    manifest_paths = write_orange_shard_manifests(
        source_path,
        manifest,
        control_root / "manifests",
        assignments=assignments,
        microbatch_sizes=microbatch_sizes,
    )
    plan = {
        "matrix_id": {
            "full": "protocol_v3_event_orange_4090d8_v1",
            "oom-recovery": (
                "protocol_v3_event_orange_4090d8_oom_recovery_v1"
            ),
            "gwn-recovery": (
                "protocol_v3_event_orange_4090d8_gwn_recovery_v1"
            ),
        }[args.mode],
        "execution_layout": {
            "full": "single_host_eight_gpu_method_sharded",
            "oom-recovery": "same_host_failed_task_microbatch_recovery",
            "gwn-recovery": "same_host_corrected_evaluation_input_recovery",
        }[args.mode],
        "created_at": _utc_now(),
        "source_manifest": str(source_path),
        "source_manifest_sha256": _sha256_file(source_path),
        "code_sha": args.code_sha.lower(),
        "cache_bundle_id": args.cache_bundle_id,
        "cache_bundle_digest": args.cache_bundle_digest.lower(),
        "microbatch_sizes": microbatch_sizes,
        "assignments": {
            str(gpu): [task.task_id for task in tasks]
            for gpu, tasks in shards.items()
        },
    }
    _write_json_atomic(control_root / "plan.json", plan)

    children: dict[int, subprocess.Popen[bytes]] = {}
    log_streams: dict[int, object] = {}
    try:
        for gpu in assignments:
            log_path = control_root / "logs" / f"gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("ab", buffering=0)
            log_streams[gpu] = log_stream
            command = [
                args.python_executable,
                "-X",
                "utf8",
                "-m",
                "scripts.remote.windows.protocol_v3_baseline_runner",
                "--allow-shard",
                "--manifest",
                str(manifest_paths[gpu]),
                "--repo-root",
                str(repo_root),
                "--results-root",
                str(results_root / f"gpu{gpu}"),
                "--cache-dir",
                str(cache_dir),
                "--graph-dir",
                str(graph_dir),
                "--cache-bundle-id",
                args.cache_bundle_id,
                "--cache-bundle-digest",
                args.cache_bundle_digest.lower(),
                "--code-sha",
                args.code_sha.lower(),
                "--python",
                args.python_executable,
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTHONUNBUFFERED"] = "1"
            children[gpu] = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        while True:
            return_codes = {
                gpu: child.poll() for gpu, child in children.items()
            }
            state = (
                "completed"
                if all(code == 0 for code in return_codes.values())
                else "failed"
                if all(code is not None for code in return_codes.values())
                else "running"
            )
            _write_json_atomic(
                control_root / "controller_status.json",
                {
                    "matrix_id": plan["matrix_id"],
                    "state": state,
                    "updated_at": _utc_now(),
                    "controller_pid": os.getpid(),
                    "children": {
                        str(gpu): {
                            "pid": children[gpu].pid,
                            "return_code": return_codes[gpu],
                            "tasks": plan["assignments"][str(gpu)],
                        }
                        for gpu in children
                    },
                },
            )
            if all(code is not None for code in return_codes.values()):
                return 0 if all(code == 0 for code in return_codes.values()) else 1
            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        for stream in log_streams.values():
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
