#!/usr/bin/env python3
"""Run the nine v3 event extension tasks across one host's GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

from scripts.remote.linux.protocol_v3_event_multi_gpu import (
    aggregate_queue_status,
    shard_name,
    write_shard_manifests,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    CANONICAL_MANIFEST,
    SUPPORT_MANIFEST,
    load_extension_manifest,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import Manifest, Task


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_task_calibration(path: Path | str, tasks: Sequence[Task]) -> dict[str, float]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = value.get("task_seconds") if isinstance(value, dict) else None
    expected = {task.task_id for task in tasks}
    if value.get("schema_version") != 1 or not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("extension calibration must bind every canonical task exactly once")
    result = {}
    for task_id, seconds in raw.items():
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError(f"invalid extension calibration for {task_id}")
        result[task_id] = float(seconds)
    return result


def assign_tasks_lpt(
    tasks: Sequence[Task],
    *,
    gpu_count: int,
    task_seconds: Mapping[str, float],
) -> dict[int, tuple[Task, ...]]:
    if not 1 <= gpu_count <= len(tasks):
        raise ValueError("gpu_count must be between one and the number of tasks")
    if set(task_seconds) != {task.task_id for task in tasks}:
        raise ValueError("task calibration does not match the selected tasks")
    buckets: dict[int, list[Task]] = {gpu: [] for gpu in range(gpu_count)}
    loads = {gpu: 0.0 for gpu in range(gpu_count)}
    for task in sorted(tasks, key=lambda item: (-task_seconds[item.task_id], item.task_id)):
        gpu = min(loads, key=lambda item: (loads[item], item))
        buckets[gpu].append(task)
        loads[gpu] += float(task_seconds[task.task_id])
    return {
        gpu: tuple(sorted(members, key=lambda task: task.task_id))
        for gpu, members in buckets.items()
    }


def build_plan(
    *,
    manifest: Manifest,
    shards: Mapping[int, Sequence[Task]],
    matrix_id: str,
    host_id: str,
    code_sha: str,
    cache_bundle_id: str,
    cache_bundle_digest: str,
    support_manifest_sha256: str,
    task_seconds: Mapping[str, float],
    gpu_count: int,
) -> dict[str, object]:
    plan: dict[str, object] = {
        "matrix_id": matrix_id,
        "host_id": host_id,
        "execution_layout": "single_host_multi_gpu_interleaved",
        "source_manifest_sha256": manifest.digest,
        "code_sha": code_sha,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "support_manifest_sha256": support_manifest_sha256,
        "gpu_count": gpu_count,
        "task_count": sum(len(tasks) for tasks in shards.values()),
        "assignments": {
            str(gpu): [task.task_id for task in tasks]
            for gpu, tasks in sorted(shards.items())
        },
        "task_seconds": {key: float(task_seconds[key]) for key in sorted(task_seconds)},
        "task_microbatch_sizes": {
            task.task_id: 8
            for tasks in shards.values()
            for task in tasks
            if task.region == "Orange" and task.model in {"igstgnn", "ndr_net"}
        },
        "shard_seconds": {
            str(gpu): float(sum(task_seconds[task.task_id] for task in tasks))
            for gpu, tasks in sorted(shards.items())
        },
    }
    plan["plan_digest"] = _sha256_json(plan)
    return plan


def build_extension_nas_plan(
    *,
    matrix_id: str,
    host_id: str,
    manifest: Manifest,
    canonical_manifest: Path,
    canonical_manifest_sha256: str,
    support_manifest: Path,
    support_manifest_sha256: str,
    support_root: Path,
    node_order: Path,
    shards: Mapping[int, Sequence[Task]],
    results_root: Path,
    control_root: Path,
    cache_dir: Path,
    graph_dir: Path,
    cache_bundle_id: str,
    cache_bundle_digest: str,
    code_sha: str,
    python_executable: str,
) -> dict[str, object]:
    canonical_order = [task.task_id for task in manifest.tasks]
    sources: list[dict[str, object]] = []
    assigned: list[str] = []
    for gpu in sorted(shards):
        members = {task.task_id for task in shards[gpu]}
        task_ids = [task_id for task_id in canonical_order if task_id in members]
        assigned.extend(task_ids)
        sources.append(
            {
                "source_id": shard_name(gpu),
                "results_root": str(results_root / shard_name(gpu)),
                "code_sha": code_sha,
                "task_ids": task_ids,
            }
        )
    if [item["source_id"] for item in sources] != sorted(
        str(item["source_id"]) for item in sources
    ):
        raise ValueError("extension NAS sources must be sorted")
    if len(assigned) != len(manifest.tasks) or set(assigned) != set(canonical_order):
        raise ValueError("extension NAS task union must equal the canonical matrix")
    return {
        "plan_version": 1,
        "runner_kind": "event_extension",
        "matrix_id": matrix_id,
        "host_id": host_id,
        "canonical_manifest": str(canonical_manifest),
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": support_manifest_sha256,
        "support_root": str(support_root),
        "node_order": str(node_order),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "cache_dir": str(cache_dir),
        "graph_dir": str(graph_dir),
        "python_executable": python_executable,
        "expected_task_count": len(assigned),
        "source_layout": "interleaved",
        "extension_options": {
            "igstgnn_data_parallel": False,
            "max_microbatch_size": 0,
            "task_microbatch_sizes": {
                task.task_id: 8
                for tasks in shards.values()
                for task in tasks
                if task.region == "Orange"
                and task.model in {"igstgnn", "ndr_net"}
            },
        },
        "sources": sources,
        "completion_gate": {
            "results_root": str(results_root),
            "runner_exit_code": str(control_root / "runner.exitcode"),
            "expected_tasks": len(assigned),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--support-manifest", type=Path, default=SUPPORT_MANIFEST)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--expect-plan-digest")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    control_root = args.control_root.resolve()
    results_root = args.results_root.resolve()
    control_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    pid_path = control_root / "controller.pid"
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing_pid = -1
        if existing_pid != os.getpid() and _pid_is_running(existing_pid):
            raise SystemExit(f"another extension controller is running: {existing_pid}")
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    manifest = load_extension_manifest(args.manifest)
    support = verify_extension_support(
        support_manifest_path=args.support_manifest,
        support_root=args.support_root,
        node_order=args.node_order,
    )
    costs = load_task_calibration(args.calibration, manifest.tasks)
    shards = assign_tasks_lpt(
        manifest.tasks, gpu_count=args.gpu_count, task_seconds=costs
    )
    shard_manifests = write_shard_manifests(
        args.manifest.resolve(),
        shards,
        control_root / "manifests",
        results_root_prefix=(
            "outputs/protocol_v3_noleak_dedup/remote_runs/" + args.matrix_id
        ),
    )
    plan = build_plan(
        manifest=manifest,
        shards=shards,
        matrix_id=args.matrix_id,
        host_id=args.host_id,
        code_sha=args.code_sha.lower(),
        cache_bundle_id=args.cache_bundle_id,
        cache_bundle_digest=args.cache_bundle_digest.lower(),
        support_manifest_sha256=support.manifest_sha256,
        task_seconds=costs,
        gpu_count=args.gpu_count,
    )
    plan.update(
        {
            "created_at": _utc_now(),
            "source_manifest": str(args.manifest.resolve()),
            "support_manifest": str(args.support_manifest.resolve()),
            "calibration": str(args.calibration.resolve()),
            "calibration_sha256": _sha256_file(args.calibration.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "graph_dir": str(args.graph_dir.resolve()),
            "support_root": str(support.support_root),
            "node_order": str(support.node_order),
        }
    )
    plan["plan_digest"] = _sha256_json(
        {
            key: value
            for key, value in plan.items()
            if key not in {"created_at", "plan_digest"}
        }
    )
    _write_json_atomic(control_root / "plan.json", plan)
    nas_plan = build_extension_nas_plan(
        matrix_id=args.matrix_id,
        host_id=args.host_id,
        manifest=manifest,
        canonical_manifest=args.manifest.resolve(),
        canonical_manifest_sha256=_sha256_file(args.manifest.resolve()),
        support_manifest=args.support_manifest.resolve(),
        support_manifest_sha256=support.manifest_sha256,
        support_root=support.support_root,
        node_order=support.node_order,
        shards=shards,
        results_root=results_root,
        control_root=control_root,
        cache_dir=args.cache_dir.resolve(),
        graph_dir=args.graph_dir.resolve(),
        cache_bundle_id=args.cache_bundle_id,
        cache_bundle_digest=args.cache_bundle_digest.lower(),
        code_sha=args.code_sha.lower(),
        python_executable=args.python_executable,
    )
    _write_json_atomic(control_root / "nas_backup_plan.json", nas_plan)
    if args.expect_plan_digest and args.expect_plan_digest != plan["plan_digest"]:
        raise SystemExit("reviewed extension plan digest does not match")
    if args.plan_only:
        print(
            f"PLAN_ONLY_OK tasks=9 gpus={args.gpu_count} "
            f"makespan_seconds={max(plan['shard_seconds'].values()):.3f} "
            f"plan_digest={plan['plan_digest']}"
        )
        return 0

    if torch_device_count() < args.gpu_count:
        raise SystemExit("extension plan requests more GPUs than are visible")
    children: dict[int, subprocess.Popen[bytes]] = {}
    logs: dict[int, object] = {}
    try:
        for gpu, tasks in sorted(shards.items()):
            log_path = control_root / "logs" / f"{shard_name(gpu)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab", buffering=0)
            logs[gpu] = handle
            command = [
                args.python_executable, "-X", "utf8", "-m",
                "scripts.remote.windows.protocol_v3_event_extension_runner",
                "--allow-shard", "--manifest", str(shard_manifests[gpu]),
                "--support-manifest", str(args.support_manifest.resolve()),
                "--repo-root", str(repo_root),
                "--results-root", str(results_root / shard_name(gpu)),
                "--cache-dir", str(args.cache_dir.resolve()),
                "--graph-dir", str(args.graph_dir.resolve()),
                "--support-root", str(support.support_root),
                "--node-order", str(support.node_order),
                "--cache-bundle-id", args.cache_bundle_id,
                "--cache-bundle-digest", args.cache_bundle_digest.lower(),
                "--code-sha", args.code_sha.lower(),
                "--python", args.python_executable,
            ]
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": args.alloc_conf,
                }
            )
            children[gpu] = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while True:
            return_codes = {gpu: process.poll() for gpu, process in children.items()}
            status = aggregate_queue_status(
                manifest=manifest,
                shards=shards,
                results_root=results_root,
                return_codes=return_codes,
                cache_bundle_id=args.cache_bundle_id,
                cache_bundle_digest=args.cache_bundle_digest.lower(),
                code_sha=args.code_sha.lower(),
                total_tasks=9,
            )
            _write_json_atomic(results_root / "queue_status.json", status)
            _write_json_atomic(
                control_root / "controller_status.json",
                {
                    "matrix_id": args.matrix_id,
                    "host_id": args.host_id,
                    "state": status["state"],
                    "updated_at": status["updated_at"],
                    "controller_pid": os.getpid(),
                    "children": {
                        str(gpu): {
                            "pid": children[gpu].pid,
                            "return_code": return_codes[gpu],
                            "tasks": [task.task_id for task in shards[gpu]],
                        }
                        for gpu in children
                    },
                },
            )
            if all(code is not None for code in return_codes.values()):
                exit_code = 0 if status["state"] == "completed" else 1
                (control_root / "runner.exitcode").write_text(
                    f"{exit_code}\n", encoding="ascii"
                )
                return exit_code
            time.sleep(max(1.0, args.poll_seconds))
    finally:
        for handle in logs.values():
            handle.close()


def torch_device_count() -> int:
    import torch

    return torch.cuda.device_count()


if __name__ == "__main__":
    raise SystemExit(main())
