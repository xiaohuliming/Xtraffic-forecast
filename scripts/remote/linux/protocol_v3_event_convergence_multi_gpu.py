#!/usr/bin/env python3
"""Plan and run the 39-task v3 seed-42 convergence audit on one GPU host."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

from scripts.remote.linux.protocol_v3_event_extension_multi_gpu import (
    _pid_is_running,
    _sha256_file,
    _sha256_json,
    _utc_now,
    _write_json_atomic,
    load_task_calibration,
    torch_device_count,
    verify_loading_audit,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    CANONICAL_MANIFEST,
    MATRIX_ID,
    REPO_ROOT,
    load_convergence_manifest,
)
from scripts.remote.linux.protocol_v3_event_multi_gpu import (
    PlanError,
    _allocate_gpus,
    aggregate_queue_status,
    build_nas_plan,
    shard_name,
    write_shard_manifests,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    SUPPORT_MANIFEST,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import Manifest, Task


def default_microbatch_sizes(tasks: Sequence[Task]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for task in tasks:
        if task.region != "Orange":
            continue
        if task.model in {"sttn", "d2stgnn"}:
            sizes[task.task_id] = 16
        elif task.model in {"igstgnn", "ndr_net"}:
            sizes[task.task_id] = 8
    return sizes


def assign_tasks_grouped_lpt(
    tasks: Sequence[Task],
    *,
    gpu_count: int,
    task_seconds: Mapping[str, float],
    microbatch_sizes: Mapping[str, int],
) -> dict[int, tuple[Task, ...]]:
    if not 1 <= gpu_count <= len(tasks):
        raise PlanError("gpu_count must be between one and the number of tasks")
    expected = {task.task_id for task in tasks}
    if set(task_seconds) != expected:
        raise PlanError("task calibration must bind every selected task exactly once")
    if not set(microbatch_sizes) <= expected:
        raise PlanError("microbatch override names an unselected task")

    groups: dict[int | None, list[Task]] = {}
    for task in tasks:
        groups.setdefault(microbatch_sizes.get(task.task_id), []).append(task)
    group_loads = {
        key: sum(float(task_seconds[task.task_id]) for task in members)
        for key, members in groups.items()
    }
    allocation = _allocate_gpus(
        group_loads,
        {key: len(members) for key, members in groups.items()},
        gpu_count,
    )
    buckets: dict[int, list[Task]] = {}
    cursor = 0
    for key in sorted(groups, key=lambda value: (value is not None, value or 0)):
        slots = list(range(cursor, cursor + allocation[key]))
        cursor += allocation[key]
        loads = {slot: 0.0 for slot in slots}
        for task in sorted(
            groups[key],
            key=lambda item: (-float(task_seconds[item.task_id]), item.task_id),
        ):
            target = min(slots, key=lambda slot: (loads[slot], slot))
            buckets.setdefault(target, []).append(task)
            loads[target] += float(task_seconds[task.task_id])
    return {
        gpu: tuple(sorted(members, key=lambda task: task.task_id))
        for gpu, members in sorted(buckets.items())
    }


def select_global_assignment_shards(
    tasks: Sequence[Task],
    *,
    plan: Mapping[str, object],
    global_gpu_ids: Sequence[int],
    expected: Mapping[str, object],
) -> tuple[dict[int, tuple[Task, ...]], str, int]:
    """Select host-local shards from one reviewed global assignment plan."""
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"global assignment plan identity mismatch: {field}")
    observed_digest = plan.get("plan_digest")
    if not isinstance(observed_digest, str) or len(observed_digest) != 64:
        raise ValueError("global assignment plan digest is invalid")
    computed_digest = _sha256_json(
        {
            key: value
            for key, value in plan.items()
            if key not in {"created_at", "plan_digest"}
        }
    )
    if computed_digest != observed_digest:
        raise ValueError("global assignment plan digest mismatch")

    gpu_count = plan.get("gpu_count")
    task_count = plan.get("task_count")
    assignments = plan.get("assignments")
    microbatch_sizes = plan.get("microbatch_sizes")
    if (
        isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or gpu_count <= 0
        or task_count != len(tasks)
        or not isinstance(assignments, dict)
        or set(assignments) != {str(index) for index in range(gpu_count)}
        or not isinstance(microbatch_sizes, dict)
    ):
        raise ValueError("global assignment plan topology is invalid")
    selected = tuple(global_gpu_ids)
    if (
        not selected
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selected)
        or len(set(selected)) != len(selected)
        or any(value < 0 or value >= gpu_count for value in selected)
    ):
        raise ValueError("global GPU IDs must be a unique nonempty plan subset")

    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("canonical convergence tasks are not unique")
    all_assigned: list[str] = []
    for gpu in range(gpu_count):
        task_ids = assignments[str(gpu)]
        if not isinstance(task_ids, list) or not task_ids:
            raise ValueError("global assignment shard must be a nonempty task list")
        if any(not isinstance(task_id, str) or task_id not in by_id for task_id in task_ids):
            raise ValueError("global assignment contains a non-canonical task")
        groups = {microbatch_sizes.get(task_id) for task_id in task_ids}
        if len(groups) != 1:
            raise ValueError("global assignment mixes microbatch groups on one GPU")
        all_assigned.extend(task_ids)
    if len(all_assigned) != len(set(all_assigned)) or set(all_assigned) != set(by_id):
        raise ValueError("global assignment must cover canonical tasks exactly once")

    local = {
        local_gpu: tuple(by_id[task_id] for task_id in assignments[str(global_gpu)])
        for local_gpu, global_gpu in enumerate(selected)
    }
    return local, observed_digest, gpu_count


def build_convergence_nas_plan(
    *,
    matrix_id: str,
    host_id: str,
    manifest: Manifest,
    canonical_manifest: Path,
    canonical_manifest_sha256: str,
    support_manifest: Path,
    support_manifest_sha256: str,
    loading_audit: Path,
    loading_audit_sha256: str,
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
    microbatch_sizes: Mapping[str, int],
) -> dict[str, object]:
    plan = build_nas_plan(
        matrix_id=matrix_id,
        host_id=host_id,
        manifest=manifest,
        canonical_manifest=canonical_manifest,
        canonical_manifest_sha256=canonical_manifest_sha256,
        shards=shards,
        results_root=results_root,
        control_root=control_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=cache_bundle_id,
        cache_bundle_digest=cache_bundle_digest,
        code_sha=code_sha,
        python_executable=python_executable,
        microbatch_sizes=microbatch_sizes,
    )
    plan.update(
        {
            "runner_kind": "event_convergence",
            "support_manifest": str(support_manifest),
            "support_manifest_sha256": support_manifest_sha256,
            "loading_audit": str(loading_audit),
            "loading_audit_sha256": loading_audit_sha256,
            "support_root": str(support_root),
            "node_order": str(node_order),
            "extension_options": {
                "igstgnn_data_parallel": False,
                "max_microbatch_size": 0,
            },
            "required_task_artifacts": ["training_diagnostics.json"],
        }
    )
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--support-manifest", type=Path, default=SUPPORT_MANIFEST)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--loading-audit", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--matrix-id", default=MATRIX_ID)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument(
        "--assignment-plan",
        type=Path,
        help="reviewed global plan whose selected shards this host must run",
    )
    parser.add_argument(
        "--global-gpu-ids",
        nargs="+",
        type=int,
        help="global plan GPU IDs mapped in order to this host's local GPUs",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--expect-plan-digest")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.matrix_id != MATRIX_ID:
        raise SystemExit(f"matrix-id must equal {MATRIX_ID}")
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
            raise SystemExit(
                f"another convergence controller is running: {existing_pid}"
            )
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")

    manifest_path = args.manifest.resolve()
    manifest = load_convergence_manifest(manifest_path)
    support_manifest = args.support_manifest.resolve()
    support = verify_extension_support(
        support_manifest_path=support_manifest,
        support_root=args.support_root.resolve(),
        node_order=args.node_order.resolve(),
    )
    loading_audit_path = args.loading_audit.resolve()
    loading_audit_sha256 = verify_loading_audit(loading_audit_path)
    calibration_path = args.calibration.resolve()
    task_seconds = load_task_calibration(calibration_path, manifest.tasks)
    microbatch_sizes = default_microbatch_sizes(manifest.tasks)
    global_plan_digest = None
    global_gpu_count = None
    global_gpu_ids = None
    if (args.assignment_plan is None) != (args.global_gpu_ids is None):
        raise SystemExit("assignment-plan and global-gpu-ids must be provided together")
    if args.assignment_plan is None:
        shards = assign_tasks_grouped_lpt(
            manifest.tasks,
            gpu_count=args.gpu_count,
            task_seconds=task_seconds,
            microbatch_sizes=microbatch_sizes,
        )
    else:
        try:
            assignment_plan = json.loads(
                args.assignment_plan.resolve().read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read global assignment plan: {exc}") from exc
        if not isinstance(assignment_plan, dict):
            raise SystemExit("global assignment plan must be a JSON object")
        try:
            shards, global_plan_digest, global_gpu_count = (
                select_global_assignment_shards(
                    manifest.tasks,
                    plan=assignment_plan,
                    global_gpu_ids=args.global_gpu_ids,
                    expected={
                        "matrix_id": args.matrix_id,
                        "source_manifest_sha256": _sha256_file(manifest_path),
                        "loading_audit_sha256": loading_audit_sha256,
                        "calibration_sha256": _sha256_file(calibration_path),
                        "code_sha": args.code_sha.lower(),
                        "cache_bundle_id": args.cache_bundle_id,
                        "cache_bundle_digest": args.cache_bundle_digest.lower(),
                    },
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        global_gpu_ids = tuple(args.global_gpu_ids)
        if len(shards) != args.gpu_count:
            raise SystemExit("local gpu-count must equal selected global GPU count")
    local_task_ids = {task.task_id for members in shards.values() for task in members}
    local_manifest = replace(
        manifest,
        tasks=tuple(task for task in manifest.tasks if task.task_id in local_task_ids),
    )
    local_microbatch_sizes = {
        task_id: size
        for task_id, size in microbatch_sizes.items()
        if task_id in local_task_ids
    }
    shard_manifests = write_shard_manifests(
        manifest_path,
        shards,
        control_root / "manifests",
        results_root_prefix=(
            "outputs/protocol_v3_noleak_dedup/remote_runs/" + args.matrix_id
        ),
        microbatch_sizes=local_microbatch_sizes,
    )
    shard_seconds = {
        str(gpu): float(sum(task_seconds[task.task_id] for task in tasks))
        for gpu, tasks in sorted(shards.items())
    }
    plan: dict[str, object] = {
        "matrix_id": args.matrix_id,
        "host_id": args.host_id,
        "execution_layout": (
            "single_host_multi_gpu_interleaved_convergence"
            if global_plan_digest is None
            else "dual_host_global_plan_shard_convergence"
        ),
        "created_at": _utc_now(),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "support_manifest": str(support_manifest),
        "support_manifest_sha256": support.manifest_sha256,
        "loading_audit": str(loading_audit_path),
        "loading_audit_sha256": loading_audit_sha256,
        "calibration": str(calibration_path),
        "calibration_sha256": _sha256_file(calibration_path),
        "code_sha": args.code_sha.lower(),
        "cache_bundle_id": args.cache_bundle_id,
        "cache_bundle_digest": args.cache_bundle_digest.lower(),
        "gpu_count": args.gpu_count,
        "task_count": len(local_manifest.tasks),
        "microbatch_sizes": dict(sorted(local_microbatch_sizes.items())),
        "alloc_conf": args.alloc_conf,
        "assignments": {
            str(gpu): [task.task_id for task in tasks]
            for gpu, tasks in sorted(shards.items())
        },
        "task_seconds": {
            task_id: float(task_seconds[task_id]) for task_id in sorted(local_task_ids)
        },
        "shard_seconds": shard_seconds,
    }
    if global_plan_digest is not None:
        plan.update(
            {
                "assignment_plan": str(args.assignment_plan.resolve()),
                "assignment_plan_sha256": _sha256_file(args.assignment_plan.resolve()),
                "global_plan_digest": global_plan_digest,
                "global_gpu_count": global_gpu_count,
                "global_gpu_ids": list(global_gpu_ids or ()),
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
    nas_plan = build_convergence_nas_plan(
        matrix_id=args.matrix_id,
        host_id=args.host_id,
        manifest=local_manifest,
        canonical_manifest=manifest_path,
        canonical_manifest_sha256=_sha256_file(manifest_path),
        support_manifest=support_manifest,
        support_manifest_sha256=support.manifest_sha256,
        loading_audit=loading_audit_path,
        loading_audit_sha256=loading_audit_sha256,
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
        microbatch_sizes=local_microbatch_sizes,
    )
    if global_plan_digest is not None:
        nas_plan.update(
            {
                "execution_layout": "dual_host_global_plan_shard_convergence",
                "assignment_plan": str(args.assignment_plan.resolve()),
                "assignment_plan_sha256": _sha256_file(args.assignment_plan.resolve()),
                "global_plan_digest": global_plan_digest,
                "global_gpu_count": global_gpu_count,
                "global_gpu_ids": list(global_gpu_ids or ()),
            }
        )
    _write_json_atomic(control_root / "nas_backup_plan.json", nas_plan)
    if args.expect_plan_digest and args.expect_plan_digest != plan["plan_digest"]:
        raise SystemExit("reviewed convergence plan digest does not match")
    if args.plan_only:
        print(
            f"PLAN_ONLY_OK tasks={len(local_manifest.tasks)} gpus={args.gpu_count} "
            f"makespan_seconds={max(shard_seconds.values()):.3f} "
            f"plan_digest={plan['plan_digest']}"
        )
        return 0

    if torch_device_count() < args.gpu_count:
        raise SystemExit("convergence plan requests more GPUs than are visible")
    children: dict[int, subprocess.Popen[bytes]] = {}
    logs: dict[int, object] = {}
    try:
        for gpu, tasks in sorted(shards.items()):
            log_path = control_root / "logs" / f"{shard_name(gpu)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab", buffering=0)
            logs[gpu] = handle
            command = [
                args.python_executable,
                "-X",
                "utf8",
                "-m",
                "scripts.remote.linux.protocol_v3_event_convergence_runner",
                "--allow-shard",
                "--manifest",
                str(shard_manifests[gpu]),
                "--support-manifest",
                str(support_manifest),
                "--repo-root",
                str(repo_root),
                "--results-root",
                str(results_root / shard_name(gpu)),
                "--cache-dir",
                str(args.cache_dir.resolve()),
                "--graph-dir",
                str(args.graph_dir.resolve()),
                "--support-root",
                str(support.support_root),
                "--node-order",
                str(support.node_order),
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
        exit_path = control_root / "runner.exitcode"
        if exit_path.exists():
            exit_path.unlink()
        while True:
            return_codes = {gpu: process.poll() for gpu, process in children.items()}
            status = aggregate_queue_status(
                manifest=local_manifest,
                shards=shards,
                results_root=results_root,
                return_codes=return_codes,
                cache_bundle_id=args.cache_bundle_id,
                cache_bundle_digest=args.cache_bundle_digest.lower(),
                code_sha=args.code_sha.lower(),
                total_tasks=len(local_manifest.tasks),
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
                            "global_gpu_id": (
                                None if global_gpu_ids is None else global_gpu_ids[gpu]
                            ),
                        }
                        for gpu in children
                    },
                },
            )
            if all(code is not None for code in return_codes.values()):
                exit_code = 0 if status["state"] == "completed" else 1
                exit_path.write_text(f"{exit_code}\n", encoding="ascii")
                return exit_code
            time.sleep(max(1.0, args.poll_seconds))
    finally:
        for handle in logs.values():
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
