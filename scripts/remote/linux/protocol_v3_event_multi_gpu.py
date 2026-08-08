#!/usr/bin/env python3
"""Run a region-complete v3 event baseline shard across N GPUs on one host."""

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

from scripts.remote.windows.protocol_v3_runner import Manifest, Task
from scripts.remote.windows.protocol_v3_baseline_runner import (
    load_baseline_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT / "config" / "protocol_v3" / "table4_event_baselines_phase1_v1.json"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "config"
    / "protocol_v3"
    / "event_baseline_gpu_calibration_v1.json"
)
SUPPORTED_REGIONS = ("Alameda", "ContraCosta", "Orange")
MAX_GPU_COUNT = 99
PLAN_DIGEST_FIELDS = (
    "matrix_id",
    "host_id",
    "execution_layout",
    "source_manifest_sha256",
    "calibration_sha256",
    "code_sha",
    "cache_bundle_id",
    "cache_bundle_digest",
    "regions",
    "gpu_count",
    "task_count",
    "microbatch_sizes",
    "alloc_conf",
    "assignments",
)
DEFAULT_ALLOC_CONF = "expandable_segments:True"


class PlanError(RuntimeError):
    """Raised when a multi-GPU host plan violates its contract."""


def shard_name(gpu: int) -> str:
    return f"gpu{gpu:02d}"


def plan_digest(plan: Mapping[str, object]) -> str:
    subset = {name: plan.get(name) for name in PLAN_DIGEST_FIELDS}
    payload = json.dumps(
        subset, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def load_calibration(path: Path | str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("calibration_version") != 1:
        raise PlanError("unsupported calibration_version")
    epoch_seconds = payload.get("epoch_seconds")
    region_scale = payload.get("region_scale")
    if not isinstance(epoch_seconds, dict) or not epoch_seconds:
        raise PlanError("calibration epoch_seconds must be a nonempty object")
    if not isinstance(region_scale, dict) or not region_scale:
        raise PlanError("calibration region_scale must be a nonempty object")
    for name, value in list(epoch_seconds.items()) + list(region_scale.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(f"calibration entry {name} must be a number")
        if not value > 0:
            raise PlanError(f"calibration entry {name} must be positive")
    return payload


def task_cost(task: Task, calibration: Mapping[str, object]) -> float:
    epoch_seconds = calibration["epoch_seconds"]
    region_scale = calibration["region_scale"]
    if task.model not in epoch_seconds:
        raise PlanError(f"calibration has no epoch cost for model {task.model}")
    if task.region not in region_scale:
        raise PlanError(f"calibration has no scale for region {task.region}")
    return float(epoch_seconds[task.model]) * float(region_scale[task.region])


def select_region_tasks(
    manifest: Manifest,
    regions: Sequence[str],
) -> tuple[Task, ...]:
    requested = tuple(dict.fromkeys(regions))
    if not requested:
        raise PlanError("at least one region is required")
    unknown = [region for region in requested if region not in SUPPORTED_REGIONS]
    if unknown:
        raise PlanError(f"unsupported regions: {unknown}")
    selected = tuple(task for task in manifest.tasks if task.region in requested)
    by_region: dict[str, set[str]] = {}
    for task in selected:
        by_region.setdefault(task.region, set()).add(task.model)
    complete = {task.model for task in manifest.tasks if task.region == requested[0]}
    for region in requested:
        if by_region.get(region) != complete:
            raise PlanError(
                f"region {region} must contribute every canonical method on one host"
            )
    return selected


def _allocate_gpus(
    group_loads: Mapping[object, float],
    group_sizes: Mapping[object, int],
    gpu_count: int,
) -> dict[object, int]:
    keys = sorted(group_loads, key=lambda key: (key is not None, key))
    if len(keys) > gpu_count:
        raise PlanError(
            f"{len(keys)} distinct microbatch groups need at least that many GPUs, "
            f"got {gpu_count}"
        )
    allocation = {key: 1 for key in keys}
    for _ in range(gpu_count - len(keys)):
        # A group can never use more GPUs than it has tasks; a task is atomic.
        candidates = [key for key in keys if allocation[key] < group_sizes[key]]
        if not candidates:
            break
        target = max(
            candidates,
            key=lambda key: (group_loads[key] / allocation[key], key is None),
        )
        allocation[target] += 1
    return allocation


def assign_tasks_lpt(
    tasks: Sequence[Task],
    gpu_count: int,
    calibration: Mapping[str, object],
    *,
    microbatch_sizes: Mapping[str, int] | None = None,
) -> dict[int, tuple[Task, ...]]:
    """Pack tasks onto GPUs, keeping each microbatch group on its own GPU set.

    A shard manifest carries one training config, so tasks that need different
    microbatch sizes can never share a GPU. Tasks that need the same size can.
    """
    if not 1 <= gpu_count <= MAX_GPU_COUNT:
        raise PlanError(f"gpu_count must be between 1 and {MAX_GPU_COUNT}")
    if len(tasks) < gpu_count:
        raise PlanError("gpu_count must not exceed the number of tasks")
    overrides = dict(microbatch_sizes or {})
    groups: dict[object, list[Task]] = {}
    for task in tasks:
        groups.setdefault(overrides.get(task.task_id), []).append(task)
    group_loads = {
        key: sum(task_cost(task, calibration) for task in members)
        for key, members in groups.items()
    }
    allocation = _allocate_gpus(
        group_loads, {key: len(members) for key, members in groups.items()}, gpu_count
    )
    buckets: list[list[Task]] = [[] for _ in range(gpu_count)]
    cursor = 0
    for key in sorted(groups, key=lambda value: (value is not None, value)):
        slots = list(range(cursor, cursor + allocation[key]))
        cursor += allocation[key]
        loads = {slot: 0.0 for slot in slots}
        ordered = sorted(
            groups[key],
            key=lambda task: (-task_cost(task, calibration), task.task_id),
        )
        for task in ordered:
            target = min(slots, key=lambda slot: (loads[slot], slot))
            buckets[target].append(task)
            loads[target] += task_cost(task, calibration)
    used = {gpu: tuple(bucket) for gpu, bucket in enumerate(buckets) if bucket}
    if not used:
        raise PlanError("no GPU received any task")
    return used


def shard_load_units(
    shards: Mapping[int, Sequence[Task]],
    calibration: Mapping[str, object],
) -> dict[int, float]:
    return {
        gpu: round(sum(task_cost(task, calibration) for task in tasks), 3)
        for gpu, tasks in shards.items()
    }


def write_shard_manifests(
    source_path: Path,
    shards: Mapping[int, Sequence[Task]],
    output_dir: Path,
    *,
    results_root_prefix: str,
    microbatch_sizes: Mapping[str, int] | None = None,
) -> dict[int, Path]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_tasks = {task["task_id"]: task for task in payload["tasks"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for gpu, tasks in shards.items():
        shard_payload = dict(payload)
        shard_payload["training"] = dict(payload["training"])
        if microbatch_sizes:
            declared = {
                microbatch_sizes.get(task.task_id) for task in tasks
            }
            if declared != {None}:
                if None in declared or len(declared) != 1:
                    raise PlanError(
                        f"GPU {gpu} mixes microbatch settings {sorted(declared, key=str)}; "
                        "a shard manifest carries exactly one training config"
                    )
                shard_payload["training"]["microbatch_size"] = int(
                    declared.pop()
                )
        shard_payload["default_results_root"] = f"{results_root_prefix}_{shard_name(gpu)}"
        shard_payload["tasks"] = [source_tasks[task.task_id] for task in tasks]
        path = output_dir / f"shard_{shard_name(gpu)}.json"
        path.write_text(
            json.dumps(shard_payload, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        paths[gpu] = path
    return paths


def _read_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def aggregate_queue_status(
    *,
    manifest: Manifest,
    shards: Mapping[int, Sequence[Task]],
    results_root: Path,
    return_codes: Mapping[int, int | None],
    cache_bundle_id: str,
    cache_bundle_digest: str,
    code_sha: str,
    total_tasks: int,
) -> dict[str, object]:
    completed = 0
    skipped = 0
    failed_count = 0
    current_tasks: dict[str, str] = {}
    shard_states: dict[str, object] = {}
    for gpu in sorted(shards):
        queue = _read_json_if_present(
            results_root / shard_name(gpu) / "queue_status.json"
        )
        state = "pending" if queue is None else str(queue.get("state"))
        if queue is not None:
            completed += int(queue.get("completed_tasks") or 0)
            skipped += int(queue.get("skipped_valid_tasks") or 0)
            current = queue.get("current_task")
            if state == "failed":
                failed_count += 1
            if isinstance(current, str) and state == "running":
                current_tasks[str(gpu)] = current
        if return_codes.get(gpu) not in (None, 0):
            failed_count += 1 if state != "failed" else 0
        shard_states[str(gpu)] = {
            "state": state,
            "return_code": return_codes.get(gpu),
            "task_ids": [task.task_id for task in shards[gpu]],
            "completed_tasks": 0 if queue is None else queue.get("completed_tasks"),
        }
    every_child_exited = all(code is not None for code in return_codes.values())
    every_child_zero = all(code == 0 for code in return_codes.values())
    all_shards_completed = all(
        entry["state"] == "completed" for entry in shard_states.values()
    )
    if failed_count:
        state = "failed"
    elif (
        every_child_exited
        and every_child_zero
        and all_shards_completed
        and completed == total_tasks
    ):
        state = "completed"
    elif every_child_exited:
        state = "failed"
    else:
        state = "running"
    ordered_current = [current_tasks[key] for key in sorted(current_tasks)]
    return {
        "state": state,
        "updated_at": _utc_now(),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "code_sha": code_sha,
        "canonical_manifest_digest": manifest.digest,
        "total_tasks": total_tasks,
        "completed_tasks": completed,
        "skipped_valid_tasks": skipped,
        "failed_count": failed_count,
        "current_task": None if state == "completed" else (
            ordered_current[0] if ordered_current else None
        ),
        "current_tasks": current_tasks,
        "shards": shard_states,
    }


def build_nas_plan(
    *,
    matrix_id: str,
    host_id: str,
    manifest: Manifest,
    canonical_manifest: Path,
    canonical_manifest_sha256: str,
    shards: Mapping[int, Sequence[Task]],
    results_root: Path,
    control_root: Path,
    cache_dir: Path,
    graph_dir: Path,
    cache_bundle_id: str,
    cache_bundle_digest: str,
    code_sha: str,
    python_executable: str,
    microbatch_sizes: Mapping[str, int] | None = None,
) -> dict[str, object]:
    canonical_order = [task.task_id for task in manifest.tasks]
    sources = []
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
    source_ids = [source["source_id"] for source in sources]
    if source_ids != sorted(source_ids):
        raise PlanError("NAS plan sources must be emitted in sorted source_id order")
    ordered = [task_id for task_id in canonical_order if task_id in set(assigned)]
    if sorted(assigned) != sorted(ordered):
        raise PlanError("NAS plan task union does not match the canonical selection")
    plan: dict[str, object] = {
        "plan_version": 1,
        "matrix_id": matrix_id,
        "host_id": host_id,
        "canonical_manifest": str(canonical_manifest),
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "cache_dir": str(cache_dir),
        "graph_dir": str(graph_dir),
        "python_executable": python_executable,
        "expected_task_count": len(assigned),
        "source_layout": "interleaved",
        "sources": sources,
        "completion_gate": {
            "results_root": str(results_root),
            "runner_exit_code": str(control_root / "runner.exitcode"),
            "expected_tasks": len(assigned),
        },
    }
    if microbatch_sizes:
        plan["task_overrides"] = {
            task_id: {"microbatch_size": int(size)}
            for task_id, size in sorted(microbatch_sizes.items())
        }
    return plan


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
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument(
        "--microbatch",
        action="append",
        default=[],
        metavar="TASK_ID=SIZE",
        help="fallback activation limit; the task is isolated to its own shard",
    )
    parser.add_argument(
        "--alloc-conf",
        default=DEFAULT_ALLOC_CONF,
        help="PYTORCH_CUDA_ALLOC_CONF for every shard; part of the run contract "
             "because the tightest task was measured under it",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--expect-plan-digest",
        default=None,
        help="refuse to launch unless the regenerated plan matches this digest",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write plan, shard manifests and NAS plan without launching trainers",
    )
    return parser.parse_args(argv)


def _parse_microbatch(
    values: Sequence[str],
    tasks: Sequence[Task],
    batch_size: int,
) -> dict[str, int]:
    known = {task.task_id for task in tasks}
    sizes: dict[str, int] = {}
    for value in values:
        task_id, _, raw = value.partition("=")
        if not task_id or not raw.isdigit():
            raise PlanError(f"invalid microbatch override: {value}")
        size = int(raw)
        if not 1 <= size <= batch_size:
            raise PlanError(
                f"microbatch override for {task_id} must be between 1 and "
                f"{batch_size}, got {size}"
            )
        if task_id not in known:
            raise PlanError(f"microbatch override names an unselected task: {task_id}")
        sizes[task_id] = size
    return sizes


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
            raise SystemExit(f"another controller is already running: {existing_pid}")
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")

    manifest = load_baseline_manifest(source_path)
    if {task.sampling_protocol for task in manifest.tasks} != {"event"}:
        raise SystemExit("multi-GPU controller requires the event manifest")
    calibration = load_calibration(args.calibration)
    tasks = select_region_tasks(manifest, args.regions)
    batch_size = manifest.training.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise SystemExit("canonical training.batch_size is invalid")
    microbatch_sizes = _parse_microbatch(args.microbatch, tasks, batch_size)
    shards = assign_tasks_lpt(
        tasks,
        args.gpu_count,
        calibration,
        microbatch_sizes=microbatch_sizes,
    )
    manifest_paths = write_shard_manifests(
        source_path,
        shards,
        control_root / "manifests",
        results_root_prefix=(
            "outputs/protocol_v3_noleak_dedup/remote_runs/" + args.matrix_id
        ),
        microbatch_sizes=microbatch_sizes,
    )
    canonical_sha = _sha256_file(source_path)
    plan = {
        "matrix_id": args.matrix_id,
        "host_id": args.host_id,
        "execution_layout": "single_host_multi_gpu_region_complete",
        "created_at": _utc_now(),
        "source_manifest": str(source_path),
        "source_manifest_sha256": canonical_sha,
        "calibration": str(args.calibration.resolve()),
        "calibration_sha256": _sha256_file(args.calibration.resolve()),
        "code_sha": args.code_sha.lower(),
        "cache_bundle_id": args.cache_bundle_id,
        "cache_bundle_digest": args.cache_bundle_digest.lower(),
        "regions": list(dict.fromkeys(args.regions)),
        "gpu_count": args.gpu_count,
        "task_count": len(tasks),
        "microbatch_sizes": microbatch_sizes or None,
        "alloc_conf": args.alloc_conf,
        "assignments": {
            str(gpu): [task.task_id for task in shard] for gpu, shard in shards.items()
        },
        "shard_load_units": {
            str(gpu): value
            for gpu, value in shard_load_units(shards, calibration).items()
        },
    }
    plan["plan_digest"] = plan_digest(plan)
    _write_json_atomic(control_root / "plan.json", plan)
    nas_plan = build_nas_plan(
        matrix_id=args.matrix_id,
        host_id=args.host_id,
        manifest=manifest,
        canonical_manifest=source_path,
        canonical_manifest_sha256=canonical_sha,
        shards=shards,
        results_root=results_root,
        control_root=control_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=args.cache_bundle_id,
        cache_bundle_digest=args.cache_bundle_digest.lower(),
        code_sha=args.code_sha.lower(),
        python_executable=args.python_executable,
        microbatch_sizes=microbatch_sizes,
    )
    _write_json_atomic(control_root / "nas_backup_plan.json", nas_plan)

    if args.expect_plan_digest and args.expect_plan_digest != plan["plan_digest"]:
        raise SystemExit(
            "plan digest mismatch: reviewed "
            f"{args.expect_plan_digest}, regenerated {plan['plan_digest']}"
        )

    if args.plan_only:
        print(
            "PLAN_ONLY_OK "
            f"tasks={len(tasks)} gpus={args.gpu_count} "
            f"makespan_units={max(shard_load_units(shards, calibration).values()):.1f} "
            f"plan_digest={plan['plan_digest']}"
        )
        return 0

    import torch

    device_count = torch.cuda.device_count()
    if device_count < args.gpu_count:
        raise SystemExit(
            f"plan requires {args.gpu_count} CUDA devices, got {device_count}"
        )

    children: dict[int, subprocess.Popen[bytes]] = {}
    log_streams: dict[int, object] = {}
    exit_path = control_root / "runner.exitcode"
    if exit_path.exists():
        exit_path.unlink()
    try:
        for gpu in sorted(shards):
            log_path = control_root / "logs" / f"{shard_name(gpu)}.log"
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
                str(results_root / shard_name(gpu)),
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
            if args.alloc_conf:
                environment["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf
            children[gpu] = subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        while True:
            return_codes = {gpu: child.poll() for gpu, child in children.items()}
            status = aggregate_queue_status(
                manifest=manifest,
                shards=shards,
                results_root=results_root,
                return_codes=return_codes,
                cache_bundle_id=args.cache_bundle_id,
                cache_bundle_digest=args.cache_bundle_digest.lower(),
                code_sha=args.code_sha.lower(),
                total_tasks=len(tasks),
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
                            "tasks": plan["assignments"][str(gpu)],
                        }
                        for gpu in children
                    },
                },
            )
            if all(code is not None for code in return_codes.values()):
                exit_code = 0 if status["state"] == "completed" else 1
                exit_path.write_text(f"{exit_code}\n", encoding="ascii")
                return exit_code
            time.sleep(max(args.poll_seconds, 1.0))
    finally:
        for stream in log_streams.values():
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
