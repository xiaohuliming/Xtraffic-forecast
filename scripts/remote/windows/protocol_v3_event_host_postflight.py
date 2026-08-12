#!/usr/bin/env python3
"""Read-only validation and export allowlist for a multi-root event shard."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SOURCE_LAYOUTS = ("canonical_contiguous", "interleaved")
ALLOWED_TASK_OVERRIDES = ("microbatch_size",)


def _bootstrap_repo_root() -> Path:
    configured = os.environ.get("XTRAFFIC_REPO_ROOT")
    marker = Path("scripts/remote/windows/protocol_v3_baseline_runner.py")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / marker).is_file():
            return root
    source = Path(__file__).resolve()
    for root in source.parents:
        if (root / marker).is_file():
            return root
    raise RuntimeError(
        "cannot locate XTraffic repository; set XTRAFFIC_REPO_ROOT"
    )


REPO_ROOT = _bootstrap_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.remote.windows import protocol_v3_runner as runner  # noqa: E402
from scripts.remote.windows.protocol_v3_baseline_runner import (  # noqa: E402
    build_baseline_invocation,
    load_baseline_manifest,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (  # noqa: E402
    ExtensionInputs,
    build_extension_invocation,
    load_extension_manifest,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_event_bist_ndr_runner import (  # noqa: E402
    build_bist_ndr_invocation,
    load_bist_ndr_manifest,
)
from scripts.remote.windows.protocol_v3_event_fdn_runner import (  # noqa: E402
    build_fdn_invocation,
    load_fdn_manifest,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (  # noqa: E402
    ConvergenceInputs,
    build_convergence_invocation,
    load_convergence_manifest,
)


class PostflightError(RuntimeError):
    """Raised when the host shard cannot be accepted read-only."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise PostflightError(f"symbolic link is forbidden: {path}")
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PostflightError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.stat(path, follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PostflightError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        return runner._read_strict_json_object(path)
    except runner.ManifestError as exc:
        raise PostflightError(str(exc)) from exc


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PostflightError(f"{name} must be a nonempty string")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostflightError(f"{name} must be an integer")
    return value


def _required_sha(payload: Mapping[str, object], name: str, size: int) -> str:
    value = _required_string(payload, name).lower()
    if len(value) != size or any(ch not in "0123456789abcdef" for ch in value):
        raise PostflightError(f"{name} must be a {size}-character hex digest")
    return value


def _resolve_path(plan_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (plan_path.parent / path).resolve()


def _collect_regular_files(path: Path) -> list[Path]:
    if path.is_symlink():
        raise PostflightError(f"symbolic link is forbidden: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise PostflightError(f"required export path is missing: {path}")
    files: list[Path] = []
    for child in path.rglob("*"):
        if child.is_symlink():
            raise PostflightError(f"symbolic link is forbidden: {child}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise PostflightError(f"not a regular export artifact: {child}")
        files.append(child)
    return files


def _source_layout(payload: Mapping[str, object]) -> str:
    layout = payload.get("source_layout", "canonical_contiguous")
    if layout not in SOURCE_LAYOUTS:
        raise PostflightError(f"unsupported source_layout: {layout!r}")
    return str(layout)


def _task_overrides(
    payload: Mapping[str, object],
    manifest,
) -> dict[str, dict[str, int]]:
    raw = payload.get("task_overrides")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise PostflightError("task_overrides must be a nonempty object")
    batch_size = manifest.training.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise PostflightError("canonical training.batch_size is invalid")
    overrides: dict[str, dict[str, int]] = {}
    for task_id, value in raw.items():
        if not isinstance(task_id, str) or not isinstance(value, dict) or not value:
            raise PostflightError(f"invalid task override entry: {task_id!r}")
        unknown = sorted(set(value).difference(ALLOWED_TASK_OVERRIDES))
        if unknown:
            raise PostflightError(f"unsupported task override keys: {unknown}")
        size = value.get("microbatch_size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise PostflightError(
                f"{task_id} microbatch_size must be an integer"
            )
        if not 1 <= size <= batch_size:
            raise PostflightError(
                f"{task_id} microbatch_size must be between 1 and {batch_size}"
            )
        overrides[task_id] = {"microbatch_size": size}
    return overrides


def _manifest_for_task(manifest, override: Mapping[str, int] | None):
    if not override:
        return manifest
    training = dict(manifest.training)
    training.update(override)
    return dataclasses.replace(manifest, training=training)


def _validate_assignment_order(
    *,
    layout: str,
    assigned: Sequence[str],
    per_source: Sequence[Sequence[str]],
    source_ids: Sequence[str],
    canonical_order: Sequence[str],
) -> None:
    present = set(assigned)
    expected_order = [task_id for task_id in canonical_order if task_id in present]
    if layout == "canonical_contiguous":
        if list(assigned) != expected_order:
            raise PostflightError("assigned tasks must follow canonical manifest order")
        return
    if list(source_ids) != sorted(source_ids):
        raise PostflightError(
            "interleaved sources must be listed in sorted source_id order"
        )
    position = {task_id: index for index, task_id in enumerate(canonical_order)}
    for source_id, task_ids in zip(source_ids, per_source):
        indices = [position[task_id] for task_id in task_ids]
        if indices != sorted(indices):
            raise PostflightError(
                f"source {source_id} tasks must follow canonical manifest order"
            )
    if sorted(assigned) != sorted(expected_order):
        raise PostflightError("assigned tasks must cover the canonical selection")


def _completion_gate(plan_path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    gate = payload.get("completion_gate")
    if not isinstance(gate, dict):
        raise PostflightError("completion_gate must be an object")
    results_root = _resolve_path(
        plan_path, _required_string(gate, "results_root")
    )
    queue_path = results_root / "queue_status.json"
    queue_before = _load_json(queue_path)
    expected_tasks = _required_int(gate, "expected_tasks")
    expected = {
        "state": "completed",
        "total_tasks": expected_tasks,
        "completed_tasks": expected_tasks,
        "current_task": None,
    }
    for name, value in expected.items():
        if queue_before.get(name) != value:
            raise PostflightError(
                f"completion gate queue {name} mismatch: expected "
                f"{value!r}, got {queue_before.get(name)!r}"
            )
    failed = queue_before.get("failed_count", 0)
    if failed != 0:
        raise PostflightError(
            f"completion gate queue failed_count must be zero, got {failed!r}"
        )
    exit_path = _resolve_path(
        plan_path, _required_string(gate, "runner_exit_code")
    )
    try:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        raise PostflightError(f"invalid runner exit code file: {exit_path}") from exc
    if exit_code != 0:
        raise PostflightError(f"runner exit code must be zero, got {exit_code}")
    if _load_json(queue_path) != queue_before:
        raise PostflightError("completion gate queue changed during validation")
    return {
        "results_root": str(results_root),
        "expected_tasks": expected_tasks,
        "runner_exit_code_path": str(exit_path),
        "runner_exit_code": exit_code,
        "queue_status": queue_before,
        "queue_status_sha256": sha256_file(queue_path),
    }


def build_postflight_report(plan_path: Path | str) -> dict[str, object]:
    plan_path = Path(plan_path).expanduser().resolve()
    plan = _load_json(plan_path)
    if _required_int(plan, "plan_version") != 1:
        raise PostflightError("unsupported plan_version")
    canonical_path = _resolve_path(
        plan_path, _required_string(plan, "canonical_manifest")
    )
    expected_manifest_sha = _required_sha(
        plan, "canonical_manifest_sha256", 64
    )
    if sha256_file(canonical_path) != expected_manifest_sha:
        raise PostflightError("canonical manifest SHA-256 mismatch")
    runner_kind = plan.get("runner_kind", "baseline")
    if runner_kind not in {
        "baseline",
        "event_extension",
        "event_bist_ndr",
        "event_fdn",
        "event_convergence",
    }:
        raise PostflightError(f"unsupported runner_kind: {runner_kind!r}")
    try:
        if runner_kind == "baseline":
            manifest = load_baseline_manifest(canonical_path)
        elif runner_kind == "event_extension":
            manifest = load_extension_manifest(canonical_path)
        elif runner_kind == "event_bist_ndr":
            manifest = load_bist_ndr_manifest(canonical_path)
        elif runner_kind == "event_fdn":
            manifest = load_fdn_manifest(canonical_path)
        else:
            manifest = load_convergence_manifest(canonical_path)
    except runner.ManifestError as exc:
        raise PostflightError(str(exc)) from exc

    protocol_version = _required_string(plan, "protocol_version")
    cache_schema_version = _required_int(plan, "cache_schema_version")
    if protocol_version != manifest.protocol_version:
        raise PostflightError("plan protocol_version mismatch")
    if cache_schema_version != manifest.cache_schema_version:
        raise PostflightError("plan cache_schema_version mismatch")
    expected_task_count = _required_int(plan, "expected_task_count")
    if expected_task_count <= 0:
        raise PostflightError("expected_task_count must be positive")
    cache_bundle_id = _required_string(plan, "cache_bundle_id")
    cache_bundle_digest = _required_sha(plan, "cache_bundle_digest", 64)
    cache_dir = _resolve_path(plan_path, _required_string(plan, "cache_dir"))
    graph_dir = _resolve_path(plan_path, _required_string(plan, "graph_dir"))
    python_executable = _required_string(plan, "python_executable")

    layout = _source_layout(plan)
    if runner_kind in {"baseline", "event_fdn"}:
        if runner_kind == "event_fdn" and plan.get("task_overrides") is not None:
            raise PostflightError("event_fdn does not accept task_overrides")
        task_overrides = (
            _task_overrides(plan, manifest)
            if runner_kind == "baseline"
            else {}
        )
        extension = None
        support_report: dict[str, object] = {}
    else:
        if (
            runner_kind in {"event_extension", "event_bist_ndr"}
            and plan.get("task_overrides") is not None
        ):
            raise PostflightError(
                f"{runner_kind} does not accept task_overrides"
            )
        task_overrides = (
            _task_overrides(plan, manifest)
            if runner_kind == "event_convergence"
            else {}
        )
        support_manifest = _resolve_path(
            plan_path, _required_string(plan, "support_manifest")
        )
        expected_support_sha = _required_sha(
            plan, "support_manifest_sha256", 64
        )
        if sha256_file(support_manifest) != expected_support_sha:
            raise PostflightError("support manifest SHA-256 mismatch")
        loading_audit = _resolve_path(
            plan_path, _required_string(plan, "loading_audit")
        )
        expected_loading_audit_sha = _required_sha(
            plan, "loading_audit_sha256", 64
        )
        if sha256_file(loading_audit) != expected_loading_audit_sha:
            raise PostflightError("loading audit SHA-256 mismatch")
        support_root = _resolve_path(
            plan_path, _required_string(plan, "support_root")
        )
        node_order = _resolve_path(
            plan_path, _required_string(plan, "node_order")
        )
        try:
            verified_support = verify_extension_support(
                support_manifest_path=support_manifest,
                support_root=support_root,
                node_order=node_order,
            )
        except runner.ManifestError as exc:
            raise PostflightError(str(exc)) from exc
        raw_options = plan.get("extension_options")
        expected_option_keys = {
            "igstgnn_data_parallel",
            "max_microbatch_size",
        }
        if runner_kind in {"event_extension", "event_bist_ndr"}:
            expected_option_keys.add("task_microbatch_sizes")
        if not isinstance(raw_options, dict) or set(raw_options) != expected_option_keys:
            raise PostflightError("extension_options must contain the exact contract")
        data_parallel = raw_options.get("igstgnn_data_parallel")
        microbatch_size = raw_options.get("max_microbatch_size")
        task_microbatch_sizes = raw_options.get("task_microbatch_sizes")
        if not isinstance(data_parallel, bool):
            raise PostflightError("igstgnn_data_parallel must be boolean")
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or microbatch_size < 0
        ):
            raise PostflightError("max_microbatch_size must be a nonnegative integer")
        if runner_kind in {"event_extension", "event_bist_ndr"}:
            methods = (
                {"igstgnn", "ndr_net"}
                if runner_kind == "event_extension"
                else {"bist_ndr"}
            )
            expected_task_microbatches = {
                task.task_id: 8
                for task in manifest.tasks
                if task.region == "Orange" and task.model in methods
            }
            if task_microbatch_sizes != expected_task_microbatches:
                raise PostflightError("task_microbatch_sizes do not match the contract")
        extension = ExtensionInputs(
            support_root=verified_support.support_root,
            node_order=verified_support.node_order,
            support_manifest_sha256=verified_support.manifest_sha256,
            igstgnn_data_parallel=data_parallel,
            max_microbatch_size=microbatch_size,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            python_executable=python_executable,
        )
        support_report = {
            "support_manifest": str(support_manifest),
            "support_manifest_sha256": expected_support_sha,
            "loading_audit": str(loading_audit),
            "loading_audit_sha256": expected_loading_audit_sha,
            "support_root": str(verified_support.support_root),
            "node_order": str(verified_support.node_order),
            "extension_options": dict(raw_options),
        }
    raw_sources = plan.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PostflightError("sources must be a nonempty list")
    canonical_by_id = {task.task_id: task for task in manifest.tasks}
    per_source_tasks: list[list[str]] = []
    ordered_source_ids: list[str] = []
    assigned: list[str] = []
    source_reports: list[dict[str, object]] = []
    task_reports: list[dict[str, object]] = []
    file_records: list[dict[str, object]] = []
    state_counts: Counter[str] = Counter()
    source_ids: set[str] = set()

    for source_payload in raw_sources:
        if not isinstance(source_payload, dict):
            raise PostflightError("each source must be an object")
        source_id = _required_string(source_payload, "source_id")
        if source_id in source_ids or not source_id.replace("_", "").isalnum():
            raise PostflightError(f"unsafe or duplicate source_id: {source_id}")
        source_ids.add(source_id)
        results_root = _resolve_path(
            plan_path, _required_string(source_payload, "results_root")
        )
        code_sha = _required_sha(source_payload, "code_sha", 40)
        task_ids = source_payload.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or any(not isinstance(value, str) for value in task_ids)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise PostflightError(f"source {source_id} task_ids are invalid")
        unknown = sorted(set(task_ids).difference(canonical_by_id))
        if unknown:
            raise PostflightError(
                f"source {source_id} contains unknown tasks: {unknown}"
            )
        assigned.extend(task_ids)
        per_source_tasks.append(list(task_ids))
        ordered_source_ids.append(source_id)
        actual_task_root = results_root / "tasks"
        if actual_task_root.is_symlink() or not actual_task_root.is_dir():
            raise PostflightError(f"source task root is invalid: {actual_task_root}")
        actual_task_ids = {path.name for path in actual_task_root.iterdir()}
        expected_source_ids = set(task_ids)
        if actual_task_ids != expected_source_ids:
            raise PostflightError(
                f"source {source_id} task directory mismatch: "
                f"missing={sorted(expected_source_ids - actual_task_ids)}, "
                f"extra={sorted(actual_task_ids - expected_source_ids)}"
            )
        run_inputs = runner.ProtocolV3Inputs(
            results_root=results_root,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            cache_bundle_id=cache_bundle_id,
            cache_bundle_digest=cache_bundle_digest,
            code_sha=code_sha,
            python_executable=python_executable,
        )
        source_task_reports: list[dict[str, object]] = []
        allowed: set[Path] = set()
        queue_path = results_root / "queue_status.json"
        allowed.update(_collect_regular_files(queue_path))
        queue_status = _load_json(queue_path)
        for task_id in task_ids:
            task = canonical_by_id[task_id]
            task_manifest = _manifest_for_task(
                manifest, task_overrides.get(task_id)
            )
            probe = results_root / ".staging" / task_id / "__POSTFLIGHT__"
            if runner_kind == "baseline":
                invocation = build_baseline_invocation(
                    task, task_manifest, run_inputs, probe
                )
            elif runner_kind == "event_extension":
                invocation = build_extension_invocation(
                    task, task_manifest, extension, probe  # type: ignore[arg-type]
                )
            elif runner_kind == "event_bist_ndr":
                invocation = build_bist_ndr_invocation(
                    task, task_manifest, extension, probe  # type: ignore[arg-type]
                )
            elif runner_kind == "event_fdn":
                invocation = build_fdn_invocation(
                    task, task_manifest, run_inputs, probe
                )
            else:
                invocation = build_convergence_invocation(
                    task,
                    task_manifest,
                    ConvergenceInputs(
                        protocol=run_inputs,
                        extension=extension,  # type: ignore[arg-type]
                    ),
                    probe,
                )
            config_hash = runner.derive_config_hash(
                task,
                task_manifest,
                run_inputs,
                invocation_config=invocation.config,
            )
            result_dir = results_root / task.output_dir
            try:
                runner.validate_result(
                    result_dir,
                    task,
                    task_manifest,
                    run_inputs,
                    expected_config_hash=config_hash,
                    required_artifacts=invocation.required_artifacts,
                    required_metrics=invocation.required_metrics,
                )
            except runner.ResultValidationError as exc:
                raise PostflightError(f"{task_id}: {exc}") from exc
            state_path = results_root / "state/tasks" / f"{task_id}.json"
            state = _load_json(state_path)
            if state.get("task_id") != task_id:
                raise PostflightError(f"task state identity mismatch: {task_id}")
            task_state = state.get("state")
            if task_state not in {"completed", "skipped_valid"}:
                raise PostflightError(
                    f"task state must be completed or skipped_valid for "
                    f"{task_id}, got {task_state!r}"
                )
            if state.get("config_hash") != config_hash:
                raise PostflightError(
                    f"task state config_hash mismatch: {task_id}"
                )
            state_counts[str(task_state)] += 1
            allowed.update(_collect_regular_files(result_dir))
            allowed.update(_collect_regular_files(state_path))
            log_dir = results_root / "logs" / task_id
            if log_dir.exists() or log_dir.is_symlink():
                allowed.update(_collect_regular_files(log_dir))
            item = {
                "task_id": task_id,
                "source_id": source_id,
                "state": task_state,
                "code_sha": code_sha,
                "config_hash": config_hash,
                "invocation_config": dict(invocation.config),
            }
            source_task_reports.append(item)
            task_reports.append(item)

        source_file_records: list[dict[str, object]] = []
        for local_path in sorted(allowed, key=lambda item: item.as_posix()):
            relative = local_path.relative_to(results_root).as_posix()
            record = {
                "path": f"sources/{source_id}/results/{relative}",
                "source_path": str(local_path.resolve()),
                "size_bytes": local_path.stat().st_size,
                "sha256": sha256_file(local_path),
                "source_id": source_id,
            }
            source_file_records.append(record)
            file_records.append(record)
        source_reports.append(
            {
                "source_id": source_id,
                "results_root": str(results_root),
                "code_sha": code_sha,
                "queue_status": queue_status,
                "task_ids": list(task_ids),
                "validated_tasks": len(source_task_reports),
                "file_count": len(source_file_records),
                "total_bytes": sum(
                    int(record["size_bytes"]) for record in source_file_records
                ),
                "files_digest": _sha256_json(source_file_records),
            }
        )

    if len(assigned) != expected_task_count or len(set(assigned)) != len(assigned):
        raise PostflightError(
            "assigned task union must be unique and equal expected_task_count"
        )
    unassigned_overrides = sorted(set(task_overrides).difference(assigned))
    if unassigned_overrides:
        raise PostflightError(
            f"task_overrides name unassigned tasks: {unassigned_overrides}"
        )
    _validate_assignment_order(
        layout=layout,
        assigned=assigned,
        per_source=per_source_tasks,
        source_ids=ordered_source_ids,
        canonical_order=[task.task_id for task in manifest.tasks],
    )
    gate_report = _completion_gate(plan_path, plan)
    plan_after = _load_json(plan_path)
    if plan_after != plan:
        raise PostflightError("backup plan changed during validation")
    if not file_records:
        raise PostflightError("export allowlist is empty")
    return {
        "status": "verified",
        "runner_kind": runner_kind,
        "matrix_id": _required_string(plan, "matrix_id"),
        "host_id": _required_string(plan, "host_id"),
        "protocol_version": protocol_version,
        "cache_schema_version": cache_schema_version,
        "cache_bundle_id": cache_bundle_id,
        "cache_bundle_digest": cache_bundle_digest,
        "canonical_manifest": str(canonical_path),
        "canonical_manifest_sha256": expected_manifest_sha,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "source_layout": layout,
        "task_overrides": task_overrides,
        "validated_tasks": len(task_reports),
        "task_state_counts": dict(state_counts),
        "tasks": task_reports,
        "sources": source_reports,
        "completion_gate": gate_report,
        "file_count": len(file_records),
        "total_bytes": sum(int(record["size_bytes"]) for record in file_records),
        "files_digest": _sha256_json(file_records),
        "files": file_records,
        **support_report,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    *,
    report: Mapping[str, object],
    report_path: Path,
    file_list_path: Path,
    transfer_manifest_path: Path,
    marker_path: Path,
) -> None:
    files = report.get("files")
    if not isinstance(files, list):
        raise PostflightError("report files must be a list")
    report_text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    paths = [str(record["path"]) for record in files]
    manifest_lines = [
        f"{record['sha256']}  {record['path']}" for record in files
    ]
    _atomic_write(report_path, report_text)
    _atomic_write(file_list_path, "\n".join(paths) + "\n")
    _atomic_write(
        transfer_manifest_path, "\n".join(manifest_lines) + "\n"
    )
    marker = {
        "status": "verified",
        "matrix_id": report["matrix_id"],
        "host_id": report["host_id"],
        "validated_tasks": report["validated_tasks"],
        "file_count": report["file_count"],
        "total_bytes": report["total_bytes"],
        "files_digest": report["files_digest"],
        "plan_sha256": report["plan_sha256"],
        "postflight_sha256": sha256_file(report_path),
        "file_list_sha256": sha256_file(file_list_path),
        "transfer_manifest_sha256": sha256_file(transfer_manifest_path),
    }
    _atomic_write(
        marker_path,
        json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_postflight_report(args.plan)
        write_outputs(
            report=report,
            report_path=args.report,
            file_list_path=args.file_list,
            transfer_manifest_path=args.transfer_manifest,
            marker_path=args.marker,
        )
    except (OSError, PostflightError) as exc:
        print(f"EVENT_HOST_POSTFLIGHT_FAILED {exc}", file=sys.stderr)
        return 1
    print(
        "EVENT_HOST_POSTFLIGHT_OK "
        f"host={report['host_id']} tasks={report['validated_tasks']} "
        f"files={report['file_count']} bytes={report['total_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
