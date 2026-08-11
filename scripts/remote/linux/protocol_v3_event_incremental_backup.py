#!/usr/bin/env python3
"""Incrementally mirror completed protocol-v3 tasks to a disaster-backup tree.

The formal finalizer remains the only component allowed to publish a matrix. This
sidecar copies only runner-validated, atomically promoted task outputs. It never
copies staging checkpoints and never promotes the remote tree.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts.remote.windows import protocol_v3_runner as runner
from scripts.remote.windows.protocol_v3_baseline_runner import (
    build_baseline_invocation,
    load_baseline_manifest,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    ExtensionInputs,
    build_extension_invocation,
    load_extension_manifest,
    verify_extension_support,
)
from scripts.remote.linux.protocol_v3_event_convergence_runner import (
    ConvergenceInputs,
    build_convergence_invocation,
    load_convergence_manifest,
)


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_RESERVE_BYTES = 256 * 1024 * 1024


class BackupError(RuntimeError):
    """Raised when an incremental backup cannot be proven safe."""


@dataclass(frozen=True)
class FileRecord:
    source_path: Path
    remote_relative: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CompletedTaskSnapshot:
    matrix_id: str
    host_id: str
    source_id: str
    task_id: str
    config_hash: str
    state_sha256: str
    files: tuple[FileRecord, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "verified_completed_task_backup",
            "matrix_id": self.matrix_id,
            "host_id": self.host_id,
            "source_id": self.source_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "state_sha256": self.state_sha256,
            "files": [
                {
                    "path": item.remote_relative,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
            "files_digest": _sha256_json(
                [
                    {
                        "path": item.remote_relative,
                        "size_bytes": item.size_bytes,
                        "sha256": item.sha256,
                    }
                    for item in self.files
                ]
            ),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BackupError(f"required JSON is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError(f"cannot read JSON strictly: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackupError(f"JSON root must be an object: {path}")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise BackupError(f"plan field {key} must be a nonempty string")
    return value


def _safe_component(value: str, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise BackupError(f"unsafe {label}: {value!r}")
    return value


def _regular_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise BackupError(f"backup source directory is invalid: {root}")
    files: list[Path] = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise BackupError(f"backup source contains a symlink: {child}")
        if child.is_file():
            files.append(child)
        elif not child.is_dir():
            raise BackupError(f"backup source contains a special file: {child}")
    return files


def _manifest_for_task(manifest: Any, override: object) -> Any:
    if override is None:
        return manifest
    if not isinstance(override, dict) or set(override) != {"microbatch_size"}:
        raise BackupError("task override is invalid")
    size = override["microbatch_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BackupError("task microbatch override is invalid")
    training = dict(manifest.training)
    training["microbatch_size"] = size
    return dataclasses.replace(manifest, training=training)


def _load_training_context(plan: Mapping[str, Any]) -> tuple[Any, Any]:
    canonical = Path(_required_string(plan, "canonical_manifest"))
    runner_kind = plan.get("runner_kind", "baseline")
    if runner_kind == "baseline":
        return load_baseline_manifest(canonical), build_baseline_invocation
    if runner_kind not in {"event_extension", "event_convergence"}:
        raise BackupError(f"unsupported runner_kind: {runner_kind!r}")
    manifest = (
        load_extension_manifest(canonical)
        if runner_kind == "event_extension"
        else load_convergence_manifest(canonical)
    )
    verified = verify_extension_support(
        support_manifest_path=Path(_required_string(plan, "support_manifest")),
        support_root=Path(_required_string(plan, "support_root")),
        node_order=Path(_required_string(plan, "node_order")),
    )
    options = plan.get("extension_options")
    if not isinstance(options, dict):
        raise BackupError("extension_options is invalid")
    extension = ExtensionInputs(
        support_root=verified.support_root,
        node_order=verified.node_order,
        support_manifest_sha256=verified.manifest_sha256,
        igstgnn_data_parallel=bool(options.get("igstgnn_data_parallel", False)),
        max_microbatch_size=int(options.get("max_microbatch_size", 0)),
        cache_dir=Path(_required_string(plan, "cache_dir")),
        graph_dir=Path(_required_string(plan, "graph_dir")),
        python_executable=_required_string(plan, "python_executable"),
    )
    if runner_kind == "event_extension":
        return manifest, lambda task, current, run_inputs, probe: (
            build_extension_invocation(task, current, extension, probe)
        )
    return manifest, lambda task, current, run_inputs, probe: (
        build_convergence_invocation(
            task,
            current,
            ConvergenceInputs(protocol=run_inputs, extension=extension),
            probe,
        )
    )


def discover_completed_tasks(
    plan_path: Path,
    *,
    known_state_sha256: Mapping[str, str] | None = None,
) -> list[CompletedTaskSnapshot]:
    """Validate and snapshot every completed task named by a NAS plan."""
    plan = _read_json_object(plan_path)
    matrix_id = _safe_component(_required_string(plan, "matrix_id"), "matrix_id")
    host_id = _safe_component(_required_string(plan, "host_id"), "host_id")
    manifest, invocation_builder = _load_training_context(plan)
    canonical_by_id = {task.task_id: task for task in manifest.tasks}
    overrides = plan.get("task_overrides") or {}
    if not isinstance(overrides, dict):
        raise BackupError("task_overrides must be an object")
    sources = plan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise BackupError("plan sources must be a nonempty list")

    known_state_sha256 = known_state_sha256 or {}
    snapshots: list[CompletedTaskSnapshot] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise BackupError("plan source entry must be an object")
        source_id = _safe_component(
            _required_string(source, "source_id"), "source_id"
        )
        results_root = Path(_required_string(source, "results_root")).resolve()
        code_sha = _required_string(source, "code_sha").lower()
        task_ids = source.get("task_ids")
        if not isinstance(task_ids, list) or any(
            not isinstance(item, str) for item in task_ids
        ):
            raise BackupError(f"source {source_id} task_ids are invalid")
        run_inputs = runner.ProtocolV3Inputs(
            results_root=results_root,
            cache_dir=Path(_required_string(plan, "cache_dir")),
            graph_dir=Path(_required_string(plan, "graph_dir")),
            cache_bundle_id=_required_string(plan, "cache_bundle_id"),
            cache_bundle_digest=_required_string(plan, "cache_bundle_digest").lower(),
            code_sha=code_sha,
            python_executable=_required_string(plan, "python_executable"),
        )
        for task_id in task_ids:
            if task_id in seen:
                raise BackupError(f"duplicate task in plan: {task_id}")
            seen.add(task_id)
            task = canonical_by_id.get(task_id)
            if task is None:
                raise BackupError(f"unknown canonical task: {task_id}")
            state_path = results_root / "state/tasks" / f"{task_id}.json"
            if not state_path.exists():
                continue
            state = _read_json_object(state_path)
            if state.get("state") != "completed":
                continue
            state_sha256 = _sha256_file(state_path)
            if known_state_sha256.get(task_id) == state_sha256:
                continue
            if state.get("task_id") != task_id or state.get("returncode") != 0:
                raise BackupError(f"completed task state is invalid: {task_id}")

            task_manifest = _manifest_for_task(manifest, overrides.get(task_id))
            probe = results_root / ".staging" / task_id / "__INCREMENTAL_BACKUP__"
            invocation = invocation_builder(
                task,
                task_manifest,
                run_inputs,
                probe,
            )
            config_hash = runner.derive_config_hash(
                task,
                task_manifest,
                run_inputs,
                invocation_config=invocation.config,
            )
            if state.get("config_hash") != config_hash:
                raise BackupError(f"completed task config_hash mismatch: {task_id}")
            result_dir = results_root / task.output_dir
            try:
                final_dir = Path(str(state.get("final_dir"))).resolve(strict=True)
                expected_dir = result_dir.resolve(strict=True)
            except OSError as exc:
                raise BackupError(f"completed task result is missing: {task_id}") from exc
            if final_dir != expected_dir:
                raise BackupError(f"completed task final_dir mismatch: {task_id}")
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
                raise BackupError(f"completed task validation failed: {task_id}: {exc}") from exc

            local_files = _regular_files(result_dir) + [state_path]
            log_dir = results_root / "logs" / task_id
            local_files.extend(_regular_files(log_dir))
            records: list[FileRecord] = []
            for local_path in sorted(set(local_files), key=lambda item: item.as_posix()):
                try:
                    relative = local_path.resolve(strict=True).relative_to(results_root)
                except (OSError, ValueError) as exc:
                    raise BackupError(
                        f"completed task file escapes source root: {local_path}"
                    ) from exc
                remote_relative = (
                    PurePosixPath("sources")
                    / source_id
                    / "results"
                    / PurePosixPath(relative.as_posix())
                ).as_posix()
                size = local_path.stat().st_size
                records.append(
                    FileRecord(
                        source_path=local_path,
                        remote_relative=remote_relative,
                        size_bytes=size,
                        sha256=_sha256_file(local_path),
                    )
                )
            snapshots.append(
                CompletedTaskSnapshot(
                    matrix_id=matrix_id,
                    host_id=host_id,
                    source_id=source_id,
                    task_id=task_id,
                    config_hash=config_hash,
                    state_sha256=state_sha256,
                    files=tuple(records),
                )
            )
    return snapshots


class RemoteStore:
    def __init__(self, *, target: str, key: Path, root: str) -> None:
        if not target or any(char.isspace() for char in target):
            raise BackupError("NAS target is invalid")
        if key.is_symlink() or not key.is_file():
            raise BackupError(f"SSH key is missing or a symlink: {key}")
        remote_root = PurePosixPath(root)
        if not remote_root.is_absolute() or ".." in remote_root.parts:
            raise BackupError(f"remote backup root is unsafe: {root}")
        self.target = target
        self.key = key
        self.root = remote_root
        self.ssh_options = [
            "-n",
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=6",
        ]

    def _script(self, source: str) -> str:
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        completed = subprocess.run(
            ["ssh", *self.ssh_options, self.target, f"echo {encoded} | base64 -d | bash -s"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise BackupError(
                "remote command failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        return completed.stdout

    def available_bytes(self) -> int:
        root = shlex.quote(self.root.as_posix())
        output = self._script(
            f"set -eu\nmkdir -p {root}\ndf -Pk {root} | tail -1 | awk '{{print $4}}'\n"
        ).strip()
        if not output.isdigit():
            raise BackupError(f"invalid remote capacity response: {output!r}")
        return int(output) * 1024

    def file_matches(self, relative: str, sha256: str) -> bool:
        path = shlex.quote((self.root / PurePosixPath(relative)).as_posix())
        output = self._script(
            "set -eu\n"
            f"if [ -f {path} ] && [ \"$(sha256sum {path} | awk '{{print $1}}')\" = {shlex.quote(sha256)} ]; then\n"
            "  echo MATCH\nelse\n  echo MISSING\nfi\n"
        ).strip()
        if output not in {"MATCH", "MISSING"}:
            raise BackupError(f"invalid remote file response: {output!r}")
        return output == "MATCH"

    def upload(self, record: FileRecord) -> None:
        destination = self.root / PurePosixPath(record.remote_relative)
        parent = shlex.quote(destination.parent.as_posix())
        final = shlex.quote(destination.as_posix())
        part_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        part = shlex.quote(part_path.as_posix())
        self._script(f"set -eu\nmkdir -p {parent}\nrm -f {part}\n")
        completed = subprocess.run(
            [
                "scp",
                "-O",
                *self.ssh_options[1:],
                str(record.source_path),
                f"{self.target}:{part_path.as_posix()}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise BackupError(f"scp failed for {record.remote_relative}: {completed.stderr.strip()}")
        if (
            record.source_path.stat().st_size != record.size_bytes
            or _sha256_file(record.source_path) != record.sha256
        ):
            raise BackupError(f"local completed artifact changed: {record.source_path}")
        self._script(
            "set -eu\n"
            f"test \"$(stat -c %s {part})\" = {record.size_bytes}\n"
            f"test \"$(sha256sum {part} | awk '{{print $1}}')\" = {shlex.quote(record.sha256)}\n"
            f"mv -f {part} {final}\nsync\n"
        )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _backup_snapshot(
    snapshot: CompletedTaskSnapshot,
    *,
    store: RemoteStore,
    control_root: Path,
    reserve_bytes: int,
) -> dict[str, Any]:
    manifest = snapshot.manifest()
    marker_relative = (
        PurePosixPath("_completed_tasks")
        / snapshot.source_id
        / f"{snapshot.task_id}.json"
    ).as_posix()
    manifest_path = (
        control_root
        / "incremental_manifests"
        / snapshot.source_id
        / f"{snapshot.task_id}.json"
    )
    manifest["verified_at"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    marker_record = FileRecord(
        source_path=manifest_path,
        remote_relative=marker_relative,
        size_bytes=manifest_path.stat().st_size,
        sha256=_sha256_file(manifest_path),
    )
    if store.file_matches(marker_relative, marker_record.sha256):
        return {**manifest, "marker_sha256": marker_record.sha256, "reused": True}

    missing = [
        item for item in snapshot.files if not store.file_matches(item.remote_relative, item.sha256)
    ]
    required = sum(item.size_bytes for item in missing) + reserve_bytes
    available = store.available_bytes()
    if available < required:
        raise BackupError(
            f"remote capacity gate failed for {snapshot.task_id}: "
            f"available={available} required={required}"
        )
    for item in missing:
        store.upload(item)
    for item in snapshot.files:
        if not store.file_matches(item.remote_relative, item.sha256):
            raise BackupError(f"remote verification failed: {item.remote_relative}")
    store.upload(marker_record)
    if not store.file_matches(marker_relative, marker_record.sha256):
        raise BackupError(f"remote task marker verification failed: {snapshot.task_id}")
    return {**manifest, "marker_sha256": marker_record.sha256, "reused": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--nas", required=True)
    parser.add_argument("--nas-base", required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.poll_seconds < 1 or args.reserve_bytes < 0:
        raise SystemExit("poll-seconds and reserve-bytes are invalid")
    plan = _read_json_object(args.plan)
    matrix_id = _safe_component(_required_string(plan, "matrix_id"), "matrix_id")
    host_id = _safe_component(_required_string(plan, "host_id"), "host_id")
    expected = plan.get("expected_task_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise SystemExit("expected_task_count is invalid")
    remote_root = (
        PurePosixPath(args.nas_base)
        / ".incremental"
        / matrix_id
        / host_id
    ).as_posix()
    store = None if args.dry_run else RemoteStore(
        target=args.nas, key=args.ssh_key, root=remote_root
    )
    state_path = args.control_root / "incremental_backup_state.json"
    log_path = args.control_root / "incremental_backup.log"
    backed: dict[str, Any] = {}
    if state_path.exists():
        old_state = _read_json_object(state_path)
        old_backed = old_state.get("backed_up")
        if isinstance(old_backed, dict):
            backed = dict(old_backed)

    def log(message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    while True:
        try:
            known_states = {
                task_id: item["state_sha256"]
                for task_id, item in backed.items()
                if isinstance(item, dict)
                and isinstance(item.get("state_sha256"), str)
            }
            snapshots = discover_completed_tasks(
                args.plan, known_state_sha256=known_states
            )
            for snapshot in snapshots:
                prior = backed.get(snapshot.task_id)
                if (
                    isinstance(prior, dict)
                    and prior.get("state_sha256") == snapshot.state_sha256
                ):
                    continue
                if args.dry_run:
                    result = {
                        **snapshot.manifest(),
                        "verified_at": _utc_now(),
                        "dry_run": True,
                    }
                else:
                    assert store is not None
                    log(f"BACKUP_START task={snapshot.task_id}")
                    result = _backup_snapshot(
                        snapshot,
                        store=store,
                        control_root=args.control_root,
                        reserve_bytes=args.reserve_bytes,
                    )
                    log(
                        f"BACKUP_OK task={snapshot.task_id} "
                        f"files={len(snapshot.files)} bytes="
                        f"{sum(item.size_bytes for item in snapshot.files)}"
                    )
                backed[snapshot.task_id] = result
                _atomic_write_json(
                    state_path,
                    {
                        "schema_version": 1,
                        "status": "running" if len(backed) < expected else "completed",
                        "matrix_id": matrix_id,
                        "host_id": host_id,
                        "remote_root": remote_root,
                        "expected_tasks": expected,
                        "backed_up_tasks": len(backed),
                        "updated_at": _utc_now(),
                        "backed_up": backed,
                    },
                )
            if len(backed) >= expected:
                log(f"BACKUP_ALL_DONE tasks={len(backed)}/{expected}")
                return 0
            if args.once:
                log(f"BACKUP_ONCE_DONE tasks={len(backed)}/{expected}")
                return 0
        except (BackupError, OSError, subprocess.SubprocessError) as exc:
            log(f"BACKUP_RETRY error={exc}")
            if args.once:
                return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
