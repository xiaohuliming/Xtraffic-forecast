#!/usr/bin/env python3
"""Read-only postflight validation for one protocol-v3 GPU shard."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


def _bootstrap_repo_root() -> Path:
    configured = os.environ.get("XTRAFFIC_REPO_ROOT")
    runner_relative = Path(
        "scripts/remote/windows/protocol_v3_runner.py"
    )
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / runner_relative).is_file():
            raise RuntimeError(
                "XTRAFFIC_REPO_ROOT does not contain "
                f"{runner_relative.as_posix()}: {candidate}"
            )
        return candidate
    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / runner_relative).is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the XTraffic repository from "
        f"{source}; set XTRAFFIC_REPO_ROOT to the repository root"
    )


SCRIPT_REPO_ROOT = _bootstrap_repo_root()
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from scripts.remote.windows import protocol_v3_runner as runner  # noqa: E402


def _load_distributed_module():
    module_name = "scripts.remote.distributed.protocol_v3_distributed"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "scripts.remote.distributed",
            module_name,
        }:
            raise
    sibling = Path(__file__).resolve().with_name(
        "protocol_v3_distributed.py"
    )
    if not sibling.is_file():
        raise RuntimeError(
            "cannot import protocol_v3_distributed and no sibling copy "
            f"exists at {sibling}"
        )
    standalone_name = "_xtraffic_protocol_v3_distributed"
    spec = importlib.util.spec_from_file_location(
        standalone_name,
        sibling,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load distributed shard module from {sibling}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[standalone_name] = module
    spec.loader.exec_module(module)
    return module


distributed = _load_distributed_module()


class PostflightError(RuntimeError):
    """Raised when a completed shard cannot be verified read-only."""


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


def _serialize_run_inputs(
    run_inputs: runner.ProtocolV3Inputs,
) -> dict[str, str]:
    return {
        "results_root": str(run_inputs.results_root.resolve()),
        "cache_dir": str(run_inputs.cache_dir.resolve()),
        "graph_dir": str(run_inputs.graph_dir.resolve()),
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "python_executable": run_inputs.python_executable,
    }


def sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise PostflightError(
            f"result tree contains a symbolic link: {path}"
        )
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PostflightError(f"result artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.stat(path, follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise PostflightError(
            f"result artifact changed while hashing: {path}"
        )
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        return runner._read_strict_json_object(path)
    except runner.ManifestError as exc:
        raise PostflightError(str(exc)) from exc


def _validate_queue(
    queue: Mapping[str, object],
    manifest: runner.Manifest,
    run_inputs: runner.ProtocolV3Inputs,
    minimum_skip_count: int,
) -> tuple[int, str]:
    expected = {
        "state": "completed",
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "manifest_digest": manifest.digest,
        "total_tasks": len(manifest.tasks),
        "completed_tasks": len(manifest.tasks),
        "current_task": None,
    }
    for field, expected_value in expected.items():
        actual = queue.get(field)
        if actual != expected_value:
            raise PostflightError(
                f"queue {field} mismatch: expected "
                f"{expected_value!r}, got {actual!r}"
            )
    actual_skip_count = queue.get("skipped_valid_tasks")
    if (
        isinstance(actual_skip_count, bool)
        or not isinstance(actual_skip_count, int)
        or actual_skip_count < minimum_skip_count
        or actual_skip_count > len(manifest.tasks)
    ):
        raise PostflightError(
            "queue skipped_valid_tasks must be an integer between "
            f"{minimum_skip_count} and {len(manifest.tasks)}, got "
            f"{actual_skip_count!r}"
        )
    source_completed_at = queue.get("updated_at")
    if (
        not isinstance(source_completed_at, str)
        or not source_completed_at.strip()
    ):
        raise PostflightError(
            "queue updated_at must be a nonempty completion timestamp"
        )
    return actual_skip_count, source_completed_at


def _validate_task_directories(
    results_root: Path,
    manifest: runner.Manifest,
) -> None:
    task_root = results_root / "tasks"
    if task_root.is_symlink() or not task_root.is_dir():
        raise PostflightError(
            f"task directory root is missing or a symbolic link: {task_root}"
        )
    expected = {task.task_id for task in manifest.tasks}
    actual = {path.name for path in task_root.iterdir()}
    if actual != expected:
        raise PostflightError(
            "task directory mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for task_id in sorted(expected):
        path = task_root / task_id
        if path.is_symlink() or not path.is_dir():
            raise PostflightError(
                f"task directory is invalid for {task_id}: {path}"
            )


def verify_environment(
    repo_root: Path,
    run_inputs: runner.ProtocolV3Inputs,
) -> dict[str, object]:
    runner._validate_run_inputs(run_inputs)
    runner._verify_git(repo_root.resolve(), run_inputs.code_sha)
    cache_payload = runner._verify_cache_bundle_release(run_inputs)
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()
    cache_manifest_path = (
        run_inputs.cache_dir / "cache_bundle_manifest.json"
    )
    return {
        "git": {
            "head": run_inputs.code_sha,
            "branch": branch,
            "clean": True,
            "repo_root": str(repo_root.resolve()),
        },
        "cache": {
            "status": cache_payload.get("status"),
            "manifest_path": str(cache_manifest_path.resolve()),
            "manifest_sha256": sha256_file(cache_manifest_path),
            "bundle_id": run_inputs.cache_bundle_id,
            "bundle_digest": run_inputs.cache_bundle_digest,
        },
    }


def runtime_provenance(
    run_inputs: runner.ProtocolV3Inputs,
) -> dict[str, object]:
    probe = r"""
import json
import platform
import socket
import sys
import h5py
import numpy
import pandas
import torch
print(json.dumps({
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "h5py": h5py.__version__,
    "pandas": pandas.__version__,
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [run_inputs.python_executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PostflightError(f"runtime provenance probe failed: {exc}") from exc
    if completed.returncode != 0:
        raise PostflightError(
            "runtime provenance probe failed with code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        runtime = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise PostflightError(
            "runtime provenance probe returned invalid JSON"
        ) from exc
    if not isinstance(runtime, dict):
        raise PostflightError(
            "runtime provenance probe did not return an object"
        )

    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PostflightError(f"GPU provenance probe failed: {exc}") from exc
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise PostflightError(
            "GPU provenance probe failed: "
            f"{gpu.stderr.strip() or 'empty nvidia-smi output'}"
        )
    runtime["gpu"] = [
        line.strip() for line in gpu.stdout.splitlines() if line.strip()
    ]
    runtime["requested_python_executable"] = run_inputs.python_executable
    return runtime


def _collect_regular_files(path: Path) -> list[Path]:
    if path.is_symlink():
        raise PostflightError(
            f"export allowlist contains a symbolic link: {path}"
        )
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise PostflightError(
            f"required export path is missing or invalid: {path}"
        )
    files: list[Path] = []
    for child in path.rglob("*"):
        if child.is_symlink():
            raise PostflightError(
                f"export allowlist contains a symbolic link: {child}"
            )
        if child.is_dir():
            continue
        if not child.is_file():
            raise PostflightError(
                f"export allowlist contains a non-file artifact: {child}"
            )
        files.append(child)
    return files


def _hash_export_allowlist(
    results_root: Path,
    manifest: runner.Manifest,
) -> list[dict[str, object]]:
    allowed_files: set[Path] = set()
    allowed_files.update(
        _collect_regular_files(results_root / "queue_status.json")
    )
    for task in manifest.tasks:
        allowed_files.update(
            _collect_regular_files(results_root / task.output_dir)
        )
        allowed_files.update(
            _collect_regular_files(
                results_root
                / "state/tasks"
                / f"{task.task_id}.json"
            )
        )
        log_dir = results_root / "logs" / task.task_id
        if log_dir.exists() or log_dir.is_symlink():
            allowed_files.update(_collect_regular_files(log_dir))

    records: list[dict[str, object]] = []
    for path in sorted(allowed_files, key=lambda item: item.as_posix()):
        relative = path.relative_to(results_root).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise PostflightError("export allowlist contains no files")
    return records


def build_postflight_report(
    *,
    shard_name: str,
    canonical: runner.Manifest,
    manifest: runner.Manifest,
    repo_root: Path,
    run_inputs: runner.ProtocolV3Inputs,
) -> dict[str, object]:
    if shard_name not in distributed.SHARD_SPECS:
        raise PostflightError(f"unknown shard: {shard_name}")
    all_shards = distributed.build_shard_manifests(canonical)
    expected_manifest = all_shards[shard_name]
    distributed.validate_shard_manifests(
        canonical,
        {
            **all_shards,
            shard_name: manifest,
        },
    )
    if manifest.tasks != expected_manifest.tasks:
        raise PostflightError(
            f"manifest does not match fixed {shard_name} assignment"
        )

    spec = distributed.SHARD_SPECS[shard_name]
    results_root = run_inputs.results_root.resolve()
    queue_path = results_root / "queue_status.json"
    queue_before = _load_json(queue_path)
    queue_skip_count, source_completed_at = _validate_queue(
        queue_before,
        manifest,
        run_inputs,
        len(spec.expected_skip_numbers),
    )
    _validate_task_directories(results_root, manifest)
    environment = verify_environment(repo_root, run_inputs)
    runtime = runtime_provenance(run_inputs)

    inherited_skip_ids = {
        task.task_id
        for task in manifest.tasks
        if distributed.task_number(task) in spec.expected_skip_numbers
    }
    state_counts: Counter[str] = Counter()
    task_reports: list[dict[str, object]] = []
    for task in manifest.tasks:
        probe_dir = (
            results_root / ".staging" / task.task_id / "__POSTFLIGHT__"
        )
        invocation = runner.build_production_invocation(
            task,
            manifest,
            run_inputs,
            probe_dir,
        )
        config_hash = runner.derive_config_hash(
            task,
            manifest,
            run_inputs,
            invocation_config=invocation.config,
        )
        result_dir = results_root / task.output_dir
        runner.validate_result(
            result_dir,
            task,
            manifest,
            run_inputs,
            expected_config_hash=config_hash,
            required_artifacts=invocation.required_artifacts,
            required_metrics=invocation.required_metrics,
        )
        state_path = (
            results_root / "state/tasks" / f"{task.task_id}.json"
        )
        task_state = _load_json(state_path)
        if task_state.get("task_id") != task.task_id:
            raise PostflightError(
                f"task state identity mismatch for {task.task_id}"
            )
        actual_state = task_state.get("state")
        if actual_state not in {"completed", "skipped_valid"}:
            raise PostflightError(
                f"task state mismatch for {task.task_id}: expected "
                f"'completed' or 'skipped_valid', got {actual_state!r}"
            )
        if (
            task.task_id in inherited_skip_ids
            and actual_state != "skipped_valid"
        ):
            raise PostflightError(
                "inherited skipped task was not preserved as "
                f"skipped_valid: {task.task_id}"
            )
        if task_state.get("config_hash") != config_hash:
            raise PostflightError(
                f"task state config_hash mismatch for {task.task_id}"
            )
        state_counts[actual_state] += 1

        task_reports.append(
            {
                "task_id": task.task_id,
                "task_number": distributed.task_number(task),
                "state": actual_state,
                "config_hash": config_hash,
                "invocation_config": dict(invocation.config),
            }
        )

    if sum(state_counts.values()) != len(manifest.tasks):
        raise PostflightError(
            "task state counts do not cover the complete shard: "
            f"{dict(state_counts)}"
        )
    if state_counts["skipped_valid"] != queue_skip_count:
        raise PostflightError(
            "queue skipped_valid_tasks differs from task states: "
            f"queue={queue_skip_count}, "
            f"states={state_counts['skipped_valid']}"
        )
    file_records = _hash_export_allowlist(results_root, manifest)
    for task, task_report in zip(manifest.tasks, task_reports, strict=True):
        prefix = task.output_dir.as_posix() + "/"
        task_files = [
            record
            for record in file_records
            if str(record["path"]).startswith(prefix)
        ]
        if not task_files:
            raise PostflightError(
                f"task result directory is empty: "
                f"{results_root / task.output_dir}"
            )
        task_report.update(
            {
                "file_count": len(task_files),
                "total_bytes": sum(
                    int(record["size_bytes"]) for record in task_files
                ),
                "files_digest": _sha256_json(task_files),
            }
        )
    queue_after = _load_json(queue_path)
    if queue_after != queue_before:
        raise PostflightError(
            "queue status changed during postflight validation"
        )

    canonical_payload = _load_json(canonical.source_path)
    actual_skip_ids = sorted(
        report["task_id"]
        for report in task_reports
        if report["state"] == "skipped_valid"
    )
    return {
        "status": "verified",
        "verified_at": source_completed_at,
        "source_completed_at": source_completed_at,
        "shard": shard_name,
        "results_root": str(results_root),
        "validated_tasks": len(task_reports),
        "expected_skipped_valid_tasks": len(spec.expected_skip_numbers),
        "minimum_skipped_valid_tasks": len(spec.expected_skip_numbers),
        "expected_skipped_task_ids": sorted(inherited_skip_ids),
        "actual_skipped_valid_tasks": queue_skip_count,
        "actual_skipped_task_ids": actual_skip_ids,
        "task_state_counts": {
            "completed": state_counts["completed"],
            "skipped_valid": state_counts["skipped_valid"],
        },
        "run_inputs": _serialize_run_inputs(run_inputs),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "code_sha": run_inputs.code_sha,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "manifest_raw_sha256": canonical.digest,
        "manifest_canonical_json_sha256": _sha256_json(canonical_payload),
        "queue_status": queue_before,
        "environment": environment,
        "runtime": runtime,
        "total_bytes": sum(
            int(record["size_bytes"]) for record in file_records
        ),
        "file_count": len(file_records),
        "files_digest": _sha256_json(file_records),
        "export_paths": [
            str(record["path"]) for record in file_records
        ],
        "tasks": task_reports,
        "files": file_records,
    }


def _resolve_from_repo(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def _report_inside_results(report: Path, results_root: Path) -> bool:
    try:
        report.resolve().relative_to(results_root.resolve())
    except ValueError:
        return False
    return True


def write_report(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=tuple(distributed.SHARD_SPECS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            SCRIPT_REPO_ROOT
            / "scripts/remote/windows/protocol_v3_manifest.json"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=SCRIPT_REPO_ROOT,
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        canonical = runner.load_manifest(
            args.manifest,
            require_production_matrix=True,
        )
        manifest = distributed.build_shard_manifests(canonical)[args.shard]
        spec = distributed.SHARD_SPECS[args.shard]
        repo_root = args.repo_root.resolve()
        results_value = args.results_root or spec.default_results_root
        cache_value = args.cache_dir or manifest.default_cache_dir
        graph_value = args.graph_dir or manifest.default_graph_dir
        run_inputs = runner.ProtocolV3Inputs(
            results_root=_resolve_from_repo(repo_root, results_value),
            cache_dir=_resolve_from_repo(repo_root, cache_value),
            graph_dir=_resolve_from_repo(repo_root, graph_value),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
        if _report_inside_results(args.report, run_inputs.results_root):
            raise PostflightError(
                "postflight report must be outside the results root"
            )
        payload = build_postflight_report(
            shard_name=args.shard,
            canonical=canonical,
            manifest=manifest,
            repo_root=repo_root,
            run_inputs=run_inputs,
        )
        write_report(args.report, payload)
    except (
        OSError,
        PostflightError,
        distributed.ShardValidationError,
        runner.ManifestError,
        runner.ResultValidationError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"POSTFLIGHT_FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        "POSTFLIGHT_OK "
        f"shard={args.shard} tasks={payload['validated_tasks']} "
        f"files={payload['file_count']} report={args.report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
