#!/usr/bin/env python3
# ruff: noqa: E402
"""Fail-closed, resumable Windows runner for the FDN benchmark v1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.fdn_benchmark_v1 import (
    ARTIFACT_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    VARIANTS,
    load_benchmark_data,
    sha256_file,
    strict_json_write,
)


DEFAULT_MANIFEST = ROOT / "config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json"
REQUIRED_COMMON = {
    "summary.json",
    "config.json",
    "ckpt_best.pt",
    "test_predictions.npz",
    "training_diagnostics.jsonl",
    "train.log",
}


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    seed: int
    variant: str


@dataclass(frozen=True)
class Manifest:
    path: Path
    raw: Mapping[str, object]
    matrix_id: str
    datasets: Mapping[str, Mapping[str, object]]
    training: Mapping[str, object]
    tasks: tuple[Task, ...]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path | str) -> Manifest:
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("matrix_schema_version") != 1:
        raise RunnerError("matrix schema version must be 1")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise RunnerError("benchmark protocol version mismatch")
    if raw.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise RunnerError("artifact schema version mismatch")
    datasets = raw.get("datasets")
    training = raw.get("training")
    tasks_raw = raw.get("tasks")
    if not isinstance(datasets, dict) or set(datasets) != {"pems04", "pems08"}:
        raise RunnerError("phase-1 datasets must be exactly pems04 and pems08")
    if not isinstance(training, dict) or not isinstance(tasks_raw, list):
        raise RunnerError("manifest training or tasks section is invalid")
    expected_training = {
        "epochs": 100,
        "batch_size": 48,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip": 5.0,
        "nhid": 32,
        "history_steps": 12,
        "horizon_steps": 12,
        "scheduler": "CosineAnnealingLR_per_optimizer_step",
        "eta_min": 0.00001,
        "checkpoint_selection": "global_validation_mae_all_valid_observations",
        "early_stopping": False,
        "device": "cuda:0",
    }
    if training != expected_training:
        raise RunnerError("phase-1 training contract changed")
    tasks: list[Task] = []
    for item in tasks_raw:
        if not isinstance(item, dict):
            raise RunnerError("task entry must be an object")
        task = Task(
            task_id=str(item.get("task_id", "")),
            dataset=str(item.get("dataset", "")),
            seed=int(item.get("seed", -1)),
            variant=str(item.get("variant", "")),
        )
        if not task.task_id.startswith("fdnb") or len(task.task_id) > 100:
            raise RunnerError(f"invalid task id: {task.task_id!r}")
        if task.dataset not in datasets or task.seed not in {42, 1, 2}:
            raise RunnerError(f"invalid task scope: {task}")
        if task.variant not in VARIANTS:
            raise RunnerError(f"invalid task variant: {task.variant}")
        tasks.append(task)
    expected_cells = {
        (dataset, seed, variant)
        for dataset in ("pems04", "pems08")
        for seed in (42, 1, 2)
        for variant in VARIANTS
    }
    actual_cells = {(task.dataset, task.seed, task.variant) for task in tasks}
    if len(tasks) != 24 or actual_cells != expected_cells:
        raise RunnerError("phase-1 manifest must cover the exact 24-cell matrix")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RunnerError("task IDs must be unique")
    return Manifest(
        path=manifest_path,
        raw=raw,
        matrix_id=str(raw.get("matrix_id", "")),
        datasets=datasets,
        training=training,
        tasks=tuple(tasks),
    )


def derive_config_hash(task: Task, manifest: Manifest, code_sha: str) -> str:
    dataset = manifest.datasets[task.dataset]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "matrix_id": manifest.matrix_id,
        "task": task.__dict__,
        "training": manifest.training,
        "dataset": dataset,
        "code_sha": code_sha,
    }
    return _canonical_sha256(payload)


def _nested_metrics_from_arrays(
    prediction: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    if prediction.shape != actual.shape or mask.shape != actual.shape:
        raise RunnerError("prediction artifact shapes differ")
    if prediction.ndim != 3 or prediction.shape[2] != 12:
        raise RunnerError("prediction artifact must have shape (S, N, 12)")
    if prediction.dtype != np.float32 or actual.dtype != np.float32 or mask.dtype != np.bool_:
        raise RunnerError("prediction artifact dtypes are invalid")
    if not np.isfinite(prediction).all() or not np.isfinite(actual[mask]).all():
        raise RunnerError("prediction artifact contains non-finite values")
    error = prediction.astype(np.float64) - actual.astype(np.float64)
    absolute = np.abs(error)

    def scope(selected: np.ndarray) -> dict[str, object]:
        count = int(selected.sum())
        if count <= 0:
            raise RunnerError("prediction artifact has an empty metric scope")
        return {
            "mae": float(absolute[selected].sum(dtype=np.float64) / count),
            "rmse": float(math.sqrt(np.square(error[selected]).sum(dtype=np.float64) / count)),
            "mape_percent": float(
                100.0
                * (absolute[selected] / np.maximum(np.abs(actual[selected].astype(np.float64)), 1e-6)).sum(
                    dtype=np.float64
                )
                / count
            ),
            "valid_count": count,
        }

    result = scope(mask)
    for horizon in (2, 5, 11):
        selected = mask[:, :, horizon]
        scoped = scope(selected)
        result[f"h{horizon + 1}"] = scoped
    return result


def _assert_close(actual: object, expected: object, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RunnerError(f"{label} metric object differs")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, int):
        if actual != expected:
            raise RunnerError(f"{label} count differs: {actual!r} != {expected!r}")
        return
    if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
        raise RunnerError(f"{label} is not finite")
    if not math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-7):
        raise RunnerError(f"{label} differs: {actual!r} != {expected!r}")


def validate_result(
    result_dir: Path,
    task: Task,
    manifest: Manifest,
    *,
    code_sha: str,
    config_hash: str,
    data_dir: Path,
) -> dict[str, object]:
    required = set(REQUIRED_COMMON)
    if task.variant.startswith("fdn_"):
        required.add("decomp_mask.json")
    actual_names = {path.name for path in result_dir.iterdir() if path.is_file()}
    if not required.issubset(actual_names):
        raise RunnerError(f"{task.task_id} lacks required artifacts: {sorted(required - actual_names)}")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "task_id": task.task_id,
        "dataset": task.dataset,
        "variant": task.variant,
        "seed": task.seed,
        "code_sha": code_sha,
        "config_hash": config_hash,
    }
    for key, expected in identity.items():
        if summary.get(key) != expected:
            raise RunnerError(f"{task.task_id} summary {key} mismatch")
    dataset = manifest.datasets[task.dataset]
    if summary.get("data_sha256") != dataset["data_sha256"]:
        raise RunnerError(f"{task.task_id} data SHA mismatch")
    if summary.get("adjacency_sha256") != dataset["adjacency_sha256"]:
        raise RunnerError(f"{task.task_id} adjacency SHA mismatch")
    with np.load(result_dir / "test_predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"prediction", "actual", "mask", "forecast_origin"}:
            raise RunnerError(f"{task.task_id} prediction schema mismatch")
        prediction = archive["prediction"]
        actual = archive["actual"]
        mask = archive["mask"]
        origins = archive["forecast_origin"]
    source = load_benchmark_data(
        task.dataset,
        data_dir,
        expected_data_sha256=str(dataset["data_sha256"]),
        expected_adjacency_sha256=str(dataset["adjacency_sha256"]),
    )
    expected_origins = np.asarray(source.anchors_by_split["test"], dtype=np.int64)
    if origins.dtype != np.int64 or not np.array_equal(origins, expected_origins):
        raise RunnerError(f"{task.task_id} forecast origins do not match the formal test split")
    recomputed = _nested_metrics_from_arrays(prediction, actual, mask)
    _assert_close(summary.get("test_metrics"), recomputed, f"{task.task_id}.test_metrics")
    diagnostics = (result_dir / "training_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    if len(diagnostics) != int(manifest.training["epochs"]):
        raise RunnerError(f"{task.task_id} does not contain 100 epoch records")
    final_record = json.loads(diagnostics[-1])
    if final_record.get("epoch") != int(manifest.training["epochs"]):
        raise RunnerError(f"{task.task_id} final epoch record mismatch")
    files = {
        name: {
            "size_bytes": int((result_dir / name).stat().st_size),
            "sha256": sha256_file(result_dir / name),
        }
        for name in sorted(required)
    }
    return {"summary": summary, "files": files, "files_digest": _canonical_sha256(files)}


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RunnerError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _write_queue(path: Path, payload: Mapping[str, object]) -> None:
    strict_json_write(path, payload)


def run_queue(
    manifest: Manifest,
    *,
    repo_root: Path,
    data_dir: Path,
    results_root: Path,
    code_sha: str,
    python_executable: str,
) -> int:
    head = _git_output(repo_root, "rev-parse", "HEAD").lower()
    if head != code_sha:
        raise RunnerError(f"checkout SHA mismatch: {head} != {code_sha}")
    if _git_output(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise RunnerError("formal benchmark checkout has tracked modifications")
    for dataset, identity in manifest.datasets.items():
        load_benchmark_data(
            dataset,
            data_dir,
            expected_data_sha256=str(identity["data_sha256"]),
            expected_adjacency_sha256=str(identity["adjacency_sha256"]),
        )
    results_root.mkdir(parents=True, exist_ok=True)
    tasks_root = results_root / "tasks"
    state_root = results_root / "state/tasks"
    staging_root = results_root / ".staging"
    for path in (tasks_root, state_root, staging_root):
        path.mkdir(parents=True, exist_ok=True)
    queue_path = results_root / "queue_status.json"
    completed = 0
    skipped = 0
    started_at = time.time()
    for position, task in enumerate(manifest.tasks, start=1):
        config_hash = derive_config_hash(task, manifest, code_sha)
        result_dir = tasks_root / task.task_id
        if result_dir.is_dir():
            validation = validate_result(
                result_dir,
                task,
                manifest,
                code_sha=code_sha,
                config_hash=config_hash,
                data_dir=data_dir,
            )
            completed += 1
            skipped += 1
            strict_json_write(
                state_root / f"{task.task_id}.json",
                {
                    "state": "skipped_valid",
                    "task_id": task.task_id,
                    "config_hash": config_hash,
                    "validation": validation,
                },
            )
            continue
        _write_queue(
            queue_path,
            {
                "state": "running",
                "matrix_id": manifest.matrix_id,
                "total_tasks": len(manifest.tasks),
                "completed_tasks": completed,
                "skipped_valid_tasks": skipped,
                "failed_count": 0,
                "current_task": task.task_id,
                "current_position": position,
                "started_at_unix": started_at,
            },
        )
        attempt = staging_root / task.task_id / "attempt_0001"
        if attempt.exists():
            raise RunnerError(f"stale formal attempt exists for {task.task_id}: {attempt}")
        work = attempt / "work"
        attempt.mkdir(parents=True, exist_ok=False)
        dataset = manifest.datasets[task.dataset]
        training = manifest.training
        command = [
            python_executable,
            "-X",
            "utf8",
            "scripts/fdn_benchmark_v1.py",
            "--dataset",
            task.dataset,
            "--variant",
            task.variant,
            "--seed",
            str(task.seed),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(work),
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--nhid",
            str(training["nhid"]),
            "--device",
            str(training["device"]),
            "--task-id",
            task.task_id,
            "--code-sha",
            code_sha,
            "--config-hash",
            config_hash,
            "--expected-data-sha256",
            str(dataset["data_sha256"]),
            "--expected-adjacency-sha256",
            str(dataset["adjacency_sha256"]),
        ]
        log_path = attempt / "train.log"
        with log_path.open("xb") as log:
            process = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            strict_json_write(
                state_root / f"{task.task_id}.json",
                {
                    "state": "failed",
                    "task_id": task.task_id,
                    "config_hash": config_hash,
                    "returncode": int(process.returncode),
                    "attempt_dir": str(attempt),
                },
            )
            _write_queue(
                queue_path,
                {
                    "state": "failed",
                    "matrix_id": manifest.matrix_id,
                    "total_tasks": len(manifest.tasks),
                    "completed_tasks": completed,
                    "skipped_valid_tasks": skipped,
                    "failed_count": 1,
                    "current_task": task.task_id,
                    "returncode": int(process.returncode),
                },
            )
            return 3
        os.replace(log_path, work / "train.log")
        validation = validate_result(
            work,
            task,
            manifest,
            code_sha=code_sha,
            config_hash=config_hash,
            data_dir=data_dir,
        )
        os.replace(work, result_dir)
        completed += 1
        strict_json_write(
            state_root / f"{task.task_id}.json",
            {
                "state": "completed",
                "task_id": task.task_id,
                "config_hash": config_hash,
                "attempt_dir": str(attempt),
                "validation": validation,
            },
        )
    _write_queue(
        queue_path,
        {
            "state": "completed",
            "matrix_id": manifest.matrix_id,
            "total_tasks": len(manifest.tasks),
            "completed_tasks": completed,
            "skipped_valid_tasks": skipped,
            "failed_count": 0,
            "current_task": None,
            "started_at_unix": started_at,
            "completed_at_unix": time.time(),
            "manifest_sha256": sha256_file(manifest.path),
            "code_sha": code_sha,
        },
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        code_sha = args.code_sha.lower()
        if len(code_sha) != 40 or any(char not in "0123456789abcdef" for char in code_sha):
            raise RunnerError("code SHA must be a full lowercase Git SHA")
        return run_queue(
            load_manifest(args.manifest),
            repo_root=args.repo_root.resolve(),
            data_dir=args.data_dir.resolve(),
            results_root=args.results_root.resolve(),
            code_sha=code_sha,
            python_executable=args.python_executable,
        )
    except (OSError, ValueError, RunnerError, json.JSONDecodeError) as exc:
        print(f"RUNNER_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
