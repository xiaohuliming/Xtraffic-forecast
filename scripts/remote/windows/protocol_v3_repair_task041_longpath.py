#!/usr/bin/env python3
"""Repair the audited task041 post-inference Windows path-length failure.

This is an incident-specific control-plane tool. It never trains or evaluates a
model. It accepts exactly the evidence captured from task041 attempt_0001,
reconstructs the two files that the trainer could not materialize, validates
the complete result with the protocol-v3 runner, and atomically promotes it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

import numpy as np
import torch


sys.dont_write_bytecode = True


TASK_ID = (
    "041_fullwindow_fourier_dual_net_learnable_k3_"
    "contracosta_seed42"
)
CODE_SHA = "3a4952837aaf6fbda1403d50b610b1bf40c6098a"
CONFIG_HASH = (
    "786992454b5d960603e73d16254d38684d7cb81328b587675648c451b8a074d7"
)
RUNNER_SHA256 = (
    "870c17aeae8cda5003f82eee159eac0226956087fe768d2a5d6a9fb43df9a123"
)
CACHE_BUNDLE_ID = "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4"
CACHE_BUNDLE_DIGEST = (
    "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"
)

QUEUE_SHA256 = (
    "1c30c4f9c82a04ce50d91172cd2a80abd3ace6042ee668afa82f276c8a26684e"
)
FAILED_STATE_SHA256 = (
    "d5db1dfbd0b879ff22c59f130fe49ac500d397e7f7d4b52b2d685519c8ea1f09"
)
OUTER_LOG_SHA256 = (
    "aed2a3a7ba9994b490d8ebc4946e1b63533255da86caab66a81822a51eb5ff3b"
)

EXISTING_ARTIFACTS = {
    "ckpt_best.pt": (
        2_618_187,
        "1b94f341ce4f8894c5c3bd688b8d26fb880569e4910351d977498fb57f97ac6c",
    ),
    "test_predictions.npz": (
        481_714_248,
        "375a57ba553857e2d7b1532a98abbaca8ec6fcbd7a2dfcbc4491ed5f91b27620",
    ),
    "train.log": (
        7_981,
        "c66121be8b5ee2cb8d66b0f4b0b62de04b25d4bdcbc036a29e68fa91cf53cea8",
    ),
}
GENERATED_ARTIFACTS = {
    "summary.json": (
        7_942,
        "115c0cfba5d6a256a27011a6169797ac48606cb90a914a942919854e399e27ac",
    ),
    "decomp_mask.json": (
        223,
        "55ae377b62287d5098a895c5608cb8050d79520df2a5936f0754c0a18bef5d5a",
    ),
}

EXPECTED_PRODUCER_MAE = 11.21213436126709
EXPECTED_AUDIT_MAE = 11.212104746452523
EXPECTED_VALID_COUNT = 103_270_922
EXPECTED_BEST_EPOCH = 28
EXPECTED_BEST_VAL = 11.049783944550832
EXPECTED_CUDA_MASK = [
    0.8248609304428101,
    0.7294601798057556,
    0.8947920799255371,
    0.46656280755996704,
    0.32573673129081726,
    0.18384388089179993,
    0.19463671743869781,
]
EXPECTED_TASK_NUMBERS = frozenset(
    {
        *range(1, 5),
        *range(7, 11),
        *range(13, 17),
        *range(19, 31),
        *range(40, 43),
    }
)
EXPECTED_NPZ_KEYS = frozenset(
    {
        "region_code",
        "sample_start",
        "region_node_idx",
        "pred_raw_flow",
        "actual_future_flow",
        "y_mask_flow",
        "affected_mask",
    }
)


class RepairError(RuntimeError):
    """Raised when any task041 repair precondition is not exact."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(
    path: Path,
    *,
    size: int | None,
    sha256: str,
    label: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise RepairError(f"{label} is missing, not a file, or a link: {path}")
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise RepairError(
            f"{label} size mismatch: expected {size}, got {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != sha256:
        raise RepairError(
            f"{label} SHA-256 mismatch: expected {sha256}, got {actual_sha256}"
        )


def _require_exact_mapping(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RepairError(
                f"{label} {key} mismatch: expected {value!r}, "
                f"got {payload.get(key)!r}"
            )


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_strict_json(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_text(encoding="utf-8")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RepairError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise RepairError(f"{path} contains non-finite constant {value}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError(f"{path} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RepairError(f"{path} must contain a JSON object")
    return payload, raw


def _load_runner(path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise RepairError(f"runner is missing, not a file, or a link: {path}")
    spec = importlib.util.spec_from_file_location(
        "xtraffic_task041_repair_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RepairError(f"cannot load runner module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task_number(task_id: str) -> int:
    match = re.match(r"^(\d{3})_", task_id)
    if match is None:
        raise RepairError(f"task id has no canonical number: {task_id}")
    return int(match.group(1))


def _verify_clean_checkout(repo_root: Path) -> None:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RepairError(f"cannot verify frozen checkout: {exc}") from exc
    if head != CODE_SHA:
        raise RepairError(
            f"frozen checkout SHA mismatch: expected {CODE_SHA}, got {head}"
        )
    if dirty:
        raise RepairError("frozen training checkout is dirty")


def _validate_log(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8")
    epochs = [
        int(value)
        for value in re.findall(r"^==> ep(\d+) train L_main=", text, re.MULTILINE)
    ]
    if epochs != list(range(1, 31)):
        raise RepairError(f"outer log epoch sequence is invalid: {epochs}")
    required = (
        "best val L_main: 11.0498",
        "=== test inference ===",
        "test MAE: all=11.212",
        "dist_net\\run_integrity.py",
        "write_summary_json",
        "FileNotFoundError: [Errno 2]",
        ".summary.json.",
        ".tmp",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RepairError(
            "outer log lacks the exact post-inference writer failure evidence: "
            + ", ".join(missing)
        )


def _validate_npz(
    path: Path,
    *,
    region_data: object,
) -> tuple[float, float, int]:
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != EXPECTED_NPZ_KEYS:
            raise RepairError(
                f"prediction keys mismatch: {sorted(archive.files)}"
            )
        region_code = archive["region_code"]
        sample_start = archive["sample_start"]
        region_node_idx = archive["region_node_idx"]
        prediction = archive["pred_raw_flow"]
        actual = archive["actual_future_flow"]
        valid = archive["y_mask_flow"]
        affected = archive["affected_mask"]

        expected_shape = (21_013, 12, 496)
        if prediction.shape != expected_shape or actual.shape != expected_shape:
            raise RepairError("prediction or target shape mismatch")
        if valid.shape != expected_shape or affected.shape != (21_013, 496):
            raise RepairError("prediction mask shape mismatch")
        if prediction.dtype != np.float32 or actual.dtype != np.float32:
            raise RepairError("prediction and target must be float32")
        if valid.dtype != np.bool_ or affected.dtype != np.bool_:
            raise RepairError("prediction masks must be bool")
        if region_code.shape != (21_013,) or region_code.dtype != np.int64:
            raise RepairError("region_code schema mismatch")
        if not np.all(region_code == 1):
            raise RepairError("region_code must identify ContraCosta")
        if sample_start.shape != (21_013,) or sample_start.dtype != np.int64:
            raise RepairError("sample_start schema mismatch")
        if (
            np.unique(sample_start).size != sample_start.size
            or np.any(np.diff(sample_start) <= 0)
        ):
            raise RepairError("sample_start must be strictly increasing and unique")
        if region_node_idx.shape != (496,) or region_node_idx.dtype != np.int64:
            raise RepairError("region_node_idx schema mismatch")
        if np.unique(region_node_idx).size != region_node_idx.size:
            raise RepairError("region_node_idx must be unique")
        if affected.any():
            raise RepairError("full-window affected_mask must be all false")
        if not np.isfinite(prediction).all():
            raise RepairError("prediction contains a non-finite value")
        if not np.isfinite(actual[valid]).all():
            raise RepairError("valid target contains a non-finite value")

        expected_starts = region_data.sample_start[
            region_data.split == 2
        ].astype(np.int64, copy=False)
        if not np.array_equal(sample_start, expected_starts):
            raise RepairError("prediction sample_start differs from formal cache")
        if not np.array_equal(
            region_node_idx,
            region_data.region_idx.astype(np.int64, copy=False),
        ):
            raise RepairError("prediction region_node_idx differs from formal cache")

        block_lo = int(sample_start[0]) + 1
        block_hi = int(sample_start[-1]) + 1 + int(region_data.T_p)
        flow_block, mask_block = region_data._slice_traffic(
            block_lo,
            block_hi,
        )
        flow = flow_block[..., 0]
        flow_mask = mask_block[..., 0]
        offsets = sample_start - sample_start[0]
        for start in range(0, sample_start.size, 64):
            stop = min(start + 64, sample_start.size)
            for local, offset in enumerate(offsets[start:stop]):
                expected_actual = flow[
                    int(offset) : int(offset) + int(region_data.T_p)
                ]
                expected_mask = flow_mask[
                    int(offset) : int(offset) + int(region_data.T_p)
                ]
                if not np.array_equal(actual[start + local], expected_actual):
                    raise RepairError(
                        f"target differs from cache at sample {start + local}"
                    )
                if not np.array_equal(valid[start + local], expected_mask):
                    raise RepairError(
                        f"mask differs from cache at sample {start + local}"
                    )

        valid_count = int(valid.sum())
        if valid_count != EXPECTED_VALID_COUNT:
            raise RepairError(
                f"valid_count mismatch: {valid_count} != {EXPECTED_VALID_COUNT}"
            )
        producer = float(np.abs(prediction - actual)[valid].mean())
        if producer != EXPECTED_PRODUCER_MAE:
            raise RepairError(
                f"producer MAE mismatch: {producer!r} "
                f"!= {EXPECTED_PRODUCER_MAE!r}"
            )

        error_sum = 0.0
        audit_count = 0
        for start in range(0, prediction.shape[0], 64):
            stop = min(start + 64, prediction.shape[0])
            selected = valid[start:stop]
            error = np.abs(
                prediction[start:stop][selected].astype(np.float64)
                - actual[start:stop][selected].astype(np.float64)
            )
            error_sum += float(error.sum())
            audit_count += int(error.size)
        audit = error_sum / audit_count
        if not math.isclose(
            audit,
            EXPECTED_AUDIT_MAE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RepairError(
                f"float64 audit MAE mismatch: {audit!r} "
                f"!= {EXPECTED_AUDIT_MAE!r}"
            )
        return producer, audit, valid_count


def _load_checkpoint(path: Path, *, expected_work: Path) -> tuple[float, list[float]]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RepairError(f"cannot load hash-locked checkpoint: {exc}") from exc
    expected_keys = {
        "C_x",
        "N",
        "T_h",
        "T_p",
        "config",
        "epoch",
        "model_state",
        "val_L_main",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_keys:
        raise RepairError("checkpoint top-level schema mismatch")
    if checkpoint["epoch"] != EXPECTED_BEST_EPOCH:
        raise RepairError("checkpoint best epoch mismatch")
    best_val = float(checkpoint["val_L_main"])
    if best_val != EXPECTED_BEST_VAL:
        raise RepairError("checkpoint best validation metric mismatch")
    if (
        checkpoint["C_x"] != 3
        or checkpoint["N"] != 496
        or checkpoint["T_h"] != 12
        or checkpoint["T_p"] != 12
    ):
        raise RepairError("checkpoint dimensions mismatch")

    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise RepairError("checkpoint config must be a mapping")
    expected_config = {
        "region": "ContraCosta",
        "tag": None,
        "decomp_mode": "learnable",
        "K": 3,
        "use_time_emb": False,
        "main_blocks": 4,
        "main_layers": 2,
        "pert_blocks": 4,
        "pert_layers": 2,
        "use_cross_attn": False,
        "cross_attn_dim": 16,
        "cross_attn_heads": 2,
        "use_gated_fusion": False,
        "gate_d_node": 8,
        "gate_hidden": 32,
        "gate_l1_lambda": 0.0,
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "nhid": 32,
        "dropout": 0.3,
        "seed": 42,
        "device": "cuda:0",
        "protocol": "full_window",
        "stride": 1,
        "train_frac": 0.7,
        "val_frac": 0.1,
        "patience": 6,
        "smoke": False,
    }
    _require_exact_mapping(config, expected_config, label="checkpoint config")
    if Path(config.get("out_dir", "")).resolve() != expected_work.resolve():
        raise RepairError("checkpoint output path mismatch")

    state = checkpoint["model_state"]
    if not isinstance(state, Mapping):
        raise RepairError("checkpoint model_state must be a mapping")
    logits = state.get("decomp.bin_logit")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.dtype != torch.float32
        or tuple(logits.shape) != (7,)
        or not torch.isfinite(logits).all()
    ):
        raise RepairError("checkpoint decomposition logits are invalid")
    if not torch.cuda.is_available():
        raise RepairError("task041 repair requires the original CUDA producer")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5080":
        raise RepairError(
            f"unexpected CUDA producer: {torch.cuda.get_device_name(0)!r}"
        )
    mask = torch.sigmoid(logits.to("cuda:0")).detach().cpu().tolist()
    if mask != EXPECTED_CUDA_MASK:
        raise RepairError(f"CUDA decomposition mask mismatch: {mask!r}")
    return best_val, mask


def _require_generated_bytes(
    name: str,
    content: bytes,
) -> None:
    expected_size, expected_sha256 = GENERATED_ARTIFACTS[name]
    actual_sha256 = _sha256_bytes(content)
    if len(content) != expected_size or actual_sha256 != expected_sha256:
        raise RepairError(
            f"generated {name} mismatch: size={len(content)}, "
            f"sha256={actual_sha256}"
        )


def _write_new_file(path: Path, content: bytes, *, temporary_name: str) -> None:
    if path.exists() or path.is_symlink():
        raise RepairError(f"repair refuses to overwrite {path}")
    temporary = path.with_name(temporary_name)
    if temporary.exists() or temporary.is_symlink():
        raise RepairError(f"repair temporary path is not fresh: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_replace_file(
    path: Path,
    content: bytes,
    *,
    temporary_name: str,
) -> None:
    temporary = path.with_name(temporary_name)
    if temporary.exists() or temporary.is_symlink():
        raise RepairError(f"repair temporary path is not fresh: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    runner_path = args.runner.resolve()
    manifest_path = args.manifest.resolve()
    results_root = args.results_root.resolve()
    cache_dir = args.cache_dir.resolve()
    graph_dir = args.graph_dir.resolve()
    if args.code_sha.lower() != CODE_SHA:
        raise RepairError("repair code_sha is not the frozen training SHA")
    if args.cache_bundle_id != CACHE_BUNDLE_ID:
        raise RepairError("repair cache bundle id mismatch")
    if args.cache_bundle_digest.lower() != CACHE_BUNDLE_DIGEST:
        raise RepairError("repair cache bundle digest mismatch")

    _verify_clean_checkout(repo_root)
    _require_file(
        runner_path,
        size=None,
        sha256=RUNNER_SHA256,
        label="repair validator runner",
    )
    runner = _load_runner(runner_path)
    canonical = runner.load_manifest(
        manifest_path,
        require_production_matrix=True,
    )
    shard_tasks = tuple(
        task
        for task in canonical.tasks
        if _task_number(task.task_id) in EXPECTED_TASK_NUMBERS
    )
    if len(shard_tasks) != 27:
        raise RepairError("5080 shard must contain exactly 27 tasks")
    task = next((item for item in shard_tasks if item.task_id == TASK_ID), None)
    if task is None or shard_tasks.index(task) != 25:
        raise RepairError("task041 must be the 26th task in the 5080 shard")
    manifest = canonical.__class__(
        manifest_version=canonical.manifest_version,
        protocol_version=canonical.protocol_version,
        cache_schema_version=canonical.cache_schema_version,
        default_results_root=canonical.default_results_root,
        default_cache_dir=canonical.default_cache_dir,
        default_graph_dir=canonical.default_graph_dir,
        training=canonical.training,
        tasks=shard_tasks,
        source_path=canonical.source_path,
        digest=canonical.digest,
    )
    run_inputs = runner.ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=cache_dir,
        graph_dir=graph_dir,
        cache_bundle_id=args.cache_bundle_id,
        cache_bundle_digest=args.cache_bundle_digest.lower(),
        code_sha=args.code_sha.lower(),
        python_executable=args.python_executable,
    )
    runner._validate_run_inputs(run_inputs)
    runner._verify_cache_bundle_release(run_inputs)
    runner._validate_results_root_layout(results_root, manifest)

    attempt_dir = (
        results_root / ".staging" / TASK_ID / "attempt_0001"
    )
    invocation = runner.build_production_invocation(
        task,
        manifest,
        run_inputs,
        attempt_dir,
    )
    config_hash = runner.derive_config_hash(
        task,
        manifest,
        run_inputs,
        invocation_config=invocation.config,
    )
    if config_hash != CONFIG_HASH:
        raise RepairError(
            f"config hash mismatch: expected {CONFIG_HASH}, got {config_hash}"
        )
    expected_relpath = (
        Path("work") / "learnable_K3_fullwindow" / "ContraCosta"
    )
    if (
        invocation.result_relpath != expected_relpath
        or invocation.required_artifacts
        != (
            "summary.json",
            "ckpt_best.pt",
            "test_predictions.npz",
            "train.log",
            "decomp_mask.json",
        )
        or invocation.required_metrics != ("all",)
    ):
        raise RepairError("task041 production invocation contract changed")

    staged_result = attempt_dir / invocation.result_relpath
    queue_path = results_root / "queue_status.json"
    state_path = results_root / "state" / "tasks" / f"{TASK_ID}.json"
    outer_log = results_root / "logs" / TASK_ID / "attempt_0001.log"
    final_dir = results_root / task.output_dir
    incident_path = (
        results_root
        / "state"
        / "control_incidents"
        / "task041_longpath_repair.json"
    )
    repair_lock = results_root / "state" / ".r41.lock"

    _require_file(
        queue_path,
        size=None,
        sha256=QUEUE_SHA256,
        label="queue status",
    )
    _require_file(
        state_path,
        size=None,
        sha256=FAILED_STATE_SHA256,
        label="failed task state",
    )
    _require_file(
        outer_log,
        size=None,
        sha256=OUTER_LOG_SHA256,
        label="outer trainer log",
    )
    queue, queue_raw = _read_strict_json(queue_path)
    failed_state, failed_state_raw = _read_strict_json(state_path)
    _require_exact_mapping(
        queue,
        {
            "state": "failed",
            "protocol_version": manifest.protocol_version,
            "cache_schema_version": manifest.cache_schema_version,
            "cache_bundle_id": run_inputs.cache_bundle_id,
            "cache_bundle_digest": run_inputs.cache_bundle_digest,
            "code_sha": run_inputs.code_sha,
            "manifest_digest": manifest.digest,
            "total_tasks": 27,
            "completed_tasks": 25,
            "skipped_valid_tasks": 24,
            "current_task": TASK_ID,
            "error": "trainer exited with code 1",
        },
        label="queue",
    )
    _require_exact_mapping(
        failed_state,
        {
            "state": "failed",
            "task_id": TASK_ID,
            "attempt": 1,
            "returncode": 1,
            "error": "trainer exited with code 1",
            "staging_dir": str(attempt_dir),
            "log_path": str(outer_log),
        },
        label="failed task state",
    )
    _validate_log(outer_log)
    if final_dir.exists() or final_dir.is_symlink():
        raise RepairError(f"task041 final directory must not exist: {final_dir}")
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise RepairError("task041 attempt directory is missing or a link")
    if staged_result.is_symlink() or not staged_result.is_dir():
        raise RepairError("task041 staged result is missing or a link")
    if (attempt_dir.parent / "attempt_0002").exists():
        raise RepairError("task041 attempt_0002 must not exist")
    actual_names = {child.name for child in staged_result.iterdir()}
    if actual_names != set(EXISTING_ARTIFACTS):
        raise RepairError(
            f"task041 staged contents mismatch: {sorted(actual_names)}"
        )
    for name, (size, sha256) in EXISTING_ARTIFACTS.items():
        _require_file(
            staged_result / name,
            size=size,
            sha256=sha256,
            label=name,
        )
    if incident_path.exists() or incident_path.is_symlink():
        raise RepairError(f"repair incident path already exists: {incident_path}")
    if repair_lock.exists() or repair_lock.is_symlink():
        raise RepairError(f"repair lock already exists: {repair_lock}")

    os.environ["XTRAFFIC_CODE_SHA"] = CODE_SHA
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from dist_net.data import FullWindowRegionData

    region_data = FullWindowRegionData(
        "ContraCosta",
        cache_dir,
        graph_dir,
        lazy=True,
        stride=1,
        train_frac=0.7,
        val_frac=0.1,
    )
    producer, audit, valid_count = _validate_npz(
        staged_result / "test_predictions.npz",
        region_data=region_data,
    )
    best_val, mask = _load_checkpoint(
        staged_result / "ckpt_best.pt",
        expected_work=attempt_dir / "work",
    )
    metadata = region_data.result_metadata()
    expected_training_git = {
        "git_sha": CODE_SHA,
        "git_branch": "verified-deployment",
        "git_dirty": False,
    }
    if metadata.get("training_git") != expected_training_git:
        raise RepairError(
            f"training provenance mismatch: {metadata.get('training_git')!r}"
        )

    summary: dict[str, object] = {
        "all": producer,
        "affected": None,
        "unaffected": producer,
        "best_val_L_main": best_val,
        "decomp_mode": "learnable",
        "K": 3,
        "final_decomp_mask": mask,
        **metadata,
    }
    identity = {
        "task_id": TASK_ID,
        "region": "ContraCosta",
        "model": "fourier_dual_net",
        "variant": "learnable_k3",
        "seed": 42,
        "protocol_version": manifest.protocol_version,
        "sampling_protocol": "full_window",
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "config_hash": config_hash,
    }
    for key, value in identity.items():
        if key in summary and summary[key] != value:
            raise RepairError(f"summary metadata conflicts at {key}")
        summary[key] = value
    summary["valid_count"] = valid_count
    summary_bytes = _strict_json_bytes(summary)
    decomp_bytes = json.dumps(
        {"mode": "learnable", "K": 3, "mask": mask},
        indent=2,
    ).encode("utf-8")
    _require_generated_bytes("summary.json", summary_bytes)
    _require_generated_bytes("decomp_mask.json", decomp_bytes)
    runner._validate_prediction_artifact(
        staged_result / "test_predictions.npz",
        task,
        summary,
        required_metrics=("all",),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_OK",
                    "task_id": TASK_ID,
                    "config_hash": config_hash,
                    "producer_mae": producer,
                    "audit_mae": audit,
                    "valid_count": valid_count,
                    "summary_bytes": len(summary_bytes),
                    "summary_sha256": _sha256_bytes(summary_bytes),
                    "decomp_mask_bytes": len(decomp_bytes),
                    "decomp_mask_sha256": _sha256_bytes(decomp_bytes),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    repair_lock.parent.mkdir(parents=True, exist_ok=True)
    with repair_lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} task={TASK_ID}\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for path, sha256, label in (
            (queue_path, QUEUE_SHA256, "queue status"),
            (state_path, FAILED_STATE_SHA256, "failed task state"),
            (outer_log, OUTER_LOG_SHA256, "outer trainer log"),
        ):
            _require_file(path, size=None, sha256=sha256, label=label)
        for name, (size, sha256) in EXISTING_ARTIFACTS.items():
            _require_file(
                staged_result / name,
                size=size,
                sha256=sha256,
                label=name,
            )
        if final_dir.exists():
            raise RepairError("task041 final directory appeared during repair")

        _write_new_file(
            staged_result / "summary.json",
            summary_bytes,
            temporary_name=".r41s.tmp",
        )
        _write_new_file(
            staged_result / "decomp_mask.json",
            decomp_bytes,
            temporary_name=".r41d.tmp",
        )
        runner.validate_result(
            staged_result,
            task,
            manifest,
            run_inputs,
            expected_config_hash=config_hash,
            required_artifacts=invocation.required_artifacts,
            required_metrics=invocation.required_metrics,
        )
        repaired_hashes = {}
        for name, (size, sha256) in {
            **EXISTING_ARTIFACTS,
            **GENERATED_ARTIFACTS,
        }.items():
            _require_file(
                staged_result / name,
                size=size,
                sha256=sha256,
                label=name,
            )
            repaired_hashes[name] = sha256

        control_sha256 = _sha256_file(Path(__file__).resolve())
        runner_sha256 = _sha256_file(runner_path)
        incident = {
            "status": "prepared_for_atomic_promotion",
            "recorded_at": _utc_now(),
            "task_id": TASK_ID,
            "attempt": 1,
            "failure_kind": "windows_legacy_path_limit_after_inference",
            "summary_target_length": len(str(staged_result / "summary.json")),
            "failed_temporary_length": 282,
            "trainer_returncode": 1,
            "repair_returncode": 0,
            "code_sha": CODE_SHA,
            "config_hash": CONFIG_HASH,
            "original_queue_status_sha256": QUEUE_SHA256,
            "original_failed_state_sha256": FAILED_STATE_SHA256,
            "original_outer_log_sha256": OUTER_LOG_SHA256,
            "original_failed_state": failed_state_raw,
            "original_queue_status": queue_raw,
            "repaired_artifact_sha256": repaired_hashes,
            "repair_control_script_sha256": control_sha256,
            "repair_runner_sha256": runner_sha256,
            "staged_validation": "passed",
        }
        incident_path.parent.mkdir(parents=True, exist_ok=True)
        _write_new_file(
            incident_path,
            _strict_json_bytes(incident),
            temporary_name=".r41i.tmp",
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_result, final_dir)
        runner.validate_result(
            final_dir,
            task,
            manifest,
            run_inputs,
            expected_config_hash=config_hash,
            required_artifacts=invocation.required_artifacts,
            required_metrics=invocation.required_metrics,
        )
        for name, (size, sha256) in {
            **EXISTING_ARTIFACTS,
            **GENERATED_ARTIFACTS,
        }.items():
            _require_file(
                final_dir / name,
                size=size,
                sha256=sha256,
                label=f"promoted {name}",
            )

        completed_state = {
            "state": "completed",
            "updated_at": _utc_now(),
            "task_id": TASK_ID,
            "attempt": 1,
            "returncode": 0,
            "trainer_returncode": 1,
            "repair_returncode": 0,
            "config_hash": config_hash,
            "final_dir": str(final_dir),
            "log_path": str(outer_log),
            "repair_kind": (
                "task041_longpath_post_training_artifact_completion"
            ),
            "original_failed_state": failed_state_raw,
            "original_failed_state_sha256": FAILED_STATE_SHA256,
            "original_queue_status_sha256": QUEUE_SHA256,
            "repaired_artifact_sha256": repaired_hashes,
            "repair_incident_path": str(incident_path),
            "repair_control_script_sha256": control_sha256,
            "repair_runner_sha256": runner_sha256,
            "staged_validation": "passed",
            "promoted_validation": "passed",
        }
        _atomic_replace_file(
            state_path,
            _strict_json_bytes(completed_state),
            temporary_name=".r41t.tmp",
        )
    finally:
        if repair_lock.exists():
            repair_lock.unlink()
    print(
        json.dumps(
            {
                "status": "REPAIR_OK",
                "task_id": TASK_ID,
                "final_dir": str(final_dir),
                "summary_sha256": GENERATED_ARTIFACTS["summary.json"][1],
                "decomp_mask_sha256": GENERATED_ARTIFACTS[
                    "decomp_mask.json"
                ][1],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RepairError as exc:
        print(f"REPAIR_FAILED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
