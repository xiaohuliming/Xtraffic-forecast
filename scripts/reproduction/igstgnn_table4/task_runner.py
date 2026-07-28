from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .bist_factory import build_bist_factory
from .execution import (
    capture_test_predictions,
    run_upstream_training,
    save_model_state_checkpoint,
)
from .igstgnn_factory import build_igstgnn_factory
from .largest_factory import build_largest_factory
from .manifest import METHODS, load_table4_manifest
from .profiles import load_profile_set
from .result_artifacts import write_completed_task_artifacts
from .runtime_data import (
    IncidentStandardScaler,
    TupleLoaderView,
    load_official_adjacency,
    load_official_incident_stats,
    load_official_sensor_info,
)
from .secure_io import (
    open_verified_regular_file,
    read_verified_regular_file,
    secure_io_provenance,
    verify_secure_directory,
)
from .training_data import VerifiedSplitBundle, load_verified_split_bundle
from .upstream_lock import load_upstream_lock, upstream_lock_digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)
_LARGEST_METHODS = frozenset(METHODS[:11])
_MAX_SPLIT_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_TRAINING_SOURCE_BYTES = 16 * 1024 * 1024
_COMMON_TRAINING_CODE_FILES = (
    "task_runner.py",
    "execution.py",
    "manifest.py",
    "metrics.py",
    "profiles.py",
    "result_artifacts.py",
    "restricted_npy.py",
    "runtime_data.py",
    "secure_io.py",
    "training_data.py",
    "upstream_lock.py",
)


@dataclass(frozen=True)
class Table4TaskSpec:
    manifest: Mapping[str, Any]
    task: Mapping[str, Any]
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedRuntimeBundle:
    bundle: VerifiedSplitBundle
    bundle_dir: Path
    adjacency: np.ndarray
    scaler: IncidentStandardScaler
    sensor_dimensions: SimpleNamespace
    incident_stats: Mapping[str, Any]


class _TaskLogger:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def info(self, message: Any, *args: Any) -> None:
        text = str(message)
        if args:
            text = text % args
        self._handle.write(text.rstrip("\n") + "\n")
        self._handle.flush()

    warning = info
    error = info


def load_table4_task_spec(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    task_id: str,
) -> Table4TaskSpec:
    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
    )
    profile_set = load_profile_set(PROFILE_PATH)
    if (
        manifest["profile_set_id"] != profile_set["profile_set_id"]
        or manifest["profile_set_sha256"] != profile_set["profile_set_sha256"]
    ):
        raise ValueError("Matrix manifest is not bound to the current profile set")
    if manifest["source_lock_digest"] != upstream_lock_digest(load_upstream_lock()):
        raise ValueError("Matrix manifest is not bound to the current upstream lock")
    for task in manifest["tasks"]:
        method = task["method"]
        profile = profile_set["profiles"][method]
        if task["config_sha256"] != profile_set["config_hashes"][method]:
            raise ValueError(f"Matrix task profile config does not match {method}")
        if (
            task["profile_id"] != profile["profile_id"]
            or task["source_id"] != profile["source_id"]
        ):
            raise ValueError(f"Matrix task profile identity does not match {method}")
        effective_seed = profile["effective"]["seed"]
        if method == "bist":
            if (
                effective_seed is not None
                or profile["effective"].get("seed_policy")
                != "upstream_runtime_random_draw"
            ):
                raise ValueError("BiST runtime seed policy is invalid")
        elif task["seed"] != effective_seed:
            raise ValueError(f"Matrix task seed does not match {method}")
    matches = [task for task in manifest["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Task ID is not in the canonical matrix: {task_id}")
    task = matches[0]
    return Table4TaskSpec(
        manifest=manifest,
        task=task,
        profile=profile_set["profiles"][task["method"]],
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _read_bound_split_node_count(
    *,
    bundle_dir: Path,
    expected_manifest_sha256: str,
    expected_region: str,
) -> tuple[int, Path, dict[str, dict[str, Any]]]:
    directory = verify_secure_directory(
        bundle_dir,
        purpose="Table 4 split bundle",
    )
    payload = read_verified_regular_file(
        directory / "split_manifest.json",
        purpose="Table 4 split manifest",
        max_bytes=_MAX_SPLIT_MANIFEST_BYTES,
        require_nonempty=True,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(
        observed_sha256,
        expected_manifest_sha256,
    ):
        raise ValueError("Split manifest SHA-256 does not match the matrix")
    manifest = json.loads(
        bytes(payload).decode("utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    sensors = (
        manifest.get("safe_structure", {}).get("sensors")
        if isinstance(manifest, dict)
        else None
    )
    node_count = sensors.get("row_count") if isinstance(sensors, dict) else None
    if manifest.get("region") != expected_region:
        raise ValueError("Split manifest region does not match the matrix task")
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("Split manifest node count is invalid")
    support_files = manifest.get("support_files")
    required_support = {
        "sensors.csv",
        "incident_stats.npz",
        "adj_matrix.npy",
    }
    if not isinstance(support_files, dict) or not required_support.issubset(
        support_files
    ):
        raise ValueError("Split manifest support file bindings are invalid")
    runtime_bindings = {}
    for filename in sorted(required_support):
        entry = support_files[filename]
        if not isinstance(entry, dict):
            raise ValueError(f"Split manifest binding for {filename} is invalid")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if (
            type(expected_bytes) is not int
            or expected_bytes <= 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"Split manifest binding for {filename} is invalid")
        runtime_bindings[filename] = {
            "expected_bytes": expected_bytes,
            "expected_sha256": expected_sha256,
        }
    return node_count, directory, runtime_bindings


def _load_verified_runtime_bundle(
    *,
    splits_root: Path | str,
    region: str,
    expected_manifest_sha256: str,
    profile_effective: Mapping[str, Any],
    task_seed: int,
    include_incident: bool,
) -> VerifiedRuntimeBundle:
    bundle_dir = Path(splits_root) / region
    node_count, bundle_dir, support_bindings = _read_bound_split_node_count(
        bundle_dir=bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_region=region,
    )
    sensor_info, sensor_dimensions = load_official_sensor_info(
        bundle_dir / "sensors.csv",
        node_count=node_count,
        **support_bindings["sensors.csv"],
    )
    incident_stats = load_official_incident_stats(
        bundle_dir / "incident_stats.npz",
        **support_bindings["incident_stats.npz"],
    )
    if incident_stats["normalized"] is not True:
        raise ValueError("Official incident records must already be normalized")
    scaler = IncidentStandardScaler(
        mean=incident_stats["mean"],
        std=incident_stats["std"],
    )
    adjacency = load_official_adjacency(
        bundle_dir / "adj_matrix.npy",
        node_count=node_count,
        **support_bindings["adj_matrix.npy"],
    )
    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        batch_size=profile_effective["batch_size_by_region"][region],
        input_dim=profile_effective["input_contract"]["input_dim"],
        include_incident=include_incident,
        sensor_info=sensor_info if include_incident else None,
        seed=task_seed,
        loader_contract=profile_effective["loader_contract"],
    )
    if (
        bundle.node_count != node_count
        or bundle.manifest_sha256 != expected_manifest_sha256
    ):
        raise RuntimeError("Verified split loader returned a different bundle")
    return VerifiedRuntimeBundle(
        bundle=bundle,
        bundle_dir=bundle_dir,
        adjacency=adjacency,
        scaler=scaler,
        sensor_dimensions=sensor_dimensions,
        incident_stats=incident_stats,
    )


def _tuple_dataloader(bundle: VerifiedSplitBundle) -> dict[str, TupleLoaderView]:
    return {
        key: TupleLoaderView(bundle[key])
        for key in ("train_loader", "val_loader", "test_loader")
    }


def _build_factory(
    *,
    spec: Table4TaskSpec,
    runtime: VerifiedRuntimeBundle,
    device: torch.device,
    logger: _TaskLogger,
    output_dir: Path,
) -> Any:
    method = spec.task["method"]
    effective = spec.profile["effective"]
    if method in _LARGEST_METHODS:
        return build_largest_factory(
            method=method,
            profile_effective=effective,
            node_num=runtime.bundle.node_count,
            device=device,
            adjacency=runtime.adjacency,
            dataloader=_tuple_dataloader(runtime.bundle),
            scaler=runtime.scaler,
            logger=logger,
            output_dir=output_dir,
        )
    if method == "bist":
        return build_bist_factory(
            profile_effective=effective,
            manifest_frozen_seed=spec.task["seed"],
            node_num=runtime.bundle.node_count,
            device=device,
            dataloader=_tuple_dataloader(runtime.bundle),
            scaler=runtime.scaler,
            logger=logger,
            output_dir=output_dir,
        )
    if method == "igstgnn":
        return build_igstgnn_factory(
            profile_effective=effective,
            node_num=runtime.bundle.node_count,
            device=device,
            adjacency=runtime.adjacency,
            dataloader=runtime.bundle,
            scaler=runtime.scaler,
            sensor_dimensions=runtime.sensor_dimensions,
            logger=logger,
            output_dir=output_dir,
            dataset=spec.task["region"],
            data_path=runtime.bundle_dir,
        )
    raise ValueError(f"Unsupported Table 4 method: {method}")


def _prepare_results_root(path: Path | str) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    return verify_secure_directory(
        root,
        purpose="Table 4 results root",
    )


def _prepare_result_target(
    *,
    results_root: Path,
    task: Mapping[str, Any],
) -> Path:
    relative = Path(task["output_dir"])
    if relative.parts != ("tasks", task["task_id"]):
        raise ValueError("Task output path does not match its canonical identity")
    tasks_dir = results_root / "tasks"
    if tasks_dir.exists() or tasks_dir.is_symlink():
        try:
            tasks_dir = verify_secure_directory(
                tasks_dir,
                purpose="Results tasks directory",
            )
        except ValueError as error:
            raise ValueError("Results tasks directory must not be a symlink") from error
    else:
        tasks_dir.mkdir(mode=0o700)
        tasks_dir = verify_secure_directory(
            tasks_dir,
            purpose="Results tasks directory",
        )
    target = tasks_dir / task["task_id"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Task directory already exists: {target}")
    return target


def _hash_regular_source(path: Path) -> str:
    digest = hashlib.sha256()
    observed_bytes = 0
    with open_verified_regular_file(
        path,
        purpose="Training source",
        require_nonempty=True,
        max_bytes=_MAX_TRAINING_SOURCE_BYTES,
    ) as opened:
        for chunk in iter(
            lambda: opened.handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
            observed_bytes += len(chunk)
        if observed_bytes != opened.stat_result.st_size:
            raise ValueError(f"Training source could not be read completely: {path}")
    return digest.hexdigest()


def _training_code_identity(method: str) -> tuple[str, dict[str, str]]:
    factory_filename = (
        "igstgnn_factory.py"
        if method == "igstgnn"
        else "bist_factory.py"
        if method == "bist"
        else "largest_factory.py"
    )
    filenames = (*_COMMON_TRAINING_CODE_FILES, factory_filename)
    directory = Path(__file__).resolve().parent
    file_hashes = {
        filename: _hash_regular_source(directory / filename) for filename in filenames
    }
    payload = json.dumps(
        file_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), file_hashes


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but MPS is unavailable")
    return device


def _runtime_provenance(
    *,
    spec: Table4TaskSpec,
    runtime: VerifiedRuntimeBundle,
    device: torch.device,
    factory_provenance: Mapping[str, Any],
    training_code_sha256: str,
    training_code_files: Mapping[str, str],
) -> dict[str, Any]:
    device_info: dict[str, Any] = {
        "requested": str(device),
        "type": device.type,
        "index": device.index,
    }
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        device_info["resolved_index"] = index
        device_info["name"] = torch.cuda.get_device_name(index)
    declared_patches = list(spec.profile["compatibility_patches"])
    factory_patches = list(
        factory_provenance.get(
            "compatibility_patches_applied_by_factory",
            factory_provenance.get("compatibility_patches_applied", []),
        )
    )
    if not set(factory_patches).issubset(declared_patches):
        raise ValueError("Factory reported an undeclared compatibility patch")
    secure_io_identity = secure_io_provenance()

    def declared_in(candidates: set[str]) -> list[str]:
        return [patch for patch in declared_patches if patch in candidates]

    patch_stages = {
        "applied_by_data_preparation": declared_in({"restricted_split_preparation_v1"}),
        "applied_by_loader": declared_in(
            {
                "incident_object_split_adapter_v1",
                "verified_split_loader_source_semantics_v2",
            }
        ),
        "applied_by_factory": factory_patches,
        "applied_by_result_pipeline": declared_in({"isolated_artifact_paths_v1"}),
        "applied_by_runtime_io": (
            declared_in({"windows_secure_io_compatibility_v1"})
            if str(secure_io_identity["active_boundary"]).startswith("windows_")
            else []
        ),
    }
    not_applicable_patches = (
        []
        if str(secure_io_identity["active_boundary"]).startswith("windows_")
        else declared_in({"windows_secure_io_compatibility_v1"})
    )
    accounted_for = {
        patch for patches in patch_stages.values() for patch in patches
    } | set(not_applicable_patches)
    unresolved_patches = [
        patch for patch in declared_patches if patch not in accounted_for
    ]
    loader_contract = dict(spec.profile["effective"]["loader_contract"])
    batch_size = spec.profile["effective"]["batch_size_by_region"][spec.task["region"]]
    loader_splits = {}
    for split in ("train", "val", "test"):
        loader = runtime.bundle[f"{split}_loader"]
        consumed_samples = (
            loader.size
            if loader_contract["partial_batch"] == "retain"
            else min(loader.size, loader.num_batch * batch_size)
        )
        loader_splits[split] = {
            "available_samples": loader.size,
            "num_batches": loader.num_batch,
            "consumed_samples": consumed_samples,
            "dropped_tail_samples": loader.size - consumed_samples,
        }
    return {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "training_code_sha256": training_code_sha256,
        "training_code_files": dict(training_code_files),
        "matrix_manifest_sha256": spec.manifest["manifest_sha256"],
        "release_digest": spec.manifest["release_digest"],
        "source_lock_digest": spec.manifest["source_lock_digest"],
        "profile_set_sha256": spec.manifest["profile_set_sha256"],
        "config_sha256": spec.task["config_sha256"],
        "split_bundle_manifest_sha256": (spec.task["split_bundle_manifest_sha256"]),
        "task_id": spec.task["task_id"],
        "method": spec.task["method"],
        "region": spec.task["region"],
        "seed": spec.task["seed"],
        "device": device_info,
        "host": {
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
        "secure_io": secure_io_identity,
        "incident_stats": {
            "mean": float(runtime.incident_stats["mean"]),
            "std": float(runtime.incident_stats["std"]),
            "normalized": bool(runtime.incident_stats["normalized"]),
        },
        "profile_authority": dict(spec.profile["authority"]),
        "public_material_gaps": list(spec.profile["public_material_gaps"]),
        "effective_profile": dict(spec.profile["effective"]),
        "loader": {
            "contract": loader_contract,
            "batch_size": batch_size,
            "splits": loader_splits,
        },
        "compatibility_patches": {
            "declared": declared_patches,
            **patch_stages,
            "not_applicable": not_applicable_patches,
            "unresolved": unresolved_patches,
        },
        "factory": dict(factory_provenance),
    }


def run_table4_task(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    task_id: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
) -> Path:
    spec = load_table4_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        task_id=task_id,
    )
    root = _prepare_results_root(results_root)
    target = _prepare_result_target(results_root=root, task=spec.task)
    device_value = _resolve_device(device)
    include_incident = spec.task["method"] == "igstgnn"
    runtime = _load_verified_runtime_bundle(
        splits_root=splits_root,
        region=spec.task["region"],
        expected_manifest_sha256=(spec.task["split_bundle_manifest_sha256"]),
        profile_effective=spec.profile["effective"],
        task_seed=spec.task["seed"],
        include_incident=include_incident,
    )
    initial_code_sha256, initial_code_files = _training_code_identity(
        spec.task["method"]
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.task['task_id']}.run.",
        dir=root,
    ) as scratch_name:
        scratch = Path(scratch_name)
        train_log = scratch / "train.log"
        checkpoint = scratch / "checkpoint.pt"
        with train_log.open("x", encoding="utf-8") as log_handle:
            logger = _TaskLogger(log_handle)
            logger.info(
                "Starting %s with manifest %s",
                spec.task["task_id"],
                manifest_sha256,
            )
            with contextlib.redirect_stdout(log_handle):
                with contextlib.redirect_stderr(log_handle):
                    factory = _build_factory(
                        spec=spec,
                        runtime=runtime,
                        device=device_value,
                        logger=logger,
                        output_dir=scratch,
                    )
                    run_upstream_training(
                        method=spec.task["method"],
                        engine=factory.engine,
                    )
                    predictions, labels = capture_test_predictions(
                        method=spec.task["method"],
                        engine=factory.engine,
                    )
                    save_model_state_checkpoint(
                        model=factory.model,
                        target=checkpoint,
                    )
            final_code_sha256, final_code_files = _training_code_identity(
                spec.task["method"]
            )
            if (
                final_code_sha256 != initial_code_sha256
                or final_code_files != initial_code_files
            ):
                raise RuntimeError("Training code changed while the task was running")
            logger.info("Captured prediction shape: %s", predictions.shape)
            log_handle.flush()
            os.fsync(log_handle.fileno())
        provenance = _runtime_provenance(
            spec=spec,
            runtime=runtime,
            device=device_value,
            factory_provenance=factory.provenance,
            training_code_sha256=final_code_sha256,
            training_code_files=final_code_files,
        )
        return write_completed_task_artifacts(
            target_dir=target,
            task=spec.task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance=provenance,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one locked IGSTGNN Table 4 reproduction task."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    target = run_table4_task(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        task_id=args.task_id,
        splits_root=args.splits_root,
        results_root=args.results_root,
        device=args.device,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
