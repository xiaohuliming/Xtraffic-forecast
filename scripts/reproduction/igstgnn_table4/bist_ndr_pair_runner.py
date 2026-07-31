from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from collections.abc import Sequence

import numpy as np

from .bist_ndr_pair_manifest import (
    PAIR_CLAIM_SCOPE,
    load_bist_ndr_pair_manifest,
)
from .bist_ndr_pair_model import build_bist_ndr_pair_model
from .ndr_extension_runner import (
    NDRTrainingStatistics,
    fit_training_climatology,
    prepare_ndr_batch,
)
from .result_artifacts import write_completed_task_artifacts
from .runtime_data import load_official_incident_stats
from .secure_io import (
    open_verified_regular_file,
    secure_io_provenance,
    verify_secure_directory,
)
from .training_data import VerifiedSplitBundle, load_verified_split_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MAX_TRAINING_SOURCE_BYTES = 16 * 1024 * 1024
_TRAINING_CODE_PATHS = (
    "scripts/reproduction/igstgnn_table4/bist_ndr_pair_runner.py",
    "scripts/reproduction/igstgnn_table4/bist_ndr_pair_manifest.py",
    "scripts/reproduction/igstgnn_table4/bist_ndr_pair_model.py",
    "scripts/reproduction/igstgnn_table4/bist_factory.py",
    "scripts/reproduction/igstgnn_table4/upstream_lock.py",
    "scripts/reproduction/igstgnn_table4/ndr_extension_runner.py",
    "scripts/reproduction/igstgnn_table4/training_data.py",
    "scripts/reproduction/igstgnn_table4/restricted_npy.py",
    "scripts/reproduction/igstgnn_table4/result_artifacts.py",
    "scripts/reproduction/igstgnn_table4/metrics.py",
    "scripts/reproduction/igstgnn_table4/runtime_data.py",
    "scripts/reproduction/igstgnn_table4/secure_io.py",
    "scripts/train_staeformer_deseason.py",
    "dist_net/metrics.py",
    "fourier_dual_net/rgdn.py",
    "baselines/GraphWaveNet/model.py",
    "baselines/STAEformer/STAEformer.py",
    ".third_party/igstgnn_table4/pristine/bist_dd94adf/src/models/bist.py",
    ".third_party/igstgnn_table4/pristine/bist_dd94adf/src/base/model.py",
)


@dataclass(frozen=True)
class BiSTNDRPairTaskSpec:
    manifest: Mapping[str, Any]
    task: Mapping[str, Any]
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class BiSTNDRPairRuntimeBundle:
    bundle: VerifiedSplitBundle
    source_normalization_mean: float
    source_normalization_std: float


def _forward(model: Any, tensors: Mapping[str, Any]) -> Any:
    return model(
        tensors["flow_hist"],
        tensors["x_baseline"],
        tensors["y_baseline"],
        tensors["time_feat"],
    )


def backward_bist_ndr_pair_batch(
    *,
    model: Any,
    tensors: Mapping[str, Any],
    loss_batch: Mapping[str, Any],
    loss_args: Any,
    device: Any,
    max_microbatch_size: int,
) -> tuple[float, tuple[int, ...], int]:
    from scripts.train_staeformer_deseason import (
        build_ndr_loss_normalizers,
        compute_loss,
    )

    if type(max_microbatch_size) is not int or max_microbatch_size <= 0:
        raise ValueError("max_microbatch_size must be a positive integer")
    batch_size = int(tensors["target"].shape[0])
    if batch_size <= 0:
        raise ValueError("BiST-NDR paired training batch is empty")
    use_microbatching = batch_size > max_microbatch_size
    normalizers = None
    if use_microbatching:
        if model.oracle_sup and model.dual:
            normalizers = build_ndr_loss_normalizers(
                model,
                loss_batch,
                baseline=tensors["y_baseline"],
                persistence=tensors["flow_hist"][:, :, -1:].expand_as(
                    tensors["y_baseline"]
                ),
                device=device,
            )
        else:
            valid_count = loss_batch["y_mask"][..., 0].to(device).float().sum()
            if int(valid_count.item()) <= 0:
                raise ValueError("BiST paired full batch has no valid observations")
            normalizers = {"valid_count": valid_count.detach()}

    total_loss = 0.0
    microbatch_count = 0
    for start in range(0, batch_size, max_microbatch_size):
        stop = min(start + max_microbatch_size, batch_size)
        chunk = slice(start, stop)
        micro_tensors = {
            key: value[chunk]
            for key, value in tensors.items()
        }
        micro_loss_batch = {
            key: value[chunk]
            for key, value in loss_batch.items()
        }
        prediction = _forward(model, micro_tensors)
        loss, _ = compute_loss(
            model,
            prediction,
            micro_loss_batch,
            loss_args,
            device,
            normalizers=normalizers,
        )
        loss.backward()
        total_loss += float(loss.item())
        microbatch_count += 1
    prediction_shape = (
        batch_size,
        int(tensors["target"].shape[1]),
        int(tensors["target"].shape[2]),
    )
    if not np.isfinite(total_loss):
        raise RuntimeError("BiST-NDR paired batch produced a non-finite loss")
    return total_loss, prediction_shape, microbatch_count


def load_bist_ndr_pair_task_spec(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    task_id: str,
) -> BiSTNDRPairTaskSpec:
    manifest = load_bist_ndr_pair_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    matches = [
        task for task in manifest["tasks"] if task["task_id"] == task_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Task ID is not in the canonical BiST-NDR pair matrix: {task_id}"
        )
    task = matches[0]
    profile = manifest["profile"]
    method_profile = profile["methods"][task["method"]]
    if (
        task["config_sha256"] != manifest["profile_sha256"]
        or task["profile_id"] != profile["profile_id"]
        or task["source_id"] != method_profile["source_id"]
        or task["seed"] not in profile["effective"]["seeds"]
    ):
        raise ValueError("BiST-NDR paired task is not bound to its profile")
    return BiSTNDRPairTaskSpec(
        manifest=manifest,
        task=task,
        profile=profile,
    )


def _hash_regular_source(path: Path) -> str:
    digest = hashlib.sha256()
    observed_bytes = 0
    with open_verified_regular_file(
        path,
        purpose="BiST-NDR paired training source",
        require_nonempty=True,
        max_bytes=_MAX_TRAINING_SOURCE_BYTES,
    ) as opened:
        for chunk in iter(lambda: opened.handle.read(1024 * 1024), b""):
            digest.update(chunk)
            observed_bytes += len(chunk)
        if observed_bytes != opened.stat_result.st_size:
            raise ValueError(f"Training source could not be read completely: {path}")
    return digest.hexdigest()


def _training_code_identity() -> tuple[str, dict[str, str]]:
    file_hashes = {
        relative: _hash_regular_source(PROJECT_ROOT / relative)
        for relative in _TRAINING_CODE_PATHS
    }
    payload = json.dumps(
        file_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), file_hashes


def _resolve_device(value: str) -> Any:
    import torch

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested but MPS is unavailable")
    return device


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_runtime_bundle(
    *,
    spec: BiSTNDRPairTaskSpec,
    splits_root: Path | str,
) -> BiSTNDRPairRuntimeBundle:
    effective = spec.profile["effective"]
    region = spec.task["region"]
    bundle_dir = Path(splits_root) / region
    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=spec.task["split_bundle_manifest_sha256"],
        batch_size=effective["batch_size_by_region"][region],
        input_dim=3,
        target_dim=3,
        include_incident=False,
        sensor_info=None,
        seed=spec.task["seed"],
        loader_contract=effective["loader_contract"],
    )
    entry = bundle.manifest.get("support_files", {}).get("incident_stats.npz")
    if not isinstance(entry, dict):
        raise ValueError("Official flow normalization binding is missing")
    statistics = load_official_incident_stats(
        bundle_dir / "incident_stats.npz",
        expected_bytes=entry.get("bytes"),
        expected_sha256=entry.get("sha256"),
    )
    if statistics["normalized"] is not True:
        raise ValueError("Official Table 4 flow must be stored in normalized units")
    return BiSTNDRPairRuntimeBundle(
        bundle=bundle,
        source_normalization_mean=float(statistics["mean"]),
        source_normalization_std=float(statistics["std"]),
    )


def _tensor_batch(
    batch: Mapping[str, Any],
    statistics: NDRTrainingStatistics,
    *,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    prepared = prepare_ndr_batch(batch, statistics)
    tensors = {
        key: torch.from_numpy(value).to(device)
        for key, value in prepared.items()
    }
    loss_batch = {
        "y_true": tensors["target"].unsqueeze(-1),
        "y_mask": tensors["target_mask"].unsqueeze(-1),
    }
    return tensors, loss_batch


def _max_microbatch_size(
    *,
    spec: BiSTNDRPairTaskSpec,
) -> int:
    batch_size = spec.profile["effective"]["batch_size_by_region"][
        spec.task["region"]
    ]
    return min(batch_size, 8) if spec.task["region"] == "Orange" else batch_size


def _evaluate_validation(
    *,
    model: Any,
    loader: Any,
    statistics: NDRTrainingStatistics,
    device: Any,
) -> float:
    import torch

    was_training = model.training
    model.eval()
    absolute_error_sum = np.float64(0)
    valid_count = 0
    with torch.no_grad():
        for batch in loader.get_iterator():
            tensors, _ = _tensor_batch(batch, statistics, device=device)
            prediction = _forward(model, tensors)
            mask = tensors["target_mask"]
            selected = torch.abs(prediction - tensors["target"])[mask]
            absolute_error_sum += np.float64(
                selected.detach().double().sum().cpu().item()
            )
            valid_count += int(mask.sum().cpu().item())
    if was_training:
        model.train()
    if valid_count <= 0:
        raise RuntimeError("Validation split contains no valid observations")
    return float(absolute_error_sum / valid_count)


def _capture_test_predictions(
    *,
    model: Any,
    loader: Any,
    statistics: NDRTrainingStatistics,
    node_count: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    predictions = np.empty((loader.size, 12, node_count), dtype=np.float32)
    labels = np.empty_like(predictions)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader.get_iterator():
            tensors, _ = _tensor_batch(batch, statistics, device=device)
            prediction = (
                _forward(model, tensors)
                .permute(0, 2, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            label = (
                tensors["target"]
                .permute(0, 2, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            batch_size = prediction.shape[0]
            predictions[cursor : cursor + batch_size] = prediction
            labels[cursor : cursor + batch_size] = label
            cursor += batch_size
    if cursor != loader.size:
        raise RuntimeError(
            "Paired test loader did not retain every official split sample"
        )
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise RuntimeError("BiST-NDR pairing produced non-finite test arrays")
    return predictions, labels


def _prepare_result_target(
    *,
    results_root: Path | str,
    task: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    root = verify_secure_directory(root, purpose="BiST-NDR paired results root")
    relative = Path(task["output_dir"])
    if relative.parts != ("tasks", task["task_id"]):
        raise ValueError("BiST-NDR paired output path is not canonical")
    tasks_root = root / "tasks"
    if tasks_root.exists() or tasks_root.is_symlink():
        tasks_root = verify_secure_directory(
            tasks_root,
            purpose="BiST-NDR paired tasks root",
        )
    else:
        tasks_root.mkdir(mode=0o700)
        tasks_root = verify_secure_directory(
            tasks_root,
            purpose="BiST-NDR paired tasks root",
        )
    target = tasks_root / task["task_id"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Task directory already exists: {target}")
    return root, target


def _loader_provenance(bundle: VerifiedSplitBundle) -> dict[str, Any]:
    return {
        split: {
            "available_samples": bundle[f"{split}_loader"].size,
            "num_batches": bundle[f"{split}_loader"].num_batch,
            "consumed_samples": bundle[f"{split}_loader"].size,
            "dropped_tail_samples": 0,
        }
        for split in ("train", "val", "test")
    }


def _runtime_provenance(
    *,
    spec: BiSTNDRPairTaskSpec,
    bundle: VerifiedSplitBundle,
    statistics: NDRTrainingStatistics,
    device: Any,
    model_provenance: Mapping[str, Any],
    parameter_count: int,
    training_code_sha256: str,
    training_code_files: Mapping[str, str],
    best_validation_mae: float,
    best_epoch: int,
) -> dict[str, Any]:
    import torch

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
    baseline_sha256 = hashlib.sha256(
        np.ascontiguousarray(statistics.baseline).tobytes()
    ).hexdigest()
    batch_size = spec.profile["effective"]["batch_size_by_region"][
        spec.task["region"]
    ]
    max_microbatch_size = _max_microbatch_size(spec=spec)
    return {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": spec.manifest["matrix_id"],
        "matrix_manifest_sha256": spec.manifest["manifest_sha256"],
        "base_matrix_id": spec.manifest["base_matrix_id"],
        "base_manifest_sha256": spec.manifest["base_manifest_sha256"],
        "claim_scope": PAIR_CLAIM_SCOPE,
        "validity_label": spec.manifest["validity_label"],
        "training_code_sha256": training_code_sha256,
        "training_code_files": dict(training_code_files),
        "config_sha256": spec.task["config_sha256"],
        "split_bundle_manifest_sha256": spec.task[
            "split_bundle_manifest_sha256"
        ],
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
        "secure_io": secure_io_provenance(),
        "selected_method_profile": dict(
            spec.profile["methods"][spec.task["method"]]
        ),
        "effective_profile": dict(spec.profile["effective"]),
        "profile_authority": dict(spec.profile["authority"]),
        "known_limitations": list(spec.profile["known_limitations"]),
        "compatibility_patches": list(spec.profile["compatibility_patches"]),
        "model": {
            **dict(model_provenance),
            "trainable_parameter_count": parameter_count,
        },
        "loader": {
            "contract": dict(
                spec.profile["effective"]["loader_contract"]
            ),
            "batch_size": batch_size,
            "target_dim": 3,
            "splits": _loader_provenance(bundle),
        },
        "training_batch_adapter": {
            "mode": (
                "direct_full_batch"
                if max_microbatch_size == batch_size
                else "loss_normalized_microbatch_accumulation_v1"
            ),
            "declared_batch_size": batch_size,
            "max_microbatch_size": max_microbatch_size,
            "optimizer_updates_per_effective_batch": 1,
            "scheduler_steps_per_effective_batch": 1,
        },
        "training_only_climatology": {
            "used_by_model": spec.task["method"] == "bist_ndr",
            "baseline_sha256": baseline_sha256,
            "shape": list(statistics.baseline.shape),
            "fit_split": "train",
            "fit_positions": ["x", "y"],
            "zero_is_null": True,
            "valid_training_points": statistics.valid_training_points,
            "source_normalization": {
                "mean": statistics.source_normalization_mean,
                "std": statistics.source_normalization_std,
                "inverse_transformed_before_loss": True,
            },
            "flow_mean": statistics.flow_mean,
            "flow_std": statistics.flow_std,
            "residual_std": statistics.residual_std,
            "residual_std_by_node_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    statistics.residual_std_by_node
                ).tobytes()
            ).hexdigest(),
        },
        "checkpoint_selection": {
            "metric": "validation_masked_mae_all_valid_points",
            "best_value": best_validation_mae,
            "best_epoch": best_epoch,
        },
        "paired_evaluation_contract": {
            "test_observation_set": (
                "identical_retained_official_split_samples_for_both_methods"
            ),
            "all_partial_batches_retained": True,
            "comparison_requires_matching_region_and_seed": True,
        },
    }


def run_bist_ndr_pair_task(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    task_id: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
) -> Path:
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    spec = load_bist_ndr_pair_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        task_id=task_id,
    )
    root, target = _prepare_result_target(
        results_root=results_root,
        task=spec.task,
    )
    device_value = _resolve_device(device)
    _set_seed(spec.task["seed"])
    runtime = _load_runtime_bundle(spec=spec, splits_root=splits_root)
    bundle = runtime.bundle
    statistics = fit_training_climatology(
        bundle.train_loader,
        node_count=bundle.node_count,
        source_normalization_mean=runtime.source_normalization_mean,
        source_normalization_std=runtime.source_normalization_std,
    )
    initial_code_sha256, initial_code_files = _training_code_identity()
    effective = spec.profile["effective"]
    optimizer_config = effective["optimizer"]
    loss_args = SimpleNamespace(
        lambda_gate=effective["loss"]["lambda_gate"],
        lambda_spec=effective["loss"]["lambda_specialization"],
    )
    epochs = effective["max_epochs"]
    max_microbatch_size = _max_microbatch_size(spec=spec)
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.task['task_id']}.run.",
        dir=root,
    ) as scratch_name:
        scratch = Path(scratch_name)
        train_log = scratch / "train.log"
        checkpoint = scratch / "checkpoint.pt"
        with train_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle):
                with contextlib.redirect_stderr(log_handle):
                    print(
                        f"Starting {spec.task['task_id']} "
                        f"manifest={manifest_sha256}",
                        flush=True,
                    )
                    print(
                        "paired loader partial_batch=retain "
                        f"train={bundle.train_loader.size} "
                        f"val={bundle.val_loader.size} "
                        f"test={bundle.test_loader.size}",
                        flush=True,
                    )
                    print(
                        "training-only climatology "
                        f"valid_points={statistics.valid_training_points} "
                        f"flow_mean={statistics.flow_mean:.6f} "
                        f"flow_std={statistics.flow_std:.6f} "
                        f"residual_std={statistics.residual_std:.6f}",
                        flush=True,
                    )
                    model_result = build_bist_ndr_pair_model(
                        method=spec.task["method"],
                        node_count=bundle.node_count,
                        effective_profile=effective,
                        statistics=statistics,
                        device=device_value,
                    )
                    model = model_result.model
                    parameter_count = sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                    print(
                        f"method={spec.task['method']} "
                        f"parameters={parameter_count} "
                        f"effective_batch={effective['batch_size_by_region'][spec.task['region']]} "
                        f"max_microbatch={max_microbatch_size}",
                        flush=True,
                    )
                    optimizer = Adam(
                        model.parameters(),
                        lr=optimizer_config["learning_rate"],
                        weight_decay=optimizer_config["weight_decay"],
                    )
                    scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=max(
                            1,
                            epochs * bundle.train_loader.num_batch,
                        ),
                        eta_min=optimizer_config["scheduler"]["eta_min"],
                    )
                    best_validation_mae = float("inf")
                    best_epoch = 0
                    for epoch in range(1, epochs + 1):
                        model.train()
                        bundle.train_loader.shuffle()
                        weighted_loss = 0.0
                        sample_count = 0
                        for batch in bundle.train_loader.get_iterator():
                            tensors, loss_batch = _tensor_batch(
                                batch,
                                statistics,
                                device=device_value,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            batch_loss, _, _ = backward_bist_ndr_pair_batch(
                                model=model,
                                tensors=tensors,
                                loss_batch=loss_batch,
                                loss_args=loss_args,
                                device=device_value,
                                max_microbatch_size=max_microbatch_size,
                            )
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                optimizer_config["gradient_clip"],
                            )
                            optimizer.step()
                            scheduler.step()
                            batch_size = int(tensors["target"].shape[0])
                            weighted_loss += batch_loss * batch_size
                            sample_count += batch_size
                        validation_mae = _evaluate_validation(
                            model=model,
                            loader=bundle.val_loader,
                            statistics=statistics,
                            device=device_value,
                        )
                        print(
                            f"epoch={epoch:03d} "
                            f"train_loss={weighted_loss / max(sample_count, 1):.8f} "
                            f"validation_mae={validation_mae:.8f}",
                            flush=True,
                        )
                        if validation_mae < best_validation_mae:
                            best_validation_mae = validation_mae
                            best_epoch = epoch
                            torch.save(
                                {
                                    "model_state": model.state_dict(),
                                    "task": dict(spec.task),
                                    "best_validation_mae": best_validation_mae,
                                    "best_epoch": best_epoch,
                                    "statistics": {
                                        "baseline": np.array(
                                            statistics.baseline,
                                            copy=True,
                                        ),
                                        "flow_mean": statistics.flow_mean,
                                        "flow_std": statistics.flow_std,
                                        "residual_std": statistics.residual_std,
                                        "residual_std_by_node": np.array(
                                            statistics.residual_std_by_node,
                                            copy=True,
                                        ),
                                        "source_normalization_mean": (
                                            statistics.source_normalization_mean
                                        ),
                                        "source_normalization_std": (
                                            statistics.source_normalization_std
                                        ),
                                    },
                                },
                                checkpoint,
                            )
                    if best_epoch <= 0 or not checkpoint.is_file():
                        raise RuntimeError(
                            "BiST-NDR paired task did not produce a checkpoint"
                        )
                    saved = torch.load(
                        checkpoint,
                        map_location=device_value,
                        weights_only=False,
                    )
                    model.load_state_dict(saved["model_state"])
                    predictions, labels = _capture_test_predictions(
                        model=model,
                        loader=bundle.test_loader,
                        statistics=statistics,
                        node_count=bundle.node_count,
                        device=device_value,
                    )
                    print(
                        f"best_epoch={best_epoch} "
                        f"best_validation_mae={best_validation_mae:.8f} "
                        f"prediction_shape={predictions.shape}",
                        flush=True,
                    )
            log_handle.flush()
            os.fsync(log_handle.fileno())
        final_code_sha256, final_code_files = _training_code_identity()
        if (
            final_code_sha256 != initial_code_sha256
            or final_code_files != initial_code_files
        ):
            raise RuntimeError(
                "BiST-NDR paired training code changed during the task"
            )
        provenance = _runtime_provenance(
            spec=spec,
            bundle=bundle,
            statistics=statistics,
            device=device_value,
            model_provenance=model_result.provenance,
            parameter_count=parameter_count,
            training_code_sha256=final_code_sha256,
            training_code_files=final_code_files,
            best_validation_mae=best_validation_mae,
            best_epoch=best_epoch,
        )
        return write_completed_task_artifacts(
            target_dir=target,
            task=spec.task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance=provenance,
            claim_scope=PAIR_CLAIM_SCOPE,
        )


def probe_bist_ndr_pair_task(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    task_id: str,
    splits_root: Path | str,
    device: str,
) -> None:
    import torch

    spec = load_bist_ndr_pair_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        task_id=task_id,
    )
    device_value = _resolve_device(device)
    _set_seed(spec.task["seed"])
    runtime = _load_runtime_bundle(spec=spec, splits_root=splits_root)
    bundle = runtime.bundle
    statistics = fit_training_climatology(
        bundle.train_loader,
        node_count=bundle.node_count,
        source_normalization_mean=runtime.source_normalization_mean,
        source_normalization_std=runtime.source_normalization_std,
    )
    model_result = build_bist_ndr_pair_model(
        method=spec.task["method"],
        node_count=bundle.node_count,
        effective_profile=spec.profile["effective"],
        statistics=statistics,
        device=device_value,
    )
    model = model_result.model
    batch = next(bundle.train_loader.get_iterator())
    tensors, loss_batch = _tensor_batch(
        batch,
        statistics,
        device=device_value,
    )
    effective = spec.profile["effective"]
    model.zero_grad(set_to_none=True)
    loss_value, prediction_shape, microbatch_count = (
        backward_bist_ndr_pair_batch(
            model=model,
            tensors=tensors,
            loss_batch=loss_batch,
            loss_args=SimpleNamespace(
                lambda_gate=effective["loss"]["lambda_gate"],
                lambda_spec=effective["loss"]["lambda_specialization"],
            ),
            device=device_value,
            max_microbatch_size=_max_microbatch_size(spec=spec),
        )
    )
    if not np.isfinite(loss_value):
        raise RuntimeError("BiST-NDR paired probe produced non-finite values")
    peak_bytes = None
    if device_value.type == "cuda":
        torch.cuda.synchronize(device_value)
        peak_bytes = int(torch.cuda.max_memory_allocated(device_value))
    print(
        "BIST_NDR_PAIR_SMOKE_OK "
        f"task={task_id} method={spec.task['method']} "
        f"shape={prediction_shape} loss={loss_value:.8f} "
        f"microbatches={microbatch_count} "
        f"peak_allocated_bytes={peak_bytes}",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one locked BiST versus BiST-NDR paired task."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    common = {
        "manifest_path": args.manifest,
        "manifest_sha256": args.manifest_sha256,
        "base_manifest_path": args.base_manifest,
        "base_manifest_sha256": args.base_manifest_sha256,
        "task_id": args.task_id,
        "splits_root": args.splits_root,
        "device": args.device,
    }
    if args.smoke:
        probe_bist_ndr_pair_task(**common)
        return 0
    if args.results_root is None:
        raise ValueError("--results-root is required unless --smoke is used")
    target = run_bist_ndr_pair_task(
        **common,
        results_root=args.results_root,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
