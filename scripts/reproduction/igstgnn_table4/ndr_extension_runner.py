from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from collections.abc import Sequence

import numpy as np

from .ndr_extension_manifest import (
    load_ndr_extension_manifest,
    load_ndr_extension_profile,
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
_CLAIM_SCOPE = "same_protocol_project_method_extension"
_TRAINING_CODE_PATHS = (
    "scripts/reproduction/igstgnn_table4/ndr_extension_runner.py",
    "scripts/reproduction/igstgnn_table4/ndr_extension_manifest.py",
    "scripts/reproduction/igstgnn_table4/training_data.py",
    "scripts/reproduction/igstgnn_table4/restricted_npy.py",
    "scripts/reproduction/igstgnn_table4/result_artifacts.py",
    "scripts/reproduction/igstgnn_table4/metrics.py",
    "scripts/reproduction/igstgnn_table4/runtime_data.py",
    "scripts/reproduction/igstgnn_table4/secure_io.py",
    "scripts/train_staeformer_deseason.py",
    "dist_net/metrics.py",
    "fourier_dual_net/deseason.py",
    "fourier_dual_net/rgdn.py",
    "baselines/GraphWaveNet/model.py",
    "baselines/STAEformer/STAEformer.py",
)


@dataclass(frozen=True)
class NDRTrainingStatistics:
    baseline: np.ndarray
    flow_mean: float
    flow_std: float
    residual_std: float
    residual_std_by_node: np.ndarray
    valid_training_points: int
    source_normalization_mean: float
    source_normalization_std: float


@dataclass(frozen=True)
class NDRTable4TaskSpec:
    manifest: Mapping[str, Any]
    task: Mapping[str, Any]
    profile: Mapping[str, Any]


@dataclass(frozen=True)
class NDRRuntimeBundle:
    bundle: VerifiedSplitBundle
    source_normalization_mean: float
    source_normalization_std: float


def _time_slots(data: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(data, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{label} must have shape batch, time, node, 3")
    if value.shape[2] <= 0:
        raise ValueError(f"{label} must contain at least one node")
    time_of_day = np.asarray(value[..., 1], dtype=np.float64)
    day_of_week = np.asarray(value[..., 2], dtype=np.float64)
    if not np.isfinite(time_of_day).all() or not np.isfinite(day_of_week).all():
        raise ValueError(f"{label} time features must be finite")
    if not np.allclose(time_of_day, time_of_day[:, :, :1], rtol=0, atol=1e-6):
        raise ValueError(f"{label} time of day differs across nodes")
    if not np.allclose(day_of_week, day_of_week[:, :, :1], rtol=0, atol=1e-6):
        raise ValueError(f"{label} day of week differs across nodes")
    tod_scaled = time_of_day[:, :, 0] * 288
    dow_scaled = day_of_week[:, :, 0] * 7
    tod = np.rint(tod_scaled).astype(np.int64)
    dow = np.rint(dow_scaled).astype(np.int64)
    if (
        not np.allclose(tod_scaled, tod, rtol=0, atol=1e-4)
        or not np.allclose(dow_scaled, dow, rtol=0, atol=1e-4)
        or np.any(tod < 0)
        or np.any(tod >= 288)
        or np.any(dow < 0)
        or np.any(dow >= 7)
    ):
        raise ValueError(f"{label} time features are outside the weekly grid")
    return dow, tod


def _iter_flow_and_slots(
    loader: Any,
    *,
    source_normalization_mean: float,
    source_normalization_std: float,
) -> Any:
    for batch in loader.get_iterator():
        if not isinstance(batch, Mapping):
            raise ValueError("Verified loader batch must be a mapping")
        for key in ("x_data", "y_data"):
            data = np.asarray(batch[key], dtype=np.float32)
            dow, tod = _time_slots(data, label=key)
            flow = np.asarray(
                data[..., 0] * np.float32(source_normalization_std)
                + np.float32(source_normalization_mean),
                dtype=np.float32,
            )
            flow = np.where(np.abs(flow) > 1e-4, flow, np.float32(0))
            yield flow, dow, tod


def fit_training_climatology(
    train_loader: Any,
    *,
    node_count: int,
    source_normalization_mean: float = 0.0,
    source_normalization_std: float = 1.0,
) -> NDRTrainingStatistics:
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("node_count must be a positive integer")
    if (
        not np.isfinite(source_normalization_mean)
        or not np.isfinite(source_normalization_std)
        or source_normalization_std <= 0
    ):
        raise ValueError("Source normalization statistics are invalid")
    counts = np.zeros((7, 288), dtype=np.int64)
    flow_sum = np.float64(0)
    flow_square_sum = np.float64(0)
    valid_training_points = 0
    observed_positions = 0

    iterator_args = {
        "source_normalization_mean": float(source_normalization_mean),
        "source_normalization_std": float(source_normalization_std),
    }
    for flow, dow, tod in _iter_flow_and_slots(train_loader, **iterator_args):
        if flow.shape[2] != node_count:
            raise ValueError("Training flow node count does not match the split bundle")
        if not np.isfinite(flow).all():
            raise ValueError("Training flow contains non-finite values")
        np.add.at(counts, (dow.reshape(-1), tod.reshape(-1)), 1)
        mask = flow != 0
        values = np.asarray(flow[mask], dtype=np.float64)
        flow_sum += values.sum(dtype=np.float64)
        flow_square_sum += np.square(values).sum(dtype=np.float64)
        valid_training_points += int(values.size)
        observed_positions += int(flow.shape[0] * flow.shape[1])
    if observed_positions <= 0 or valid_training_points <= 0:
        raise ValueError("Training loader contains no valid flow observations")

    cell_values: list[list[np.ndarray | None]] = [
        [
            np.empty((int(counts[dow, tod]), node_count), dtype=np.float32)
            if counts[dow, tod]
            else None
            for tod in range(288)
        ]
        for dow in range(7)
    ]
    cursors = np.zeros((7, 288), dtype=np.int64)
    for flow, dow, tod in _iter_flow_and_slots(train_loader, **iterator_args):
        for batch_index in range(flow.shape[0]):
            for step in range(flow.shape[1]):
                day_index = int(dow[batch_index, step])
                time_index = int(tod[batch_index, step])
                cursor = int(cursors[day_index, time_index])
                values = cell_values[day_index][time_index]
                if values is None:
                    raise RuntimeError("Climatology allocation is inconsistent")
                values[cursor] = flow[batch_index, step]
                cursors[day_index, time_index] += 1
    if not np.array_equal(cursors, counts):
        raise RuntimeError("Training loader changed while fitting climatology")

    baseline = np.zeros((7, 288, node_count), dtype=np.float32)
    for day_index in range(7):
        for time_index in range(288):
            values = cell_values[day_index][time_index]
            if values is None:
                continue
            masked = np.where(values != 0, values, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(masked, axis=0)
            baseline[day_index, time_index] = np.where(
                np.isfinite(median),
                median,
                np.float32(0),
            )

    residual_sum = np.float64(0)
    residual_square_sum = np.float64(0)
    residual_count = 0
    node_sum = np.zeros(node_count, dtype=np.float64)
    node_square_sum = np.zeros(node_count, dtype=np.float64)
    node_count_valid = np.zeros(node_count, dtype=np.int64)
    for flow, dow, tod in _iter_flow_and_slots(train_loader, **iterator_args):
        expected = baseline[dow, tod]
        mask = flow != 0
        residual = np.asarray(flow - expected, dtype=np.float64)
        selected = residual[mask]
        residual_sum += selected.sum(dtype=np.float64)
        residual_square_sum += np.square(selected).sum(dtype=np.float64)
        residual_count += int(selected.size)
        node_sum += np.where(mask, residual, 0).sum(axis=(0, 1), dtype=np.float64)
        node_square_sum += np.where(mask, residual * residual, 0).sum(
            axis=(0, 1),
            dtype=np.float64,
        )
        node_count_valid += mask.sum(axis=(0, 1), dtype=np.int64)
    if residual_count != valid_training_points:
        raise RuntimeError("Training loader changed while fitting residual scale")

    flow_mean = float(flow_sum / valid_training_points)
    flow_variance = max(
        float(flow_square_sum / valid_training_points - flow_mean * flow_mean),
        0.0,
    )
    flow_std = float(np.sqrt(flow_variance))
    residual_mean = float(residual_sum / residual_count)
    residual_variance = max(
        float(
            residual_square_sum / residual_count
            - residual_mean * residual_mean
        ),
        0.0,
    )
    residual_std = float(np.sqrt(residual_variance))
    node_mean = np.divide(
        node_sum,
        node_count_valid,
        out=np.zeros_like(node_sum),
        where=node_count_valid > 0,
    )
    node_variance = np.divide(
        node_square_sum,
        node_count_valid,
        out=np.zeros_like(node_square_sum),
        where=node_count_valid > 0,
    ) - np.square(node_mean)
    node_std = np.sqrt(np.maximum(node_variance, 0))
    node_std = np.where(
        (node_count_valid > 0) & np.isfinite(node_std) & (node_std > 1e-6),
        node_std,
        1.0,
    ).astype(np.float32)
    if not np.isfinite(flow_std) or flow_std <= 1e-6:
        flow_std = 1.0
    if not np.isfinite(residual_std) or residual_std <= 1e-6:
        residual_std = 1.0
    baseline.setflags(write=False)
    node_std.setflags(write=False)
    return NDRTrainingStatistics(
        baseline=baseline,
        flow_mean=flow_mean,
        flow_std=flow_std,
        residual_std=residual_std,
        residual_std_by_node=node_std,
        valid_training_points=valid_training_points,
        source_normalization_mean=float(source_normalization_mean),
        source_normalization_std=float(source_normalization_std),
    )


def prepare_ndr_batch(
    batch: Mapping[str, Any],
    statistics: NDRTrainingStatistics,
) -> dict[str, np.ndarray]:
    x_data = np.asarray(batch["x_data"], dtype=np.float32)
    y_data = np.asarray(batch["y_data"], dtype=np.float32)
    x_dow, x_tod = _time_slots(x_data, label="x_data")
    y_dow, y_tod = _time_slots(y_data, label="y_data")
    x_baseline = statistics.baseline[x_dow, x_tod]
    y_baseline = statistics.baseline[y_dow, y_tod]
    flow_hist = np.asarray(
        x_data[..., 0] * np.float32(statistics.source_normalization_std)
        + np.float32(statistics.source_normalization_mean),
        dtype=np.float32,
    )
    target_time_major = np.asarray(
        y_data[..., 0] * np.float32(statistics.source_normalization_std)
        + np.float32(statistics.source_normalization_mean),
        dtype=np.float32,
    )
    flow_hist = np.where(np.abs(flow_hist) > 1e-4, flow_hist, np.float32(0))
    target_time_major = np.where(
        np.abs(target_time_major) > 1e-4,
        target_time_major,
        np.float32(0),
    )
    target = np.asarray(
        target_time_major.transpose(0, 2, 1),
        dtype=np.float32,
    )
    return {
        "flow_hist": np.asarray(
            flow_hist.transpose(0, 2, 1),
            dtype=np.float32,
        ),
        "x_baseline": np.asarray(
            x_baseline.transpose(0, 2, 1),
            dtype=np.float32,
        ),
        "y_baseline": np.asarray(
            y_baseline.transpose(0, 2, 1),
            dtype=np.float32,
        ),
        "time_feat": np.asarray(x_data[:, :, 0, 1:3], dtype=np.float32),
        "target": target,
        "target_mask": np.asarray(target != 0, dtype=np.bool_),
    }


def load_ndr_table4_task_spec(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    task_id: str,
) -> NDRTable4TaskSpec:
    manifest = load_ndr_extension_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    profile = load_ndr_extension_profile(
        Path(manifest_path).parent / manifest["profile_path"],
        expected_sha256=manifest["profile_sha256"],
    )
    tasks = [task for task in manifest["tasks"] if task["task_id"] == task_id]
    if len(tasks) != 1:
        raise ValueError(f"Task ID is not in the canonical NDR extension: {task_id}")
    task = tasks[0]
    if (
        task["config_sha256"] != manifest["profile_sha256"]
        or task["profile_id"] != profile["profile_id"]
        or task["source_id"] != profile["source_id"]
        or task["seed"] not in profile["effective"]["seeds"]
    ):
        raise ValueError("NDR extension task is not bound to the current profile")
    return NDRTable4TaskSpec(manifest=manifest, task=task, profile=profile)


def _hash_regular_source(path: Path) -> str:
    digest = hashlib.sha256()
    observed_bytes = 0
    with open_verified_regular_file(
        path,
        purpose="NDR extension training source",
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
    spec: NDRTable4TaskSpec,
    splits_root: Path | str,
) -> NDRRuntimeBundle:
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
    return NDRRuntimeBundle(
        bundle=bundle,
        source_normalization_mean=float(statistics["mean"]),
        source_normalization_std=float(statistics["std"]),
    )


def _make_model(
    *,
    node_count: int,
    statistics: NDRTrainingStatistics,
    device: Any,
) -> Any:
    import torch
    from scripts.train_staeformer_deseason import (
        STAEformerDeseason,
        VARIANTS,
    )

    model = STAEformerDeseason(
        node_count,
        12,
        12,
        **dict(VARIANTS["s0g"]),
    ).to(device)
    with torch.no_grad():
        model.sd_res.fill_(statistics.residual_std)
        model.flow_mu.fill_(statistics.flow_mean)
        model.flow_sd.fill_(statistics.flow_std)
        model.sd_node.copy_(
            torch.from_numpy(
                np.array(statistics.residual_std_by_node, copy=True)
            ).to(device)
        )
    return model


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


def _forward(model: Any, tensors: Mapping[str, Any]) -> Any:
    return model(
        tensors["flow_hist"],
        tensors["x_baseline"],
        tensors["y_baseline"],
        tensors["time_feat"],
    )


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
        raise RuntimeError("Test loader did not retain every official split sample")
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise RuntimeError("NDR extension produced non-finite predictions or labels")
    return predictions, labels


def _prepare_result_target(
    *,
    results_root: Path | str,
    task: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    root = verify_secure_directory(root, purpose="NDR extension results root")
    relative = Path(task["output_dir"])
    if relative.parts != ("tasks", task["task_id"]):
        raise ValueError("NDR extension output path is not canonical")
    tasks_root = root / "tasks"
    if tasks_root.exists() or tasks_root.is_symlink():
        tasks_root = verify_secure_directory(
            tasks_root,
            purpose="NDR extension tasks root",
        )
    else:
        tasks_root.mkdir(mode=0o700)
        tasks_root = verify_secure_directory(
            tasks_root,
            purpose="NDR extension tasks root",
        )
    target = tasks_root / task["task_id"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Task directory already exists: {target}")
    return root, target


def _loader_provenance(bundle: VerifiedSplitBundle) -> dict[str, Any]:
    splits = {}
    for split in ("train", "val", "test"):
        loader = bundle[f"{split}_loader"]
        splits[split] = {
            "available_samples": loader.size,
            "num_batches": loader.num_batch,
            "consumed_samples": loader.size,
            "dropped_tail_samples": 0,
        }
    return splits


def _runtime_provenance(
    *,
    spec: NDRTable4TaskSpec,
    bundle: VerifiedSplitBundle,
    statistics: NDRTrainingStatistics,
    device: Any,
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
    return {
        "schema_version": 1,
        "protocol_id": "igstgnn_table4_official_v1",
        "matrix_id": spec.manifest["matrix_id"],
        "matrix_manifest_sha256": spec.manifest["manifest_sha256"],
        "base_matrix_id": spec.manifest["base_matrix_id"],
        "base_manifest_sha256": spec.manifest["base_manifest_sha256"],
        "claim_scope": _CLAIM_SCOPE,
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
        "effective_profile": dict(spec.profile["effective"]),
        "profile_authority": dict(spec.profile["authority"]),
        "known_limitations": list(spec.profile["known_limitations"]),
        "compatibility_patches": list(spec.profile["compatibility_patches"]),
        "loader": {
            "contract": dict(spec.profile["effective"]["loader_contract"]),
            "batch_size": spec.profile["effective"]["batch_size_by_region"][
                spec.task["region"]
            ],
            "target_dim": 3,
            "splits": _loader_provenance(bundle),
        },
        "training_only_climatology": {
            "baseline_sha256": baseline_sha256,
            "shape": list(statistics.baseline.shape),
            "fit_split": "train",
            "fit_positions": ["x", "y"],
            "zero_is_null": True,
            "valid_training_points": statistics.valid_training_points,
            "source_normalization": {
                "mean": statistics.source_normalization_mean,
                "std": statistics.source_normalization_std,
                "inverse_transformed_before_model": True,
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
    }


def run_ndr_extension_task(
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
    from scripts.train_staeformer_deseason import compute_loss

    spec = load_ndr_table4_task_spec(
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
                        f"Starting {spec.task['task_id']} manifest={manifest_sha256}",
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
                    model = _make_model(
                        node_count=bundle.node_count,
                        statistics=statistics,
                        device=device_value,
                    )
                    parameter_count = sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                    print(f"parameters={parameter_count}", flush=True)
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
                            prediction = _forward(model, tensors)
                            loss, _ = compute_loss(
                                model,
                                prediction,
                                loss_batch,
                                loss_args,
                                device_value,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                optimizer_config["gradient_clip"],
                            )
                            optimizer.step()
                            scheduler.step()
                            batch_size = int(prediction.shape[0])
                            weighted_loss += float(loss.item()) * batch_size
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
                        raise RuntimeError("NDR extension did not produce a checkpoint")
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
            raise RuntimeError("NDR extension training code changed during the task")
        provenance = _runtime_provenance(
            spec=spec,
            bundle=bundle,
            statistics=statistics,
            device=device_value,
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
            claim_scope=_CLAIM_SCOPE,
        )


def probe_ndr_extension_task(
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
    from scripts.train_staeformer_deseason import compute_loss

    spec = load_ndr_table4_task_spec(
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
    model = _make_model(
        node_count=bundle.node_count,
        statistics=statistics,
        device=device_value,
    )
    batch = next(bundle.train_loader.get_iterator())
    tensors, loss_batch = _tensor_batch(
        batch,
        statistics,
        device=device_value,
    )
    prediction = _forward(model, tensors)
    effective = spec.profile["effective"]
    loss, _ = compute_loss(
        model,
        prediction,
        loss_batch,
        SimpleNamespace(
            lambda_gate=effective["loss"]["lambda_gate"],
            lambda_spec=effective["loss"]["lambda_specialization"],
        ),
        device_value,
    )
    loss.backward()
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss):
        raise RuntimeError("NDR extension probe produced non-finite values")
    peak_bytes = None
    if device_value.type == "cuda":
        torch.cuda.synchronize(device_value)
        peak_bytes = int(torch.cuda.max_memory_allocated(device_value))
    print(
        "NDR_EXTENSION_SMOKE_OK "
        f"task={task_id} shape={tuple(prediction.shape)} "
        f"loss={float(loss.item()):.8f} peak_allocated_bytes={peak_bytes}",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one locked NDR-Net Table 4 extension task."
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
        probe_ndr_extension_task(**common)
        return 0
    if args.results_root is None:
        raise ValueError("--results-root is required unless --smoke is used")
    target = run_ndr_extension_task(
        **common,
        results_root=args.results_root,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
