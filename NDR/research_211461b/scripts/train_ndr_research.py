#!/usr/bin/env python3
"""Isolated, resumable NDR development using v3 event train and validation only."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dist_net.metrics import MaskedMAEAccumulator
from dist_net.protocol import git_worktree_provenance
from ndr_research.data import (
    ResearchData,
    array_digest,
    canonical_digest,
    load_research_data,
    prepare_research_batch,
    sha256_file,
)
from ndr_research.models import (
    RESEARCH_VARIANTS,
    build_research_loss_normalizers,
    compute_research_loss,
    forward_research,
    make_research_model,
    research_factory_kwargs,
    research_persistence,
)

CHECKPOINT_SCHEMA = 1


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    paths = {Path(__file__).resolve()}
    paths.update((root / "ndr_research").glob("*.py"))
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(root):
                relative = path.relative_to(root)
                if relative.parts[0] not in {"tests", "outputs"}:
                    paths.add(path)
    files = {
        path.relative_to(root).as_posix(): {"sha256": sha256_file(path),
                                          "size_bytes": path.stat().st_size}
        for path in sorted(paths) if path.is_file() and path.is_relative_to(root)
    }
    if not files:
        raise ValueError("source snapshot contains no source files")
    return {"files": files, "files_digest": canonical_digest(files),
            "git": git_worktree_provenance(root)}


def require_snapshot(snapshot: Mapping[str, Any], root: Path = ROOT) -> None:
    if canonical_digest(snapshot["files"]) != snapshot["files_digest"]:
        raise ValueError("source snapshot digest is invalid")
    for relative, declaration in snapshot["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("source snapshot path escapes the checkout")
        if not path.is_file() or sha256_file(path) != declaration["sha256"]:
            raise ValueError(f"source snapshot changed: {relative}")


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0],
                  "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
                  "position": numpy_state[2], "gaussian": numpy_state[3],
                  "cached_gaussian": numpy_state[4]},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], value["keys"].cpu().numpy().astype(np.uint32),
                         value["position"], value["gaussian"], value["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("strict resume requires the original CUDA RNG topology")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def runtime_identity(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "numpy": np.__version__, "cuda": torch.version.cuda,
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "precision": "float32",
    }


def configure_runtime(device_name: str, seed: int, threads: int = 3) -> torch.device:
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("research training supports explicit CPU smoke or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is forbidden")
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return device


def batch_tensors(batch: Mapping[str, Any], data: ResearchData,
                  device: torch.device) -> dict[str, torch.Tensor]:
    prepared = prepare_research_batch(batch, data.statistics)
    tensors = {
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in prepared.items() if key != "sample_start"
    }
    valid = tensors["target_mask"].bool()
    if not torch.isfinite(tensors["target"][valid]).all():
        raise FloatingPointError("validation/training has non-finite observed targets")
    # The original NDR auxiliary loss multiplies by masks after arithmetic.
    tensors["target"] = torch.where(valid, tensors["target"], 0.0)
    return tensors


def model_forward(model: Any, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    prediction = forward_research(
        model, tensors["flow_hist"], tensors["x_baseline"],
        tensors["y_baseline"], tensors["time_feat"],
        history_mask=tensors["history_mask"],
    )
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("forecast must be finite everywhere and match target shape")
    return prediction


def backward_effective_batch(
    model: Any, tensors: Mapping[str, torch.Tensor], *, microbatch_size: int,
    lambda_gate: float = 1.0, lambda_specialization: float = 0.3,
) -> dict[str, Any]:
    count = int(tensors["target_mask"].sum().item())
    metric = MaskedMAEAccumulator()
    parts: dict[str, float] = {}
    normalizers = None
    if count:
        with torch.no_grad():
            persistence = research_persistence(
                model, tensors["flow_hist"], tensors["x_baseline"],
                history_mask=tensors["history_mask"],
            ).expand_as(tensors["y_baseline"])
            normalizers = build_research_loss_normalizers(
                model, tensors["target"], tensors["target_mask"],
                baseline=tensors["y_baseline"], persistence=persistence,
            )
    batch_size = int(tensors["flow_hist"].shape[0])
    for start in range(0, batch_size, microbatch_size):
        chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
        prediction = model_forward(model, chunk)
        metric.update(prediction, chunk["target"], chunk["target_mask"])
        if not chunk["target_mask"].any():
            continue
        loss, local_parts = compute_research_loss(
            model, prediction, chunk["target"], chunk["target_mask"],
            lambda_gate=lambda_gate, lambda_specialization=lambda_specialization,
            normalizers=normalizers,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("training objective is non-finite")
        loss.backward()
        for key, value in local_parts.items():
            parts[key] = parts.get(key, 0.0) + float(value)
    objective = parts.get("mae", 0.0) + lambda_gate * parts.get("gate", 0.0)
    objective += lambda_specialization * parts.get("spec", 0.0)
    return {"error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
            "objective": objective, "loss_parts": parts}


def optimize_batch(model: Any, optimizer: Any, scheduler: Any,
                   tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = backward_effective_batch(
        model, tensors, microbatch_size=config["microbatch_size"],
        lambda_gate=config["lambda_gate"],
        lambda_specialization=config["lambda_specialization"],
    )
    result["updated"] = bool(result["valid_count"])
    if result["updated"]:
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("gradient is non-finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()
               if value.is_floating_point()):
            raise FloatingPointError("optimizer produced a non-finite model state")
    return result


def selected_val_indices(data: ResearchData, max_batches: int | None) -> np.ndarray:
    count = data.val.size if max_batches is None else min(data.val.size, max_batches * data.val.bs)
    return data.val.indices[:count]


def evaluation_identity(data: ResearchData, max_batches: int | None) -> dict[str, Any]:
    indices = selected_val_indices(data, max_batches)
    return {
        "split": "val", "samples": int(indices.size),
        "sample_indices_sha256": array_digest(indices),
        "forecast_origins_sha256": array_digest(data.region.sample_start[indices]),
        "region_cache_digest": data.identity["region_cache_digest"],
        "all_validation_samples": bool(indices.size == data.val.size),
    }


def evaluate(model: Any, data: ResearchData, device: torch.device, *,
             microbatch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    metric = MaskedMAEAccumulator()
    starts = []
    target_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    horizon_errors = np.zeros(data.region.T_p, dtype=np.float64)
    horizon_counts = np.zeros(data.region.T_p, dtype=np.int64)
    data.val.start_epoch()
    batches = min(data.val.num_batch, max_batches or data.val.num_batch)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch = data.val.next_batch()
                starts.extend(np.asarray(batch["sample_start"], dtype=np.int64).tolist())
                tensors = batch_tensors(batch, data, device)
                target_digest.update(tensors["target"].cpu().contiguous().numpy().tobytes())
                mask_digest.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
                for start in range(0, tensors["target"].shape[0], microbatch_size):
                    chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
                    prediction = model_forward(model, chunk)
                    metric.update(prediction, chunk["target"], chunk["target_mask"])
                    valid = chunk["target_mask"].bool()
                    errors = torch.where(valid, (prediction - chunk["target"]).abs(), 0.0)
                    horizon_errors += errors.double().sum(dim=(0, 1)).cpu().numpy()
                    horizon_counts += valid.sum(dim=(0, 1)).cpu().numpy()
        identity = evaluation_identity(data, max_batches)
        if array_digest(np.asarray(starts, dtype=np.int64)) != identity["forecast_origins_sha256"]:
            raise ValueError("validation stream differs from the declared sample collection")
        horizon_mae = [float(error / count) if count else None
                       for error, count in zip(horizon_errors, horizon_counts)]
        return {"mae": metric.compute(), "global_masked_mae": metric.compute(),
                "horizon_mae": horizon_mae,
                "horizon_valid_counts": horizon_counts.tolist(),
                "average_horizon_mae": float(np.mean(horizon_mae)) if horizon_counts.all() else None,
                "absolute_error_sum": metric._absolute_error_sum,
                "valid_count": metric.valid_count, "batches": batches,
                "selection": identity, "observed_targets_sha256": target_digest.hexdigest(),
                "observation_masks_sha256": mask_digest.hexdigest(),
                "finite_predictions_everywhere": True}
    finally:
        model.train(was_training)


def _loader_checkpoint(loader: Any) -> dict[str, Any]:
    state = loader.state_dict()
    state["order"] = torch.from_numpy(state["order"].copy())
    return state


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any,
                    data: ResearchData, config: Mapping[str, Any],
                    progress: Mapping[str, Any], selection_metric: Mapping[str, Any] | None) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "config": dict(config),
        "config_digest": canonical_digest(config), "model_factory": research_factory_kwargs(model),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": capture_rng_state(),
        "loaders": {"train": _loader_checkpoint(data.train), "val": _loader_checkpoint(data.val)},
        "data_identity": data.identity, "progress": copy.deepcopy(progress),
        "selection_metric": dict(selection_metric) if selection_metric is not None else None,
        "resume_boundary": "completed_effective_batch_or_epoch",
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"),
                    {"sha256": digest, "size_bytes": path.stat().st_size,
                     "config_digest": payload["config_digest"]})
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if path.stat().st_size != marker["size_bytes"] or sha256_file(path) != marker["sha256"]:
        raise ValueError("checkpoint content hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if (canonical_digest(checkpoint["config"]) != checkpoint["config_digest"]
            or checkpoint["config_digest"] != marker["config_digest"]):
        raise ValueError("checkpoint configuration digest mismatch")
    required = {"model", "optimizer", "scheduler", "rng", "loaders", "progress", "data_identity"}
    if not required.issubset(checkpoint):
        raise ValueError("checkpoint is not strictly resumable")
    return checkpoint


def create_optimizer(model: Any, config: Mapping[str, Any], batches: int) -> tuple[Any, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * batches, eta_min=config["eta_min"],
    )
    return optimizer, scheduler


def restore_training(checkpoint: Mapping[str, Any], model: Any, optimizer: Any,
                     scheduler: Any, data: ResearchData, config: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint["config_digest"] != canonical_digest(config) or checkpoint["data_identity"] != data.identity:
        raise ValueError("strict resume code/config/cache/sample identity mismatch")
    require_snapshot(config["source_snapshot"])
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for name in ("train", "val"):
        getattr(data, name).load_state_dict(checkpoint["loaders"][name])
    restore_rng_state(checkpoint["rng"])
    return copy.deepcopy(checkpoint["progress"])


def empty_partial() -> dict[str, Any]:
    return {"error_sum": 0.0, "valid_count": 0, "objective_sum": 0.0,
            "optimized_batches": 0, "batches": 0, "seconds": 0.0}


def write_history(output: Path, history: list[dict[str, Any]]) -> None:
    temporary = output / ".training_diagnostics.tmp"
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                 for row in history), encoding="utf-8")
    os.replace(temporary, output / "training_diagnostics.jsonl")


def train_run(*, model: Any, data: ResearchData, device: torch.device,
              config: Mapping[str, Any], output: Path,
              checkpoint: Mapping[str, Any] | None = None,
              max_wallclock_seconds: float | None = None) -> dict[str, Any]:
    train_batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    optimizer, scheduler = create_optimizer(model, config, train_batches)
    progress: dict[str, Any] = {
        "epoch": 0, "stage": "start", "best_epoch": None, "best_val_mae": None,
        "history": [], "updates": 0, "partial_epoch": empty_partial(), "elapsed_seconds": 0.0,
    }
    if checkpoint is not None:
        progress = restore_training(checkpoint, model, optimizer, scheduler, data, config)
    write_history(output, progress["history"])
    stop = {"requested": False}
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(
                signum, lambda *_: stop.__setitem__("requested", True))
        except ValueError:
            pass
    started = time.monotonic()
    accumulated = progress["elapsed_seconds"]

    def stopping() -> bool:
        return stop["requested"] or (max_wallclock_seconds is not None
                                    and time.monotonic() - started >= max_wallclock_seconds)

    def persist_last() -> None:
        progress["elapsed_seconds"] = accumulated + time.monotonic() - started
        save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                        data=data, config=config, progress=progress, selection_metric=None)

    try:
        atomic_json(output / "run_state.json", {"status": "running", "config_digest": canonical_digest(config)})
        while progress["epoch"] < config["epochs"]:
            if progress["stage"] == "start":
                data.train.start_epoch()
                progress["partial_epoch"] = empty_partial()
                progress["stage"] = "train"
            while progress["stage"] == "train" and data.train.current_ind < train_batches:
                if stopping():
                    persist_last()
                    result = {"status": "stopped_resumable", "reason": "signal" if stop["requested"] else "wallclock",
                              "completed_epochs": progress["epoch"], "next_batch": data.train.current_ind,
                              "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config)}
                    atomic_json(output / "run_state.json", result)
                    return result
                step_started = time.monotonic()
                batch = data.train.next_batch()
                tensors = batch_tensors(batch, data, device)
                step = optimize_batch(model, optimizer, scheduler, tensors, config)
                partial = progress["partial_epoch"]
                partial["error_sum"] += step["error_sum"]
                partial["valid_count"] += step["valid_count"]
                partial["objective_sum"] += step["objective"]
                partial["optimized_batches"] += int(step["updated"])
                partial["batches"] += 1
                partial["seconds"] += time.monotonic() - step_started
                progress["updates"] += int(step["updated"])
                if data.train.current_ind == 1 or data.train.current_ind % config["log_interval"] == 0:
                    print(json.dumps({"epoch": progress["epoch"] + 1, "batch": data.train.current_ind,
                                      "batches": train_batches, "updates": progress["updates"],
                                      "objective": step["objective"]}), flush=True)
            progress["stage"] = "validate"
            if stopping():
                persist_last()
                result = {"status": "stopped_resumable", "reason": "signal" if stop["requested"] else "wallclock",
                          "completed_epochs": progress["epoch"], "next_stage": "validate",
                          "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config)}
                atomic_json(output / "run_state.json", result)
                return result
            val_started = time.monotonic()
            validation = evaluate(model, data, device, microbatch_size=config["microbatch_size"],
                                  max_batches=config["max_val_batches"])
            partial = progress["partial_epoch"]
            if partial["valid_count"] == 0:
                raise ValueError("training epoch has no valid target observations")
            row = {"epoch": progress["epoch"] + 1,
                   "train_mae": partial["error_sum"] / partial["valid_count"],
                   "train_valid_count": partial["valid_count"],
                   "train_objective_batch_mean": partial["objective_sum"] / max(1, partial["optimized_batches"]),
                   "train_seconds": partial["seconds"], "validation_seconds": time.monotonic() - val_started,
                   "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                   "updates": progress["updates"]}
            progress["history"].append(row)
            improved = progress["best_val_mae"] is None or validation["mae"] < progress["best_val_mae"]
            progress["epoch"] += 1
            progress["stage"] = "start"
            if improved:
                progress["best_val_mae"] = validation["mae"]
                progress["best_epoch"] = progress["epoch"]
            progress["elapsed_seconds"] = accumulated + time.monotonic() - started
            write_history(output, progress["history"])
            if improved:
                save_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                                data=data, config=config, progress=progress, selection_metric=validation)
            persist_last()
            print(json.dumps(row, sort_keys=True), flush=True)
        best_validation = progress["history"][progress["best_epoch"] - 1]["validation"]
        result = {"status": "training_completed_validation_pending_independent_verification",
                  "completed_epochs": progress["epoch"], "best_epoch": progress["best_epoch"],
                  "best_val_mae": progress["best_val_mae"], "updates": progress["updates"],
                  "best_validation": best_validation,
                  "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config),
                  "test_evaluated": False, "elapsed_seconds": progress["elapsed_seconds"]}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def verify_checkpoint(*, checkpoint_path: Path, data: ResearchData, device: torch.device,
                      output: Path, current_runtime: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    require_snapshot(config["source_snapshot"])
    if checkpoint["data_identity"] != data.identity:
        raise ValueError("independent verification cache/sample identity mismatch")
    if current_runtime != config["runtime"]:
        raise ValueError("independent verification runtime differs from the training runtime")
    expected = checkpoint["selection_metric"]
    if expected is None:
        raise ValueError("verification requires a selected best checkpoint with a recorded validation metric")
    if evaluation_identity(data, config["max_val_batches"]) != expected["selection"]:
        raise ValueError("independent verification selects a different validation collection")
    model = make_research_model(**(checkpoint["model_factory"] | {"device": str(device)}))
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate(model, data, device, microbatch_size=config["microbatch_size"],
                      max_batches=config["max_val_batches"])
    identity_keys = ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256")
    if any(actual[key] != expected[key] for key in identity_keys):
        raise ValueError("verification did not reproduce the exact validation observations")
    if not math.isclose(actual["mae"], expected["mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent checkpoint validation MAE mismatch")
    status = "verified_checkpoint_complete_validation"
    if config["claim_scope"] == "smoke_only":
        status = ("verified_smoke_checkpoint_complete_validation"
                  if actual["selection"]["all_validation_samples"]
                  else "verified_smoke_checkpoint_subset")
    result = {
        "status": status,
        "checkpoint_sha256": sha256_file(checkpoint_path), "config_digest": checkpoint["config_digest"],
        "validation": actual, "expected_validation_mae": expected["mae"],
        "absolute_mae_difference": abs(actual["mae"] - expected["mae"]),
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "cache_bundle_digest": data.identity["cache_bundle_digest"],
        "test_evaluated": False, "claim_scope": config["claim_scope"],
    }
    atomic_json(output / "verification.json", result)
    return result


def calibrate(model: Any, data: ResearchData, device: torch.device,
              config: Mapping[str, Any], *, warmup: int, timed: int, output: Path) -> dict[str, Any]:
    optimizer, scheduler = create_optimizer(model, config, data.train.num_batch)
    durations = []
    data.train.start_epoch()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for number in range(warmup + timed):
        if data.train.current_ind >= data.train.num_batch:
            data.train.start_epoch()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch = data.train.next_batch()
        tensors = batch_tensors(batch, data, device)
        optimize_batch(model, optimizer, scheduler, tensors, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if number >= warmup:
            durations.append(time.perf_counter() - started)
    train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    data.val.start_epoch()
    val_durations = []
    with torch.no_grad():
        for number in range(warmup + timed):
            if data.val.current_ind >= data.val.num_batch:
                data.val.start_epoch()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = data.val.next_batch()
            tensors = batch_tensors(batch, data, device)
            for start in range(0, tensors["target"].shape[0], config["microbatch_size"]):
                chunk = {key: value[start:start + config["microbatch_size"]] for key, value in tensors.items()}
                model_forward(model, chunk)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if number >= warmup:
                val_durations.append(time.perf_counter() - started)
    train_mean, val_mean = float(np.mean(durations)), float(np.mean(val_durations))
    result = {
        "status": "calibrated_engineering_only", "claim_scope": "calibration_only",
        "warmup_batches": warmup, "timed_batches": timed,
        "mean_train_batch_seconds": train_mean, "mean_val_batch_seconds": val_mean,
        "estimated_train_val_seconds": config["epochs"] * (train_mean * data.train.num_batch + val_mean * data.val.num_batch),
        "train_peak_allocated_bytes": train_peak,
        "val_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "config_digest": canonical_digest(config), "test_evaluated": False,
    }
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate", "verify"), default="train")
    parser.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    parser.add_argument("--variant", choices=RESEARCH_VARIANTS, default="baseline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=0.00001)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-specialization", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--reliable-anchor", action="store_true")
    parser.add_argument("--max-reference-age", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-wallclock-seconds", type=float)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--lazy", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "microbatch_size", "hidden_dim", "timed_batches", "threads", "log_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.microbatch_size > args.batch_size:
        raise ValueError("microbatch size cannot exceed the effective batch size")
    if args.warmup_batches < 0:
        raise ValueError("warmup batch count cannot be negative")
    for name in ("max_train_batches", "max_val_batches", "max_wallclock_seconds"):
        if getattr(args, name) is not None and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            raise ValueError(f"{name} must be positive when provided")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("verify requires --checkpoint")
    if args.mode != "verify" and args.checkpoint is not None:
        raise ValueError("--checkpoint is only for independent verification")
    if args.resume is not None and args.mode != "train":
        raise ValueError("resume is only available for train mode")
    if args.resume is not None and args.resume.name != "last.pt":
        raise ValueError("strict continuation must use last.pt")
    if args.mode == "verify" and (args.max_train_batches is not None or args.max_val_batches is not None):
        raise ValueError("verification must reuse the checkpoint validation collection without subset overrides")
    for name in ("learning_rate", "gradient_clip", "eta_min"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "lambda_gate", "lambda_specialization"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def reject_identity_overrides(args: argparse.Namespace, saved: Mapping[str, Any],
                              arguments: list[str]) -> None:
    mutable = {"mode", "device", "data_dir", "graph_dir", "output_dir", "checkpoint",
               "resume", "max_wallclock_seconds", "warmup_batches", "timed_batches", "lazy"}
    for name, current in vars(args).items():
        option = "--" + name.replace("_", "-")
        explicit = any(value == option or value.startswith(option + "=") for value in arguments)
        if explicit and name not in mutable and current != saved.get(name):
            raise ValueError(f"resume/verify cannot override frozen identity field: {option}")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    validate_arguments(args)
    previous = load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous is not None else None
    if saved is not None:
        reject_identity_overrides(args, saved, arguments)
    seed, threads = (saved["seed"], saved["threads"]) if saved else (args.seed, args.threads)
    device = configure_runtime(args.device, seed, threads)
    runtime = runtime_identity(device)
    if saved is not None:
        require_snapshot(saved["source_snapshot"])
        if runtime != saved["runtime"]:
            raise ValueError("resume/verify runtime must exactly match the original runtime")
    data = load_research_data(
        region_name=saved["region"] if saved else args.region,
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        batch_size=saved["batch_size"] if saved else args.batch_size,
        seed=seed, preload=not args.lazy,
    )
    output = args.output_dir or ((args.resume or args.checkpoint).parent if previous else None)
    if output is None:
        raise ValueError("new runs require an explicit isolated --output-dir")
    output = output.resolve()
    if args.resume and output != args.resume.resolve().parent:
        raise ValueError("resume output must be the original run directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_checkpoint(checkpoint_path=args.checkpoint, data=data, device=device,
                                   output=output, current_runtime=runtime)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if previous:
        config = saved
        factory = previous["model_factory"] | {"device": str(device)}
    else:
        config = {name: getattr(args, name) for name in (
            "region", "variant", "epochs", "seed", "batch_size", "microbatch_size",
            "learning_rate", "weight_decay", "gradient_clip", "eta_min", "lambda_gate",
            "lambda_specialization", "hidden_dim", "reliable_anchor", "max_reference_age",
            "max_train_batches", "max_val_batches", "log_interval", "threads")}
        config.update({"schema_version": 1, "purpose": "v3_event_train_validation_development",
                       "claim_scope": "smoke_only" if args.max_train_batches is not None or args.max_val_batches is not None else "validation_development_only",
                       "source_snapshot": source_snapshot(), "runtime": runtime,
                       "data_identity": data.identity, "precision": "FP32",
                       "early_stopping": False, "test_evaluation": False,
                       "val_selection": evaluation_identity(data, args.max_val_batches)})
        factory = {"node_count": data.region.N, "variant": args.variant,
                   "residual_std": data.statistics.residual_std,
                   "residual_std_by_node": data.statistics.residual_std_by_node.tolist(),
                   "flow_mean": data.statistics.flow_mean, "flow_std": data.statistics.flow_std,
                   "reliable_anchor": args.reliable_anchor, "max_reference_age": args.max_reference_age,
                   "hidden_dim": args.hidden_dim, "device": str(device)}
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
        atomic_json(output / "source_snapshot.json", config["source_snapshot"])
        atomic_json(output / "data_identity.json", data.identity)
    model = make_research_model(**factory)
    if args.mode == "calibrate":
        result = calibrate(model, data, device, config, warmup=args.warmup_batches,
                           timed=args.timed_batches, output=output)
    else:
        result = train_run(model=model, data=data, device=device, config=config, output=output,
                           checkpoint=previous, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
