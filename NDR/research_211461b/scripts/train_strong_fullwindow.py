#!/usr/bin/env python3
"""Independent full-window raw STAEformer and locked native BiST development.

Only full-window source train/validation data are accessible. This runner never
constructs a test loader or executes the old automatic-test scripts. Native
BiST recipes are capped at a declared 30/100 epochs, not a reproduction of its
500-epoch convergence budget. Its native training BN uses a full effective
batch, including the retained tail. Matched-recipe BN microbatch changes require
an explicit declaration and remain a different recipe.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dist_net.metrics import MaskedMAEAccumulator
from dist_net.protocol import git_worktree_provenance
from ndr_research.data import canonical_digest, sha256_file
from ndr_research.fullwindow_data import FullWindowStatistics, legacy_array_digest
from ndr_research.fullwindow_training import load_training_data, validation_identity as shared_validation_identity
from ndr_research.strong_references import METHODS, RECIPES, make_strong_reference, reference_recipe
from scripts.train_ndr_research import (
    atomic_json, capture_rng_state, configure_runtime, empty_partial, require_snapshot,
    restore_rng_state, runtime_identity, write_history,
)
from scripts.train_ndr_strong_reference import (
    _calibration_validation_batch, _require_model_identity, create_reference_optimizer,
    optimize_reference, strong_forward,
)
from scripts.train_ndr_transfer import checkpoint_safe

SCHEMA = "fullwindow_strong_checkpoint_v1"
FORMAL_SCOPE = "fullwindow_validation_development_capped"
INPUT_FIELDS = ("flow_hist", "time_feat", "target", "target_mask")


def source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the real import closure, including raw-material and sampling code."""
    root = Path(root).resolve()
    pending = [root / "scripts/train_strong_fullwindow.py"]
    files: set[Path] = set()

    def enqueue(parts: tuple[str, ...]) -> None:
        for length in range(1, len(parts) + 1):
            package = root.joinpath(*parts[:length], "__init__.py")
            if package.is_file():
                pending.append(package)
        module = root.joinpath(*parts).with_suffix(".py")
        if module.is_file():
            pending.append(module)

    while pending:
        path = pending.pop().resolve()
        if path in files or not path.is_file() or not path.is_relative_to(root):
            continue
        files.add(path)
        package = path.relative_to(root).parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                for item in node.names:
                    enqueue(tuple(item.name.split(".")))
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else ()
                module = base + (tuple(node.module.split(".")) if node.module else ())
                if module:
                    enqueue(module)
                    for item in node.names:
                        if item.name != "*":
                            enqueue(module + tuple(item.name.split(".")))
    for relative in ("config/reproduction/igstgnn_table4/profiles.public_material_v2.json",
                     "scripts/reproduction/igstgnn_table4/upstreams.lock.json"):
        files.add(root / relative)
    declarations = {path.relative_to(root).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(files)}
    return {"files": declarations, "files_digest": canonical_digest(declarations),
            "git": git_worktree_provenance(root), "scope": "fullwindow_strong_static_import_closure_and_locked_profiles"}


def fullwindow_recipe(method: str, recipe: str, *, region: str, epochs: int,
                      batch_size: int | None = None, microbatch_size: int | None = None,
                      allow_bn_microbatch_override: bool = False) -> dict[str, Any]:
    result = reference_recipe(method, recipe, region=region, epochs=epochs,
                              batch_size=batch_size, microbatch_size=microbatch_size)
    changed_bn = result["batch_normalization_scope"] == "microbatch_override"
    if method == "bist" and recipe == "native" and changed_bn:
        raise ValueError("native BiST requires full effective training batches, not BN microbatches")
    if changed_bn and not allow_bn_microbatch_override:
        raise ValueError("matched BiST BN microbatch changes require an explicit declaration")
    result.update(sampling_protocol="full_window", budget_label=f"fullwindow_capped_{epochs}_epoch_development",
                  native_tail_override="native_drop_to_fullwindow_retain" if method == "bist" else None,
                  native_convergence_reproduced=False,
                  normalizer="weekly7_float64_unique_raw_observed_train_prefix",
                  legacy27_normalizer_difference="legacy_scalar_mean_std_used_float32_reductions",
                  bn_microbatch_override_declared=changed_bn and allow_bn_microbatch_override,
                  full_validation_point_weighted=True)
    return result


def tensors_from_batch(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {name: torch.as_tensor(np.array(batch[name], copy=True), device=device) for name in INPUT_FIELDS}
    if tensors["target_mask"].dtype != torch.bool:
        raise ValueError("full-window strong objective requires the original boolean mask")
    observed = tensors["target_mask"]
    if not torch.isfinite(tensors["target"][observed]).all():
        raise FloatingPointError("observed training/validation targets must be finite")
    tensors["target"] = torch.where(observed, tensors["target"], 0.0)
    return tensors


def validation_identity(data: Any, max_batches: int | None = None) -> dict[str, Any]:
    return shared_validation_identity(data, max_batches)


def evaluate_strong(model: Any, data: Any, device: torch.device, *, microbatch_size: int,
                    max_batches: int | None = None) -> dict[str, Any]:
    if microbatch_size <= 0:
        raise ValueError("validation microbatch size must be positive")
    was_training = model.training
    model.eval()
    metric = MaskedMAEAccumulator()
    errors, counts = np.zeros(12, np.float64), np.zeros(12, np.int64)
    target_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    origins = []
    data.val.start_epoch(0)
    batches = min(data.val.num_batch, max_batches or data.val.num_batch)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch = data.val.get_next()
                if batch is None:
                    raise RuntimeError("validation loader exhausted before its declared collection")
                origins.extend(batch["sample_start"].tolist())
                tensors = tensors_from_batch(batch, device)
                target_hash.update(tensors["target"].cpu().contiguous().numpy().tobytes())
                mask_hash.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
                for start in range(0, len(tensors["target"]), microbatch_size):
                    chunk = {name: value[start:start + microbatch_size] for name, value in tensors.items()}
                    prediction = strong_forward(model, chunk)
                    valid = chunk["target_mask"]
                    metric.update(prediction, chunk["target"], valid)
                    absolute = torch.where(valid, (prediction - chunk["target"]).abs(), 0)
                    errors += absolute.double().sum((0, 1)).cpu().numpy()
                    counts += valid.sum((0, 1)).cpu().numpy()
        selection = validation_identity(data, max_batches)
        if legacy_array_digest(np.asarray(origins, np.int64)) != selection["forecast_origins_sha256"]:
            raise ValueError("validation collection differs from the declared full-window origins")
        horizon = [float(error / count) if count else None for error, count in zip(errors, counts)]
        collection = {"sampling_protocol": "full_window", "split": "val",
                      "sampling_digest": selection["sampling_digest"], "samples": selection["samples"],
                      "forecast_origins_sha256": selection["forecast_origins_sha256"],
                      "observed_targets_sha256": target_hash.hexdigest(), "observation_masks_sha256": mask_hash.hexdigest(),
                      "all_declared_split_samples": selection["all_validation_samples"],
                      "hash_format": "C_order_sanitized_flow0_targets_and_boolean_mask_bytes; origins_legacy_array_digest"}
        return {"mae": metric.compute(), "global_masked_mae": metric.compute(), "collection": collection,
                "absolute_error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
                "horizon_mae": horizon, "horizon_valid_counts": counts.tolist(),
                "average_horizon_mae": float(np.mean(horizon)) if counts.all() else None,
                "selection": selection, "observed_targets_sha256": target_hash.hexdigest(),
                "observation_masks_sha256": mask_hash.hexdigest(), "finite_predictions_everywhere": True}
    finally:
        model.train(was_training)


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any,
                    data: Any, config: Mapping[str, Any], progress: Mapping[str, Any],
                    selection_metric: Mapping[str, Any] | None = None) -> None:
    payload = {"schema_version": SCHEMA, "sampling_protocol": "full_window",
               "model_class": "StrongReference", "native_model_class": type(model.native).__name__,
               "config": dict(config), "config_digest": canonical_digest(config),
               "model_factory": model.factory_kwargs(), "model_upstream": copy.deepcopy(model.upstream),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": capture_rng_state(), "loaders": {name: getattr(data, name).state_dict() for name in ("train", "val")},
               "data_identity": data.identity, "progress": copy.deepcopy(progress),
               "reader_identity": data.reader.identity, "sampling_identity": data.reader.sampling_identity,
               "statistics_state": data.statistics.state_dict(), "statistics_identity": dict(data.statistics.identity),
               "selection_metric": copy.deepcopy(selection_metric),
               "resume_boundary": "completed_effective_batch_or_epoch", "test_evaluated": False}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint_safe(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"),
                    {"sha256": digest, "size_bytes": path.stat().st_size, "config_digest": payload["config_digest"]})
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if sha256_file(path) != marker["sha256"] or path.stat().st_size != marker["size_bytes"]:
        raise ValueError("full-window checkpoint byte identity differs from its committed marker")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"config", "config_digest", "model_factory", "model_upstream", "model", "optimizer", "scheduler",
                "rng", "loaders", "data_identity", "progress", "selection_metric", "reader_identity",
                "sampling_identity", "statistics_state", "statistics_identity", "model_class"}
    if checkpoint.get("schema_version") != SCHEMA or checkpoint.get("sampling_protocol") != "full_window" or not required <= set(checkpoint):
        raise ValueError("checkpoint is not a complete independent full-window strong checkpoint")
    if checkpoint.get("test_evaluated") is not False or checkpoint["config"].get("sampling_protocol") != "full_window":
        raise ValueError("full-window development checkpoint has an incompatible test or sampling scope")
    if checkpoint["config"].get("schema_version") != SCHEMA:
        raise ValueError("full-window checkpoint config has a different schema")
    if canonical_digest(checkpoint["config"]) != checkpoint["config_digest"] or marker["config_digest"] != checkpoint["config_digest"]:
        raise ValueError("full-window checkpoint recipe identity is inconsistent")
    statistics = FullWindowStatistics.from_state(checkpoint["statistics_state"])
    if dict(statistics.identity) != checkpoint["statistics_identity"] or checkpoint["model_class"] != "StrongReference":
        raise ValueError("full-window checkpoint statistics or real model class are inconsistent")
    return checkpoint


def _check_data_and_model(checkpoint: Mapping[str, Any], model: Any, data: Any, config: Mapping[str, Any]) -> None:
    if checkpoint["config_digest"] != canonical_digest(config) or checkpoint["data_identity"] != data.identity:
        raise ValueError("strict full-window data/config/sampling/statistics identity mismatch")
    if dict(runtime_identity(next(model.parameters()).device)) != config["runtime"]:
        raise ValueError("strict full-window restoration/verification runtime differs")
    if (checkpoint["reader_identity"] != data.reader.identity or checkpoint["sampling_identity"] != data.reader.sampling_identity
            or checkpoint["statistics_identity"] != dict(data.statistics.identity)):
        raise ValueError("strict full-window reader/sampling/statistics identity mismatch")
    if data.identity.get("sampling_protocol") != "full_window" or data.reader.history_policy != "canonical_causal_imputation":
        raise ValueError("strong controls require independent full-window canonical clean histories")
    require_snapshot(config["source_snapshot"])
    _require_model_identity(model, checkpoint)
    factory = checkpoint["model_factory"]
    if (factory["method"], factory["node_count"], factory["flow_mean"], factory["flow_std"]) != (
        config["method"], data.reader.N, data.statistics.flow_mean, data.statistics.flow_std,
    ):
        raise ValueError("native constructor normalization differs from frozen raw train statistics")


def _require_native_normalizer(model: Any, data: Any) -> None:
    if model.flow_mean.item() != np.float32(data.statistics.flow_mean) or model.flow_std.item() != np.float32(data.statistics.flow_std):
        raise ValueError("native model scalar buffers differ from frozen unique train statistics")


def restore_checkpoint(checkpoint: Mapping[str, Any], model: Any, optimizer: Any, scheduler: Any,
                       data: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    _check_data_and_model(checkpoint, model, data, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    _require_native_normalizer(model, data)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for name in ("train", "val"):
        getattr(data, name).load_state_dict(checkpoint["loaders"][name])
    restore_rng_state(checkpoint["rng"])
    progress = copy.deepcopy(checkpoint["progress"])
    if (type(progress["epoch"]) is not int or progress["stage"] not in ("start", "train", "validate")
            or not 0 <= progress["epoch"] <= config["epochs"]):
        raise ValueError("full-window checkpoint has an invalid strict progress cursor")
    batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    if progress["stage"] in ("train", "validate"):
        committed = progress["partial_epoch"]["batches"]
        cursor = min(committed * data.train.bs, data.train.size)
        if (type(committed) is not int or not 0 <= committed <= batches or data.train.cursor != cursor
                or data.train.epoch != progress["epoch"]
                or progress["stage"] == "validate" and committed != batches):
            raise ValueError("strict loader cursor/epoch disagrees with committed optimization progress")
    return progress


def train_strong(*, model: Any, data: Any, device: torch.device, config: Mapping[str, Any],
                 output: Path, checkpoint: Mapping[str, Any] | None = None,
                 max_wallclock_seconds: float | None = None) -> dict[str, Any]:
    require_snapshot(config["source_snapshot"])
    batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    limit = min(data.train.size, batches * data.train.bs)
    optimizer, scheduler = create_reference_optimizer(model, config, batches)
    progress = {"epoch": 0, "stage": "start", "best_epoch": None, "best_val_mae": None,
                "updates": 0, "history": [], "partial_epoch": empty_partial(), "elapsed_seconds": 0.0}
    if checkpoint is not None:
        progress = restore_checkpoint(checkpoint, model, optimizer, scheduler, data, config)
    stop, handlers = {"requested": False}, {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            handlers[signum] = signal.signal(signum, lambda *_: stop.__setitem__("requested", True))
        except ValueError:
            pass
    started, previous_elapsed = time.monotonic(), progress["elapsed_seconds"]

    def stopped() -> bool:
        return stop["requested"] or (max_wallclock_seconds is not None and time.monotonic() - started >= max_wallclock_seconds)

    def save_last() -> None:
        progress["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)

    def stopped_result() -> dict[str, Any]:
        save_last()
        result = {"status": "stopped_resumable_fullwindow", "completed_epochs": progress["epoch"],
                  "stage": progress["stage"], "loader_cursor": data.train.cursor,
                  "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config),
                  "sampling_protocol": "full_window", "test_evaluated": False}
        atomic_json(output / "run_state.json", result)
        return result

    try:
        while progress["epoch"] < config["epochs"]:
            if progress["stage"] == "start":
                data.train.start_epoch(progress["epoch"])
                progress["partial_epoch"] = empty_partial()
                progress["stage"] = "train"
            while progress["stage"] == "train" and data.train.cursor < limit:
                if stopped():
                    return stopped_result()
                tick = time.monotonic()
                batch = data.train.get_next()
                if batch is None:
                    raise RuntimeError("training loader exhausted before the declared full-window batch count")
                step = optimize_reference(model, optimizer, scheduler, tensors_from_batch(batch, device), config)
                partial = progress["partial_epoch"]
                for name, source in (("error_sum", "error_sum"), ("valid_count", "valid_count"), ("objective_sum", "objective")):
                    partial[name] += step[source]
                partial["optimized_batches"] += int(step["updated"])
                if step["updated"]:
                    partial["learning_rate_last_update"] = step["learning_rate_used"]
                partial["batches"] += 1
                partial["seconds"] += time.monotonic() - tick
                progress["updates"] += int(step["updated"])
            progress["stage"] = "validate"
            if stopped():
                return stopped_result()
            validation = evaluate_strong(model, data, device, microbatch_size=config["microbatch_size"], max_batches=config["max_val_batches"])
            partial = progress["partial_epoch"]
            if not partial["valid_count"]:
                raise ValueError("full-window training epoch has no observed targets")
            used_lr = partial.get("learning_rate_last_update")
            if config["training_recipe"]["scheduler_step_unit"] == "completed_epoch":
                scheduler.step()
            progress["epoch"] += 1
            progress["stage"] = "start"
            row = {"epoch": progress["epoch"], "train_mae": partial["error_sum"] / partial["valid_count"],
                   "train_valid_count": partial["valid_count"], "train_batches": partial["batches"],
                   "validation": validation, "learning_rate_used": used_lr,
                   "learning_rate_next": float(optimizer.param_groups[0]["lr"]), "updates": progress["updates"]}
            progress["history"].append(row)
            if progress["best_val_mae"] is None or validation["global_masked_mae"] < progress["best_val_mae"]:
                progress["best_val_mae"], progress["best_epoch"] = validation["global_masked_mae"], progress["epoch"]
                save_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data,
                                config=config, progress=progress, selection_metric=validation)
            save_last()
            write_history(output, progress["history"])
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        require_snapshot(config["source_snapshot"])
        result = {"status": "fullwindow_strong_training_completed_validation_pending_independent_verification",
                  "completed_epochs": progress["epoch"], "best_epoch": progress["best_epoch"],
                  "best_val_mae": progress["best_val_mae"], "updates": progress["updates"],
                  "best_validation": progress["history"][progress["best_epoch"] - 1]["validation"],
                  "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config),
                  "sampling_protocol": "full_window", "epoch_budget_label": config["training_recipe"]["budget_label"],
                  "native_convergence_reproduced": False, "test_evaluated": False}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def verify_strong(*, checkpoint_path: Path, data: Any, device: torch.device, output: Path,
                  runtime: Mapping[str, Any], bist_checkout: Path | None = None) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if os.getpid() == config["training_process_id"]:
        raise PermissionError("independent verification must use another process")
    if dict(runtime) != config["runtime"]:
        raise ValueError("independent verification runtime differs from training")
    expected = checkpoint["selection_metric"]
    if expected is None or expected["selection"] != validation_identity(data, config["max_val_batches"]):
        raise ValueError("independent verification requires a selected checkpoint and its exact collection")
    factory = checkpoint["model_factory"] | {"device": str(device)}
    if bist_checkout is not None:
        factory["bist_checkout"] = bist_checkout
    model = make_strong_reference(**factory)
    _check_data_and_model(checkpoint, model, data, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    _require_native_normalizer(model, data)
    actual = evaluate_strong(model, data, device, microbatch_size=config["microbatch_size"], max_batches=config["max_val_batches"])
    for key in ("valid_count", "selection", "collection", "observed_targets_sha256", "observation_masks_sha256", "horizon_valid_counts"):
        if actual[key] != expected[key]:
            raise ValueError("independent full-window validation collection differs")
    if not math.isclose(actual["global_masked_mae"], expected["global_masked_mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent full-window global validation MAE differs")
    for left, right in zip(actual["horizon_mae"], expected["horizon_mae"], strict=True):
        if (left is None) != (right is None) or (left is not None and not math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-5)):
            raise ValueError("independent full-window horizon validation MAE differs")
    formal = config["claim_scope"] == FORMAL_SCOPE
    if formal and not actual["selection"]["all_validation_samples"]:
        raise PermissionError("formal verification cannot use a validation subset")
    summary_path = checkpoint_path.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "fullwindow_strong_training_completed_validation_pending_independent_verification"
            or summary["config_digest"] != checkpoint["config_digest"] or summary["completed_epochs"] != config["epochs"]
            or summary["best_epoch"] != checkpoint["progress"]["best_epoch"]
            or summary["test_evaluated"] is not False or summary["best_validation"] != expected):
        raise PermissionError("verification requires completed frozen training and this exact best checkpoint")
    require_snapshot(config["source_snapshot"])
    result = {"schema_version": SCHEMA,
              "status": "verified_fullwindow_strong_complete_validation_capped" if formal else "verified_fullwindow_strong_smoke_only",
              "claim_scope": config["claim_scope"], "validation": actual, "sampling_protocol": "full_window",
              "sampling_identity": data.reader.sampling_identity, "statistics_identity": dict(data.statistics.identity),
              "reader_identity": data.reader.identity, "model_class": "StrongReference",
              "data_identity_digest": data.identity["identity_digest"], "checkpoint_sha256": sha256_file(checkpoint_path),
              "summary_sha256": sha256_file(summary_path), "config_digest": checkpoint["config_digest"],
              "training_process_id": config["training_process_id"], "verification_process_id": os.getpid(),
              "source_files_digest": config["source_snapshot"]["files_digest"],
              "source_snapshot_files_digest": config["source_snapshot"]["files_digest"],
              "native_convergence_reproduced": False, "test_evaluated": False}
    atomic_json(output / "verification.json", result)
    return result


def _next_full_batch(loader: Any) -> tuple[Mapping[str, Any], int]:
    if loader.size < loader.bs:
        raise ValueError("calibration requires a complete effective batch in each split")
    skipped = 0
    while True:
        batch = loader.get_next()
        if batch is None:
            loader.start_epoch()
            continue
        if len(batch["flow_hist"]) == loader.bs:
            return batch, skipped
        skipped += 1


def calibrate_strong(*, model: Any, data: Any, device: torch.device, config: Mapping[str, Any],
                     warmup_batches: int, timed_batches: int, output: Path) -> dict[str, Any]:
    if type(warmup_batches) is not int or warmup_batches < 0 or type(timed_batches) is not int or timed_batches <= 0:
        raise ValueError("calibration requires nonnegative warmup and positive timed counts")
    if data.train.bs != config["batch_size"] or data.val.bs != config["batch_size"]:
        raise ValueError("calibration loaders differ from the frozen effective batch")
    require_snapshot(config["source_snapshot"])
    optimizer, scheduler = create_reference_optimizer(model, config, data.train.num_batch)
    was_training, phases = model.training, {}
    try:
        for phase, loader in (("train", data.train), ("val", data.val)):
            loader.start_epoch()
            model.train(phase == "train")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            durations, sizes, skipped_tails, valid_count = [], [], 0, 0
            for number in range(warmup_batches + timed_batches):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                batch, skipped = _next_full_batch(loader)
                tensors = tensors_from_batch(batch, device)
                if phase == "train":
                    observed = optimize_reference(model, optimizer, scheduler, tensors, config)["valid_count"]
                else:
                    with torch.no_grad():
                        observed = _calibration_validation_batch(model, tensors, config["microbatch_size"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - tick
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("calibration produced an invalid full-batch duration")
                skipped_tails += skipped
                if number >= warmup_batches:
                    durations.append(elapsed)
                    sizes.append(len(batch["flow_hist"]))
                    valid_count += int(observed)
            phases[phase] = {"warmup_batches": warmup_batches, "timed_batches": len(durations),
                             "timed_batch_seconds": durations, "timed_effective_batch_sizes": sizes,
                             "mean_batch_seconds": float(np.mean(durations)), "p90_batch_seconds": float(np.percentile(durations, 90)),
                             "timed_valid_count": valid_count, "skipped_retained_tails": skipped_tails,
                             "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                             "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
    finally:
        model.train(was_training)
    mean_estimate = config["requested_epochs"] * (phases["train"]["mean_batch_seconds"] * data.train.num_batch + phases["val"]["mean_batch_seconds"] * data.val.num_batch)
    p90_estimate = config["requested_epochs"] * (phases["train"]["p90_batch_seconds"] * data.train.num_batch + phases["val"]["p90_batch_seconds"] * data.val.num_batch)
    require_snapshot(config["source_snapshot"])
    result = {"schema_version": SCHEMA, "status": "calibrated_fullwindow_cuda_full_effective_batches" if device.type == "cuda" else "calibrated_fullwindow_cpu_engineering_only",
              "claim_scope": "calibration_only", "sampling_protocol": "full_window", "cuda_budget_eligible": device.type == "cuda",
              "device_type": device.type, "phases": phases, "budget_epochs": config["requested_epochs"],
              "train_num_batches": data.train.num_batch, "val_num_batches": data.val.num_batch,
              "estimated_train_val_seconds": mean_estimate, "p90_estimated_train_val_seconds": p90_estimate,
              "estimate_formula": "epochs * (train_full_batch_seconds * train_batches + validation_full_batch_seconds * validation_batches)",
              "timing_scope": "data_loading_transfer_actual_main_masked_objective_optimizer_or_complete_validation_reduction",
              "warmup_excluded_from_timing": True, "tail_retained_in_real_training_and_validation": True,
              "batch_normalization_scope": config["training_recipe"]["batch_normalization_scope"],
              "prediction_error_used_for_configuration_selection": False,
              "data_identity": data.identity, "training_recipe": config["training_recipe"],
              "runtime": config["runtime"], "config_digest": canonical_digest(config),
              "native_convergence_reproduced": False, "test_evaluated": False}
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "verify", "calibrate"), default="train")
    parser.add_argument("--method", choices=METHODS, default="staeformer")
    parser.add_argument("--recipe", choices=RECIPES, default="native")
    parser.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    parser.add_argument("--epochs", choices=(30, 100), type=int, default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--allow-bn-microbatch-override", action="store_true")
    parser.add_argument("--seed", choices=(42, 1, 2), type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bist-checkout", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-wallclock-seconds", type=float)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--timed-batches", type=int, default=20)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--lazy", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.threads <= 0 or args.smoke_epochs <= 0 or args.warmup_batches < 0 or args.timed_batches <= 0:
        raise ValueError("thread/smoke/calibration counts are invalid")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("verify requires a selected full-window --checkpoint")
    if args.checkpoint is not None and args.mode != "verify":
        raise ValueError("--checkpoint is reserved for independent verify")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("strict resume requires train mode and last.pt")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or not args.smoke):
            raise ValueError("subset limits require a positive explicit smoke-only run")
    if args.device.startswith("cpu") and not args.smoke and args.resume is None and args.mode != "verify":
        raise ValueError("CPU is restricted to explicit engineering smoke; CUDA fallback is forbidden")
    if args.max_wallclock_seconds is not None and (not math.isfinite(args.max_wallclock_seconds) or args.max_wallclock_seconds <= 0):
        raise ValueError("wallclock limit must be finite and positive")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    validate_arguments(args)
    previous = load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous else None
    if saved:
        mutable = {"mode", "device", "bist_checkout", "data_dir", "graph_dir", "output_dir", "checkpoint", "resume", "max_wallclock_seconds", "warmup_batches", "timed_batches"}
        for name, value in vars(args).items():
            option = "--" + name.replace("_", "-")
            expected = saved.get("requested_epochs") if name == "epochs" else saved.get(name)
            if name not in mutable and any(item == option or item.startswith(option + "=") for item in arguments) and value != expected:
                raise ValueError(f"strict full-window resume/verify cannot override {option}")
    device = configure_runtime(args.device, saved["seed"] if saved else args.seed, saved["threads"] if saved else args.threads)
    runtime = runtime_identity(device)
    if saved and runtime != saved["runtime"]:
        raise ValueError("strict full-window resume/verify runtime differs from training")
    if device.type == "cpu" and (saved["claim_scope"] if saved else "smoke_only" if args.smoke else "formal") != "smoke_only":
        raise PermissionError("CPU cannot be classified as formal full-window development")
    recipe = saved["training_recipe"] if saved else fullwindow_recipe(args.method, args.recipe, region=args.region, epochs=args.epochs,
                                                                     batch_size=args.batch_size, microbatch_size=args.microbatch_size,
                                                                     allow_bn_microbatch_override=args.allow_bn_microbatch_override)
    data = load_training_data(region=saved["region"] if saved else args.region,
                              data_dir=args.data_dir, graph_dir=args.graph_dir, batch_size=recipe["batch_size"],
                              seed=saved["seed"] if saved else args.seed, preload=not (saved["lazy"] if saved else args.lazy))
    if data.identity.get("sampling_protocol") != "full_window" or data.reader.history_policy != "canonical_causal_imputation":
        raise PermissionError("full-window controls cannot use event data or observed-zero clean history")
    if not args.smoke and not saved and not data.reader.contract.formal:
        raise PermissionError("formal full-window runs require the exact frozen raw-material release")
    output = args.output_dir or ((args.resume or args.checkpoint).parent if previous else None)
    if output is None:
        raise ValueError("a new full-window run requires an independent output directory")
    if args.resume is not None and output.resolve() != args.resume.parent.resolve():
        raise ValueError("strict resume must retain its original output directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_strong(checkpoint_path=args.checkpoint, data=data, device=device, output=output, runtime=runtime, bist_checkout=args.bist_checkout)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return result
    factory = previous["model_factory"] if previous else {"method": args.method, "node_count": data.reader.N,
               "flow_mean": data.statistics.flow_mean, "flow_std": data.statistics.flow_std, "bist_checkout": args.bist_checkout}
    if previous and args.bist_checkout is not None:
        factory = factory | {"bist_checkout": args.bist_checkout}
    model = make_strong_reference(**(factory | {"device": str(device)}))
    if saved:
        config = saved
    else:
        config = {name: getattr(args, name) for name in ("method", "recipe", "region", "seed", "threads", "lazy", "smoke", "smoke_epochs", "max_train_batches", "max_val_batches", "allow_bn_microbatch_override", "warmup_batches", "timed_batches")}
        config.update(schema_version=SCHEMA, sampling_protocol="full_window", model_class="StrongReference",
                      epochs=args.smoke_epochs if args.smoke else args.epochs, requested_epochs=args.epochs,
                      batch_size=recipe["batch_size"], microbatch_size=recipe["microbatch_size"], training_recipe=recipe,
                      claim_scope="calibration_only" if args.mode == "calibrate" else "smoke_only" if args.smoke else FORMAL_SCOPE,
                      training_process_id=os.getpid(), runtime=runtime, data_identity=data.identity,
                      source_snapshot=source_snapshot(), model_upstream=model.upstream,
                      history_policy=data.reader.history_policy, precision="FP32", test_loader_constructed=False,
                      test_evaluated=False, early_stopping=False, native_convergence_reproduced=False,
                      validation_selection=validation_identity(data, args.max_val_batches),
                      scalar_fit_precision="float64", legacy27_scalar_fit_precision="float32")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
        atomic_json(output / "data_identity.json", data.identity)
        atomic_json(output / "source_snapshot.json", config["source_snapshot"])
    if args.mode == "calibrate":
        result = calibrate_strong(model=model, data=data, device=device, config=config,
                                  warmup_batches=args.warmup_batches, timed_batches=args.timed_batches, output=output)
    else:
        result = train_strong(model=model, data=data, device=device, config=config, output=output,
                              checkpoint=previous, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
