#!/usr/bin/env python3
"""Independent full-window NDR and SharedGraphNDR train/validation development."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
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
from ndr_research.data import canonical_digest, sha256_file
from ndr_research.fullwindow_data import FullWindowStatistics, legacy_array_digest
from ndr_research.fullwindow_training import (
    METHODS, FullWindowTrainingData, batch_tensors, factory_from_data, load_training_data,
    model_forward, objective_recipe, optimize_batch, rebuild_model, saved_factory, validation_identity,
)
from scripts.train_ndr_research import (
    atomic_json, capture_rng_state, create_optimizer, empty_partial, reject_identity_overrides,
    require_snapshot, restore_rng_state, source_snapshot as checkout_snapshot, write_history,
)
from scripts.train_ndr_transfer import checkpoint_safe, configure_transfer_runtime, graph_chunk, transfer_runtime_identity

CHECKPOINT_SCHEMA = "fullwindow_ndr_checkpoint_v1"
FORMAL_SCOPE = "fullwindow_validation_development_only"


def source_snapshot() -> dict[str, Any]:
    complete = checkout_snapshot(ROOT)
    names = {Path(__file__).resolve().relative_to(ROOT).as_posix()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT)
            if relative.parts[0] not in {"tests", "outputs"}:
                names.add(relative.as_posix())
    names.update({"ndr_research/fullwindow_data.py", "ndr_research/fullwindow_training.py",
                  "ndr_research/models.py", "ndr_research/shared_graph.py", "ndr_research/graph_loss.py",
                  "dist_net/protocol.py", "scripts/train_ndr_research.py", "scripts/train_ndr_transfer.py"})
    files = {name: record for name, record in complete["files"].items() if name in names}
    for name in names:
        path = ROOT / name
        files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {"files": files, "files_digest": canonical_digest(files), "git": complete["git"],
            "scope": "fullwindow_sampling_model_objective_and_imported_python_dependencies"}


def loader_batches(loader: Any) -> int:
    return (loader.cursor + loader.batch_size - 1) // loader.batch_size


def evaluate(model: Any, data: FullWindowTrainingData, device: torch.device, *,
             method: str, microbatch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    metric = MaskedMAEAccumulator()
    origins = []
    targets_sha, masks_sha = hashlib.sha256(), hashlib.sha256()
    horizon_errors, horizon_counts = np.zeros(12, np.float64), np.zeros(12, np.int64)
    data.val.start_epoch()
    batches = min(data.val.num_batch, max_batches or data.val.num_batch)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch = data.val.get_next()
                if batch is None:
                    raise ValueError("full-window validation exhausted before its frozen collection")
                origins.extend(batch["sample_start"].tolist())
                tensors = batch_tensors(batch, device)
                targets_sha.update(tensors["target"].cpu().contiguous().numpy().tobytes())
                masks_sha.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
                for low in range(0, tensors["target"].shape[0], microbatch_size):
                    chunk = graph_chunk(tensors, low, low + microbatch_size)
                    prediction = model_forward(model, chunk, method)
                    metric.update(prediction, chunk["target"], chunk["target_mask"])
                    errors = torch.where(chunk["target_mask"], (prediction - chunk["target"]).abs(), 0.0)
                    horizon_errors += errors.double().sum(dim=(0, 1)).cpu().numpy()
                    horizon_counts += chunk["target_mask"].sum(dim=(0, 1)).cpu().numpy()
        selection = validation_identity(data, max_batches)
        if legacy_array_digest(np.asarray(origins, np.int64)) != selection["forecast_origins_sha256"]:
            raise ValueError("full-window validation origins differ from their frozen selection")
        horizons = [float(error / count) if count else None for error, count in zip(horizon_errors, horizon_counts)]
        collection = {"sampling_protocol": "full_window", "split": "val", "sampling_digest": selection["sampling_digest"],
                      "samples": selection["samples"], "forecast_origins_sha256": selection["forecast_origins_sha256"],
                      "observed_targets_sha256": targets_sha.hexdigest(), "observation_masks_sha256": masks_sha.hexdigest(),
                      "all_declared_split_samples": selection["all_validation_samples"],
                      "hash_format": "C_order_sanitized_flow0_targets_and_boolean_mask_bytes; origins_legacy_array_digest"}
        return {"mae": metric.compute(), "global_masked_mae": metric.compute(),
                "absolute_error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
                "horizon_mae": horizons, "horizon_valid_counts": horizon_counts.tolist(),
                "average_horizon_mae": float(np.mean(horizons)) if horizon_counts.all() else None,
                "selection": selection, "collection": collection, "batches": batches,
                "observed_targets_sha256": targets_sha.hexdigest(), "observation_masks_sha256": masks_sha.hexdigest(),
                "finite_predictions_everywhere": True}
    finally:
        model.train(was_training)


def _loader_checkpoint(loader: Any) -> dict[str, Any]:
    return checkpoint_safe(loader.state_dict())


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any,
                    data: FullWindowTrainingData, config: Mapping[str, Any],
                    progress: Mapping[str, Any], selection_metric: Mapping[str, Any] | None) -> None:
    payload = {"schema_version": CHECKPOINT_SCHEMA, "model_class": type(model).__name__,
               "model_factory": saved_factory(model, config["method"]), "config": dict(config),
               "config_digest": canonical_digest(config), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": capture_rng_state(),
               "loaders": {name: _loader_checkpoint(getattr(data, name)) for name in ("train", "val")},
               "statistics_state": checkpoint_safe(data.statistics.state_dict()),
               "statistics_identity": dict(data.statistics.identity), "reader_identity": data.reader.identity,
               "sampling_identity": data.reader.sampling_identity, "data_identity": dict(data.identity),
               "progress": copy.deepcopy(progress), "selection_metric": copy.deepcopy(selection_metric),
               "sampling_protocol": "full_window", "resume_boundary": "completed_effective_batch_or_epoch",
               "test_numeric_api_available": False, "test_evaluated": False}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"),
                    {"sha256": digest, "size_bytes": path.stat().st_size, "config_digest": payload["config_digest"]})
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if path.stat().st_size != marker["size_bytes"] or sha256_file(path) != marker["sha256"]:
        raise ValueError("full-window checkpoint byte identity mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "model_factory", "model_class", "config", "config_digest", "optimizer", "scheduler", "rng",
                "loaders", "progress", "data_identity", "reader_identity", "sampling_identity", "statistics_state", "statistics_identity", "selection_metric"}
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or not required.issubset(checkpoint):
        raise ValueError("checkpoint is not a complete independent full-window NDR checkpoint")
    config = checkpoint["config"]
    if (canonical_digest(config) != checkpoint["config_digest"] or marker["config_digest"] != checkpoint["config_digest"]):
        raise ValueError("full-window checkpoint frozen configuration differs")
    if (checkpoint.get("sampling_protocol") != "full_window" or checkpoint.get("test_numeric_api_available") is not False
            or checkpoint.get("test_evaluated") is not False):
        raise ValueError("full-window checkpoint has another sampling protocol or test access")
    for field in ("data_identity", "reader_identity", "sampling_identity", "statistics_identity", "model_factory", "model_class"):
        if checkpoint[field] != config.get(field):
            raise ValueError(f"full-window checkpoint configuration differs from {field}")
    statistics = FullWindowStatistics.from_state(checkpoint["statistics_state"])
    if dict(statistics.identity) != checkpoint["statistics_identity"]:
        raise ValueError("full-window numeric statistics differ from their frozen identity")
    reader = checkpoint["reader_identity"]
    sampling = checkpoint["sampling_identity"]
    if (reader["sampling_protocol"] != "full_window" or reader["role"] != "source"
            or reader["history_policy"] != "canonical_causal_imputation" or reader["sampling"] != sampling
            or statistics.identity["sampling_digest"] != sampling["sampling_digest"]
            or statistics.identity["reader_identity_digest"] != reader["identity_digest"]
            or statistics.identity["fit_start"] != 0 or statistics.identity["fit_end"] != sampling["train_end"]):
        raise ValueError("full-window checkpoint statistics/sampling/reader scope differs")
    model = rebuild_model(config["method"], checkpoint["model_factory"])
    if type(model).__name__ != checkpoint["model_class"]:
        raise ValueError("full-window checkpoint does not contain its declared real model class")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def restore_training(checkpoint: Mapping[str, Any], model: Any, optimizer: Any, scheduler: Any,
                     data: FullWindowTrainingData, config: Mapping[str, Any]) -> dict[str, Any]:
    if (checkpoint["config_digest"] != canonical_digest(config) or checkpoint["data_identity"] != dict(data.identity)
            or checkpoint["statistics_identity"] != dict(data.statistics.identity)
            or checkpoint["reader_identity"] != data.reader.identity or checkpoint["sampling_identity"] != data.reader.sampling_identity
            or checkpoint["model_factory"] != saved_factory(model, config["method"])
            or config["runtime"] != transfer_runtime_identity(next(model.parameters()).device)):
        raise ValueError("full-window resume method/statistics/material/sampling/runtime differs")
    require_snapshot(config["source_snapshot"])
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for name in ("train", "val"):
        getattr(data, name).load_state_dict(checkpoint["loaders"][name])
    restore_rng_state(checkpoint["rng"])
    progress = copy.deepcopy(checkpoint["progress"])
    batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    if (type(progress["epoch"]) is not int or not 0 <= progress["epoch"] <= config["epochs"]
            or progress["stage"] not in {"start", "train", "validate"}):
        raise ValueError("full-window restored epoch/stage is invalid")
    if progress["stage"] in {"train", "validate"}:
        expected_cursor = min(progress["partial_epoch"]["batches"] * data.train.batch_size, data.train.size)
        if (data.train.cursor != expected_cursor or not 0 <= loader_batches(data.train) <= batches
                or data.train.epoch != progress["epoch"]
                or progress["stage"] == "validate" and loader_batches(data.train) != batches):
            raise ValueError("full-window resume loader cursor/epoch differs from committed progress")
    return progress


def train_run(*, model: Any, data: FullWindowTrainingData, device: torch.device,
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
                data.train.start_epoch(epoch=progress["epoch"])
                progress["partial_epoch"] = empty_partial()
                progress["stage"] = "train"
            while progress["stage"] == "train" and loader_batches(data.train) < train_batches:
                if stopping():
                    persist_last()
                    result = {"status": "stopped_resumable", "reason": "signal" if stop["requested"] else "wallclock",
                              "completed_epochs": progress["epoch"], "next_batch": loader_batches(data.train),
                              "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config)}
                    atomic_json(output / "run_state.json", result)
                    return result
                step_started = time.monotonic()
                batch = data.train.get_next()
                tensors = batch_tensors(batch, device)
                step = optimize_batch(model, optimizer, scheduler, tensors, config)
                partial = progress["partial_epoch"]
                partial["error_sum"] += step["absolute_error_sum"]
                partial["valid_count"] += step["valid_count"]
                partial["objective_sum"] += step["objective"]
                partial["optimized_batches"] += int(step["updated"])
                partial["batches"] += 1
                partial["seconds"] += time.monotonic() - step_started
                progress["updates"] += int(step["updated"])
                if loader_batches(data.train) == 1 or loader_batches(data.train) % config["log_interval"] == 0:
                    print(json.dumps({"epoch": progress["epoch"] + 1, "batch": loader_batches(data.train),
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
            validation = evaluate(model, data, device, method=config["method"], microbatch_size=config["microbatch_size"],
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
                  "sampling_protocol": "full_window", "method": config["method"],
                  "test_evaluated": False, "access_audit": data.reader.access_audit(), "elapsed_seconds": progress["elapsed_seconds"]}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def verify_checkpoint(*, checkpoint_path: Path, data: FullWindowTrainingData, device: torch.device,
                      output: Path, current_runtime: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if config["training_process_id"] == os.getpid():
        raise ValueError("full-window verification requires an independent process")
    require_snapshot(config["source_snapshot"])
    if (checkpoint["data_identity"] != dict(data.identity) or checkpoint["statistics_identity"] != dict(data.statistics.identity)
            or checkpoint["reader_identity"] != data.reader.identity or checkpoint["sampling_identity"] != data.reader.sampling_identity
            or checkpoint["model_factory"] != factory_from_data(config["method"], data, graph_hidden_dim=config["graph_hidden_dim"])):
        raise ValueError("independent full-window data/statistics/material/model identity differs")
    if dict(current_runtime) != config["runtime"]:
        raise ValueError("full-window verification runtime differs from training")
    selected = checkpoint["selection_metric"]
    if selected is None or validation_identity(data, config["max_val_batches"]) != selected["selection"]:
        raise ValueError("verification requires the selected checkpoint's exact full-window validation collection")
    model = rebuild_model(config["method"], checkpoint["model_factory"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate(model, data, device, method=config["method"], microbatch_size=config["microbatch_size"], max_batches=config["max_val_batches"])
    for key in ("valid_count", "selection", "collection", "observed_targets_sha256", "observation_masks_sha256"):
        if actual[key] != selected[key]:
            raise ValueError("full-window verification did not reproduce exact observations and sampling")
    if not math.isclose(actual["global_masked_mae"], selected["global_masked_mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("full-window selected MAE did not reproduce")
    status = "verified_fullwindow_ndr_checkpoint_complete_validation"
    if config["claim_scope"] == "smoke_only":
        status = "verified_fullwindow_ndr_smoke_complete_validation" if actual["selection"]["all_validation_samples"] else "verified_fullwindow_ndr_smoke_validation_subset"
    elif config["claim_scope"] != FORMAL_SCOPE or not actual["selection"]["all_validation_samples"]:
        raise PermissionError("formal full-window verification needs complete full-window validation")
    result = {"schema_version": 1, "status": status, "method": config["method"],
              "model_class": checkpoint["model_class"], "model_factory": checkpoint["model_factory"],
              "checkpoint_sha256": sha256_file(checkpoint_path), "config_digest": checkpoint["config_digest"],
              "source_files_digest": config["source_snapshot"]["files_digest"], "data_identity": dict(data.identity),
              "statistics_identity": dict(data.statistics.identity), "reader_identity": data.reader.identity,
              "sampling_identity": data.reader.sampling_identity, "sampling_protocol": "full_window",
              "validation": actual, "expected_validation_mae": selected["global_masked_mae"],
              "absolute_mae_difference": abs(actual["global_masked_mae"] - selected["global_masked_mae"]),
              "training_process_id": config["training_process_id"], "verification_process_id": os.getpid(),
              "claim_scope": config["claim_scope"], "test_evaluated": False, "access_audit": data.reader.access_audit()}
    atomic_json(output / "verification.json", result)
    return result


def _next_full_batch(loader: Any) -> Mapping[str, Any]:
    if loader.size < loader.batch_size:
        raise ValueError("full-window calibration needs complete effective batches")
    for _ in range(loader.num_batch + 1):
        if loader.cursor >= loader.size:
            loader.start_epoch()
        batch = loader.get_next()
        if batch is not None and batch["flow_hist"].shape[0] == loader.batch_size:
            return batch
    raise ValueError("full-window calibration could not find a complete batch")


def calibrate(model: Any, data: FullWindowTrainingData, device: torch.device, config: Mapping[str, Any], *,
              warmup: int, timed: int, output: Path) -> dict[str, Any]:
    optimizer, scheduler = create_optimizer(model, config, data.train.num_batch)
    timings, objectives = {}, []
    for split, loader in (("train", data.train), ("val", data.val)):
        loader.start_epoch()
        durations = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for number in range(warmup + timed):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            tensors = batch_tensors(_next_full_batch(loader), device)
            if split == "train":
                step = optimize_batch(model, optimizer, scheduler, tensors, config)
                objectives.append(step["parts"])
            else:
                model.eval()
                metric = MaskedMAEAccumulator()
                horizon_errors, horizon_counts = np.zeros(12, np.float64), np.zeros(12, np.int64)
                targets_sha, masks_sha = hashlib.sha256(), hashlib.sha256()
                targets_sha.update(tensors["target"].cpu().contiguous().numpy().tobytes())
                masks_sha.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
                with torch.no_grad():
                    for low in range(0, tensors["target"].shape[0], config["microbatch_size"]):
                        chunk = graph_chunk(tensors, low, low + config["microbatch_size"])
                        prediction = model_forward(model, chunk, config["method"])
                        metric.update(prediction, chunk["target"], chunk["target_mask"])
                        errors = torch.where(chunk["target_mask"], (prediction - chunk["target"]).abs(), 0.0)
                        horizon_errors += errors.double().sum(dim=(0, 1)).cpu().numpy()
                        horizon_counts += chunk["target_mask"].sum(dim=(0, 1)).cpu().numpy()
                if metric.valid_count:
                    metric.compute()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if number >= warmup:
                durations.append(time.perf_counter() - tick)
        timings[split] = {"mean_batch_seconds": float(np.mean(durations)), "full_effective_batch_size": loader.batch_size,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                          "warmup_batches": warmup, "timed_batches": timed,
                          "validation_metric_and_target_hash_work_included": split == "val"}
    result = {"status": "calibrated_fullwindow_ndr_engineering_only", "method": config["method"],
              "claim_scope": "calibration_only", "splits": timings, "actual_train_objective_parts": objectives,
              "objective_recipe": config["objective_recipe"], "sampling_protocol": "full_window",
              "estimated_full_train_val_seconds": config["epochs"] * sum(timings[name]["mean_batch_seconds"] * getattr(data, name).num_batch for name in ("train", "val")),
              "timing_includes_reader": True, "config_digest": canonical_digest(config), "test_evaluated": False,
              "access_audit": data.reader.access_audit()}
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate", "verify"), default="train")
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    parser.add_argument("--teacher", choices=("original", "projected"))
    parser.add_argument("--graph-hidden-dim", type=int, choices=(64, 240), default=64)
    parser.add_argument("--epochs", type=int, choices=(30, 100), default=30)
    parser.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-wallclock-seconds", type=float)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lazy", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    objective_recipe(args.method, args.teacher)
    for name in ("batch_size", "microbatch_size", "threads", "timed_batches", "log_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.microbatch_size > args.batch_size or args.warmup_batches < 0 or args.epochs not in (30, 100):
        raise ValueError("microbatch, warmup or fixed epoch budget is invalid")
    for name in ("max_train_batches", "max_val_batches", "max_wallclock_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if args.mode == "verify" and (args.checkpoint is None or args.max_train_batches is not None or args.max_val_batches is not None):
        raise ValueError("verification requires best checkpoint and its original subset identity")
    if args.mode != "verify" and args.checkpoint is not None:
        raise ValueError("checkpoint is only for independent full-window verification")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("continuation requires train mode and last.pt")
    if args.mode == "train" and torch.device(args.device).type == "cpu" and (args.max_train_batches is None or args.max_val_batches is None):
        raise ValueError("CPU full-window training requires explicit bounded train and validation smoke subsets")
    for name in ("learning_rate", "gradient_clip", "eta_min"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    previous = load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous else None
    if saved is not None:
        reject_identity_overrides(args, saved, arguments)
        for name in ("region", "method", "teacher", "epochs", "graph_hidden_dim", "batch_size", "microbatch_size"):
            option = "--" + name.replace("_", "-")
            if not any(value == option or value.startswith(option + "=") for value in arguments):
                setattr(args, name, saved[name])
        if args.mode == "train":
            args.max_train_batches, args.max_val_batches = saved["max_train_batches"], saved["max_val_batches"]
        if args.lazy and not saved["lazy"]:
            raise ValueError("full-window continuation cannot change the frozen preload policy")
    elif args.teacher is None:
        args.teacher = "projected" if args.method == "projected_oracle" else "original"
    validate_arguments(args)
    device = configure_transfer_runtime(args.device, saved["seed"] if saved else args.seed, saved["threads"] if saved else args.threads)
    runtime = transfer_runtime_identity(device)
    if saved is not None:
        require_snapshot(saved["source_snapshot"])
        if runtime != saved["runtime"]:
            raise ValueError("strict full-window verification/continuation requires the original runtime")
    data = load_training_data(region=saved["region"] if saved else args.region, data_dir=args.data_dir,
                              graph_dir=args.graph_dir, batch_size=saved["batch_size"] if saved else args.batch_size,
                              seed=saved["seed"] if saved else args.seed, preload=not (saved["lazy"] if saved else args.lazy))
    output = args.output_dir or ((args.resume or args.checkpoint).parent if previous else None)
    if output is None:
        raise ValueError("a new full-window run requires an isolated output directory")
    output = output.resolve()
    if args.resume and output != args.resume.resolve().parent:
        raise ValueError("full-window continuation must retain its original output directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_checkpoint(checkpoint_path=args.checkpoint, data=data, device=device, output=output, current_runtime=runtime)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    factory = saved["model_factory"] if saved else factory_from_data(args.method, data, graph_hidden_dim=args.graph_hidden_dim)
    model = rebuild_model(saved["method"] if saved else args.method, factory, device)
    if saved is None:
        config = {name: getattr(args, name) for name in ("method", "teacher", "region", "graph_hidden_dim", "epochs", "seed",
                  "batch_size", "microbatch_size", "learning_rate", "weight_decay", "gradient_clip", "eta_min",
                  "max_train_batches", "max_val_batches", "threads", "log_interval", "lazy")}
        config.update(schema_version=CHECKPOINT_SCHEMA, model_class=type(model).__name__, model_factory=factory,
                      sampling_protocol="full_window", history_policy="canonical_causal_imputation",
                      claim_scope="smoke_only" if args.max_train_batches is not None or args.max_val_batches is not None else FORMAL_SCOPE,
                      objective_recipe=objective_recipe(args.method, args.teacher), lambda_gate=1.0, lambda_specialization=0.3,
                      source_snapshot=source_snapshot(), runtime=runtime, training_process_id=os.getpid(),
                      data_identity=dict(data.identity), statistics_identity=dict(data.statistics.identity),
                      reader_identity=data.reader.identity, sampling_identity=data.reader.sampling_identity,
                      selection_metric="complete_global_observed_point_mae", precision="FP32",
                      early_stopping=False, test_evaluated=False, test_numeric_api_available=False,
                      val_selection=validation_identity(data, args.max_val_batches))
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("config", config), ("source_snapshot", config["source_snapshot"]), ("data_identity", dict(data.identity)),
                            ("statistics_identity", dict(data.statistics.identity)), ("reader_identity", data.reader.identity),
                            ("sampling_identity", data.reader.sampling_identity), ("loading", dict(data.loading))):
            atomic_json(output / (name + ".json"), value)
    else:
        config = saved
    atomic_json(output / "model_spec.json", model.research_spec)
    if args.mode == "calibrate":
        result = calibrate(model, data, device, config, warmup=args.warmup_batches, timed=args.timed_batches, output=output)
    else:
        result = train_run(model=model, data=data, device=device, config=config, output=output,
                           checkpoint=previous, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
