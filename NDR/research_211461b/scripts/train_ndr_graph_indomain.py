#!/usr/bin/env python3
"""Single-region v3 event SharedGraphNDR development, never a transfer fold."""
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
from ndr_research.data import ResearchData, array_digest, canonical_digest, load_research_data, sha256_file
from ndr_research.graph_indomain import HIDDEN_DIMS, batch_tensors, graph_identity, model_config, normalizer_identity
from ndr_research.graph_loss import TEACHERS
from ndr_research.shared_graph import SharedGraphNDR
from scripts.train_ndr_research import (
    atomic_json, capture_rng_state, empty_partial, evaluation_identity, reject_identity_overrides,
    require_snapshot, restore_rng_state, source_snapshot as checkout_snapshot, write_history,
)
from scripts.train_ndr_transfer import (
    configure_transfer_runtime, forward_graph, graph_chunk, optimize_graph_batch, transfer_runtime_identity,
)

CHECKPOINT_SCHEMA = "graph_indomain_checkpoint_v1"
MODEL_CLASS = "SharedGraphNDR"


def source_snapshot() -> dict[str, Any]:
    full = checkout_snapshot(ROOT)
    dependencies = {Path(__file__).resolve().relative_to(ROOT).as_posix()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
            relative = path.relative_to(ROOT)
            if relative.parts[0] not in {"tests", "outputs"}:
                dependencies.add(relative.as_posix())
    dependencies.update({"ndr_research/graph_indomain.py", "ndr_research/shared_graph.py",
                         "ndr_research/graph_loss.py", "ndr_research/data.py",
                         "scripts/train_ndr_research.py", "scripts/train_ndr_transfer.py"})
    files = {name: record for name, record in full["files"].items() if name in dependencies}
    for name in dependencies:
        path = ROOT / name
        files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {"files": files, "files_digest": canonical_digest(files), "git": full["git"],
            "scope": "graph_indomain_entrypoint_and_imported_local_python_dependencies"}


def model_forward(model: SharedGraphNDR, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return forward_graph(model, tensors)


def optimize_batch(model: Any, optimizer: Any, scheduler: Any,
                   tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, Any]:
    step = optimize_graph_batch(model, optimizer, scheduler, tensors, config)
    return step | {"error_sum": step["absolute_error_sum"]}


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
                    chunk = graph_chunk(tensors, start, start + microbatch_size)
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


def save_checkpoint(path: Path, *, model: SharedGraphNDR, optimizer: Any, scheduler: Any,
                    data: ResearchData, config: Mapping[str, Any],
                    progress: Mapping[str, Any], selection_metric: Mapping[str, Any] | None) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "model_class": MODEL_CLASS,
        "model_config": model.get_config(), "config": dict(config),
        "config_digest": canonical_digest(config), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "loaders": {name: _loader_checkpoint(getattr(data, name)) for name in ("train", "val")},
        "data_identity": data.identity, "normalizer_identity": normalizer_identity(data),
        "graph_identity": graph_identity(data), "progress": copy.deepcopy(progress),
        "selection_metric": copy.deepcopy(selection_metric),
        "resume_boundary": "completed_effective_batch_or_epoch",
        "test_loader_constructed": False, "test_evaluated": False,
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
        raise ValueError("graph indomain checkpoint content hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "model_config", "config", "config_digest", "optimizer", "scheduler", "rng",
                "loaders", "progress", "data_identity", "normalizer_identity", "graph_identity", "selection_metric"}
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA or not required.issubset(checkpoint):
        raise ValueError("checkpoint is not a resumable graph indomain checkpoint")
    config = checkpoint["config"]
    if (canonical_digest(config) != checkpoint["config_digest"]
            or checkpoint["config_digest"] != marker["config_digest"]):
        raise ValueError("graph indomain checkpoint configuration digest mismatch")
    if (checkpoint.get("model_class") != MODEL_CLASS or config.get("model_class") != MODEL_CLASS
            or checkpoint["model_config"] != model_config(config["hidden_dim"])
            or config.get("model_config") != checkpoint["model_config"]):
        raise ValueError("checkpoint does not bind the fixed real SharedGraphNDR architecture")
    for key in ("data_identity", "normalizer_identity", "graph_identity"):
        if config.get(key) != checkpoint[key]:
            raise ValueError(f"checkpoint configuration differs from its {key}")
    if checkpoint.get("test_evaluated") is not False or checkpoint.get("test_loader_constructed") is not False:
        raise ValueError("single-region development checkpoint has test access")
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
    if (checkpoint["model_config"] != model.get_config()
            or checkpoint["normalizer_identity"] != normalizer_identity(data)
            or checkpoint["graph_identity"] != graph_identity(data)
            or config["runtime"] != transfer_runtime_identity(next(model.parameters()).device)):
        raise ValueError("strict graph resume model/normalizer/graph/runtime identity mismatch")
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
        raise ValueError("strict graph resume epoch/stage is invalid")
    if progress["stage"] in {"train", "validate"}:
        cursor = data.train.current_ind
        if (not 0 <= cursor <= batches or progress["partial_epoch"]["batches"] != cursor
                or progress["stage"] == "validate" and cursor != batches):
            raise ValueError("strict graph resume progress differs from the committed loader cursor")
    return progress


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
    if os.getpid() == config["training_process_id"]:
        raise ValueError("graph verification must reload the checkpoint in an independent process")
    require_snapshot(config["source_snapshot"])
    if (checkpoint["normalizer_identity"] != normalizer_identity(data)
            or checkpoint["graph_identity"] != graph_identity(data)):
        raise ValueError("verification regional train normalization or static graph differs")
    if checkpoint["data_identity"] != data.identity:
        raise ValueError("independent verification cache/sample identity mismatch")
    if current_runtime != config["runtime"]:
        raise ValueError("independent verification runtime differs from the training runtime")
    expected = checkpoint["selection_metric"]
    if expected is None:
        raise ValueError("verification requires a selected best checkpoint with a recorded validation metric")
    if evaluation_identity(data, config["max_val_batches"]) != expected["selection"]:
        raise ValueError("independent verification selects a different validation collection")
    model = SharedGraphNDR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate(model, data, device, microbatch_size=config["microbatch_size"],
                      max_batches=config["max_val_batches"])
    identity_keys = ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256")
    if any(actual[key] != expected[key] for key in identity_keys):
        raise ValueError("verification did not reproduce the exact validation observations")
    if not math.isclose(actual["mae"], expected["mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent checkpoint validation MAE mismatch")
    status = "verified_graph_indomain_checkpoint_complete_validation"
    if config["claim_scope"] == "smoke_only":
        status = ("verified_graph_indomain_smoke_complete_validation"
                  if actual["selection"]["all_validation_samples"]
                  else "verified_graph_indomain_smoke_validation_subset")
    if config["claim_scope"] == "validation_development_only" and not actual["selection"]["all_validation_samples"]:
        raise ValueError("formal graph verification requires complete global validation")
    result = {
        "schema_version": 1, "model_class": MODEL_CLASS,
        "training_process_id": config["training_process_id"], "verification_process_id": os.getpid(),
        "normalizer_identity": normalizer_identity(data), "graph_identity": graph_identity(data),
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


def _next_full_batch(loader: Any) -> dict[str, Any]:
    if loader.size < loader.bs:
        raise ValueError("calibration requires a full effective batch per train/val split")
    for _ in range(loader.num_batch + 1):
        if loader.current_ind >= loader.num_batch:
            loader.start_epoch()
        batch = loader.next_batch()
        if batch["x_data"].shape[0] == loader.bs:
            return batch
    raise ValueError("calibration did not find a complete effective batch")


def calibrate(model: Any, data: ResearchData, device: torch.device,
              config: Mapping[str, Any], *, warmup: int, timed: int, output: Path) -> dict[str, Any]:
    optimizer, scheduler = create_optimizer(model, config, data.train.num_batch)
    reports = {}
    for split, loader in (("train", data.train), ("val", data.val)):
        loader.start_epoch()
        durations = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for number in range(warmup + timed):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            tensors = batch_tensors(_next_full_batch(loader), data, device)
            if split == "train":
                optimize_batch(model, optimizer, scheduler, tensors, config)
            else:
                model.eval()
                with torch.no_grad():
                    for start in range(0, tensors["target"].shape[0], config["microbatch_size"]):
                        model_forward(model, graph_chunk(tensors, start, start + config["microbatch_size"]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if number >= warmup:
                durations.append(time.perf_counter() - started)
        reports[split] = {"mean_batch_seconds": float(np.mean(durations)), "timed_batches": timed,
                          "warmup_batches": warmup, "full_effective_batch_size": loader.bs,
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    estimate = config["epochs"] * sum(reports[name]["mean_batch_seconds"] * getattr(data, name).num_batch
                                       for name in ("train", "val"))
    result = {"status": "calibrated_graph_indomain_engineering_only", "claim_scope": "calibration_only",
              "region": config["region"], "splits": reports, "estimated_full_train_val_seconds": estimate,
              "timing_includes_reader": True, "config_digest": canonical_digest(config),
              "test_loader_constructed": False, "test_evaluated": False}
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate", "verify"), default="train")
    parser.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    parser.add_argument("--teacher", choices=TEACHERS, default="original")
    parser.add_argument("--hidden-dim", type=int, choices=HIDDEN_DIMS, default=64)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, choices=(30, 100), default=30)
    parser.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=0.00001)
    parser.add_argument("--lambda-gate", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--lambda-specialization", type=float, choices=(0.3,), default=0.3)
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
    model_config(args.hidden_dim)
    for name in ("batch_size", "microbatch_size", "timed_batches", "threads", "log_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.microbatch_size > args.batch_size or args.warmup_batches < 0:
        raise ValueError("microbatch or warmup count is invalid")
    for name in ("max_train_batches", "max_val_batches", "max_wallclock_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("verify requires --checkpoint")
    if args.mode != "verify" and args.checkpoint is not None:
        raise ValueError("checkpoint is only for independent validation verification")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("strict continuation must use train mode and last.pt")
    if args.mode == "verify" and (args.max_train_batches is not None or args.max_val_batches is not None):
        raise ValueError("verification must preserve the checkpoint's frozen validation collection")
    if args.mode == "train" and torch.device(args.device).type == "cpu" and (args.max_train_batches is None or args.max_val_batches is None):
        raise ValueError("CPU graph training requires explicit bounded train and validation smoke subsets")
    for name in ("learning_rate", "gradient_clip", "eta_min"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if args.lambda_gate != 1.0 or args.lambda_specialization != 0.3:
        raise ValueError("graph oracle and specialization weights are predeclared as 1.0 and 0.3")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    previous = load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous is not None else None
    if saved is not None:
        reject_identity_overrides(args, saved, arguments)
        for name in ("region", "hidden_dim", "teacher", "epochs", "batch_size", "microbatch_size"):
            if not any(value == "--" + name.replace("_", "-") or value.startswith("--" + name.replace("_", "-") + "=") for value in arguments):
                setattr(args, name, saved[name])
        if args.mode == "train":
            args.max_train_batches, args.max_val_batches = saved["max_train_batches"], saved["max_val_batches"]
    validate_arguments(args)
    seed, threads = (saved["seed"], saved["threads"]) if saved else (args.seed, args.threads)
    device = configure_transfer_runtime(args.device, seed, threads)
    runtime = transfer_runtime_identity(device)
    if saved is not None:
        require_snapshot(saved["source_snapshot"])
        if runtime != saved["runtime"]:
            raise ValueError("strict graph resume/verify runtime must match the original runtime")
        if args.lazy and not saved["lazy"]:
            raise ValueError("resume/verify cannot change the frozen preload policy")
    data = load_research_data(region_name=saved["region"] if saved else args.region,
                              data_dir=args.data_dir, graph_dir=args.graph_dir,
                              batch_size=saved["batch_size"] if saved else args.batch_size,
                              seed=seed, preload=not (saved["lazy"] if saved else args.lazy))
    output = args.output_dir or ((args.resume or args.checkpoint).parent if previous else None)
    if output is None:
        raise ValueError("new graph runs require an isolated --output-dir")
    output = output.resolve()
    if args.resume and output != args.resume.resolve().parent:
        raise ValueError("strict continuation must use the original output directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_checkpoint(checkpoint_path=args.checkpoint, data=data, device=device,
                                   output=output, current_runtime=runtime)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if saved is None:
        config = {name: getattr(args, name) for name in (
            "region", "teacher", "hidden_dim", "epochs", "seed", "batch_size", "microbatch_size",
            "learning_rate", "weight_decay", "gradient_clip", "eta_min", "lambda_gate",
            "lambda_specialization", "max_train_batches", "max_val_batches", "log_interval", "threads", "lazy")}
        config.update({"schema_version": CHECKPOINT_SCHEMA, "model_class": MODEL_CLASS,
                       "model_config": model_config(args.hidden_dim), "method": "SharedGraphNDR_indomain",
                       "purpose": "v3_event_single_region_train_validation_development",
                       "claim_scope": "smoke_only" if args.max_train_batches is not None or args.max_val_batches is not None else "validation_development_only",
                       "source_snapshot": source_snapshot(), "runtime": runtime,
                       "training_process_id": os.getpid(),
                       "data_identity": data.identity, "normalizer_identity": normalizer_identity(data),
                       "graph_identity": graph_identity(data), "precision": "FP32",
                       "early_stopping": False, "test_evaluation": False,
                       "selection_split": "val", "selection_metric": "complete_global_masked_mae",
                       "val_selection": evaluation_identity(data, args.max_val_batches),
                       "training_recipe": {"optimizer": "Adam", "scheduler": "CosineAnnealingLR_per_completed_effective_update",
                                           "loss": "original_graph_NDR_raw_flow_masked_mae_plus_oracle_plus_specialization",
                                           "teacher": args.teacher, "lambda_gate": 1.0, "lambda_specialization": 0.3,
                                           "normalization": "regional_train_only_weekly_climatology_and_node_residual_scale",
                                           "effective_batch_size": args.batch_size, "microbatch_size": args.microbatch_size}})
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        for name, value in (("config", config), ("source_snapshot", config["source_snapshot"]),
                            ("data_identity", data.identity), ("normalizer_identity", config["normalizer_identity"]),
                            ("graph_identity", config["graph_identity"])):
            atomic_json(output / (name + ".json"), value)
    else:
        config = saved
    model = SharedGraphNDR(**config["model_config"]).to(device)
    atomic_json(output / "model_spec.json", model.research_spec | {"training_scope": "regional_supervised"})
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
