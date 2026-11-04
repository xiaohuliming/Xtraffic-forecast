#!/usr/bin/env python3
"""Source-only, node-inductive NDR training and independent source validation."""
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
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dist_net.metrics import MaskedMAEAccumulator
from ndr_research.data import array_digest, canonical_digest, sha256_file
from ndr_research.graph_loss import TEACHERS, build_graph_loss_normalizers, compute_graph_loss
from ndr_research.shared_graph import SharedGraphNDR, canonical_history
from scripts.train_ndr_research import (
    atomic_json,
    capture_rng_state,
    configure_runtime,
    load_checkpoint,
    reject_identity_overrides,
    require_snapshot,
    restore_rng_state,
    runtime_identity,
    source_snapshot as full_checkout_snapshot,
    write_history,
)

REGIONS = ("Alameda", "ContraCosta", "Orange")
CHECKPOINT_SCHEMA = 1
FORWARD_FIELDS = ("flow_hist", "x_baseline", "y_baseline", "time_feat",
                  "history_mask", "edge_index", "node_scale")


def transfer_source_snapshot() -> dict[str, Any]:
    """Bind imported candidate dependencies without sealing unrelated baselines."""
    snapshot = full_checkout_snapshot(ROOT)
    dependencies = {Path(__file__).resolve().relative_to(ROOT).as_posix()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(ROOT):
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in {"tests", "outputs"}:
            dependencies.add(relative.as_posix())
    files = {name: record for name, record in snapshot["files"].items() if name in dependencies}
    core = {"ndr_research/shared_graph.py", "ndr_research/transfer_data.py",
            "ndr_research/graph_loss.py", "scripts/train_ndr_transfer.py"}
    for name in core:
        path = ROOT / name
        files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return {"files": files, "files_digest": canonical_digest(files), "git": snapshot["git"],
            "scope": "candidate_entrypoint_and_imported_python_dependencies"}


source_snapshot = transfer_source_snapshot


def configure_transfer_runtime(device_name: str, seed: int, threads: int = 3) -> torch.device:
    device = configure_runtime(device_name, seed, threads)
    # Record a deterministic SDP recipe before any graph-model CUDA operation.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return device


def transfer_runtime_identity(device: torch.device) -> dict[str, Any]:
    return runtime_identity(device) | {
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }


def validate_fold(source_regions: list[str] | tuple[str, ...], heldout_region: str) -> tuple[str, str]:
    sources = tuple(source_regions)
    if len(sources) != 2 or len(set(sources)) != 2 or any(region not in REGIONS for region in sources):
        raise ValueError("a leave-one-region-out fold requires two distinct known source regions")
    if heldout_region not in REGIONS or heldout_region in sources or set(sources) | {heldout_region} != set(REGIONS):
        raise ValueError("heldout region must be the excluded third region")
    return sources


def epoch_region_plan(counts: Mapping[str, int], *, seed: int, epoch: int) -> list[str]:
    if len(counts) != 2 or any(type(count) is not int or count <= 0 for count in counts.values()):
        raise ValueError("region plan requires two positive effective-batch counts")
    plan = [region for region in sorted(counts) for _ in range(counts[region])]
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 713]))
    rng.shuffle(plan)
    return plan


def region_objective_multipliers(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {region: total / (len(counts) * count) for region, count in counts.items()}


def checkpoint_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, copy=True))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(checkpoint_safe(item) for item in value)
    if isinstance(value, list):
        return [checkpoint_safe(item) for item in value]
    return value


def graph_batch_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    required = set(FORWARD_FIELDS) | {"target", "target_mask", "sample_start"}
    if set(batch) != required:
        raise ValueError(f"graph training batch whitelist mismatch: {sorted(set(batch) ^ required)}")
    tensors = {key: torch.as_tensor(np.array(value, copy=True), device=device)
               for key, value in batch.items() if key != "sample_start"}
    for key in ("flow_hist", "x_baseline", "y_baseline", "time_feat", "node_scale", "target"):
        tensors[key] = tensors[key].float()
    if tensors["edge_index"].dtype not in (torch.int32, torch.int64):
        raise ValueError("road graph must use explicit integer edge indices")
    target = tensors["target"]
    mask = tensors["target_mask"]
    if target.ndim != 3 or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("graph target and explicit boolean mask must match (B,N,H)")
    if not torch.isfinite(target[mask]).all():
        raise FloatingPointError("graph observed targets must be finite")
    tensors["target"] = torch.where(mask, target, 0.0)
    tensors["edge_index"] = tensors["edge_index"].long()
    node_scale = tensors["node_scale"]
    batch_size, nodes, _ = target.shape
    if node_scale.shape == (nodes,):
        node_scale = node_scale.unsqueeze(0).expand(batch_size, -1)
    elif node_scale.shape == (batch_size, nodes, 1):
        node_scale = node_scale.squeeze(-1)
    elif node_scale.shape != (batch_size, nodes):
        raise ValueError("runtime node scale is incompatible with this region")
    tensors["node_scale"] = node_scale.contiguous()
    return tensors


def graph_chunk(tensors: Mapping[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value if key == "edge_index" else value[start:stop]
            for key, value in tensors.items()}


def forward_graph(model: SharedGraphNDR, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    prediction = model(**{key: tensors[key] for key in FORWARD_FIELDS})
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("graph forecast must be finite everywhere and match target shape")
    return prediction


def backward_graph_batch(
    model: SharedGraphNDR, tensors: Mapping[str, torch.Tensor], *, microbatch_size: int,
    teacher: str = "original", lambda_gate: float = 1.0,
    lambda_specialization: float = 0.3, objective_multiplier: float = 1.0,
) -> dict[str, Any]:
    metric = MaskedMAEAccumulator()
    valid_count = int(tensors["target_mask"].sum().item())
    normalizers = None
    if valid_count:
        with torch.no_grad():
            state = canonical_history(tensors["flow_hist"], tensors["x_baseline"], tensors["history_mask"])
            normalizers = build_graph_loss_normalizers(
                tensors["target"], tensors["target_mask"], baseline=tensors["y_baseline"],
                persistence=state.reference.unsqueeze(-1), node_scale=tensors["node_scale"], teacher=teacher,
            )
    parts: dict[str, float] = {}
    for start in range(0, tensors["target"].shape[0], microbatch_size):
        chunk = graph_chunk(tensors, start, start + microbatch_size)
        prediction = forward_graph(model, chunk)
        metric.update(prediction, chunk["target"], chunk["target_mask"])
        if not chunk["target_mask"].any():
            continue
        loss, local_parts = compute_graph_loss(
            model, prediction, chunk["target"], chunk["target_mask"], teacher=teacher,
            lambda_gate=lambda_gate, lambda_specialization=lambda_specialization, normalizers=normalizers,
        )
        (loss * objective_multiplier).backward()
        for name, value in local_parts.items():
            parts[name] = parts.get(name, 0.0) + value
    objective = parts.get("mae", 0.0) + lambda_gate * parts.get("gate", 0.0)
    objective += lambda_specialization * parts.get("spec", 0.0)
    return {"absolute_error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
            "objective": objective, "weighted_objective": objective * objective_multiplier,
            "parts": parts, "updated": bool(valid_count)}


def optimize_graph_batch(model: Any, optimizer: Any, scheduler: Any,
                         tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any],
                         *, objective_multiplier: float = 1.0) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = backward_graph_batch(
        model, tensors, microbatch_size=config["microbatch_size"], teacher=config["teacher"],
        lambda_gate=config["lambda_gate"], lambda_specialization=config["lambda_specialization"],
        objective_multiplier=objective_multiplier,
    )
    if result["updated"]:
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("graph gradient is non-finite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
            raise FloatingPointError("graph optimizer produced non-finite parameters")
    return result


def val_selection_identity(loader: Any, reader: Any, max_batches: int | None) -> dict[str, Any]:
    count = loader.size if max_batches is None else min(loader.size, max_batches * loader.batch_size)
    indices = loader.indices[:count]
    return {"selection_split": "source_val", "samples": count,
            "sample_indices_sha256": array_digest(indices),
            "forecast_origins_sha256": array_digest(reader.sample_start[indices]),
            "reader_identity": reader.identity,
            "all_validation_samples": count == loader.size}


def evaluate_source_region(model: Any, data: Any, region: str, device: torch.device,
                           *, microbatch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    if region not in data.readers or region in {getattr(data, "heldout_region", None)}:
        raise ValueError("source validation cannot evaluate a heldout reader")
    reader, loader = data.readers[region], data.val_loaders[region]
    if reader.role != "source":
        raise ValueError("checkpoint selection requires source-role readers")
    loader.start_epoch()
    batches = min(loader.num_batch, max_batches or loader.num_batch)
    metric = MaskedMAEAccumulator()
    errors = np.zeros(model.horizon, dtype=np.float64)
    counts = np.zeros(model.horizon, dtype=np.int64)
    origins: list[int] = []
    target_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    with torch.no_grad():
        for _ in range(batches):
            batch = loader.get_next()
            if batch is None:
                raise ValueError("source validation exhausted before the declared collection")
            origins.extend(np.asarray(batch["sample_start"], dtype=np.int64).tolist())
            tensors = graph_batch_tensors(batch, device)
            target_hash.update(tensors["target"].cpu().contiguous().numpy().tobytes())
            mask_hash.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
            for start in range(0, tensors["target"].shape[0], microbatch_size):
                chunk = graph_chunk(tensors, start, start + microbatch_size)
                prediction = forward_graph(model, chunk)
                metric.update(prediction, chunk["target"], chunk["target_mask"])
                error = torch.where(chunk["target_mask"], (prediction - chunk["target"]).abs(), 0.0)
                errors += error.double().sum(dim=(0, 1)).cpu().numpy()
                counts += chunk["target_mask"].sum(dim=(0, 1)).cpu().numpy()
    selection = val_selection_identity(loader, reader, max_batches)
    if array_digest(np.asarray(origins, dtype=np.int64)) != selection["forecast_origins_sha256"]:
        raise ValueError("source validation stream differs from its frozen collection")
    horizons = [float(error / count) if count else None for error, count in zip(errors, counts)]
    return {"global_masked_mae": metric.compute(), "absolute_error_sum": metric._absolute_error_sum,
            "valid_count": metric.valid_count, "horizon_mae": horizons,
            "horizon_valid_counts": counts.tolist(),
            "average_horizon_mae": float(np.mean(horizons)) if counts.all() else None,
            "selection": selection, "observed_targets_sha256": target_hash.hexdigest(),
            "observation_masks_sha256": mask_hash.hexdigest(), "finite_predictions_everywhere": True}


def evaluate_sources(model: Any, data: Any, device: torch.device, *,
                     microbatch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    try:
        regions = {region: evaluate_source_region(model, data, region, device,
                                                 microbatch_size=microbatch_size, max_batches=max_batches)
                   for region in data.readers}
    finally:
        model.train(was_training)
    if len(regions) != 2:
        raise ValueError("checkpoint selection requires exactly two source regions")
    averages = [value["average_horizon_mae"] for value in regions.values()]
    return {"selection_split": "source_val", "regions": regions,
            "macro_global_mae": float(np.mean([value["global_masked_mae"] for value in regions.values()])),
            "macro_average_horizon_mae": float(np.mean(averages)) if all(value is not None for value in averages) else None,
            "all_source_validation_samples": all(value["selection"]["all_validation_samples"] for value in regions.values()),
            "source_regions": list(data.readers), "target_region_used": False}


def source_statistics_payload(data: Any) -> Any:
    return checkpoint_safe(data.preprocessing.to_state())


def apply_source_loading(data: Any, *, lazy: bool) -> dict[str, Any]:
    if lazy:
        return {"policy": "source_lazy_raw_windows", "regions": list(data.readers),
                "heldout_preloaded": False, "bytes": 0}
    regions = {}
    for region, reader in data.readers.items():
        if reader.role != "source":
            raise ValueError("source preload cannot contain a target-role reader")
        regions[region] = reader.preload_source_train_val()
    return {"policy": "source_eager_channel_zero_prefix_to_val_end", "regions": regions,
            "heldout_preloaded": False, "bytes": sum(value["bytes"] for value in regions.values())}


def save_transfer_checkpoint(path: Path, *, model: SharedGraphNDR, optimizer: Any,
                             scheduler: Any, data: Any, config: Mapping[str, Any],
                             progress: Mapping[str, Any], selection_metric: Mapping[str, Any] | None) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "model_class": "SharedGraphNDR",
        "model_config": model.get_config(), "model": model.state_dict(),
        "config": dict(config), "config_digest": canonical_digest(config),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(), "data_identity": data.identity,
        "source_statistics": source_statistics_payload(data),
        "source_statistics_identity": data.preprocessing.identity,
        "loaders": {split: {region: checkpoint_safe(loader.state_dict()) for region, loader in loaders.items()}
                    for split, loaders in (("train", data.train_loaders), ("val", data.val_loaders))},
        "progress": copy.deepcopy(progress),
        "selection_metric": dict(selection_metric) if selection_metric is not None else None,
        "selection_split": "source_val", "heldout_region": config["heldout_region"],
        "source_regions": config["source_regions"], "target_data_read": False,
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


def load_transfer_checkpoint(path: Path) -> dict[str, Any]:
    from ndr_research.transfer_data import TransferPreprocessing
    checkpoint = load_checkpoint(path)
    required = {"source_statistics", "source_statistics_identity", "model_config", "source_regions", "heldout_region"}
    if not required.issubset(checkpoint) or checkpoint.get("model_class") != "SharedGraphNDR":
        raise ValueError("checkpoint is not a node-inductive source-only model")
    validate_fold(checkpoint["source_regions"], checkpoint["heldout_region"])
    if checkpoint.get("selection_split") != "source_val" or checkpoint.get("target_data_read") is not False:
        raise ValueError("checkpoint selection used non-source development data")
    preprocessing = TransferPreprocessing.from_state(checkpoint["source_statistics"])
    if preprocessing.mode != "source_only" or preprocessing.identity != checkpoint["source_statistics_identity"]:
        raise ValueError("checkpoint numeric source statistics are not bound to its source-only identity")
    return checkpoint


def create_transfer_optimizer(model: Any, config: Mapping[str, Any], total_batches: int) -> tuple[Any, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * total_batches, eta_min=config["eta_min"])
    return optimizer, scheduler


def restore_transfer_training(checkpoint: Mapping[str, Any], model: Any, optimizer: Any,
                              scheduler: Any, data: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if (checkpoint["config_digest"] != canonical_digest(config) or checkpoint["data_identity"] != data.identity
            or checkpoint["source_statistics_identity"] != data.preprocessing.identity):
        raise ValueError("transfer resume code/config/source/template identity mismatch")
    require_snapshot(config["source_snapshot"])
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for split, loaders in (("train", data.train_loaders), ("val", data.val_loaders)):
        if set(checkpoint["loaders"][split]) != set(loaders):
            raise ValueError("transfer resume loader-region identity mismatch")
        for region, loader in loaders.items():
            loader.load_state_dict(checkpoint["loaders"][split][region])
    restore_rng_state(checkpoint["rng"])
    progress = copy.deepcopy(checkpoint["progress"])
    counts = planned_batch_counts(data, config)
    if progress["plan"]:
        expected_plan = epoch_region_plan(counts, seed=config["seed"], epoch=progress["epoch"])
        if progress["plan"] != expected_plan or not 0 <= progress["plan_cursor"] <= len(expected_plan):
            raise ValueError("transfer resume deterministic region plan identity mismatch")
        consumed = progress["plan"][:progress["plan_cursor"]]
        for region, loader in data.train_loaders.items():
            expected_cursor = min(consumed.count(region) * loader.batch_size, loader.size)
            if loader.cursor != expected_cursor:
                raise ValueError("transfer resume loader cursor differs from the committed region plan")
    return progress


def planned_batch_counts(data: Any, config: Mapping[str, Any]) -> dict[str, int]:
    return {region: min(loader.num_batch, config["max_train_batches"] or loader.num_batch)
            for region, loader in data.train_loaders.items()}


def empty_region_partial() -> dict[str, Any]:
    return {"absolute_error_sum": 0.0, "valid_count": 0, "objective_sum": 0.0,
            "batches": 0, "optimized_batches": 0}


def train_transfer_run(*, model: SharedGraphNDR, data: Any, device: torch.device,
                       config: Mapping[str, Any], output: Path,
                       checkpoint: Mapping[str, Any] | None = None,
                       max_wallclock_seconds: float | None = None) -> dict[str, Any]:
    counts = planned_batch_counts(data, config)
    multipliers = region_objective_multipliers(counts)
    optimizer, scheduler = create_transfer_optimizer(model, config, sum(counts.values()))
    progress: dict[str, Any] = {
        "epoch": 0, "stage": "start", "plan": [], "plan_cursor": 0,
        "best_epoch": None, "best_macro_source_val_mae": None, "history": [], "updates": 0,
        "partial_regions": {region: empty_region_partial() for region in counts}, "elapsed_seconds": 0.0,
    }
    if checkpoint is not None:
        progress = restore_transfer_training(checkpoint, model, optimizer, scheduler, data, config)
    write_history(output, progress["history"])
    stopped = {"requested": False}
    handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            handlers[signum] = signal.signal(signum, lambda *_: stopped.__setitem__("requested", True))
        except ValueError:
            pass
    started = time.monotonic()
    accumulated = progress["elapsed_seconds"]

    def should_stop() -> bool:
        return stopped["requested"] or (max_wallclock_seconds is not None
                                       and time.monotonic() - started >= max_wallclock_seconds)

    def persist_last() -> None:
        progress["elapsed_seconds"] = accumulated + time.monotonic() - started
        save_transfer_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                                 data=data, config=config, progress=progress, selection_metric=None)

    def stop_result() -> dict[str, Any]:
        persist_last()
        result = {"status": "stopped_resumable", "completed_epochs": progress["epoch"],
                  "stage": progress["stage"], "plan_cursor": progress["plan_cursor"],
                  "reason": "signal" if stopped["requested"] else "wallclock",
                  "claim_scope": config["claim_scope"], "target_data_read": False}
        atomic_json(output / "run_state.json", result)
        return result

    try:
        atomic_json(output / "run_state.json", {"status": "running", "target_data_read": False,
                                                "config_digest": canonical_digest(config)})
        while progress["epoch"] < config["epochs"]:
            if progress["stage"] == "start":
                for loader in data.train_loaders.values():
                    loader.start_epoch()
                progress["plan"] = epoch_region_plan(counts, seed=config["seed"], epoch=progress["epoch"])
                progress["plan_cursor"] = 0
                progress["partial_regions"] = {region: empty_region_partial() for region in counts}
                progress["stage"] = "train"
            while progress["stage"] == "train" and progress["plan_cursor"] < len(progress["plan"]):
                if should_stop():
                    return stop_result()
                region = progress["plan"][progress["plan_cursor"]]
                batch = data.train_loaders[region].get_next()
                if batch is None:
                    raise ValueError("source training loader exhausted before its region plan")
                tensors = graph_batch_tensors(batch, device)
                step = optimize_graph_batch(model, optimizer, scheduler, tensors, config,
                                             objective_multiplier=multipliers[region])
                partial = progress["partial_regions"][region]
                partial["absolute_error_sum"] += step["absolute_error_sum"]
                partial["valid_count"] += step["valid_count"]
                partial["objective_sum"] += step["objective"]
                partial["batches"] += 1
                partial["optimized_batches"] += int(step["updated"])
                progress["updates"] += int(step["updated"])
                progress["plan_cursor"] += 1
                if progress["plan_cursor"] == 1 or progress["plan_cursor"] % config["log_interval"] == 0:
                    print(json.dumps({"epoch": progress["epoch"] + 1, "region": region,
                                      "batch": progress["plan_cursor"], "batches": len(progress["plan"]),
                                      "updates": progress["updates"], "objective": step["objective"]}), flush=True)
            progress["stage"] = "validate"
            if should_stop():
                return stop_result()
            validation = evaluate_sources(model, data, device, microbatch_size=config["microbatch_size"],
                                          max_batches=config["max_val_batches"])
            region_metrics = {}
            for region, partial in progress["partial_regions"].items():
                if partial["valid_count"] == 0 or partial["batches"] != counts[region]:
                    raise ValueError("source train epoch did not cover every planned usable batch")
                region_metrics[region] = partial | {
                    "global_masked_mae": partial["absolute_error_sum"] / partial["valid_count"],
                    "objective_batch_mean": partial["objective_sum"] / max(1, partial["optimized_batches"]),
                    "objective_multiplier": multipliers[region]}
            row = {"epoch": progress["epoch"] + 1, "train_regions": region_metrics,
                   "train_macro_objective": float(np.mean([value["objective_batch_mean"] for value in region_metrics.values()])),
                   "source_validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                   "updates": progress["updates"]}
            progress["history"].append(row)
            improved = (progress["best_macro_source_val_mae"] is None
                        or validation["macro_global_mae"] < progress["best_macro_source_val_mae"])
            progress["epoch"] += 1
            progress["stage"] = "start"
            progress["plan"] = []
            progress["plan_cursor"] = 0
            if improved:
                progress["best_epoch"] = progress["epoch"]
                progress["best_macro_source_val_mae"] = validation["macro_global_mae"]
            progress["elapsed_seconds"] = accumulated + time.monotonic() - started
            write_history(output, progress["history"])
            if improved:
                save_transfer_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                                         data=data, config=config, progress=progress, selection_metric=validation)
            persist_last()
            print(json.dumps(row, sort_keys=True), flush=True)
        result = {
            "status": "training_completed_source_validation_pending_independent_verification",
            "completed_epochs": progress["epoch"], "best_epoch": progress["best_epoch"],
            "best_macro_source_val_mae": progress["best_macro_source_val_mae"],
            "best_source_validation": progress["history"][progress["best_epoch"] - 1]["source_validation"],
            "source_regions": config["source_regions"], "heldout_region": config["heldout_region"],
            "claim_scope": config["claim_scope"], "target_data_read": False,
            "test_evaluated": False, "config_digest": canonical_digest(config),
            "elapsed_seconds": progress["elapsed_seconds"], "updates": progress["updates"],
        }
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def verify_transfer_checkpoint(*, checkpoint_path: Path, data: Any, device: torch.device,
                               output: Path, current_runtime: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = load_transfer_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    require_snapshot(config["source_snapshot"])
    if (data.identity != checkpoint["data_identity"]
            or data.preprocessing.identity != checkpoint["source_statistics_identity"]):
        raise ValueError("source-only verification data/template identity mismatch")
    if current_runtime != config["runtime"]:
        raise ValueError("source-only verification requires the original runtime")
    expected = checkpoint["selection_metric"]
    if expected is None or expected.get("selection_split") != "source_val":
        raise ValueError("verification requires a selected source-validation best checkpoint")
    model = SharedGraphNDR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate_sources(model, data, device, microbatch_size=config["microbatch_size"],
                              max_batches=config["max_val_batches"])
    if set(actual["regions"]) != set(expected["regions"]):
        raise ValueError("verification source regions differ")
    for region, observed in actual["regions"].items():
        recorded = expected["regions"][region]
        for key in ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256"):
            if observed[key] != recorded[key]:
                raise ValueError("verification did not reproduce the exact source validation collection")
        if not math.isclose(observed["global_masked_mae"], recorded["global_masked_mae"], rel_tol=2e-6, abs_tol=2e-5):
            raise ValueError("independent source validation metric mismatch")
    if not math.isclose(actual["macro_global_mae"], expected["macro_global_mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent macro source validation metric mismatch")
    status = "verified_transfer_source_checkpoint_complete_validation"
    if config["claim_scope"] == "smoke_only":
        status = ("verified_smoke_complete_source_validation" if actual["all_source_validation_samples"]
                  else "verified_smoke_source_validation_subset")
    result = {"schema_version": 1, "status": status, "source_validation": actual,
              "expected_macro_global_mae": expected["macro_global_mae"],
              "absolute_mae_difference": abs(actual["macro_global_mae"] - expected["macro_global_mae"]),
              "checkpoint_sha256": sha256_file(checkpoint_path), "config_digest": checkpoint["config_digest"],
              "source_statistics_identity": checkpoint["source_statistics_identity"],
              "source_files_digest": config["source_snapshot"]["files_digest"],
              "source_snapshot_files_digest": config["source_snapshot"]["files_digest"],
              "selection_metric_digest": canonical_digest(expected),
              "all_source_validation_samples": actual["all_source_validation_samples"],
              "model_config_sha256": canonical_digest(checkpoint["model_config"]),
              "model_config": checkpoint["model_config"], "source_regions": config["source_regions"],
              "heldout_region": config["heldout_region"], "claim_scope": config["claim_scope"],
              "target_data_read": False, "test_evaluated": False}
    atomic_json(output / "verification.json", result)
    return result


def calibrate_transfer(*, model: Any, data: Any, device: torch.device, config: Mapping[str, Any],
                       output: Path, warmup: int, timed: int) -> dict[str, Any]:
    counts = planned_batch_counts(data, config)
    multipliers = region_objective_multipliers(counts)
    optimizer, scheduler = create_transfer_optimizer(model, config, sum(counts.values()))
    region_reports = {}
    for region in data.readers:
        report = {}
        for split, loader in (("train", data.train_loaders[region]), ("val", data.val_loaders[region])):
            loader.start_epoch()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for number in range(warmup + timed):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                batch = loader.get_next()
                if batch is None:
                    loader.start_epoch()
                    batch = loader.get_next()
                if batch is None:
                    raise ValueError("calibration source loader is empty")
                tensors = graph_batch_tensors(batch, device)
                if split == "train":
                    optimize_graph_batch(model, optimizer, scheduler, tensors, config,
                                         objective_multiplier=multipliers[region])
                else:
                    model.eval()
                    with torch.no_grad():
                        for start in range(0, tensors["target"].shape[0], config["microbatch_size"]):
                            forward_graph(model, graph_chunk(tensors, start, start + config["microbatch_size"]))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if number >= warmup:
                    durations.append(time.perf_counter() - started)
            report[split] = {"mean_batch_seconds": float(np.mean(durations)),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                             "timed_batches": timed, "warmup_batches": warmup}
        region_reports[region] = report
    estimate = config["epochs"] * sum(
        region_reports[region]["train"]["mean_batch_seconds"] * data.train_loaders[region].num_batch
        + region_reports[region]["val"]["mean_batch_seconds"] * data.val_loaders[region].num_batch
        for region in data.readers)
    result = {"status": "calibrated_source_only_engineering", "claim_scope": "calibration_only",
              "regions": region_reports, "estimated_train_source_val_seconds": estimate,
              "timing_includes_reader": True, "config_digest": canonical_digest(config),
              "target_data_read": False, "test_evaluated": False}
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "calibrate", "verify"), default="train")
    parser.add_argument("--source-regions", nargs=2, choices=REGIONS, default=["Alameda", "ContraCosta"])
    parser.add_argument("--heldout-region", choices=REGIONS, default="Orange")
    parser.add_argument("--teacher", choices=TEACHERS, default="original")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, choices=(30, 100), default=30)
    parser.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--road-layers", type=int, default=2)
    parser.add_argument("--latent-tokens", type=int, choices=(8, 16), default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--eta-min", type=float, default=0.00001)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--lambda-specialization", type=float, default=0.3)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-wallclock-seconds", type=float)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--lazy", action="store_true", help="Disable source-only train/validation prefix preloading.")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    validate_fold(args.source_regions, args.heldout_region)
    for name in ("batch_size", "microbatch_size", "hidden_dim", "road_layers", "num_heads",
                 "timed_batches", "log_interval", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.microbatch_size > args.batch_size or args.warmup_batches < 0:
        raise ValueError("microbatch or warmup count is invalid")
    for name in ("max_train_batches", "max_val_batches", "max_wallclock_seconds"):
        if getattr(args, name) is not None and (not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            raise ValueError(f"{name} must be positive and finite when supplied")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("verify requires a best checkpoint")
    if args.mode != "verify" and args.checkpoint is not None:
        raise ValueError("checkpoint is only for independent source validation")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("strict continuation must use train mode and last.pt")
    if args.mode == "verify" and (args.max_train_batches is not None or args.max_val_batches is not None):
        raise ValueError("verification cannot override the frozen source validation collection")
    if args.mode == "train" and torch.device(args.device).type == "cpu" and args.max_train_batches is None:
        raise ValueError("CPU training is engineering smoke only and requires --max-train-batches")
    for name in ("learning_rate", "gradient_clip", "eta_min"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("weight_decay", "lambda_gate", "lambda_specialization"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must lie in [0,1)")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    from ndr_research.transfer_data import load_transfer_sources
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    args.source_regions = sorted(args.source_regions)
    checkpoint = load_transfer_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    config = checkpoint["config"] if checkpoint is not None else None
    if config is not None:
        reject_identity_overrides(args, config, arguments)
        restored_fields = ["source_regions", "heldout_region"]
        if args.mode == "train":
            restored_fields.extend(("max_train_batches", "max_val_batches"))
        for name in restored_fields:
            option = "--" + name.replace("_", "-")
            if not any(value == option or value.startswith(option + "=") for value in arguments):
                setattr(args, name, config[name])
    validate_arguments(args)
    device = configure_transfer_runtime(args.device, config["seed"] if config else args.seed,
                                        config["threads"] if config else args.threads)
    runtime = transfer_runtime_identity(device)
    if config is not None:
        require_snapshot(config["source_snapshot"])
        if runtime != config["runtime"]:
            raise ValueError("strict transfer resume/verify requires the original runtime")
    sources = config["source_regions"] if config else args.source_regions
    heldout = config["heldout_region"] if config else args.heldout_region
    data = load_transfer_sources(source_regions=sources, heldout_region=heldout,
                                 data_dir=args.data_dir, graph_dir=args.graph_dir,
                                 batch_size=config["batch_size"] if config else args.batch_size,
                                 seed=config["seed"] if config else args.seed)
    if set(data.readers) != set(sources) or heldout in data.readers:
        raise ValueError("training data unexpectedly contains a heldout region")
    if config is not None and args.lazy and config["loading_policy"] != "source_lazy_raw_windows":
        raise ValueError("resume/verify cannot override the frozen source loading policy")
    lazy = args.lazy if config is None else config["loading_policy"] == "source_lazy_raw_windows"
    loading = apply_source_loading(data, lazy=lazy)
    output = args.output_dir or ((args.resume or args.checkpoint).parent if checkpoint else None)
    if output is None:
        raise ValueError("new transfer runs require an explicit isolated output directory")
    output = output.resolve()
    if args.resume and output != args.resume.resolve().parent:
        raise ValueError("transfer continuation must retain the original output directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_transfer_checkpoint(checkpoint_path=args.checkpoint, data=data, device=device,
                                            output=output, current_runtime=runtime)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if config is None:
        model_config = {name: getattr(args, name) for name in
                        ("hidden_dim", "road_layers", "latent_tokens", "num_heads", "dropout")}
        model_config.update(history_steps=12, horizon=12)
        config = {name: getattr(args, name) for name in
                  ("source_regions", "heldout_region", "teacher", "epochs", "seed", "batch_size",
                   "microbatch_size", "hidden_dim", "road_layers", "latent_tokens", "num_heads", "dropout",
                   "learning_rate", "weight_decay", "gradient_clip", "eta_min", "lambda_gate",
                   "lambda_specialization", "max_train_batches", "max_val_batches", "threads", "log_interval")}
        config.update({
            "schema_version": 1, "model_class": "SharedGraphNDR", "model_config": model_config,
            "claim_scope": "smoke_only" if args.max_train_batches is not None or args.max_val_batches is not None else "source_validation_development_only",
            "selection_split": "source_val", "selection_metric": "mean_of_source_region_global_masked_mae",
            "objective_weighting": "region_uniform_mean_of_effective_batch_objectives",
            "runtime": runtime, "source_snapshot": source_snapshot(), "data_identity": data.identity,
            "source_statistics_identity": data.preprocessing.identity,
            "target_data_read": False, "test_evaluation": False, "early_stopping": False,
            "precision": "FP32", "complete_source_train_epoch": args.max_train_batches is None,
            "loading_policy": loading["policy"],
        })
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
        atomic_json(output / "source_snapshot.json", config["source_snapshot"])
        atomic_json(output / "data_identity.json", data.identity)
        atomic_json(output / "source_loading.json", loading)
    model = SharedGraphNDR(**config["model_config"]).to(device)
    if args.mode == "calibrate":
        result = calibrate_transfer(model=model, data=data, device=device, config=config, output=output,
                                    warmup=args.warmup_batches, timed=args.timed_batches)
    else:
        result = train_transfer_run(model=model, data=data, device=device, config=config, output=output,
                                    checkpoint=checkpoint, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
