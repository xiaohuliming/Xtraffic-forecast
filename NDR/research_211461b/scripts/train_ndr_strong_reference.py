#!/usr/bin/env python3
"""Resumable same-region STAEformer/BiST controls with train/validation-only v3 data."""
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
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dist_net.metrics import MaskedMAEAccumulator
from dist_net.protocol import git_worktree_provenance
from ndr_research.data import ResearchData, array_digest, canonical_digest, load_research_data, sha256_file
from ndr_research.strong_references import METHODS, PLUGIN_LOSS, PLUGIN_METHODS, RECIPES, make_strong_reference, reference_recipe
from scripts.train_ndr_research import (
    atomic_json, batch_tensors, capture_rng_state, configure_runtime, empty_partial,
    evaluation_identity, require_snapshot, restore_rng_state, runtime_identity,
    write_history,
)
from scripts.train_staeformer_deseason import build_ndr_loss_normalizers, compute_loss

SCHEMA = "strong_reference_checkpoint_v1"


def source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Snapshot this runner's local import closure, not unrelated research modules."""
    root = Path(root).resolve()
    pending = [root / "scripts/train_ndr_strong_reference.py", root / "ndr_research/strong_references.py"]
    files: set[Path] = set()

    def enqueue_module(parts: tuple[str, ...]) -> None:
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    enqueue_module(tuple(item.name.split(".")))
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else ()
                module = base + (tuple(node.module.split(".")) if node.module else ())
                if module:
                    enqueue_module(module)
                    for item in node.names:
                        if item.name != "*":
                            enqueue_module(module + tuple(item.name.split(".")))
    # baselines/GraphWaveNet/model.py is reached by a bare ``import model`` inside
    # fourier_dual_net/rgdn.py, which the AST closure cannot see; the plugin loss
    # module imports rgdn, so the file is declared explicitly.
    for relative in ("config/reproduction/igstgnn_table4/profiles.public_material_v2.json",
                     "scripts/reproduction/igstgnn_table4/upstreams.lock.json",
                     "baselines/GraphWaveNet/model.py"):
        path = root / relative
        if path.is_file():
            files.add(path)
    declarations = {path.relative_to(root).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(files)}
    if not declarations:
        raise ValueError("strong-reference source dependency closure is empty")
    return {"files": declarations, "files_digest": canonical_digest(declarations),
            "git": git_worktree_provenance(root), "scope": "strong_runner_static_local_import_closure_and_locked_profiles"}


def is_plugin_model(model: Any) -> bool:
    return getattr(model, "method", None) in PLUGIN_METHODS


def plugin_loss_weights(recipe: Mapping[str, Any], *, completed_updates: int | None = None,
                        planned_updates: int | None = None) -> dict[str, Any]:
    """Resolve the gate distillation and specialization weights for the next effective update.

    The original contract is constant. The anneal variant interpolates linearly from the
    initial values at zero completed updates to the final values once the completed count
    reaches ``ceil(end_fraction * planned_updates)``, and stays there. Because the count is
    the checkpointed ``progress["updates"]``, a strict resume continues the schedule exactly.
    """
    plugin_loss = recipe.get("plugin_loss") or PLUGIN_LOSS
    schedule = plugin_loss.get("schedule") or {"name": "constant"}
    if schedule["name"] == "constant":
        return {"schedule": "constant", "completed_updates": completed_updates,
                "lambda_gate": float(plugin_loss["lambda_gate"]),
                "lambda_specialization": float(plugin_loss["lambda_specialization"])}
    if schedule["name"] != "linear_anneal_to_zero":
        raise ValueError(f"unknown plugin loss schedule {schedule['name']!r}")
    if type(completed_updates) is not int or completed_updates < 0 or type(planned_updates) is not int or planned_updates <= 0:
        raise ValueError("annealed plugin losses need the completed and planned effective update counts")
    fraction_of_plan = float(schedule["end_fraction_of_planned_updates"])
    if not 0.0 < fraction_of_plan <= 1.0:
        raise ValueError("the anneal end must be a positive fraction of the planned updates")
    end = max(1, math.ceil(fraction_of_plan * planned_updates))
    fraction = min(1.0, completed_updates / end)
    weights = {}
    for name in ("lambda_gate", "lambda_specialization"):
        initial, final = float(schedule[f"{name}_initial"]), float(schedule[f"{name}_final"])
        weights[name] = initial + (final - initial) * fraction
    return {"schedule": schedule["name"], "completed_updates": completed_updates,
            "planned_total_updates": planned_updates, "anneal_end_updates": end,
            "anneal_fraction": fraction, **weights}


def strong_forward(model: Any, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if is_plugin_model(model):
        prediction = model(tensors["flow_hist"], tensors["time_feat"], tensors["x_baseline"], tensors["y_baseline"])
    else:
        prediction = model(tensors["flow_hist"], tensors["time_feat"])
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("strong-reference forecasts must be finite and match the full horizon")
    return prediction


def _backward_plugin_batch(model: Any, tensors: Mapping[str, torch.Tensor], *, microbatch_size: int,
                           lambda_gate: float, lambda_specialization: float) -> dict[str, Any]:
    """Original NDR objective on the BiST plugin with exact full-effective-batch normalizers.

    The gate distillation and specialization denominators are fixed from the complete
    effective batch before activation microbatching, exactly as the verified pair runner
    and the original s0g microbatch contract do, so the summed microbatch losses equal
    the single full-batch loss.
    """
    plugin = model.plugin
    full_loss_batch = {"y_true": tensors["target"].unsqueeze(-1), "y_mask": tensors["target_mask"].unsqueeze(-1)}
    with torch.no_grad():
        normalizers = build_ndr_loss_normalizers(
            plugin, full_loss_batch, baseline=tensors["y_baseline"],
            persistence=tensors["flow_hist"][:, :, -1:].expand_as(tensors["y_baseline"]),
            device=tensors["target"].device,
        )
    loss_args = SimpleNamespace(lambda_gate=lambda_gate, lambda_spec=lambda_specialization)
    metric = MaskedMAEAccumulator()
    objective = 0.0
    parts: dict[str, float] = {}
    for start in range(0, tensors["target"].shape[0], microbatch_size):
        chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
        prediction = strong_forward(model, chunk)
        mask = chunk["target_mask"].bool()
        metric.update(prediction, chunk["target"], mask)
        if not mask.any():
            continue
        loss, local_parts = compute_loss(
            plugin, prediction, {"y_true": chunk["target"].unsqueeze(-1), "y_mask": chunk["target_mask"].unsqueeze(-1)},
            loss_args, tensors["target"].device, normalizers=normalizers,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("plugin objective is nonfinite")
        loss.backward()
        objective += float(loss.detach())
        for key, value in local_parts.items():
            parts[key] = parts.get(key, 0.0) + float(value)
    return {"error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
            "objective": objective, "updated": True, "loss_parts": parts}


def backward_reference_batch(model: Any, tensors: Mapping[str, torch.Tensor], *, microbatch_size: int,
                             lambda_gate: float = PLUGIN_LOSS["lambda_gate"],
                             lambda_specialization: float = PLUGIN_LOSS["lambda_specialization"]) -> dict[str, Any]:
    count = int(tensors["target_mask"].sum().item())
    if microbatch_size <= 0:
        raise ValueError("microbatch size must be positive")
    if count == 0:
        return {"error_sum": 0.0, "valid_count": 0, "objective": 0.0, "updated": False}
    if is_plugin_model(model):
        return _backward_plugin_batch(model, tensors, microbatch_size=microbatch_size,
                                      lambda_gate=lambda_gate, lambda_specialization=lambda_specialization)
    metric = MaskedMAEAccumulator()
    objective = 0.0
    for start in range(0, tensors["target"].shape[0], microbatch_size):
        chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
        prediction = strong_forward(model, chunk)
        mask = chunk["target_mask"].bool()
        metric.update(prediction, chunk["target"], mask)
        if not mask.any():
            continue
        loss = (prediction[mask] - chunk["target"][mask]).abs().sum() / count
        if not torch.isfinite(loss):
            raise FloatingPointError("reference main objective is nonfinite")
        loss.backward()
        objective += float(loss.detach())
    return {"error_sum": metric._absolute_error_sum, "valid_count": metric.valid_count,
            "objective": objective, "updated": True}


def create_reference_optimizer(model: Any, config: Mapping[str, Any], batches: int) -> tuple[Any, Any]:
    recipe = config["training_recipe"]
    options = recipe["optimizer"]
    regularization = recipe.get("plugin_regularization")
    if regularization:
        # The regularize variant keeps the native BiST weight decay and raises it only on the
        # added plugin parameters, as two explicit Adam parameter groups.
        groups = model.parameter_groups()
        declared = regularization["parameter_groups"]
        if set(groups) != set(declared) or set(declared) != {"native_bist", "ndr_plugin"}:
            raise ValueError("plugin regularization parameter groups do not match the model partition")
        if sum(len(group) for group in groups.values()) != sum(1 for _ in model.parameters()) or not groups["ndr_plugin"]:
            raise ValueError("plugin regularization must partition every parameter exactly once")
        if declared["native_bist"]["weight_decay"] != options["weight_decay"]:
            raise ValueError("the native BiST group must keep the recipe weight decay")
        parameters = [{"params": groups[name], "weight_decay": float(declared[name]["weight_decay"]), "name": name}
                      for name in ("native_bist", "ndr_plugin")]
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=options["learning_rate"],
                                 weight_decay=options["weight_decay"], eps=options["epsilon"])
    schedule = options["scheduler"]
    if schedule["name"] == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule["milestones"], gamma=schedule["gamma"])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] * batches, eta_min=schedule["eta_min"])
    return optimizer, scheduler


def optimize_reference(model: Any, optimizer: Any, scheduler: Any, tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any],
                       *, completed_updates: int | None = None, planned_updates: int | None = None) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    used_lr = float(optimizer.param_groups[0]["lr"])
    weights = plugin_loss_weights(config["training_recipe"], completed_updates=completed_updates,
                                  planned_updates=planned_updates) if is_plugin_model(model) else None
    result = backward_reference_batch(model, tensors, microbatch_size=config["microbatch_size"],
                                      lambda_gate=weights["lambda_gate"] if weights else PLUGIN_LOSS["lambda_gate"],
                                      lambda_specialization=weights["lambda_specialization"] if weights else PLUGIN_LOSS["lambda_specialization"])
    result["learning_rate_used"] = used_lr if result["updated"] else None
    result["plugin_loss_weights"] = weights
    if result["updated"]:
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("reference gradients are absent/nonfinite")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training_recipe"]["optimizer"]["gradient_clip"])
        optimizer.step()
        if config["training_recipe"]["scheduler_step_unit"] == "completed_effective_update":
            scheduler.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
            raise FloatingPointError("reference optimizer produced nonfinite parameters/buffers")
    return result


def evaluate_reference(model: Any, data: ResearchData, device: torch.device, *, microbatch_size: int, max_batches: int | None = None) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    accumulator = MaskedMAEAccumulator()
    starts = []
    target_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    errors = np.zeros(12, dtype=np.float64)
    counts = np.zeros(12, dtype=np.int64)
    data.val.start_epoch()
    batches = min(data.val.num_batch, max_batches or data.val.num_batch)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch = data.val.next_batch()
                starts.extend(np.asarray(batch["sample_start"], dtype=np.int64).tolist())
                tensors = batch_tensors(batch, data, device)
                target_hash.update(tensors["target"].cpu().contiguous().numpy().tobytes())
                mask_hash.update(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
                for start in range(0, tensors["target"].shape[0], microbatch_size):
                    chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
                    prediction = strong_forward(model, chunk)
                    valid = chunk["target_mask"].bool()
                    accumulator.update(prediction, chunk["target"], valid)
                    error = torch.where(valid, (prediction - chunk["target"]).abs(), 0.0)
                    errors += error.double().sum((0, 1)).cpu().numpy()
                    counts += valid.sum((0, 1)).cpu().numpy()
        selection = evaluation_identity(data, max_batches)
        if array_digest(np.asarray(starts, dtype=np.int64)) != selection["forecast_origins_sha256"]:
            raise ValueError("reference validation origin collection is inconsistent")
        horizon = [float(error / count) if count else None for error, count in zip(errors, counts)]
        return {"mae": accumulator.compute(), "global_masked_mae": accumulator.compute(),
                "absolute_error_sum": accumulator._absolute_error_sum, "valid_count": accumulator.valid_count,
                "horizon_mae": horizon, "horizon_valid_counts": counts.tolist(),
                "average_horizon_mae": float(np.mean(horizon)) if counts.all() else None,
                "selection": selection, "observed_targets_sha256": target_hash.hexdigest(),
                "observation_masks_sha256": mask_hash.hexdigest(), "finite_predictions_everywhere": True}
    finally:
        model.train(was_training)


def _loader_state(loader: Any) -> dict[str, Any]:
    value = loader.state_dict()
    value["order"] = torch.from_numpy(value["order"].copy())
    return value


def save_reference_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any,
                              data: ResearchData, config: Mapping[str, Any], progress: Mapping[str, Any],
                              selection_metric: Mapping[str, Any] | None = None) -> None:
    payload = {"schema_version": SCHEMA, "config": dict(config), "config_digest": canonical_digest(config),
               "model_factory": model.factory_kwargs(), "model_upstream": copy.deepcopy(model.upstream),
               "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng": capture_rng_state(), "loaders": {name: _loader_state(getattr(data, name)) for name in ("train", "val")},
               "data_identity": data.identity, "progress": copy.deepcopy(progress),
               "selection_metric": copy.deepcopy(selection_metric), "resume_boundary": "completed_effective_batch_or_epoch"}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": digest, "size_bytes": path.stat().st_size, "config_digest": payload["config_digest"]})
    finally:
        temporary.unlink(missing_ok=True)


def load_reference_checkpoint(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if sha256_file(path) != marker["sha256"] or path.stat().st_size != marker["size_bytes"]:
        raise ValueError("strong-reference checkpoint byte identity mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {"config", "config_digest", "model_factory", "model_upstream", "model", "optimizer", "scheduler", "rng", "loaders", "data_identity", "progress", "selection_metric"}
    if value.get("schema_version") != SCHEMA or not required.issubset(value):
        raise ValueError("strong-reference checkpoint is not complete/resumable")
    if canonical_digest(value["config"]) != value["config_digest"] or marker["config_digest"] != value["config_digest"]:
        raise ValueError("strong-reference recipe identity mismatch")
    return value


def _require_model_identity(model: Any, checkpoint: Mapping[str, Any]) -> None:
    observed, expected = copy.deepcopy(model.upstream), copy.deepcopy(checkpoint["model_upstream"])
    # A pristine locked upstream may be mounted at a different path for verification.
    observed.pop("checkout", None)
    expected.pop("checkout", None)
    if observed != expected:
        raise ValueError("strong-reference locked model/upstream identity mismatch")


def restore_reference(checkpoint: Mapping[str, Any], model: Any, optimizer: Any, scheduler: Any,
                      data: ResearchData, config: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint["config_digest"] != canonical_digest(config) or checkpoint["data_identity"] != data.identity:
        raise ValueError("strong-reference strict resume config/cache/split identity mismatch")
    require_snapshot(config["source_snapshot"])
    _require_model_identity(model, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for name in ("train", "val"):
        getattr(data, name).load_state_dict(checkpoint["loaders"][name])
    restore_rng_state(checkpoint["rng"])
    return copy.deepcopy(checkpoint["progress"])


def train_reference(*, model: Any, data: ResearchData, device: torch.device,
                    config: Mapping[str, Any], output: Path, checkpoint: Mapping[str, Any] | None = None,
                    max_wallclock_seconds: float | None = None) -> dict[str, Any]:
    train_batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    planned_updates = int(config["epochs"]) * int(train_batches)
    optimizer, scheduler = create_reference_optimizer(model, config, train_batches)
    progress = {"epoch": 0, "stage": "start", "best_epoch": None, "best_val_mae": None,
                "updates": 0, "history": [], "partial_epoch": empty_partial(), "elapsed_seconds": 0.0,
                "planned_total_updates": planned_updates, "plugin_loss_schedule": None}
    if checkpoint is not None:
        progress = restore_reference(checkpoint, model, optimizer, scheduler, data, config)
    stop = {"requested": False}
    handlers = {}
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
        save_reference_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)

    try:
        while progress["epoch"] < config["epochs"]:
            if progress["stage"] == "start":
                data.train.start_epoch()
                progress["partial_epoch"] = empty_partial()
                progress["stage"] = "train"
            while progress["stage"] == "train" and data.train.current_ind < train_batches:
                if stopped():
                    save_last()
                    result = {"status": "stopped_resumable", "completed_epochs": progress["epoch"], "next_batch": data.train.current_ind, "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config)}
                    atomic_json(output / "run_state.json", result)
                    return result
                step_started = time.monotonic()
                tensors = batch_tensors(data.train.next_batch(), data, device)
                step = optimize_reference(model, optimizer, scheduler, tensors, config,
                                          completed_updates=int(progress["updates"]), planned_updates=planned_updates)
                partial = progress["partial_epoch"]
                partial["error_sum"] += step["error_sum"]
                partial["valid_count"] += step["valid_count"]
                partial["objective_sum"] += step["objective"]
                partial["optimized_batches"] += int(step["updated"])
                if step["updated"]:
                    partial["learning_rate_last_update"] = step["learning_rate_used"]
                    progress["plugin_loss_schedule"] = step["plugin_loss_weights"]
                partial["batches"] += 1
                partial["seconds"] += time.monotonic() - step_started
                progress["updates"] += int(step["updated"])
            progress["stage"] = "validate"
            if stopped():
                save_last()
                result = {"status": "stopped_resumable", "completed_epochs": progress["epoch"], "next_stage": "validate", "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config)}
                atomic_json(output / "run_state.json", result)
                return result
            validation = evaluate_reference(model, data, device, microbatch_size=config["microbatch_size"], max_batches=config["max_val_batches"])
            partial = progress["partial_epoch"]
            if not partial["valid_count"]:
                raise ValueError("strong-reference training epoch has no valid observations")
            used_lr = partial.get("learning_rate_last_update")
            if config["training_recipe"]["scheduler_step_unit"] == "completed_epoch":
                scheduler.step()
            progress["epoch"] += 1
            progress["stage"] = "start"
            row = {"epoch": progress["epoch"], "train_mae": partial["error_sum"] / partial["valid_count"],
                   "train_valid_count": partial["valid_count"], "validation": validation,
                   "learning_rate_used": used_lr, "learning_rate_next": optimizer.param_groups[0]["lr"], "updates": progress["updates"],
                   "plugin_loss_weights_last_update": copy.deepcopy(progress.get("plugin_loss_schedule"))}
            progress["history"].append(row)
            improved = progress["best_val_mae"] is None or validation["mae"] < progress["best_val_mae"]
            if improved:
                progress["best_val_mae"], progress["best_epoch"] = validation["mae"], progress["epoch"]
                save_reference_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress, selection_metric=validation)
            save_last()
            write_history(output, progress["history"])
            print(json.dumps(row, sort_keys=True), flush=True)
        result = {"status": "training_completed_validation_pending_independent_verification",
                  "completed_epochs": progress["epoch"], "best_epoch": progress["best_epoch"],
                  "best_val_mae": progress["best_val_mae"], "updates": progress["updates"],
                  "planned_total_updates": planned_updates,
                  "plugin_loss_schedule_last_update": copy.deepcopy(progress.get("plugin_loss_schedule")),
                  "claim_scope": config["claim_scope"], "config_digest": canonical_digest(config), "test_evaluated": False}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def verify_reference(*, checkpoint_path: Path, data: ResearchData, device: torch.device,
                     output: Path, runtime: Mapping[str, Any], bist_checkout: Path | None = None) -> dict[str, Any]:
    checkpoint = load_reference_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if os.getpid() == config["training_process_id"]:
        raise ValueError("verification must reload the checkpoint in an independent process")
    require_snapshot(config["source_snapshot"])
    if checkpoint["data_identity"] != data.identity or dict(runtime) != config["runtime"]:
        raise ValueError("independent verification data/runtime identity mismatch")
    expected = checkpoint["selection_metric"]
    if expected is None or expected["selection"] != evaluation_identity(data, config["max_val_batches"]):
        raise ValueError("verification must use the selected checkpoint's exact validation collection")
    kwargs = checkpoint["model_factory"] | {"device": str(device)}
    if bist_checkout is not None:
        kwargs["bist_checkout"] = bist_checkout
    model = make_strong_reference(**kwargs)
    _require_model_identity(model, checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate_reference(model, data, device, microbatch_size=config["microbatch_size"], max_batches=config["max_val_batches"])
    for key in ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256"):
        if actual[key] != expected[key]:
            raise ValueError("independent validation observation collection differs")
    if not math.isclose(actual["mae"], expected["mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent validation metrics do not reproduce")
    if config["claim_scope"] != "smoke_only" and not actual["selection"]["all_validation_samples"]:
        raise ValueError("formal development verification must cover full source validation")
    result = {"status": "verified_smoke_only" if config["claim_scope"] == "smoke_only" else "verified_complete_source_validation_development",
              "claim_scope": config["claim_scope"], "validation": actual,
              "checkpoint_sha256": sha256_file(checkpoint_path), "config_digest": checkpoint["config_digest"],
              "training_process_id": config["training_process_id"], "verification_process_id": os.getpid(),
              "source_files_digest": config["source_snapshot"]["files_digest"], "test_evaluated": False}
    atomic_json(output / "verification.json", result)
    return result


def _next_full_reference_batch(loader: Any) -> tuple[Mapping[str, Any], int]:
    """Skip retained tails for calibration while leaving actual epoch training intact."""
    if loader.size < loader.bs:
        raise ValueError("calibration needs at least one complete effective batch per split")
    skipped = 0
    while True:
        try:
            batch = loader.next_batch()
        except StopIteration:
            loader.start_epoch()
            continue
        if batch["x_data"].shape[0] == loader.bs:
            return batch, skipped
        skipped += 1


def _calibration_validation_batch(model: Any, tensors: Mapping[str, torch.Tensor], microbatch_size: int) -> int:
    """Exercise the full validation reduction/transfer path without reporting scores."""
    accumulator = MaskedMAEAccumulator()
    # Match the production validation's observation-digest and horizon-reduction work.
    hashlib.sha256(tensors["target"].cpu().contiguous().numpy().tobytes())
    hashlib.sha256(tensors["target_mask"].cpu().contiguous().numpy().tobytes())
    for start in range(0, tensors["target"].shape[0], microbatch_size):
        chunk = {key: value[start:start + microbatch_size] for key, value in tensors.items()}
        prediction = strong_forward(model, chunk)
        valid = chunk["target_mask"].bool()
        accumulator.update(prediction, chunk["target"], valid)
        errors = torch.where(valid, (prediction - chunk["target"]).abs(), 0.0)
        errors.double().sum((0, 1)).cpu().numpy()
        valid.sum((0, 1)).cpu().numpy()
    return accumulator.valid_count


def calibrate_reference(*, model: Any, data: ResearchData, device: torch.device,
                        config: Mapping[str, Any], warmup_batches: int, timed_batches: int,
                        output: Path) -> dict[str, Any]:
    if type(warmup_batches) is not int or warmup_batches < 0 or type(timed_batches) is not int or timed_batches <= 0:
        raise ValueError("calibration needs nonnegative warmup and positive timed batch counts")
    if data.train.bs != config["batch_size"] or data.val.bs != config["batch_size"]:
        raise ValueError("calibration loaders must match the declared complete effective batch")
    optimizer, scheduler = create_reference_optimizer(model, config, data.train.num_batch)
    planned_updates = int(config["epochs"]) * int(data.train.num_batch)
    completed_updates = 0
    summaries = {}
    was_training = model.training
    try:
        for phase, loader in (("train", data.train), ("val", data.val)):
            loader.start_epoch()
            model.train(phase == "train")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            durations, batch_sizes = [], []
            skipped_tails = valid_count = 0
            for number in range(warmup_batches + timed_batches):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                batch, skipped = _next_full_reference_batch(loader)
                tensors = batch_tensors(batch, data, device)
                size = int(tensors["flow_hist"].shape[0])
                if size != config["batch_size"]:
                    raise ValueError("calibration measured an incomplete effective batch")
                if phase == "train":
                    # Only plugin models consume the update counts; native controls keep the original call.
                    counts = {"completed_updates": completed_updates, "planned_updates": planned_updates} if is_plugin_model(model) else {}
                    step = optimize_reference(model, optimizer, scheduler, tensors, config, **counts)
                    completed_updates += int(step["updated"])
                    observed = int(step["valid_count"])
                else:
                    with torch.no_grad():
                        observed = _calibration_validation_batch(model, tensors, config["microbatch_size"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("calibration clock produced an invalid complete-batch duration")
                skipped_tails += skipped
                if number >= warmup_batches:
                    durations.append(elapsed)
                    batch_sizes.append(size)
                    valid_count += observed
            summaries[phase] = {
                "warmup_batches": warmup_batches, "timed_batches": len(durations),
                "timed_batch_seconds": durations, "mean_batch_seconds": float(np.mean(durations)),
                "p90_batch_seconds": float(np.percentile(durations, 90)),
                "timed_effective_batch_sizes": batch_sizes, "timed_valid_count": valid_count,
                "skipped_retained_tails": skipped_tails,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
            }
    finally:
        model.train(was_training)
    train_mean, val_mean = summaries["train"]["mean_batch_seconds"], summaries["val"]["mean_batch_seconds"]
    budget_epochs = int(config.get("requested_epochs", config["epochs"]))
    estimate = budget_epochs * (train_mean * data.train.num_batch + val_mean * data.val.num_batch)
    p90_estimate = budget_epochs * (summaries["train"]["p90_batch_seconds"] * data.train.num_batch + summaries["val"]["p90_batch_seconds"] * data.val.num_batch)
    result = {
        "status": "calibrated_cuda_full_effective_batches" if device.type == "cuda" else "calibrated_cpu_engineering_only",
        "claim_scope": "calibration_only", "cuda_budget_eligible": device.type == "cuda",
        "device_type": device.type, "phases": summaries,
        "warmup_batches": warmup_batches, "timed_batches": timed_batches,
        "mean_train_batch_seconds": train_mean, "mean_val_batch_seconds": val_mean,
        "p90_train_batch_seconds": summaries["train"]["p90_batch_seconds"],
        "p90_val_batch_seconds": summaries["val"]["p90_batch_seconds"],
        "train_peak_allocated_bytes": summaries["train"]["peak_allocated_bytes"],
        "train_peak_reserved_bytes": summaries["train"]["peak_reserved_bytes"],
        "val_peak_allocated_bytes": summaries["val"]["peak_allocated_bytes"],
        "val_peak_reserved_bytes": summaries["val"]["peak_reserved_bytes"],
        "budget_epochs": budget_epochs, "train_num_batches": data.train.num_batch, "val_num_batches": data.val.num_batch,
        "estimated_train_val_seconds": estimate, "p90_estimated_train_val_seconds": p90_estimate,
        "estimate_formula": "epochs * (mean_train_batch_seconds * train_num_batches + mean_val_batch_seconds * val_num_batches)",
        "memory_peak_scope": "warmup_and_timed_phase_including_persistent_optimizer_state",
        "timing_scope": "data_loading_transfer_full_effective_model_step_and_validation_reductions",
        "prediction_error_used_for_configuration_selection": False,
        "batch_normalization_scope": config["training_recipe"]["batch_normalization_scope"],
        "config_digest": canonical_digest(config), "test_evaluated": False,
    }
    atomic_json(output / "calibration.json", result)
    return result


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "verify", "calibrate"), default="train")
    parser.add_argument("--method", choices=METHODS, default="staeformer")
    parser.add_argument("--recipe", choices=RECIPES, default="native")
    parser.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    parser.add_argument("--epochs", type=int, choices=(30, 100), default=30)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
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
    if args.threads <= 0 or args.smoke_epochs <= 0:
        raise ValueError("thread/smoke epoch counts must be positive")
    if args.warmup_batches < 0 or args.timed_batches <= 0:
        raise ValueError("calibration warmup must be nonnegative and timed batches positive")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("independent verification requires --checkpoint")
    if args.checkpoint is not None and args.mode != "verify":
        raise ValueError("--checkpoint is reserved for independent verification")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("strict resume requires train mode and last.pt")
    for name in ("max_train_batches", "max_val_batches"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            raise ValueError("smoke batch limits must be positive")
        if getattr(args, name) is not None and not args.smoke:
            raise ValueError("subset overrides require explicit --smoke and cannot become formal results")
    if args.device.startswith("cpu") and not args.smoke and args.resume is None and args.mode != "verify":
        raise ValueError("CPU is restricted to explicit smoke; CUDA fallback is forbidden")
    if args.max_wallclock_seconds is not None and (not math.isfinite(args.max_wallclock_seconds) or args.max_wallclock_seconds <= 0):
        raise ValueError("wallclock limit must be finite and positive")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    validate_arguments(args)
    previous = load_reference_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous is not None else None
    if saved:
        mutable = {"mode", "device", "bist_checkout", "data_dir", "graph_dir", "output_dir", "checkpoint", "resume", "max_wallclock_seconds", "warmup_batches", "timed_batches"}
        for name, value in vars(args).items():
            option = "--" + name.replace("_", "-")
            expected = saved.get("requested_epochs") if name == "epochs" else saved.get(name)
            if name not in mutable and any(item == option or item.startswith(option + "=") for item in arguments) and value != expected:
                raise ValueError(f"strict resume/verify cannot change frozen recipe: {option}")
    device = configure_runtime(args.device, saved["seed"] if saved else args.seed, saved["threads"] if saved else args.threads)
    runtime = runtime_identity(device)
    if saved and runtime != saved["runtime"]:
        raise ValueError("strict resume/verify must reproduce the original runtime")
    if device.type == "cpu" and (saved["claim_scope"] if saved else "smoke_only" if args.smoke else "formal") != "smoke_only":
        raise ValueError("CPU execution cannot be classified as formal reference development")
    recipe = saved["training_recipe"] if saved else reference_recipe(args.method, args.recipe, region=args.region, epochs=args.epochs, batch_size=args.batch_size, microbatch_size=args.microbatch_size)
    data = load_research_data(region_name=saved["region"] if saved else args.region, data_dir=args.data_dir, graph_dir=args.graph_dir,
                              batch_size=recipe["batch_size"], seed=saved["seed"] if saved else args.seed, preload=not (saved["lazy"] if saved else args.lazy))
    output = args.output_dir or ((args.resume or args.checkpoint).parent if previous else None)
    if output is None:
        raise ValueError("a new reference run requires a regional independent output directory")
    if args.resume is not None and output.resolve() != args.resume.parent.resolve():
        raise ValueError("strict resume must use the original output directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify_reference(checkpoint_path=args.checkpoint, data=data, device=device, output=output, runtime=runtime, bist_checkout=args.bist_checkout)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    factory = previous["model_factory"] if previous else {"method": args.method, "node_count": data.region.N,
               "flow_mean": data.statistics.flow_mean, "flow_std": data.statistics.flow_std, "bist_checkout": args.bist_checkout}
    if not previous and args.method in PLUGIN_METHODS:
        # The plugin anchors and loss scales are the same train-only climatology statistics
        # the NDR research trainer records; they are frozen into the checkpoint factory.
        factory.update({"residual_std": data.statistics.residual_std,
                        "residual_std_by_node": data.statistics.residual_std_by_node.tolist()})
    if previous and args.bist_checkout is not None:
        factory = factory | {"bist_checkout": args.bist_checkout}
    model = make_strong_reference(**(factory | {"device": str(device)}))
    if saved:
        config = saved
    else:
        config = {name: getattr(args, name) for name in ("method", "recipe", "region", "seed", "threads", "lazy", "smoke", "smoke_epochs", "max_train_batches", "max_val_batches", "warmup_batches", "timed_batches")}
        config.update({"schema_version": SCHEMA, "epochs": args.smoke_epochs if args.smoke else args.epochs,
                       "requested_epochs": args.epochs, "batch_size": recipe["batch_size"], "microbatch_size": recipe["microbatch_size"],
                       "training_recipe": recipe, "claim_scope": "smoke_only" if args.smoke else "validation_development_only",
                       "training_process_id": os.getpid(), "runtime": runtime, "data_identity": data.identity,
                       "source_snapshot": source_snapshot(), "model_upstream": model.upstream,
                       "test_loader_constructed": False, "test_evaluated": False,
                       "validation_selection": evaluation_identity(data, args.max_val_batches)})
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config)
        atomic_json(output / "data_identity.json", data.identity)
        atomic_json(output / "source_snapshot.json", config["source_snapshot"])
    if args.mode == "calibrate":
        result = calibrate_reference(model=model, data=data, device=device, config=config,
                                     warmup_batches=args.warmup_batches, timed_batches=args.timed_batches, output=output)
    else:
        result = train_reference(model=model, data=data, device=device, config=config, output=output, checkpoint=previous, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
