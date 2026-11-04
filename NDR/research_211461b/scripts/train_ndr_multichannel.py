#!/usr/bin/env python3
"""Multichannel NDR-Net development using v3 event train and validation only.

The candidate adds occupancy and speed residual history to the original s0g backbone
input; gating, anchors, heads, loss and optimizer recipe stay the baseline ones. The
training loop, strict resume, checkpoints and independent verification are the
functions of the unmodified scripts/train_ndr_research.py, loaded as an independent
module instance whose four data and model touch points are bound to the multichannel
versions defined here. The baseline trainer and every research module recorded in the
stage-1 source snapshots stay byte-identical.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ndr_research.data import canonical_digest
from ndr_research.multichannel import (
    AUXILIARY_BATCH_KEYS,
    MULTICHANNEL_VARIANT,
    MultiChannelData,
    MultiChannelNDR,
    load_multichannel_data,
    make_multichannel_model,
    multichannel_factory_kwargs,
    prepare_multichannel_batch,
)

BASELINE_TRAINER = ROOT / "scripts" / "train_ndr_research.py"
CORE_COLLABORATORS = ("batch_tensors", "model_forward", "research_factory_kwargs", "make_research_model")
CONFIG_ARGUMENTS = (
    "region", "variant", "epochs", "seed", "batch_size", "microbatch_size",
    "learning_rate", "weight_decay", "gradient_clip", "eta_min", "lambda_gate",
    "lambda_specialization", "hidden_dim", "reliable_anchor", "max_reference_age",
    "max_train_batches", "max_val_batches", "log_interval", "threads",
)


def batch_tensors(batch: Mapping[str, Any], data: MultiChannelData,
                  device: torch.device) -> dict[str, torch.Tensor]:
    prepared = prepare_multichannel_batch(batch, data.statistics, data.auxiliary_statistics)
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


def model_forward(model: MultiChannelNDR, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not isinstance(model, MultiChannelNDR):
        raise TypeError("the multichannel trainer runs only MultiChannelNDR")
    prediction = model(tensors["flow_hist"], tensors["x_baseline"], tensors["y_baseline"],
                       tensors["time_feat"], *(tensors[key] for key in AUXILIARY_BATCH_KEYS))
    if prediction.shape != tensors["target"].shape or not torch.isfinite(prediction).all():
        raise FloatingPointError("forecast must be finite everywhere and match target shape")
    return prediction


def load_baseline_trainer(**collaborators: Any) -> ModuleType:
    """An independent instance of the unmodified baseline trainer with bound collaborators."""
    if set(collaborators) != set(CORE_COLLABORATORS):
        raise ValueError("bind exactly the batch, forward and factory collaborators")
    spec = importlib.util.spec_from_file_location("_ndr_multichannel_trainer_core", BASELINE_TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError("the baseline research trainer is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in collaborators.items():
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"the baseline trainer has no collaborator named {name}")
        setattr(module, name, value)
    return module


core = load_baseline_trainer(
    batch_tensors=batch_tensors, model_forward=model_forward,
    research_factory_kwargs=multichannel_factory_kwargs, make_research_model=make_multichannel_model,
)


def calibrate(model: MultiChannelNDR, data: MultiChannelData, device: torch.device,
              config: Mapping[str, Any], *, warmup: int, timed: int, output: Path) -> dict[str, Any]:
    """The baseline calibration, adding p90 batch times and the budget eligibility fields."""
    optimizer, scheduler = core.create_optimizer(model, config, data.train.num_batch)
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
        tensors = batch_tensors(data.train.next_batch(), data, device)
        core.optimize_batch(model, optimizer, scheduler, tensors, config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if number >= warmup:
            durations.append(time.perf_counter() - started)
    train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    train_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
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
            tensors = batch_tensors(data.val.next_batch(), data, device)
            for start in range(0, tensors["target"].shape[0], config["microbatch_size"]):
                chunk = {key: value[start:start + config["microbatch_size"]] for key, value in tensors.items()}
                model_forward(model, chunk)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if number >= warmup:
                val_durations.append(time.perf_counter() - started)
    train_mean, val_mean = float(np.mean(durations)), float(np.mean(val_durations))
    train_p90, val_p90 = float(np.percentile(durations, 90)), float(np.percentile(val_durations, 90))
    batches = {"train": data.train.num_batch, "val": data.val.num_batch}
    result = {
        "status": "calibrated_engineering_only", "claim_scope": "calibration_only",
        "warmup_batches": warmup, "timed_batches": timed,
        "mean_train_batch_seconds": train_mean, "mean_val_batch_seconds": val_mean,
        "p90_train_batch_seconds": train_p90, "p90_val_batch_seconds": val_p90,
        "timed_train_batch_seconds": durations, "timed_val_batch_seconds": val_durations,
        "estimated_train_val_seconds": config["epochs"] * (train_mean * batches["train"] + val_mean * batches["val"]),
        "p90_estimated_train_val_seconds": config["epochs"] * (train_p90 * batches["train"] + val_p90 * batches["val"]),
        "budget_epochs": config["epochs"], "train_num_batches": batches["train"], "val_num_batches": batches["val"],
        "train_peak_allocated_bytes": train_peak, "train_peak_reserved_bytes": train_reserved,
        "val_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "val_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "device_type": device.type, "cuda_budget_eligible": device.type == "cuda",
        "config_digest": canonical_digest(config), "test_evaluated": False,
    }
    core.atomic_json(output / "calibration.json", result)
    return result


def multichannel_recipe(data: MultiChannelData) -> dict[str, Any]:
    """Declared input-scope change; every other recipe field is the baseline's."""
    return {
        "auxiliary_channels": list(data.auxiliary_channels),
        "auxiliary_residual_std": [value.residual_std for value in data.auxiliary_statistics],
        "backbone_input": ("flow residual channel unchanged; occupancy and speed residuals added "
                           "through a zero-initialized bias-free projection to the flow input embedding"),
        "standardization": "train-only weekly median climatology and global observed residual std per channel",
        "missing_policy": "unobserved auxiliary points enter as zero residual",
        "gate_features": "flow only, unchanged",
        "anchors_heads_and_loss": "raw flow units, unchanged",
        "initial_function": "identical to the baseline s0g at construction",
        "training_loop": "unmodified scripts/train_ndr_research.py functions",
        "data_scope": "flow, occupancy and speed history; not a flow-only comparison",
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = core.argument_parser()
    parser.description = __doc__
    variant = parser._option_string_actions["--variant"]
    variant.choices = (MULTICHANNEL_VARIANT,)
    variant.default = MULTICHANNEL_VARIANT
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    core.validate_arguments(args)
    if args.variant != MULTICHANNEL_VARIANT or args.reliable_anchor:
        raise ValueError("this entrypoint trains only the multichannel variant with the original anchor")
    previous = core.load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = previous["config"] if previous is not None else None
    if saved is not None:
        if saved.get("variant") != MULTICHANNEL_VARIANT:
            raise ValueError("the checkpoint was not produced by the multichannel trainer")
        core.reject_identity_overrides(args, saved, arguments)
    seed, threads = (saved["seed"], saved["threads"]) if saved else (args.seed, args.threads)
    device = core.configure_runtime(args.device, seed, threads)
    runtime = core.runtime_identity(device)
    if saved is not None:
        core.require_snapshot(saved["source_snapshot"])
        if runtime != saved["runtime"]:
            raise ValueError("resume/verify runtime must exactly match the original runtime")
    data = load_multichannel_data(
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
        result = core.verify_checkpoint(checkpoint_path=args.checkpoint, data=data, device=device,
                                        output=output, current_runtime=runtime)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if previous:
        config = saved
        factory = previous["model_factory"] | {"device": str(device)}
    else:
        config = {name: getattr(args, name) for name in CONFIG_ARGUMENTS}
        config.update({"schema_version": 1, "purpose": "v3_event_train_validation_development",
                       "claim_scope": "smoke_only" if args.max_train_batches is not None or args.max_val_batches is not None else "validation_development_only",
                       "source_snapshot": core.source_snapshot(), "runtime": runtime,
                       "data_identity": data.identity, "precision": "FP32",
                       "early_stopping": False, "test_evaluation": False,
                       "val_selection": core.evaluation_identity(data, args.max_val_batches),
                       "multichannel_recipe": multichannel_recipe(data)})
        factory = {"node_count": data.region.N, "variant": args.variant,
                   "residual_std": data.statistics.residual_std,
                   "residual_std_by_node": data.statistics.residual_std_by_node.tolist(),
                   "flow_mean": data.statistics.flow_mean, "flow_std": data.statistics.flow_std,
                   "reliable_anchor": args.reliable_anchor, "max_reference_age": args.max_reference_age,
                   "hidden_dim": args.hidden_dim,
                   "auxiliary_channels": list(data.auxiliary_channels),
                   "auxiliary_residual_std": [value.residual_std for value in data.auxiliary_statistics],
                   "device": str(device)}
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        core.atomic_json(output / "config.json", config)
        core.atomic_json(output / "source_snapshot.json", config["source_snapshot"])
        core.atomic_json(output / "data_identity.json", data.identity)
    model = make_multichannel_model(**factory)
    if args.mode == "calibrate":
        result = calibrate(model, data, device, config, warmup=args.warmup_batches,
                           timed=args.timed_batches, output=output)
    else:
        result = core.train_run(model=model, data=data, device=device, config=config, output=output,
                                checkpoint=previous, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
