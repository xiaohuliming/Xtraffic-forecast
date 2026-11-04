#!/usr/bin/env python3
"""Independent masked v3 DualCast-S train/verify/calibrate; no test loader."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
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
from ndr_research.data import canonical_digest, load_research_data, sha256_file
from ndr_research.dualcast_training import RECIPES, compute_dualcast_loss, dualcast_recipe, make_dualcast_model, regional_adjacency
from scripts.train_ndr_research import atomic_json, batch_tensors, capture_rng_state, configure_runtime, evaluation_identity, require_snapshot, runtime_identity, write_history
from scripts.train_ndr_strong_reference import (
    _calibration_validation_batch, _next_full_reference_batch,
    evaluate_reference, restore_reference,
)

SCHEMA = "dualcast_v3_complete_checkpoint_v1"


def source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    pending = [root / "scripts/train_dualcast_v3.py", root / "ndr_research/dualcast_training.py"]
    paths = set()

    def module(parts):
        for length in range(1, len(parts) + 1):
            package = root.joinpath(*parts[:length], "__init__.py")
            if package.is_file():
                pending.append(package)
        path = root.joinpath(*parts).with_suffix(".py")
        if path.is_file():
            pending.append(path)

    while pending:
        path = pending.pop().resolve()
        if path in paths or not path.is_file() or not path.is_relative_to(root):
            continue
        paths.add(path)
        package = path.relative_to(root).parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for value in node.names:
                    module(tuple(value.name.split(".")))
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else ()
                name = base + (tuple(node.module.split(".")) if node.module else ())
                if name:
                    module(name)
                    for value in node.names:
                        if value.name != "*":
                            module(name + tuple(value.name.split(".")))
    for name in ("config/reproduction/igstgnn_table4/profiles.public_material_v2.json", "scripts/reproduction/igstgnn_table4/upstreams.lock.json"):
        if (root / name).is_file():
            paths.add(root / name)
    files = {path.relative_to(root).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in sorted(paths)}
    return {"files": files, "files_digest": canonical_digest(files), "git": git_worktree_provenance(root),
            "scope": "independent_dualcast_entrypoint_static_import_closure"}


def create_optimizer(model: Any, config: Mapping[str, Any]) -> tuple[Any, Any]:
    value = config["training_recipe"]
    options = value["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"], eps=options["epsilon"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=value["scheduler"]["step_size"], gamma=value["scheduler"]["gamma"])
    return optimizer, scheduler


def optimize_batch(model: Any, optimizer: Any, tensors: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(tensors["target_mask"].sum())
    if tensors["flow_hist"].shape[0] > config["batch_size"]:
        raise ValueError("DualCast auxiliary batch is larger than its frozen full-effective batch")
    if not count:
        return {"updated": False, "valid_count": 0, "error_sum": 0.0, "objective": 0.0, "omitted_undefined_auxiliary": {}}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(tensors["flow_hist"], tensors["time_feat"])
    loss, parts = compute_dualcast_loss(model, prediction, tensors["target"], tensors["target_mask"], config["training_recipe"])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("DualCast full-effective gradients are absent or nonfinite")
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values() if value.is_floating_point()):
        raise FloatingPointError("DualCast optimizer produced nonfinite parameters/buffers")
    accumulator = MaskedMAEAccumulator()
    accumulator.update(prediction, tensors["target"], tensors["target_mask"])
    return parts | {"updated": True, "valid_count": accumulator.valid_count, "error_sum": accumulator._absolute_error_sum}


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any, data: Any,
                    config: Mapping[str, Any], progress: Mapping[str, Any], selection_metric=None) -> None:
    loaders = {}
    for name in ("train", "val"):
        state = getattr(data, name).state_dict()
        state["order"] = torch.from_numpy(state["order"].copy())
        loaders[name] = state
    value = {"schema_version": SCHEMA, "config": dict(config), "config_digest": canonical_digest(config),
             "model_factory": model.factory_kwargs(), "model_upstream": copy.deepcopy(model.upstream),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "rng": capture_rng_state(), "loaders": loaders, "data_identity": data.identity,
             "progress": copy.deepcopy(progress), "selection_metric": copy.deepcopy(selection_metric),
             "resume_boundary": "completed_full_effective_batch_or_epoch"}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": digest, "size_bytes": path.stat().st_size, "config_digest": value["config_digest"]})
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if marker["sha256"] != sha256_file(path) or marker["size_bytes"] != path.stat().st_size:
        raise ValueError("DualCast checkpoint bytes changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {"config", "model_factory", "model_upstream", "model", "optimizer", "scheduler", "rng", "loaders", "data_identity", "progress", "selection_metric"}
    if value.get("schema_version") != SCHEMA or not required.issubset(value):
        raise ValueError("DualCast checkpoint is not strictly complete")
    if value["config_digest"] != canonical_digest(value["config"]) or value["config_digest"] != marker["config_digest"]:
        raise ValueError("DualCast frozen recipe digest changed")
    return value


def train_run(*, model: Any, data: Any, device: torch.device, config: Mapping[str, Any], output: Path,
              checkpoint: Mapping[str, Any] | None = None, max_wallclock_seconds: float | None = None) -> dict[str, Any]:
    optimizer, scheduler = create_optimizer(model, config)
    progress = {"epoch": 0, "stage": "start", "updates": 0, "best_epoch": None, "best_val_mae": None,
                "history": [], "partial": {"error_sum": 0.0, "valid_count": 0, "auxiliary_omissions": {}}, "elapsed_seconds": 0.0}
    if checkpoint is not None:
        progress = restore_reference(checkpoint, model, optimizer, scheduler, data, config)
    batches = min(data.train.num_batch, config["max_train_batches"] or data.train.num_batch)
    stop, previous = {"requested": False}, {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.signal(signum, lambda *_: stop.__setitem__("requested", True))
        except ValueError:
            pass
    started, prior_elapsed = time.monotonic(), progress["elapsed_seconds"]

    def stopped():
        return stop["requested"] or (max_wallclock_seconds is not None and time.monotonic() - started >= max_wallclock_seconds)

    def persist():
        progress["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        save_checkpoint(output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)

    try:
        while progress["epoch"] < config["epochs"]:
            if progress["stage"] == "start":
                data.train.start_epoch()
                progress["partial"] = {"error_sum": 0.0, "valid_count": 0, "auxiliary_omissions": {}}
                progress["stage"] = "train"
            while progress["stage"] == "train" and data.train.current_ind < batches:
                if stopped():
                    persist()
                    result = {"status": "stopped_resumable", "completed_epochs": progress["epoch"], "next_batch": data.train.current_ind, "claim_scope": config["claim_scope"]}
                    atomic_json(output / "run_state.json", result)
                    return result
                tensors = batch_tensors(data.train.next_batch(), data, device)
                step = optimize_batch(model, optimizer, tensors, config)
                progress["partial"]["error_sum"] += step["error_sum"]
                progress["partial"]["valid_count"] += step["valid_count"]
                omissions = Counter(progress["partial"]["auxiliary_omissions"])
                omissions.update(f"{term}:{reason}" for term, reason in step["omitted_undefined_auxiliary"].items())
                progress["partial"]["auxiliary_omissions"] = dict(omissions)
                progress["updates"] += int(step["updated"])
                if step["updated"]:
                    with (output / "auxiliary_audit.jsonl").open("a", encoding="utf-8") as handle:
                        json.dump({"epoch": progress["epoch"] + 1, "batch": data.train.current_ind,
                                   **{key: value for key, value in step.items() if key not in ("error_sum",)}}, handle, sort_keys=True, allow_nan=False)
                        handle.write("\n")
            progress["stage"] = "validate"
            if stopped():
                persist()
                return {"status": "stopped_resumable", "completed_epochs": progress["epoch"], "next_stage": "validate", "claim_scope": config["claim_scope"]}
            validation = evaluate_reference(model, data, device, microbatch_size=config["batch_size"], max_batches=config["max_val_batches"])
            if not progress["partial"]["valid_count"]:
                raise ValueError("DualCast training epoch contains no observed targets")
            used_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            progress["epoch"] += 1
            progress["stage"] = "start"
            row = {"epoch": progress["epoch"], "train_mae": progress["partial"]["error_sum"] / progress["partial"]["valid_count"],
                   "validation": validation, "auxiliary_omissions": progress["partial"]["auxiliary_omissions"],
                   "learning_rate_used": used_lr, "learning_rate_next": optimizer.param_groups[0]["lr"], "updates": progress["updates"]}
            progress["history"].append(row)
            improved = progress["best_val_mae"] is None or validation["mae"] < progress["best_val_mae"]
            if improved:
                progress["best_val_mae"], progress["best_epoch"] = validation["mae"], progress["epoch"]
                save_checkpoint(output / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress, selection_metric=validation)
            persist()
            write_history(output, progress["history"])
            print(json.dumps(row, sort_keys=True), flush=True)
        result = {"status": "training_complete_pending_independent_source_validation", "claim_scope": config["claim_scope"],
                  "completed_epochs": progress["epoch"], "best_epoch": progress["best_epoch"], "best_val_mae": progress["best_val_mae"],
                  "config_digest": canonical_digest(config), "recipe": config["training_recipe"]["name"], "exact_native_claim": False, "test_evaluated": False}
        atomic_json(output / "summary.json", result)
        atomic_json(output / "run_state.json", result)
        return result
    except Exception as error:
        atomic_json(output / "failure.json", {"status": "failed_closed", "error_type": type(error).__name__, "message": str(error),
                                              "claim_scope": config["claim_scope"], "recipe": config["training_recipe"]["name"], "test_evaluated": False})
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def verify(*, checkpoint_path: Path, data: Any, device: torch.device, runtime: Mapping[str, Any], output: Path, checkout: Path | None = None) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if os.getpid() == config["training_process_id"]:
        raise ValueError("DualCast engineering verification must use an independent process")
    require_snapshot(config["source_snapshot"])
    if checkpoint["data_identity"] != data.identity or dict(runtime) != config["runtime"]:
        raise ValueError("DualCast verification data/runtime identity changed")
    expected = checkpoint["selection_metric"]
    if expected is None or expected["selection"] != evaluation_identity(data, config["max_val_batches"]):
        raise ValueError("DualCast verification needs the exact selected validation collection")
    kwargs = checkpoint["model_factory"] | {"device": str(device)}
    if checkout is not None:
        kwargs["checkout"] = checkout
    model = make_dualcast_model(**kwargs)
    observed, declared = copy.deepcopy(model.upstream), copy.deepcopy(checkpoint["model_upstream"])
    observed.pop("checkout", None)
    declared.pop("checkout", None)
    if observed != declared:
        raise ValueError("DualCast author AST/source/adjacency identity changed")
    model.load_state_dict(checkpoint["model"], strict=True)
    actual = evaluate_reference(model, data, device, microbatch_size=config["batch_size"], max_batches=config["max_val_batches"])
    if any(actual[key] != expected[key] for key in ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256")):
        raise ValueError("DualCast verification did not reproduce all recorded source observations")
    if not math.isclose(actual["mae"], expected["mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("DualCast independent source metric recomputation differs")
    if config["claim_scope"] != "smoke_only" and not actual["selection"]["all_validation_samples"]:
        raise ValueError("DualCast complete development verification requires full source validation")
    result = {"status": "verified_smoke_source_collection" if config["claim_scope"] == "smoke_only" else "verified_full_source_validation_compatibility_reference",
              "claim_scope": config["claim_scope"], "recipe": config["training_recipe"]["name"], "exact_native_claim": False,
              "checkpoint_sha256": sha256_file(checkpoint_path), "config_digest": checkpoint["config_digest"],
              "validation": actual, "training_process_id": config["training_process_id"], "verification_process_id": os.getpid(),
              "author_files_digest": model.upstream["author_files_digest"], "test_evaluated": False}
    atomic_json(output / "verification.json", result)
    return result


def calibrate(*, model: Any, data: Any, device: torch.device, config: Mapping[str, Any], output: Path, warmup: int, timed: int) -> dict[str, Any]:
    optimizer, scheduler = create_optimizer(model, config)
    phases = {}
    for phase, loader in (("train", data.train), ("val", data.val)):
        loader.start_epoch()
        model.train(phase == "train")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations, omissions = [], Counter()
        for number in range(warmup + timed):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch, _ = _next_full_reference_batch(loader)
            tensors = batch_tensors(batch, data, device)
            if phase == "train":
                result = optimize_batch(model, optimizer, tensors, config)
                omissions.update(f"{term}:{reason}" for term, reason in result["omitted_undefined_auxiliary"].items())
            else:
                with torch.no_grad():
                    _calibration_validation_batch(model, tensors, config["batch_size"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if number >= warmup:
                durations.append(elapsed)
        phases[phase] = {"warmup_batches": warmup, "timed_batches": timed, "seconds": durations,
                         "mean_seconds": float(np.mean(durations)), "p90_seconds": float(np.percentile(durations, 90)),
                         "auxiliary_omissions": dict(omissions),
                         "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
                         "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
    epochs = config["requested_epochs"]
    result = {"status": "calibrated_full_dualcast_objective_cuda" if device.type == "cuda" else "calibrated_cpu_engineering_only",
              "claim_scope": "calibration_only", "cuda_budget_eligible": device.type == "cuda", "phases": phases,
              "recipe": config["training_recipe"]["name"], "full_effective_batch": config["batch_size"],
              "estimated_train_val_seconds": epochs * (phases["train"]["mean_seconds"] * data.train.num_batch + phases["val"]["mean_seconds"] * data.val.num_batch),
              "p90_estimated_train_val_seconds": epochs * (phases["train"]["p90_seconds"] * data.train.num_batch + phases["val"]["p90_seconds"] * data.val.num_batch),
              "prediction_error_used_for_configuration_selection": False, "exact_native_claim": False,
              "config_digest": canonical_digest(config), "test_evaluated": False}
    atomic_json(output / "calibration.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("train", "verify", "calibrate"), default="train")
    result.add_argument("--recipe", choices=RECIPES, default=RECIPES[1])
    result.add_argument("--checkout", type=Path)
    result.add_argument("--region", choices=("Alameda", "ContraCosta", "Orange"), default="Alameda")
    result.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    result.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, choices=(30, 100), default=100)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--seed", type=int, choices=(42, 1, 2), default=42)
    result.add_argument("--embed-dim", type=int, default=64)
    result.add_argument("--heads", type=int, default=8)
    result.add_argument("--layers", type=int, default=1)
    result.add_argument("--threads", type=int, default=3)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--smoke-epochs", type=int, default=1)
    result.add_argument("--max-train-batches", type=int)
    result.add_argument("--max-val-batches", type=int)
    result.add_argument("--max-wallclock-seconds", type=float)
    result.add_argument("--warmup-batches", type=int, default=3)
    result.add_argument("--timed-batches", type=int, default=20)
    result.add_argument("--lazy", action="store_true")
    return result


def validate(args):
    if any(getattr(args, field) <= 0 for field in ("batch_size", "threads", "smoke_epochs", "embed_dim", "heads", "layers", "timed_batches")) or args.warmup_batches < 0 or args.embed_dim % args.heads:
        raise ValueError("DualCast batch/runtime/model/calibration dimensions are invalid")
    if args.mode == "verify" and args.checkpoint is None:
        raise ValueError("DualCast verify needs a selected checkpoint")
    if args.checkpoint is not None and args.mode != "verify":
        raise ValueError("checkpoint is reserved for independent verification")
    if args.resume is not None and (args.mode != "train" or args.resume.name != "last.pt"):
        raise ValueError("DualCast strict resume requires last.pt in train mode")
    if args.checkout is None and args.resume is None and args.checkpoint is None:
        raise ValueError("a new DualCast run needs an explicit locked author checkout")
    if args.device.startswith("cpu") and not args.smoke and args.resume is None and args.mode != "verify":
        raise ValueError("CPU is restricted to explicit DualCast engineering smoke")
    for field in ("max_train_batches", "max_val_batches"):
        if getattr(args, field) is not None and (not args.smoke or getattr(args, field) <= 0):
            raise ValueError("DualCast subset collection requires positive explicit smoke limits")
    if args.max_wallclock_seconds is not None and (not math.isfinite(args.max_wallclock_seconds) or args.max_wallclock_seconds <= 0):
        raise ValueError("DualCast wallclock limit must be positive and finite")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(arguments)
    validate(args)
    checkpoint = load_checkpoint(args.resume or args.checkpoint) if args.resume or args.checkpoint else None
    saved = checkpoint["config"] if checkpoint else None
    if saved:
        mutable = {"mode", "checkout", "data_dir", "graph_dir", "output_dir", "resume", "checkpoint", "device", "max_wallclock_seconds", "warmup_batches", "timed_batches"}
        for field, value in vars(args).items():
            option = "--" + field.replace("_", "-")
            expected = saved.get("requested_epochs") if field == "epochs" else saved.get(field)
            if field not in mutable and any(item == option or item.startswith(option + "=") for item in arguments) and value != expected:
                raise ValueError("DualCast cannot override frozen recipe identity: " + option)
    device = configure_runtime(args.device, saved["seed"] if saved else args.seed, saved["threads"] if saved else args.threads)
    runtime = runtime_identity(device)
    if saved and runtime != saved["runtime"]:
        raise ValueError("DualCast resume/verify runtime identity differs")
    if device.type == "cpu" and saved and saved["claim_scope"] != "smoke_only":
        raise ValueError("CPU cannot restore a formal DualCast training identity")
    recipe = saved["training_recipe"] if saved else dualcast_recipe(args.recipe, batch_size=args.batch_size, epochs=args.epochs)
    data = load_research_data(region_name=saved["region"] if saved else args.region, data_dir=args.data_dir, graph_dir=args.graph_dir,
                              batch_size=recipe["batch_size"], seed=saved["seed"] if saved else args.seed, preload=not (saved["lazy"] if saved else args.lazy))
    output = args.output_dir or ((args.resume or args.checkpoint).parent if checkpoint else None)
    if output is None:
        raise ValueError("DualCast needs an independent regional output directory")
    if args.resume and output.resolve() != args.resume.parent.resolve():
        raise ValueError("DualCast resume must retain the original run directory")
    if args.mode == "verify":
        output.mkdir(parents=True, exist_ok=True)
        result = verify(checkpoint_path=args.checkpoint, data=data, device=device, runtime=runtime, output=output, checkout=args.checkout)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    factory = checkpoint["model_factory"] if checkpoint else {"checkout": args.checkout, "adjacency": regional_adjacency(data.region),
               "flow_mean": data.statistics.flow_mean, "flow_std": data.statistics.flow_std,
               "embed_dim": args.embed_dim, "heads": args.heads, "layers": args.layers}
    if checkpoint and args.checkout is not None:
        factory = factory | {"checkout": args.checkout}
    model = make_dualcast_model(**(factory | {"device": str(device)}))
    if saved:
        config = saved
    else:
        config = {field: getattr(args, field) for field in ("recipe", "region", "seed", "threads", "lazy", "smoke", "smoke_epochs", "max_train_batches", "max_val_batches", "batch_size", "embed_dim", "heads", "layers")}
        config.update({"schema_version": SCHEMA, "epochs": args.smoke_epochs if args.smoke else args.epochs,
                       "requested_epochs": args.epochs, "training_recipe": recipe,
                       "claim_scope": "smoke_only" if args.smoke else "validation_development_compatibility_reference",
                       "training_process_id": os.getpid(), "runtime": runtime, "data_identity": data.identity,
                       "source_snapshot": source_snapshot(), "model_upstream": model.upstream,
                       "test_loader_constructed": False, "test_evaluated": False})
        if args.mode == "calibrate":
            config["claim_scope"] = "calibration_only"
        output.mkdir(parents=True, exist_ok=False)
        for filename, value in (("config.json", config), ("data_identity.json", data.identity), ("source_snapshot.json", config["source_snapshot"])):
            atomic_json(output / filename, value)
    result = calibrate(model=model, data=data, device=device, config=config, output=output, warmup=args.warmup_batches, timed=args.timed_batches) if args.mode == "calibrate" else train_run(model=model, data=data, device=device, config=config, output=output, checkpoint=checkpoint, max_wallclock_seconds=args.max_wallclock_seconds)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
