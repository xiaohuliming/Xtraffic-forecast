from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from collections.abc import Sequence

import numpy as np

from .bist_factory import build_bist_factory
from .execution import (
    capture_test_predictions,
    run_upstream_training,
    save_model_state_checkpoint,
)
from .manifest import load_table4_manifest
from .native100_compare_manifest import load_native100_compare_manifest
from .ndr_extension_manifest import load_ndr_extension_profile
from .ndr_extension_runner import (
    NDRTable4TaskSpec,
    probe_ndr_task_spec,
    run_ndr_task_spec,
)
from . import ndr_extension_runner as ndr_runner_module
from .profiles import load_profile_set
from .result_artifacts import write_completed_task_artifacts
from .task_runner import (
    PROFILE_PATH,
    Table4TaskSpec,
    _TaskLogger,
    _load_verified_runtime_bundle,
    _prepare_result_target,
    _prepare_results_root,
    _resolve_device,
    _runtime_provenance,
    _training_code_identity as _base_training_code_identity,
    _tuple_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
NDR_BASE_PROFILE_PATH = CONFIG_ROOT / "ndr_extension_profile_v1.json"
NATIVE100_CLAIM_SCOPE = "same_table_native_recipe_100_epoch_supplement"
_EXTRA_TRAINING_CODE_PATHS = (
    "scripts/reproduction/igstgnn_table4/native100_compare_runner.py",
    "scripts/reproduction/igstgnn_table4/native100_compare_manifest.py",
)


@dataclass(frozen=True)
class Native100TaskSpec:
    manifest: Mapping[str, Any]
    task: Mapping[str, Any]
    method_profile: Mapping[str, Any]
    base_profile: Mapping[str, Any]
    base_effective: Mapping[str, Any]
    effective: Mapping[str, Any]
    runtime_profile: Mapping[str, Any]


def _base_profile(method: str, *, expected_sha256: str) -> dict[str, Any]:
    if method == "bist":
        profile_set = load_profile_set(PROFILE_PATH)
        if profile_set["config_hashes"]["bist"] != expected_sha256:
            raise ValueError("Native 100 BiST base profile SHA-256 mismatch")
        return copy.deepcopy(profile_set["profiles"]["bist"])
    if method == "ndr_net":
        return copy.deepcopy(
            load_ndr_extension_profile(
                NDR_BASE_PROFILE_PATH,
                expected_sha256=expected_sha256,
            )
        )
    raise ValueError(f"Unsupported native 100 method: {method}")


def load_native100_task_spec(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    task_id: str,
) -> Native100TaskSpec:
    manifest = load_native100_compare_manifest(
        manifest_path,
        expected_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        expected_base_manifest_sha256=base_manifest_sha256,
    )
    matches = [task for task in manifest["tasks"] if task["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"Task ID is not in the native 100 matrix: {task_id}")
    task = matches[0]
    method_profile = manifest["profile"]["methods"][task["method"]]
    base_profile = _base_profile(
        task["method"],
        expected_sha256=method_profile["base_config_sha256"],
    )
    base_effective = copy.deepcopy(base_profile["effective"])
    effective = copy.deepcopy(base_effective)
    effective["max_epochs"] = 100
    runtime_profile = copy.deepcopy(base_profile)
    runtime_profile["effective"] = copy.deepcopy(effective)
    runtime_profile["profile_id"] = manifest["profile"]["profile_id"]
    runtime_profile["source_id"] = task["source_id"]
    return Native100TaskSpec(
        manifest=manifest,
        task=task,
        method_profile=method_profile,
        base_profile=base_profile,
        base_effective=base_effective,
        effective=effective,
        runtime_profile=runtime_profile,
    )


def _native_training_code_identity(method: str) -> tuple[str, dict[str, str]]:
    if method == "bist":
        _, files = _base_training_code_identity("bist")
    elif method == "ndr_net":
        _, files = ndr_runner_module._training_code_identity()
    else:
        raise ValueError(f"Unsupported native 100 method: {method}")
    merged = dict(files)
    for relative in _EXTRA_TRAINING_CODE_PATHS:
        source = PROJECT_ROOT / relative
        merged[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.dumps(merged, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest(), merged


def _bound_bist_spec(
    spec: Native100TaskSpec,
    *,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
) -> Table4TaskSpec:
    base_manifest = load_table4_manifest(
        base_manifest_path,
        expected_sha256=base_manifest_sha256,
    )
    bound_manifest = {
        **base_manifest,
        "manifest_sha256": spec.manifest["manifest_sha256"],
    }
    return Table4TaskSpec(
        manifest=bound_manifest,
        task=spec.task,
        profile=spec.runtime_profile,
    )


def _run_bist_native100_task(
    *,
    spec: Native100TaskSpec,
    base_manifest_path: Path | str,
    base_manifest_sha256: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
) -> Path:
    table_spec = _bound_bist_spec(
        spec,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
    )
    root = _prepare_results_root(results_root)
    target = _prepare_result_target(results_root=root, task=spec.task)
    device_value = _resolve_device(device)
    runtime = _load_verified_runtime_bundle(
        splits_root=splits_root,
        region=spec.task["region"],
        expected_manifest_sha256=spec.task["split_bundle_manifest_sha256"],
        profile_effective=spec.effective,
        task_seed=spec.task["seed"],
        include_incident=False,
    )
    initial_code_sha256, initial_code_files = _native_training_code_identity(
        "bist"
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{spec.task['task_id']}.run.",
        dir=root,
    ) as scratch_name:
        scratch = Path(scratch_name)
        train_log = scratch / "train.log"
        checkpoint = scratch / "checkpoint.pt"
        with train_log.open("x", encoding="utf-8") as log_handle:
            logger = _TaskLogger(log_handle)
            logger.info(
                "Starting %s with manifest %s",
                spec.task["task_id"],
                spec.manifest["manifest_sha256"],
            )
            logger.info(
                "Native BiST recipe max_epochs override 500 to 100, patience 20"
            )
            with contextlib.redirect_stdout(log_handle):
                with contextlib.redirect_stderr(log_handle):
                    factory = build_bist_factory(
                        profile_effective=spec.base_effective,
                        manifest_frozen_seed=spec.task["seed"],
                        node_num=runtime.bundle.node_count,
                        device=device_value,
                        dataloader=_tuple_dataloader(runtime.bundle),
                        scaler=runtime.scaler,
                        logger=logger,
                        output_dir=scratch,
                        max_epochs_override=100,
                    )
                    run_upstream_training(method="bist", engine=factory.engine)
                    predictions, labels = capture_test_predictions(
                        method="bist",
                        engine=factory.engine,
                    )
                    save_model_state_checkpoint(
                        model=factory.model,
                        target=checkpoint,
                    )
            final_code_sha256, final_code_files = _native_training_code_identity(
                "bist"
            )
            if (
                final_code_sha256 != initial_code_sha256
                or final_code_files != initial_code_files
            ):
                raise RuntimeError("Native 100 training code changed during BiST")
            logger.info("Captured prediction shape: %s", predictions.shape)
            log_handle.flush()
            os.fsync(log_handle.fileno())
        provenance = _runtime_provenance(
            spec=table_spec,
            runtime=runtime,
            device=device_value,
            factory_provenance=factory.provenance,
            training_code_sha256=final_code_sha256,
            training_code_files=final_code_files,
        )
        provenance.update(
            {
                "matrix_id": spec.manifest["matrix_id"],
                "base_matrix_id": spec.manifest["base_matrix_id"],
                "base_manifest_sha256": spec.manifest["base_manifest_sha256"],
                "claim_scope": NATIVE100_CLAIM_SCOPE,
                "native_recipe_override": dict(spec.method_profile["override"]),
            }
        )
        return write_completed_task_artifacts(
            target_dir=target,
            task=spec.task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance=provenance,
            claim_scope=NATIVE100_CLAIM_SCOPE,
        )


def _ndr_spec(spec: Native100TaskSpec) -> NDRTable4TaskSpec:
    return NDRTable4TaskSpec(
        manifest=spec.manifest,
        task=spec.task,
        profile=spec.runtime_profile,
    )


def run_native100_task(
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
    spec = load_native100_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        task_id=task_id,
    )
    if spec.task["method"] == "bist":
        return _run_bist_native100_task(
            spec=spec,
            base_manifest_path=base_manifest_path,
            base_manifest_sha256=base_manifest_sha256,
            splits_root=splits_root,
            results_root=results_root,
            device=device,
        )
    if spec.task["method"] == "ndr_net":
        return run_ndr_task_spec(
            spec=_ndr_spec(spec),
            splits_root=splits_root,
            results_root=results_root,
            device=device,
            claim_scope=NATIVE100_CLAIM_SCOPE,
            training_code_identity_fn=lambda: _native_training_code_identity(
                "ndr_net"
            ),
        )
    raise ValueError(f"Unsupported native 100 method: {spec.task['method']}")


def run_bist_probe_step(engine: Any) -> dict[str, Any]:
    import torch

    loader = engine._dataloader["train_loader"]
    loader.shuffle()
    model_input, label = next(loader.get_iterator())
    engine._optimizer.zero_grad()
    model_input, label = engine._to_device(
        engine._to_tensor(
            [model_input, label[:, : engine._args.horizon]]
        )
    )
    prediction = engine.model(model_input, label)
    prediction, label = engine._inverse_transform([prediction, label])
    mask_value = torch.tensor(0, device=label.device)
    if label.min() < 1:
        mask_value = label.min()
    loss = engine._loss_fn(prediction, label, mask_value)
    loss.backward()
    if engine._clip_grad_value != 0:
        torch.nn.utils.clip_grad_norm_(
            engine.model.parameters(),
            engine._clip_grad_value,
        )
    engine._optimizer.step()
    engine._iter_cnt += 1
    return {
        "loss": float(loss.detach().cpu()),
        "prediction_shape": tuple(prediction.shape),
    }


def probe_native100_task(
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

    spec = load_native100_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        base_manifest_path=base_manifest_path,
        base_manifest_sha256=base_manifest_sha256,
        task_id=task_id,
    )
    if spec.task["method"] == "ndr_net":
        probe_ndr_task_spec(
            spec=_ndr_spec(spec),
            splits_root=splits_root,
            device=device,
        )
        return
    device_value = _resolve_device(device)
    runtime = _load_verified_runtime_bundle(
        splits_root=splits_root,
        region=spec.task["region"],
        expected_manifest_sha256=spec.task["split_bundle_manifest_sha256"],
        profile_effective=spec.effective,
        task_seed=spec.task["seed"],
        include_incident=False,
    )
    with tempfile.TemporaryDirectory(prefix="native100_bist_probe.") as scratch:
        logger = _TaskLogger(open(os.devnull, "w", encoding="utf-8"))
        try:
            factory = build_bist_factory(
                profile_effective=spec.base_effective,
                manifest_frozen_seed=spec.task["seed"],
                node_num=runtime.bundle.node_count,
                device=device_value,
                dataloader=_tuple_dataloader(runtime.bundle),
                scaler=runtime.scaler,
                logger=logger,
                output_dir=scratch,
                max_epochs_override=100,
            )
            probe = run_bist_probe_step(factory.engine)
        finally:
            logger._handle.close()
    values = np.asarray([probe["loss"]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("Native 100 BiST probe produced non-finite values")
    peak_bytes = None
    if device_value.type == "cuda":
        torch.cuda.synchronize(device_value)
        peak_bytes = int(torch.cuda.max_memory_allocated(device_value))
    print(
        "NATIVE100_BIST_SMOKE_OK "
        f"task={task_id} shape={probe['prediction_shape']} "
        f"loss={probe['loss']:.8f} "
        f"peak_allocated_bytes={peak_bytes}",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a native-recipe BiST or NDR-Net 100 epoch Table 4 task."
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
        probe_native100_task(**common)
        return 0
    if args.results_root is None:
        raise ValueError("--results-root is required unless --smoke is used")
    target = run_native100_task(
        **common,
        results_root=args.results_root,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
