from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .execution import (
    capture_test_predictions,
    run_upstream_training,
    save_model_state_checkpoint,
)
from .result_artifacts import write_completed_task_artifacts
from .task_runner import (
    _TaskLogger,
    _build_factory,
    _prepare_result_target,
    _prepare_results_root,
    _resolve_device,
    _runtime_provenance,
    _training_code_identity,
    _load_verified_runtime_bundle,
    load_table4_task_spec,
)


_STGCN_ADAPTER = "stgcn-idempotent-self-loop-removal"
_MULTI_GPU_ADAPTER = "multi-gpu-full-batch"
_MULTI_GPU_METHODS = frozenset({"dgcrn", "d2stgnn", "igstgnn"})
_ADAPTERS = (_STGCN_ADAPTER, _MULTI_GPU_ADAPTER)


class RecoveryDataParallel(torch.nn.DataParallel):
    @staticmethod
    def _to_replica_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, torch.device):
            return device if value.type == "cuda" else value
        if isinstance(value, list):
            moved = [
                RecoveryDataParallel._to_replica_device(item, device)
                for item in value
            ]
            return (
                moved
                if any(new is not old for new, old in zip(moved, value, strict=True))
                else value
            )
        if isinstance(value, tuple):
            moved = tuple(
                RecoveryDataParallel._to_replica_device(item, device)
                for item in value
            )
            return (
                moved
                if any(new is not old for new, old in zip(moved, value, strict=True))
                else value
            )
        if isinstance(value, dict):
            moved = {
                key: RecoveryDataParallel._to_replica_device(item, device)
                for key, item in value.items()
            }
            if all(moved[key] is item for key, item in value.items()):
                return value
            copied = value.copy()
            copied.update(moved)
            return copied
        return value

    def replicate(
        self,
        module: torch.nn.Module,
        device_ids: list[int],
    ) -> list[torch.nn.Module]:
        replicas = super().replicate(module, device_ids)
        internal_names = {"_parameters", "_buffers", "_modules"}
        for replica, device_id in zip(replicas, device_ids, strict=True):
            device = torch.device("cuda", device_id)
            for child in replica.modules():
                for name, value in tuple(vars(child).items()):
                    if name in internal_names:
                        continue
                    moved = self._to_replica_device(value, device)
                    if moved is not value:
                        setattr(child, name, moved)
        return replicas

    @property
    def horizon(self) -> int:
        return self.module.horizon

    def param_num(self) -> int:
        return self.module.param_num()


def _warm_up_replica_linalg(device_ids: list[int]) -> None:
    for device_id in device_ids:
        device = torch.device("cuda", device_id)
        with torch.cuda.device(device):
            torch.linalg.qr(torch.eye(2, dtype=torch.float32, device=device))
            torch.cuda.synchronize(device)


def wrap_model_for_recovery(
    *,
    method: str,
    model: torch.nn.Module,
    engine: Any,
    device: torch.device,
    adapter: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if method not in _MULTI_GPU_METHODS or adapter != _MULTI_GPU_ADAPTER:
        raise ValueError("The requested multi-GPU adapter is not valid for this method")
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError("Multi-GPU recovery requires primary device cuda:0")
    if engine.model is not model:
        raise ValueError("Recovery engine and model identity do not match")
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) < 2:
        raise ValueError("Multi-GPU recovery requires at least two visible GPUs")
    replica_linalg_warmup = method == "dgcrn"
    if replica_linalg_warmup:
        _warm_up_replica_linalg(device_ids)
    wrapped = RecoveryDataParallel(
        model,
        device_ids=device_ids,
        output_device=0,
    )
    engine.model = wrapped
    return (
        wrapped,
        {
            "type": "torch_data_parallel",
            "name": "replica_device_full_batch_v1",
            "device_ids": device_ids,
            "full_batch_preserved": True,
            "optimizer_step_contract_preserved": True,
            "unregistered_tensor_policy": "replica_device_copy",
            "replica_linalg_warmup": replica_linalg_warmup,
            "upstream_source_modified": False,
        },
    )


def adapt_runtime_for_recovery(
    *,
    method: str,
    runtime: Any,
    adapter: str,
) -> tuple[Any, dict[str, Any]]:
    if method != "stgcn" or adapter != _STGCN_ADAPTER:
        raise ValueError("The requested recovery adapter is not valid for this method")
    adjacency = np.asarray(runtime.adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("STGCN recovery adjacency must be square")
    if not np.issubdtype(adjacency.dtype, np.number):
        raise TypeError("STGCN recovery adjacency must be numeric")
    if not np.isfinite(adjacency).all():
        raise ValueError("STGCN recovery adjacency must contain only finite values")
    diagonal = np.diag(adjacency)
    if not np.array_equal(diagonal, np.zeros_like(diagonal)):
        raise ValueError(
            "STGCN idempotent self-loop removal requires a zero diagonal"
        )
    adapted_adjacency = np.array(adjacency, copy=True)
    adapted_adjacency += np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    return (
        replace(runtime, adjacency=adapted_adjacency),
        {
            "type": "input_semantics_adapter",
            "name": "stgcn_idempotent_self_loop_removal_v1",
            "source_diagonal": "zero",
            "upstream_identity_subtraction_preserved": True,
            "effective_loop_free_graph_preserved": True,
        },
    )


def _controller_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_recovery_task(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    task_id: str,
    splits_root: Path | str,
    results_root: Path | str,
    device: str,
    adapter: str,
) -> Path:
    spec = load_table4_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        task_id=task_id,
    )
    method = spec.task["method"]
    if adapter == _STGCN_ADAPTER and method != "stgcn":
        raise ValueError("The STGCN recovery adapter requires an STGCN task")
    if adapter == _MULTI_GPU_ADAPTER and method not in _MULTI_GPU_METHODS:
        raise ValueError(
            "The multi-GPU recovery adapter requires DGCRN, D2STGNN, or IGSTGNN"
        )
    root = _prepare_results_root(results_root)
    target = _prepare_result_target(results_root=root, task=spec.task)
    device_value = _resolve_device(device)
    runtime = _load_verified_runtime_bundle(
        splits_root=splits_root,
        region=spec.task["region"],
        expected_manifest_sha256=(spec.task["split_bundle_manifest_sha256"]),
        profile_effective=spec.profile["effective"],
        task_seed=spec.task["seed"],
        include_incident=method == "igstgnn",
    )
    initial_code_sha256, initial_code_files = _training_code_identity(method)
    initial_controller_sha256 = _controller_sha256()
    input_adapter: dict[str, Any] | None = None
    if adapter == _STGCN_ADAPTER:
        runtime, input_adapter = adapt_runtime_for_recovery(
            method=method,
            runtime=runtime,
            adapter=adapter,
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{task_id}.recovery.",
        dir=root,
    ) as scratch_name:
        scratch = Path(scratch_name)
        train_log = scratch / "train.log"
        checkpoint = scratch / "checkpoint.pt"
        with train_log.open("x", encoding="utf-8") as log_handle:
            logger = _TaskLogger(log_handle)
            logger.info(
                "Starting recovery for %s with manifest %s and adapter %s",
                task_id,
                manifest_sha256,
                adapter,
            )
            with contextlib.redirect_stdout(log_handle):
                with contextlib.redirect_stderr(log_handle):
                    factory = _build_factory(
                        spec=spec,
                        runtime=runtime,
                        device=device_value,
                        logger=logger,
                        output_dir=scratch,
                    )
                    execution_adapter = input_adapter
                    if adapter == _MULTI_GPU_ADAPTER:
                        _, execution_adapter = wrap_model_for_recovery(
                            method=method,
                            model=factory.model,
                            engine=factory.engine,
                            device=device_value,
                            adapter=adapter,
                        )
                    if execution_adapter is None:
                        raise RuntimeError("Recovery adapter provenance is missing")
                    execution_adapter = {
                        **execution_adapter,
                        "controller_sha256": initial_controller_sha256,
                    }
                    factory.provenance["execution_adapter"] = execution_adapter
                    run_upstream_training(
                        method=method,
                        engine=factory.engine,
                    )
                    predictions, labels = capture_test_predictions(
                        method=method,
                        engine=factory.engine,
                    )
                    save_model_state_checkpoint(
                        model=factory.engine.model,
                        target=checkpoint,
                    )
            final_code_sha256, final_code_files = _training_code_identity(method)
            final_controller_sha256 = _controller_sha256()
            if (
                final_code_sha256 != initial_code_sha256
                or final_code_files != initial_code_files
                or final_controller_sha256 != initial_controller_sha256
            ):
                raise RuntimeError("Recovery training code changed during the task")
            logger.info("Captured prediction shape: %s", predictions.shape)
            log_handle.flush()
            os.fsync(log_handle.fileno())
        provenance = _runtime_provenance(
            spec=spec,
            runtime=runtime,
            device=device_value,
            factory_provenance=factory.provenance,
            training_code_sha256=final_code_sha256,
            training_code_files=final_code_files,
        )
        provenance["recovery_execution"] = {
            "adapter": adapter,
            "controller_sha256": final_controller_sha256,
            "base_training_code_sha256": final_code_sha256,
        }
        return write_completed_task_artifacts(
            target_dir=target,
            task=spec.task,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance=provenance,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one locked Table 4 task through an explicit, provenance-bound "
            "hardware or input-semantics recovery adapter."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--splits-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--adapter", required=True, choices=_ADAPTERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    target = run_recovery_task(
        manifest_path=arguments.manifest,
        manifest_sha256=arguments.manifest_sha256,
        task_id=arguments.task_id,
        splits_root=arguments.splits_root,
        results_root=arguments.results_root,
        device=arguments.device,
        adapter=arguments.adapter,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
