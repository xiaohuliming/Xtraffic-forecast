from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from collections.abc import Sequence
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch.nn.parallel._functions import Scatter

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


def primary_light_chunk_sizes(
    *,
    batch_size: int,
    device_count: int,
) -> list[int]:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(device_count, bool)
        or not isinstance(device_count, int)
        or device_count <= 0
    ):
        raise ValueError("device_count must be a positive integer")
    active_devices = min(batch_size, device_count)
    quotient, remainder = divmod(batch_size, active_devices)
    return [
        quotient + (1 if index > 0 and index <= remainder else 0)
        for index in range(active_devices)
    ]


def _is_namedtuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and hasattr(value, "_asdict")
        and hasattr(value, "_fields")
    )


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

    @staticmethod
    def _batch_size(inputs: tuple[Any, ...], dim: int) -> int:
        pending = list(inputs)
        while pending:
            value = pending.pop(0)
            if torch.is_tensor(value):
                if value.ndim <= dim:
                    continue
                return int(value.shape[dim])
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        raise ValueError("Multi-GPU recovery inputs do not contain a batch tensor")

    @staticmethod
    def _scatter_value(
        value: Any,
        *,
        target_gpus: list[int],
        chunk_sizes: list[int],
        dim: int,
    ) -> list[Any]:
        def scatter_map(item: Any) -> list[Any]:
            if torch.is_tensor(item):
                return list(Scatter.apply(target_gpus, chunk_sizes, dim, item))
            if _is_namedtuple(item):
                return [
                    type(item)(*values)
                    for values in zip(
                        *map(scatter_map, item),
                        strict=False,
                    )
                ]
            if isinstance(item, tuple) and item:
                return list(zip(*map(scatter_map, item), strict=False))
            if isinstance(item, list) and item:
                return [
                    list(values)
                    for values in zip(
                        *map(scatter_map, item),
                        strict=False,
                    )
                ]
            if isinstance(item, dict) and item:
                return [
                    type(item)(values)
                    for values in zip(
                        *map(scatter_map, item.items()),
                        strict=False,
                    )
                ]
            return [item for _ in target_gpus]

        try:
            return scatter_map(value)
        finally:
            scatter_map = None

    def scatter(
        self,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        device_ids: list[int],
    ) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
        batch_size = self._batch_size(inputs, self.dim)
        chunk_sizes = primary_light_chunk_sizes(
            batch_size=batch_size,
            device_count=len(device_ids),
        )
        active_devices = device_ids[: len(chunk_sizes)]
        scattered_inputs = (
            self._scatter_value(
                inputs,
                target_gpus=active_devices,
                chunk_sizes=chunk_sizes,
                dim=self.dim,
            )
            if inputs
            else []
        )
        scattered_kwargs = (
            self._scatter_value(
                kwargs,
                target_gpus=active_devices,
                chunk_sizes=chunk_sizes,
                dim=self.dim,
            )
            if kwargs
            else []
        )
        if len(scattered_inputs) < len(scattered_kwargs):
            scattered_inputs.extend(
                () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
            )
        elif len(scattered_kwargs) < len(scattered_inputs):
            scattered_kwargs.extend(
                {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
            )
        return tuple(scattered_inputs), tuple(scattered_kwargs)

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


def _d2stgnn_microbatch_train_batch(engine: Any) -> tuple[float, float, float]:
    engine.model.train()
    train_loss: list[float] = []
    train_mape: list[float] = []
    train_rmse: list[float] = []
    train_loader = engine._dataloader["train_loader"]
    train_loader.shuffle()
    for x_data, label_data in train_loader.get_iterator():
        engine._optimizer.zero_grad()
        full_label = engine._to_device(engine._to_tensor([label_data]))[0]
        full_label = engine._inverse_transform(full_label)
        mask_value = torch.tensor(0.0, device=full_label.device)
        if full_label.min() < 1:
            mask_value = full_label.min()
        if engine._iter_cnt == 0:
            print("check mask value", mask_value)

        engine._iter_cnt += 1
        if engine._iter_cnt < engine._warm_step:
            engine._cl_len = engine._horizon
        elif engine._iter_cnt == engine._warm_step:
            engine._cl_len = 1
        elif (
            (engine._iter_cnt - engine._warm_step) % engine._cl_step == 0
            and engine._cl_len < engine._horizon
        ):
            engine._cl_len += 1

        full_label = full_label[:, : engine._cl_len, :, :]
        full_mask = full_label != mask_value
        total_valid = int(full_mask.sum().item())
        if total_valid <= 0:
            raise ValueError("D2STGNN microbatch has no valid target observations")
        batch_size = int(len(x_data))
        microbatch_count = min(engine._recovery_microbatch_count, batch_size)
        index_chunks = np.array_split(np.arange(batch_size), microbatch_count)
        weighted_loss = 0.0
        mape_sum = 0.0
        squared_error_sum = 0.0
        for indices in index_chunks:
            micro_x, micro_label = engine._to_device(
                engine._to_tensor([x_data[indices], label_data[indices]])
            )
            prediction = engine.model(micro_x, micro_label)
            prediction, micro_label = engine._inverse_transform(
                [prediction, micro_label]
            )
            prediction = prediction[:, : engine._cl_len, :, :]
            micro_label = micro_label[:, : engine._cl_len, :, :]
            micro_mask = micro_label != mask_value
            micro_valid = int(micro_mask.sum().item())
            micro_loss = engine._loss_fn(
                prediction,
                micro_label,
                mask_value,
            )
            loss_weight = micro_valid / total_valid
            (micro_loss * loss_weight).backward()
            weighted_loss += micro_loss.detach().item() * loss_weight

            absolute_error = torch.abs(prediction.detach() - micro_label)
            safe_mape = torch.where(
                micro_mask,
                absolute_error / micro_label,
                torch.zeros_like(absolute_error),
            )
            safe_mape = torch.where(
                torch.isnan(safe_mape),
                torch.zeros_like(safe_mape),
                safe_mape,
            )
            mape_sum += safe_mape.sum().item()
            squared_error_sum += torch.where(
                micro_mask,
                absolute_error.square(),
                torch.zeros_like(absolute_error),
            ).sum().item()
        if engine._clip_grad_value != 0:
            torch.nn.utils.clip_grad_norm_(
                engine.model.parameters(),
                engine._clip_grad_value,
            )
        engine._optimizer.step()
        train_loss.append(weighted_loss)
        train_mape.append(mape_sum / total_valid)
        train_rmse.append(float(np.sqrt(squared_error_sum / total_valid)))
    return (
        float(np.mean(train_loss)),
        float(np.mean(train_mape)),
        float(np.mean(train_rmse)),
    )


def install_d2stgnn_microbatch_training(
    *,
    engine: Any,
    microbatches: int,
) -> dict[str, Any]:
    if (
        isinstance(microbatches, bool)
        or not isinstance(microbatches, int)
        or microbatches < 2
    ):
        raise ValueError("D2STGNN recovery requires at least two microbatches")
    if hasattr(engine, "_recovery_microbatch_count"):
        raise ValueError("D2STGNN microbatch recovery is already installed")
    engine._recovery_microbatch_count = microbatches
    engine.train_batch = MethodType(_d2stgnn_microbatch_train_batch, engine)
    return {
        "microbatch_count": microbatches,
        "effective_batch_preserved": True,
        "masked_loss_global_weighting": True,
        "optimizer_steps_per_batch": 1,
        "gradient_clips_per_batch": 1,
        "curriculum_updates_per_batch": 1,
    }


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
            "primary_chunk_policy": "smallest_remainder_chunk",
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
    microbatches: int = 1,
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
    if (
        isinstance(microbatches, bool)
        or not isinstance(microbatches, int)
        or microbatches <= 0
    ):
        raise ValueError("microbatches must be a positive integer")
    if microbatches > 1 and (
        adapter != _MULTI_GPU_ADAPTER or method != "d2stgnn"
    ):
        raise ValueError(
            "Microbatch recovery is valid only for multi-GPU D2STGNN"
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
                    if microbatches > 1:
                        microbatch_adapter = install_d2stgnn_microbatch_training(
                            engine=factory.engine,
                            microbatches=microbatches,
                        )
                        execution_adapter = {
                            **execution_adapter,
                            "gradient_accumulation": microbatch_adapter,
                        }
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
    parser.add_argument("--microbatches", type=int, default=1)
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
        microbatches=arguments.microbatches,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
