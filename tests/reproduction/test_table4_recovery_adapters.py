from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from scripts.reproduction.igstgnn_table4.recovery_task_runner import (
    adapt_runtime_for_recovery,
    install_d2stgnn_microbatch_training,
    main,
    primary_light_chunk_sizes,
    wrap_model_for_recovery,
)


@dataclass(frozen=True)
class _Runtime:
    adjacency: np.ndarray


def test_stgcn_recovery_preserves_an_already_loop_free_graph() -> None:
    adjacency = np.asarray(
        [
            [0.0, 0.75, 0.0],
            [0.25, 0.0, 0.5],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )

    adapted, provenance = adapt_runtime_for_recovery(
        method="stgcn",
        runtime=_Runtime(adjacency=adjacency),
        adapter="stgcn-idempotent-self-loop-removal",
    )

    np.testing.assert_array_equal(
        adapted.adjacency - np.eye(3, dtype=np.float64),
        adjacency,
    )
    np.testing.assert_array_equal(adjacency.diagonal(), np.zeros(3))
    assert provenance == {
        "type": "input_semantics_adapter",
        "name": "stgcn_idempotent_self_loop_removal_v1",
        "source_diagonal": "zero",
        "upstream_identity_subtraction_preserved": True,
        "effective_loop_free_graph_preserved": True,
    }


def test_multi_gpu_recovery_preserves_the_declared_full_batch(monkeypatch) -> None:
    class _Model(torch.nn.Module):
        horizon = 12

        def param_num(self) -> int:
            return 456

    class _Engine:
        def __init__(self, model: torch.nn.Module) -> None:
            self.model = model

    class _FakeDataParallel:
        def __init__(self, module, *, device_ids, output_device):
            self.module = module
            self.device_ids = device_ids
            self.output_device = output_device

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        "scripts.reproduction.igstgnn_table4.recovery_task_runner."
        "RecoveryDataParallel",
        _FakeDataParallel,
    )
    model = _Model()
    engine = _Engine(model)

    wrapped, provenance = wrap_model_for_recovery(
        method="d2stgnn",
        model=model,
        engine=engine,
        device=torch.device("cuda:0"),
        adapter="multi-gpu-full-batch",
    )

    assert engine.model is wrapped
    assert wrapped.module is model
    assert wrapped.device_ids == [0, 1, 2]
    assert wrapped.output_device == 0
    assert provenance == {
        "type": "torch_data_parallel",
        "name": "replica_device_full_batch_v1",
        "device_ids": [0, 1, 2],
        "full_batch_preserved": True,
        "optimizer_step_contract_preserved": True,
        "primary_chunk_policy": "smallest_remainder_chunk",
        "unregistered_tensor_policy": "replica_device_copy",
        "replica_linalg_warmup": False,
        "upstream_source_modified": False,
    }


def test_primary_gpu_receives_the_smallest_remainder_chunk() -> None:
    assert primary_light_chunk_sizes(batch_size=64, device_count=5) == [
        12,
        13,
        13,
        13,
        13,
    ]


def test_d2stgnn_microbatches_match_one_full_masked_optimizer_step() -> None:
    class _Loader:
        num_batch = 1

        def __init__(self, batch) -> None:
            self.batch = batch

        def shuffle(self) -> None:
            pass

        def get_iterator(self):
            return iter([self.batch])

    class _Model(torch.nn.Module):
        horizon = 1

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.75))

        def forward(self, value, label=None):
            return value * self.weight

        def param_num(self) -> int:
            return 1

    class _Engine:
        def __init__(self, model, batch) -> None:
            self.model = model
            self._dataloader = {"train_loader": _Loader(batch)}
            self._optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            self._clip_grad_value = 0
            self._iter_cnt = 0
            self._warm_step = 100
            self._cl_step = 10
            self._cl_len = 0
            self._horizon = 1
            self._device = torch.device("cpu")

        def _to_tensor(self, values):
            return [
                torch.as_tensor(value, dtype=torch.float32)
                for value in values
            ]

        def _to_device(self, values):
            return [value.to(self._device) for value in values]

        def _inverse_transform(self, values):
            return values

        @staticmethod
        def _loss_fn(prediction, label, mask_value):
            mask = (label != mask_value).float()
            mask = mask / mask.mean()
            loss = torch.abs(prediction - label) * mask
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            return loss.mean()

    x_data = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(
        4, 1, 1, 1
    )
    y_data = np.asarray([0.0, 3.0, 2.0, 8.0], dtype=np.float32).reshape(
        4, 1, 1, 1
    )
    batch = (x_data, y_data)
    full_model = _Model()
    micro_model = _Model()
    micro_model.load_state_dict(full_model.state_dict())

    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.1)
    full_optimizer.zero_grad()
    full_prediction = full_model(torch.as_tensor(x_data))
    full_label = torch.as_tensor(y_data)
    full_loss = _Engine._loss_fn(
        full_prediction,
        full_label,
        torch.tensor(0.0),
    )
    full_loss.backward()
    full_optimizer.step()

    micro_engine = _Engine(micro_model, batch)
    provenance = install_d2stgnn_microbatch_training(
        engine=micro_engine,
        microbatches=2,
    )
    micro_engine.train_batch()

    torch.testing.assert_close(micro_model.weight, full_model.weight)
    assert micro_engine._iter_cnt == 1
    assert provenance == {
        "microbatch_count": 2,
        "effective_batch_preserved": True,
        "masked_loss_global_weighting": True,
        "optimizer_steps_per_batch": 1,
        "gradient_clips_per_batch": 1,
        "curriculum_updates_per_batch": 1,
    }
    assert primary_light_chunk_sizes(batch_size=48, device_count=4) == [
        12,
        12,
        12,
        12,
    ]


def test_multi_gpu_recovery_rejects_a_non_primary_device() -> None:
    model = torch.nn.Linear(1, 1)

    with pytest.raises(ValueError, match="primary device"):
        wrap_model_for_recovery(
            method="igstgnn",
            model=model,
            engine=type("_Engine", (), {"model": model})(),
            device=torch.device("cuda:1"),
            adapter="multi-gpu-full-batch",
        )


def test_recovery_cli_forwards_the_selected_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    observed = {}

    def _run_recovery_task(**kwargs):
        observed.update(kwargs)
        return tmp_path / "tasks" / "005_alameda_stgcn"

    monkeypatch.setattr(
        "scripts.reproduction.igstgnn_table4.recovery_task_runner."
        "run_recovery_task",
        _run_recovery_task,
    )

    exit_code = main(
        [
            "--manifest",
            str(tmp_path / "matrix.json"),
            "--manifest-sha256",
            "a" * 64,
            "--task-id",
            "005_alameda_stgcn",
            "--splits-root",
            str(tmp_path / "splits"),
            "--results-root",
            str(tmp_path / "results"),
            "--device",
            "cuda:0",
            "--adapter",
            "stgcn-idempotent-self-loop-removal",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "manifest_path": tmp_path / "matrix.json",
        "manifest_sha256": "a" * 64,
        "task_id": "005_alameda_stgcn",
        "splits_root": tmp_path / "splits",
        "results_root": tmp_path / "results",
        "device": "cuda:0",
        "adapter": "stgcn-idempotent-self-loop-removal",
        "microbatches": 1,
    }
