from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from scripts.reproduction.igstgnn_table4.recovery_task_runner import (
    adapt_runtime_for_recovery,
    main,
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
        "unregistered_tensor_policy": "replica_device_copy",
        "upstream_source_modified": False,
    }


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
    }
