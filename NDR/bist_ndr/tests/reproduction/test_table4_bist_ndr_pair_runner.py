from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.bist_ndr_pair_manifest import (
    load_bist_ndr_pair_profile,
)
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_model import (
    build_bist_ndr_pair_model,
)
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_runner import (
    backward_bist_ndr_pair_batch,
)
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_matrix_runner import (
    build_bist_ndr_pair_task_command,
    build_bist_ndr_pair_task_environment,
)
from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    NDRTrainingStatistics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "bist_ndr_pair_profile_v1.json"
)


def _effective():
    digest = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    return load_bist_ndr_pair_profile(
        PROFILE_PATH,
        expected_sha256=digest,
    )["effective"]


def _statistics(node_count: int) -> NDRTrainingStatistics:
    return NDRTrainingStatistics(
        baseline=np.zeros((7, 288, node_count), dtype=np.float32),
        flow_mean=20.0,
        flow_std=5.0,
        residual_std=3.0,
        residual_std_by_node=np.ones(node_count, dtype=np.float32),
        valid_training_points=100,
        source_normalization_mean=10.0,
        source_normalization_std=2.0,
    )


def _tensors():
    generator = torch.Generator().manual_seed(5)
    flow_hist = torch.rand((4, 3, 12), generator=generator) * 10 + 15
    x_baseline = torch.full_like(flow_hist, 18.0)
    y_baseline = torch.full_like(flow_hist, 21.0)
    target = torch.rand((4, 3, 12), generator=generator) * 10 + 15
    time_feat = torch.zeros((4, 12, 2))
    time_feat[..., 0] = 10 / 96
    time_feat[..., 1] = 2 / 7
    return {
        "flow_hist": flow_hist,
        "x_baseline": x_baseline,
        "y_baseline": y_baseline,
        "time_feat": time_feat,
        "target": target,
        "target_mask": torch.ones_like(target, dtype=torch.bool),
    }


def test_paired_microbatching_matches_the_effective_full_batch_for_both_methods():
    effective = _effective()
    statistics = _statistics(node_count=3)
    tensors = _tensors()
    loss_batch = {
        "y_true": tensors["target"].unsqueeze(-1),
        "y_mask": tensors["target_mask"].unsqueeze(-1),
    }
    loss_args = SimpleNamespace(lambda_gate=1.0, lambda_spec=0.3)

    for method in ("bist_aligned", "bist_ndr"):
        torch.manual_seed(101)
        full = build_bist_ndr_pair_model(
            method=method,
            node_count=3,
            effective_profile=effective,
            statistics=statistics,
            device="cpu",
        ).model
        torch.manual_seed(101)
        micro = build_bist_ndr_pair_model(
            method=method,
            node_count=3,
            effective_profile=effective,
            statistics=statistics,
            device="cpu",
        ).model
        full.eval()
        micro.eval()

        full_loss, _, full_chunks = backward_bist_ndr_pair_batch(
            model=full,
            tensors=tensors,
            loss_batch=loss_batch,
            loss_args=loss_args,
            device=torch.device("cpu"),
            max_microbatch_size=4,
        )
        micro_loss, _, micro_chunks = backward_bist_ndr_pair_batch(
            model=micro,
            tensors=tensors,
            loss_batch=loss_batch,
            loss_args=loss_args,
            device=torch.device("cpu"),
            max_microbatch_size=2,
        )

        np.testing.assert_allclose(micro_loss, full_loss, rtol=1e-5, atol=1e-5)
        assert full_chunks == 1
        assert micro_chunks == 2
        full_gradients = dict(full.named_parameters())
        for name, parameter in micro.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                full_gradients[name].grad,
                rtol=2e-4,
                atol=2e-5,
            )


def test_pair_task_command_binds_the_locked_matrix_and_result_root():
    command = build_bist_ndr_pair_task_command(
        python_executable=Path("/venv/bin/python"),
        manifest_path=Path("/repo/config/matrix.pair.json"),
        manifest_sha256="1" * 64,
        base_manifest_path=Path("/repo/config/matrix.base.json"),
        base_manifest_sha256="2" * 64,
        task_id="pair010_bist_ndr_alameda_seed42",
        splits_root=Path("/data/official_splits"),
        results_root=Path("/results/pair_gpu0"),
        device="cuda:0",
    )

    assert command[0] == str(Path("/venv/bin/python"))
    assert command[1:4] == ["-X", "utf8", "-m"]
    assert command[4] == (
        "scripts.reproduction.igstgnn_table4.bist_ndr_pair_runner"
    )
    assert command[command.index("--manifest-sha256") + 1] == "1" * 64
    assert command[command.index("--base-manifest-sha256") + 1] == "2" * 64
    assert command[command.index("--task-id") + 1] == (
        "pair010_bist_ndr_alameda_seed42"
    )
    assert command[command.index("--results-root") + 1] == str(
        Path("/results/pair_gpu0")
    )


def test_pair_task_environment_preserves_the_host_python_package_path():
    environment = build_bist_ndr_pair_task_environment(
        {"PYTHONPATH": str(Path("/host/site-packages")), "KEEP": "yes"}
    )

    python_paths = environment["PYTHONPATH"].split(os.pathsep)
    assert python_paths[0] == str(PROJECT_ROOT)
    assert python_paths[1] == str(Path("/host/site-packages"))
    assert environment["KEEP"] == "yes"
