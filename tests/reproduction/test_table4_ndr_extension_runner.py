from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    fit_training_climatology,
    prepare_ndr_batch,
)
from scripts.train_staeformer_deseason import (
    build_ndr_loss_normalizers,
    compute_loss,
)


class _StaticLoader:
    def __init__(self, batch):
        self._batch = batch

    def get_iterator(self):
        yield self._batch


class _TinyDualNDR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_logit = torch.nn.Parameter(torch.tensor(0.2))
        self.normal_bias = torch.nn.Parameter(torch.tensor(0.4))
        self.disturbance_bias = torch.nn.Parameter(torch.tensor(-0.3))
        self.register_buffer("sd_node", torch.tensor([1.0, 2.0]))
        self.oracle_sup = True
        self.dual = True
        self.abank = False
        self.aux = None

    def forward(self, baseline, persistence):
        gate = torch.sigmoid(self.gate_logit).expand_as(baseline)
        normal = baseline + self.normal_bias
        disturbance = persistence + self.disturbance_bias
        self.aux = {
            "g": gate,
            "B": baseline,
            "P": persistence,
            "yn": normal,
            "yd": disturbance,
        }
        return gate * normal + (1.0 - gate) * disturbance


def test_two_microbatch_ndr_loss_matches_full_batch_loss_and_gradients():
    target = torch.tensor(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    mask = torch.ones_like(target)
    baseline = target - 0.7
    persistence = target + 0.9
    args = SimpleNamespace(lambda_gate=1.0, lambda_spec=0.3)

    full_model = _TinyDualNDR()
    full_prediction = full_model(baseline, persistence)
    full_loss, _ = compute_loss(
        full_model,
        full_prediction,
        {
            "y_true": target.unsqueeze(-1),
            "y_mask": mask.unsqueeze(-1),
        },
        args,
        torch.device("cpu"),
    )
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in full_model.named_parameters()
    }

    micro_model = _TinyDualNDR()
    normalizers = build_ndr_loss_normalizers(
        micro_model,
        {
            "y_true": target.unsqueeze(-1),
            "y_mask": mask.unsqueeze(-1),
        },
        baseline=baseline,
        persistence=persistence,
        device=torch.device("cpu"),
    )
    micro_losses = []
    for chunk in (slice(0, 2), slice(2, 4)):
        prediction = micro_model(
            baseline[chunk],
            persistence[chunk],
        )
        loss, _ = compute_loss(
            micro_model,
            prediction,
            {
                "y_true": target[chunk].unsqueeze(-1),
                "y_mask": mask[chunk].unsqueeze(-1),
            },
            args,
            torch.device("cpu"),
            normalizers=normalizers,
        )
        loss.backward()
        micro_losses.append(loss.detach())

    torch.testing.assert_close(sum(micro_losses), full_loss.detach())
    for name, parameter in micro_model.named_parameters():
        torch.testing.assert_close(parameter.grad, full_gradients[name])


def test_ndr_model_module_import_does_not_require_h5py():
    project_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "h5py" or name.startswith("h5py."):
        raise ModuleNotFoundError("h5py is intentionally unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import scripts.train_staeformer_deseason
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _training_batch():
    x_data = np.zeros((2, 12, 2, 3), dtype=np.float32)
    y_data = np.zeros((2, 12, 2, 3), dtype=np.float32)
    x_ticks = np.arange(12, dtype=np.float32)
    y_ticks = np.arange(12, 24, dtype=np.float32)
    x_data[..., 1] = x_ticks[None, :, None] / 288
    y_data[..., 1] = y_ticks[None, :, None] / 288
    x_data[..., 2] = 0
    y_data[..., 2] = 0
    x_data[0, :, 0, 0] = 2
    x_data[1, :, 0, 0] = 4
    y_data[0, :, 0, 0] = 6
    y_data[1, :, 0, 0] = 8
    return {
        "x_data": x_data,
        "y_data": y_data,
        "sample_indices": np.array([0, 1], dtype=np.int64),
    }


def test_ndr_climatology_uses_only_the_supplied_training_loader():
    batch = _training_batch()

    statistics = fit_training_climatology(
        _StaticLoader(batch),
        node_count=2,
    )
    prepared = prepare_ndr_batch(batch, statistics)

    assert statistics.baseline.shape == (7, 288, 2)
    assert statistics.baseline[0, 0, 0] == 3
    assert statistics.baseline[0, 12, 0] == 7
    assert statistics.baseline[0, 100, 0] == 0
    assert statistics.baseline[:, :, 1].sum() == 0
    assert statistics.flow_mean == 5
    np.testing.assert_allclose(statistics.flow_std, np.sqrt(5), rtol=1e-6)
    assert statistics.residual_std == 1
    np.testing.assert_allclose(statistics.residual_std_by_node, [1, 1])
    assert prepared["flow_hist"].shape == (2, 2, 12)
    assert prepared["target"].shape == (2, 2, 12)
    np.testing.assert_array_equal(prepared["x_baseline"][:, 0], 3)
    np.testing.assert_array_equal(prepared["y_baseline"][:, 0], 7)
    assert prepared["target_mask"][:, 0].all()
    assert not prepared["target_mask"][:, 1].any()


def test_ndr_adapter_restores_official_normalized_flow_to_raw_units():
    raw = _training_batch()
    normalized = {
        **raw,
        "x_data": np.array(raw["x_data"], copy=True),
        "y_data": np.array(raw["y_data"], copy=True),
    }
    normalized["x_data"][..., 0] = (normalized["x_data"][..., 0] - 10) / 2
    normalized["y_data"][..., 0] = (normalized["y_data"][..., 0] - 10) / 2

    statistics = fit_training_climatology(
        _StaticLoader(normalized),
        node_count=2,
        source_normalization_mean=10,
        source_normalization_std=2,
    )
    prepared = prepare_ndr_batch(normalized, statistics)

    assert statistics.source_normalization_mean == 10
    assert statistics.source_normalization_std == 2
    assert statistics.baseline[0, 0, 0] == 3
    assert statistics.baseline[0, 12, 0] == 7
    np.testing.assert_array_equal(prepared["flow_hist"][:, 0], [[2] * 12, [4] * 12])
    np.testing.assert_array_equal(prepared["target"][:, 0], [[6] * 12, [8] * 12])
    assert not prepared["target_mask"][:, 1].any()
