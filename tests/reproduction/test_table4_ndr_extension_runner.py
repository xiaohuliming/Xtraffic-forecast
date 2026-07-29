from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    fit_training_climatology,
    prepare_ndr_batch,
)


class _StaticLoader:
    def __init__(self, batch):
        self._batch = batch

    def get_iterator(self):
        yield self._batch


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
