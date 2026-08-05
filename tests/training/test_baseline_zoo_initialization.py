from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from scripts.train_baseline_zoo import build_model


def test_astgcn_factory_applies_the_upstream_engine_initialization() -> None:
    torch.manual_seed(42)

    model, _, _ = build_model(
        "astgcn",
        10,
        torch.device("cpu"),
        np.ones((10, 10), dtype=np.float32),
        Namespace(),
    )

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_dstagnn_factory_applies_the_upstream_engine_initialization(
    monkeypatch,
) -> None:
    import dstagnn

    class UninitializedDSTAGNN(torch.nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.full((3, 3), float("nan")))
            self.vector = torch.nn.Parameter(torch.full((3,), float("nan")))

    monkeypatch.setattr(dstagnn, "DSTAGNN", UninitializedDSTAGNN)
    model, _, _ = build_model(
        "dstagnn",
        10,
        torch.device("cpu"),
        np.ones((10, 10), dtype=np.float32),
        Namespace(),
    )

    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_dgcrn_preserves_single_item_partial_batch_dimension() -> None:
    model, _, _ = build_model(
        "dgcrn",
        4,
        torch.device("cpu"),
        np.ones((4, 4), dtype=np.float32),
        Namespace(),
    )
    model.eval()
    model_input = torch.zeros(1, 12, 4, 3)
    label = torch.zeros(1, 12, 4, 1)

    with torch.no_grad():
        prediction = model(
            model_input,
            label,
            batches_seen=0,
            task_level=12,
        )

    assert prediction.shape == (1, 12, 4, 1)
