from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from scripts.train_baseline_zoo import (
    build_model,
    resolve_training_recipe,
    validate_training_contract,
)


def test_dcrnn_convergence_recipe_restores_native_scheduler_without_changing_fixed30() -> None:
    fixed = resolve_training_recipe("dcrnn", "fixed30")
    convergence = resolve_training_recipe("dcrnn", "convergence100")

    assert fixed["lr"] == convergence["lr"] == 1e-2
    assert fixed["clip"] == convergence["clip"] == 5.0
    assert fixed["scheduler"] == {
        "type": "MultiStepLR",
        "milestones": [10, 20, 27],
        "gamma": 0.1,
    }
    assert convergence["scheduler"] == {
        "type": "MultiStepLR",
        "milestones": [10, 50, 90],
        "gamma": 0.1,
    }


def test_convergence_recipe_rejects_short_budget_or_early_patience() -> None:
    validate_training_contract(
        Namespace(recipe_profile="convergence100", epochs=100, patience=30, seed=42)
    )

    with pytest.raises(ValueError, match="100 epochs"):
        validate_training_contract(
            Namespace(recipe_profile="convergence100", epochs=30, patience=30, seed=42)
        )
    with pytest.raises(ValueError, match="patience 30"):
        validate_training_contract(
            Namespace(recipe_profile="convergence100", epochs=100, patience=6, seed=42)
        )


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
