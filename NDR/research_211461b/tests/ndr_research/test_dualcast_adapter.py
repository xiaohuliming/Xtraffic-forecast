"""Reduction gradients, real calendar semantics and pinned reference loading."""

from pathlib import Path

import pytest
import torch

from ndr_research.dualcast_adapter import (
    DualCastSReference,
    reference_calendar,
    torch_degree,
    torch_sum_scatter,
)

CHECKOUT = Path(
    "/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/references/DualCast"
)


def test_sum_reduction_preserves_every_duplicate_edge_and_its_gradient():
    x = torch.tensor(
        [[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0], [17.0, 19.0]]], requires_grad=True
    )
    indices = torch.tensor([0, 1, 0, 2])
    reduced = torch_sum_scatter(x, indices, dim=-2)
    torch.testing.assert_close(
        reduced, torch.tensor([[[13.0, 16.0], [5.0, 7.0], [17.0, 19.0]]])
    )
    weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    (reduced * weights).sum().backward()
    torch.testing.assert_close(x.grad, weights[:, indices])
    torch.testing.assert_close(
        torch_degree(indices, 4, dtype=torch.float32),
        torch.tensor([2.0, 1.0, 1.0, 0.0]),
    )


def test_real_weekdays_align_with_author_workday_and_weekend_prototype_groups():
    calendar = torch.zeros(7, 12, 2)
    calendar[..., 0] = 90 / 288
    calendar[..., 1] = torch.arange(7)[:, None] / 7
    te = reference_calendar(calendar)
    assert te[:, 0, 0].tolist() == [3, 4, 5, 6, 0, 1, 2]
    assert te[:, 0, 2].tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert te[:, 0, 1].tolist() == [90] * 7


@pytest.mark.parametrize("batch", [1, 3])
def test_pinned_author_model_handles_the_retained_one_sample_tail_batch(batch):
    if not CHECKOUT.exists():
        pytest.skip("author reference is supplied outside versioned source")
    torch.set_num_threads(2)
    torch.manual_seed(25)
    model = DualCastSReference(
        checkout=CHECKOUT,
        adjacency=torch.eye(8),
        flow_mean=70,
        flow_std=20,
        embed_dim=16,
        heads=4,
    )
    history = 50 + torch.rand(batch, 8, 12) * 30
    calendar = torch.zeros(batch, 12, 2)
    calendar[..., 0] = 90 / 288
    for i in range(batch):
        calendar[i, :, 1] = (i % 7) / 7
    model.train()
    predicted = model(history, calendar)
    assert predicted.shape == (batch, 8, 12)
    assert model.last_aux["environment"].shape == (batch, 16)
    assert torch.isfinite(predicted).all()
    (predicted - 60).abs().mean().backward()
    assert model.base.conv1.weight.grad is not None
    assert torch.isfinite(model.base.conv1.weight.grad).all()
    assert model.provenance["model_config"]["num_prototypes"] == 17


def test_modified_reference_bytes_are_rejected_before_model_execution(tmp_path):
    from ndr_research.dualcast_adapter import load_reference_class

    (tmp_path / "DualCastS/model").mkdir(parents=True)
    (tmp_path / "DualCastS/model/model_.py").write_text(
        "raise RuntimeError('must never execute')"
    )
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="identity changed"):
        load_reference_class(tmp_path)
