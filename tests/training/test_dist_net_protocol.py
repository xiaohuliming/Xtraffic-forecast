from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.train_dist_net import run_forward_loss


class _DummyModel:
    def __call__(self, batch, edge_index):
        prediction = torch.zeros_like(batch["y_true"])
        return {
            "pred": prediction,
            "pred_normal_final": prediction,
        }


def test_blind_mode_never_uses_post_event_affected_labels():
    batch = {
        "incident_feat": torch.ones(2, 3, 4),
        "incident_mask": torch.ones(2, 3, dtype=torch.bool),
        "rel_feat": torch.ones(2, 3, 5, 4),
        "y_true": torch.ones(2, 5, 2, 1),
        "y_baseline": torch.zeros(2, 5, 2, 1),
        "y_mask": torch.ones(2, 5, 2, 1, dtype=torch.bool),
    }
    args = SimpleNamespace(
        blind=True,
        single_branch=False,
        alpha=1.0,
        beta=0.3,
        gamma=0.5,
        lam_aff=0.3,
    )

    with patch(
        "scripts.train_dist_net.compute_losses",
        return_value={"L_total": torch.tensor(0.0)},
    ) as compute:
        run_forward_loss(_DummyModel(), batch, edge_index=None, args=args)

    kwargs = compute.call_args.kwargs
    assert kwargs["beta"] == 0.0
    assert kwargs["gamma"] == 0.0
    assert not kwargs["affected_mask"].any()
    assert not batch["incident_feat"].any()
    assert not batch["incident_mask"].any()
    assert not batch["rel_feat"].any()
