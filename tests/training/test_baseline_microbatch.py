from __future__ import annotations

import numpy as np
import torch

from scripts.train_baseline_zoo import (
    backward_effective_masked_mae_batch,
    masked_mae,
)


class _ToyForecast(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., :1] * self.weight


def test_microbatch_backward_matches_one_effective_masked_mae_batch() -> None:
    x = torch.arange(4 * 3 * 2 * 3, dtype=torch.float32).reshape(4, 3, 2, 3)
    target = torch.linspace(0.0, 4.0, 4 * 2 * 3).reshape(4, 2, 3)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0, 0] = False

    full = _ToyForecast()
    full_prediction = full(x).squeeze(-1).permute(0, 2, 1).contiguous()
    full_loss = masked_mae(full_prediction, target, mask)
    full_loss.backward()

    split = _ToyForecast()
    split.load_state_dict(full.state_dict())
    split_loss = backward_effective_masked_mae_batch(
        model=split,
        name="toy",
        x=x,
        label=None,
        y_true=target,
        y_mask=mask,
        iter_cnt=0,
        horizon=3,
        node_count=2,
        device=torch.device("cpu"),
        mean=0.0,
        std=1.0,
        max_microbatch_size=2,
    )

    np.testing.assert_allclose(split_loss, float(full_loss.item()), rtol=1e-6)
    torch.testing.assert_close(split.weight.grad, full.weight.grad)
