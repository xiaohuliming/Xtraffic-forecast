"""Faithful PyTorch implementation of DCRNN's model core.

The implementation follows Li et al., ICLR 2018:

* bidirectional random-walk diffusion convolution;
* diffusion convolution inside a GRU cell (DCGRU);
* a multi-layer encoder-decoder;
* curriculum learning / scheduled sampling during training.

Tensor convention throughout this file is ``(batch, time, node, feature)``.
The model has no knowledge of XTraffic incidents or affected-node labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class DiffusionGraphConv(nn.Module):
    """Diffusion convolution over one or more random-walk supports."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        max_diffusion_step: int,
        num_supports: int,
        bias_start: float = 0.0,
    ) -> None:
        super().__init__()
        if max_diffusion_step < 0:
            raise ValueError("max_diffusion_step must be non-negative")
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.output_dim = int(output_dim)
        self.max_diffusion_step = int(max_diffusion_step)
        self.num_supports = int(num_supports)

        n_terms = 1 + self.num_supports * self.max_diffusion_step
        self.projection = nn.Linear(
            n_terms * (self.input_dim + self.state_dim), self.output_dim
        )
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, bias_start)

    @staticmethod
    def _support_mm(support: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply an ``(N,N)`` support to ``x`` shaped ``(B,N,F)``."""
        batch, nodes, features = x.shape
        flat = x.permute(1, 0, 2).reshape(nodes, batch * features)
        if support.layout in {torch.sparse_coo, torch.sparse_csr}:
            propagated = torch.sparse.mm(support, flat)
        else:
            propagated = support @ flat
        return propagated.reshape(nodes, batch, features).permute(1, 0, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(supports) != self.num_supports:
            raise ValueError(
                f"expected {self.num_supports} supports, received {len(supports)}"
            )
        x0 = torch.cat([inputs, state], dim=-1)
        terms = [x0]
        if self.max_diffusion_step:
            for support in supports:
                x_prev = x0
                x_curr = self._support_mm(support, x0)
                terms.append(x_curr)
                for _step in range(2, self.max_diffusion_step + 1):
                    x_next = 2.0 * self._support_mm(support, x_curr) - x_prev
                    terms.append(x_next)
                    x_prev, x_curr = x_curr, x_next
        return self.projection(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose affine maps are replaced by diffusion convolutions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_diffusion_step: int,
        num_supports: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate = DiffusionGraphConv(
            input_dim=input_dim,
            state_dim=hidden_dim,
            output_dim=2 * hidden_dim,
            max_diffusion_step=max_diffusion_step,
            num_supports=num_supports,
            bias_start=1.0,
        )
        self.candidate = DiffusionGraphConv(
            input_dim=input_dim,
            state_dim=hidden_dim,
            output_dim=hidden_dim,
            max_diffusion_step=max_diffusion_step,
            num_supports=num_supports,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        supports: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        reset, update = torch.sigmoid(self.gate(inputs, state, supports)).chunk(
            2, dim=-1
        )
        candidate = torch.tanh(self.candidate(inputs, reset * state, supports))
        return update * state + (1.0 - update) * candidate


class DCRNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_diffusion_step: int,
        num_supports: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cells = nn.ModuleList(
            [
                DCGRUCell(
                    input_dim=input_dim if layer == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_supports=num_supports,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(
        self, inputs: torch.Tensor, supports: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        batch, _steps, nodes, _features = inputs.shape
        states = [inputs.new_zeros(batch, nodes, self.hidden_dim) for _ in self.cells]
        for step in range(inputs.shape[1]):
            layer_input = inputs[:, step]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(layer_input, states[layer], supports)
                layer_input = states[layer]
        return states


class DCRNNDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_diffusion_step: int,
        num_supports: int,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.cells = nn.ModuleList(
            [
                DCGRUCell(
                    input_dim=output_dim if layer == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_supports=num_supports,
                )
                for layer in range(num_layers)
            ]
        )
        # The official DCGRU decoder projection is a weight matrix without bias.
        self.projection = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward_step(
        self,
        inputs: torch.Tensor,
        states: list[torch.Tensor],
        supports: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        next_states: list[torch.Tensor] = []
        layer_input = inputs
        for layer, cell in enumerate(self.cells):
            next_state = cell(layer_input, states[layer], supports)
            next_states.append(next_state)
            layer_input = next_state
        return self.projection(layer_input), next_states


class DCRNN(nn.Module):
    """Diffusion Convolutional Recurrent Neural Network.

    ``supports`` must contain the forward and backward random-walk matrices. They
    are registered once on the top-level module and shared by every DCGRU cell.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        horizon: int,
        supports: Sequence[torch.Tensor],
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_diffusion_step: int = 2,
        cl_decay_steps: int = 2000,
    ) -> None:
        super().__init__()
        if len(supports) == 0:
            raise ValueError("DCRNN requires at least one graph support")
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_diffusion_step = int(max_diffusion_step)
        self.cl_decay_steps = int(cl_decay_steps)

        for index, support in enumerate(supports):
            if tuple(support.shape) != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"support {index} has shape {tuple(support.shape)}, expected "
                    f"({self.num_nodes}, {self.num_nodes})"
                )
            if support.layout == torch.sparse_coo:
                support = support.coalesce()
            # Graph supports are deterministically rebuilt from the saved config.
            # Keeping them non-persistent avoids duplicating sparse matrices in
            # every checkpoint while they still follow model.to(device).
            self.register_buffer(f"support_{index}", support, persistent=False)
        self.num_supports = len(supports)

        self.encoder = DCRNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_diffusion_step=max_diffusion_step,
            num_supports=self.num_supports,
        )
        self.decoder = DCRNNDecoder(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_diffusion_step=max_diffusion_step,
            num_supports=self.num_supports,
        )

    @property
    def supports(self) -> list[torch.Tensor]:
        return [getattr(self, f"support_{i}") for i in range(self.num_supports)]

    def sampling_threshold(self, batches_seen: int) -> float:
        """Curriculum-learning threshold from the original DCRNN code."""
        if self.cl_decay_steps <= 0:
            return 0.0
        decay = float(self.cl_decay_steps)
        return decay / (decay + math.exp(float(batches_seen) / decay))

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        batches_seen: int = 0,
        use_curriculum_learning: bool = True,
    ) -> torch.Tensor:
        """Predict a standardized sequence.

        Args:
            inputs: ``(B, T_in, N, input_dim)``.
            labels: optional standardized targets ``(B, horizon, N, output_dim)``.
            labels_mask: target validity mask with the same shape as ``labels``.
            batches_seen: optimizer-step count for scheduled sampling.
            use_curriculum_learning: only effective while the module is training.
        """
        if inputs.ndim != 4 or inputs.shape[2] != self.num_nodes:
            raise ValueError(
                f"inputs must be (B,T,{self.num_nodes},F), got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"input feature dimension {inputs.shape[-1]} != {self.input_dim}"
            )
        if labels is not None and tuple(labels.shape[1:]) != (
            self.horizon,
            self.num_nodes,
            self.output_dim,
        ):
            raise ValueError(
                f"labels must be (B,{self.horizon},{self.num_nodes},{self.output_dim})"
            )

        states = self.encoder(inputs, self.supports)
        decoder_input = inputs.new_zeros(
            inputs.shape[0], self.num_nodes, self.output_dim
        )
        outputs = []
        threshold = self.sampling_threshold(batches_seen)
        for step in range(self.horizon):
            output, states = self.decoder.forward_step(
                decoder_input, states, self.supports
            )
            outputs.append(output)
            decoder_input = output
            if (
                self.training
                and use_curriculum_learning
                and labels is not None
                and torch.rand((), device=inputs.device).item() < threshold
            ):
                teacher = labels[:, step]
                if labels_mask is not None:
                    teacher = torch.where(labels_mask[:, step].bool(), teacher, output)
                decoder_input = teacher
        return torch.stack(outputs, dim=1)
