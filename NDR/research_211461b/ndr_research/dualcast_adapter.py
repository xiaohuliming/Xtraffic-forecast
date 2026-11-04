"""Pinned DualCast-S reference with native Torch reductions and masked units.

The public model body is loaded unchanged from its author checkout. Only its
unused MessagePassing import and degree/scatter imports are resolved locally,
so no legacy binary PyG extensions or shared environment changes are required.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

UPSTREAM_COMMIT = "50425c12f2fc4e6ba6069d028540d2373e5e945a"
MODEL_SHA256 = "47462de01c827296a8aed4b6e628bd7dd041fee64a209f026c7452cc3c65d336"
MODEL_RELATIVE_PATH = "DualCastS/model/model_.py"


def torch_degree(index: torch.Tensor, num_nodes: int, dtype=None) -> torch.Tensor:
    if index.dtype != torch.long or index.ndim != 1:
        raise ValueError("degree indices must be a long vector")
    if num_nodes < 0 or (
        index.numel() and (index.min() < 0 or index.max() >= num_nodes)
    ):
        raise ValueError("degree indices leave the declared graph")
    output = torch.zeros(num_nodes, device=index.device, dtype=dtype)
    return output.scatter_add_(0, index, output.new_ones(index.numel()))


def torch_sum_scatter(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    reduce: str = "sum",
) -> torch.Tensor:
    """The exact sum-only subset used by the pinned author's model."""
    if reduce not in {"sum", "add"}:
        raise ValueError("the reference only requires sum reductions")
    dim %= source.ndim
    if (
        index.dtype != torch.long
        or index.ndim != 1
        or index.numel() != source.shape[dim]
    ):
        raise ValueError("scatter indices must match the selected source dimension")
    if index.device != source.device or (index.numel() and index.min() < 0):
        raise ValueError("scatter index device or range mismatch")
    shape = list(source.shape)
    shape[dim] = int(index.max()) + 1 if index.numel() else 0
    output = source.new_zeros(shape)
    return output.index_add_(dim, index, source)


def load_reference_class(checkout: Path | str):
    checkout = Path(checkout).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=20,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            "DualCast reference Git identity changed or is unavailable"
        ) from error
    if commit != UPSTREAM_COMMIT:
        raise ValueError("DualCast reference Git identity changed")
    path = checkout / MODEL_RELATIVE_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256
    ):
        raise ValueError("DualCast reference model bytes changed")
    syntax = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    syntax.body = [
        node
        for node in syntax.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            in {"torch_geometric.nn", "torch_geometric.utils", "torch_scatter"}
        )
    ]
    namespace = {
        "__name__": "ndr_research_locked_dualcast",
        "degree": torch_degree,
        "scatter": torch_sum_scatter,
    }
    exec(compile(syntax, str(path), "exec"), namespace)
    return namespace["STTN"]


def reference_calendar(time_feat: torch.Tensor) -> torch.Tensor:
    """Map real normalized weekday to the author's Friday-origin categories."""
    if (
        time_feat.ndim != 3
        or time_feat.shape[-1] != 2
        or not torch.isfinite(time_feat).all()
    ):
        raise ValueError("DualCast calendar must be finite (B, T, 2)")
    if not ((time_feat >= 0) & (time_feat < 1)).all():
        raise ValueError("normalized historical calendar is outside its grid")
    tod_scaled = time_feat[..., 0] * 288
    dow_scaled = time_feat[..., 1] * 7
    tod, dow = tod_scaled.round().long(), dow_scaled.round().long()
    if not torch.allclose(tod_scaled, tod.to(tod_scaled), atol=1e-4, rtol=0):
        raise ValueError("time of day is not on a five-minute grid")
    if not torch.allclose(dow_scaled, dow.to(dow_scaled), atol=1e-4, rtol=0):
        raise ValueError("weekday is not on the integer weekly grid")
    friday_origin = (dow - 4) % 7
    weekend = (dow >= 5).long()
    return torch.stack((friday_origin, tod, weekend), -1)


class DualCastSReference(nn.Module):
    """A regional strong reference, not a node-inductive transfer model."""

    def __init__(
        self,
        *,
        checkout: Path | str,
        adjacency: torch.Tensor,
        flow_mean: float,
        flow_std: float,
        device: str | torch.device = "cpu",
        embed_dim: int = 64,
        layers: int = 1,
        heads: int = 8,
        forward_expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        device = torch.device(device)
        if (
            adjacency.ndim != 2
            or adjacency.shape[0] != adjacency.shape[1]
            or adjacency.shape[0] < 2
        ):
            raise ValueError(
                "DualCast regional adjacency must be square with at least two nodes"
            )
        if not torch.isfinite(adjacency).all() or flow_std <= 0:
            raise ValueError("reference adjacency and fitted scale must be finite")
        if not torch.isfinite(torch.tensor([flow_mean, flow_std])).all():
            raise ValueError("reference normalization must be finite")
        self.register_buffer("flow_mean", torch.tensor(flow_mean, dtype=torch.float32))
        self.register_buffer("flow_std", torch.tensor(flow_std, dtype=torch.float32))
        self.nodes = adjacency.shape[0]
        config = dict(
            input_dim=1,
            output_dim=1,
            embed_dim=embed_dim,
            L=layers,
            K=heads,
            T=288,
            forward_expansion=forward_expansion,
            dropout_rate=dropout,
            num_his=12,
            num_pred=12,
            num_prototypes=17,
        )
        reference = load_reference_class(checkout)
        self.base = reference(
            SimpleNamespace(**config),
            adjacency.to(device=device, dtype=torch.float32),
            device,
        )
        self.to(device)
        self.last_aux: dict[str, Any] | None = None
        self.provenance = {
            "model": "author_DualCastS_STTN",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_model_sha256": MODEL_SHA256,
            "model_config": config,
            "compatibility": [
                "native_torch_degree_and_sum_scatter",
                "preserve_global_feature_batch_axis",
            ],
            "normalizer_fit": "unique_raw_observed_source_training_timestamps",
            "calendar": "real_calendar_to_friday_origin_weekday; weekend_only_holiday_flag",
            "claim_scope": "same_v3_regional_reference; not_original_PeMS_score_or_inductive_transfer",
        }

    def forward(self, flow_hist: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        if flow_hist.ndim != 3 or flow_hist.shape[1:] != (self.nodes, 12):
            raise ValueError("regional reference history must match (B, fitted N, 12)")
        if (
            time_feat.shape != (flow_hist.shape[0], 12, 2)
            or not torch.isfinite(flow_hist).all()
        ):
            raise ValueError("reference history/calendar are invalid")
        normalized = ((flow_hist - self.flow_mean) / self.flow_std).permute(0, 2, 1)
        intrinsic, environment, prototypes, anchors, prediction = self.base(
            normalized, reference_calendar(time_feat)
        )
        batch = flow_hist.shape[0]
        self.last_aux = {
            "intrinsic": intrinsic.reshape(batch, -1),
            "environment": environment.reshape(batch, -1),
            "prototype_members": prototypes,
            "anchors": anchors,
        }
        if (
            prediction.shape != (batch, 12, self.nodes)
            or not torch.isfinite(prediction).all()
        ):
            raise FloatingPointError("author reference produced invalid predictions")
        return prediction.permute(0, 2, 1) * self.flow_std + self.flow_mean
