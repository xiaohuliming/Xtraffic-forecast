"""Occupancy and speed residual inputs for the original NDR-Net s0g; no training entrypoint.

The baseline research modules stay byte-identical, so checkpoints of the stage-1
reference runs remain admissible under their recorded source snapshots. This module
only adds: a train-only climatology for any cache channel, a loader that also yields the
auxiliary history of the same samples, and a model whose backbone input embedding
receives the auxiliary residuals through a zero-initialized projection.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.protocol_v3_event_extension_data import WeeklyClimatology, _weekly_slots_from_history
from scripts.train_staeformer_deseason import STAEformerDeseason, VARIANTS

from .data import (
    ResearchData,
    ResearchEventLoader,
    array_digest,
    load_research_data,
    prepare_research_batch,
)
from .models import research_factory_kwargs

MULTICHANNEL_VARIANT = "multichannel"
CACHE_CHANNEL_NAMES = ("flow", "occupancy", "speed")
AUXILIARY_CHANNELS = {"occupancy": 1, "speed": 2}
MULTICHANNEL_AUXILIARY_CHANNELS = ("occupancy", "speed")
AUXILIARY_BATCH_KEYS = ("aux_hist", "aux_baseline", "aux_mask")


def checked_auxiliary_channels(names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names or len(set(names)) != len(names) or any(name not in AUXILIARY_CHANNELS for name in names):
        raise ValueError("auxiliary channels must be distinct names among occupancy and speed")
    return names


def fit_channel_climatology(region_data: Any, channel: int) -> WeeklyClimatology:
    """fit_weekly_climatology for one cache channel, from raw train timestamps only."""
    if type(channel) is not int or not 0 <= channel < len(CACHE_CHANNEL_NAMES):
        raise ValueError("cache channel index is outside flow, occupancy and speed")
    train_end = int(region_data.train_end)
    node_count = int(region_data.N)
    if train_end <= 0 or node_count <= 0:
        raise ValueError("region training boundary is invalid")
    if region_data.flow_series is None:
        values_all, mask_all = region_data._slice_traffic(0, train_end)
    else:
        values_all = region_data.flow_series[:train_end]
        mask_all = region_data.flow_mask[:train_end]
    values = np.asarray(values_all[..., channel], dtype=np.float32)
    mask = np.asarray(mask_all[..., channel], dtype=np.bool_)
    if values.shape != (train_end, node_count) or mask.shape != values.shape:
        raise ValueError("training channel tensors have an invalid shape")
    observed = np.asarray(values[mask], dtype=np.float64)
    if observed.size == 0 or not np.isfinite(observed).all():
        raise ValueError("training channel has no finite observed values")
    mean = float(observed.mean(dtype=np.float64))
    std = float(observed.std(dtype=np.float64))
    if not np.isfinite(std) or std <= 1e-6:
        std = 1.0
    dow = np.asarray(region_data.dow[:train_end], dtype=np.int64)
    tod = np.asarray(region_data.tod[:train_end], dtype=np.int64)
    baseline = np.zeros((7, 288, node_count), dtype=np.float32)
    for day in range(7):
        for slot in range(288):
            selected = np.flatnonzero((dow == day) & (tod == slot))
            if selected.size == 0:
                continue
            slot_values = np.where(mask[selected], values[selected], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(slot_values, axis=0)
            baseline[day, slot] = np.where(np.isfinite(medians), medians, np.float32(0))
    residual = np.asarray(values - baseline[dow, tod], dtype=np.float64)
    residual_std = float(residual[mask].std(dtype=np.float64))
    if not np.isfinite(residual_std) or residual_std <= 1e-6:
        residual_std = 1.0
    residual_std_by_node = np.ones(node_count, dtype=np.float32)
    for node in range(node_count):
        node_values = residual[:, node][mask[:, node]]
        if node_values.size:
            value = float(node_values.std(dtype=np.float64))
            if np.isfinite(value) and value > 1e-6:
                residual_std_by_node[node] = value
    baseline.setflags(write=False)
    residual_std_by_node.setflags(write=False)
    return WeeklyClimatology(
        baseline=baseline, flow_mean=mean, flow_std=std, residual_std=residual_std,
        residual_std_by_node=residual_std_by_node, valid_training_points=int(observed.size),
    )


class MultiChannelEventLoader(ResearchEventLoader):
    """The research loader, plus the auxiliary history of the very same samples."""

    def __init__(self, *, auxiliary_channels: Sequence[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.auxiliary_channels = checked_auxiliary_channels(auxiliary_channels)

    def _build_batch(self, local_indices: np.ndarray) -> dict[str, Any]:
        batch = super()._build_batch(local_indices)
        channels = [AUXILIARY_CHANNELS[name] for name in self.auxiliary_channels]
        values, masks = [], []
        for row, index in enumerate(local_indices):
            sample = self.region_data.get_sample(int(index))
            history = np.asarray(sample["x_hist"], dtype=np.float32)
            if (int(sample["sample_start"]) != int(batch["sample_start"][row])
                    or not np.array_equal(history[..., 0].T, batch["x_raw_flow"][row])):
                raise RuntimeError("auxiliary history is not aligned with the flow history")
            values.append(history[..., channels])
            masks.append(np.asarray(sample["x_hist_mask"], dtype=np.bool_)[..., channels])
        batch["x_raw_aux"] = np.stack(values)
        batch["x_aux_mask"] = np.stack(masks)
        return batch


def prepare_auxiliary_history(
    batch: Mapping[str, Any], auxiliary_statistics: Sequence[WeeklyClimatology],
) -> dict[str, np.ndarray]:
    """Raw auxiliary history with its train-only weekly climatology and observation mask."""
    if "x_raw_aux" not in batch or "x_aux_mask" not in batch:
        raise ValueError("auxiliary statistics require auxiliary history in the batch")
    values = np.asarray(batch["x_raw_aux"], dtype=np.float32)
    mask = np.asarray(batch["x_aux_mask"])
    channels = len(auxiliary_statistics)
    if channels == 0 or values.ndim != 4 or values.shape[-1] != channels or mask.shape != values.shape:
        raise ValueError("auxiliary history must have shape (B, N, T_h, C) matching its statistics")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("auxiliary observation mask must contain zero or one")
    x_dow, x_tod, _, _ = _weekly_slots_from_history(np.asarray(batch["time_feat"], dtype=np.float32))
    baseline = np.stack(
        [statistics.baseline[x_dow, x_tod].transpose(0, 2, 1) for statistics in auxiliary_statistics],
        axis=-1,
    ).astype(np.float32, copy=False)
    if baseline.shape != values.shape:
        raise ValueError("auxiliary climatology does not match the auxiliary history")
    return {"aux_hist": values, "aux_baseline": baseline, "aux_mask": mask.astype(np.bool_)}


def prepare_multichannel_batch(
    batch: Mapping[str, Any], statistics: WeeklyClimatology,
    auxiliary_statistics: Sequence[WeeklyClimatology],
) -> dict[str, np.ndarray]:
    """The unchanged research batch, plus the auxiliary history arrays."""
    prepared = prepare_research_batch(batch, statistics)
    prepared.update(prepare_auxiliary_history(batch, auxiliary_statistics))
    return prepared


@dataclass
class MultiChannelData(ResearchData):
    auxiliary_channels: tuple[str, ...]
    auxiliary_statistics: tuple[WeeklyClimatology, ...]


def cache_channel_names(traffic_path: Path | str) -> tuple[str, ...]:
    with h5py.File(traffic_path, "r") as source:
        names = source.attrs.get("channels")
        if names is None:
            raise ValueError("traffic cache does not declare its channel names")
        return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in names)


def load_multichannel_data(
    *, region_name: str, data_dir: Path | str, graph_dir: Path | str,
    batch_size: int = 16, seed: int = 42, preload: bool = True,
    auxiliary_channels: Sequence[str] = MULTICHANNEL_AUXILIARY_CHANNELS,
) -> MultiChannelData:
    """The unchanged research data, with loaders that also yield auxiliary history."""
    names = checked_auxiliary_channels(auxiliary_channels)
    base = load_research_data(region_name=region_name, data_dir=data_dir, graph_dir=graph_dir,
                              batch_size=batch_size, seed=seed, preload=preload)
    region = base.region
    if cache_channel_names(region.traffic_path) != CACHE_CHANNEL_NAMES:
        raise ValueError("traffic cache channels differ from flow, occupancy and speed")
    statistics = tuple(fit_channel_climatology(region, AUXILIARY_CHANNELS[name]) for name in names)
    loaders = {}
    for name, original in (("train", base.train), ("val", base.val)):
        loader = MultiChannelEventLoader(
            split=name, region_data=region, indices=original.indices, batch_size=original.bs,
            flow_mean=base.statistics.flow_mean, flow_std=base.statistics.flow_std,
            sensor_support=None, include_incident=False, shuffle=original.should_shuffle,
            drop_last=original.drop_last, seed=seed, auxiliary_channels=names,
        )
        if loader._rng.bit_generator.state != original._rng.bit_generator.state:
            raise RuntimeError("multichannel loader does not reproduce the research batch order")
        loaders[name] = loader
    identity = dict(base.identity)
    identity["cache_channel_names"] = list(CACHE_CHANNEL_NAMES)
    identity["auxiliary_channels"] = [
        {"name": name, "cache_channel_index": AUXILIARY_CHANNELS[name],
         "fit_range": [0, region.train_end],
         "fit_positions": "unique_raw_observed_training_timestamps",
         "weekly_climatology_sha256": array_digest(value.baseline),
         "node_residual_std_sha256": array_digest(value.residual_std_by_node),
         "channel_mean": value.flow_mean, "channel_std": value.flow_std,
         "residual_std": value.residual_std,
         "valid_training_points": value.valid_training_points}
        for name, value in zip(names, statistics)
    ]
    identity["auxiliary_missing_policy"] = "unobserved_points_enter_as_zero_residual"
    return MultiChannelData(region, base.statistics, loaders["train"], loaders["val"], identity,
                            names, statistics)


class MultiChannelNDR(STAEformerDeseason):
    """Original s0g whose backbone also sees occupancy and speed residual channels.

    The auxiliary residuals pass through a bias-free projection whose output is added to
    the output of the original flow input projection, through a forward hook on that
    layer. The projection is zero at construction, so the initial forecast equals the
    original s0g forecast. Gate features, anchors, both heads and the loss keep flow only.
    """

    def __init__(
        self,
        node_count: int,
        *,
        auxiliary_channels: Sequence[str],
        auxiliary_residual_std: Sequence[float],
    ) -> None:
        super().__init__(node_count, 12, 12, **dict(VARIANTS["s0g"]))
        names = checked_auxiliary_channels(auxiliary_channels)
        scales = torch.as_tensor(list(auxiliary_residual_std), dtype=torch.float32)
        if scales.shape != (len(names),) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("auxiliary residual scales must be finite, positive and one per channel")
        self.auxiliary_channels = names
        self.register_buffer("aux_sd", scales.clone())
        # Created after every original parameter and without the torch RNG, so a seeded
        # build draws exactly the original s0g initial weights.
        self.aux_input_proj = nn.Parameter(
            torch.zeros(self.backbone.input_embedding_dim, len(names))
        )
        self._aux_embedding: torch.Tensor | None = None
        self.backbone.input_proj.register_forward_hook(self._add_auxiliary_embedding)

    def _add_auxiliary_embedding(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor,
    ) -> torch.Tensor:
        if self._aux_embedding is None:
            raise RuntimeError("the multichannel backbone must be called through MultiChannelNDR.forward")
        if self._aux_embedding.shape != output.shape:
            raise ValueError("auxiliary embedding does not match the flow input embedding")
        return output + self._aux_embedding

    def auxiliary_signal(
        self, aux_hist: torch.Tensor, aux_baseline: torch.Tensor, aux_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standardized residuals; unobserved points enter as exactly zero."""
        if aux_mask.dtype != torch.bool and not torch.logical_or(aux_mask == 0, aux_mask == 1).all():
            raise ValueError("auxiliary observation mask must contain only zero or one")
        valid = aux_mask.bool()
        if not torch.isfinite(aux_baseline).all():
            raise FloatingPointError("auxiliary climatology must be finite")
        filled = torch.where(valid, aux_hist, aux_baseline)
        if not torch.isfinite(filled).all():
            raise FloatingPointError("observed auxiliary history must be finite")
        return (filled - aux_baseline) / self.aux_sd

    def forward(
        self,
        flow_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        time_feat: torch.Tensor,
        aux_hist: torch.Tensor,
        aux_baseline: torch.Tensor,
        aux_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (*flow_hist.shape, len(self.auxiliary_channels))
        if aux_hist.shape != expected or aux_baseline.shape != expected or aux_mask.shape != expected:
            raise ValueError("auxiliary history, climatology and mask must have shape (B, N, 12, C)")
        signal = self.auxiliary_signal(aux_hist, aux_baseline, aux_mask)
        self._aux_embedding = F.linear(signal.transpose(1, 2), self.aux_input_proj)
        try:
            return super().forward(flow_hist, x_baseline, y_baseline, time_feat)
        finally:
            self._aux_embedding = None


def make_multichannel_model(
    node_count: int,
    *,
    variant: str = MULTICHANNEL_VARIANT,
    residual_std: float = 1.0,
    residual_std_by_node: Any = None,
    flow_mean: float = 0.0,
    flow_std: float = 1.0,
    reliable_anchor: bool = False,
    max_reference_age: int = 2,
    hidden_dim: int = 16,
    auxiliary_channels: Sequence[str] = MULTICHANNEL_AUXILIARY_CHANNELS,
    auxiliary_residual_std: Sequence[float] | None = None,
    device: torch.device | str = "cpu",
) -> MultiChannelNDR:
    """Accept the research factory recipe; the flow statistics follow make_research_model."""
    if variant != MULTICHANNEL_VARIANT:
        raise ValueError("the multichannel factory builds only the multichannel variant")
    if type(node_count) is not int or node_count <= 0:
        raise ValueError("node count must be positive")
    if reliable_anchor or max_reference_age != 2 or hidden_dim != 16:
        raise ValueError("the multichannel variant keeps the original causal anchor and router")
    if auxiliary_residual_std is None:
        raise ValueError("multichannel NDR requires fitted auxiliary residual scales")
    model = MultiChannelNDR(node_count, auxiliary_channels=auxiliary_channels,
                            auxiliary_residual_std=auxiliary_residual_std)
    scales = torch.as_tensor(
        residual_std_by_node if residual_std_by_node is not None else torch.ones(node_count),
        dtype=torch.float32,
    )
    numbers = torch.tensor([residual_std, flow_mean, flow_std], dtype=torch.float64)
    if not torch.isfinite(numbers).all() or residual_std <= 0 or flow_std <= 0:
        raise ValueError("fitted flow statistics must be finite with positive scales")
    if scales.shape != (node_count,) or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise ValueError("fitted node residual scales must be positive with shape (N,)")
    with torch.no_grad():
        model.sd_res.fill_(residual_std)
        model.sd_node.copy_(scales)
        model.flow_mu.fill_(flow_mean)
        model.flow_sd.fill_(flow_std)
    model.research_spec = {
        "variant": MULTICHANNEL_VARIANT,
        "backbone": "original_STAEformer_s0g",
        "backbone_global_normalization": "unchanged",
        "dual_residual_heads": "unchanged",
        "teacher": "original_soft_distance",
        "history_policy": "original_causal_cache",
        "router": "original_horizon_monotone",
        "router_hidden_dim": 0,
        "reliable_anchor": False,
        "max_reference_age": None,
        "stale_reference_fallback": None,
        "parameter_node_independence": "router_only; original_backbone_retains_node_embeddings",
        "backbone_input": "flow_residual_plus_auxiliary_residual_embedding",
        "auxiliary_channels": list(model.auxiliary_channels),
        "auxiliary_projection": "bias_free_linear_zero_initialized_added_to_flow_input_projection",
        "auxiliary_standardization": "train_only_weekly_median_and_global_observed_residual_std",
        "auxiliary_missing_policy": "unobserved_points_enter_as_zero_residual",
        "gate_features": "flow_only_unchanged",
        "anchors_heads_and_loss": "raw_flow_units_unchanged",
        "initial_function": "identical_to_baseline_s0g",
    }
    return model.to(torch.device(device))


def multichannel_factory_kwargs(model: MultiChannelNDR) -> dict[str, Any]:
    """The research factory recipe plus the auxiliary channel names and scales."""
    if not isinstance(model, MultiChannelNDR) or model.research_spec["variant"] != MULTICHANNEL_VARIANT:
        raise ValueError("multichannel factory kwargs require a MultiChannelNDR")
    return research_factory_kwargs(model) | {
        "auxiliary_channels": list(model.auxiliary_channels),
        "auxiliary_residual_std": model.aux_sd.detach().cpu().tolist(),
    }
