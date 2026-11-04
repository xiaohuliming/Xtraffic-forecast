"""Frozen same-region event evaluation for real NDR, STAEformer and BiST.

Clean native baselines retain the original audited v3 causal cache inputs.
Added missingness rebuilds history using only retained original observations.
No fitting, checkpoint selection or target labels enter a forecast function.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from ndr_research.data import array_digest
from ndr_research.cohort import training_recipe_identity
from ndr_research.models import forward_research, make_research_model
from ndr_research.shared_graph import SharedGraphNDR, canonical_history
from ndr_research.strong_references import make_strong_reference
from ndr_research.transfer_data import (
    REGIONS,
    PROTOCOL,
    TransferPreprocessing,
    TransferRegionReader,
    TestEvaluationGate,
    _fingerprint,
)
from ndr_research.transfer_evaluation import (
    SCENARIOS,
    MaskedForecastSums,
    _StreamArrayHash,
    canonical_sha256,
    file_sha256,
    perturb_history,
    predict_history_only,
)
from scripts.protocol_v3_event_extension_data import fit_weekly_climatology
from scripts.train_ndr_research import require_snapshot
from scripts.train_ndr_strong_reference import (
    _require_model_identity,
    SCHEMA as STRONG_CHECKPOINT_SCHEMA,
)
from scripts.train_ndr_graph_indomain import (
    CHECKPOINT_SCHEMA as REGIONAL_GRAPH_SCHEMA,
    load_checkpoint as load_regional_graph_checkpoint,
)
from ndr_research.graph_indomain import model_config as regional_graph_config

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("ndr", "strong_reference", "shared_graph", "graph_indomain")


def read_selected_checkpoint(path: Path | str, *, kind: str) -> dict[str, Any]:
    path = Path(path)
    marker = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if file_sha256(path) != marker.get("sha256") or path.stat().st_size != marker.get(
        "size_bytes"
    ):
        raise ValueError("selected checkpoint sidecar/content mismatch")
    checkpoint = (
        load_regional_graph_checkpoint(path)
        if kind == "graph_indomain"
        else torch.load(path, map_location="cpu", weights_only=True)
    )
    schema = {
        "strong_reference": STRONG_CHECKPOINT_SCHEMA,
        "graph_indomain": REGIONAL_GRAPH_SCHEMA,
    }.get(kind, 1)
    if checkpoint.get("schema_version") != schema or not {
        "model",
        "config",
        "data_identity",
        "selection_metric",
    } <= set(checkpoint):
        raise ValueError("unsupported selected checkpoint schema")
    digest = canonical_sha256(checkpoint["config"])
    if digest != checkpoint.get("config_digest") or digest != marker.get(
        "config_digest"
    ):
        raise ValueError("selected checkpoint configuration digest mismatch")
    if checkpoint["selection_metric"] is None:
        raise ValueError("checkpoint has no recorded validation selection")
    if checkpoint["config"].get("data_identity") != checkpoint["data_identity"]:
        raise ValueError(
            "checkpoint's data identity differs from its frozen configuration"
        )
    return checkpoint


def inspect_selected_model(
    checkpoint_path: Path | str,
    verification_path: Path | str,
    summary_path: Path | str,
    *,
    kind: str,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    """Metadata-only admission; no cache reader or target collection is created."""
    if kind not in KINDS:
        raise ValueError("unsupported regional checkpoint kind")
    checkpoint = read_selected_checkpoint(checkpoint_path, kind=kind)
    config, selected = checkpoint["config"], checkpoint["selection_metric"]
    require_snapshot(config["source_snapshot"], source_root)
    if (
        config.get("max_train_batches") is not None
        or config.get("max_val_batches") is not None
    ):
        raise PermissionError(
            "limited-batch smoke checkpoints cannot enter final regional evaluation"
        )
    expected_scope = (
        "source_validation_development_only"
        if kind == "shared_graph"
        else "validation_development_only"
    )
    if config.get("claim_scope") != expected_scope:
        raise PermissionError(
            "checkpoint is not a complete formal validation development run"
        )
    seed = config.get("seed")
    if type(seed) is not int or seed not in (42, 1, 2):
        raise ValueError("regional checkpoint has no declared training seed")
    verified = json.loads(Path(verification_path).read_text())
    summary = json.loads(Path(summary_path).read_text())
    status = {
        "ndr": "verified_checkpoint_complete_validation",
        "strong_reference": "verified_complete_source_validation_development",
        "shared_graph": "verified_transfer_source_checkpoint_complete_validation",
        "graph_indomain": "verified_graph_indomain_checkpoint_complete_validation",
    }[kind]
    if (
        verified.get("status") != status
        or verified.get("checkpoint_sha256") != file_sha256(checkpoint_path)
        or verified.get("config_digest") != checkpoint["config_digest"]
        or verified.get("source_files_digest")
        != config["source_snapshot"]["files_digest"]
        or verified.get("test_evaluated") is not False
    ):
        raise PermissionError(
            "checkpoint does not have its exact complete independent validation verification"
        )
    if (
        summary.get("completed_epochs") != config["epochs"]
        or summary.get("test_evaluated") is not False
        or summary.get("config_digest") != checkpoint["config_digest"]
    ):
        raise PermissionError(
            "development training is not complete at its frozen epoch budget"
        )
    if kind == "shared_graph":
        if (
            checkpoint.get("model_class") != "SharedGraphNDR"
            or checkpoint.get("target_data_read") is not False
            or selected.get("selection_split") != "source_val"
            or selected.get("all_source_validation_samples") is not True
        ):
            raise PermissionError(
                "shared graph checkpoint did not use complete source-only selection"
            )
        observed = verified["source_validation"]
        for region, row in selected["regions"].items():
            actual = observed["regions"][region]
            _validate_validation_pair(row, actual)
        identity = checkpoint["source_statistics_identity"]
        if (
            identity.get("mode") != "source_only"
            or identity.get("target_statistics_used") is not False
            or verified.get("source_statistics_identity") != identity
            or verified.get("model_config_sha256")
            != canonical_sha256(checkpoint["model_config"])
            or verified.get("selection_metric_digest") != canonical_sha256(selected)
        ):
            raise PermissionError(
                "shared candidate verification does not bind source-only statistics or the selected model recipe"
            )
        bundle = identity["bundle_digest"]
        training_regions = tuple(checkpoint["source_regions"])
        architecture = dict(checkpoint["model_config"])
        family_name = f"shared_graph_h{architecture['hidden_dim']}"
    else:
        _validate_validation_pair(selected, verified["validation"])
        identity = checkpoint["data_identity"]
        if identity.get("training_analysis_labels") is not False or identity.get(
            "fit_range"
        ) != [0, identity["train_end"]]:
            raise PermissionError(
                "regional fit identity used analysis labels or a non-training prefix"
            )
        bundle = identity["cache_bundle_digest"]
        training_regions = (config["region"],)
        if kind == "graph_indomain":
            architecture = dict(checkpoint["model_config"])
            if (
                checkpoint.get("model_class") != "SharedGraphNDR"
                or architecture != regional_graph_config(config["hidden_dim"])
                or config.get("model_config") != architecture
            ):
                raise ValueError(
                    "regional graph checkpoint has an invalid model class or fixed recipe"
                )
            for field in ("normalizer_identity", "graph_identity"):
                if (
                    checkpoint[field] != config.get(field)
                    or verified.get(field) != checkpoint[field]
                ):
                    raise ValueError(
                        "regional graph verification does not bind its train statistics/static roads"
                    )
            normalizer = checkpoint["normalizer_identity"]
            if (
                normalizer["model_registered_node_statistics"] is not False
                or any(
                    normalizer[key] != identity[key]
                    for key in (
                        "region",
                        "region_cache_digest",
                        "fit_range",
                        "weekly_climatology_sha256",
                        "node_residual_std_sha256",
                        "valid_training_points",
                        "training_analysis_labels",
                    )
                )
                or checkpoint["graph_identity"]["region"] != config["region"]
            ):
                raise ValueError(
                    "regional graph fit/road identities differ from its audited regional data"
                )
            family_name = f"graph_indomain_h{architecture['hidden_dim']}"
        elif kind == "ndr":
            factory = checkpoint["model_factory"]
            architecture = {
                name: factory[name]
                for name in (
                    "variant",
                    "reliable_anchor",
                    "max_reference_age",
                    "hidden_dim",
                )
            }
            family_name = "ndr_" + factory["variant"]
        else:
            factory = checkpoint["model_factory"]
            architecture = {
                "method": factory["method"],
                "training_recipe": config["training_recipe"],
            }
            family_name = "raw_" + factory["method"]
    family = {
        "checkpoint_kind": kind,
        "model_family": family_name,
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "architecture": architecture,
        "training_recipe_identity": training_recipe_identity(config),
        "training_runtime": config["runtime"],
    }
    return {
        "checkpoint": checkpoint,
        "kind": kind,
        "training_seed": seed,
        "training_regions": training_regions,
        "bundle_digest": bundle,
        "model_family": family_name,
        "family_identity": family,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "verification_sha256": file_sha256(verification_path),
        "summary_sha256": file_sha256(summary_path),
        "config_digest": checkpoint["config_digest"],
        "validation_selection_sha256": canonical_sha256(selected),
        "data_identity_sha256": canonical_sha256(checkpoint["data_identity"]),
    }


def _validate_validation_pair(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    if (
        expected.get("selection", {}).get("all_validation_samples") is not True
        or observed.get("selection", {}).get("all_validation_samples") is not True
    ):
        raise PermissionError("validation collection is a subset")
    for key in (
        "valid_count",
        "selection",
        "observed_targets_sha256",
        "observation_masks_sha256",
    ):
        if expected[key] != observed[key]:
            raise ValueError(
                "complete validation collection differs from its selected checkpoint"
            )
    left = expected.get("global_masked_mae", expected.get("mae"))
    right = observed.get("global_masked_mae", observed.get("mae"))
    if not math.isclose(left, right, rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("independent validation did not reproduce the selected metric")


def admit_regional_gate(
    gate_path: Path | str, *, source_root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(gate_path).resolve()
    gate = json.loads(path.read_text())
    if (
        gate.get("schema_version") != 1
        or gate.get("sampling_protocol") != "event"
        or gate.get("protocol_version") != PROTOCOL
        or gate.get("split") != "test"
        or gate.get("structure_development_closed") is not True
        or gate.get("selection_frozen") is not True
        or gate.get("test_used_for_structure_selection") is not False
        or gate.get("region") not in REGIONS
        or not gate.get("authorization_id")
    ):
        raise PermissionError(
            "regional evaluation needs an explicit closed-development event-test gate"
        )
    if gate.get("scenarios") != list(SCENARIOS):
        raise ValueError("gate must predeclare the five complete missingness scenarios")
    if (
        gate.get("input_policy")
        != "native_clean_and_retained_observation_causal_corruption"
    ):
        raise ValueError(
            "gate does not freeze the original-clean/causal-corruption policy"
        )
    if (
        type(gate.get("perturbation_seed")) is not int
        or not 0 <= gate["perturbation_seed"] < 2**32
    ):
        raise ValueError("gate must freeze a uint32 perturbation seed")
    if gate.get("comparison_scope") not in (
        "matched_indomain_training",
        "declared_mixed_training_scope",
    ):
        raise ValueError("training supervision/data scope must be explicitly declared")
    if (
        type(gate.get("max_test_evaluations")) is not int
        or not 1 <= gate["max_test_evaluations"] <= 100
        or not isinstance(gate.get("evaluation_ledger"), str)
        or not gate["evaluation_ledger"].strip()
    ):
        raise ValueError(
            "regional gate needs a fixed shared ledger and predeclared test inference budget"
        )
    models = {}
    for declaration in gate.get("models", []):
        name = declaration.get("model_id", "")
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in models:
            raise ValueError("model identifiers must be unique safe lowercase names")
        paths = {}
        for key in ("checkpoint", "verification", "summary"):
            value = Path(declaration[key])
            paths[key] = value if value.is_absolute() else path.parent / value
            if file_sha256(paths[key]) != declaration.get(key + "_sha256"):
                raise ValueError(
                    "gate's checkpoint or development artifact hash mismatch"
                )
        admitted = inspect_selected_model(
            paths["checkpoint"],
            paths["verification"],
            paths["summary"],
            kind=declaration["checkpoint_kind"],
            source_root=source_root,
        )
        for key in (
            "validation_selection_sha256",
            "data_identity_sha256",
            "config_digest",
            "family_identity",
            "training_seed",
        ):
            if declaration.get(key) != admitted[key]:
                raise ValueError(f"gate does not bind {key} for {name}")
        if admitted["bundle_digest"] != gate["cache_bundle_digest"]:
            raise ValueError("regional comparison mixes cache bundles")
        if gate["comparison_scope"] == "matched_indomain_training" and admitted[
            "training_regions"
        ] != (gate["region"],):
            raise PermissionError(
                "extra-source or zero-shot model cannot be called matched regional training"
            )
        if admitted["kind"] != "shared_graph" and admitted["training_regions"] != (
            gate["region"],
        ):
            raise ValueError(
                "regional node-identity baseline belongs to another region"
            )
        admitted.update(model_id=name, paths=paths)
        models[name] = admitted
    if len(models) < 2:
        raise ValueError(
            "frozen regional evaluation needs at least two real checkpoint models"
        )
    gate["gate_sha256"] = file_sha256(path)
    return gate, models


@dataclass
class RegionalTrainingPreprocessing:
    statistics: Any
    identity: Mapping[str, Any]
    mode: str = "indomain_train_fitted"

    def validate_reader(self, reader: TransferRegionReader) -> None:
        if (
            self.identity["region"] != reader.region
            or self.identity["region_cache_digest"] != reader.identity["region_digest"]
        ):
            raise ValueError("regional train statistics do not match the exact reader")

    def lookup(self, dow: np.ndarray, tod: np.ndarray, nodes: int) -> np.ndarray:
        if nodes != self.statistics.baseline.shape[-1]:
            raise ValueError("regional climatology node count differs")
        return np.array(self.statistics.baseline[dow, tod], copy=True)

    def scales(self, nodes: int) -> np.ndarray:
        if nodes != len(self.statistics.residual_std_by_node):
            raise ValueError("regional scales node count differs")
        return np.array(self.statistics.residual_std_by_node, copy=True)


def fit_regional_training_preprocessing(
    reader: TransferRegionReader,
) -> RegionalTrainingPreprocessing:
    """Reproduce original NDR statistics using only whitelisted raw train values."""
    if reader.role != "source":
        raise PermissionError(
            "regional fitting must use an explicitly supervised training region"
        )
    flow = np.empty((reader.train_end, reader.N, 1), dtype=np.float32)
    mask = np.empty(flow.shape, dtype=bool)
    for start, stop, values, observed in reader.iter_source_training():
        flow[start:stop, :, 0], mask[start:stop, :, 0] = values, observed
    view = SimpleNamespace(
        train_end=reader.train_end,
        N=reader.N,
        flow_series=flow,
        flow_mask=mask,
        dow=reader.dow,
        tod=reader.tod,
    )
    statistics = fit_weekly_climatology(view)
    identity = {
        "mode": "indomain_train_fitted",
        "region": reader.region,
        "cache_bundle_digest": reader.contract.bundle_digest,
        "region_cache_digest": reader.identity["region_digest"],
        "fit_range": [0, reader.train_end],
        "fit_positions": "unique_raw_observed_training_timestamps",
        "weekly_climatology_sha256": array_digest(statistics.baseline),
        "node_residual_std_sha256": array_digest(statistics.residual_std_by_node),
        "flow_mean": statistics.flow_mean,
        "flow_std": statistics.flow_std,
        "residual_std": statistics.residual_std,
        "valid_training_points": statistics.valid_training_points,
        "test_statistics_used": False,
    }
    identity["identity_digest"] = canonical_sha256(identity)
    return RegionalTrainingPreprocessing(statistics, identity)


def require_regional_fit(
    admitted: Mapping[str, Any], preprocessing: RegionalTrainingPreprocessing
) -> None:
    if admitted["kind"] == "shared_graph":
        return
    recorded = admitted["checkpoint"]["data_identity"]
    for key in (
        "region",
        "region_cache_digest",
        "fit_range",
        "weekly_climatology_sha256",
        "node_residual_std_sha256",
        "valid_training_points",
    ):
        if recorded[key] != preprocessing.identity[key]:
            raise ValueError(
                f"regional train-only preprocessing differs from checkpoint: {key}"
            )
    for key in ("flow_mean", "flow_std", "residual_std"):
        if not math.isclose(
            recorded[key], preprocessing.identity[key], rel_tol=2e-7, abs_tol=2e-6
        ):
            raise ValueError("regional training normalization did not reproduce")
    if admitted["kind"] == "graph_indomain":
        # The reader's graph is also checked immediately before inference.
        if recorded["region"] != admitted["checkpoint"]["graph_identity"]["region"]:
            raise ValueError("regional graph identity belongs to another county")


class RegionalForecastAdapter(torch.nn.Module):
    """Seven-argument label-free bridge; native clean and corrupted inputs differ."""

    def __init__(self, model: torch.nn.Module, *, kind: str) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError("unsupported regional forecast adapter")
        self.model, self.kind, self.scenario = model, kind, "clean"

    def forward(
        self,
        flow_hist,
        x_baseline,
        y_baseline,
        time_feat,
        history_mask,
        edge_index,
        node_scale,
    ):
        if self.kind in ("shared_graph", "graph_indomain"):
            return self.model(
                flow_hist,
                x_baseline,
                y_baseline,
                time_feat,
                history_mask,
                edge_index,
                node_scale,
            )
        history = (
            flow_hist
            if self.scenario == "clean"
            else canonical_history(flow_hist, x_baseline, history_mask).filled
        )
        if self.kind == "strong_reference":
            return self.model(history, time_feat)
        return forward_research(
            self.model, history, x_baseline, y_baseline, time_feat, history_mask
        )


def rebuild_regional_model(
    admitted: Mapping[str, Any],
    device: torch.device | str,
    *,
    bist_checkout: Path | None = None,
) -> RegionalForecastAdapter:
    checkpoint, kind = admitted["checkpoint"], admitted["kind"]
    if kind == "ndr":
        model = make_research_model(
            **(checkpoint["model_factory"] | {"device": str(device)})
        )
    elif kind == "strong_reference":
        factory = dict(checkpoint["model_factory"])
        if factory["method"] == "bist" and bist_checkout is not None:
            factory["bist_checkout"] = bist_checkout
        model = make_strong_reference(**(factory | {"device": str(device)}))
        _require_model_identity(model, checkpoint)
    else:
        model = SharedGraphNDR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    adapter = RegionalForecastAdapter(model, kind=kind).to(device).eval()
    if kind == "graph_indomain":
        adapter.graph_identity = copy.deepcopy(checkpoint["graph_identity"])
    return adapter


def read_native_clean_history(
    reader: TransferRegionReader, indices: np.ndarray
) -> np.ndarray:
    """Whitelisted original imputed history only, with stop exactly at origin+1."""
    path = reader.paths["traffic"]
    if _fingerprint(path) != reader._fingerprints["traffic"]:
        raise RuntimeError("native history cache changed since its formal verification")
    before = path.stat()
    rows = []
    with h5py.File(path, "r") as handle:
        for origin in reader.sample_start[indices]:
            start, stop = int(origin) - reader.T_h + 1, int(origin) + 1
            if start < 0 or stop > reader.T or stop - start != reader.T_h:
                raise ValueError(
                    "native clean history is outside the audited causal window"
                )
            value = reader._dataset(
                handle,
                "traffic",
                "flow_series_imputed",
                (slice(start, stop), slice(None), 0),
                "native_clean_history",
            )
            rows.append(np.asarray(value, dtype=np.float32).T)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ) or _fingerprint(path) != reader._fingerprints["traffic"]:
        raise RuntimeError("native history cache changed during its read")
    values = np.stack(rows)
    if not np.isfinite(values).all():
        raise FloatingPointError("original v3 native imputed history is not finite")
    return values


def _model_batch(
    batch: Mapping[str, np.ndarray], reader: TransferRegionReader, preprocessing: Any
) -> dict[str, np.ndarray]:
    result = dict(batch)
    history = batch["sample_start"][:, None] + np.arange(-reader.T_h + 1, 1)
    future = batch["sample_start"][:, None] + np.arange(1, reader.T_p + 1)
    result["x_baseline"] = np.stack(
        [
            preprocessing.lookup(reader.dow[row], reader.tod[row], reader.N).T
            for row in history
        ]
    )
    result["y_baseline"] = np.stack(
        [
            preprocessing.lookup(reader.dow[row], reader.tod[row], reader.N).T
            for row in future
        ]
    )
    result["node_scale"] = preprocessing.scales(reader.N)
    return result


def evaluate_regional_models(
    models: Mapping[str, RegionalForecastAdapter],
    *,
    reader: TransferRegionReader,
    preprocessing: RegionalTrainingPreprocessing,
    model_preprocessing: Mapping[str, Any],
    evaluation_grant: TestEvaluationGate,
    batch_size: int = 8,
    perturbation_seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """One full event-test target collection, shared by all frozen checkpoint models."""
    if set(models) != set(model_preprocessing) or len(models) < 2:
        raise ValueError("regional models must have explicit matching preprocessing")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("regional evaluation batch size must be positive")
    indices = reader.indices("test")
    for adapter in models.values():
        if adapter.kind == "graph_indomain":
            roads = adapter.graph_identity
            if (
                roads["region"] != reader.region
                or roads["nodes"] != reader.N
                or roads["edge_index_sha256"]
                != array_digest(reader.edge_index.astype(np.int64))
            ):
                raise ValueError(
                    "regional graph forecast road identity differs from its training checkpoint"
                )
    metrics = {
        name: {scenario: MaskedForecastSums() for scenario in SCENARIOS}
        for name in models
    }
    blocks = {name: {scenario: {} for scenario in SCENARIOS} for name in models}
    durations = {name: {scenario: 0.0 for scenario in SCENARIOS} for name in models}
    removed = {scenario: 0 for scenario in SCENARIOS}
    streams, original_observed = {}, 0
    started = time.perf_counter()
    for low in range(0, len(indices), batch_size):
        selected = indices[low : low + batch_size]
        batch = reader.get_batch(
            selected, preprocessing, split="test", evaluation_gate=evaluation_grant
        )
        native = read_native_clean_history(reader, selected)
        original_observed += int(batch["history_mask"].sum())
        target, mask = batch["target"].copy(), batch["target_mask"].copy()
        days = (
            reader.times_ns[batch["sample_start"]]
            .astype("datetime64[ns]")
            .astype("datetime64[D]")
            .astype(str)
        )
        for field in ("sample_start", "target", "target_mask"):
            streams.setdefault(
                field, _StreamArrayHash(len(indices), batch[field])
            ).update(batch[field])
        for scenario in SCENARIOS:
            corrupted = perturb_history(batch, scenario, seed=perturbation_seed)
            removed[scenario] += int(
                batch["history_mask"].sum() - corrupted["history_mask"].sum()
            )
            for name, adapter in models.items():
                current = _model_batch(corrupted, reader, model_preprocessing[name])
                adapter.scenario = scenario
                if scenario == "clean" and adapter.kind in ("ndr", "strong_reference"):
                    current["flow_hist"] = native.copy()
                tick = time.perf_counter()
                prediction = predict_history_only(adapter, current, device)
                durations[name][scenario] += time.perf_counter() - tick
                if not np.array_equal(batch["target"], target) or not np.array_equal(
                    batch["target_mask"], mask
                ):
                    raise RuntimeError("regional forecast mutated target observations")
                metrics[name][scenario].update(prediction, target, mask)
                for day in np.unique(days):
                    keep = days == day
                    blocks[name][scenario].setdefault(
                        str(day), MaskedForecastSums()
                    ).update(prediction[keep], target[keep], mask[keep])
    collection = {
        "samples": len(indices),
        "forecast_origins_sha256": streams["sample_start"].hexdigest(),
        "targets_sha256": streams["target"].hexdigest(),
        "target_masks_sha256": streams["target_mask"].hexdigest(),
        "all_declared_split_samples": True,
        "day_basis": "audited cache times_ns at forecast origin",
        "hash_format": "sha256(dtype_string+json(full_shape)+C_order_bytes)",
    }
    reports = {}
    for name, adapter in models.items():
        scenarios = {}
        for scenario in SCENARIOS:
            scenarios[scenario] = {
                **metrics[name][scenario].result(),
                "collection": copy.deepcopy(collection),
                "day_blocks": {
                    day: value.totals()
                    for day, value in sorted(blocks[name][scenario].items())
                },
                "forward_seconds": durations[name][scenario],
                "perturbation_seed": perturbation_seed,
                "history_observed_points_original": original_observed,
                "history_observed_points_removed": removed[scenario],
            }
        reports[name] = {
            "collection": copy.deepcopy(collection),
            "scenarios": scenarios,
            "input_policy": {
                "clean": "original_v3_causal_cache"
                if adapter.kind in ("ndr", "strong_reference")
                else "frozen_candidate_own_observation_sanitization",
                "corrupted": "retained_raw_observations_then_in_window_causal_fill_then_frozen_train_climatology",
            },
            "target_used_for_forward": False,
        }
    return {
        "collection": collection,
        "models": reports,
        "wallclock_seconds": time.perf_counter() - started,
        "target_windows_read_once_for_all_models": True,
        "original_native_history_extra_read_is_history_only": True,
    }


def model_preprocessing_from_admission(
    admitted: Mapping[str, Any], regional: RegionalTrainingPreprocessing
) -> Any:
    if admitted["kind"] != "shared_graph":
        require_regional_fit(admitted, regional)
        return regional
    checkpoint = admitted["checkpoint"]
    source = TransferPreprocessing.from_state(checkpoint["source_statistics"])
    if source.identity != checkpoint["source_statistics_identity"]:
        raise ValueError(
            "shared candidate numerical preprocessing differs from its checkpoint"
        )
    return source


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class RegionalEvaluationLedger:
    """Fixed region/model-family budget, independent of gate paths and seeds.

    Each attempt costs one slot per distinct training seed in its checkpoint
    cohort. Three seeds plus one independent repeat require six slots. A failed
    reserved attempt consumes its slots. Identifiers/auth receipts never define
    a new family; one shared ledger must be retained for the complete study.
    """

    def __init__(
        self,
        gate: Mapping[str, Any],
        admitted: Mapping[str, Any],
        gate_path: Path | str,
        output: Path | str,
    ) -> None:
        import uuid

        declared = Path(gate["evaluation_ledger"])
        self.path = (
            declared
            if declared.is_absolute()
            else Path(gate_path).resolve().parent / declared
        )
        self.gate, self.output = gate, str(Path(output).resolve())
        families = sorted(
            {
                json.dumps(
                    spec["family_identity"], sort_keys=True, separators=(",", ":")
                )
                for spec in admitted.values()
            }
        )
        self.family_payload = {
            "region": gate["region"],
            "model_families": [json.loads(value) for value in families],
        }
        self.family_sha = canonical_sha256(self.family_payload)
        self.seeds = sorted({spec["training_seed"] for spec in admitted.values()})
        self.checkpoints = {
            name: spec["checkpoint_sha256"] for name, spec in admitted.items()
        }
        self.attempt_id = uuid.uuid4().hex

    def _append(self, status: str, details: Mapping[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            rows = (
                [
                    json.loads(line)
                    for line in self.path.read_text().splitlines()
                    if line.strip()
                ]
                if self.path.exists()
                else []
            )
            previous = "0" * 64
            for row in rows:
                payload = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if (
                    payload["previous_sha256"] != previous
                    or canonical_sha256(payload) != row["record_sha256"]
                ):
                    raise ValueError("regional evaluation ledger hash chain changed")
                previous = row["record_sha256"]
            family = [
                row
                for row in rows
                if row["status"] == "reserved"
                and row["family_sha256"] == self.family_sha
            ]
            budget = self.gate["max_test_evaluations"]
            if family and any(
                row["family_max_test_evaluations"] != budget for row in family
            ):
                raise PermissionError(
                    "regional model-family test budget cannot change after first access"
                )
            consumed = sum(row["training_seed_inference_slots"] for row in family)
            cost = len(self.seeds) if status == "reserved" else 0
            if status == "reserved" and consumed + cost > budget:
                raise PermissionError(
                    "regional model-family test inference budget is exhausted"
                )
            if status != "reserved" and not any(
                row["attempt_id"] == self.attempt_id and row["status"] == "reserved"
                for row in rows
            ):
                raise ValueError(
                    "regional attempt was not reserved before reading test"
                )
            row = {
                "status": status,
                "attempt_id": self.attempt_id,
                "unix_time_ns": time.time_ns(),
                "previous_sha256": previous,
                "family_sha256": self.family_sha,
                "family": self.family_payload,
                "family_max_test_evaluations": budget,
                "training_seed_inference_slots": cost,
                "consumed_slots_after": consumed + cost,
                "training_seeds": self.seeds,
                "checkpoint_cohort": self.checkpoints,
                "gate_sha256": self.gate["gate_sha256"],
                "authorization_id": self.gate["authorization_id"],
                "output": self.output,
                "details": dict(details),
            }
            row["record_sha256"] = canonical_sha256(row)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return row
        finally:
            lock.unlink(missing_ok=True)

    def reserve(self) -> dict[str, Any]:
        return self._append(
            "reserved",
            {"unit": "distinct_training_seed_inference_slots_per_cohort_access"},
        )

    def finish(self, status: str, **details: Any) -> dict[str, Any]:
        return self._append(status, details)
