"""Gate-bound same-region full-window temporal holdout evaluation."""

from __future__ import annotations

import copy
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from ndr_research.cohort import training_recipe_identity
from ndr_research.fullwindow_data import (
    FullWindowRegionReader,
    FullWindowStatistics,
    SPLIT_CODES,
    legacy_array_digest,
)
from ndr_research.fullwindow_training import rebuild_model
from ndr_research.indomain_evaluation import RegionalEvaluationLedger, state_digest
from ndr_research.models import forward_research, research_factory_kwargs
from ndr_research.shared_graph import canonical_history
from ndr_research.strong_references import make_strong_reference
from ndr_research.transfer_data import PROTOCOL, _fingerprint, _json_hash
from ndr_research.transfer_evaluation import (
    BATCH_FIELDS,
    FORWARD_FIELDS,
    SCENARIOS,
    MaskedForecastSums,
    _StreamArrayHash,
    canonical_sha256,
    file_sha256,
    perturb_history,
    predict_history_only,
)
from scripts.train_ndr_research import require_snapshot

ROOT = Path(__file__).resolve().parents[1]
GATE_SCHEMA = "fullwindow_frozen_eval_v1"
REPORT_STATUS = "frozen_fullwindow_evaluation_complete"
KINDS = ("fullwindow_ndr", "fullwindow_strong")
SEEDS = (42, 1, 2)
FAMILY_SCOPE = "same_region_fullwindow_model_families_across_seeds_checkpoints_identifiers_and_authorizations"


def rebuild_fullwindow_model(spec: Mapping[str, Any], device="cpu", bist_checkout=None):
    checkpoint = spec["checkpoint"]
    if spec["kind"] == "fullwindow_ndr":
        model = rebuild_model(
            checkpoint["config"]["method"], checkpoint["model_factory"], device
        )
    else:
        factory = dict(checkpoint["model_factory"])
        if bist_checkout is not None and factory["method"] == "bist":
            factory["bist_checkout"] = bist_checkout
        model = make_strong_reference(**factory, device=device)
        from scripts.train_strong_fullwindow import _require_model_identity

        _require_model_identity(model, checkpoint)
    if type(model).__name__ != checkpoint["model_class"]:
        raise ValueError(
            "full-window checkpoint class differs from its actual constructed model"
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def inspect_fullwindow_model(
    checkpoint_path,
    verification_path,
    summary_path,
    *,
    kind,
    source_root=ROOT,
    bist_checkout=None,
):
    if kind == "fullwindow_ndr":
        from scripts.train_ndr_fullwindow import load_checkpoint

        schema, scope, status = (
            "fullwindow_ndr_checkpoint_v1",
            "fullwindow_validation_development_only",
            "verified_fullwindow_ndr_checkpoint_complete_validation",
        )
        summary_status = (
            "training_completed_validation_pending_independent_verification"
        )
    elif kind == "fullwindow_strong":
        from scripts.train_strong_fullwindow import load_checkpoint

        schema, scope, status = (
            "fullwindow_strong_checkpoint_v1",
            "fullwindow_validation_development_capped",
            "verified_fullwindow_strong_complete_validation_capped",
        )
        summary_status = "fullwindow_strong_training_completed_validation_pending_independent_verification"
    else:
        raise ValueError("unsupported independent full-window checkpoint kind")
    checkpoint = load_checkpoint(Path(checkpoint_path))
    config, reader, sampling = (
        checkpoint["config"],
        checkpoint["reader_identity"],
        checkpoint["sampling_identity"],
    )
    statistics = FullWindowStatistics.from_state(checkpoint["statistics_state"])
    if (
        checkpoint["schema_version"] != schema
        or config.get("schema_version") != schema
        or config.get("model_class") != checkpoint["model_class"]
        or config.get("claim_scope") != scope
        or config.get("sampling_protocol") != "full_window"
        or config.get("max_train_batches") is not None
        or config.get("max_val_batches") is not None
        or config.get("test_evaluated") is not False
        or config.get("seed") not in SEEDS
        or config.get("epochs") not in (30, 100)
    ):
        raise PermissionError(
            "full-window final evaluation requires complete independent full-window development"
        )
    require_snapshot(config["source_snapshot"], source_root)
    required_sources = {
        "ndr_research/fullwindow_data.py",
        "ndr_research/fullwindow_training.py",
    }
    required_sources.add(
        "scripts/train_ndr_fullwindow.py"
        if kind == "fullwindow_ndr"
        else "scripts/train_strong_fullwindow.py"
    )
    if not required_sources.issubset(config["source_snapshot"]["files"]):
        raise ValueError("full-window checkpoint omits its actual training/data source")
    if (
        reader.get("sampling_protocol") != "full_window"
        or reader.get("role") != "source"
        or reader.get("history_policy") != "canonical_causal_imputation"
        or reader.get("sampling") != sampling
        or reader.get("test_numeric_api_available") is not False
        or reader.get("incident_or_affected_values_read") is not False
        or dict(statistics.identity) != checkpoint["statistics_identity"]
        or statistics.identity["reader_identity_digest"] != reader["identity_digest"]
        or statistics.identity["sampling_digest"] != sampling["sampling_digest"]
        or statistics.identity["fit_start"] != 0
        or statistics.identity["fit_end"] != sampling["train_end"]
        or statistics.identity.get("scalar_reduction_dtype") != "float64"
        or sampling.get("event_sample_identity_used") is not False
        or sampling.get("target_disjoint_between_splits") is not True
    ):
        raise PermissionError(
            "full-window statistics and raw sampling are not bound to the saved training reader"
        )
    identity = checkpoint["data_identity"]
    for field, expected in (
        ("reader_identity", reader),
        ("sampling_identity", sampling),
        ("statistics_identity", dict(statistics.identity)),
    ):
        if (
            identity.get(field) != expected
            or config.get("data_identity", {}).get(field) != expected
        ):
            raise ValueError("full-window checkpoint/data/config identities disagree")
    selected = checkpoint["selection_metric"]
    collection, selection = (
        selected.get("collection", {}),
        selected.get("selection", {}),
    )
    if (
        selection.get("all_validation_samples") is not True
        or selection.get("samples") != sampling["counts"]["val"]
        or selection.get("sampling_protocol") != "full_window"
        or selection.get("sampling_digest") != sampling["sampling_digest"]
        or selection.get("reader_identity_digest") != reader["identity_digest"]
        or selection.get("statistics_identity_digest")
        != statistics.identity["identity_digest"]
        or collection.get("all_declared_split_samples") is not True
        or collection.get("sampling_protocol") != "full_window"
    ):
        raise PermissionError(
            "full-window checkpoint was not selected on its complete full-window validation"
        )
    verified, summary = (
        json.loads(Path(verification_path).read_text()),
        json.loads(Path(summary_path).read_text()),
    )
    checkpoint_sha = file_sha256(checkpoint_path)
    bindings = {
        "schema_version": 1 if kind == "fullwindow_ndr" else schema,
        "status": status,
        "claim_scope": scope,
        "checkpoint_sha256": checkpoint_sha,
        "config_digest": checkpoint["config_digest"],
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "model_class": checkpoint["model_class"],
        "reader_identity": reader,
        "sampling_identity": sampling,
        "statistics_identity": dict(statistics.identity),
        "sampling_protocol": "full_window",
        "test_evaluated": False,
        "training_process_id": config["training_process_id"],
    }
    for field, value in bindings.items():
        if verified.get(field) != value:
            raise PermissionError(
                f"independent full-window verification does not bind {field}"
            )
    if kind == "fullwindow_ndr" and (
        verified.get("model_factory") != checkpoint["model_factory"]
        or verified.get("data_identity") != identity
    ):
        raise ValueError(
            "NDR full-window verification does not bind factory and data identities"
        )
    if kind == "fullwindow_strong" and (
        verified.get("data_identity_digest") != identity["identity_digest"]
        or verified.get("source_snapshot_files_digest")
        != config["source_snapshot"]["files_digest"]
    ):
        raise ValueError(
            "strong full-window verification does not bind data and source identities"
        )
    if (
        type(verified.get("verification_process_id")) is not int
        or verified["verification_process_id"] <= 0
        or verified["verification_process_id"] == config["training_process_id"]
    ):
        raise PermissionError(
            "full-window verification did not run in an independent process"
        )
    observed = verified.get("validation", {})
    for field in (
        "valid_count",
        "selection",
        "collection",
        "observed_targets_sha256",
        "observation_masks_sha256",
        "horizon_valid_counts",
    ):
        if selected.get(field) != observed.get(field):
            raise ValueError(
                "independent full-window validation changed the selected observations"
            )
    if not math.isclose(
        selected["global_masked_mae"],
        observed.get("global_masked_mae", float("inf")),
        rel_tol=2e-6,
        abs_tol=2e-5,
    ):
        raise ValueError("independent full-window validation MAE differs")
    if (
        summary.get("status") != summary_status
        or summary.get("completed_epochs") != config["epochs"]
        or summary.get("best_epoch") != checkpoint["progress"]["best_epoch"]
        or summary.get("config_digest") != checkpoint["config_digest"]
        or summary.get("test_evaluated") is not False
        or canonical_sha256(summary.get("best_validation"))
        != canonical_sha256(selected)
    ):
        raise PermissionError(
            "full-window training budget is incomplete or its selected checkpoint differs"
        )
    if kind == "fullwindow_strong" and verified.get("summary_sha256") != file_sha256(
        summary_path
    ):
        raise ValueError(
            "strong full-window verification does not bind the final summary"
        )
    method = config["method"]
    factory = copy.deepcopy(checkpoint["model_factory"])
    architecture = {
        key: value for key, value in factory.items() if key != "bist_checkout"
    }
    recipe = training_recipe_identity(config)
    for field in ("model_upstream", "warmup_batches", "timed_batches"):
        recipe.pop(field, None)
    family = {
        "checkpoint_kind": kind,
        "model_class": checkpoint["model_class"],
        "method": method,
        "sampling_protocol": "full_window",
        "architecture": architecture,
        "training_recipe_identity": recipe,
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "training_runtime": config["runtime"],
        "sampling_digest": sampling["sampling_digest"],
        "statistics_identity_digest": statistics.identity["identity_digest"],
    }
    spec = {
        "checkpoint": checkpoint,
        "kind": kind,
        "checkpoint_kind": kind,
        "method": method,
        "model_class": checkpoint["model_class"],
        "native_model_class": checkpoint.get("native_model_class"),
        "model_family": method
        + (f"_h{factory['hidden_dim']}" if method == "shared_graph" else ""),
        "training_seed": config["seed"],
        "training_regions": [reader["region"]],
        "training_scope": "same_region_full_window_supervised_temporal_split",
        "sampling_protocol": "full_window",
        "family_identity": family,
        "training_recipe": recipe,
        "training_runtime": config["runtime"],
        "source_snapshot": config["source_snapshot"],
        "reader_identity": reader,
        "sampling_identity": sampling,
        "statistics_identity": dict(statistics.identity),
        "data_identity": identity,
        "checkpoint_sha256": checkpoint_sha,
        "verification_sha256": file_sha256(verification_path),
        "summary_sha256": file_sha256(summary_path),
        "config_digest": checkpoint["config_digest"],
        "validation_selection_sha256": canonical_sha256(selected),
        "normalizer_scalar_dtype": "float64",
        "legacy27_scalar_dtype": "float32",
        "native_convergence_reproduced": False,
    }
    model = rebuild_fullwindow_model(spec, bist_checkout=bist_checkout)
    if kind == "fullwindow_strong":
        if (
            checkpoint["model_factory"]["method"] != method
            or checkpoint["model_factory"]["node_count"]
            != statistics.baseline.shape[-1]
        ):
            raise ValueError("strong full-window factory method or node scope differs")
        for field in ("flow_mean", "flow_std"):
            if checkpoint["model_factory"][field] != getattr(statistics, field):
                raise ValueError(
                    "strong full-window factory does not use its exact saved train statistics"
                )
        if model.native.__class__.__name__ != checkpoint.get("native_model_class"):
            raise ValueError("strong checkpoint has a false native model class")
        for field in ("flow_mean", "flow_std"):
            if float(getattr(model, field)) != float(
                np.float32(getattr(statistics, field))
            ):
                raise ValueError(
                    "strong registered scalar differs from saved train-only statistics"
                )
    elif method != "shared_graph":
        for buffer, field in (
            ("flow_mu", "flow_mean"),
            ("flow_sd", "flow_std"),
            ("sd_res", "residual_std"),
        ):
            if float(getattr(model, buffer)) != float(
                np.float32(getattr(statistics, field))
            ):
                raise ValueError(
                    "NDR registered scalar differs from saved train-only statistics"
                )
        if not np.array_equal(
            model.sd_node.detach().numpy(), statistics.residual_std_by_node
        ):
            raise ValueError(
                "NDR registered node scales differ from saved train-only statistics"
            )
    spec["model_state_sha256"] = state_digest(model)
    spec["parameter_count"] = sum(value.numel() for value in model.parameters())
    return spec


def admit_fullwindow_gate(gate_path, *, source_root=ROOT, bist_checkout=None):
    path = Path(gate_path).resolve()
    gate = json.loads(path.read_text())
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or gate.get("sampling_protocol") != "full_window"
        or gate.get("protocol_version") != PROTOCOL
        or gate.get("split") != "test"
        or gate.get("evaluation_role") != "same_region_temporal_heldout"
        or gate.get("structure_development_closed") is not True
        or gate.get("selection_frozen") is not True
        or gate.get("test_used_for_structure_selection") is not False
        or not gate.get("authorization_id")
        or gate.get("training_seeds") != list(SEEDS)
        or gate.get("max_test_evaluations") != 6
        or not gate.get("evaluation_ledger")
        or gate.get("scenarios") != list(SCENARIOS)
    ):
        raise PermissionError(
            "full-window evaluation requires a closed three-seed frozen test gate and six slots"
        )
    if (
        type(gate.get("perturbation_seed")) is not int
        or not 0 <= gate["perturbation_seed"] < 2**32
    ):
        raise ValueError("full-window gate needs a frozen uint32 perturbation seed")
    admitted = {}
    for row in gate.get("models", []):
        name = row.get("model_id", "")
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in admitted:
            raise ValueError("full-window model identifiers must be safe and unique")
        paths = {
            field: Path(row[field])
            for field in ("checkpoint", "verification", "summary")
        }
        paths = {
            field: value if value.is_absolute() else path.parent / value
            for field, value in paths.items()
        }
        if any(
            file_sha256(value) != row.get(field + "_sha256")
            for field, value in paths.items()
        ):
            raise ValueError("frozen full-window development artifacts changed")
        spec = inspect_fullwindow_model(
            paths["checkpoint"],
            paths["verification"],
            paths["summary"],
            kind=row["checkpoint_kind"],
            source_root=source_root,
            bist_checkout=bist_checkout,
        )
        for field in (
            "family_identity",
            "training_seed",
            "model_class",
            "config_digest",
            "validation_selection_sha256",
            "model_state_sha256",
        ):
            if row.get(field) != spec[field]:
                raise ValueError(f"full-window gate does not bind {field}")
        if spec["training_regions"] != [gate["region"]]:
            raise PermissionError(
                "full-window temporal holdout belongs to a different trained region"
            )
        spec.update(model_id=name, paths=paths)
        admitted[name] = spec
    families = {}
    for spec in admitted.values():
        families.setdefault(canonical_sha256(spec["family_identity"]), []).append(spec)
    if len(families) < 2 or any(
        len(rows) != 3 or {row["training_seed"] for row in rows} != set(SEEDS)
        for rows in families.values()
    ):
        raise ValueError(
            "each of at least two full-window model families needs exactly three frozen seeds"
        )
    for field in ("reader_identity", "sampling_identity", "statistics_identity"):
        if (
            len({canonical_sha256(spec[field]) for spec in admitted.values()}) != 1
            or gate.get(field) != next(iter(admitted.values()))[field]
        ):
            raise ValueError(
                "full-window model cohort mixes raw readers, sampling or saved training statistics"
            )
    gate["gate_sha256"] = file_sha256(path)
    return gate, admitted


@dataclass(frozen=True)
class FullWindowFrozenGrant:
    authorization_id: str
    gate_sha256: str
    checkpoint_cohort_sha256: str
    region: str
    source_reader_identity_digest: str
    sampling_digest: str
    statistics_identity_digest: str

    def __post_init__(self):
        if not self.authorization_id:
            raise PermissionError(
                "full-window numerical test access needs explicit frozen authorization"
            )
        for field in (
            "gate_sha256",
            "checkpoint_cohort_sha256",
            "source_reader_identity_digest",
            "sampling_digest",
            "statistics_identity_digest",
        ):
            if not re.fullmatch(r"[a-f0-9]{64}", getattr(self, field)):
                raise ValueError("full-window grant requires exact SHA-256 identities")


class FullWindowFrozenReader:
    """Separate numerical test reader; the immutable development API stays closed."""

    role = "same_region_temporal_heldout"

    def __init__(
        self,
        region,
        data_dir,
        graph_dir,
        *,
        grant,
        statistics,
        source_reader_identity,
        sampling_identity,
        contract=None,
        bundle_manifest=None,
    ):
        if not isinstance(grant, FullWindowFrozenGrant):
            raise PermissionError(
                "full-window test reader requires an admitted frozen grant"
            )
        self._grant, self._statistics = grant, statistics
        self.metadata = FullWindowRegionReader(
            region,
            data_dir,
            graph_dir,
            role="target",
            contract=contract,
            bundle_manifest=bundle_manifest,
            history_policy="canonical_causal_imputation",
        )
        source = self.metadata.identity
        source["role"] = "source"
        source.pop("identity_digest")
        source["identity_digest"] = _json_hash(source)
        if (
            source != source_reader_identity
            or self.metadata.sampling_identity != sampling_identity
            or grant.region != region
            or grant.source_reader_identity_digest != source["identity_digest"]
            or grant.sampling_digest != sampling_identity["sampling_digest"]
            or grant.statistics_identity_digest
            != statistics.identity["identity_digest"]
            or statistics.identity["reader_identity_digest"]
            != source["identity_digest"]
        ):
            raise ValueError(
                "full-window grant/source reader/sampling/saved statistics do not align"
            )
        self.source_reader_identity = copy.deepcopy(source)
        self._reads = 0
        self._read_bounds = []
        self._identity = {
            "evaluation_role": self.role,
            "sampling_protocol": "full_window",
            "region": region,
            "source_reader_identity": source,
            "sampling_identity": sampling_identity,
            "statistics_identity": dict(statistics.identity),
            "gate_sha256": grant.gate_sha256,
            "checkpoint_cohort_sha256": grant.checkpoint_cohort_sha256,
            "test_numeric_api_available": True,
        }

    grant = property(lambda self: self._grant)
    statistics = property(lambda self: self._statistics)
    identity = property(lambda self: copy.deepcopy(self._identity))

    def __getattr__(self, name):
        if name in {
            "region",
            "N",
            "T",
            "T_h",
            "T_p",
            "train_end",
            "val_end",
            "times_ns",
            "region_idx",
            "dow",
            "tod",
            "edge_index",
            "sample_start",
            "split",
            "sampling_identity",
        }:
            return getattr(self.metadata, name)
        raise AttributeError(name)

    def indices(self):
        return np.flatnonzero(self.split == SPLIT_CODES["test"])

    def require_unchanged(self):
        for name, path in self.metadata.paths.items():
            if _fingerprint(path) != self.metadata._fingerprints[name]:
                raise RuntimeError(
                    "full-window frozen raw material changed during evaluation"
                )
        source = self.metadata.identity
        source["role"] = "source"
        source.pop("identity_digest")
        source["identity_digest"] = _json_hash(source)
        if (
            source != self.source_reader_identity
            or self.statistics.identity["identity_digest"]
            != self.grant.statistics_identity_digest
        ):
            raise RuntimeError(
                "full-window source metadata or saved statistics changed"
            )

    def get_test_batch(self, indices, *, grant):
        if grant != self.grant:
            raise PermissionError("full-window test window request has another grant")
        selected = np.asarray(indices)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or not selected.size
            or len(np.unique(selected)) != len(selected)
            or (selected < 0).any()
            or (selected >= len(self.split)).any()
            or not np.all(self.split[selected] == 2)
        ):
            raise PermissionError(
                "full-window frozen reader only exposes exact declared test windows"
            )
        self.require_unchanged()
        rows = []
        with h5py.File(self.metadata.paths["traffic"], "r") as handle:
            for index in selected:
                anchor = int(self.sample_start[index])
                low, high = anchor - 11, anchor + 13
                if not self.val_end <= anchor + 1 or high > self.T:
                    raise PermissionError(
                        "full-window test targets cross the frozen temporal holdout"
                    )
                selection = (slice(low, high), slice(None), 0)
                flow = np.asarray(handle["flow_series_imputed"][selection])
                mask = np.asarray(handle["flow_mask"][selection])
                if (
                    flow.shape != (24, self.N)
                    or mask.dtype != np.bool_
                    or not np.isfinite(flow).all()
                    or (flow < 0).any()
                ):
                    raise FloatingPointError(
                        "frozen canonical flow0 test window is invalid"
                    )
                stats = self.statistics
                rows.append(
                    {
                        "flow_hist": flow[:12].T.copy(),
                        "history_mask": mask[:12].T.copy(),
                        "x_baseline": stats.lookup(
                            self.dow[low : anchor + 1],
                            self.tod[low : anchor + 1],
                            self.N,
                        ).T,
                        "y_baseline": stats.lookup(
                            self.dow[anchor + 1 : high],
                            self.tod[anchor + 1 : high],
                            self.N,
                        ).T,
                        "time_feat": np.stack(
                            (
                                self.tod[low : anchor + 1] / 288,
                                self.dow[low : anchor + 1] / 7,
                            ),
                            -1,
                        ).astype(np.float32),
                        "target": np.where(mask[12:], flow[12:], np.float32(0)).T,
                        "target_mask": mask[12:].T.copy(),
                    }
                )
                self._reads += 1
                self._read_bounds.append([low, high])
        self.require_unchanged()
        result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        result.update(
            sample_start=self.sample_start[selected].copy(),
            edge_index=self.edge_index.copy(),
            node_scale=self.statistics.scales(self.N),
        )
        return result

    def access_audit(self):
        return {
            "region": self.region,
            "role": self.role,
            "sampling_protocol": "full_window",
            "test_windows_read": self._reads,
            "numeric_fields": ["flow_series_imputed", "flow_mask"],
            "numeric_channels": [0],
            "test_window_bounds_sha256": canonical_sha256(self._read_bounds),
            "incident_or_affected_values_read": False,
            "development_test_api_modified": False,
            "metadata_access": self.metadata.access_audit(),
        }


class FullWindowForecastAdapter(torch.nn.Module):
    def __init__(self, model, *, kind, method):
        super().__init__()
        self.model, self.kind, self.method, self.scenario = model, kind, method, "clean"

    def frozen_config(self):
        if self.method == "shared_graph":
            factory = self.model.get_config()
        elif self.kind == "fullwindow_strong":
            factory = self.model.factory_kwargs()
        else:
            factory = research_factory_kwargs(self.model)
        return {"kind": self.kind, "method": self.method, "factory": factory}

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
        if self.method == "shared_graph":
            return self.model(
                flow_hist,
                x_baseline,
                y_baseline,
                time_feat,
                history_mask,
                edge_index,
                node_scale,
            )
        own_sanitization = self.kind == "fullwindow_ndr" and self.method in {
            "mask_signed",
            "sanitized_control",
        }
        history = (
            flow_hist
            if self.scenario == "clean" or own_sanitization
            else canonical_history(flow_hist, x_baseline, history_mask).filled
        )
        if self.kind == "fullwindow_strong":
            return self.model(history, time_feat)
        return forward_research(
            self.model, history, x_baseline, y_baseline, time_feat, history_mask
        )


def evaluate_fullwindow_models(
    models, *, reader, grant, batch_size=8, perturbation_seed=42, device="cpu"
):
    if (
        reader.role != "same_region_temporal_heldout"
        or len(models) < 2
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise PermissionError(
            "full-window evaluation needs a dedicated frozen test reader and real eval-mode models"
        )
    if any(model.training for model in models.values()):
        raise PermissionError("full-window models must already be frozen in eval mode")
    states = {name: state_digest(model.model) for name, model in models.items()}
    constructors = {name: model.frozen_config() for name, model in models.items()}
    modes = {
        name: [part.training for part in model.modules()]
        for name, model in models.items()
    }
    indices = reader.indices()
    streams = {}
    seen = set()
    observed_total = 0
    sums = {
        name: {scenario: MaskedForecastSums() for scenario in SCENARIOS}
        for name in models
    }
    blocks = {name: {scenario: {} for scenario in SCENARIOS} for name in models}
    elapsed = {name: {scenario: 0.0 for scenario in SCENARIOS} for name in models}
    removed = {scenario: 0 for scenario in SCENARIOS}
    origins_all = []
    for low in range(0, len(indices), batch_size):
        batch = reader.get_test_batch(indices[low : low + batch_size], grant=grant)
        if set(batch) != BATCH_FIELDS:
            raise ValueError(
                "full-window test batch violates the forecast/metric whitelist"
            )
        origins = batch["sample_start"]
        target, mask = batch["target"].copy(), batch["target_mask"].copy()
        if len(np.unique(origins)) != len(origins) or any(
            int(origin) in seen for origin in origins
        ):
            raise ValueError("duplicated full-window forecast origin")
        seen.update(map(int, origins))
        origins_all.extend(origins.tolist())
        days = (
            reader.times_ns[origins]
            .astype("datetime64[ns]")
            .astype("datetime64[D]")
            .astype(str)
        )
        observed_total += int(batch["history_mask"].sum())
        for field in ("sample_start", "target", "target_mask"):
            streams.setdefault(
                field, _StreamArrayHash(len(indices), batch[field])
            ).update(batch[field])
        for scenario in SCENARIOS:
            corrupted = (
                batch
                if scenario == "clean"
                else perturb_history(batch, scenario, seed=perturbation_seed)
            )
            removed[scenario] += int(
                batch["history_mask"].sum() - corrupted["history_mask"].sum()
            )
            for name, adapter in models.items():
                adapter.scenario = scenario
                started = time.perf_counter()
                prediction = predict_history_only(adapter, corrupted, device)
                elapsed[name][scenario] += time.perf_counter() - started
                if not np.array_equal(batch["target"], target) or not np.array_equal(
                    batch["target_mask"], mask
                ):
                    raise RuntimeError("full-window forecast mutated targets")
                sums[name][scenario].update(prediction, target, mask)
                for day in np.unique(days):
                    keep = days == day
                    blocks[name][scenario].setdefault(
                        str(day), MaskedForecastSums()
                    ).update(prediction[keep], target[keep], mask[keep])
    legacy_origins = legacy_array_digest(np.asarray(origins_all, np.int64))
    if legacy_origins != reader.sampling_identity["split_origin_sha256"]["test"]:
        raise ValueError("full-window test collection differs from its frozen sampling")
    if states != {
        name: state_digest(model.model) for name, model in models.items()
    } or modes != {
        name: [part.training for part in model.modules()]
        for name, model in models.items()
    }:
        raise RuntimeError("full-window evaluation mutated model state or eval mode")
    if constructors != {name: model.frozen_config() for name, model in models.items()}:
        raise RuntimeError("full-window evaluation mutated frozen model configuration")
    reader.require_unchanged()
    collection = {
        "sampling_protocol": "full_window",
        "split": "test",
        "samples": len(indices),
        "sampling_digest": reader.sampling_identity["sampling_digest"],
        "forecast_origins_sha256": streams["sample_start"].hexdigest(),
        "fullwindow_forecast_origins_legacy_sha256": legacy_origins,
        "targets_sha256": streams["target"].hexdigest(),
        "target_masks_sha256": streams["target_mask"].hexdigest(),
        "all_declared_split_samples": True,
        "day_basis": "audited raw traffic times_ns at forecast origin",
        "hash_format": "sha256(dtype_string+json(full_shape)+C_order_bytes)",
        "target_disjoint_between_splits": True,
        "adjacent_origins_share_targets_within_split": True,
    }
    reports = {
        name: {
            "collection": copy.deepcopy(collection),
            "scenarios": {
                scenario: {
                    **sums[name][scenario].result(),
                    "collection": copy.deepcopy(collection),
                    "day_blocks": {
                        day: value.totals()
                        for day, value in sorted(blocks[name][scenario].items())
                    },
                    "forward_seconds": elapsed[name][scenario],
                    "perturbation_seed": perturbation_seed,
                    "history_observed_points_original": observed_total,
                    "history_observed_points_removed": removed[scenario],
                }
                for scenario in SCENARIOS
            },
            "target_used_for_forward": False,
        }
        for name in models
    }
    return {
        "collection": collection,
        "models": reports,
        "model_state_sha256": states,
        "target_windows_read_once_for_all_models": True,
        "targets_used_for_forward": False,
        "forecast_input_fields": list(FORWARD_FIELDS),
    }


class FullWindowEvaluationLedger(RegionalEvaluationLedger):
    def reserve(self):
        return self._append(
            "reserved",
            {
                "unit": "distinct_training_seed_inference_slots_per_cohort_access",
                "family_scope": FAMILY_SCOPE,
                "sampling_protocol": "full_window",
            },
        )
