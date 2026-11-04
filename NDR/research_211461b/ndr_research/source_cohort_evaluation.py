"""Frozen two-source checkpoint cohorts with one paired heldout collection."""

from __future__ import annotations

import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ndr_research.cohort import preprocessing_semantics, training_recipe_identity
from ndr_research.data import array_digest
from ndr_research.indomain_evaluation import RegionalEvaluationLedger, state_digest, _validate_validation_pair
from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.shared_staeformer import SHARED_STAEFORMER_CONFIG, SharedSTAEformer
from ndr_research.transfer_data import PROTOCOL, REGIONS, TransferPreprocessing
from ndr_research.transfer_evaluation import (
    BATCH_FIELDS, CONTROL_KINDS, FORWARD_FIELDS, SCENARIOS, HistoryBaseline,
    MaskedForecastSums, _StreamArrayHash, canonical_sha256, file_sha256,
    perturb_history, predict_history_only,
)
from scripts.train_ndr_research import require_snapshot
from scripts.train_ndr_transfer import load_transfer_checkpoint
from scripts.train_shared_staeformer_transfer import load_reference_checkpoint

ROOT = Path(__file__).resolve().parents[1]
GATE_SCHEMA = "source_cohort_frozen_v1"
CERTIFICATE_SCHEMA = "source_cohort_independent_validation_v1"
RECEIPT_SCHEMA = "source_cohort_runner_process_receipt_v1"
REPORT_STATUS = "frozen_source_cohort_evaluation_complete"
KINDS = ("shared_graph", "shared_staeformer")
CLASSES = {"shared_graph": "SharedGraphNDR", "shared_staeformer": "SharedSTAEformer"}
SEEDS = (42, 1, 2)
MODES = ("source_only", "historical_calibrated_transfer")
FAMILY_SCOPE = "two_source_model_families_and_heldout_across_seeds_modes_prefixes"


def source_pool_digest(preprocessing: TransferPreprocessing) -> str:
    return canonical_sha256({"template": array_digest(preprocessing.template),
                             "residual_scale": array_digest(np.asarray(preprocessing.residual_scale, dtype=np.float32)),
                             "semantics": preprocessing_semantics(preprocessing.identity)})


def rebuild_source_model(admitted: Mapping[str, Any], device: torch.device | str = "cpu") -> torch.nn.Module:
    checkpoint = admitted["checkpoint"]
    factory = SharedGraphNDR if admitted["kind"] == "shared_graph" else SharedSTAEformer
    model = factory(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def inspect_source_model(
    checkpoint_path: Path | str, verification_path: Path | str, summary_path: Path | str, *,
    kind: str, certificate_path: Path | str | None = None, source_root: Path = ROOT,
    require_certificate: bool = True, allow_pending_verification: bool = False,
) -> dict[str, Any]:
    """Admit only fully trained true-class checkpoints selected on complete source val."""
    if kind not in KINDS:
        raise ValueError("source cohort supports real SharedGraphNDR and SharedSTAEformer checkpoints")
    loader = load_transfer_checkpoint if kind == "shared_graph" else load_reference_checkpoint
    checkpoint = loader(Path(checkpoint_path))
    config = checkpoint["config"]
    if checkpoint.get("schema_version") != 1 or config.get("schema_version") != 1:
        raise ValueError("source checkpoint schema differs from the declared schema one")
    if checkpoint.get("model_class") != CLASSES[kind] or config.get("model_class") != CLASSES[kind]:
        raise ValueError("source cohort checkpoint does not contain its real model class")
    if config.get("model_config") != checkpoint["model_config"]:
        raise ValueError("source model config differs between checkpoint and training config")
    require_snapshot(config["source_snapshot"], source_root)
    required_sources = {"ndr_research/shared_graph.py", "ndr_research/transfer_data.py", "scripts/train_ndr_transfer.py"}
    required_sources |= ({"ndr_research/graph_loss.py"} if kind == "shared_graph" else
                         {"ndr_research/shared_staeformer.py", "baselines/STAEformer/STAEformer.py", "scripts/train_shared_staeformer_transfer.py"})
    if not required_sources.issubset(config["source_snapshot"]["files"]):
        raise ValueError("source checkpoint snapshot omits its actual class, training or data source")
    if (config.get("max_train_batches") is not None or config.get("max_val_batches") is not None
            or config.get("claim_scope") != "source_validation_development_only"
            or config.get("complete_source_train_epoch") is not True
            or config.get("target_data_read") is not False or config.get("test_evaluation") is not False):
        raise PermissionError("source cohort requires complete formal source-only training and validation")
    if type(config.get("seed")) is not int or config["seed"] not in SEEDS or config.get("epochs") not in (30, 100):
        raise ValueError("source checkpoint needs a declared research seed and fixed 30/100 epoch budget")
    sources, heldout = tuple(checkpoint["source_regions"]), checkpoint["heldout_region"]
    if (len(sources) != 2 or len(set(sources)) != 2 or heldout in sources
            or set((*sources, heldout)) != set(REGIONS)
            or config.get("source_regions") != list(sources) or config.get("heldout_region") != heldout):
        raise ValueError("source cohort checkpoint is not its declared two-source leave-one-region-out fold")
    selected = checkpoint["selection_metric"]
    if (not isinstance(selected, Mapping) or selected.get("selection_split") != "source_val"
            or selected.get("all_source_validation_samples") is not True
            or selected.get("target_region_used") is not False
            or set(selected.get("regions", {})) != set(sources)):
        raise PermissionError("source checkpoint was not selected on full source-only validation")
    for region, row in selected["regions"].items():
        collection = row.get("selection", {})
        reader_identity = collection.get("reader_identity", {})
        if (collection.get("selection_split") != "source_val" or collection.get("all_validation_samples") is not True
                or reader_identity.get("role") != "source" or reader_identity.get("region") != region
                or collection.get("samples") != reader_identity.get("counts", {}).get("val")):
            raise PermissionError("source validation selection does not cover its exact source reader metadata")
    source = TransferPreprocessing.from_state(checkpoint["source_statistics"])
    identity = checkpoint["source_statistics_identity"]
    if (source.mode != "source_only" or source.identity != identity
            or identity.get("target_statistics_used") is not False
            or set(identity.get("source_regions", [])) != set(sources)
            or config.get("source_statistics_identity") != identity
            or config.get("data_identity") != checkpoint["data_identity"]):
        raise PermissionError("source checkpoint statistics are not its exact frozen two-source pool")
    if kind == "shared_staeformer" and config.get("architecture") != SHARED_STAEFORMER_CONFIG:
        raise ValueError("STAEformer checkpoint is not the fixed shared temporal adaptation")
    verified = json.loads(Path(verification_path).read_text()) if verification_path is not None else None
    if verified is None and not allow_pending_verification:
        raise PermissionError("source cohort needs its full source-validation verification")
    summary = json.loads(Path(summary_path).read_text())
    checkpoint_sha = file_sha256(checkpoint_path)
    bindings = {"schema_version": 1, "status": "verified_transfer_source_checkpoint_complete_validation",
                "checkpoint_sha256": checkpoint_sha, "config_digest": checkpoint["config_digest"],
                "source_statistics_identity": identity, "source_snapshot_files_digest": config["source_snapshot"]["files_digest"],
                "source_files_digest": config["source_snapshot"]["files_digest"],
                "selection_metric_digest": canonical_sha256(selected), "model_config_sha256": canonical_sha256(checkpoint["model_config"]),
                "source_regions": list(sources), "heldout_region": heldout, "all_source_validation_samples": True,
                "claim_scope": "source_validation_development_only", "target_data_read": False, "test_evaluated": False}
    if verified is not None:
        for name, value in bindings.items():
            if verified.get(name) != value:
                raise PermissionError(f"complete independent source validation does not bind {name}")
        observed = verified.get("source_validation", {})
        if (observed.get("all_source_validation_samples") is not True or observed.get("target_region_used") is not False
                or observed.get("selection_split") != "source_val" or set(observed.get("regions", {})) != set(sources)):
            raise PermissionError("source verification does not reproduce a complete source collection")
        for region in sources:
            _validate_validation_pair(selected["regions"][region], observed["regions"][region])
        if not math.isclose(selected["macro_global_mae"], observed["macro_global_mae"], rel_tol=2e-6, abs_tol=2e-5):
            raise ValueError("source validation macro metric differs from its selected checkpoint")
    if (summary.get("status") != "training_completed_source_validation_pending_independent_verification"
            or summary.get("completed_epochs") != config["epochs"]
            or summary.get("config_digest") != checkpoint["config_digest"]
            or summary.get("test_evaluated") is not False or summary.get("target_data_read") is not False
            or summary.get("source_regions") != list(sources) or summary.get("heldout_region") != heldout
            or summary.get("best_epoch") != checkpoint["progress"]["best_epoch"]
            or canonical_sha256(summary.get("best_source_validation")) != canonical_sha256(selected)):
        raise PermissionError("source training has not completed its frozen budget with this selected best checkpoint")
    architecture = (dict(checkpoint["model_config"]) if kind == "shared_graph"
                    else {"architecture": dict(SHARED_STAEFORMER_CONFIG), "adaptation": "shared_temporal_12_by_1_by_80"})
    family = {"checkpoint_kind": kind, "model_class": CLASSES[kind], "architecture": architecture,
              "source_files_digest": config["source_snapshot"]["files_digest"],
              "training_recipe_identity": training_recipe_identity(config), "training_runtime": config["runtime"],
              "source_regions": list(sources), "heldout_region": heldout,
              "source_pool_numeric_semantics_sha256": source_pool_digest(source)}
    admitted = {"checkpoint": checkpoint, "kind": kind, "checkpoint_kind": kind, "model_class": CLASSES[kind],
                "model_family": f"shared_graph_h{architecture['hidden_dim']}" if kind == "shared_graph" else "shared_staeformer_temporal",
                "training_seed": config["seed"], "training_regions": sources, "source_regions": list(sources),
                "heldout_region": heldout, "bundle_digest": identity["bundle_digest"],
                "family_identity": family, "training_recipe": training_recipe_identity(config),
                "training_runtime": config["runtime"], "model_config": checkpoint["model_config"],
                "source_snapshot": copy.deepcopy(config["source_snapshot"]),
                "source_statistics_identity": copy.deepcopy(identity), "data_identity": copy.deepcopy(checkpoint["data_identity"]),
                "training_scope": "two_source_supervised_target_disjoint_event",
                "checkpoint_sha256": checkpoint_sha, "verification_sha256": file_sha256(verification_path) if verification_path is not None else None,
                "summary_sha256": file_sha256(summary_path), "config_digest": checkpoint["config_digest"],
                "validation_selection_sha256": canonical_sha256(selected),
                "data_identity_sha256": canonical_sha256(checkpoint["data_identity"]),
                "source_pool_numeric_semantics_sha256": source_pool_digest(source)}
    if require_certificate:
        if certificate_path is None:
            raise PermissionError("source cohort needs a fresh-process full source-validation certificate")
        certificate = json.loads(Path(certificate_path).read_text())
        receipt = certificate.get("schema_version") == RECEIPT_SCHEMA
        expected = {"schema_version": RECEIPT_SCHEMA if receipt else CERTIFICATE_SCHEMA,
                    "status": "recorded_independent_complete_source_validation" if receipt else "certified_independent_complete_source_validation",
                    "model_class": CLASSES[kind], "checkpoint_sha256": checkpoint_sha,
                    "verification_sha256": admitted["verification_sha256"], "summary_sha256": admitted["summary_sha256"],
                    "config_digest": admitted["config_digest"], "source_snapshot_files_digest": config["source_snapshot"]["files_digest"],
                    "runtime": config["runtime"], "data_identity": checkpoint["data_identity"],
                    "selection_metric_digest": admitted["validation_selection_sha256"],
                    "source_pool_numeric_semantics_sha256": admitted["source_pool_numeric_semantics_sha256"],
                    "verification_execution": "fresh_runner_verify_process" if receipt else "fresh_cli_process_checkpoint_reconstruction",
                    "checkpoint_existed_before_verifier_entry": True, "source_validation_complete": True,
                    "target_data_read": False, "test_evaluated": False, "training_executed_in_verifier": False}
        for name, value in expected.items():
            if certificate.get(name) != value:
                raise PermissionError(f"source-validation certificate does not bind {name}")
        if type(certificate.get("verification_process_id")) is not int or certificate["verification_process_id"] <= 0:
            raise ValueError("source-validation certificate has no independent process identity")
        process_start, checkpoint_modified = certificate.get("verifier_process_started_ns"), certificate.get("checkpoint_mtime_ns")
        if type(process_start) is not int or type(checkpoint_modified) is not int or not 0 < checkpoint_modified < process_start:
            raise PermissionError("checkpoint did not exist before the verifier process started")
        training_pid = config.get("training_process_id")
        if training_pid is not None and certificate["verification_process_id"] == training_pid:
            raise PermissionError("source verification process equals its training process")
        if receipt and (certificate.get("returncode") != 0 or certificate.get("runner_mode") != "verify"):
            raise PermissionError("source validation process receipt is not a successful verify-only process")
        reproduced = verified["source_validation"] if receipt else certificate.get("source_validation", {})
        if (reproduced.get("selection_split") != "source_val" or reproduced.get("all_source_validation_samples") is not True
                or reproduced.get("target_region_used") is not False or set(reproduced.get("regions", {})) != set(sources)):
            raise PermissionError("independent certificate does not contain complete source-only validation")
        for region in sources:
            _validate_validation_pair(selected["regions"][region], reproduced.get("regions", {}).get(region, {}))
        if not math.isclose(selected["macro_global_mae"], reproduced.get("macro_global_mae", float("inf")), rel_tol=2e-6, abs_tol=2e-5):
            raise ValueError("independent certificate macro source metric differs")
        admitted["certificate_sha256"] = file_sha256(certificate_path)
    # Actual class reconstruction rejects relabelled state dictionaries.
    model = rebuild_source_model(admitted)
    if kind == "shared_staeformer":
        center, scale = model.flow_center.item(), model.flow_scale.item()
        if center != np.float32(np.mean(source.template, dtype=np.float64)) or scale != np.float32(source.residual_scale):
            raise ValueError("STAEformer registered normalization differs from its source pool")
    admitted["model_state_sha256"] = state_digest(model)
    admitted["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    admitted["model_spec"] = copy.deepcopy(model.research_spec)
    return admitted


def admit_source_cohort_gate(gate_path: Path | str, *, source_root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(gate_path).resolve()
    gate = json.loads(path.read_text())
    if (gate.get("schema_version") != GATE_SCHEMA or gate.get("protocol_version") != PROTOCOL
            or gate.get("sampling_protocol") != "event" or gate.get("split") != "test"
            or gate.get("structure_development_closed") is not True or gate.get("selection_frozen") is not True
            or gate.get("test_used_for_structure_selection") is not False or not gate.get("authorization_id")
            or gate.get("comparison_scope") != "matched_two_source_training"
            or gate.get("training_seeds") != list(SEEDS)):
        raise PermissionError("source cohort requires an explicit closed-development three-seed event-test gate")
    if gate.get("scenarios") != list(SCENARIOS) or gate.get("control_kinds") != list(CONTROL_KINDS):
        raise ValueError("source cohort must freeze five scenarios and both fixed source-only controls")
    modes = gate.get("permitted_evaluation_modes")
    if modes not in (["source_only"], list(MODES)) or gate.get("evaluation_mode") not in modes:
        raise ValueError("source cohort must predeclare source-only or both transfer modes")
    expected_budget = len(SEEDS) * len(modes) * 2
    if gate.get("max_test_evaluations") != expected_budget or not gate.get("evaluation_ledger"):
        raise ValueError("three seeds plus independent repeat require six or twelve fixed inference slots")
    if type(gate.get("perturbation_seed")) is not int or not 0 <= gate["perturbation_seed"] < 2**32:
        raise ValueError("source cohort perturbation seed must be a frozen uint32")
    if gate["evaluation_mode"] == "historical_calibrated_transfer":
        prefix = gate.get("historical_prefix", {})
        if type(prefix.get("prefix_end")) is not int or prefix["prefix_end"] <= 0 or not prefix.get("authorization_id"):
            raise PermissionError("historical calibration needs an explicitly authorized fixed prefix")
    elif "historical_prefix" in gate:
        raise PermissionError("source-only evaluation cannot use a target prefix")
    admitted = {}
    for row in gate.get("models", []):
        name = row.get("model_id", "")
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in admitted:
            raise ValueError("source cohort model identifiers must be safe and unique")
        paths = {}
        for field in ("checkpoint", "verification", "summary", "certificate"):
            value = Path(row[field])
            paths[field] = value if value.is_absolute() else path.parent / value
            if file_sha256(paths[field]) != row.get(field + "_sha256"):
                raise ValueError("source cohort checkpoint or source acceptance artifact changed")
        spec = inspect_source_model(paths["checkpoint"], paths["verification"], paths["summary"],
                                    kind=row["checkpoint_kind"], certificate_path=paths["certificate"], source_root=source_root)
        for field in ("family_identity", "training_seed", "model_class", "config_digest", "validation_selection_sha256",
                      "data_identity_sha256", "source_pool_numeric_semantics_sha256", "model_state_sha256"):
            if row.get(field) != spec[field]:
                raise ValueError(f"source cohort gate does not bind {field}")
        if (spec["bundle_digest"] != gate.get("cache_bundle_digest")
                or spec["source_regions"] != gate.get("source_regions") or spec["heldout_region"] != gate.get("region")):
            raise ValueError("source cohort mixes cache bundles or leave-one-region-out folds")
        spec.update(model_id=name, paths=paths)
        admitted[name] = spec
    for kind in KINDS:
        current = [spec for spec in admitted.values() if spec["kind"] == kind]
        if len(current) != 3 or {spec["training_seed"] for spec in current} != set(SEEDS):
            raise ValueError("each true source model requires exactly the three frozen seeds")
        if len({canonical_sha256(spec["family_identity"]) for spec in current}) != 1:
            raise ValueError("source seeds change architecture, recipe, runtime or source pool semantics")
    if len(admitted) != 6 or len({spec["source_pool_numeric_semantics_sha256"] for spec in admitted.values()}) != 1:
        raise ValueError("six real checkpoint models must share the same source pool numeric semantics")
    gate["gate_sha256"] = file_sha256(path)
    gate["family_scope"] = FAMILY_SCOPE
    return gate, admitted


class SourceCohortLedger(RegionalEvaluationLedger):
    """The inherited family excludes seed, mode and calibration prefix identities."""

    def reserve(self) -> dict[str, Any]:
        return self._append("reserved", {"unit": "distinct_training_seed_inference_slots_per_cohort_access",
                                          "family_scope": FAMILY_SCOPE, "evaluation_mode": self.gate["evaluation_mode"],
                                          "permitted_evaluation_modes": self.gate["permitted_evaluation_modes"]})


def batch_with_preprocessing(batch: Mapping[str, np.ndarray], reader: Any, preprocessing: Any) -> dict[str, np.ndarray]:
    result = dict(batch)
    history = batch["sample_start"][:, None] + np.arange(-11, 1)
    future = batch["sample_start"][:, None] + np.arange(1, 13)
    for field, positions in (("x_baseline", history), ("y_baseline", future)):
        result[field] = np.stack([preprocessing.lookup(reader.dow[row], reader.tod[row], reader.N).T for row in positions])
    result["node_scale"] = preprocessing.scales(reader.N)
    return result


def evaluate_source_cohort(
    models: Mapping[str, torch.nn.Module], *, reader: Any, preprocessing: Any,
    source_preprocessing: Any, evaluation_grant: Any, batch_size: int = 8,
    perturbation_seed: int = 42, device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Read every test window once, sharing targets and perturbations across models."""
    if len(models) < 2 or set(models) & set(CONTROL_KINDS):
        raise ValueError("source evaluation requires real models separate from fixed controls")
    if reader.role != "target" or source_preprocessing.mode != "source_only":
        raise PermissionError("source evaluation requires a target-role reader and frozen source-only controls")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("source evaluation batch size must be positive")
    source_preprocessing.validate_reader(reader)
    preprocessing.validate_reader(reader)
    complete = {**models, **{kind: HistoryBaseline(kind).to(device).eval() for kind in CONTROL_KINDS}}
    for model in complete.values():
        if model.training:
            raise PermissionError("source cohort models must already be frozen in eval mode")
    states = {name: state_digest(model) for name, model in complete.items()}
    constructors = {name: model.get_config() for name, model in models.items() if hasattr(model, "get_config")}
    training_modes = {name: [module.training for module in model.modules()] for name, model in complete.items()}
    indices = reader.indices("test")
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not indices.size or len(np.unique(indices)) != len(indices):
        raise ValueError("source cohort requires one nonempty unique full test index collection")
    metrics = {name: {scenario: MaskedForecastSums() for scenario in SCENARIOS} for name in complete}
    blocks = {name: {scenario: {} for scenario in SCENARIOS} for name in complete}
    durations = {name: {scenario: 0.0 for scenario in SCENARIOS} for name in complete}
    removed = {scenario: 0 for scenario in SCENARIOS}
    streams, origins_seen, original_observed = {}, set(), 0
    started = time.perf_counter()
    for low in range(0, len(indices), batch_size):
        selected = indices[low:low + batch_size]
        batch = reader.get_batch(selected, preprocessing, split="test", evaluation_gate=evaluation_grant)
        if set(batch) != BATCH_FIELDS:
            raise ValueError("source target batch violates the target-free forecast whitelist")
        origins = np.asarray(batch["sample_start"])
        if origins.shape != (len(selected),) or origins.dtype.kind not in "iu" or (origins < 0).any() or (origins >= len(reader.times_ns)).any():
            raise ValueError("source cohort origins do not align with real audited timestamps")
        if any(int(origin) in origins_seen for origin in origins) or len(set(map(int, origins))) != len(origins):
            raise ValueError("source cohort evaluation origins are duplicated")
        origins_seen.update(map(int, origins))
        days = reader.times_ns[origins].astype("datetime64[ns]").astype("datetime64[D]").astype(str)
        target, mask = batch["target"].copy(), batch["target_mask"].copy()
        original_observed += int(batch["history_mask"].sum())
        for field in ("sample_start", "target", "target_mask"):
            streams.setdefault(field, _StreamArrayHash(len(indices), batch[field])).update(batch[field])
        for scenario in SCENARIOS:
            corrupted = perturb_history(batch, scenario, seed=perturbation_seed)
            removed[scenario] += int(batch["history_mask"].sum() - corrupted["history_mask"].sum())
            controls = batch_with_preprocessing(corrupted, reader, source_preprocessing)
            for name, model in complete.items():
                current = controls if name in CONTROL_KINDS else corrupted
                for field in ("target", "target_mask", "sample_start", "flow_hist", "history_mask"):
                    if not np.array_equal(current[field], corrupted[field], equal_nan=True):
                        raise ValueError("source controls changed the shared history or target collection")
                tick = time.perf_counter()
                prediction = predict_history_only(model, current, device)
                durations[name][scenario] += time.perf_counter() - tick
                if not np.array_equal(batch["target"], target, equal_nan=True) or not np.array_equal(batch["target_mask"], mask):
                    raise RuntimeError("source cohort forward mutated target observations")
                metrics[name][scenario].update(prediction, target, mask)
                for day in np.unique(days):
                    keep = days == day
                    blocks[name][scenario].setdefault(str(day), MaskedForecastSums()).update(prediction[keep], target[keep], mask[keep])
    if {name: state_digest(model) for name, model in complete.items()} != states:
        raise RuntimeError("source cohort evaluation mutated frozen model state")
    if ({name: model.get_config() for name, model in models.items() if hasattr(model, "get_config")} != constructors
            or {name: [module.training for module in model.modules()] for name, model in complete.items()} != training_modes):
        raise RuntimeError("source cohort evaluation mutated its frozen model configuration or eval mode")
    collection = {"samples": len(indices), "forecast_origins_sha256": streams["sample_start"].hexdigest(),
                  "targets_sha256": streams["target"].hexdigest(), "target_masks_sha256": streams["target_mask"].hexdigest(),
                  "all_declared_split_samples": True, "day_basis": "audited cache times_ns at forecast origin",
                  "hash_format": "sha256(dtype_string+json(full_shape)+C_order_bytes)"}
    reports = {}
    for name in complete:
        scenarios = {scenario: {**metrics[name][scenario].result(), "collection": copy.deepcopy(collection),
                               "day_blocks": {day: totals.totals() for day, totals in sorted(blocks[name][scenario].items())},
                               "forward_seconds": durations[name][scenario], "perturbation_seed": perturbation_seed,
                               "history_observed_points_original": original_observed,
                               "history_observed_points_removed": removed[scenario]} for scenario in SCENARIOS}
        reports[name] = {"collection": copy.deepcopy(collection), "scenarios": scenarios,
                         "source_only_control": name in CONTROL_KINDS, "target_used_for_forward": False}
    return {"collection": collection, "models": reports, "model_state_sha256": states,
            "target_windows_read_once_for_all_models": True, "targets_used_for_forward": False,
            "forecast_input_fields": list(FORWARD_FIELDS), "wallclock_seconds": time.perf_counter() - started}
