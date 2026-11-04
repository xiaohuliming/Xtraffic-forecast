"""Frozen transfer evaluation, observation-only corruptions and paired day blocks.

This module never selects a model or fits preprocessing. An explicit gate binds
the selected source-validation checkpoint before any held-out window is read.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


FORWARD_FIELDS = ("flow_hist", "x_baseline", "y_baseline", "time_feat",
                  "history_mask", "edge_index", "node_scale")
BATCH_FIELDS = frozenset((*FORWARD_FIELDS, "target", "target_mask", "sample_start"))
SCENARIOS = ("clean", "random_missing_10pct", "random_missing_30pct",
             "last_missing_3steps", "last_missing_6steps")
REQUIRED_CANDIDATE_FILES = frozenset(("ndr_research/shared_graph.py",
                                    "ndr_research/transfer_data.py",
                                    "ndr_research/graph_loss.py",
                                    "scripts/train_ndr_transfer.py"))
VERIFIED_SOURCE_STATUS = "verified_transfer_source_checkpoint_complete_validation"
MAPE_MIN_ABSOLUTE_TARGET = 1e-4
CONTROL_KINDS = ("source_only_persistence", "source_only_pooled_climatology")
CANDIDATE_KINDS = ("shared_graph_ndr", *CONTROL_KINDS)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def write_new_json(path: Path | str, value: Any) -> None:
    """Exclusive creation prevents overwriting an earlier held-out report."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def require_candidate_sources(snapshot: Mapping[str, Any], root: Path | str) -> None:
    files = snapshot.get("files", {})
    if not REQUIRED_CANDIDATE_FILES.issubset(files):
        raise ValueError("candidate snapshot omits a model, data or training source")
    if canonical_sha256(files) != snapshot.get("files_digest"):
        raise ValueError("candidate source snapshot digest is inconsistent")
    root = Path(root).resolve()
    for relative, identity in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError("candidate source path escapes the checkout or is a symlink")
        if not path.is_file() or file_sha256(path) != identity.get("sha256"):
            raise ValueError(f"candidate source hash changed: {relative}")
        if path.stat().st_size != identity.get("size_bytes"):
            raise ValueError(f"candidate source size changed: {relative}")


@dataclass(frozen=True)
class FrozenEvaluationGate:
    authorization_id: str
    candidate_sha: str
    checkpoint_sha256: str
    heldout_region: str
    source_regions: tuple[str, str]
    mode: str
    split: str
    seed: int
    max_test_evaluations: int
    ledger_path: Path
    gate_sha256: str
    source_verification_sha256: str
    selection_sha256: str
    prefix_end: int | None = None
    prefix_authorization_id: str | None = None


def _gate_path(value: Any, gate_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"gate requires {field}")
    path = Path(value)
    return (path if path.is_absolute() else gate_path.parent / path).resolve()


def validate_frozen_gate(gate_path: Path | str, checkpoint_path: Path | str,
                         checkpoint: Mapping[str, Any], *, source_root: Path | str) -> FrozenEvaluationGate:
    """Verify code, checkpoint, full source validation and explicit frozen selection."""
    gate_path, checkpoint_path = Path(gate_path).resolve(), Path(checkpoint_path).resolve()
    declaration = json.loads(gate_path.read_text(encoding="utf-8"))
    if declaration.get("schema_version") != 1 or declaration.get("selection_frozen") is not True:
        raise PermissionError("evaluation needs an explicit schema-one frozen selection gate")
    authorization_id = declaration.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise PermissionError("evaluation gate requires an authorization identity")
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != _checked_sha(declaration.get("modelcheckpoint_sha"), "modelcheckpoint_sha"):
        raise ValueError("frozen checkpoint hash differs from the evaluation gate")
    config = checkpoint["config"]
    if canonical_sha256(config) != checkpoint.get("config_digest"):
        raise ValueError("checkpoint configuration digest is inconsistent")
    snapshot = config["source_snapshot"]
    require_candidate_sources(snapshot, source_root)
    candidate_sha = _checked_sha(declaration.get("candidate_sha"), "candidate_sha")
    if candidate_sha != snapshot["files_digest"]:
        raise ValueError("candidate identity differs from its source snapshot")
    model_config_sha = canonical_sha256(checkpoint["model_config"])
    if model_config_sha != declaration.get("model_config_sha"):
        raise ValueError("model recipe differs from the frozen gate")
    sources = tuple(checkpoint["source_regions"])
    heldout = checkpoint["heldout_region"]
    if (len(sources) != 2 or len(set(sources)) != 2 or heldout in sources
            or set((*sources, heldout)) != {"Alameda", "ContraCosta", "Orange"}):
        raise ValueError("checkpoint is not a complete leave-one-region-out fold")
    if declaration.get("source_regions") != list(sources) or declaration.get("heldout_region") != heldout:
        raise ValueError("gate source/held-out regions differ from the selected checkpoint")
    if (checkpoint.get("selection_split") != "source_val" or checkpoint.get("target_data_read") is not False
            or config.get("claim_scope") != "source_validation_development_only"):
        raise PermissionError("held-out evaluation requires a source-only development checkpoint")
    selection = checkpoint.get("selection_metric")
    if (not isinstance(selection, Mapping) or selection.get("selection_split") != "source_val"
            or selection.get("all_source_validation_samples") is not True
            or selection.get("target_region_used") is not False
            or set(selection.get("regions", {})) != set(sources)):
        raise PermissionError("checkpoint was not selected on complete source-only validation")
    selection_sha = canonical_sha256(selection)
    if declaration.get("source_validation_selection_sha") != selection_sha:
        raise ValueError("gate does not bind the original source-validation selection")
    source_identity = checkpoint["source_statistics_identity"]
    if (source_identity.get("mode") != "source_only" or source_identity.get("target_statistics_used") is not False
            or set(source_identity.get("source_regions", [])) != set(sources)
            or source_identity.get("bundle_digest") != declaration.get("cache_bundle_digest")):
        raise PermissionError("checkpoint statistics are not the declared source-only pool")
    verify_path = _gate_path(declaration.get("source_validation_verification"), gate_path,
                             "source_validation_verification")
    verify_sha = file_sha256(verify_path)
    if verify_sha != declaration.get("source_validation_verification_sha"):
        raise ValueError("source-validation verification report hash mismatch")
    verified = json.loads(verify_path.read_text(encoding="utf-8"))
    bindings = {
        "status": VERIFIED_SOURCE_STATUS, "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha, "config_digest": checkpoint["config_digest"],
        "source_statistics_identity": source_identity,
        "source_snapshot_files_digest": candidate_sha, "selection_metric_digest": selection_sha,
        "model_config_sha256": model_config_sha, "source_regions": list(sources),
        "heldout_region": heldout, "all_source_validation_samples": True,
        "claim_scope": "source_validation_development_only", "target_data_read": False,
        "test_evaluated": False,
    }
    for key, expected in bindings.items():
        if verified.get(key) != expected:
            raise PermissionError(f"source-validation verification does not bind {key}")
    reproduced = verified.get("source_validation", {})
    if (reproduced.get("selection_split") != "source_val"
            or reproduced.get("all_source_validation_samples") is not True
            or reproduced.get("target_region_used") is not False
            or set(reproduced.get("regions", {})) != set(sources)):
        raise PermissionError("verification does not certify complete source-only validation")
    for region in sources:
        before, after = selection["regions"][region], reproduced["regions"][region]
        for key in ("valid_count", "selection", "observed_targets_sha256", "observation_masks_sha256"):
            if before.get(key) != after.get(key):
                raise ValueError("source-validation collection differs from the frozen selection")
        if before.get("selection", {}).get("all_validation_samples") is not True:
            raise PermissionError("source-validation collection is a subset")
        if not math.isclose(before["global_masked_mae"], after["global_masked_mae"], rel_tol=2e-6, abs_tol=2e-5):
            raise ValueError("source-validation MAE did not reproduce the selected checkpoint")
    if not math.isclose(selection["macro_global_mae"], reproduced["macro_global_mae"], rel_tol=2e-6, abs_tol=2e-5):
        raise ValueError("macro source-validation MAE did not reproduce")
    mode, split = declaration.get("evaluation_mode"), declaration.get("split")
    if mode not in ("source_only", "historical_calibrated_transfer") or split not in ("val", "test"):
        raise ValueError("gate must fix a transfer mode and a held-out val/test split")
    seed = declaration.get("perturbation_seed")
    maximum = declaration.get("max_test_evaluations")
    if type(seed) is not int or not 0 <= seed < 2**32 or type(maximum) is not int or not 1 <= maximum <= 100:
        raise ValueError("gate needs a fixed uint32 perturbation seed and positive test budget")
    if declaration.get("scenarios") != list(SCENARIOS):
        raise ValueError("gate must freeze exactly the five declared missingness scenarios")
    if declaration.get("candidate_kinds") != list(CANDIDATE_KINDS):
        raise ValueError("gate must predeclare the candidate and both fixed source-only controls")
    prefix_end = prefix_authorization = None
    if mode == "historical_calibrated_transfer":
        prefix = declaration.get("historical_prefix", {})
        prefix_end, prefix_authorization = prefix.get("prefix_end"), prefix.get("authorization_id")
        if type(prefix_end) is not int or prefix_end <= 0 or not isinstance(prefix_authorization, str) or not prefix_authorization.strip():
            raise PermissionError("historical calibration needs a separately authorized fixed prefix")
    elif "historical_prefix" in declaration:
        raise PermissionError("source-only evaluation cannot contain a target calibration prefix")
    return FrozenEvaluationGate(authorization_id, candidate_sha, checkpoint_sha, heldout, sources,
                                mode, split, seed, maximum,
                                _gate_path(declaration.get("evaluation_ledger"), gate_path, "evaluation_ledger"),
                                file_sha256(gate_path), verify_sha, selection_sha,
                                prefix_end, prefix_authorization)


class EvaluationLedger:
    """One frozen family budget across checkpoint seeds and transfer modes.

    The family key is candidate source digest plus held-out region. The first
    started attempt fixes its test limit. Target-validation attempts are logged
    independently; failures still consume a reserved test attempt.
    """

    def __init__(self, gate: FrozenEvaluationGate, output: Path | str) -> None:
        self.gate, self.output = gate, str(Path(output).resolve())
        self.attempt_id = uuid.uuid4().hex

    def _append(self, action: str, details: Mapping[str, Any]) -> dict[str, Any]:
        path = self.gate.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(path.name + ".lock")
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.close(descriptor)
            rows = []
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            previous = "0" * 64
            for row in rows:
                signature = row.get("record_sha256")
                payload = {key: value for key, value in row.items() if key != "record_sha256"}
                if payload.get("previous_sha256") != previous or canonical_sha256(payload) != signature:
                    raise ValueError("evaluation ledger hash chain is invalid")
                previous = signature
            family = [row for row in rows if row["action"] == "started"
                      and row["candidate_sha"] == self.gate.candidate_sha
                      and row["heldout_region"] == self.gate.heldout_region]
            if family:
                frozen_limit = family[0].get("family_max_test_evaluations")
                if type(frozen_limit) is not int or any(row.get("family_max_test_evaluations") != frozen_limit for row in family):
                    raise ValueError("evaluation ledger lacks one consistent frozen family test budget")
                if self.gate.max_test_evaluations != frozen_limit:
                    raise PermissionError("the frozen family test-evaluation limit cannot change after first access")
            used = sum(row["split"] == "test" for row in family)
            val_used = sum(row["split"] == "val" for row in family)
            if action == "started" and self.gate.split == "test" and used >= self.gate.max_test_evaluations:
                raise PermissionError("frozen candidate test-evaluation budget is exhausted")
            if action != "started" and not any(row["attempt_id"] == self.attempt_id and row["action"] == "started" for row in rows):
                raise ValueError("ledger attempt was not reserved before reading held-out data")
            row = {"action": action, "attempt_id": self.attempt_id,
                   "previous_sha256": previous, "unix_time_ns": time.time_ns(),
                   "candidate_sha": self.gate.candidate_sha,
                   "checkpoint_sha256": self.gate.checkpoint_sha256,
                   "heldout_region": self.gate.heldout_region, "split": self.gate.split,
                   "mode": self.gate.mode, "gate_sha256": self.gate.gate_sha256,
                   "candidate_kind": "shared_graph_ndr", "control_candidate_kinds": list(CONTROL_KINDS),
                   "family_max_test_evaluations": self.gate.max_test_evaluations,
                   "family_budget_scope": "candidate_source_digest_and_heldout_region_across_checkpoint_seeds_and_transfer_modes",
                   "authorization_id": self.gate.authorization_id, "output": self.output,
                   "test_access_attempt_number": used + int(action == "started" and self.gate.split == "test"),
                   "target_val_access_attempt_number": val_used + int(action == "started" and self.gate.split == "val"),
                   "details": dict(details)}
            row["record_sha256"] = canonical_sha256(row)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return row
        finally:
            lock.unlink(missing_ok=True)

    def reserve(self) -> dict[str, Any]:
        return self._append("started", {"scenarios": list(SCENARIOS), "seed": self.gate.seed,
                                        "candidate_kinds": list(CANDIDATE_KINDS)})

    def finish(self, *, status: str, report_sha256: str | None = None, error_type: str | None = None) -> dict[str, Any]:
        return self._append(status, {"report_sha256": report_sha256, "error_type": error_type})


def perturb_history(batch: Mapping[str, np.ndarray], scenario: str, *, seed: int) -> dict[str, np.ndarray]:
    """Corrupt only observable history, deterministically per forecast origin."""
    if set(batch) != BATCH_FIELDS:
        raise ValueError("evaluation batch is not the forecast/metric whitelist")
    if scenario not in SCENARIOS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("unknown or unfrozen observation perturbation")
    flow = np.asarray(batch["flow_hist"])
    mask = np.asarray(batch["history_mask"])
    origins = np.asarray(batch["sample_start"])
    if flow.ndim != 3 or mask.dtype != np.bool_ or mask.shape != flow.shape:
        raise ValueError("history requires matching B,N,T flow and boolean mask")
    if origins.shape != (flow.shape[0],) or origins.dtype.kind not in "iu" or (origins < 0).any():
        raise ValueError("perturbation needs an integer origin for every sample")
    if not np.isfinite(flow[mask]).all() or (flow[mask] < 0).any():
        raise FloatingPointError("original observed history must be finite and nonnegative")
    observed = mask.copy()
    if scenario.startswith("random_missing"):
        probability = 0.1 if scenario == "random_missing_10pct" else 0.3
        code = 10 if probability == 0.1 else 30
        for index, origin in enumerate(origins):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(origin), code]))
            observed[index] &= rng.random(flow.shape[1:]) >= probability
    elif scenario.startswith("last_missing"):
        steps = 3 if scenario == "last_missing_3steps" else 6
        if steps > flow.shape[-1]:
            raise ValueError("history is shorter than the frozen contiguous outage")
        observed[..., -steps:] = False
    result = dict(batch)
    result["history_mask"] = observed
    result["flow_hist"] = np.where(observed, flow, np.float32(0)).astype(flow.dtype, copy=False)
    return result


def predict_history_only(model: Any, batch: Mapping[str, np.ndarray], device: torch.device | str) -> np.ndarray:
    """Explicit seven-argument forecast boundary; targets stay on the metric side."""
    if set(batch) != BATCH_FIELDS:
        raise ValueError("evaluation batch contains fields outside the forecast/metric whitelist")
    signature = inspect.signature(model.forward)
    if tuple(signature.parameters) != FORWARD_FIELDS:
        raise ValueError("model forward must expose exactly the seven history-only inputs")
    device = torch.device(device)
    tensors = {
        key: torch.as_tensor(np.array(batch[key], copy=True), device=device,
                             dtype=torch.bool if key == "history_mask" else torch.long if key == "edge_index" else torch.float32)
        for key in FORWARD_FIELDS
    }
    if model.training:
        raise PermissionError("transfer evaluation requires a frozen eval-mode model")
    with torch.inference_mode():
        prediction = model(*(tensors[key] for key in FORWARD_FIELDS))
    if not isinstance(prediction, torch.Tensor) or prediction.shape != np.shape(batch["target"]):
        raise ValueError("prediction does not match the target's B,N,H metric dimensions")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("forecast contains non-finite predictions")
    return prediction.detach().cpu().numpy().copy()


class HistoryBaseline(torch.nn.Module):
    """Fixed source-only controls with the same seven-input forecast boundary."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in CONTROL_KINDS:
            raise ValueError("unknown fixed history baseline")
        self.candidate_kind = kind

    def forward(self, flow_hist, x_baseline, y_baseline, time_feat,
                history_mask, edge_index, node_scale):
        if self.candidate_kind == "source_only_pooled_climatology":
            return y_baseline.clone()
        from .shared_graph import canonical_history
        reference = canonical_history(flow_hist, x_baseline, history_mask).reference
        return reference.unsqueeze(-1).expand_as(y_baseline).clone()


class MaskedForecastSums:
    """Float64 observation-level totals, with independent MAPE denominators."""

    def __init__(self, horizon: int = 12) -> None:
        self.horizon = horizon
        self.absolute = np.zeros(horizon, dtype=np.float64)
        self.squared = np.zeros(horizon, dtype=np.float64)
        self.percentage = np.zeros(horizon, dtype=np.float64)
        self.count = np.zeros(horizon, dtype=np.int64)
        self.percentage_count = np.zeros(horizon, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        prediction, target, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
        if prediction.ndim != 3 or prediction.shape != target.shape or mask.shape != target.shape or prediction.shape[-1] != self.horizon:
            raise ValueError("metrics require aligned B,N,H arrays")
        if mask.dtype != np.bool_:
            raise ValueError("target observation mask must be boolean")
        if not np.isfinite(prediction).all() or not np.isfinite(target[mask]).all():
            raise FloatingPointError("metrics received non-finite forecasts or valid targets")
        for horizon in range(self.horizon):
            valid = mask[..., horizon]
            actual = target[..., horizon][valid].astype(np.float64)
            error = np.abs(prediction[..., horizon][valid].astype(np.float64) - actual)
            self.absolute[horizon] += error.sum(dtype=np.float64)
            self.squared[horizon] += np.square(error).sum(dtype=np.float64)
            self.count[horizon] += actual.size
            relative = np.abs(actual) > MAPE_MIN_ABSOLUTE_TARGET
            self.percentage[horizon] += (100 * error[relative] / np.abs(actual[relative])).sum(dtype=np.float64)
            self.percentage_count[horizon] += int(relative.sum())

    def totals(self) -> dict[str, Any]:
        return {"absolute_error_sum": self.absolute.tolist(), "squared_error_sum": self.squared.tolist(),
                "percentage_error_sum": self.percentage.tolist(), "valid_count": self.count.tolist(),
                "percentage_valid_count": self.percentage_count.tolist()}

    def result(self) -> dict[str, Any]:
        count, relative_count = int(self.count.sum()), int(self.percentage_count.sum())
        if count == 0:
            raise ValueError("evaluation has zero valid target observations")
        def divide(numerators, denominators):
            return [float(value / count) if count else None for value, count in zip(numerators, denominators)]
        horizon_mae = divide(self.absolute, self.count)
        horizon_mse = divide(self.squared, self.count)
        return {"global_masked_mae": float(self.absolute.sum() / count),
                "global_masked_rmse": float(np.sqrt(self.squared.sum() / count)),
                "global_masked_mape": float(self.percentage.sum() / relative_count) if relative_count else None,
                "valid_count": count, "percentage_valid_count": relative_count,
                "horizon_mae": horizon_mae,
                "horizon_rmse": [float(np.sqrt(x)) if x is not None else None for x in horizon_mse],
                "horizon_mape": divide(self.percentage, self.percentage_count), "sums": self.totals()}


class _StreamArrayHash:
    def __init__(self, samples: int, value: np.ndarray) -> None:
        self.shape = (samples, *value.shape[1:])
        self.dtype = value.dtype
        self.rows = 0
        self.digest = hashlib.sha256()
        self.digest.update(str(self.dtype).encode())
        self.digest.update(json.dumps(self.shape).encode())

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.dtype != self.dtype or values.shape[1:] != self.shape[1:]:
            raise ValueError("streamed evaluation collection changed shape or dtype")
        self.digest.update(np.ascontiguousarray(values).tobytes())
        self.rows += values.shape[0]

    def hexdigest(self) -> str:
        if self.rows != self.shape[0]:
            raise ValueError("evaluation collection is incomplete")
        return self.digest.hexdigest()


def evaluate_frozen_batches(model: Any, *, indices: np.ndarray,
                            get_batch: Callable[[np.ndarray], Mapping[str, np.ndarray]],
                            times_ns: np.ndarray, device: torch.device | str = "cpu",
                            batch_size: int = 16, seed: int = 42,
                            controls: Mapping[str, Any] | None = None,
                            control_batch_factory: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """Read each target window once, then run all frozen corruption scenarios."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not indices.size or np.unique(indices).size != indices.size:
        raise ValueError("evaluation needs a nonempty unique declared split index")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    times_ns = np.asarray(times_ns)
    if times_ns.ndim != 1 or times_ns.dtype.kind not in "iu":
        raise ValueError("day blocks require the audited real timestamp vector")
    controls = dict(controls or {})
    if controls and set(controls) != set(CONTROL_KINDS):
        raise ValueError("both fixed source-only controls must be evaluated together")
    models = {"shared_graph_ndr": model, **controls}
    for current in models.values():
        current.eval()
    sums = {kind: {scenario: MaskedForecastSums() for scenario in SCENARIOS} for kind in models}
    blocks = {kind: {scenario: {} for scenario in SCENARIOS} for kind in models}
    elapsed = {kind: {scenario: 0.0 for scenario in SCENARIOS} for kind in models}
    removed = {kind: {scenario: 0 for scenario in SCENARIOS} for kind in models}
    streams: dict[str, _StreamArrayHash] = {}
    origins_seen: set[int] = set()
    original_observed = 0
    started = time.perf_counter()
    for start in range(0, len(indices), batch_size):
        batch = get_batch(indices[start:start + batch_size])
        if set(batch) != BATCH_FIELDS:
            raise ValueError("target reader returned a field outside its explicit whitelist")
        origins = np.asarray(batch["sample_start"])
        if origins.shape != (min(batch_size, len(indices) - start),) or (origins < 0).any() or (origins >= len(times_ns)).any():
            raise ValueError("forecast origins do not align with the requested full split")
        if any(int(origin) in origins_seen for origin in origins) or len(set(map(int, origins))) != len(origins):
            raise ValueError("forecast origins are duplicated in the evaluation collection")
        origins_seen.update(map(int, origins))
        days = times_ns[origins].astype("datetime64[ns]").astype("datetime64[D]").astype(str)
        original_observed += int(np.asarray(batch["history_mask"]).sum())
        for key in ("sample_start", "target", "target_mask"):
            if key not in streams:
                streams[key] = _StreamArrayHash(len(indices), np.asarray(batch[key]))
            streams[key].update(np.asarray(batch[key]))
        target_before = np.array(batch["target"], copy=True)
        mask_before = np.array(batch["target_mask"], copy=True)
        control_batch = control_batch_factory(batch) if controls and control_batch_factory else batch
        for key in ("target", "target_mask", "sample_start", "flow_hist", "history_mask"):
            if not np.array_equal(control_batch[key], batch[key], equal_nan=True):
                raise ValueError("control preprocessing changed the paired target or original history collection")
        for kind, current in models.items():
            current_batch = batch if kind == "shared_graph_ndr" else control_batch
            for scenario in SCENARIOS:
                corrupted = perturb_history(current_batch, scenario, seed=seed)
                step_started = time.perf_counter()
                prediction = predict_history_only(current, corrupted, device)
                elapsed[kind][scenario] += time.perf_counter() - step_started
                if not np.array_equal(batch["target"], target_before, equal_nan=True) or not np.array_equal(batch["target_mask"], mask_before):
                    raise RuntimeError("history perturbation or forward mutated held-out targets")
                sums[kind][scenario].update(prediction, target_before, mask_before)
                removed[kind][scenario] += int(np.asarray(batch["history_mask"]).sum() - corrupted["history_mask"].sum())
                for day in np.unique(days):
                    selected = days == day
                    block = blocks[kind][scenario].setdefault(str(day), MaskedForecastSums())
                    block.update(prediction[selected], target_before[selected], mask_before[selected])
    collection = {"samples": len(indices), "forecast_origins_sha256": streams["sample_start"].hexdigest(),
                  "targets_sha256": streams["target"].hexdigest(), "target_masks_sha256": streams["target_mask"].hexdigest(),
                  "hash_format": "sha256(dtype_string+json(full_shape)+C_order_bytes)",
                  "all_declared_split_samples": True, "day_basis": "audited cache times_ns at forecast origin"}
    reports = {}
    for kind in models:
        scenarios = {}
        for scenario in SCENARIOS:
            scenarios[scenario] = {**sums[kind][scenario].result(), "forward_seconds": elapsed[kind][scenario],
                                   "samples_per_forward_second": len(indices) / elapsed[kind][scenario] if elapsed[kind][scenario] else None,
                                   "history_observed_points_original": original_observed,
                                   "history_observed_points_removed": removed[kind][scenario],
                                   "perturbation_seed": seed, "collection": copy.deepcopy(collection),
                                   "day_blocks": {day: value.totals() for day, value in sorted(blocks[kind][scenario].items())}}
        reports[kind] = {"candidate_kind": kind, "scenarios": scenarios,
                         "source_only_control": kind in CONTROL_KINDS}
    return {"collection": collection, "candidate_kind": "shared_graph_ndr",
            "scenarios": reports["shared_graph_ndr"]["scenarios"],
            "controls": {kind: reports[kind] for kind in controls},
            "target_windows_read_once_for_all_models": True, "wallclock_seconds": time.perf_counter() - started,
            "mape_absolute_target_threshold": MAPE_MIN_ABSOLUTE_TARGET,
            "prediction_units": "raw_flow", "forecast_input_fields": list(FORWARD_FIELDS),
            "targets_used_for_forward": False, "test_driven_tuning": False}


def paired_day_block_bootstrap(left: Mapping[str, Any], right: Mapping[str, Any], *,
                               samples: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Resample paired forecast days and recompute point-weighted MAE differences."""
    if left.get("collection") != right.get("collection"):
        raise ValueError("paired day statistics require identical origins, targets and masks")
    first, second = left.get("day_blocks", {}), right.get("day_blocks", {})
    if not first or set(first) != set(second):
        raise ValueError("paired forecasts do not share the same real calendar days")
    days = sorted(first)
    if len(days) < 2 or type(samples) is not int or samples < 100:
        raise ValueError("paired uncertainty needs at least two days and 100 resamples")
    counts = np.asarray([sum(first[day]["valid_count"]) for day in days], dtype=np.float64)
    for day in days:
        if first[day]["valid_count"] != second[day]["valid_count"]:
            raise ValueError("paired days have different observation denominators")
    if not (counts >= 0).all() or counts.sum() == 0:
        raise ValueError("paired day blocks contain no valid observations")
    difference = np.asarray([sum(first[day]["absolute_error_sum"]) - sum(second[day]["absolute_error_sum"])
                             for day in days], dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        draw = rng.integers(0, len(days), size=len(days))
        denominator = counts[draw].sum()
        if denominator:
            differences.append(float(difference[draw].sum() / denominator))
    if not differences:
        raise ValueError("bootstrap produced no observed resampled blocks")
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"metric": "left_minus_right_global_masked_mae", "estimate": float(difference.sum() / counts.sum()),
            "ci95": [float(low), float(high)], "forecast_day_blocks": len(days), "bootstrap_samples": samples,
            "valid_resamples": len(differences), "seed": seed, "paired": True,
            "resampling_unit": "actual calendar day of forecast origin", "claim_scope": "descriptive_paired_day_block_uncertainty"}
