#!/usr/bin/env python3
"""Fail-closed Windows runner for the XTraffic protocol-v3 45-task matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence, TextIO

import numpy as np


PROTOCOL_VERSION = "protocol_v3_noleak_dedup"
CACHE_SCHEMA_VERSION = 3
DEFAULT_CACHE_DIR = Path("outputs/dist_net/region_data_v3_noleak_dedup")
DEFAULT_RESULTS_ROOT = Path(
    "outputs/protocol_v3_noleak_dedup/remote_runs/minimal_45_5080"
)
DEFAULT_GRAPH_DIR = Path("outputs/region_graphs")
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,127}$")
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
REGION_GRAPH_KEYS = {
    "Alameda": "alameda",
    "ContraCosta": "contra_costa",
    "Orange": "orange",
}
FLOAT32_PRODUCER_METRIC_KINDS = frozenset(
    {
        "graphwavenet",
        "fourier_dual_net",
        "staeformer",
        "staeformer_icsf",
    }
)


class ManifestError(ValueError):
    """Raised when a queue manifest violates the v3 contract."""


class ResultValidationError(RuntimeError):
    """Raised when a trainer result does not satisfy its task contract."""


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    region: str
    model: str
    variant: str
    seed: int
    sampling_protocol: str
    output_dir: Path


@dataclass(frozen=True)
class Manifest:
    manifest_version: int
    protocol_version: str
    cache_schema_version: int
    default_results_root: Path
    default_cache_dir: Path
    default_graph_dir: Path
    training: Mapping[str, object]
    tasks: tuple[Task, ...]
    source_path: Path
    digest: str


@dataclass(frozen=True)
class ProtocolV3Inputs:
    results_root: Path
    cache_dir: Path
    graph_dir: Path
    cache_bundle_id: str
    cache_bundle_digest: str
    code_sha: str
    python_executable: str


@dataclass(frozen=True)
class Invocation:
    command: tuple[str, ...]
    result_relpath: Path
    required_artifacts: tuple[str, ...]
    required_metrics: tuple[str, ...]
    config: Mapping[str, object]


CommandBuilder = Callable[
    [Task, Manifest, ProtocolV3Inputs, Path],
    Invocation,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a nonempty string")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        raise ManifestError(f"{field} must be relative: {value!r}")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise ManifestError(f"{field} contains an unsafe path component: {value!r}")
    return Path(*pure.parts)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest must be a JSON object: {path}")
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _read_strict_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON document must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_cache_bundle_release(
    run_inputs: ProtocolV3Inputs,
) -> dict[str, object]:
    cache_input = run_inputs.cache_dir
    graph_input = run_inputs.graph_dir
    if cache_input.is_symlink() or not cache_input.is_dir():
        raise ManifestError(
            f"formal cache directory is missing or a symlink: {cache_input}"
        )
    if graph_input.is_symlink() or not graph_input.is_dir():
        raise ManifestError(
            f"formal graph directory is missing or a symlink: {graph_input}"
        )
    cache_dir = cache_input.resolve()
    graph_dir = graph_input.resolve()
    manifest_path = cache_dir / "cache_bundle_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestError(
            f"verified cache bundle manifest is missing: {manifest_path}"
        )
    payload = _read_strict_json_object(manifest_path)
    expected_identity = {
        "status": "verified",
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
    }
    for field, expected in expected_identity.items():
        actual = payload.get(field)
        if actual != expected:
            raise ManifestError(
                f"cache release {field} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )

    regions = payload.get("regions")
    expected_regions = set(REGION_GRAPH_KEYS)
    if not isinstance(regions, dict) or set(regions) != expected_regions:
        actual_regions = sorted(regions) if isinstance(regions, dict) else None
        raise ManifestError(
            "cache release must contain exactly Alameda, ContraCosta, "
            f"and Orange; got {actual_regions!r}"
        )

    region_digests: dict[str, str] = {}
    expected_roles = {"traffic", "samples", "relation", "graph"}
    for region, graph_key in REGION_GRAPH_KEYS.items():
        report = regions.get(region)
        if not isinstance(report, dict):
            raise ManifestError(
                f"cache release region report is invalid for {region}"
            )
        region_digest = report.get("cache_bundle_digest")
        if (
            not isinstance(region_digest, str)
            or not HEX_64_RE.fullmatch(region_digest)
        ):
            raise ManifestError(
                f"cache release region digest is invalid for {region}"
            )
        region_digests[region] = region_digest
        provenance = report.get("cache_provenance")
        artifact_hashes = (
            provenance.get("artifact_sha256")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(provenance, dict):
            raise ManifestError(
                f"cache release provenance is missing for {region}"
            )
        if provenance.get("bundle_id") != run_inputs.cache_bundle_id:
            raise ManifestError(
                f"cache release provenance bundle_id mismatch for {region}"
            )
        cache_ids: dict[str, str] = {}
        for field in (
            "traffic_cache_id",
            "samples_cache_id",
            "relation_cache_id",
        ):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(
                    f"cache release provenance {field} is invalid for {region}"
                )
            cache_ids[field] = value
        if (
            not isinstance(artifact_hashes, dict)
            or not expected_roles.issubset(artifact_hashes)
        ):
            raise ManifestError(
                f"cache release artifact hashes are incomplete for {region}"
            )
        artifact_paths = {
            "traffic": cache_dir / f"{region}_traffic.h5",
            "samples": cache_dir / f"{region}_samples.h5",
            "relation": cache_dir / f"{region}_event_rel_feat.npz",
            "graph": graph_dir / f"{graph_key}_sparse_adj.npz",
        }
        for role, path in artifact_paths.items():
            expected_hash = artifact_hashes.get(role)
            if (
                not isinstance(expected_hash, str)
                or not HEX_64_RE.fullmatch(expected_hash)
            ):
                raise ManifestError(
                    f"cache release {region} {role} hash is invalid"
                )
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ManifestError(
                    f"cache release {region} {role} artifact is missing "
                    f"or empty: {path}"
                )
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise ManifestError(
                    f"cache release {region} {role} artifact hash mismatch: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
        recomputed_region_digest = _sha256_json(
            {
                "protocol_version": PROTOCOL_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "bundle_id": run_inputs.cache_bundle_id,
                **cache_ids,
                "artifact_sha256": artifact_hashes,
            }
        )
        if recomputed_region_digest != region_digest:
            raise ManifestError(
                f"cache release {region} provenance digest mismatch: "
                f"expected {region_digest}, "
                f"recomputed {recomputed_region_digest}"
            )

    digest_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "bundle_id": run_inputs.cache_bundle_id,
        "regions": {
            region: region_digests[region]
            for region in sorted(region_digests)
        },
    }
    recomputed_digest = _sha256_json(digest_payload)
    if recomputed_digest != run_inputs.cache_bundle_digest:
        raise ManifestError(
            "cache release aggregate digest mismatch: "
            f"expected {run_inputs.cache_bundle_digest}, "
            f"recomputed {recomputed_digest}"
        )
    return payload


def load_manifest(
    path: Path | str,
    *,
    require_production_matrix: bool = True,
) -> Manifest:
    source_path = Path(path).resolve()
    payload = _read_json_object(source_path)
    if payload.get("manifest_version") != 1:
        raise ManifestError("manifest_version must equal 1")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ManifestError(
            f"protocol_version must equal {PROTOCOL_VERSION!r}"
        )
    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ManifestError(
            f"cache_schema_version must equal {CACHE_SCHEMA_VERSION}"
        )

    training = payload.get("training")
    if not isinstance(training, dict):
        raise ManifestError("training must be a JSON object")
    for key in ("epochs", "patience", "device"):
        if key not in training:
            raise ManifestError(f"training.{key} is required")
    if not isinstance(training["epochs"], int) or training["epochs"] <= 0:
        raise ManifestError("training.epochs must be a positive integer")
    if not isinstance(training["patience"], int) or training["patience"] < 0:
        raise ManifestError("training.patience must be a nonnegative integer")
    if not isinstance(training["device"], str) or not training["device"]:
        raise ManifestError("training.device must be a nonempty string")

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ManifestError("tasks must be a nonempty JSON array")
    tasks: list[Task] = []
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    valid_regions = {"Alameda", "ContraCosta", "Orange"}
    valid_kinds = {
        "staeformer_deseason",
        "staeformer_icsf",
        "graphwavenet",
        "fourier_dual_net",
        "staeformer",
    }
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise ManifestError(f"tasks[{index}] must be a JSON object")
        missing = {
            "task_id",
            "kind",
            "region",
            "model",
            "variant",
            "seed",
            "sampling_protocol",
            "output_dir",
        } - raw.keys()
        if missing:
            raise ManifestError(
                f"tasks[{index}] missing fields: {sorted(missing)}"
            )
        task_id = raw["task_id"]
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise ManifestError(f"invalid task_id at tasks[{index}]: {task_id!r}")
        task_id_key = task_id.casefold()
        if task_id_key in seen_ids:
            raise ManifestError(f"duplicate task_id: {task_id}")
        seen_ids.add(task_id_key)
        kind = raw["kind"]
        if kind not in valid_kinds:
            raise ManifestError(f"invalid task kind for {task_id}: {kind!r}")
        region = raw["region"]
        if region not in valid_regions:
            raise ManifestError(f"invalid region for {task_id}: {region!r}")
        model = raw["model"]
        variant = raw["variant"]
        if not isinstance(model, str) or not model:
            raise ManifestError(f"invalid model for {task_id}")
        if not isinstance(variant, str) or not variant:
            raise ManifestError(f"invalid variant for {task_id}")
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestError(f"invalid seed for {task_id}: {seed!r}")
        sampling_protocol = raw["sampling_protocol"]
        if sampling_protocol not in {"event", "full_window"}:
            raise ManifestError(
                f"invalid sampling_protocol for {task_id}: "
                f"{sampling_protocol!r}"
            )
        output_dir = _relative_path(raw["output_dir"], f"{task_id}.output_dir")
        if output_dir.parts != ("tasks", task_id):
            raise ManifestError(
                f"output_dir for {task_id} must equal tasks/{task_id}"
            )
        output_key = output_dir.as_posix().casefold()
        if output_key in seen_outputs:
            raise ManifestError(f"duplicate output path: {output_dir.as_posix()}")
        seen_outputs.add(output_key)
        tasks.append(
            Task(
                task_id=task_id,
                kind=str(kind),
                region=str(region),
                model=model,
                variant=variant,
                seed=seed,
                sampling_protocol=str(sampling_protocol),
                output_dir=output_dir,
            )
        )

    manifest = Manifest(
        manifest_version=1,
        protocol_version=str(payload["protocol_version"]),
        cache_schema_version=int(payload["cache_schema_version"]),
        default_results_root=_relative_path(
            payload.get("default_results_root"),
            "default_results_root",
        ),
        default_cache_dir=_relative_path(
            payload.get("default_cache_dir"),
            "default_cache_dir",
        ),
        default_graph_dir=_relative_path(
            payload.get("default_graph_dir"),
            "default_graph_dir",
        ),
        training=dict(training),
        tasks=tuple(tasks),
        source_path=source_path,
        digest=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    if require_production_matrix:
        _validate_production_matrix(manifest)
    return manifest


def _validate_production_matrix(manifest: Manifest) -> None:
    if manifest.default_cache_dir != DEFAULT_CACHE_DIR:
        raise ManifestError(
            f"production default_cache_dir must equal {DEFAULT_CACHE_DIR}"
        )
    if manifest.default_results_root != DEFAULT_RESULTS_ROOT:
        raise ManifestError(
            f"production default_results_root must equal {DEFAULT_RESULTS_ROOT}"
        )
    if manifest.default_graph_dir != DEFAULT_GRAPH_DIR:
        raise ManifestError(
            f"production default_graph_dir must equal {DEFAULT_GRAPH_DIR}"
        )

    expected: set[tuple[str, str, str, str, int, str]] = set()
    for region in ("Alameda", "ContraCosta", "Orange"):
        for variant in ("s0a", "s0c"):
            for seed in (42, 1, 2):
                expected.add(
                    (
                        "staeformer_deseason",
                        region,
                        "staeformer_deseason",
                        variant,
                        seed,
                        "event",
                    )
                )
    for region in ("Alameda", "ContraCosta"):
        for variant in ("s0f", "s0g"):
            for seed in (42, 1, 2):
                expected.add(
                    (
                        "staeformer_deseason",
                        region,
                        "staeformer_deseason",
                        variant,
                        seed,
                        "event",
                    )
                )
    for region in ("Alameda", "ContraCosta", "Orange"):
        for variant in ("base", "icsf"):
            expected.add(
                (
                    "staeformer_icsf",
                    region,
                    "staeformer_icsf",
                    variant,
                    42,
                    "event",
                )
            )
        expected.add(
            (
                "graphwavenet",
                region,
                "graphwavenet",
                "base",
                42,
                "full_window",
            )
        )
        expected.add(
            (
                "fourier_dual_net",
                region,
                "fourier_dual_net",
                "learnable_k3",
                42,
                "full_window",
            )
        )
        expected.add(
            (
                "staeformer",
                region,
                "staeformer",
                "base",
                42,
                "full_window",
            )
        )

    actual = {
        (
            task.kind,
            task.region,
            task.model,
            task.variant,
            task.seed,
            task.sampling_protocol,
        )
        for task in manifest.tasks
    }
    if len(manifest.tasks) != 45 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ManifestError(
            "production matrix must contain exactly the canonical 45 tasks; "
            f"missing={missing}, extra={extra}, count={len(manifest.tasks)}"
        )


def _production_training_config(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
) -> dict[str, object]:
    common = {
        "kind": task.kind,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "sampling_protocol": task.sampling_protocol,
        "epochs": int(manifest.training["epochs"]),
        "device": str(manifest.training["device"]),
        "cache_dir": str(run_inputs.cache_dir.resolve()),
        "graph_dir": str(run_inputs.graph_dir.resolve()),
    }
    if task.kind != "staeformer_deseason":
        common["patience"] = int(manifest.training["patience"])
    if task.sampling_protocol == "full_window":
        common.update({"stride": 1, "train_frac": 0.7, "val_frac": 0.1})
    if task.kind == "fourier_dual_net":
        common.update({"decomp_mode": "learnable", "K": 3})
    if task.kind == "staeformer_icsf":
        common["use_icsf"] = task.variant == "icsf"
    return common


def derive_config_hash(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    *,
    invocation_config: Mapping[str, object] | None = None,
) -> str:
    config = (
        dict(invocation_config)
        if invocation_config is not None
        else _production_training_config(task, manifest, run_inputs)
    )
    payload = {
        "contract": {
            "protocol_version": manifest.protocol_version,
            "cache_schema_version": manifest.cache_schema_version,
            "cache_bundle_id": run_inputs.cache_bundle_id,
            "cache_bundle_digest": run_inputs.cache_bundle_digest,
            "code_sha": run_inputs.code_sha,
        },
        "task": {
            "task_id": task.task_id,
            "region": task.region,
            "model": task.model,
            "variant": task.variant,
            "seed": task.seed,
            "sampling_protocol": task.sampling_protocol,
        },
        "training": config,
    }
    return _sha256_json(payload)


def build_production_invocation(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    attempt_dir: Path,
) -> Invocation:
    work = attempt_dir / "work"
    py = run_inputs.python_executable
    epochs = str(manifest.training["epochs"])
    patience = str(manifest.training["patience"])
    device = str(manifest.training["device"])
    cache = str(run_inputs.cache_dir)
    graph = str(run_inputs.graph_dir)
    command: list[str] = [py, "-X", "utf8"]

    if task.kind == "staeformer_deseason":
        command += [
            "scripts/train_staeformer_deseason.py",
            "--region",
            task.region,
            "--variant",
            task.variant,
            "--seed",
            str(task.seed),
            "--epochs",
            epochs,
            "--device",
            device,
            "--data_dir",
            cache,
            "--graph_dir",
            graph,
            "--out_dir",
            str(work),
        ]
        result_relpath = Path("work") / task.region / (
            f"{task.variant}_seed{task.seed}"
        )
        artifacts = ["summary.json", "ckpt_best.pt", "test_predictions.npz"]
        if task.variant in {"s0f", "s0g"}:
            artifacts.append("gate_diag.npz")
        required_metrics = ("all", "affected", "unaffected")
    elif task.kind == "staeformer_icsf":
        command += [
            "scripts/train_staeformer_icsf.py",
            "--region",
            task.region,
            "--seed",
            str(task.seed),
            "--epochs",
            epochs,
            "--patience",
            patience,
            "--device",
            device,
            "--data_dir",
            cache,
            "--graph_dir",
            graph,
            "--out_dir",
            str(work),
        ]
        if task.variant == "icsf":
            command.append("--use_icsf")
        result_relpath = Path("work") / task.region / task.variant
        artifacts = ["summary.json", "ckpt_best.pt", "test_predictions.npz"]
        required_metrics = ("all", "affected", "unaffected")
    elif task.kind == "graphwavenet":
        command += [
            "scripts/train_graphwavenet.py",
            "--region",
            task.region,
            "--seed",
            str(task.seed),
            "--epochs",
            epochs,
            "--patience",
            patience,
            "--device",
            device,
            "--data-dir",
            cache,
            "--graph-dir",
            graph,
            "--out-dir",
            str(work),
            "--protocol",
            "full_window",
            "--stride",
            "1",
            "--train_frac",
            "0.7",
            "--val_frac",
            "0.1",
        ]
        result_relpath = Path("work_fullwindow") / task.region
        artifacts = [
            "summary.json",
            "ckpt_best.pt",
            "test_predictions.npz",
            "train.log",
        ]
        required_metrics = ("all",)
    elif task.kind == "fourier_dual_net":
        command += [
            "scripts/train_fourier_dual_net.py",
            "--region",
            task.region,
            "--seed",
            str(task.seed),
            "--epochs",
            epochs,
            "--patience",
            patience,
            "--device",
            device,
            "--data_dir",
            cache,
            "--graph_dir",
            graph,
            "--out_dir",
            str(work),
            "--protocol",
            "full_window",
            "--stride",
            "1",
            "--train_frac",
            "0.7",
            "--val_frac",
            "0.1",
            "--decomp_mode",
            "learnable",
            "--K",
            "3",
        ]
        result_relpath = (
            Path("work") / "learnable_K3_fullwindow" / task.region
        )
        artifacts = [
            "summary.json",
            "ckpt_best.pt",
            "test_predictions.npz",
            "train.log",
            "decomp_mask.json",
        ]
        required_metrics = ("all",)
    elif task.kind == "staeformer":
        command += [
            "scripts/train_staeformer_xtraffic.py",
            "--region",
            task.region,
            "--seed",
            str(task.seed),
            "--epochs",
            epochs,
            "--patience",
            patience,
            "--device",
            device,
            "--data_dir",
            cache,
            "--graph_dir",
            graph,
            "--out_dir",
            str(work),
            "--protocol",
            "full_window",
            "--stride",
            "1",
            "--train_frac",
            "0.7",
            "--val_frac",
            "0.1",
        ]
        result_relpath = Path("work_fullwindow") / task.region
        artifacts = ["summary.json", "ckpt_best.pt", "test_predictions.npz"]
        required_metrics = ("all",)
    else:
        raise ManifestError(f"unsupported task kind: {task.kind}")

    return Invocation(
        command=tuple(command),
        result_relpath=result_relpath,
        required_artifacts=tuple(artifacts),
        required_metrics=tuple(required_metrics),
        config=_production_training_config(task, manifest, run_inputs),
    )


def _validate_run_inputs(run_inputs: ProtocolV3Inputs) -> None:
    if not run_inputs.cache_bundle_id.strip():
        raise ManifestError("cache_bundle_id must be nonempty")
    if not HEX_64_RE.fullmatch(run_inputs.cache_bundle_digest.lower()):
        raise ManifestError("cache_bundle_digest must be a 64-character SHA-256")
    if not HEX_40_RE.fullmatch(run_inputs.code_sha.lower()):
        raise ManifestError("code_sha must be a full 40-character Git SHA")
    if not run_inputs.python_executable:
        raise ManifestError("python_executable must be nonempty")


def _safe_artifact_path(result_dir: Path, relative: str) -> Path:
    rel = _relative_path(relative, "required artifact")
    path = result_dir / rel
    try:
        path.resolve().relative_to(result_dir.resolve())
    except ValueError as exc:
        raise ResultValidationError(
            f"artifact escapes result directory: {relative}"
        ) from exc
    return path


def _summary_metric(summary: Mapping[str, object], name: str) -> object:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return metrics[name]
    return summary.get(name)


def _binary_mask(
    value: np.ndarray,
    *,
    field: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if value.shape != expected_shape:
        raise ResultValidationError(
            f"prediction artifact {field} shape mismatch: "
            f"expected {expected_shape}, got {value.shape}"
        )
    if value.dtype == np.bool_:
        return value
    if not np.issubdtype(value.dtype, np.number):
        raise ResultValidationError(
            f"prediction artifact {field} must be boolean or binary numeric"
        )
    if not np.isfinite(value).all() or not np.isin(value, (0, 1)).all():
        raise ResultValidationError(
            f"prediction artifact {field} must contain only zero and one"
        )
    return value.astype(np.bool_, copy=False)


def _require_prediction_metric(
    summary: Mapping[str, object],
    name: str,
    producer_recomputed: float,
    *,
    audit_recomputed: float | None = None,
    valid_count: int = 1,
    producer_dtype: np.dtype[object] = np.dtype(np.float64),
    allow_producer_fallback: bool = False,
) -> None:
    reported = _summary_metric(summary, name)
    audit_value = (
        producer_recomputed
        if audit_recomputed is None
        else audit_recomputed
    )
    dtype = np.dtype(producer_dtype)
    if (
        isinstance(reported, bool)
        or not isinstance(reported, (int, float))
        or not math.isfinite(float(reported))
        or not math.isfinite(producer_recomputed)
        or not math.isfinite(audit_value)
    ):
        raise ResultValidationError(
            f"prediction metric {name!r} mismatch: "
            f"summary={reported!r}, "
            f"producer_recomputed={producer_recomputed}, "
            f"audit_recomputed={audit_value}, "
            "tolerance=finite numeric values required"
        )

    reported_value = float(reported)
    if math.isclose(
        reported_value,
        audit_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        return

    if not allow_producer_fallback or dtype != np.dtype(np.float32):
        tolerance = max(1e-6, 1e-6 * abs(audit_value))
        raise ResultValidationError(
            f"prediction metric {name!r} mismatch: "
            f"summary={reported!r}, "
            f"producer_recomputed={producer_recomputed}, "
            f"audit_recomputed={audit_value}, "
            f"tolerance={tolerance}"
        )

    producer_ulp = abs(
        float(np.spacing(np.asarray(producer_recomputed, dtype=dtype)))
    )
    summary_tolerance = 2.0 * producer_ulp
    epsilon = float(np.finfo(dtype).eps)
    reduction_levels = math.ceil(math.log2(max(valid_count, 1))) + 1
    accumulated_error = reduction_levels * epsilon
    if accumulated_error >= 1.0:
        audit_tolerance = math.inf
    else:
        audit_tolerance = (
            max(1e-6, 1e-6 * abs(audit_value))
            + accumulated_error
            / (1.0 - accumulated_error)
            * abs(producer_recomputed)
        )
    if (
        abs(reported_value - producer_recomputed) > summary_tolerance
        or abs(producer_recomputed - audit_value) > audit_tolerance
    ):
        raise ResultValidationError(
            f"prediction metric {name!r} mismatch: "
            f"summary={reported!r}, "
            f"producer_recomputed={producer_recomputed}, "
            f"audit_recomputed={audit_value}, "
            f"tolerance={summary_tolerance}, "
            f"audit_tolerance={audit_tolerance}"
        )


def _validate_prediction_artifact(
    path: Path,
    task: Task,
    summary: Mapping[str, object],
    *,
    required_metrics: Sequence[str],
) -> None:
    required_fields = {
        "region_code",
        "sample_start",
        "region_node_idx",
        "pred_raw_flow",
        "actual_future_flow",
        "y_mask_flow",
        "affected_mask",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required_fields.difference(archive.files))
            if missing:
                raise ResultValidationError(
                    "prediction artifact is missing fields: "
                    + ", ".join(missing)
                )
            arrays = {
                field: archive[field]
                for field in required_fields
            }
    except ResultValidationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise ResultValidationError(
            f"prediction artifact cannot be read safely: {exc}"
        ) from exc

    prediction = arrays["pred_raw_flow"]
    actual = arrays["actual_future_flow"]
    if (
        prediction.ndim != 3
        or prediction.shape != actual.shape
        or any(size <= 0 for size in prediction.shape)
    ):
        raise ResultValidationError(
            "prediction artifact pred_raw_flow and actual_future_flow "
            f"must share a nonempty (S, T_p, N) shape; got "
            f"{prediction.shape} and {actual.shape}"
        )
    if not (
        np.issubdtype(prediction.dtype, np.number)
        and np.issubdtype(actual.dtype, np.number)
    ):
        raise ResultValidationError(
            "prediction artifact flow arrays must be numeric"
        )
    sample_count, horizon, num_nodes = prediction.shape
    valid = _binary_mask(
        arrays["y_mask_flow"],
        field="y_mask_flow",
        expected_shape=prediction.shape,
    )
    affected = _binary_mask(
        arrays["affected_mask"],
        field="affected_mask",
        expected_shape=(sample_count, num_nodes),
    )

    sample_start = arrays["sample_start"]
    if (
        sample_start.shape != (sample_count,)
        or not np.issubdtype(sample_start.dtype, np.integer)
        or np.unique(sample_start).size != sample_count
        or np.any(np.diff(sample_start.astype(np.int64, copy=False)) <= 0)
    ):
        raise ResultValidationError(
            "prediction artifact sample_start must be a strictly increasing "
            "unique integer vector"
        )
    region_node_idx = arrays["region_node_idx"]
    if (
        region_node_idx.shape != (num_nodes,)
        or not np.issubdtype(region_node_idx.dtype, np.integer)
        or np.unique(region_node_idx).size != num_nodes
    ):
        raise ResultValidationError(
            "prediction artifact region_node_idx must be a unique integer "
            "vector with one entry per node"
        )
    expected_region_code = {
        "Alameda": 0,
        "ContraCosta": 1,
        "Orange": 2,
    }[task.region]
    region_code = arrays["region_code"]
    if (
        region_code.shape != (sample_count,)
        or not np.issubdtype(region_code.dtype, np.integer)
        or not np.all(region_code == expected_region_code)
    ):
        raise ResultValidationError(
            f"prediction artifact region_code does not identify {task.region}"
        )

    recomputed_valid_count = int(valid.sum())
    reported_valid_count = summary.get("valid_count")
    if reported_valid_count != recomputed_valid_count:
        raise ResultValidationError(
            "prediction valid_count mismatch: "
            f"summary={reported_valid_count!r}, "
            f"recomputed={recomputed_valid_count}"
        )
    if recomputed_valid_count <= 0:
        raise ResultValidationError(
            "prediction artifact contains no valid observations"
        )

    affected_3d = np.broadcast_to(
        affected[:, None, :],
        (sample_count, horizon, num_nodes),
    )
    error_sums = {"all": 0.0, "affected": 0.0, "unaffected": 0.0}
    counts = {"all": 0, "affected": 0, "unaffected": 0}
    chunk_size = 64
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        chunk_valid = valid[start:stop]
        chunk_prediction = prediction[start:stop]
        chunk_actual = actual[start:stop]
        if not np.isfinite(chunk_prediction).all():
            raise ResultValidationError(
                "prediction artifact contains a non-finite prediction"
            )
        if not np.isfinite(chunk_actual[chunk_valid]).all():
            raise ResultValidationError(
                "prediction artifact contains non-finite valid target values"
            )
        chunk_error = np.abs(
            chunk_prediction[chunk_valid].astype(np.float64)
            - chunk_actual[chunk_valid].astype(np.float64)
        )
        error_sums["all"] += float(chunk_error.sum())
        counts["all"] += int(chunk_error.size)
        chunk_affected = affected_3d[start:stop]
        for name, group in (
            ("affected", chunk_affected),
            ("unaffected", ~chunk_affected),
        ):
            selected = chunk_valid & group
            group_error = np.abs(
                chunk_prediction[selected].astype(np.float64)
                - chunk_actual[selected].astype(np.float64)
            )
            error_sums[name] += float(group_error.sum())
            counts[name] += int(group_error.size)

    allow_producer_fallback = (
        task.kind in FLOAT32_PRODUCER_METRIC_KINDS
        and prediction.dtype == np.dtype(np.float32)
        and actual.dtype == np.dtype(np.float32)
    )
    producer_error: np.ndarray | None = None
    if allow_producer_fallback:
        with np.errstate(invalid="ignore"):
            producer_error = np.abs(prediction - actual)

    def require_group_metric(name: str, selected: np.ndarray) -> None:
        audit_recomputed = error_sums[name] / counts[name]
        producer_recomputed = audit_recomputed
        producer_dtype = np.dtype(np.float64)
        if producer_error is not None:
            producer_recomputed = float(producer_error[selected].mean())
            producer_dtype = producer_error.dtype
        _require_prediction_metric(
            summary,
            name,
            producer_recomputed,
            audit_recomputed=audit_recomputed,
            valid_count=counts[name],
            producer_dtype=producer_dtype,
            allow_producer_fallback=allow_producer_fallback,
        )

    require_group_metric("all", valid)
    if task.sampling_protocol == "event":
        for name in ("affected", "unaffected"):
            if name not in required_metrics:
                continue
            if counts[name] <= 0:
                raise ResultValidationError(
                    f"prediction artifact has no valid {name} observations"
                )
            require_group_metric(
                name,
                valid
                & (
                    affected_3d
                    if name == "affected"
                    else ~affected_3d
                ),
            )


def validate_result(
    result_dir: Path,
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    *,
    expected_config_hash: str,
    required_artifacts: Sequence[str],
    required_metrics: Sequence[str],
) -> dict[str, object]:
    if not result_dir.is_dir():
        raise ResultValidationError(f"result directory is missing: {result_dir}")
    summary_path = result_dir / "summary.json"
    try:
        summary = _read_strict_json_object(summary_path)
    except ManifestError as exc:
        raise ResultValidationError(f"invalid summary.json: {exc}") from exc

    expected = {
        "task_id": task.task_id,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "protocol_version": manifest.protocol_version,
        "sampling_protocol": task.sampling_protocol,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "config_hash": expected_config_hash,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ResultValidationError(
                f"summary {key} mismatch: expected {value!r}, "
                f"got {summary.get(key)!r}"
            )

    valid_count = summary.get("valid_count")
    if (
        isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or valid_count <= 0
    ):
        raise ResultValidationError(
            f"summary valid_count must be a positive integer, got {valid_count!r}"
        )
    for metric in required_metrics:
        value = _summary_metric(summary, metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ResultValidationError(
                f"summary metric {metric!r} must be finite, got {value!r}"
            )
        if float(value) < 0:
            raise ResultValidationError(
                f"summary metric {metric!r} must be nonnegative, got {value!r}"
            )

    for relative in required_artifacts:
        artifact = _safe_artifact_path(result_dir, relative)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ResultValidationError(
                f"required artifact missing or empty: {relative}"
            )
    if "test_predictions.npz" in required_artifacts:
        _validate_prediction_artifact(
            result_dir / "test_predictions.npz",
            task,
            summary,
            required_metrics=required_metrics,
        )
    return summary


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _directory_nonempty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _validate_results_root_layout(
    results_root: Path,
    manifest: Manifest,
) -> None:
    allowed_top_level = {
        "queue_status.json",
        "tasks",
        "logs",
        ".staging",
        "state",
    }
    unexpected = sorted(
        child.name
        for child in results_root.iterdir()
        if child.name not in allowed_top_level
    )
    if unexpected:
        raise ResultValidationError(
            "results root contains unowned content: " + ", ".join(unexpected)
        )
    for name in ("tasks", "logs", ".staging", "state"):
        path = results_root / name
        if path.exists() and not path.is_dir():
            raise ResultValidationError(
                f"runner control path must be a directory: {path}"
            )
    queue_status = results_root / "queue_status.json"
    if queue_status.exists() and not queue_status.is_file():
        raise ResultValidationError(
            f"queue status path must be a file: {queue_status}"
        )
    task_root = results_root / "tasks"
    if task_root.is_dir():
        allowed_tasks = {task.task_id for task in manifest.tasks}
        unknown_tasks = sorted(
            child.name
            for child in task_root.iterdir()
            if child.name not in allowed_tasks
        )
        if unknown_tasks:
            raise ResultValidationError(
                "results root contains outputs for unknown tasks: "
                + ", ".join(unknown_tasks)
            )


def _next_attempt(results_root: Path, task_id: str) -> int:
    numbers: set[int] = set()
    for parent in (
        results_root / "logs" / task_id,
        results_root / ".staging" / task_id,
    ):
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            match = re.fullmatch(r"attempt_(\d{4})(?:\.log)?", child.name)
            if match:
                numbers.add(int(match.group(1)))
    return max(numbers, default=0) + 1


def _require_recovery_path(
    value: object,
    expected: Path,
    *,
    field: str,
) -> None:
    if not isinstance(value, str):
        raise ResultValidationError(
            f"recovery {field} must be an absolute path string"
        )
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ResultValidationError(
            f"recovery {field} must be absolute: {value!r}"
        )
    try:
        candidate_resolved = candidate.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise ResultValidationError(
            f"recovery {field} cannot be resolved: {exc}"
        ) from exc
    if candidate_resolved != expected_resolved:
        raise ResultValidationError(
            f"recovery {field} mismatch: "
            f"expected {expected}, got {candidate}"
        )


def _validate_recovery_queue_state(
    payload: Mapping[str, object] | None,
    *,
    task: Task,
    task_index: int,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
) -> None:
    if payload is None:
        raise ResultValidationError(
            "staging recovery requires the prior queue_status.json"
        )
    expected = {
        "state": "failed",
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "manifest_digest": manifest.digest,
        "total_tasks": len(manifest.tasks),
        "current_task": task.task_id,
        "completed_tasks": task_index,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ResultValidationError(
                f"prior queue {key} mismatch: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )
    skipped = payload.get("skipped_valid_tasks")
    if (
        isinstance(skipped, bool)
        or not isinstance(skipped, int)
        or skipped < 0
        or skipped > task_index
    ):
        raise ResultValidationError(
            "prior queue skipped_valid_tasks is invalid"
        )
    error = payload.get("error")
    if (
        not isinstance(error, str)
        or not error.startswith("result validation or promotion failed:")
    ):
        raise ResultValidationError(
            "prior queue error is not a result validation or promotion failure"
        )


def _recover_staged_result(
    *,
    results_root: Path,
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    invocation_probe: Invocation,
    config_hash: str,
    final_dir: Path,
    task_status_path: Path,
    command_builder: CommandBuilder,
) -> dict[str, object]:
    next_attempt = _next_attempt(results_root, task.task_id)
    attempt = next_attempt - 1
    if attempt <= 0:
        raise ResultValidationError(
            f"no staging attempt exists for {task.task_id}"
        )
    attempt_name = f"attempt_{attempt:04d}"
    attempt_dir = (
        results_root / ".staging" / task.task_id / attempt_name
    )
    log_path = (
        results_root
        / "logs"
        / task.task_id
        / f"{attempt_name}.log"
    )
    if (
        attempt_dir.is_symlink()
        or not attempt_dir.is_dir()
        or log_path.is_symlink()
        or not log_path.is_file()
        or log_path.stat().st_size <= 0
    ):
        raise ResultValidationError(
            "latest recovery attempt or its log is missing, empty, or a link"
        )
    try:
        attempt_dir.resolve(strict=True).relative_to(
            (results_root / ".staging" / task.task_id).resolve(strict=True)
        )
        log_path.resolve(strict=True).relative_to(
            (results_root / "logs" / task.task_id).resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise ResultValidationError(
            "latest recovery attempt escapes its canonical task directory"
        ) from exc

    try:
        failed_state = _read_strict_json_object(task_status_path)
    except ManifestError as exc:
        raise ResultValidationError(
            f"cannot read failed task state strictly: {exc}"
        ) from exc
    failed_state_sha256 = _sha256_file(task_status_path)
    state_attempt = failed_state.get("attempt")
    state_returncode = failed_state.get("returncode")
    if failed_state.get("state") != "failed":
        raise ResultValidationError("recovery task state is not failed")
    if failed_state.get("task_id") != task.task_id:
        raise ResultValidationError("recovery task state task_id mismatch")
    if (
        isinstance(state_attempt, bool)
        or not isinstance(state_attempt, int)
        or state_attempt != attempt
    ):
        raise ResultValidationError(
            "recovery task state does not identify the latest attempt"
        )
    if (
        isinstance(state_returncode, bool)
        or not isinstance(state_returncode, int)
        or state_returncode != 0
    ):
        raise ResultValidationError(
            "recovery requires a strict zero trainer returncode"
        )
    state_error = failed_state.get("error")
    if (
        not isinstance(state_error, str)
        or not state_error.startswith(
            "result validation or promotion failed:"
        )
    ):
        raise ResultValidationError(
            "recovery task error is not a result validation or promotion failure"
        )
    if (
        "config_hash" in failed_state
        and failed_state["config_hash"] != config_hash
    ):
        raise ResultValidationError(
            "recovery task state config_hash mismatch"
        )
    _require_recovery_path(
        failed_state.get("staging_dir"),
        attempt_dir,
        field="staging_dir",
    )
    _require_recovery_path(
        failed_state.get("log_path"),
        log_path,
        field="log_path",
    )

    invocation = command_builder(
        task,
        manifest,
        run_inputs,
        attempt_dir,
    )
    actual_hash = derive_config_hash(
        task,
        manifest,
        run_inputs,
        invocation_config=invocation.config,
    )
    if (
        actual_hash != config_hash
        or invocation.required_artifacts
        != invocation_probe.required_artifacts
        or invocation.required_metrics != invocation_probe.required_metrics
        or invocation.result_relpath != invocation_probe.result_relpath
    ):
        raise ResultValidationError(
            "recovery invocation differs from preflight"
        )

    staged_result = attempt_dir / invocation.result_relpath
    try:
        staged_result.resolve(strict=True).relative_to(
            attempt_dir.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise ResultValidationError(
            "recovery result directory escapes its staging attempt"
        ) from exc
    validate_result(
        staged_result,
        task,
        manifest,
        run_inputs,
        expected_config_hash=config_hash,
        required_artifacts=invocation.required_artifacts,
        required_metrics=invocation.required_metrics,
    )
    artifact_sha256 = {
        relative: _sha256_file(
            _safe_artifact_path(staged_result, relative)
        )
        for relative in invocation.required_artifacts
    }

    if final_dir.exists():
        if _directory_nonempty(final_dir):
            raise ResultValidationError(
                f"task final directory became nonempty: {final_dir}"
            )
        final_dir.rmdir()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_result, final_dir)
    validate_result(
        final_dir,
        task,
        manifest,
        run_inputs,
        expected_config_hash=config_hash,
        required_artifacts=invocation.required_artifacts,
        required_metrics=invocation.required_metrics,
    )
    promoted_sha256 = {
        relative: _sha256_file(
            _safe_artifact_path(final_dir, relative)
        )
        for relative in invocation.required_artifacts
    }
    if promoted_sha256 != artifact_sha256:
        raise ResultValidationError(
            "recovered artifacts changed during atomic promotion"
        )
    return {
        "state": "completed",
        "updated_at": _utc_now(),
        "task_id": task.task_id,
        "attempt": attempt,
        "returncode": 0,
        "config_hash": config_hash,
        "final_dir": str(final_dir),
        "log_path": str(log_path),
        "recovered_from_attempt": attempt,
        "recovered_failure_state_sha256": failed_state_sha256,
        "recovered_artifact_sha256": artifact_sha256,
        "recovered_control_validator_sha256": _sha256_file(
            Path(__file__).resolve()
        ),
    }


def _task_environment(
    task: Task,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    config_hash: str,
    attempt: int,
    staging_root: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "XTRAFFIC_TASK_ID": task.task_id,
            "XTRAFFIC_REGION": task.region,
            "XTRAFFIC_MODEL": task.model,
            "XTRAFFIC_VARIANT": task.variant,
            "XTRAFFIC_SEED": str(task.seed),
            "XTRAFFIC_PROTOCOL_VERSION": manifest.protocol_version,
            "XTRAFFIC_SAMPLING_PROTOCOL": task.sampling_protocol,
            "XTRAFFIC_CACHE_SCHEMA_VERSION": str(
                manifest.cache_schema_version
            ),
            "XTRAFFIC_CACHE_BUNDLE_ID": run_inputs.cache_bundle_id,
            "XTRAFFIC_CACHE_BUNDLE_DIGEST": (
                run_inputs.cache_bundle_digest
            ),
            "XTRAFFIC_CODE_SHA": run_inputs.code_sha,
            "XTRAFFIC_CONFIG_HASH": config_hash,
            "XTRAFFIC_RUN_ATTEMPT": str(attempt),
            "XTRAFFIC_STAGING_ROOT": str(staging_root.resolve()),
        }
    )
    return environment


def _verify_git(repo_root: Path, code_sha: str) -> None:
    try:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManifestError(f"cannot verify Git state: {exc}") from exc
    if actual != code_sha:
        raise ManifestError(
            f"code_sha mismatch: requested {code_sha}, checkout is {actual}"
        )
    if dirty:
        raise ManifestError("formal protocol runner requires a clean Git checkout")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _queue_payload(
    *,
    state: str,
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    completed: int,
    skipped: int,
    current_task: str | None = None,
    error: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": state,
        "updated_at": _utc_now(),
        "protocol_version": manifest.protocol_version,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "manifest_digest": manifest.digest,
        "total_tasks": len(manifest.tasks),
        "completed_tasks": completed,
        "skipped_valid_tasks": skipped,
        "current_task": current_task,
    }
    if error is not None:
        payload["error"] = error
    return payload


def run_queue(
    manifest: Manifest,
    run_inputs: ProtocolV3Inputs,
    *,
    repo_root: Path,
    command_builder: CommandBuilder = build_production_invocation,
    verify_git: bool = True,
    stdout: TextIO = sys.stdout,
    recover_staging_task: str | None = None,
) -> int:
    results_root = run_inputs.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    queue_status_path = results_root / "queue_status.json"
    completed = 0
    skipped = 0
    prior_queue: dict[str, object] | None = None
    recovery_preflight_error: str | None = None
    if recover_staging_task is not None:
        task_ids = {task.task_id for task in manifest.tasks}
        if recover_staging_task not in task_ids:
            recovery_preflight_error = (
                "recover_staging_task is not in the active manifest: "
                f"{recover_staging_task!r}"
            )
        elif queue_status_path.exists():
            try:
                prior_queue = _read_strict_json_object(queue_status_path)
            except ManifestError as exc:
                recovery_preflight_error = str(exc)

    def write_queue(
        state: str,
        *,
        current_task: str | None = None,
        error: str | None = None,
    ) -> None:
        _atomic_write_json(
            queue_status_path,
            _queue_payload(
                state=state,
                manifest=manifest,
                run_inputs=run_inputs,
                completed=completed,
                skipped=skipped,
                current_task=current_task,
                error=error,
            ),
        )

    write_queue("preflight")
    if recovery_preflight_error is not None:
        write_queue("failed", error=recovery_preflight_error)
        print(
            f"PREFLIGHT_FAILED: {recovery_preflight_error}",
            file=stdout,
            flush=True,
        )
        return 2
    preflight: dict[str, tuple[Invocation, str, bool]] = {}
    try:
        _validate_run_inputs(run_inputs)
        _verify_cache_bundle_release(run_inputs)
        _validate_results_root_layout(results_root, manifest)
        if verify_git:
            _verify_git(repo_root.resolve(), run_inputs.code_sha)
        for task in manifest.tasks:
            probe_dir = results_root / ".staging" / task.task_id / "__CONFIG__"
            invocation = command_builder(
                task,
                manifest,
                run_inputs,
                probe_dir,
            )
            config_hash = derive_config_hash(
                task,
                manifest,
                run_inputs,
                invocation_config=invocation.config,
            )
            final_dir = results_root / task.output_dir
            should_skip = False
            if final_dir.exists() and not final_dir.is_dir():
                raise ResultValidationError(
                    f"task final path is not a directory: {final_dir}"
                )
            if _directory_nonempty(final_dir):
                try:
                    validate_result(
                        final_dir,
                        task,
                        manifest,
                        run_inputs,
                        expected_config_hash=config_hash,
                        required_artifacts=invocation.required_artifacts,
                        required_metrics=invocation.required_metrics,
                    )
                except ResultValidationError as exc:
                    raise ResultValidationError(
                        f"nonempty task final directory rejected for "
                        f"{task.task_id}: {exc}"
                    ) from exc
                should_skip = True
            preflight[task.task_id] = (
                invocation,
                config_hash,
                should_skip,
            )
        if recover_staging_task is not None:
            recovery_index = next(
                index
                for index, task in enumerate(manifest.tasks)
                if task.task_id == recover_staging_task
            )
            recovery_should_skip = preflight[recover_staging_task][2]
            missing_predecessors = [
                task.task_id
                for task in manifest.tasks[:recovery_index]
                if not preflight[task.task_id][2]
            ]
            if not recovery_should_skip and missing_predecessors:
                raise ResultValidationError(
                    "staging recovery requires every preceding task to "
                    "already have a valid final result: "
                    + ", ".join(missing_predecessors)
                )
    except (ManifestError, ResultValidationError, OSError) as exc:
        write_queue("failed", error=str(exc))
        print(f"PREFLIGHT_FAILED: {exc}", file=stdout, flush=True)
        return 2

    write_queue("running")
    for task_index, task in enumerate(manifest.tasks):
        invocation_probe, config_hash, should_skip = preflight[task.task_id]
        task_status_path = (
            results_root / "state" / "tasks" / f"{task.task_id}.json"
        )
        final_dir = results_root / task.output_dir
        if should_skip:
            skipped += 1
            completed += 1
            _atomic_write_json(
                task_status_path,
                {
                    "state": "skipped_valid",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "config_hash": config_hash,
                    "final_dir": str(final_dir),
                },
            )
            print(f"SKIP_VALID {task.task_id}", file=stdout, flush=True)
            write_queue("running")
            continue

        if recover_staging_task == task.task_id:
            print(
                f"RECOVER_STAGING {task.task_id}",
                file=stdout,
                flush=True,
            )
            try:
                _validate_recovery_queue_state(
                    prior_queue,
                    task=task,
                    task_index=task_index,
                    manifest=manifest,
                    run_inputs=run_inputs,
                )
                recovered_state = _recover_staged_result(
                    results_root=results_root,
                    task=task,
                    manifest=manifest,
                    run_inputs=run_inputs,
                    invocation_probe=invocation_probe,
                    config_hash=config_hash,
                    final_dir=final_dir,
                    task_status_path=task_status_path,
                    command_builder=command_builder,
                )
            except (
                ManifestError,
                OSError,
                ValueError,
                ResultValidationError,
            ) as exc:
                error = f"staging recovery failed: {exc}"
                write_queue(
                    "failed",
                    current_task=task.task_id,
                    error=error,
                )
                print(
                    f"RECOVERY_FAILED {task.task_id}: {exc}",
                    file=stdout,
                    flush=True,
                )
                return 4
            completed += 1
            _atomic_write_json(task_status_path, recovered_state)
            print(
                f"TASK_DONE {task.task_id} recovered",
                file=stdout,
                flush=True,
            )
            write_queue("running")
            continue

        attempt = _next_attempt(results_root, task.task_id)
        attempt_name = f"attempt_{attempt:04d}"
        attempt_dir = (
            results_root / ".staging" / task.task_id / attempt_name
        )
        log_dir = results_root / "logs" / task.task_id
        log_path = log_dir / f"{attempt_name}.log"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir(parents=True, exist_ok=True)
        invocation = command_builder(
            task,
            manifest,
            run_inputs,
            attempt_dir,
        )
        actual_hash = derive_config_hash(
            task,
            manifest,
            run_inputs,
            invocation_config=invocation.config,
        )
        if (
            actual_hash != config_hash
            or invocation.required_artifacts
            != invocation_probe.required_artifacts
            or invocation.required_metrics != invocation_probe.required_metrics
        ):
            error = "task invocation changed between preflight and launch"
            _atomic_write_json(
                task_status_path,
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "error": error,
                },
            )
            write_queue("failed", current_task=task.task_id, error=error)
            print(f"TASK_FAILED {task.task_id}: {error}", file=stdout, flush=True)
            return 3
        _atomic_write_json(
            task_status_path,
            {
                "state": "running",
                "updated_at": _utc_now(),
                "task_id": task.task_id,
                "attempt": attempt,
                "config_hash": config_hash,
                "staging_dir": str(attempt_dir),
                "log_path": str(log_path),
                "command": list(invocation.command),
            },
        )
        write_queue("running", current_task=task.task_id)
        print(
            f"RUN {task.task_id} {attempt_name}",
            file=stdout,
            flush=True,
        )

        process: subprocess.Popen[bytes] | None = None
        try:
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            )
            with log_path.open("xb") as log_file:
                process = subprocess.Popen(
                    list(invocation.command),
                    cwd=str(repo_root),
                    env=_task_environment(
                        task,
                        manifest,
                        run_inputs,
                        config_hash,
                        attempt,
                        attempt_dir,
                    ),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creationflags,
                )
                returncode = process.wait()
        except KeyboardInterrupt:
            if process is not None:
                _terminate_process(process)
            error = "runner interrupted by user"
            _atomic_write_json(
                task_status_path,
                {
                    "state": "interrupted",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "error": error,
                    "staging_dir": str(attempt_dir),
                    "log_path": str(log_path),
                },
            )
            write_queue(
                "interrupted",
                current_task=task.task_id,
                error=error,
            )
            print(f"INTERRUPTED {task.task_id}", file=stdout, flush=True)
            return 130
        except OSError as exc:
            returncode = None
            error = f"cannot launch task subprocess: {exc}"
            _atomic_write_json(
                task_status_path,
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "error": error,
                },
            )
            write_queue("failed", current_task=task.task_id, error=error)
            print(f"TASK_FAILED {task.task_id}: {error}", file=stdout, flush=True)
            return 3

        if returncode != 0:
            error = f"trainer exited with code {returncode}"
            _atomic_write_json(
                task_status_path,
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "returncode": returncode,
                    "error": error,
                    "staging_dir": str(attempt_dir),
                    "log_path": str(log_path),
                },
            )
            write_queue("failed", current_task=task.task_id, error=error)
            print(f"TASK_FAILED {task.task_id}: {error}", file=stdout, flush=True)
            return 3

        staged_result = attempt_dir / invocation.result_relpath
        try:
            staged_result.resolve().relative_to(attempt_dir.resolve())
            validate_result(
                staged_result,
                task,
                manifest,
                run_inputs,
                expected_config_hash=config_hash,
                required_artifacts=invocation.required_artifacts,
                required_metrics=invocation.required_metrics,
            )
            if final_dir.exists():
                if _directory_nonempty(final_dir):
                    raise ResultValidationError(
                        f"task final directory became nonempty: {final_dir}"
                    )
                final_dir.rmdir()
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_result, final_dir)
            validate_result(
                final_dir,
                task,
                manifest,
                run_inputs,
                expected_config_hash=config_hash,
                required_artifacts=invocation.required_artifacts,
                required_metrics=invocation.required_metrics,
            )
        except (OSError, ValueError, ResultValidationError) as exc:
            error = f"result validation or promotion failed: {exc}"
            _atomic_write_json(
                task_status_path,
                {
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "returncode": returncode,
                    "error": error,
                    "staging_dir": str(attempt_dir),
                    "log_path": str(log_path),
                },
            )
            write_queue("failed", current_task=task.task_id, error=error)
            print(f"TASK_FAILED {task.task_id}: {error}", file=stdout, flush=True)
            return 4

        completed += 1
        _atomic_write_json(
            task_status_path,
            {
                "state": "completed",
                "updated_at": _utc_now(),
                "task_id": task.task_id,
                "attempt": attempt,
                "returncode": returncode,
                "config_hash": config_hash,
                "final_dir": str(final_dir),
                "log_path": str(log_path),
            },
        )
        print(f"TASK_DONE {task.task_id}", file=stdout, flush=True)
        write_queue("running")

    write_queue("completed")
    print("ALL_DONE", file=stdout, flush=True)
    return 0


def _resolve_from_repo(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "protocol_v3_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=script_dir.parents[2])
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--recover-staging-task")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(
            args.manifest,
            require_production_matrix=True,
        )
        repo_root = args.repo_root.resolve()
        results_value = args.results_root or manifest.default_results_root
        cache_value = args.cache_dir or manifest.default_cache_dir
        graph_value = args.graph_dir or manifest.default_graph_dir
        run_inputs = ProtocolV3Inputs(
            results_root=_resolve_from_repo(repo_root, results_value),
            cache_dir=_resolve_from_repo(repo_root, cache_value),
            graph_dir=_resolve_from_repo(repo_root, graph_value),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
    except ManifestError as exc:
        print(f"CONFIG_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    return run_queue(
        manifest,
        run_inputs,
        repo_root=repo_root,
        verify_git=True,
        recover_staging_task=args.recover_staging_task,
    )


if __name__ == "__main__":
    sys.exit(main())
