#!/usr/bin/env python3
"""Run the locked protocol-v3 seed-42 convergence-audit matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from scripts.remote.windows.protocol_v3_baseline_runner import (
    build_baseline_invocation,
)
from scripts.remote.windows.protocol_v3_event_extension_runner import (
    ExtensionInputs,
    SUPPORT_MANIFEST,
    build_extension_invocation,
    verify_extension_support,
)
from scripts.remote.windows.protocol_v3_runner import (
    Invocation,
    Manifest,
    ManifestError,
    ProtocolV3Inputs,
    Task,
    load_manifest,
    run_queue,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_MANIFEST = (
    REPO_ROOT / "config/protocol_v3/event_convergence_seed42_v1.json"
)
MATRIX_ID = "protocol_v3_event_convergence_seed42_39_v1"
REGIONS = ("Alameda", "ContraCosta", "Orange")
CONVERGENCE_METHODS = (
    "lstm",
    "dcrnn",
    "agcrn",
    "stgcn",
    "graphwavenet",
    "astgcn",
    "sttn",
    "dstagnn",
    "dgcrn",
    "d2stgnn",
    "bist",
    "igstgnn",
    "ndr_net",
)
BASELINE_METHODS = frozenset(CONVERGENCE_METHODS[:10])
EXTENSION_METHODS = frozenset(CONVERGENCE_METHODS[10:])


def _optimizer(
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float | None,
    scheduler: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "scheduler": None if scheduler is None else dict(scheduler),
    }


EXPECTED_METHOD_CONTRACTS: dict[str, dict[str, object]] = {
    "lstm": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(0.001, 0.0001, 5.0),
    },
    "dcrnn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(
            0.01,
            0.0,
            5.0,
            {"name": "MultiStepLR", "milestones": [10, 50, 90], "gamma": 0.1},
        ),
    },
    "agcrn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(0.001, 0.0, None),
    },
    "stgcn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(
            0.001,
            0.0005,
            None,
            {"name": "StepLR", "step_size": 10, "gamma": 0.95},
        ),
    },
    "graphwavenet": {
        "trainer": "graphwavenet_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(
            0.001,
            0.0001,
            5.0,
            {
                "name": "CosineAnnealingLR",
                "t_max": "max_epochs_times_training_batches",
                "eta_min": "learning_rate_times_0.01",
                "step_unit": "optimizer_step",
            },
        ),
    },
    "astgcn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(
            0.001,
            0.0001,
            5.0,
            {"name": "CosineAnnealingLR", "t_max": 100, "eta_min": 0.000001},
        ),
    },
    "sttn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(0.001, 0.0001, 5.0),
    },
    "dstagnn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(0.0001, 0.0, None),
    },
    "dgcrn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(0.001, 0.0001, 5.0),
    },
    "d2stgnn": {
        "trainer": "baseline_zoo_convergence100_v1",
        "max_epochs": 100,
        "patience": 30,
        "optimizer": _optimizer(
            0.002,
            0.00001,
            5.0,
            {
                "name": "MultiStepLR",
                "milestones": [1, 38, 46, 54, 62, 70, 80],
                "gamma": 0.5,
            },
        ),
    },
    "bist": {
        "trainer": "event_extension_convergence100_v1",
        "max_epochs": 100,
        "patience": 20,
        "optimizer": _optimizer(
            0.002,
            0.00001,
            5.0,
            {"name": "MultiStepLR", "milestones": [30, 60, 80, 100], "gamma": 0.5},
        ),
    },
    "igstgnn": {
        "trainer": "event_extension_convergence100_v1",
        "max_epochs": 100,
        "patience": 20,
        "optimizer": _optimizer(
            0.002,
            0.00001,
            5.0,
            {
                "name": "MultiStepLR",
                "milestones": [1, 38, 46, 54, 62, 70, 80],
                "gamma": 0.5,
            },
        ),
    },
    "ndr_net": {
        "trainer": "event_extension_convergence100_v1",
        "max_epochs": 100,
        "patience": None,
        "optimizer": _optimizer(
            0.001,
            0.0001,
            5.0,
            {
                "name": "CosineAnnealingLR",
                "t_max": "max_epochs_times_training_batches",
                "eta_min": 0.00001,
            },
        ),
    },
}


EXPECTED_TRAINING = {
    "epochs": 100,
    "patience": 30,
    "device": "cuda:0",
    "batch_size": 32,
    "stride": 1,
    "train_frac": 0.7,
    "val_frac": 0.15,
    "seed": 42,
    "checkpoint_metric": "global_validation_masked_mae_all_valid_points",
    "evaluation_mask": "explicit_v3_flow_mask",
    "partial_batch": "retain",
    "traffic_loading": "eager",
    "training_analysis_labels": False,
    "experiment_role": "convergence_audit_not_final_multiseed_table",
}


@dataclass(frozen=True)
class ConvergenceManifest(Manifest):
    matrix_id: str
    method_contracts: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class ConvergenceInputs:
    protocol: ProtocolV3Inputs
    extension: ExtensionInputs


def _load_payload(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read convergence manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError("convergence manifest must be a JSON object")
    return value


def load_convergence_manifest(
    path: Path | str,
    *,
    require_complete_matrix: bool = True,
) -> ConvergenceManifest:
    source = Path(path).resolve()
    base = load_manifest(source, require_production_matrix=False)
    payload = _load_payload(source)
    if payload.get("matrix_id") != MATRIX_ID:
        raise ManifestError(f"matrix_id must equal {MATRIX_ID!r}")
    training = dict(base.training)
    microbatch_size = training.pop("microbatch_size", None)
    if training != EXPECTED_TRAINING:
        raise ManifestError("convergence common training contract is not canonical")
    if microbatch_size is not None:
        if (
            require_complete_matrix
            or isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or not 1 <= microbatch_size <= int(EXPECTED_TRAINING["batch_size"])
        ):
            raise ManifestError(
                "training.microbatch_size is valid only for a canonical shard "
                "and must be between 1 and batch_size"
            )
    method_contracts = payload.get("method_contracts")
    if method_contracts != EXPECTED_METHOD_CONTRACTS:
        raise ManifestError("convergence method contracts are not canonical")

    expected = {
        (region, method, 42, "event")
        for region in REGIONS
        for method in CONVERGENCE_METHODS
    }
    actual = {
        (task.region, task.model, task.seed, task.sampling_protocol)
        for task in base.tasks
    }
    if require_complete_matrix:
        if len(base.tasks) != 39 or actual != expected:
            raise ManifestError("convergence matrix must contain the exact 39 tasks")
    elif len(actual) != len(base.tasks) or not actual <= expected:
        raise ManifestError("convergence shard is not a canonical task subset")

    for task in base.tasks:
        expected_kind = (
            "event_extension"
            if task.model in EXTENSION_METHODS
            else "graphwavenet"
            if task.model == "graphwavenet"
            else "baseline_zoo"
        )
        expected_variant = (
            "v3_event_aligned" if task.model in EXTENSION_METHODS else "base"
        )
        if task.kind != expected_kind or task.variant != expected_variant:
            raise ManifestError(f"invalid convergence task binding: {task.task_id}")

    return ConvergenceManifest(
        manifest_version=base.manifest_version,
        protocol_version=base.protocol_version,
        cache_schema_version=base.cache_schema_version,
        default_results_root=base.default_results_root,
        default_cache_dir=base.default_cache_dir,
        default_graph_dir=base.default_graph_dir,
        training=base.training,
        tasks=base.tasks,
        source_path=base.source_path,
        digest=base.digest,
        matrix_id=MATRIX_ID,
        method_contracts={
            name: dict(contract) for name, contract in method_contracts.items()
        },
    )


def _extension_manifest_for_task(
    manifest: ConvergenceManifest,
    task: Task,
) -> Manifest:
    contract = manifest.method_contracts[task.model]
    patience = contract["patience"]
    effective_patience = 20 if patience is None else int(patience)
    training = {
        "epochs": int(contract["max_epochs"]),
        "patience": effective_patience,
        "device": str(manifest.training["device"]),
        "seed": task.seed,
        "checkpoint_metric": manifest.training["checkpoint_metric"],
        "evaluation_mask": manifest.training["evaluation_mask"],
        "partial_batch": manifest.training["partial_batch"],
        "traffic_loading": manifest.training["traffic_loading"],
        "training_analysis_labels": manifest.training["training_analysis_labels"],
    }
    return Manifest(
        manifest_version=manifest.manifest_version,
        protocol_version=manifest.protocol_version,
        cache_schema_version=manifest.cache_schema_version,
        default_results_root=manifest.default_results_root,
        default_cache_dir=manifest.default_cache_dir,
        default_graph_dir=manifest.default_graph_dir,
        training=training,
        tasks=(task,),
        source_path=manifest.source_path,
        digest=manifest.digest,
    )


def _with_convergence_contract(
    invocation: Invocation,
    *,
    manifest: ConvergenceManifest,
    task: Task,
    command: list[str],
) -> Invocation:
    config = dict(invocation.config)
    config.update(
        {
            "matrix_id": manifest.matrix_id,
            "experiment_role": manifest.training["experiment_role"],
            "method_contract": dict(manifest.method_contracts[task.model]),
        }
    )
    artifacts = tuple(invocation.required_artifacts)
    if "training_diagnostics.json" not in artifacts:
        artifacts += ("training_diagnostics.json",)
    return Invocation(
        command=tuple(command),
        result_relpath=invocation.result_relpath,
        required_artifacts=artifacts,
        required_metrics=invocation.required_metrics,
        config=config,
    )


def build_convergence_invocation(
    task: Task,
    manifest: ConvergenceManifest,
    inputs: ConvergenceInputs,
    attempt_dir: Path,
) -> Invocation:
    if task.model not in manifest.method_contracts:
        raise ManifestError(f"missing method contract for {task.model}")
    if task.model in BASELINE_METHODS:
        base = build_baseline_invocation(
            task,
            manifest,
            inputs.protocol,
            attempt_dir,
        )
        command = list(base.command)
        if task.model == "graphwavenet":
            command += ["--run-contract", "convergence100"]
        else:
            command += ["--recipe-profile", "convergence100"]
        return _with_convergence_contract(
            base,
            manifest=manifest,
            task=task,
            command=command,
        )
    if task.model in EXTENSION_METHODS:
        current = _extension_manifest_for_task(manifest, task)
        extension = inputs.extension
        microbatch_size = manifest.training.get("microbatch_size")
        if microbatch_size is not None:
            extension = replace(
                extension,
                max_microbatch_size=int(microbatch_size),
            )
        base = build_extension_invocation(
            task,
            current,
            extension,
            attempt_dir,
        )
        command = list(base.command) + ["--run-contract", "convergence100"]
        return _with_convergence_contract(
            base,
            manifest=manifest,
            task=task,
            command=command,
        )
    raise ManifestError(f"unsupported convergence method: {task.model}")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--support-manifest", type=Path, default=SUPPORT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--allow-shard", action="store_true")
    parser.add_argument("--igstgnn-data-parallel", action="store_true")
    parser.add_argument("--max-microbatch-size", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_convergence_manifest(
            args.manifest,
            require_complete_matrix=not args.allow_shard,
        )
        root = args.repo_root.resolve()
        protocol = ProtocolV3Inputs(
            results_root=_resolve(
                root,
                args.results_root or manifest.default_results_root,
            ),
            cache_dir=_resolve(root, args.cache_dir or manifest.default_cache_dir),
            graph_dir=_resolve(root, args.graph_dir or manifest.default_graph_dir),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
        support = verify_extension_support(
            support_manifest_path=_resolve(root, args.support_manifest),
            support_root=_resolve(root, args.support_root),
            node_order=_resolve(root, args.node_order),
        )
        extension = ExtensionInputs(
            support_root=support.support_root,
            node_order=support.node_order,
            support_manifest_sha256=support.manifest_sha256,
            igstgnn_data_parallel=args.igstgnn_data_parallel,
            max_microbatch_size=args.max_microbatch_size,
            cache_dir=protocol.cache_dir,
            graph_dir=protocol.graph_dir,
            python_executable=protocol.python_executable,
        )
        inputs = ConvergenceInputs(protocol=protocol, extension=extension)
    except ManifestError as exc:
        print(f"CONFIG_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2
    return run_queue(
        manifest,
        protocol,
        repo_root=root,
        command_builder=lambda task, current, _protocol, attempt: (
            build_convergence_invocation(task, current, inputs, attempt)
        ),
        verify_git=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
