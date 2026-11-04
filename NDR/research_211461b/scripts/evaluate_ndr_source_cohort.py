#!/usr/bin/env python3
"""Certify source checkpoints and evaluate a frozen true-class three-seed cohort."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil

from ndr_research.source_cohort_evaluation import (
    CERTIFICATE_SCHEMA, CLASSES, FAMILY_SCOPE, GATE_SCHEMA, KINDS, MODES,
    REPORT_STATUS, SEEDS, SourceCohortLedger, admit_source_cohort_gate,
    evaluate_source_cohort, inspect_source_model, rebuild_source_model, source_pool_digest,
)
from ndr_research.transfer_data import (
    PROTOCOL, REGIONS, TestEvaluationGate, TransferPreprocessing, TransferRegionReader,
    fit_historical_calibration, load_transfer_sources,
)
from ndr_research.transfer_evaluation import CONTROL_KINDS, SCENARIOS, canonical_sha256, file_sha256, write_new_json
from scripts.evaluate_ndr_transfer import evaluation_source_snapshot
from scripts.train_ndr_research import require_snapshot
from scripts.train_ndr_transfer import (
    apply_source_loading, configure_transfer_runtime,
    transfer_runtime_identity, verify_transfer_checkpoint,
)
from scripts.train_shared_staeformer_transfer import verify_reference_checkpoint


def public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {name: copy.deepcopy(value) for name, value in spec.items() if name not in {"checkpoint", "paths"}}


def evaluator_snapshot() -> dict[str, Any]:
    snapshot = evaluation_source_snapshot(ROOT)
    for name in ("ndr_research/source_cohort_evaluation.py", "scripts/evaluate_ndr_source_cohort.py",
                 "ndr_research/shared_staeformer.py", "scripts/train_shared_staeformer_transfer.py",
                 "ndr_research/indomain_evaluation.py", "ndr_research/cohort.py"):
        path = ROOT / name
        snapshot["files"][name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    snapshot["files_digest"] = canonical_sha256(snapshot["files"])
    return snapshot


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect-checkpoint", "certify-source", "create-gate", "evaluate"), default="evaluate")
    parser.add_argument("--checkpoint-kind", choices=KINDS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--independent-source-certificate", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--region", choices=REGIONS)
    parser.add_argument("--authorization-id")
    parser.add_argument("--structure-development-closed", action="store_true")
    parser.add_argument("--evaluation-ledger", type=Path)
    parser.add_argument("--evaluation-mode", choices=MODES, default="source_only")
    parser.add_argument("--permitted-evaluation-modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--max-test-evaluations", type=int, default=12)
    parser.add_argument("--historical-prefix-end", type=int)
    parser.add_argument("--historical-prefix-authorization-id")
    parser.add_argument("--perturbation-seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def certify_source(args: argparse.Namespace) -> dict[str, Any]:
    """Fresh CLI process reloads and verifies source val without a target reader."""
    if __name__ != "__main__":
        raise PermissionError("source certification must run as an independent CLI process")
    if not args.checkpoint or not args.summary or not args.checkpoint_kind or not args.output:
        raise ValueError("certify-source requires a checkpoint, summary, true kind and new output directory")
    started_ns = int(psutil.Process(os.getpid()).create_time() * 1e9)
    checkpoint_modified = args.checkpoint.stat().st_mtime_ns
    if checkpoint_modified >= started_ns:
        raise PermissionError("the source checkpoint must exist before this verifier process starts")
    spec = inspect_source_model(args.checkpoint, args.verification, args.summary, kind=args.checkpoint_kind,
                                require_certificate=False, allow_pending_verification=args.verification is None)
    checkpoint, config = spec["checkpoint"], spec["checkpoint"]["config"]
    device = configure_transfer_runtime(args.device, config["seed"], config["threads"])
    runtime = transfer_runtime_identity(device)
    if runtime != config["runtime"]:
        raise ValueError("source certificate requires the original training runtime")
    before = {"checkpoint": file_sha256(args.checkpoint), "summary": file_sha256(args.summary)}
    if args.verification:
        before["verification"] = file_sha256(args.verification)
    snapshot = evaluator_snapshot()
    require_snapshot(snapshot)
    data = load_transfer_sources(config["source_regions"], config["heldout_region"], args.data_dir, args.graph_dir,
                                 batch_size=config["batch_size"], seed=config["seed"])
    if set(data.readers) != set(config["source_regions"]) or config["heldout_region"] in data.readers:
        raise PermissionError("source certification unexpectedly created a target reader")
    apply_source_loading(data, lazy=config["loading_policy"] == "source_lazy_raw_windows")
    args.output.mkdir(parents=True, exist_ok=False)
    verify = verify_transfer_checkpoint if args.checkpoint_kind == "shared_graph" else verify_reference_checkpoint
    result = verify(checkpoint_path=args.checkpoint, data=data, device=device, output=args.output, current_runtime=runtime)
    verification_path = args.verification or args.output / "verification.json"
    final = inspect_source_model(args.checkpoint, verification_path, args.summary,
                                 kind=args.checkpoint_kind, require_certificate=False)
    after = {"checkpoint": file_sha256(args.checkpoint), "summary": file_sha256(args.summary)}
    if args.verification:
        after["verification"] = file_sha256(args.verification)
    if before != after:
        raise RuntimeError("source acceptance artifacts changed during certification")
    require_snapshot(snapshot)
    require_snapshot(config["source_snapshot"])
    certificate = {"schema_version": CERTIFICATE_SCHEMA, "status": "certified_independent_complete_source_validation",
                   "model_class": CLASSES[args.checkpoint_kind], "checkpoint_kind": args.checkpoint_kind,
                   "checkpoint_sha256": before["checkpoint"], "summary_sha256": before["summary"],
                   "verification_sha256": file_sha256(verification_path), "config_digest": checkpoint["config_digest"],
                   "source_snapshot_files_digest": config["source_snapshot"]["files_digest"],
                   "selection_metric_digest": final["validation_selection_sha256"],
                   "source_pool_numeric_semantics_sha256": final["source_pool_numeric_semantics_sha256"],
                   "runtime": runtime, "data_identity": data.identity,
                   "verification_process_id": os.getpid(), "training_process_id": config.get("training_process_id"),
                   "verifier_process_started_ns": started_ns, "checkpoint_mtime_ns": checkpoint_modified,
                   "verification_execution": "fresh_cli_process_checkpoint_reconstruction",
                   "checkpoint_existed_before_verifier_entry": True, "training_executed_in_verifier": False,
                   "source_validation_complete": True, "source_validation": result["source_validation"],
                   "source_access_audits": {region: reader.access_audit() for region, reader in data.readers.items()},
                   "evaluator_source_snapshot": snapshot, "target_data_read": False, "test_evaluated": False,
                   "created_unix_time_ns": time.time_ns()}
    write_new_json(args.output / "source_certificate.json", certificate)
    return {"status": certificate["status"], "model_class": certificate["model_class"],
            "certificate": str((args.output / "source_certificate.json").resolve()),
            "verification": str(verification_path.resolve()), "target_data_read": False}


def create_gate(args: argparse.Namespace) -> dict[str, Any]:
    if (not args.model_manifest or not args.gate_json or not args.region or not args.authorization_id
            or not args.structure_development_closed or not args.evaluation_ledger or args.output is not None):
        raise ValueError("create-gate requires a model manifest, region, closed structure, authorization and shared ledger")
    manifest = json.loads(args.model_manifest.read_text())
    models, bundle, sources = [], None, None
    for row in manifest["models"]:
        paths = {name: Path(row[name]) for name in ("checkpoint", "verification", "summary", "certificate")}
        paths = {name: value if value.is_absolute() else args.model_manifest.resolve().parent / value for name, value in paths.items()}
        spec = inspect_source_model(paths["checkpoint"], paths["verification"], paths["summary"],
                                    kind=row["checkpoint_kind"], certificate_path=paths["certificate"])
        models.append({"model_id": row["model_id"], "checkpoint_kind": row["checkpoint_kind"],
                       **{name: str(value.resolve()) for name, value in paths.items()},
                       **{name + "_sha256": file_sha256(value) for name, value in paths.items()},
                       **{name: spec[name] for name in ("model_class", "family_identity", "training_seed", "config_digest",
                                                      "validation_selection_sha256", "data_identity_sha256",
                                                      "source_pool_numeric_semantics_sha256", "model_state_sha256")}})
        bundle, sources = spec["bundle_digest"], spec["source_regions"]
    gate = {"schema_version": GATE_SCHEMA, "protocol_version": PROTOCOL, "sampling_protocol": "event",
            "split": "test", "region": args.region, "source_regions": sources, "cache_bundle_digest": bundle,
            "comparison_scope": "matched_two_source_training", "training_seeds": list(SEEDS),
            "authorization_id": args.authorization_id, "structure_development_closed": True,
            "selection_frozen": True, "test_used_for_structure_selection": False,
            "models": models, "scenarios": list(SCENARIOS), "control_kinds": list(CONTROL_KINDS),
            "evaluation_mode": args.evaluation_mode, "permitted_evaluation_modes": args.permitted_evaluation_modes,
            "evaluation_ledger": str(args.evaluation_ledger.resolve()), "max_test_evaluations": args.max_test_evaluations,
            "perturbation_seed": args.perturbation_seed, "family_scope": FAMILY_SCOPE}
    if args.evaluation_mode == "historical_calibrated_transfer":
        gate["historical_prefix"] = {"prefix_end": args.historical_prefix_end, "authorization_id": args.historical_prefix_authorization_id}
    elif args.historical_prefix_end is not None or args.historical_prefix_authorization_id is not None:
        raise ValueError("source-only gate cannot declare target calibration")
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.gate_json, gate)
    try:
        admitted_gate, admitted = admit_source_cohort_gate(args.gate_json)
    except BaseException:
        args.gate_json.unlink()
        raise
    ledger = SourceCohortLedger(admitted_gate, admitted, args.gate_json, args.gate_json.parent)
    return {"status": "frozen_source_cohort_gate_created", "gate_json": str(args.gate_json.resolve()),
            "gate_sha256": admitted_gate["gate_sha256"], "family_sha256": ledger.family_sha,
            "family_scope": FAMILY_SCOPE, "target_data_read": False}


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gate_json or not args.output or args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("evaluation requires a frozen gate, new output and positive batch/thread sizes")
    if any(getattr(args, name) is not None for name in ("checkpoint", "checkpoint_kind", "verification", "summary",
                                                      "independent_source_certificate", "model_manifest", "region", "authorization_id",
                                                      "evaluation_ledger", "historical_prefix_end", "historical_prefix_authorization_id")):
        raise ValueError("evaluation obtains all model, authorization and prefix settings from its frozen gate")
    gate, admitted = admit_source_cohort_gate(args.gate_json)
    device = configure_transfer_runtime(args.device, gate["perturbation_seed"], args.threads)
    snapshot = evaluator_snapshot()
    require_snapshot(snapshot)
    source = TransferPreprocessing.from_state(next(iter(admitted.values()))["checkpoint"]["source_statistics"])
    models = {name: rebuild_source_model(spec, device) for name, spec in admitted.items()}
    before = {name: spec["model_state_sha256"] for name, spec in admitted.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    gate_bytes = args.gate_json.read_bytes()
    if hashlib.sha256(gate_bytes).hexdigest() != gate["gate_sha256"]:
        raise RuntimeError("source cohort gate changed after admission")
    with (args.output / "gate.json").open("xb") as handle:
        handle.write(gate_bytes)
    write_new_json(args.output / "evaluation_source_snapshot.json", snapshot)
    ledger = SourceCohortLedger(gate, admitted, args.gate_json, args.output)
    reservation = ledger.reserve()
    reader = None
    try:
        reader = TransferRegionReader(gate["region"], args.data_dir, args.graph_dir,
                                      role="target", bundle_manifest=args.bundle_manifest)
        if reader.contract.bundle_digest != gate["cache_bundle_digest"]:
            raise ValueError("target reader cache bundle differs from the frozen cohort")
        preprocessing = source
        if gate["evaluation_mode"] == "historical_calibrated_transfer":
            prefix = gate["historical_prefix"]
            preprocessing = fit_historical_calibration(reader, prefix_end=prefix["prefix_end"], authorization_id=prefix["authorization_id"])
        grant = TestEvaluationGate(gate["authorization_id"], canonical_sha256({name: spec["checkpoint_sha256"] for name, spec in admitted.items()}))
        report = evaluate_source_cohort(models, reader=reader, preprocessing=preprocessing,
                                        source_preprocessing=source, evaluation_grant=grant, batch_size=args.batch_size,
                                        perturbation_seed=gate["perturbation_seed"], device=device)
        if any(report["model_state_sha256"][name] != before[name] for name in models):
            raise RuntimeError("source cohort model state changed before evaluation started")
        after_gate, after_models = admit_source_cohort_gate(args.gate_json)
        if after_gate != gate or {name: spec["model_state_sha256"] for name, spec in after_models.items()} != before:
            raise RuntimeError("source gate or accepted model state changed during heldout evaluation")
        require_snapshot(snapshot)
        for spec in admitted.values():
            require_snapshot(spec["checkpoint"]["config"]["source_snapshot"])
        if source_pool_digest(source) != next(iter(admitted.values()))["source_pool_numeric_semantics_sha256"]:
            raise RuntimeError("frozen source pool was mutated during inference")
        for name, result in report["models"].items():
            if name in admitted:
                result.update(public_spec(admitted[name]))
            else:
                result.update(model_id=name, model_class="HistoryBaseline", training_scope="fixed_source_only_control")
        result = {"schema_version": GATE_SCHEMA, "status": REPORT_STATUS, **report,
                  "family_scope": FAMILY_SCOPE, "family_sha256": ledger.family_sha,
                  "family_identity": ledger.family_payload, "comparison_scope": gate["comparison_scope"],
                  "gate_sha256": gate["gate_sha256"], "region": gate["region"], "source_regions": gate["source_regions"],
                  "heldout_region": gate["region"],
                  "training_seeds": list(SEEDS), "sampling_protocol": "event", "protocol_version": PROTOCOL,
                  "cache_bundle_digest": gate["cache_bundle_digest"], "split": "test",
                  "evaluation_mode": gate["evaluation_mode"], "mode": gate["evaluation_mode"],
                  "preprocessing_identity": preprocessing.identity, "source_preprocessing_identity": source.identity,
                  "source_preprocessing_numeric_semantics_sha256": source_pool_digest(source),
                  "controls_use_source_only_preprocessing": True, "staeformer_scalar_normalization_remains_source_frozen": True,
                  "evaluation_runtime": transfer_runtime_identity(device), "evaluation_source_snapshot": snapshot,
                  "target_access_audit": reader.access_audit(), "test_evaluated": True,
                  "test_evaluation_budget_scope": FAMILY_SCOPE, "budget_reservation": reservation,
                  "test_driven_tuning": False, "method_comparison_is_gate_ablation": False,
                  "training_recipe_homogeneity": "within_each_true_model_family_across_three_seeds"}
        write_new_json(args.output / "report.json", result)
        ledger.finish("completed", report_sha256=file_sha256(args.output / "report.json"))
        return {"status": REPORT_STATUS, "report": str((args.output / "report.json").resolve()),
                "family_sha256": ledger.family_sha, "family_scope": FAMILY_SCOPE}
    except BaseException as error:
        if reader is not None:
            write_new_json(args.output / "failure_access_audit.json", reader.access_audit())
        ledger.finish("failed", error_type=type(error).__name__)
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    arguments = sys.argv[1:] if argv is None else argv
    args = argument_parser().parse_args(arguments)
    if args.mode == "evaluate":
        forbidden = ("--evaluation-mode", "--permitted-evaluation-modes", "--max-test-evaluations", "--perturbation-seed", "--structure-development-closed")
        if any(value == option or value.startswith(option + "=") for value in arguments for option in forbidden):
            raise ValueError("evaluation cannot override its frozen mode, family budget, seed or structure declaration")
    if args.mode == "inspect-checkpoint":
        if not args.checkpoint or not args.verification or not args.summary or not args.checkpoint_kind:
            raise ValueError("inspect-checkpoint requires checkpoint, verification, summary and true checkpoint kind")
        result = public_spec(inspect_source_model(args.checkpoint, args.verification, args.summary, kind=args.checkpoint_kind,
                                                 certificate_path=args.independent_source_certificate))
    elif args.mode == "certify-source":
        result = certify_source(args)
    elif args.mode == "create-gate":
        result = create_gate(args)
    else:
        result = run_evaluation(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
