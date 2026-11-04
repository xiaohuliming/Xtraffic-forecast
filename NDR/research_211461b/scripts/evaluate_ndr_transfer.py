#!/usr/bin/env python3
"""Create an explicit freeze gate or evaluate its held-out transfer collection."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.transfer_data import (
    TestEvaluationGate,
    TransferPreprocessing,
    TransferRegionReader,
    fit_historical_calibration,
)
from ndr_research.transfer_evaluation import (
    SCENARIOS,
    CANDIDATE_KINDS,
    CONTROL_KINDS,
    EvaluationLedger,
    HistoryBaseline,
    canonical_sha256,
    evaluate_frozen_batches,
    file_sha256,
    require_candidate_sources,
    validate_frozen_gate,
    write_new_json,
)
from scripts.train_ndr_transfer import (
    configure_transfer_runtime,
    load_transfer_checkpoint,
    transfer_runtime_identity,
)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Pre-register one shared ledger before the first held-out access.
The family key is candidate source-code digest plus held-out region, across
all checkpoint seeds and both transfer modes. The first started attempt fixes
the family test limit; later gates cannot increase or decrease it. Failed test
attempts consume the limit. Target-val attempts are recorded separately and
never select the model.

Example for three frozen seeds and two modes, six total test attempts:
  python scripts/evaluate_ndr_transfer.py --mode create-gate \\
    --checkpoint best.pt --gate-json frozen_seed42_source.json \\
    --source-verification verification.json --authorization-id preregistered-fold \\
    --evaluation-ledger family_ledger.jsonl --max-test-evaluations 6
Use the same ledger and limit 6 for the other five pre-registered gates.
""")
    parser.add_argument("--mode", choices=("create-gate", "evaluate"), default="evaluate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path)
    parser.add_argument("--authorization-id")
    parser.add_argument("--evaluation-mode", choices=("source_only", "historical_calibrated_transfer"), default="source_only")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--perturbation-seed", type=int, default=42)
    parser.add_argument("--max-test-evaluations", type=int, default=1,
                        help="pre-registered family limit across seeds/modes; failures count; first access locks it")
    parser.add_argument("--evaluation-ledger", type=Path,
                        help="one shared append-only ledger for the entire candidate-code/held-out family")
    parser.add_argument("--historical-prefix-end", type=int)
    parser.add_argument("--historical-prefix-authorization-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/dist_net/region_data_v3_noleak_dedup")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs/region_graphs")
    parser.add_argument("--bundle-manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def create_explicit_gate(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """No target reader is created; the CLI names one already selected checkpoint."""
    if args.source_verification is None or args.evaluation_ledger is None or not args.authorization_id:
        raise ValueError("gate creation requires source verification, authorization-id and a shared evaluation ledger")
    if args.output is not None:
        raise ValueError("gate creation does not write a target evaluation directory")
    gate = {
        "schema_version": 1, "authorization_id": args.authorization_id,
        "selection_frozen": True,
        "candidate_sha": checkpoint["config"]["source_snapshot"]["files_digest"],
        "modelcheckpoint_sha": file_sha256(args.checkpoint),
        "model_config_sha": canonical_sha256(checkpoint["model_config"]),
        "source_regions": checkpoint["source_regions"], "heldout_region": checkpoint["heldout_region"],
        "cache_bundle_digest": checkpoint["source_statistics_identity"]["bundle_digest"],
        "source_validation_selection_sha": canonical_sha256(checkpoint["selection_metric"]),
        "source_validation_verification": str(args.source_verification.resolve()),
        "source_validation_verification_sha": file_sha256(args.source_verification),
        "evaluation_mode": args.evaluation_mode, "split": args.split,
        "perturbation_seed": args.perturbation_seed, "scenarios": list(SCENARIOS),
        "candidate_kinds": list(CANDIDATE_KINDS),
        "evaluation_ledger": str(args.evaluation_ledger.resolve()),
        "max_test_evaluations": args.max_test_evaluations,
        "freeze_basis": "explicit_cli_selected_complete_source_validation_checkpoint",
        "architecture_search": False,
    }
    if args.evaluation_mode == "historical_calibrated_transfer":
        gate["historical_prefix"] = {"prefix_end": args.historical_prefix_end,
                                     "authorization_id": args.historical_prefix_authorization_id}
    elif args.historical_prefix_end is not None or args.historical_prefix_authorization_id is not None:
        raise ValueError("source-only gate cannot declare target historical calibration")
    args.gate_json.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(args.gate_json, gate)
    try:
        validate_frozen_gate(args.gate_json, args.checkpoint, checkpoint, source_root=ROOT)
    except BaseException:
        args.gate_json.unlink()
        raise
    return {"status": "explicit_frozen_gate_created", "gate_json": str(args.gate_json.resolve()),
            "gate_sha256": file_sha256(args.gate_json), "heldout_data_read": False,
            "candidate_sha": gate["candidate_sha"], "modelcheckpoint_sha": gate["modelcheckpoint_sha"],
            "heldout_region": gate["heldout_region"], "evaluation_mode": gate["evaluation_mode"]}


def _model_state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluation_source_snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Hash the evaluator's imported dependency closure separately from training."""
    root = root.resolve()
    required = {
        "scripts/evaluate_ndr_transfer.py", "ndr_research/transfer_evaluation.py",
        "ndr_research/transfer_data.py", "ndr_research/shared_graph.py",
        "ndr_research/graph_loss.py", "scripts/train_ndr_transfer.py",
    }
    paths = {root / name for name in required}
    pending = [sys.modules[name] for name in
               (__name__, "ndr_research.transfer_evaluation", "ndr_research.transfer_data",
                "ndr_research.shared_graph", "scripts.train_ndr_transfer") if name in sys.modules]
    visited = set()
    while pending:
        module = pending.pop()
        if id(module) in visited:
            continue
        visited.add(id(module))
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(root):
                relative = path.relative_to(root)
                if relative.parts[0] not in {"tests", "outputs"}:
                    paths.add(path)
        for value in tuple(vars(module).values()):
            dependency = value if isinstance(value, types.ModuleType) else sys.modules.get(getattr(value, "__module__", None))
            filename = getattr(dependency, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and path.is_relative_to(root) and path.relative_to(root).parts[0] not in {"tests", "outputs"}:
                    pending.append(dependency)
    files = {}
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError("evaluation source dependency is missing or is a symlink")
        files[path.relative_to(root).as_posix()] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return {"files": files, "files_digest": canonical_sha256(files), "source_root": str(root),
            "scope": "evaluation_entrypoint_and_imported_workspace_dependency_closure",
            "separate_from_candidate_training_identity": True}


def run_evaluation(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    gate = validate_frozen_gate(args.gate_json, args.checkpoint, checkpoint, source_root=ROOT)
    if args.output is None:
        raise ValueError("held-out evaluation requires a new explicit output directory")
    if (args.source_verification is not None or args.authorization_id is not None
            or args.evaluation_ledger is not None or args.historical_prefix_end is not None
            or args.historical_prefix_authorization_id is not None):
        raise ValueError("evaluation mode obtains all authorization and prefix settings from gate-json")
    if args.batch_size <= 0 or args.threads <= 0:
        raise ValueError("batch size and CPU thread count must be positive")
    device = configure_transfer_runtime(args.device, gate.seed, args.threads)
    evaluation_snapshot = evaluation_source_snapshot()
    require_candidate_sources(evaluation_snapshot, Path(evaluation_snapshot["source_root"]))
    source_preprocessing = TransferPreprocessing.from_state(checkpoint["source_statistics"])
    preprocessing = source_preprocessing
    if preprocessing.mode != "source_only" or preprocessing.identity != checkpoint["source_statistics_identity"]:
        raise ValueError("checkpoint source-statistics payload differs from its frozen identity")
    model = SharedGraphNDR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    state_before = _model_state_sha(model)
    args.output.mkdir(parents=True, exist_ok=False)
    gate_bytes = args.gate_json.read_bytes()
    if hashlib.sha256(gate_bytes).hexdigest() != gate.gate_sha256:
        raise RuntimeError("gate bytes changed after admission")
    with (args.output / "gate.json").open("xb") as handle:
        handle.write(gate_bytes)
    gate_declaration = json.loads(gate_bytes)
    source_verification = Path(gate_declaration["source_validation_verification"])
    if not source_verification.is_absolute():
        source_verification = args.gate_json.resolve().parent / source_verification
    verification_bytes = source_verification.read_bytes()
    if hashlib.sha256(verification_bytes).hexdigest() != gate.source_verification_sha256:
        raise RuntimeError("source-verification bytes changed after admission")
    with (args.output / "source_verification.json").open("xb") as handle:
        handle.write(verification_bytes)
    write_new_json(args.output / "evaluation_source_snapshot.json", evaluation_snapshot)
    ledger = EvaluationLedger(gate, args.output)
    reservation = ledger.reserve()
    reader = None
    try:
        reader = TransferRegionReader(gate.heldout_region, args.data_dir, args.graph_dir,
                                      role="target", bundle_manifest=args.bundle_manifest)
        indices = reader.indices(gate.split)
        if gate.mode == "historical_calibrated_transfer":
            if gate.prefix_end is None or gate.prefix_end >= int(reader.sample_start[indices].min()):
                raise ValueError("calibration prefix must be frozen strictly before the first evaluation origin")
            preprocessing = fit_historical_calibration(reader, prefix_end=gate.prefix_end,
                                                       authorization_id=gate.prefix_authorization_id)
        elif preprocessing.mode != "source_only":
            raise PermissionError("strict source-only evaluation cannot use calibrated target statistics")
        reader_gate = TestEvaluationGate(gate.authorization_id, gate.checkpoint_sha256)
        def source_control_batch(batch):
            result = dict(batch)
            origins = batch["sample_start"]
            history_indices = origins[:, None] + np.arange(-reader.T_h + 1, 1)
            future_indices = origins[:, None] + np.arange(1, reader.T_p + 1)
            result["x_baseline"] = np.stack([source_preprocessing.lookup(reader.dow[index], reader.tod[index], reader.N).T
                                              for index in history_indices])
            result["y_baseline"] = np.stack([source_preprocessing.lookup(reader.dow[index], reader.tod[index], reader.N).T
                                              for index in future_indices])
            result["node_scale"] = source_preprocessing.scales(reader.N)
            return result
        evaluated = evaluate_frozen_batches(
            model, indices=indices, times_ns=reader.times_ns, device=device,
            batch_size=args.batch_size, seed=gate.seed,
            get_batch=lambda selected: reader.get_batch(selected, preprocessing, split=gate.split,
                                                        evaluation_gate=reader_gate),
            controls={kind: HistoryBaseline(kind).to(device) for kind in CONTROL_KINDS},
            control_batch_factory=source_control_batch,
        )
        if state_before != _model_state_sha(model):
            raise RuntimeError("held-out evaluation changed frozen model parameters or buffers")
        require_candidate_sources(checkpoint["config"]["source_snapshot"], ROOT)
        require_candidate_sources(evaluation_snapshot, Path(evaluation_snapshot["source_root"]))
        if file_sha256(args.checkpoint) != gate.checkpoint_sha256:
            raise RuntimeError("checkpoint bytes changed during held-out evaluation")
        if file_sha256(args.gate_json) != gate.gate_sha256:
            raise RuntimeError("freeze gate changed during held-out evaluation")
        report = {
            "schema_version": 1, "status": "frozen_transfer_evaluation_complete",
            "claim_scope": "frozen_candidate_evaluation_no_sota_or_winner_claim",
            "candidate_sha": gate.candidate_sha, "checkpoint_sha256": gate.checkpoint_sha256,
            "gate_sha256": gate.gate_sha256, "authorization_id": gate.authorization_id,
            "source_verification_sha256": gate.source_verification_sha256,
            "source_validation_selection_sha256": gate.selection_sha256,
            "source_regions": list(gate.source_regions), "heldout_region": gate.heldout_region,
            "split": gate.split, "evaluation_mode": gate.mode,
            "source_statistics_identity": checkpoint["source_statistics_identity"],
            "evaluation_preprocessing_identity": dict(preprocessing.identity),
            "control_preprocessing_identity": dict(source_preprocessing.identity),
            "target_reader_identity": reader.identity, "target_access_audit": reader.access_audit(),
            "model_config": checkpoint["model_config"], "model_state_sha256": state_before,
            "runtime": transfer_runtime_identity(device) | {
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "evaluation_seed": gate.seed,
            }, "batch_size": args.batch_size,
            "evaluation_source_snapshot": evaluation_snapshot,
            "evaluation_source_snapshot_unchanged_after": True,
            "evaluation_ledger": str(gate.ledger_path), "attempt_id": ledger.attempt_id,
            "test_access_attempt_number": reservation["test_access_attempt_number"],
            "target_val_access_attempt_number": reservation["target_val_access_attempt_number"],
            "test_evaluation_budget_scope": reservation["family_budget_scope"],
            "test_evaluation_limit": gate.max_test_evaluations,
            "target_validation_used_for_selection": False, "target_gradient_updates": False,
            **evaluated,
        }
        write_new_json(args.output / "report.json", report)
        ledger.finish(status="completed", report_sha256=file_sha256(args.output / "report.json"))
        return report
    except BaseException as error:
        failure = {"status": "frozen_transfer_evaluation_failed", "error_type": type(error).__name__,
                   "checkpoint_sha256": gate.checkpoint_sha256, "gate_sha256": gate.gate_sha256,
                   "evaluation_source_snapshot": evaluation_snapshot,
                   "attempt_id": ledger.attempt_id,
                   "target_access_audit": reader.access_audit() if reader is not None else None}
        write_new_json(args.output / "failure.json", failure)
        ledger.finish(status="failed", error_type=type(error).__name__)
        raise


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    checkpoint = load_transfer_checkpoint(args.checkpoint)
    result = create_explicit_gate(args, checkpoint) if args.mode == "create-gate" else run_evaluation(args, checkpoint)
    print(json.dumps({key: result.get(key) for key in
                     ("status", "gate_json", "candidate_sha", "checkpoint_sha256", "heldout_region", "evaluation_mode")},
                     sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
