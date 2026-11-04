"""Synthetic-only gate and pairing tests; no real heldout reader is constructed."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.shared_staeformer import SHARED_STAEFORMER_CONFIG, SharedSTAEformer
from ndr_research.source_cohort_evaluation import (
    CERTIFICATE_SCHEMA, CLASSES, GATE_SCHEMA, KINDS, MODES, RECEIPT_SCHEMA, REPORT_STATUS, SEEDS,
    SourceCohortLedger, admit_source_cohort_gate, evaluate_source_cohort,
    inspect_source_model, source_pool_digest,
)
from ndr_research.transfer_data import PROTOCOL, TransferPreprocessing, _array_hash
from ndr_research.transfer_evaluation import CONTROL_KINDS, SCENARIOS, HistoryBaseline, canonical_sha256, file_sha256
from scripts import evaluate_ndr_source_cohort as cli
from scripts.train_ndr_transfer import checkpoint_safe, configure_transfer_runtime, transfer_runtime_identity

ROOT = Path(__file__).resolve().parents[2]


def source_pool():
    template, scale = np.full((7, 288), 40, np.float32), np.asarray(5, np.float32)
    identity = {"mode": "source_only", "bundle_id": "synthetic_bundle", "bundle_digest": "a" * 64,
                "source_regions": ["Alameda", "ContraCosta"], "source_identities": {"Alameda": "alameda", "ContraCosta": "contra"},
                "target_statistics_used": False, "template_sha256": _array_hash(template), "scale_sha256": _array_hash(scale),
                "fit_ranges": {"Alameda": [0, 20], "ContraCosta": [0, 20]},
                "fit_values": "synthetic_source_train_observations", "access_audits": {}}
    identity["identity_digest"] = canonical_sha256(identity)
    return TransferPreprocessing("source_only", template, float(scale), identity)


def synthetic_source_selection():
    rows = {}
    for region in ("Alameda", "ContraCosta"):
        selection = {"selection_split": "source_val", "samples": 2, "all_validation_samples": True,
                     "reader_identity": {"role": "source", "region": region, "counts": {"val": 2}},
                     "forecast_origins_sha256": "b" * 64, "sample_indices_sha256": "c" * 64}
        rows[region] = {"selection": selection, "valid_count": 72, "global_masked_mae": 5.0,
                        "observed_targets_sha256": "d" * 64, "observation_masks_sha256": "e" * 64}
    return {"selection_split": "source_val", "all_source_validation_samples": True, "target_region_used": False,
            "regions": rows, "source_regions": list(rows), "macro_global_mae": 5.0}


def write_synthetic_acceptance(root, kind, seed):
    """Construct test-only formal-shaped artifacts, without a performance claim."""
    root.mkdir()
    pool, selection = source_pool(), synthetic_source_selection()
    model = SharedGraphNDR(hidden_dim=8, num_heads=2) if kind == "shared_graph" else SharedSTAEformer(flow_center=40, flow_scale=5)
    required = ["ndr_research/shared_graph.py", "ndr_research/transfer_data.py", "scripts/train_ndr_transfer.py", "ndr_research/graph_loss.py",
                "ndr_research/shared_staeformer.py", "baselines/STAEformer/STAEformer.py", "scripts/train_shared_staeformer_transfer.py"]
    files = {name: {"sha256": file_sha256(ROOT / name), "size_bytes": (ROOT / name).stat().st_size} for name in required}
    snapshot = {"files": files, "files_digest": canonical_sha256(files), "git": {"git_sha": "synthetic_only"}}
    identity = {"source_regions": ["Alameda", "ContraCosta"], "heldout_region": "Orange", "mode": "source_only"}
    config = {"schema_version": 1, "model_class": CLASSES[kind], "model_config": model.get_config(),
              "architecture": dict(SHARED_STAEFORMER_CONFIG) if kind == "shared_staeformer" else model.get_config(),
              "seed": seed, "epochs": 30, "source_regions": ["Alameda", "ContraCosta"], "heldout_region": "Orange",
              "max_train_batches": None, "max_val_batches": None, "claim_scope": "source_validation_development_only",
              "complete_source_train_epoch": True, "target_data_read": False, "test_evaluation": False,
              "source_snapshot": snapshot, "source_statistics_identity": pool.identity, "data_identity": identity,
              "runtime": transfer_runtime_identity(torch.device("cpu")), "learning_rate": 0.001, "weight_decay": 0.0001,
              "eta_min": 0.00001, "gradient_clip": 5.0, "batch_size": 2, "microbatch_size": 1, "threads": 1,
              "loading_policy": "source_lazy_raw_windows", "log_interval": 100,
              "loss": "synthetic_graph_auxiliary_loss" if kind == "shared_graph" else "pure_raw_flow_masked_mae"}
    checkpoint = {"schema_version": 1, "model_class": CLASSES[kind], "config": config, "config_digest": canonical_sha256(config),
                  "model_config": model.get_config(), "model": model.state_dict(), "source_statistics": pool.to_state(),
                  "source_statistics_identity": pool.identity, "data_identity": identity,
                  "selection_metric": selection, "selection_split": "source_val", "target_data_read": False,
                  "source_regions": ["Alameda", "ContraCosta"], "heldout_region": "Orange",
                  "optimizer": {}, "scheduler": {}, "rng": {}, "loaders": {}, "progress": {"best_epoch": 2}}
    path = root / "best.pt"
    torch.save(checkpoint_safe(checkpoint), path)
    (root / "best.pt.sha256.json").write_text(json.dumps({"sha256": file_sha256(path), "size_bytes": path.stat().st_size,
                                                       "config_digest": checkpoint["config_digest"]}))
    verification = {"schema_version": 1, "status": "verified_transfer_source_checkpoint_complete_validation",
                    "checkpoint_sha256": file_sha256(path), "config_digest": checkpoint["config_digest"],
                    "source_statistics_identity": pool.identity, "source_snapshot_files_digest": snapshot["files_digest"],
                    "source_files_digest": snapshot["files_digest"], "selection_metric_digest": canonical_sha256(selection),
                    "model_config_sha256": canonical_sha256(model.get_config()), "source_regions": list(selection["regions"]),
                    "heldout_region": "Orange", "all_source_validation_samples": True,
                    "claim_scope": config["claim_scope"], "target_data_read": False, "test_evaluated": False, "source_validation": selection}
    verification_path = root / "verification.json"
    verification_path.write_text(json.dumps(verification))
    summary = {"status": "training_completed_source_validation_pending_independent_verification", "completed_epochs": 30,
               "best_epoch": 2, "config_digest": checkpoint["config_digest"], "test_evaluated": False, "target_data_read": False,
               "source_regions": list(selection["regions"]), "heldout_region": "Orange", "best_source_validation": selection}
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary))
    certificate = {"schema_version": CERTIFICATE_SCHEMA, "status": "certified_independent_complete_source_validation",
                   "model_class": CLASSES[kind], "checkpoint_sha256": file_sha256(path),
                   "verification_sha256": file_sha256(verification_path), "summary_sha256": file_sha256(summary_path),
                   "config_digest": checkpoint["config_digest"], "source_snapshot_files_digest": snapshot["files_digest"],
                   "selection_metric_digest": canonical_sha256(selection), "source_pool_numeric_semantics_sha256": source_pool_digest(pool),
                   "runtime": config["runtime"], "data_identity": identity, "verification_process_id": 12345,
                   "verifier_process_started_ns": 200, "checkpoint_mtime_ns": 100,
                   "verification_execution": "fresh_cli_process_checkpoint_reconstruction", "checkpoint_existed_before_verifier_entry": True,
                   "source_validation_complete": True, "source_validation": selection, "target_data_read": False,
                   "test_evaluated": False, "training_executed_in_verifier": False}
    certificate_path = root / "certificate.json"
    certificate_path.write_text(json.dumps(certificate))
    return path, verification_path, summary_path, certificate_path


def build_gate(tmp_path):
    declarations = []
    for kind in KINDS:
        for seed in SEEDS:
            paths = write_synthetic_acceptance(tmp_path / f"{kind}_{seed}", kind, seed)
            spec = inspect_source_model(*paths[:3], kind=kind, certificate_path=paths[3])
            declarations.append({"model_id": f"{kind}_s{seed}", "checkpoint_kind": kind,
                                 **{name: str(path) for name, path in zip(("checkpoint", "verification", "summary", "certificate"), paths)},
                                 **{name + "_sha256": file_sha256(path) for name, path in zip(("checkpoint", "verification", "summary", "certificate"), paths)},
                                 **{name: spec[name] for name in ("family_identity", "model_class", "training_seed", "config_digest",
                                                                "validation_selection_sha256", "data_identity_sha256",
                                                                "source_pool_numeric_semantics_sha256", "model_state_sha256")}})
    gate = {"schema_version": GATE_SCHEMA, "protocol_version": PROTOCOL, "sampling_protocol": "event", "split": "test",
            "structure_development_closed": True, "selection_frozen": True, "test_used_for_structure_selection": False,
            "authorization_id": "synthetic_only_gate", "comparison_scope": "matched_two_source_training",
            "training_seeds": list(SEEDS), "scenarios": list(SCENARIOS), "control_kinds": list(CONTROL_KINDS),
            "permitted_evaluation_modes": list(MODES), "evaluation_mode": "source_only", "max_test_evaluations": 12,
            "evaluation_ledger": str(tmp_path / "ledger.jsonl"), "perturbation_seed": 42, "models": declarations,
            "cache_bundle_digest": "a" * 64, "region": "Orange", "source_regions": ["Alameda", "ContraCosta"]}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    return path, gate


class SyntheticTargetReader:
    role, region, N = "target", "Orange", 3

    def __init__(self):
        self.contract = SimpleNamespace(bundle_id="synthetic_bundle", bundle_digest="a" * 64)
        self.identity = {"identity_digest": "synthetic_orange"}
        self.times_ns = (np.datetime64("2025-01-01") + np.arange(1000) * np.timedelta64(5, "m")).astype("datetime64[ns]").astype(np.int64)
        self.tod, self.dow = np.arange(1000) % 288, (np.arange(1000) // 288) % 7
        self.origins = np.asarray([20, 30, 310, 620], dtype=np.int64)
        self.reads = []

    def indices(self, split):
        assert split == "test"
        return np.arange(4, dtype=np.int64)

    def get_batch(self, indices, preprocessing, *, split, evaluation_gate):
        assert split == "test" and evaluation_gate is not None
        self.reads.extend(indices.tolist())
        origins = self.origins[indices]
        shape = (len(indices), self.N, 12)
        flow, target = np.full(shape, 30, np.float32), np.full(shape, 45, np.float32)
        mask, observed = np.ones(shape, bool), np.ones(shape, bool)
        mask[:, 0, 0] = False
        target[:, 0, 0] = np.nan
        mask[:, 1, 1] = True
        target[:, 1, 1] = 0
        observed[:, 0, -1] = False
        history = origins[:, None] + np.arange(-11, 1)
        future = origins[:, None] + np.arange(1, 13)
        return {"flow_hist": flow, "history_mask": observed, "target": target, "target_mask": mask,
                "x_baseline": np.stack([preprocessing.lookup(self.dow[row], self.tod[row], self.N).T for row in history]),
                "y_baseline": np.stack([preprocessing.lookup(self.dow[row], self.tod[row], self.N).T for row in future]),
                "time_feat": np.stack((self.tod[history] / 288, self.dow[history] / 7), -1).astype(np.float32),
                "sample_start": origins.copy(), "edge_index": np.asarray([[0, 1], [1, 2]], dtype=np.int64),
                "node_scale": preprocessing.scales(self.N)}

    def access_audit(self):
        return {"region": self.region, "role": self.role, "synthetic_only": True, "samples_read": len(self.reads)}


def test_true_source_cohort_gate_binds_six_models_and_rejects_artifact_changes(tmp_path):
    configure_transfer_runtime("cpu", 42, 1)
    path, gate = build_gate(tmp_path)
    admitted_gate, admitted = admit_source_cohort_gate(path)
    assert len(admitted) == 6 and {spec["model_class"] for spec in admitted.values()} == {"SharedGraphNDR", "SharedSTAEformer"}
    assert all(spec["training_scope"] == "two_source_supervised_target_disjoint_event" for spec in admitted.values())
    assert all(spec["training_recipe"] for spec in admitted.values())
    assert admitted_gate["gate_sha256"] == file_sha256(path)
    changed = copy.deepcopy(gate)
    changed["models"][0]["model_class"] = "SharedSTAEformer"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="model_class"):
        admit_source_cohort_gate(path)
    changed = copy.deepcopy(gate)
    changed["models"].pop()
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="three frozen seeds"):
        admit_source_cohort_gate(path)


def test_source_acceptance_rejects_subset_incomplete_training_and_bad_process_proof(tmp_path):
    configure_transfer_runtime("cpu", 42, 1)
    paths = write_synthetic_acceptance(tmp_path / "source", "shared_staeformer", 42)
    original = paths[2].read_text()
    summary = json.loads(original)
    summary["completed_epochs"] = 29
    paths[2].write_text(json.dumps(summary))
    with pytest.raises(PermissionError, match="completed its frozen budget"):
        inspect_source_model(*paths[:3], kind="shared_staeformer", certificate_path=paths[3])
    paths[2].write_text(original)
    certificate = json.loads(paths[3].read_text())
    certificate["verifier_process_started_ns"] = 99
    paths[3].write_text(json.dumps(certificate))
    with pytest.raises(PermissionError, match="before the verifier process"):
        inspect_source_model(*paths[:3], kind="shared_staeformer", certificate_path=paths[3])
    verification = json.loads(paths[1].read_text())
    verification["source_validation"]["regions"]["Alameda"]["selection"]["all_validation_samples"] = False
    paths[1].write_text(json.dumps(verification))
    with pytest.raises(PermissionError, match="subset"):
        inspect_source_model(*paths[:3], kind="shared_staeformer", certificate_path=paths[3])


def test_frozen_runner_receipt_reuses_the_complete_independent_validation(tmp_path):
    configure_transfer_runtime("cpu", 42, 1)
    paths = write_synthetic_acceptance(tmp_path / "source", "shared_staeformer", 42)
    receipt = json.loads(paths[3].read_text())
    receipt.update(schema_version=RECEIPT_SCHEMA, status="recorded_independent_complete_source_validation",
                   verification_execution="fresh_runner_verify_process", returncode=0, runner_mode="verify")
    receipt.pop("source_validation")
    paths[3].write_text(json.dumps(receipt))
    admitted = inspect_source_model(*paths[:3], kind="shared_staeformer", certificate_path=paths[3])
    assert admitted["certificate_sha256"] == file_sha256(paths[3])
    receipt["returncode"] = 1
    paths[3].write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="successful verify-only"):
        inspect_source_model(*paths[:3], kind="shared_staeformer", certificate_path=paths[3])


def test_three_seed_ledger_counts_failure_and_does_not_reset_for_mode_or_prefix(tmp_path):
    configure_transfer_runtime("cpu", 42, 1)
    path, _ = build_gate(tmp_path)
    gate, admitted = admit_source_cohort_gate(path)
    first = SourceCohortLedger(gate, admitted, path, tmp_path / "first")
    assert first.reserve()["training_seed_inference_slots"] == 3
    first.finish("failed", error_type="SyntheticFailure")
    changed = dict(gate, evaluation_mode="historical_calibrated_transfer", historical_prefix={"prefix_end": 10, "authorization_id": "prefix"})
    second = SourceCohortLedger(changed, admitted, path, tmp_path / "second")
    assert second.family_sha == first.family_sha
    assert second.reserve()["consumed_slots_after"] == 6
    second.finish("completed")
    changed["max_test_evaluations"] = 6
    with pytest.raises(PermissionError, match="cannot change after first access"):
        SourceCohortLedger(changed, admitted, path, tmp_path / "third").reserve()
    for number in (3, 4):
        current = SourceCohortLedger(gate, admitted, path, tmp_path / str(number))
        current.reserve()
        current.finish("failed")
    with pytest.raises(PermissionError, match="budget is exhausted"):
        SourceCohortLedger(gate, admitted, path, tmp_path / "fifth").reserve()


def test_real_model_comparison_reads_windows_once_and_pairs_days_masks_and_scenarios():
    configure_transfer_runtime("cpu", 42, 1)
    reader, pool = SyntheticTargetReader(), source_pool()
    models = {"graph": SharedGraphNDR(hidden_dim=8, num_heads=2).eval(),
              "staeformer": SharedSTAEformer(flow_center=40, flow_scale=5).eval()}
    result = evaluate_source_cohort(models, reader=reader, preprocessing=pool, source_preprocessing=pool,
                                    evaluation_grant=object(), batch_size=2)
    assert reader.reads == [0, 1, 2, 3]
    assert result["target_windows_read_once_for_all_models"] and not result["targets_used_for_forward"]
    assert set(result["models"]) == {"graph", "staeformer", *CONTROL_KINDS}
    for current in result["models"].values():
        assert set(current["scenarios"]) == set(SCENARIOS)
        for scenario in current["scenarios"].values():
            assert scenario["collection"] == result["collection"]
            assert scenario["valid_count"] == 4 * 35
            assert scenario["percentage_valid_count"] == 4 * 34
            assert set(scenario["day_blocks"]) == {"2025-01-01", "2025-01-02", "2025-01-03"}
            assert sum(sum(block["valid_count"]) for block in scenario["day_blocks"].values()) == 4 * 35
    assert result["models"]["graph"]["scenarios"]["last_missing_6steps"]["history_observed_points_removed"] > 0


def test_historical_mode_keeps_source_controls_and_stae_scalar_frozen():
    configure_transfer_runtime("cpu", 42, 1)
    reader, pool = SyntheticTargetReader(), source_pool()
    template, scales = np.full((7, 288, 3), 75, np.float32), np.full(3, 8, np.float32)
    identity = {"mode": "historical_calibrated_transfer", "bundle_id": "synthetic_bundle", "bundle_digest": "a" * 64,
                "target_region": "Orange", "target_identity_digest": "synthetic_orange", "prefix_end": 10,
                "template_sha256": _array_hash(template), "scale_sha256": _array_hash(scales)}
    identity["identity_digest"] = canonical_sha256(identity)
    calibrated = TransferPreprocessing("historical_calibrated_transfer", template, scales, identity)
    stae = SharedSTAEformer(flow_center=40, flow_scale=5).eval()
    models = {"staeformer": stae, "other": HistoryBaseline("source_only_persistence").eval()}
    result = evaluate_source_cohort(models, reader=reader, preprocessing=calibrated, source_preprocessing=pool,
                                    evaluation_grant=object(), batch_size=2)
    assert stae.flow_center.item() == 40 and stae.flow_scale.item() == 5
    climatology = result["models"]["source_only_pooled_climatology"]["scenarios"]["clean"]
    assert climatology["global_masked_mae"] == pytest.approx((34 * 5 + 40) / 35)


def test_mutating_state_fails_closed_and_certification_requires_fresh_cli():
    configure_transfer_runtime("cpu", 42, 1)
    class Mutating(HistoryBaseline):
        def __init__(self):
            super().__init__("source_only_persistence")
            self.register_buffer("calls", torch.zeros(1))

        def forward(self, flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale):
            self.calls.add_(1)
            return super().forward(flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale)
    reader, pool = SyntheticTargetReader(), source_pool()
    with pytest.raises(RuntimeError, match="mutated frozen model state"):
        evaluate_source_cohort({"mutating": Mutating().eval(), "second": HistoryBaseline("source_only_persistence").eval()},
                               reader=reader, preprocessing=pool, source_preprocessing=pool, evaluation_grant=object())
    with pytest.raises(PermissionError, match="independent CLI"):
        cli.certify_source(cli.argument_parser().parse_args(["--mode", "certify-source"]))


def test_end_to_end_cli_report_keeps_true_classes_and_failure_consumes_budget(tmp_path, monkeypatch):
    configure_transfer_runtime("cpu", 42, 1)
    path, _ = build_gate(tmp_path)
    reader = SyntheticTargetReader()
    monkeypatch.setattr(cli, "TransferRegionReader", lambda *_args, **_kwargs: reader)
    args = cli.argument_parser().parse_args(["--gate-json", str(path), "--output", str(tmp_path / "report"), "--batch-size", "2"])
    result = cli.run_evaluation(args)
    assert result["status"] == REPORT_STATUS and reader.reads == [0, 1, 2, 3]
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["status"] == REPORT_STATUS and report["method_comparison_is_gate_ablation"] is False
    assert report["training_seeds"] == list(SEEDS) and report["staeformer_scalar_normalization_remains_source_frozen"]
    assert len(report["models"]) == 8
    assert {row["model_class"] for row in report["models"].values()} == {"SharedGraphNDR", "SharedSTAEformer", "HistoryBaseline"}
    assert all(row["collection"] == report["collection"] for row in report["models"].values())
    def failed_reader(*_args, **_kwargs):
        raise RuntimeError("synthetic reader unavailable")
    monkeypatch.setattr(cli, "TransferRegionReader", failed_reader)
    args.output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="unavailable"):
        cli.run_evaluation(args)
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "failed"
    assert sum(row["training_seed_inference_slots"] for row in rows if row["status"] == "reserved") == 6


def test_gate_requires_exact_fixed_scenarios_and_budget_before_any_target_access(tmp_path, monkeypatch):
    configure_transfer_runtime("cpu", 42, 1)
    path, gate = build_gate(tmp_path)
    def forbidden_reader(*_args, **_kwargs):
        pytest.fail("gate rejection constructed a target reader")
    monkeypatch.setattr(cli, "TransferRegionReader", forbidden_reader)
    changed = copy.deepcopy(gate)
    changed["scenarios"] = ["clean"]
    path.write_text(json.dumps(changed))
    args = cli.argument_parser().parse_args(["--gate-json", str(path), "--output", str(tmp_path / "rejected")])
    with pytest.raises(ValueError, match="five scenarios"):
        cli.run_evaluation(args)
    changed = copy.deepcopy(gate)
    changed["max_test_evaluations"] = 3
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="six or twelve"):
        cli.run_evaluation(args)
    assert not (tmp_path / "rejected").exists()
    with pytest.raises(ValueError, match="cannot override"):
        cli.main(["--gate-json", str(path), "--output", str(tmp_path / "rejected"), "--evaluation-mode", "source_only"])
