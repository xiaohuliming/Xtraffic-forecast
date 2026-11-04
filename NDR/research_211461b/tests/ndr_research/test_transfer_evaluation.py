from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.transfer_data import (
    FIVE_MINUTES_NS,
    TestEvaluationGate as ReaderTestGate,
    TransferRegionReader,
    fit_source_only_pool,
)
from ndr_research.transfer_evaluation import (
    FORWARD_FIELDS,
    CANDIDATE_KINDS,
    CONTROL_KINDS,
    REQUIRED_CANDIDATE_FILES,
    SCENARIOS,
    VERIFIED_SOURCE_STATUS,
    EvaluationLedger,
    HistoryBaseline,
    MaskedForecastSums,
    canonical_sha256,
    evaluate_frozen_batches,
    file_sha256,
    paired_day_block_bootstrap,
    perturb_history,
    predict_history_only,
    validate_frozen_gate,
    write_new_json,
)
from scripts import evaluate_ndr_transfer as evaluator
from tests.ndr_research.test_transfer_data import make_fixture


@pytest.fixture(autouse=True)
def single_thread_cpu():
    torch.set_num_threads(1)


def numeric_batch(origins=(0, 2, 4, 290), *, nodes=1):
    origins = np.asarray(origins, dtype=np.int64)
    size = len(origins)
    history = np.arange(size * nodes * 12, dtype=np.float32).reshape(size, nodes, 12) + 1
    return {
        "flow_hist": history, "history_mask": np.ones(history.shape, dtype=bool),
        "x_baseline": np.zeros_like(history),
        "y_baseline": np.zeros((size, nodes, 12), dtype=np.float32),
        "time_feat": np.zeros((size, 12, 2), dtype=np.float32),
        "node_scale": np.ones(nodes, dtype=np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "target": np.full((size, nodes, 12), 5.0, dtype=np.float32),
        "target_mask": np.ones((size, nodes, 12), dtype=bool),
        "sample_start": origins,
    }


class ObservedHistoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forwarded = []

    def forward(self, flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale):
        self.forwarded.append(tuple(value.detach().cpu().clone() for value in
                                    (flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale)))
        return (flow_hist * history_mask).sum(-1, keepdim=True).expand_as(y_baseline) / 12


class ZeroModel(torch.nn.Module):
    def forward(self, flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale):
        return torch.zeros_like(y_baseline)


def slice_batch(batch, selection):
    return {key: value if key in ("node_scale", "edge_index") else value[selection].copy()
            for key, value in batch.items()}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_corruptions_only_remove_history_observations_and_preserve_targets(scenario):
    batch = numeric_batch(nodes=3)
    batch["history_mask"][0, 1, 0] = False
    batch["flow_hist"][0, 1, 0] = np.nan
    targets, target_mask = batch["target"].copy(), batch["target_mask"].copy()
    before = batch["history_mask"].copy()
    result = perturb_history(batch, scenario, seed=41)
    assert not (result["history_mask"] & ~before).any()
    assert np.all(result["flow_hist"][~result["history_mask"]] == 0)
    np.testing.assert_array_equal(result["flow_hist"][result["history_mask"]], batch["flow_hist"][result["history_mask"]])
    np.testing.assert_array_equal(result["target"], targets)
    np.testing.assert_array_equal(result["target_mask"], target_mask)
    np.testing.assert_array_equal(batch["history_mask"], before)
    assert np.isnan(batch["flow_hist"][0, 1, 0])
    if scenario.startswith("last_missing"):
        missing = 3 if "3steps" in scenario else 6
        assert not result["history_mask"][..., -missing:].any()
        np.testing.assert_array_equal(result["history_mask"][..., :-missing], before[..., :-missing])


@pytest.mark.parametrize("scenario", ("random_missing_10pct", "random_missing_30pct"))
def test_random_missingness_is_batch_and_order_independent(scenario):
    batch = numeric_batch(nodes=17)
    full = perturb_history(batch, scenario, seed=72)
    order = [3, 0, 2, 1]
    reordered = perturb_history(slice_batch(batch, order), scenario, seed=72)
    np.testing.assert_array_equal(reordered["history_mask"], full["history_mask"][order])
    for index in range(len(order)):
        alone = perturb_history(slice_batch(batch, [index]), scenario, seed=72)
        np.testing.assert_array_equal(alone["history_mask"][0], full["history_mask"][index])
    altered_seed = perturb_history(batch, scenario, seed=73)
    assert not np.array_equal(full["history_mask"], altered_seed["history_mask"])


def test_future_targets_and_labels_cannot_reach_seven_argument_forward():
    model = ObservedHistoryModel().eval()
    batch = perturb_history(numeric_batch(), "clean", seed=42)
    first = predict_history_only(model, batch, "cpu")
    changed = dict(batch)
    changed["target"] = np.full_like(batch["target"], np.nan)
    changed["target_mask"] = np.zeros_like(batch["target_mask"])
    second = predict_history_only(model, changed, "cpu")
    np.testing.assert_array_equal(first, second)
    assert len(model.forwarded[-1]) == len(FORWARD_FIELDS) == 7
    for original, altered in zip(model.forwarded[-2], model.forwarded[-1]):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    with pytest.raises(ValueError, match="whitelist"):
        predict_history_only(model, batch | {"affected_mask": np.ones(1)}, "cpu")
    model.train()
    with pytest.raises(PermissionError, match="eval-mode"):
        predict_history_only(model, batch, "cpu")


def test_missing_sentinels_cannot_change_real_model_forecasts():
    torch.manual_seed(42)
    model = SharedGraphNDR(hidden_dim=8, road_layers=2, latent_tokens=8, num_heads=2).eval()
    batch = numeric_batch(nodes=3)
    batch["history_mask"][..., ::3] = False
    poisoned = copy.deepcopy(batch)
    poisoned["flow_hist"][~poisoned["history_mask"]] = np.inf
    clean = perturb_history(batch, "random_missing_30pct", seed=72)
    poison = perturb_history(poisoned, "random_missing_30pct", seed=72)
    np.testing.assert_array_equal(predict_history_only(model, clean, "cpu"),
                                  predict_history_only(model, poison, "cpu"))


def test_global_metrics_use_all_points_and_independent_mape_denominator():
    target = np.zeros((2, 1, 12), dtype=np.float32)
    prediction = np.zeros_like(target)
    mask = np.zeros_like(target, dtype=bool)
    target[0, 0, :2] = [0, 10]
    target[1, 0, :2] = [2, 90]
    prediction[0, 0, :2] = [1, 8]
    prediction[1, 0, :2] = [1, 90]
    mask[0, 0, :2] = True
    mask[1, 0, 0] = True
    target[~mask] = np.nan
    total = MaskedForecastSums()
    total.update(prediction[:1], target[:1], mask[:1])
    total.update(prediction[1:], target[1:], mask[1:])
    metrics = total.result()
    assert metrics["valid_count"] == 3
    assert metrics["global_masked_mae"] == pytest.approx(4 / 3)
    assert metrics["global_masked_rmse"] == pytest.approx(np.sqrt(6 / 3))
    assert metrics["percentage_valid_count"] == 2
    assert metrics["global_masked_mape"] == pytest.approx((20 + 50) / 2)
    assert metrics["horizon_mae"][:2] == [1, 2]
    assert metrics["global_masked_mae"] != np.mean(metrics["horizon_mae"][:2])
    assert metrics["horizon_mae"][2:] == [None] * 10
    prediction[~mask] = np.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        total.update(prediction, target, mask)


def real_calendar_collection(batch_size):
    batch = numeric_batch()
    batch["target_mask"][:] = False
    for index, (value, observations) in enumerate(((1, 1), (9, 9), (9, 1), (3, 2))):
        batch["target"][index] = value
        batch["target_mask"][index, 0, :observations] = True
    times = np.datetime64("2024-01-01T23:50", "ns").astype(np.int64) + np.arange(400, dtype=np.int64) * FIVE_MINUTES_NS
    report = evaluate_frozen_batches(ZeroModel(), indices=np.arange(4, dtype=np.int64),
                                    get_batch=lambda selected: slice_batch(batch, selected),
                                    times_ns=times, batch_size=batch_size, seed=91)
    return report


def test_real_midnight_day_blocks_and_unequal_point_counts():
    report = real_calendar_collection(3)
    clean = report["scenarios"]["clean"]
    assert clean["valid_count"] == 13
    assert clean["global_masked_mae"] == pytest.approx(97 / 13)
    assert list(clean["day_blocks"]) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert sum(clean["day_blocks"]["2024-01-01"]["valid_count"]) == 1
    assert sum(clean["day_blocks"]["2024-01-02"]["valid_count"]) == 10
    assert sum(clean["day_blocks"]["2024-01-03"]["valid_count"]) == 2
    assert sum(clean["day_blocks"]["2024-01-02"]["absolute_error_sum"]) == 90
    for scenario in SCENARIOS:
        current = report["scenarios"][scenario]
        assert current["collection"] == report["collection"]
        assert current["valid_count"] == 13
        assert current["global_masked_mae"] == pytest.approx(97 / 13)
    assert report["targets_used_for_forward"] is False
    tail = report["scenarios"]["last_missing_6steps"]
    assert tail["history_observed_points_removed"] == 24


def test_collection_hashes_and_metrics_are_independent_of_evaluation_batch_size():
    first, second = real_calendar_collection(1), real_calendar_collection(3)
    assert first["collection"] == second["collection"]
    for scenario in SCENARIOS:
        for key in ("sums", "day_blocks", "global_masked_mae", "horizon_mae", "history_observed_points_removed"):
            assert first["scenarios"][scenario][key] == second["scenarios"][scenario][key]


def test_fixed_source_controls_are_paired_and_share_one_target_read():
    batch = numeric_batch(origins=(12, 13), nodes=2)
    batch["x_baseline"][:] = 17
    batch["y_baseline"][:] = 19
    batch["history_mask"][0, 0] = False
    batch["history_mask"][1, 1, -2:] = False
    expected_persistence = np.broadcast_to(np.asarray([[17, 24], [36, 46]], dtype=np.float32)[..., None], (2, 2, 12))
    clean = perturb_history(batch, "clean", seed=42)
    persistence = HistoryBaseline("source_only_persistence").eval()
    pooled = HistoryBaseline("source_only_pooled_climatology").eval()
    np.testing.assert_array_equal(predict_history_only(persistence, clean, "cpu"), expected_persistence)
    np.testing.assert_array_equal(predict_history_only(pooled, clean, "cpu"), np.full((2, 2, 12), 19, dtype=np.float32))
    reads = []
    def get_batch(indices):
        reads.append(indices.copy())
        return slice_batch(batch, indices)
    times = np.datetime64("2024-01-01T00:00", "ns").astype(np.int64) + np.arange(100) * FIVE_MINUTES_NS
    report = evaluate_frozen_batches(ZeroModel(), indices=np.arange(2, dtype=np.int64),
                                    get_batch=get_batch, times_ns=times, batch_size=2,
                                    controls={kind: HistoryBaseline(kind) for kind in CONTROL_KINDS})
    assert len(reads) == 1
    assert set(report["controls"]) == set(CONTROL_KINDS)
    for kind in CONTROL_KINDS:
        assert report["controls"][kind]["candidate_kind"] == kind
        assert report["controls"][kind]["source_only_control"] is True
        for scenario in SCENARIOS:
            assert report["controls"][kind]["scenarios"][scenario]["collection"] == report["collection"]
    assert report["controls"]["source_only_persistence"]["scenarios"]["clean"]["global_masked_mae"] == pytest.approx(25.75)


def test_controls_cannot_replace_target_collection_and_gate_freezes_both(tmp_path):
    batch = numeric_batch()
    times = np.datetime64("2024-01-01T00:00", "ns").astype(np.int64) + np.arange(400) * FIVE_MINUTES_NS
    def poisoned_preprocessor(original):
        result = dict(original)
        result["target"] = np.zeros_like(original["target"])
        return result
    with pytest.raises(ValueError, match="paired target"):
        evaluate_frozen_batches(ZeroModel(), indices=np.arange(4, dtype=np.int64),
                                get_batch=lambda index: slice_batch(batch, index), times_ns=times,
                                controls={kind: HistoryBaseline(kind) for kind in CONTROL_KINDS},
                                control_batch_factory=poisoned_preprocessor)
    root, path, checkpoint, _, _, gate_path, declaration = gate_fixture(tmp_path)
    declaration["candidate_kinds"] = ["shared_graph_ndr"]
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="both fixed source-only controls"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)


def test_paired_bootstrap_resamples_real_days_and_retains_point_weights():
    left = real_calendar_collection(2)["scenarios"]["clean"]
    right = copy.deepcopy(left)
    for block in right["day_blocks"].values():
        block["absolute_error_sum"] = [0.0] * 12
    result = paired_day_block_bootstrap(left, right, samples=10000, seed=91)
    assert result["estimate"] == pytest.approx(97 / 13)
    assert result["estimate"] != pytest.approx((1 + 9 + 3) / 3)
    assert result["ci95"] == pytest.approx([1, 9])
    assert result["forecast_day_blocks"] == 3
    assert result == paired_day_block_bootstrap(left, right, samples=10000, seed=91)
    right["day_blocks"]["2024-01-02"]["valid_count"][0] += 1
    with pytest.raises(ValueError, match="denominators"):
        paired_day_block_bootstrap(left, right, samples=100, seed=91)
    right = copy.deepcopy(left)
    right["collection"]["targets_sha256"] = "different"
    with pytest.raises(ValueError, match="identical"):
        paired_day_block_bootstrap(left, right, samples=100, seed=91)


def gate_fixture(tmp_path):
    root = tmp_path / "source"
    files = {}
    for relative in REQUIRED_CANDIDATE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
        files[relative] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    source_identity = {"mode": "source_only", "target_statistics_used": False,
                       "source_regions": ["Alameda", "ContraCosta"], "bundle_digest": "b" * 64}
    selection = {"selection_split": "source_val", "all_source_validation_samples": True,
                 "target_region_used": False, "macro_global_mae": 3.0,
                 "regions": {name: {"global_masked_mae": value, "valid_count": 13,
                                     "selection": {"all_validation_samples": True, "samples": 5},
                                     "observed_targets_sha256": "f" * 64,
                                     "observation_masks_sha256": "e" * 64}
                             for name, value in (("Alameda", 2.0), ("ContraCosta", 4.0))}}
    snapshot = {"files": files, "files_digest": canonical_sha256(files)}
    config = {"source_snapshot": snapshot, "claim_scope": "source_validation_development_only"}
    checkpoint = {"schema_version": 1, "model_class": "SharedGraphNDR", "model_config": {"hidden_dim": 8},
                  "config": config, "config_digest": canonical_sha256(config),
                  "source_regions": ["Alameda", "ContraCosta"], "heldout_region": "Orange",
                  "selection_split": "source_val", "target_data_read": False,
                  "selection_metric": selection, "source_statistics_identity": source_identity}
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"frozen source checkpoint")
    verified = {"status": VERIFIED_SOURCE_STATUS, "schema_version": 1,
                "checkpoint_sha256": file_sha256(checkpoint_path), "config_digest": checkpoint["config_digest"],
                "source_statistics_identity": source_identity, "source_snapshot_files_digest": snapshot["files_digest"],
                "selection_metric_digest": canonical_sha256(selection),
                "model_config_sha256": canonical_sha256(checkpoint["model_config"]),
                "source_regions": checkpoint["source_regions"], "heldout_region": "Orange",
                "all_source_validation_samples": True, "claim_scope": config["claim_scope"],
                "target_data_read": False, "test_evaluated": False, "source_validation": copy.deepcopy(selection)}
    verify_path = tmp_path / "verification.json"
    verify_path.write_text(json.dumps(verified))
    declaration = {"schema_version": 1, "authorization_id": "explicit-fixed-candidate",
                   "selection_frozen": True, "candidate_sha": snapshot["files_digest"],
                   "modelcheckpoint_sha": file_sha256(checkpoint_path),
                   "model_config_sha": canonical_sha256(checkpoint["model_config"]),
                   "source_regions": checkpoint["source_regions"], "heldout_region": "Orange",
                   "cache_bundle_digest": "b" * 64,
                   "source_validation_selection_sha": canonical_sha256(selection),
                   "source_validation_verification": str(verify_path),
                   "source_validation_verification_sha": file_sha256(verify_path),
                   "evaluation_mode": "source_only", "split": "test", "perturbation_seed": 42,
                   "scenarios": list(SCENARIOS), "max_test_evaluations": 1,
                   "candidate_kinds": list(CANDIDATE_KINDS),
                   "evaluation_ledger": str(tmp_path / "shared_evaluation_ledger.jsonl")}
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(declaration))
    return root, checkpoint_path, checkpoint, verify_path, verified, gate_path, declaration


def test_gate_requires_hashes_and_complete_verified_source_selection(tmp_path):
    root, path, checkpoint, verify_path, verified, gate_path, declaration = gate_fixture(tmp_path)
    gate = validate_frozen_gate(gate_path, path, checkpoint, source_root=root)
    assert gate.heldout_region == "Orange" and gate.mode == "source_only"
    verified["source_validation"]["regions"]["Alameda"]["observation_masks_sha256"] = "d" * 64
    verify_path.write_text(json.dumps(verified))
    declaration["source_validation_verification_sha"] = file_sha256(verify_path)
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="collection differs"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)


@pytest.mark.parametrize("field,value", (("status", "verified_smoke_complete_source_validation"),
                                          ("all_source_validation_samples", False),
                                          ("target_data_read", True), ("test_evaluated", True),
                                          ("selection_metric_digest", "c" * 64)))
def test_gate_rejects_formal_hash_bound_report_with_invalid_acceptance_fields(tmp_path, field, value):
    root, path, checkpoint, verify_path, verified, gate_path, declaration = gate_fixture(tmp_path)
    verified[field] = value
    verify_path.write_text(json.dumps(verified))
    declaration["source_validation_verification_sha"] = file_sha256(verify_path)
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(PermissionError, match="verification does not bind"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)


def test_gate_rejects_changed_model_sources_and_unfrozen_selection(tmp_path):
    root, path, checkpoint, _, _, gate_path, declaration = gate_fixture(tmp_path)
    source = root / "ndr_research/shared_graph.py"
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="source hash changed"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)
    source.write_text("VALUE = 1\n")
    declaration["selection_frozen"] = False
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(PermissionError, match="frozen selection"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)


def test_source_only_gate_rejects_hidden_target_calibration(tmp_path):
    root, path, checkpoint, _, _, gate_path, declaration = gate_fixture(tmp_path)
    declaration["historical_prefix"] = {"prefix_end": 123, "authorization_id": "hidden"}
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(PermissionError, match="target calibration prefix"):
        validate_frozen_gate(gate_path, path, checkpoint, source_root=root)


def test_test_budget_consumes_failed_attempts_and_different_checkpoint_cannot_reset_it(tmp_path):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    gate = validate_frozen_gate(gate_path, path, checkpoint, source_root=root)
    first = EvaluationLedger(gate, tmp_path / "first")
    assert first.reserve()["test_access_attempt_number"] == 1
    first.finish(status="failed", error_type="TimeoutError")
    second = EvaluationLedger(replace(gate, checkpoint_sha256="a" * 64), tmp_path / "second")
    with pytest.raises(PermissionError, match="budget is exhausted"):
        second.reserve()
    rows = [json.loads(line) for line in gate.ledger_path.read_text().splitlines()]
    assert [row["action"] for row in rows] == ["started", "failed"]
    assert rows[1]["previous_sha256"] == rows[0]["record_sha256"]


@pytest.mark.parametrize("first_split", ("val", "test"))
@pytest.mark.parametrize("new_limit", (5, 7))
def test_first_family_access_locks_test_limit_across_modes_and_checkpoint_seeds(tmp_path, first_split, new_limit):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    gate = replace(validate_frozen_gate(gate_path, path, checkpoint, source_root=root),
                   max_test_evaluations=6, split=first_split)
    first = EvaluationLedger(gate, tmp_path / "first")
    first.reserve()
    first.finish(status="completed")
    altered = replace(gate, max_test_evaluations=new_limit, checkpoint_sha256="a" * 64,
                      mode="historical_calibrated_transfer", split="test", prefix_end=40,
                      prefix_authorization_id="new-prefix-cannot-reset-family")
    with pytest.raises(PermissionError, match="frozen family.*limit cannot change"):
        EvaluationLedger(altered, tmp_path / "second").reserve()


def test_family_six_attempt_budget_spans_three_seeds_and_two_transfer_modes(tmp_path):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    gate = replace(validate_frozen_gate(gate_path, path, checkpoint, source_root=root), max_test_evaluations=6)
    counter = 0
    for seed, checkpoint_sha in ((42, "a" * 64), (1, "b" * 64), (2, "c" * 64)):
        for mode in ("source_only", "historical_calibrated_transfer"):
            counter += 1
            selected = replace(gate, checkpoint_sha256=checkpoint_sha, seed=seed, mode=mode)
            attempt = EvaluationLedger(selected, tmp_path / f"attempt{counter}")
            assert attempt.reserve()["test_access_attempt_number"] == counter
            attempt.finish(status="failed" if counter == 2 else "completed")
    seventh = replace(gate, checkpoint_sha256="d" * 64, prefix_end=41,
                      mode="historical_calibrated_transfer", prefix_authorization_id="another-prefix")
    with pytest.raises(PermissionError, match="budget is exhausted"):
        EvaluationLedger(seventh, tmp_path / "seventh").reserve()
    rows = [json.loads(line) for line in gate.ledger_path.read_text().splitlines()]
    started = [row for row in rows if row["action"] == "started"]
    assert len(started) == 6
    assert {row["family_max_test_evaluations"] for row in started} == {6}
    assert [row["test_access_attempt_number"] for row in started] == list(range(1, 7))


def test_target_val_attempts_are_recorded_independently_of_test_budget(tmp_path):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    gate = replace(validate_frozen_gate(gate_path, path, checkpoint, source_root=root), max_test_evaluations=6)
    for expected in (1, 2):
        attempt = EvaluationLedger(replace(gate, split="val"), tmp_path / f"val{expected}")
        row = attempt.reserve()
        assert row["target_val_access_attempt_number"] == expected
        assert row["test_access_attempt_number"] == 0
        attempt.finish(status="completed")
    test_row = EvaluationLedger(gate, tmp_path / "test").reserve()
    assert test_row["test_access_attempt_number"] == 1
    assert test_row["target_val_access_attempt_number"] == 2


def test_evaluation_source_changes_do_not_rewrite_candidate_training_identity(tmp_path):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    candidate_before = checkpoint["config"]["source_snapshot"]["files_digest"]
    for relative in ("scripts/evaluate_ndr_transfer.py", "ndr_research/transfer_evaluation.py"):
        (root / relative).write_text("EVALUATION_POLICY = 1\n")
    snapshot = evaluator.evaluation_source_snapshot(root)
    assert snapshot["separate_from_candidate_training_identity"] is True
    assert {"scripts/evaluate_ndr_transfer.py", "ndr_research/transfer_evaluation.py"} <= set(snapshot["files"])
    changed = root / "ndr_research/transfer_evaluation.py"
    changed.write_text("EVALUATION_POLICY = 2\n")
    from ndr_research.transfer_evaluation import require_candidate_sources
    with pytest.raises(ValueError, match="source hash changed"):
        require_candidate_sources(snapshot, root)
    assert evaluator.evaluation_source_snapshot(root)["files_digest"] != snapshot["files_digest"]
    # The selected training checkpoint remains valid; its code snapshot never
    # included an evaluator that the training process did not import.
    current = validate_frozen_gate(gate_path, path, checkpoint, source_root=root)
    assert current.candidate_sha == candidate_before


def test_ledger_tamper_and_existing_report_overwrite_fail_closed(tmp_path):
    root, path, checkpoint, _, _, gate_path, _ = gate_fixture(tmp_path)
    gate = validate_frozen_gate(gate_path, path, checkpoint, source_root=root)
    EvaluationLedger(gate, tmp_path / "first").reserve()
    original = gate.ledger_path.read_text()
    gate.ledger_path.write_text(original.replace("\"test\"", "\"val\""))
    with pytest.raises(ValueError, match="hash chain"):
        EvaluationLedger(gate, tmp_path / "second").reserve()
    report = tmp_path / "report.json"
    write_new_json(report, {"original": True})
    with pytest.raises(FileExistsError):
        write_new_json(report, {"replacement": True})
    assert json.loads(report.read_text()) == {"original": True}


def test_whitelisted_target_reader_uses_source_statistics_and_never_interprets_labels(tmp_path, monkeypatch):
    import h5py
    data, graph, contract = make_fixture(tmp_path)
    sources = {region: TransferRegionReader(region, data, graph, role="source", contract=contract)
               for region in ("Alameda", "ContraCosta")}
    preprocessing = fit_source_only_pool(sources)
    accesses = []
    original = h5py.Dataset.__getitem__
    def observe(dataset, selection):
        accesses.append(dataset.name)
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe)
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    gate = ReaderTestGate("synthetic-frozen-evaluation", "a" * 64)
    model = SharedGraphNDR(hidden_dim=8, road_layers=2, latent_tokens=8, num_heads=2)
    report = evaluate_frozen_batches(model, indices=target.indices("test"), times_ns=target.times_ns,
                                    get_batch=lambda selected: target.get_batch(selected, preprocessing,
                                                                                split="test", evaluation_gate=gate),
                                    batch_size=1)
    assert report["collection"]["samples"] == 2
    assert report["scenarios"]["clean"]["valid_count"] == 2 * target.N * 12
    assert set(accesses) <= {"/sample_start", "/split", "/times_ns", "/region_idx", "/flow_series_imputed", "/flow_mask"}
    assert preprocessing.mode == "source_only" and preprocessing.template.shape == (7, 288)
    assert preprocessing.identity["target_statistics_used"] is False
    assert report["targets_used_for_forward"] is False


@pytest.mark.parametrize("mode", ("source_only", "historical_calibrated_transfer"))
def test_cli_end_to_end_gate_precedes_target_labels_and_controls_remain_source_only(tmp_path, monkeypatch, mode):
    root, path, checkpoint, verify_path, verified, gate_path, declaration = gate_fixture(tmp_path / "gate_fixture")
    cache = tmp_path / "cache"
    cache.mkdir()
    data, graph, contract = make_fixture(cache)
    sources = {name: TransferRegionReader(name, data, graph, role="source", contract=contract)
               for name in ("Alameda", "ContraCosta")}
    preprocessing = fit_source_only_pool(sources)
    model = SharedGraphNDR(hidden_dim=8, road_layers=2, latent_tokens=8, num_heads=2)
    checkpoint.update(model_config=model.get_config(), model=model.state_dict(),
                      source_statistics=preprocessing.to_state(), source_statistics_identity=preprocessing.identity)
    verified["model_config_sha256"] = canonical_sha256(checkpoint["model_config"])
    verified["source_statistics_identity"] = preprocessing.identity
    verify_path.write_text(json.dumps(verified))
    declaration.update(evaluation_mode=mode, max_test_evaluations=2,
                       model_config_sha=canonical_sha256(checkpoint["model_config"]),
                       cache_bundle_digest=preprocessing.identity["bundle_digest"],
                       source_validation_verification_sha=file_sha256(verify_path))
    if mode == "historical_calibrated_transfer":
        declaration["historical_prefix"] = {"prefix_end": 40, "authorization_id": "fixed-past-only-prefix"}
    gate_path.write_text(json.dumps(declaration))
    opened = []
    def fixture_reader(*args, **kwargs):
        opened.append(args[0])
        return TransferRegionReader(*args, **kwargs, contract=contract)
    monkeypatch.setattr(evaluator, "ROOT", root)
    monkeypatch.setattr(evaluator, "TransferRegionReader", fixture_reader)
    args = evaluator.argument_parser().parse_args([
        "--checkpoint", str(path), "--gate-json", str(gate_path), "--output", str(tmp_path / "new_report"),
        "--data-dir", str(data), "--graph-dir", str(graph), "--batch-size", "1",
    ])
    saved = verified["status"]
    verified["status"] = "unverified"
    verify_path.write_text(json.dumps(verified))
    declaration["source_validation_verification_sha"] = file_sha256(verify_path)
    gate_path.write_text(json.dumps(declaration))
    with pytest.raises(PermissionError, match="verification does not bind status"):
        evaluator.run_evaluation(args, checkpoint)
    assert opened == [] and not args.output.exists()
    verified["status"] = saved
    verify_path.write_text(json.dumps(verified))
    declaration["source_validation_verification_sha"] = file_sha256(verify_path)
    gate_path.write_text(json.dumps(declaration))
    report = evaluator.run_evaluation(args, checkpoint)
    assert opened == ["Orange"]
    assert report["evaluation_mode"] == mode
    assert report["control_preprocessing_identity"]["target_statistics_used"] is False
    assert report["controls"]["source_only_pooled_climatology"]["scenarios"]["clean"]["global_masked_mae"] == pytest.approx(89980)
    assert report["controls"]["source_only_persistence"]["scenarios"]["clean"]["global_masked_mae"] == 0
    if mode == "historical_calibrated_transfer":
        assert report["evaluation_preprocessing_identity"]["prefix_end"] == 40
        assert report["evaluation_preprocessing_identity"]["target_statistics_used"] is True
    else:
        assert report["evaluation_preprocessing_identity"] == preprocessing.identity
    assert report["test_access_attempt_number"] == 1
    runtime = report["runtime"]
    assert runtime["deterministic_algorithms"] is True and runtime["cudnn_benchmark"] is False
    assert runtime["cudnn_deterministic"] is True
    assert runtime["cuda_matmul_allow_tf32"] is False and runtime["cudnn_allow_tf32"] is False
    assert runtime["flash_sdp"] is False and runtime["memory_efficient_sdp"] is False
    assert runtime["math_sdp"] is True and runtime["float32_matmul_precision"] == "highest"
    assert runtime["evaluation_seed"] == declaration["perturbation_seed"]
    assert report["evaluation_source_snapshot_unchanged_after"] is True
    evaluation_snapshot = report["evaluation_source_snapshot"]
    assert {"scripts/evaluate_ndr_transfer.py", "ndr_research/transfer_evaluation.py"} <= set(evaluation_snapshot["files"])
    assert json.loads((args.output / "evaluation_source_snapshot.json").read_text()) == evaluation_snapshot
    assert json.loads((args.output / "report.json").read_text())["status"] == "frozen_transfer_evaluation_complete"
    repeated_args = copy.copy(args)
    repeated_args.output = tmp_path / "independent_repeat_report"
    repeated = evaluator.run_evaluation(repeated_args, checkpoint)
    assert repeated["test_access_attempt_number"] == 2
    assert repeated["runtime"] == report["runtime"]
    assert repeated["model_state_sha256"] == report["model_state_sha256"]
    assert repeated["evaluation_source_snapshot"] == report["evaluation_source_snapshot"]
    for kind in ("shared_graph_ndr", *CONTROL_KINDS):
        first_metrics = report["scenarios"] if kind == "shared_graph_ndr" else report["controls"][kind]["scenarios"]
        second_metrics = repeated["scenarios"] if kind == "shared_graph_ndr" else repeated["controls"][kind]["scenarios"]
        for scenario in SCENARIOS:
            assert first_metrics[scenario]["sums"] == second_metrics[scenario]["sums"]
            assert first_metrics[scenario]["day_blocks"] == second_metrics[scenario]["day_blocks"]
    with pytest.raises(FileExistsError):
        evaluator.run_evaluation(args, checkpoint)
    assert opened == ["Orange", "Orange"]


def test_cli_requires_explicit_gate_and_creates_no_evaluation_search():
    parser = evaluator.argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "best.pt"])
    args = parser.parse_args(["--checkpoint", "best.pt", "--gate-json", "freeze.json"])
    assert args.device == "cpu" and args.mode == "evaluate"
    assert not hasattr(args, "max_val_batches") and not hasattr(args, "learning_rate")
    assert not hasattr(args, "epochs")
    help_text = parser.format_help()
    assert "six total test attempts" in help_text and "--max-test-evaluations 6" in help_text
    assert "all checkpoint seeds and both transfer modes" in help_text
    assert "Failed test" in help_text and "Target-val attempts" in help_text
