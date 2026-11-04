"""Synthetic evaluator-shaped cohorts; no real traffic labels, Orange or GPU."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from ndr_research.fullwindow_evaluation import (
    GATE_SCHEMA,
    REPORT_STATUS,
    FullWindowForecastAdapter,
    evaluate_fullwindow_models,
)
from ndr_research.fullwindow_training import make_training_data
from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.strong_references import make_strong_reference
from ndr_research.transfer_evaluation import SCENARIOS, canonical_sha256, file_sha256
from scripts.analyze_ndr_cohort import analyze_manifest, load_member
from scripts.train_ndr_transfer import configure_transfer_runtime, transfer_runtime_identity
from tests.ndr_research.test_fullwindow_evaluation import frozen_reader, synthetic_material

SEEDS = (42, 1, 2)


@pytest.fixture(scope="module")
def evaluator_output(tmp_path_factory):
    """Run the real evaluator on synthetic data before adding public metadata."""
    configure_transfer_runtime("cpu", 42, 1)
    directory = tmp_path_factory.mktemp("synthetic_fullwindow_cohort")
    data, roads, contract, source, statistics = synthetic_material(directory, steps=2000)
    reader, grant = frozen_reader(data, roads, contract, source, statistics)
    graph = SharedGraphNDR(hidden_dim=8, num_heads=1).eval()
    native = make_strong_reference(method="staeformer", node_count=source.N,
                                   flow_mean=statistics.flow_mean, flow_std=statistics.flow_std).eval()
    adapters = {
        "graph": FullWindowForecastAdapter(graph, kind="fullwindow_ndr", method="shared_graph").eval(),
        "stae": FullWindowForecastAdapter(native, kind="fullwindow_strong", method="staeformer").eval(),
    }
    output = evaluate_fullwindow_models(adapters, reader=reader, grant=grant, batch_size=64, device="cpu")
    assert len(output["models"]["graph"]["scenarios"]["clean"]["day_blocks"]) >= 2
    assert output["target_windows_read_once_for_all_models"]
    training = make_training_data(source, statistics, batch_size=16, seed=42, preload=False)
    return output, source, statistics, training.identity, graph.get_config()


def synthetic_report(evaluator_output, seed):
    """Match CLI report and inspect_fullwindow_model public_spec field layouts."""
    output, source, statistics, data_identity, graph_config = evaluator_output
    report = copy.deepcopy(output)
    report.update(schema_version=GATE_SCHEMA, status=REPORT_STATUS, region="Alameda",
                  sampling_protocol="full_window", split="test", mode="same_region_temporal_heldout",
                  evaluation_role="same_region_temporal_heldout",
                  comparison_scope="same_region_full_window_temporal_holdout", training_seeds=list(SEEDS),
                  reader_identity=source.identity, sampling_identity=source.sampling_identity,
                  statistics_identity=dict(statistics.identity),
                  evaluation_source_snapshot={"files_digest": "f" * 64},
                  scalar_fit_precision="float64", legacy27_scalar_fit_precision="float32",
                  legacy27_scores_directly_comparable=False, test_driven_tuning=False)
    names = {"graph": f"graph_seed{seed}", "stae": f"stae_seed{seed}"}
    report["models"] = {names[name]: value for name, value in report["models"].items()}
    runtime = transfer_runtime_identity(torch.device("cpu"))
    for name, method, kind, real_class, native_class in (
        (names["graph"], "shared_graph", "fullwindow_ndr", "SharedGraphNDR", None),
        (names["stae"], "staeformer", "fullwindow_strong", "StrongReference", "STAEformer"),
    ):
        recipe = {"method": method, "epochs": 100, "batch_size": 16, "microbatch_size": 8,
                  "learning_rate": 0.001, "loss": "synthetic_only_main_objective"}
        snapshot = {"files_digest": "a" * 64 if kind == "fullwindow_ndr" else "b" * 64}
        family = {"checkpoint_kind": kind, "model_class": real_class, "method": method,
                  "sampling_protocol": "full_window",
                  "architecture": copy.deepcopy(graph_config) if method == "shared_graph" else {"method": method, "node_count": source.N},
                  "training_recipe_identity": recipe, "source_files_digest": snapshot["files_digest"],
                  "training_runtime": runtime, "sampling_digest": source.sampling_identity["sampling_digest"],
                  "statistics_identity_digest": statistics.identity["identity_digest"]}
        report["models"][name].update(
            kind=kind, checkpoint_kind=kind, model_class=real_class, native_model_class=native_class,
            model_id=name, method=method, model_family="shared_graph_h8" if method == "shared_graph" else method,
            training_seed=seed, training_regions=["Alameda"],
            training_scope="same_region_full_window_supervised_temporal_split", sampling_protocol="full_window",
            family_identity=copy.deepcopy(family), training_recipe=copy.deepcopy(recipe), training_runtime=copy.deepcopy(runtime),
            source_snapshot=copy.deepcopy(snapshot), reader_identity=source.identity,
            sampling_identity=source.sampling_identity, statistics_identity=dict(statistics.identity),
            data_identity=copy.deepcopy(data_identity),
            checkpoint_sha256=canonical_sha256({"fixture": "untrained_synthetic_only", "seed": seed, "method": method}),
            normalizer_scalar_dtype="float64", legacy27_scalar_dtype="float32", native_convergence_reproduced=False,
        )
    return report


def write_report(tmp_path: Path, report: dict, name="report.json") -> dict:
    path = tmp_path / name
    path.write_text(json.dumps(report, sort_keys=True, allow_nan=False))
    return {"report": name, "report_sha256": file_sha256(path)}


def selected(tmp_path, report, seed=42, side="graph"):
    declaration = write_report(tmp_path, report)
    declaration["model"] = f"{side}_seed{seed}"
    return load_member(declaration, training_seed=seed, base=tmp_path)


def manifest_files(tmp_path, reports):
    manifest = {"schema_version": 1, "selection_frozen": True, "scenarios": list(SCENARIOS),
                "bootstrap_samples": 100, "bootstrap_seed": 42, "evidence_role": "indomain_performance", "pairs": []}
    for seed in SEEDS:
        declaration = write_report(tmp_path, reports[seed], f"seed{seed}.json")
        manifest["pairs"].append({"training_seed": seed,
                                  "left": dict(declaration, model=f"graph_seed{seed}"),
                                  "right": dict(declaration, model=f"stae_seed{seed}")})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_actual_evaluator_shape_selects_declared_model_and_preserves_identity(tmp_path, evaluator_output):
    report = synthetic_report(evaluator_output, 42)
    report["models"]["irrelevant_unselected"] = {"training_seed": 999, "model_class": "wrong"}
    normalized, identity = selected(tmp_path, report, side="stae")
    current = report["models"]["stae_seed42"]
    assert normalized["training_seed"] == 42 and normalized["region"] == "Alameda"
    assert normalized["evaluation_mode"] == "indomain"
    assert normalized["comparison_scope"] == "matched_fullwindow_regional_supervision"
    assert normalized["sampling_protocol"] == "full_window"
    assert normalized["trained_on_evaluated_region"] and normalized["training_regions"] == ["Alameda"]
    family = normalized["family_identity"]
    assert family["model_class"] == "StrongReference" and family["native_model_class"] == "STAEformer"
    assert family["model_family"] == "staeformer"
    assert family["fullwindow_model_family"]["checkpoint_kind"] == "fullwindow_strong"
    assert family["fullwindow_model_family"]["source_files_digest"] == current["source_snapshot"]["files_digest"]
    for key in ("reader_identity", "sampling_identity", "statistics_identity", "data_identity"):
        assert family[key] == current[key]
    assert family["scalar_fit_precision"] == "float64" and family["legacy27_scalar_fit_precision"] == "float32"
    assert family["legacy27_scores_directly_comparable"] is False
    assert normalized["scenarios"] == current["scenarios"]
    assert identity["model"] == "stae_seed42" and identity["checkpoint_sha256"] == current["checkpoint_sha256"]


def test_complete_three_seed_fullwindow_manifest_keeps_truthful_scope(tmp_path, evaluator_output):
    reports = {seed: synthetic_report(evaluator_output, seed) for seed in SEEDS}
    result = analyze_manifest(manifest_files(tmp_path, reports))
    assert result["sampling_protocol"] == "full_window" and result["evaluation_mode"] == "indomain"
    assert result["comparison_scope"] == "matched_fullwindow_regional_supervision"
    assert result["traffic_labels_opened"] is False and result["sota_or_winner_claim"] is False
    assert set(result["scenarios"]) == set(SCENARIOS)
    assert result["model_family_identities"]["left"]["model_class"] == "SharedGraphNDR"
    assert result["model_family_identities"]["right"]["native_model_class"] == "STAEformer"
    assert result["model_training_scopes"]["left"]["training_regions"] == ["Alameda"]
    assert result["scenarios"]["clean"]["groups"]["confirmation_seeds_1_2"]["training_seeds"] == [1, 2]


@pytest.mark.parametrize("field,value", [
    ("training_seed", 1), ("training_seed", True), ("model_class", "StrongReference"),
    ("native_model_class", "BiST"), ("checkpoint_kind", "fullwindow_strong"), ("kind", "event_ndr"),
])
def test_wrong_seed_or_real_class_is_rejected(tmp_path, evaluator_output, field, value):
    report = synthetic_report(evaluator_output, 42)
    report["models"]["graph_seed42"][field] = value
    with pytest.raises(ValueError, match="seed or true model class"):
        selected(tmp_path, report)


@pytest.mark.parametrize("field", ["reader_identity", "statistics_identity", "sampling_identity"])
@pytest.mark.parametrize("where", ["model", "top", "data"])
def test_top_model_and_data_identity_mismatch_is_rejected(tmp_path, evaluator_output, field, where):
    report = synthetic_report(evaluator_output, 42)
    model = report["models"]["graph_seed42"]
    destination = report if where == "top" else model["data_identity"] if where == "data" else model
    destination[field] = copy.deepcopy(destination[field])
    destination[field]["tampered_fixture_identity"] = True
    with pytest.raises(ValueError, match="identit|identity"):
        selected(tmp_path, report)


@pytest.mark.parametrize("field,value", [
    ("scalar_fit_precision", "float32"), ("legacy27_scalar_fit_precision", "float64"),
    ("legacy27_scores_directly_comparable", True), ("sampling_protocol", "event"),
])
def test_event_and_legacy_float32_cannot_enter_new_fullwindow_comparison(tmp_path, evaluator_output, field, value):
    report = synthetic_report(evaluator_output, 42)
    report[field] = value
    with pytest.raises(ValueError, match="sampling and scalar precision"):
        selected(tmp_path, report)


@pytest.mark.parametrize("field,value", [
    ("model_class", "STAEformerDeseason"), ("checkpoint_kind", "fullwindow_strong"),
    ("method", "bist"), ("sampling_protocol", "event"),
    ("source_files_digest", "0" * 64), ("statistics_identity_digest", "0" * 64),
    ("sampling_digest", "0" * 64),
])
def test_family_cross_bindings_are_rejected(tmp_path, evaluator_output, field, value):
    report = synthetic_report(evaluator_output, 42)
    report["models"]["graph_seed42"]["family_identity"][field] = value
    with pytest.raises(ValueError, match="family|identit"):
        selected(tmp_path, report)


@pytest.mark.parametrize("model_family", ["shared_graph_h64", "bist", "fake_ndr"])
def test_true_family_name_cannot_be_relabelled(tmp_path, evaluator_output, model_family):
    report = synthetic_report(evaluator_output, 42)
    report["models"]["graph_seed42"]["model_family"] = model_family
    with pytest.raises(ValueError, match="family"):
        selected(tmp_path, report)


def test_consistent_single_report_recipe_drift_is_rejected_across_seeds(tmp_path, evaluator_output):
    reports = {seed: synthetic_report(evaluator_output, seed) for seed in SEEDS}
    changed = reports[2]["models"]["graph_seed2"]
    changed["training_recipe"]["learning_rate"] = 0.002
    changed["family_identity"]["training_recipe_identity"] = copy.deepcopy(changed["training_recipe"])
    with pytest.raises(ValueError, match="model recipes or source identities"):
        analyze_manifest(manifest_files(tmp_path, reports))


def test_consistent_single_report_statistics_drift_is_rejected_across_seeds(tmp_path, evaluator_output):
    reports = {seed: synthetic_report(evaluator_output, seed) for seed in SEEDS}
    report = reports[2]
    report["statistics_identity"]["identity_digest"] = "0" * 64
    for model in report["models"].values():
        model["statistics_identity"] = copy.deepcopy(report["statistics_identity"])
        model["data_identity"]["statistics_identity"] = copy.deepcopy(report["statistics_identity"])
        model["family_identity"]["statistics_identity_digest"] = "0" * 64
    with pytest.raises(ValueError, match="model recipes or source identities"):
        analyze_manifest(manifest_files(tmp_path, reports))


def test_actual_event_report_mixed_with_fullwindow_manifest_fails(tmp_path, evaluator_output):
    reports = {seed: synthetic_report(evaluator_output, seed) for seed in SEEDS}
    path = manifest_files(tmp_path, reports)
    manifest = json.loads(path.read_text())
    event = copy.deepcopy(reports[2]["models"]["stae_seed2"])
    event.update(status="frozen_indomain_model_evaluation_complete", training_seed=2, split="test", region="Alameda",
                 sampling_protocol="event", evaluation_mode="indomain", model_family="staeformer",
                 comparison_scope="matched_indomain_training", training_scope="regional_supervised", trained_on_evaluated_region=True,
                 evaluation_source_snapshot={"files_digest": "f" * 64}, collection=copy.deepcopy(reports[2]["collection"]),
                 provenance={"model_config": {"method": "staeformer"}, "training_recipe_identity": event["training_recipe"],
                             "training_runtime": event["training_runtime"], "source_files_digest": "b" * 64, "training_regions": ["Alameda"]})
    declaration = write_report(tmp_path, event, "event.json")
    manifest["pairs"][2]["right"] = dict(declaration, model="stae_seed2")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mixes regions, splits, protocols"):
        analyze_manifest(path)


def test_report_hash_and_comparison_scope_are_bound(tmp_path, evaluator_output):
    report = synthetic_report(evaluator_output, 42)
    declaration = write_report(tmp_path, report)
    declaration["model"] = "graph_seed42"
    (tmp_path / declaration["report"]).write_text("{}")
    with pytest.raises(ValueError, match="report hash"):
        load_member(declaration, training_seed=42, base=tmp_path)
    report["comparison_scope"] = "source_only_transfer"
    with pytest.raises(ValueError, match="scope|boundary|boundaries"):
        selected(tmp_path, report)
