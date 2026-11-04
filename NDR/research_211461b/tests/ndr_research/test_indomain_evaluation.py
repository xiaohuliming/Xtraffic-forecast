from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from ndr_research.data import canonical_digest
from ndr_research.indomain_evaluation import (
    PROTOCOL,
    RegionalForecastAdapter,
    admit_regional_gate,
    evaluate_regional_models,
    fit_regional_training_preprocessing,
    inspect_selected_model,
    read_native_clean_history,
    rebuild_regional_model,
    state_digest,
    RegionalEvaluationLedger,
)
from ndr_research.models import make_research_model, research_factory_kwargs
from ndr_research.shared_graph import SharedGraphNDR
from ndr_research.strong_references import make_strong_reference
from ndr_research.strong_references import reference_recipe
from ndr_research.transfer_data import TransferRegionReader, TestEvaluationGate as Grant
from ndr_research.transfer_evaluation import (
    SCENARIOS,
    canonical_sha256,
    file_sha256,
    perturb_history,
    predict_history_only,
)
from scripts import evaluate_ndr_indomain as cli
from scripts import train_ndr_strong_reference as strong_trainer
from scripts import train_ndr_transfer as transfer_trainer
from ndr_research.transfer_data import load_transfer_sources
from tests.ndr_research.test_transfer_data import make_fixture, freeze_fixture


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


class RecordingNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, flow_hist, time_feat):
        self.received.append(flow_hist.detach().clone())
        return flow_hist.mean(-1, keepdim=True).expand(-1, -1, 12).clone()


def batch_for_boundary():
    flow = np.array(
        [[[5, 0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]]], dtype=np.float32
    )
    mask = np.ones_like(flow, dtype=bool)
    mask[..., 1] = False
    return {
        "flow_hist": flow,
        "history_mask": mask,
        "x_baseline": np.full_like(flow, 3),
        "y_baseline": np.full_like(flow, 3),
        "time_feat": np.zeros((1, 12, 2), dtype=np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "node_scale": np.ones(1, dtype=np.float32),
        "target": np.ones_like(flow),
        "target_mask": np.ones_like(mask),
        "sample_start": np.array([80], dtype=np.int64),
    }


def test_clean_native_input_is_preserved_and_corruption_cannot_reuse_masked_values():
    native = RecordingNative()
    adapter = RegionalForecastAdapter(native, kind="strong_reference").eval()
    batch = batch_for_boundary()
    clean = dict(batch)
    clean["flow_hist"] = batch["flow_hist"].copy()
    clean["flow_hist"][..., 1] = (
        777  # Original causal cache value, not a raw observation.
    )
    adapter.scenario = "clean"
    predict_history_only(adapter, clean, "cpu")
    assert native.received[-1][0, 0, 1] == 777
    for scenario in SCENARIOS[1:]:
        adapter.scenario = scenario
        first = perturb_history(batch, scenario, seed=42)
        poison = copy.deepcopy(batch)
        poison["flow_hist"][~poison["history_mask"]] = np.inf
        second = perturb_history(poison, scenario, seed=42)
        first_prediction = predict_history_only(adapter, first, "cpu")
        first_input = native.received[-1].clone()
        second_prediction = predict_history_only(adapter, second, "cpu")
        np.testing.assert_array_equal(first_prediction, second_prediction)
        torch.testing.assert_close(first_input, native.received[-1], rtol=0, atol=0)
        assert torch.isfinite(native.received[-1]).all()
    # An outage starts with the latest retained observation, not zero or a
    # previously computed cache value from a removed observation.
    adapter.scenario = "last_missing_3steps"
    outage = perturb_history(batch, adapter.scenario, seed=42)
    predict_history_only(adapter, outage, "cpu")
    assert native.received[-1][0, 0, -3:].tolist() == [75, 75, 75]
    first_missing = copy.deepcopy(batch)
    first_missing["history_mask"][..., :3] = False
    first_missing["flow_hist"][..., :3] = 0
    predict_history_only(adapter, first_missing, "cpu")
    assert native.received[-1][0, 0, :3].tolist() == [3, 3, 3]


def test_future_numeric_targets_do_not_change_real_ndr_or_staeformer_forecast():
    batch = batch_for_boundary()
    models = [
        RegionalForecastAdapter(make_research_model(1), kind="ndr").eval(),
        RegionalForecastAdapter(
            make_strong_reference(
                method="staeformer", node_count=1, flow_mean=10, flow_std=3
            ),
            kind="strong_reference",
        ).eval(),
        RegionalForecastAdapter(
            SharedGraphNDR(hidden_dim=8, num_heads=2), kind="shared_graph"
        ).eval(),
    ]
    changed = dict(batch)
    changed["target"] = np.full_like(batch["target"], np.nan)
    changed["target_mask"] = np.zeros_like(batch["target_mask"])
    for model in models:
        np.testing.assert_array_equal(
            predict_history_only(model, batch, "cpu"),
            predict_history_only(model, changed, "cpu"),
        )


def fixture_reader(tmp_path, *, native_missing_value=111):
    data, graph, contract = make_fixture(tmp_path)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as handle:
        handle["flow_mask"][70, 0, 0] = False
        handle["flow_series_imputed"][70, 0, 0] = native_missing_value
    contract = freeze_fixture(data, graph)
    reader = TransferRegionReader(
        "Alameda", data, graph, role="source", contract=contract
    )
    preprocessing = fit_regional_training_preprocessing(reader)
    return reader, preprocessing


def test_native_cache_reader_is_strictly_historical_and_never_interprets_analysis_arrays(
    tmp_path, monkeypatch
):
    reader, preprocessing = fixture_reader(tmp_path)
    accessed = []
    original = h5py.Dataset.__getitem__

    def track(dataset, selected):
        accessed.append((dataset.name, selected))
        return original(dataset, selected)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    indices = reader.indices("test")
    native = read_native_clean_history(reader, indices)
    assert native.shape == (2, reader.N, 12) and native[0, 0, 1] == 111
    assert {name for name, _ in accessed} == {"/flow_series_imputed"}
    for (_, selected), origin in zip(accessed, reader.sample_start[indices]):
        assert selected[0].start == origin - 11 and selected[0].stop == origin + 1
        assert selected[2] == 0
    assert preprocessing.identity["fit_range"] == [0, reader.train_end]


def test_all_real_baselines_and_candidate_share_complete_targets_masks_and_day_counts(
    tmp_path,
):
    reader, preprocessor = fixture_reader(tmp_path)
    checkout = Path(
        "/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf"
    )
    if not (checkout / "src/models/bist.py").exists():
        pytest.skip("locked public BiST checkout unavailable")
    stats = preprocessor.statistics
    ndr = make_research_model(
        reader.N,
        residual_std=stats.residual_std,
        residual_std_by_node=np.array(stats.residual_std_by_node, copy=True),
        flow_mean=stats.flow_mean,
        flow_std=stats.flow_std,
    )
    models = {
        "original_ndr": RegionalForecastAdapter(ndr, kind="ndr").eval(),
        "raw_staeformer": RegionalForecastAdapter(
            make_strong_reference(
                method="staeformer",
                node_count=reader.N,
                flow_mean=stats.flow_mean,
                flow_std=stats.flow_std,
            ),
            kind="strong_reference",
        ).eval(),
        "real_bist": RegionalForecastAdapter(
            make_strong_reference(
                method="bist",
                node_count=reader.N,
                flow_mean=stats.flow_mean,
                flow_std=stats.flow_std,
                bist_checkout=checkout,
            ),
            kind="strong_reference",
        ).eval(),
        "candidate": RegionalForecastAdapter(
            SharedGraphNDR(hidden_dim=8, num_heads=2), kind="shared_graph"
        ).eval(),
    }
    before = {name: state_digest(model) for name, model in models.items()}
    result = evaluate_regional_models(
        models,
        reader=reader,
        preprocessing=preprocessor,
        model_preprocessing={name: preprocessor for name in models},
        evaluation_grant=Grant("synthetic-frozen-cohort", "a" * 64),
        batch_size=1,
    )
    assert before == {name: state_digest(model) for name, model in models.items()}
    assert result["collection"]["samples"] == len(reader.indices("test")) == 2
    assert result["target_windows_read_once_for_all_models"] is True
    for name, report in result["models"].items():
        assert (
            report["collection"] == result["collection"]
            and report["target_used_for_forward"] is False
        )
        for scenario in SCENARIOS:
            current = report["scenarios"][scenario]
            assert current["valid_count"] == 2 * reader.N * 12
            assert current["collection"] == result["collection"]
            assert len(current["horizon_mae"]) == 12
            assert (
                sum(
                    sum(block["valid_count"])
                    for block in current["day_blocks"].values()
                )
                == current["valid_count"]
            )
            assert np.isfinite(current["global_masked_mae"])
        assert report["input_policy"]["clean"] == (
            "frozen_candidate_own_observation_sanitization"
            if name == "candidate"
            else "original_v3_causal_cache"
        )


def create_selected_ndr(tmp_path, reader, preprocessing, source_root):
    directory = tmp_path / "selected"
    directory.mkdir()
    source = source_root / "fixture_source.py"
    source_root.mkdir(exist_ok=True)
    source.write_text("FROZEN = True\n")
    files = {
        "fixture_source.py": {
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
        }
    }
    snapshot = {"files": files, "files_digest": canonical_sha256(files)}
    identity = dict(preprocessing.identity) | {
        "training_analysis_labels": False,
        "train_end": reader.train_end,
    }
    config = {
        "region": "Alameda",
        "variant": "baseline",
        "epochs": 30,
        "seed": 42,
        "max_train_batches": None,
        "max_val_batches": None,
        "claim_scope": "validation_development_only",
        "source_snapshot": snapshot,
        "data_identity": identity,
        "runtime": {"device_type": "cpu"},
    }
    model = make_research_model(
        reader.N,
        residual_std=preprocessing.statistics.residual_std,
        residual_std_by_node=np.array(
            preprocessing.statistics.residual_std_by_node, copy=True
        ),
        flow_mean=preprocessing.statistics.flow_mean,
        flow_std=preprocessing.statistics.flow_std,
    )
    selected = {
        "global_masked_mae": 1.0,
        "mae": 1.0,
        "valid_count": 72,
        "selection": {"all_validation_samples": True, "samples": 2},
        "observed_targets_sha256": "b" * 64,
        "observation_masks_sha256": "c" * 64,
    }
    payload = {
        "schema_version": 1,
        "config": config,
        "config_digest": canonical_digest(config),
        "data_identity": identity,
        "model": model.state_dict(),
        "model_factory": research_factory_kwargs(model),
        "selection_metric": selected,
    }
    cp = directory / "best.pt"
    torch.save(payload, cp)
    cp.with_suffix(".pt.sha256.json").write_text(
        json.dumps(
            {
                "sha256": file_sha256(cp),
                "size_bytes": cp.stat().st_size,
                "config_digest": payload["config_digest"],
            }
        )
    )
    verification = {
        "status": "verified_checkpoint_complete_validation",
        "checkpoint_sha256": file_sha256(cp),
        "config_digest": payload["config_digest"],
        "source_files_digest": snapshot["files_digest"],
        "test_evaluated": False,
        "validation": selected,
    }
    verify = directory / "verification.json"
    verify.write_text(json.dumps(verification))
    summary = directory / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "completed_epochs": 30,
                "test_evaluated": False,
                "config_digest": payload["config_digest"],
            }
        )
    )
    return cp, verify, summary, payload


def frozen_fixture_gate(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    reader, preprocessing = fixture_reader(cache)
    source = tmp_path / "source"
    cp, verification, summary, payload = create_selected_ndr(
        tmp_path, reader, preprocessing, source
    )
    admitted = inspect_selected_model(
        cp, verification, summary, kind="ndr", source_root=source
    )
    model_row = {
        "model_id": "baseline",
        "checkpoint_kind": "ndr",
        "checkpoint": str(cp),
        "verification": str(verification),
        "summary": str(summary),
        "checkpoint_sha256": file_sha256(cp),
        "verification_sha256": file_sha256(verification),
        "summary_sha256": file_sha256(summary),
        **{
            key: admitted[key]
            for key in (
                "validation_selection_sha256",
                "data_identity_sha256",
                "config_digest",
                "family_identity",
                "training_seed",
            )
        },
    }
    gate = {
        "schema_version": 1,
        "region": "Alameda",
        "split": "test",
        "sampling_protocol": "event",
        "protocol_version": PROTOCOL,
        "structure_development_closed": True,
        "selection_frozen": True,
        "test_used_for_structure_selection": False,
        "authorization_id": "synthetic-fixed-cohort",
        "cache_bundle_digest": reader.contract.bundle_digest,
        "scenarios": list(SCENARIOS),
        "input_policy": "native_clean_and_retained_observation_causal_corruption",
        "perturbation_seed": 42,
        "comparison_scope": "matched_indomain_training",
        "evaluation_ledger": str(tmp_path / "shared_ledger.jsonl"),
        "max_test_evaluations": 6,
        "models": [model_row, model_row | {"model_id": "second"}],
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    return reader, preprocessing, source, cp, verification, summary, payload, path, gate


def test_gate_binds_all_checkpoints_complete_validation_and_closed_structure_before_cache_read(
    tmp_path, monkeypatch
):
    reader, preprocessing, source, cp, verification, summary, payload, path, gate = (
        frozen_fixture_gate(tmp_path)
    )
    admitted = inspect_selected_model(
        cp, verification, summary, kind="ndr", source_root=source
    )
    restored = rebuild_regional_model(admitted, "cpu")
    assert restored.kind == "ndr"
    frozen, models = admit_regional_gate(path, source_root=source)
    assert set(models) == {"baseline", "second"}
    assert frozen["gate_sha256"] == file_sha256(path)
    gate["structure_development_closed"] = False
    path.write_text(json.dumps(gate))
    monkeypatch.setattr(
        cli,
        "TransferRegionReader",
        lambda *_args, **_kw: pytest.fail("test reader opened before gate admission"),
    )
    monkeypatch.setattr(cli, "ROOT", source)
    args = cli.argument_parser().parse_args(
        ["--gate-json", str(path), "--output", str(tmp_path / "output")]
    )
    with pytest.raises(PermissionError, match="closed-development"):
        cli.run_evaluation(args)
    assert not args.output.exists()
    gate["structure_development_closed"] = True
    gate["cache_bundle_digest"] = "d" * 64
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="mixes cache bundles"):
        admit_regional_gate(path, source_root=source)
    gate["cache_bundle_digest"] = reader.contract.bundle_digest
    payload["config"]["max_train_batches"] = 2
    payload["config_digest"] = canonical_digest(payload["config"])
    torch.save(payload, cp)
    cp.with_suffix(".pt.sha256.json").write_text(
        json.dumps(
            {
                "sha256": file_sha256(cp),
                "size_bytes": cp.stat().st_size,
                "config_digest": payload["config_digest"],
            }
        )
    )
    with pytest.raises(PermissionError, match="limited-batch smoke"):
        inspect_selected_model(
            cp, verification, summary, kind="ndr", source_root=source
        )


def test_cli_synthetic_full_cohort_outputs_existing_analysis_schema_without_real_test_reads(
    tmp_path, monkeypatch
):
    reader, _, source, _, _, _, _, gate_path, gate = frozen_fixture_gate(tmp_path)
    actual_snapshot = cli._evaluation_snapshot()
    monkeypatch.setattr(cli, "ROOT", source)
    monkeypatch.setattr(cli, "_evaluation_snapshot", lambda: actual_snapshot)
    monkeypatch.setattr(cli, "TransferRegionReader", lambda *_args, **_kwargs: reader)
    args = cli.argument_parser().parse_args(
        [
            "--gate-json",
            str(gate_path),
            "--output",
            str(tmp_path / "reports"),
            "--batch-size",
            "1",
            "--device",
            "cpu",
        ]
    )
    result = cli.run_evaluation(args)
    assert result["status"] == "frozen_indomain_paired_evaluation_complete"
    assert (
        result["evaluation_region_train_statistics_used_for_regional_controls"] is True
    )
    assert result["consumed_test_inference_slots"] == 1
    assert len(result["models"]) == 2
    for name in ("baseline", "second"):
        report = json.loads((args.output / (name + ".json")).read_text())
        assert report["region"] == "Alameda" and report["split"] == "test"
        assert report["training_seed"] == 42 and report["evaluation_mode"] == "indomain"
        assert (
            report["training_scope"] == "regional_supervised"
            and report["trained_on_evaluated_region"] is True
        )
        assert report["provenance"]["training_regions"] == ["Alameda"]
        assert (
            "training_recipe_identity" in report["provenance"]
            and "training_runtime" in report["provenance"]
        )
        assert (
            report["evaluation_source_snapshot"]["files_digest"]
            == actual_snapshot["files_digest"]
        )
        for scenario in SCENARIOS:
            assert report["scenarios"][scenario]["collection"] == result["collection"]
            assert report["scenarios"][scenario]["valid_count"] == 2 * reader.N * 12
    assert json.loads((args.output / "paired_matrix_report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        cli.run_evaluation(args)


def test_actual_complete_baseline_checkpoint_structure_requires_no_target_data():
    directory = Path(
        "/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/partial_verified/alameda_baseline_seed42"
    )
    if not (directory / "best.pt").exists():
        pytest.skip("retained complete baseline checkpoint unavailable")
    inspected = inspect_selected_model(
        directory / "best.pt",
        directory / "verification.json",
        directory / "summary.json",
        kind="ndr",
    )
    model = rebuild_regional_model(inspected, "cpu")
    assert (
        inspected["training_seed"] == 42 and inspected["model_family"] == "ndr_baseline"
    )
    assert sum(value.numel() for value in model.parameters()) == 1582308
    assert inspected["checkpoint"]["model_factory"]["node_count"] == 521


@pytest.mark.parametrize("method", ("staeformer", "bist"))
def test_real_strong_checkpoint_writer_schema_is_admitted_without_accepting_other_versions(
    tmp_path, method
):
    cache = tmp_path / "cache"
    cache.mkdir()
    reader, preprocessing = fixture_reader(cache)
    checkout = Path(
        "/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf"
    )
    if method == "bist" and not (checkout / "src/models/bist.py").exists():
        pytest.skip("locked public BiST checkout unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "fixture_source.py"
    source.write_text("UNCHANGED = True\n")
    files = {
        "fixture_source.py": {
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
        }
    }
    identity = dict(preprocessing.identity) | {
        "training_analysis_labels": False,
        "train_end": reader.train_end,
    }
    recipe = reference_recipe(
        method,
        "matched-ndr",
        region="Alameda",
        epochs=30,
        batch_size=2,
        microbatch_size=2,
    )
    config = {
        "region": "Alameda",
        "epochs": 30,
        "seed": 42,
        "max_train_batches": None,
        "max_val_batches": None,
        "training_recipe": recipe,
        "claim_scope": "validation_development_only",
        "source_snapshot": {"files": files, "files_digest": canonical_sha256(files)},
        "data_identity": identity,
        "runtime": {"device_type": "cpu"},
        "training_process_id": 123,
    }
    model = make_strong_reference(
        method=method,
        node_count=reader.N,
        flow_mean=preprocessing.statistics.flow_mean,
        flow_std=preprocessing.statistics.flow_std,
        bist_checkout=checkout if method == "bist" else None,
    )
    optimizer, scheduler = strong_trainer.create_reference_optimizer(model, config, 3)
    loader = SimpleNamespace(state_dict=lambda: {"order": np.arange(3, dtype=np.int64)})
    data = SimpleNamespace(identity=identity, train=loader, val=loader)
    selected = {
        "mae": 1.0,
        "valid_count": 72,
        "selection": {"all_validation_samples": True, "samples": 2},
        "observed_targets_sha256": "b" * 64,
        "observation_masks_sha256": "c" * 64,
    }
    checkpoint = tmp_path / "best.pt"
    strong_trainer.save_reference_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        config=config,
        progress={"epoch": 30},
        selection_metric=selected,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == "strong_reference_checkpoint_v1"
    verified = {
        "status": "verified_complete_source_validation_development",
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_digest": payload["config_digest"],
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "test_evaluated": False,
        "validation": selected,
    }
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(verified))
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "completed_epochs": 30,
                "config_digest": payload["config_digest"],
                "test_evaluated": False,
            }
        )
    )
    accepted = inspect_selected_model(
        checkpoint,
        verification,
        summary,
        kind="strong_reference",
        source_root=source_root,
    )
    restored = rebuild_regional_model(
        accepted, "cpu", bist_checkout=checkout if method == "bist" else None
    )
    assert state_digest(restored.model) == state_digest(model)
    payload["schema_version"] = (
        1  # The NDR integer schema must not pass for a strong reference.
    )
    torch.save(payload, checkpoint)
    checkpoint.with_suffix(".pt.sha256.json").write_text(
        json.dumps(
            {
                "sha256": file_sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "config_digest": payload["config_digest"],
            }
        )
    )
    with pytest.raises(ValueError, match="unsupported selected checkpoint schema"):
        inspect_selected_model(
            checkpoint,
            verification,
            summary,
            kind="strong_reference",
            source_root=source_root,
        )


def test_shared_regional_budget_ignores_gate_path_model_id_checkpoint_seed_and_auth(
    tmp_path,
):
    ledger_path = tmp_path / "study_ledger.jsonl"
    gate = {
        "region": "Alameda",
        "max_test_evaluations": 6,
        "evaluation_ledger": str(ledger_path),
        "gate_sha256": "a" * 64,
        "authorization_id": "first",
    }
    family_a = {
        "model_family": "original_ndr",
        "source_files_digest": "b" * 64,
        "architecture": {"variant": "baseline"},
    }
    family_b = {
        "model_family": "raw_staeformer",
        "source_files_digest": "c" * 64,
        "architecture": {"method": "staeformer"},
    }

    def specs(seed, suffix):
        return {
            f"baseline_{suffix}": {
                "family_identity": family_a,
                "training_seed": seed,
                "checkpoint_sha256": str(seed % 10) * 64,
            },
            f"reference_{suffix}": {
                "family_identity": family_b,
                "training_seed": seed,
                "checkpoint_sha256": "f" * 64,
            },
        }

    original = RegionalEvaluationLedger(
        gate,
        specs(42, "first"),
        tmp_path / "first_gate.json",
        tmp_path / "first_output",
    )
    assert original.reserve()["consumed_slots_after"] == 1
    original.finish("failed", error_type="TimeoutError")
    moved_gate = gate | {"authorization_id": "different", "gate_sha256": "d" * 64}
    moved = RegionalEvaluationLedger(
        moved_gate,
        specs(1, "renamed"),
        tmp_path / "subfolder/new_gate.json",
        tmp_path / "new_output",
    )
    assert moved.family_sha == original.family_sha
    assert moved.reserve()["consumed_slots_after"] == 2
    moved.finish("completed")
    for changed_limit in (3, 7):
        changed = RegionalEvaluationLedger(
            moved_gate | {"max_test_evaluations": changed_limit},
            specs(2, "third"),
            tmp_path / "another_gate.json",
            tmp_path / "third",
        )
        with pytest.raises(PermissionError, match="budget cannot change"):
            changed.reserve()
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [
        row["training_seed_inference_slots"]
        for row in rows
        if row["status"] == "reserved"
    ] == [1, 1]


def test_real_shared_graph_checkpoint_writer_and_source_verification_binding(tmp_path):
    data_dir, graph_dir, contract = make_fixture(tmp_path)
    data = load_transfer_sources(
        ["Alameda", "ContraCosta"],
        "Orange",
        data_dir,
        graph_dir,
        2,
        42,
        contract=contract,
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "fixture_source.py"
    source.write_text("FIXED = True\n")
    files = {
        "fixture_source.py": {
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
        }
    }
    model = SharedGraphNDR(hidden_dim=8, num_heads=2)
    config = {
        "model_class": "SharedGraphNDR",
        "model_config": model.get_config(),
        "epochs": 30,
        "seed": 42,
        "source_regions": ["Alameda", "ContraCosta"],
        "heldout_region": "Orange",
        "max_train_batches": None,
        "max_val_batches": None,
        "claim_scope": "source_validation_development_only",
        "source_snapshot": {"files": files, "files_digest": canonical_sha256(files)},
        "data_identity": data.identity,
        "runtime": {"device_type": "cpu"},
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "eta_min": 0.00001,
    }
    rows = {
        region: {
            "global_masked_mae": 1.0,
            "valid_count": reader.N * 24,
            "selection": {"all_validation_samples": True, "samples": 2},
            "observed_targets_sha256": "b" * 64,
            "observation_masks_sha256": "c" * 64,
        }
        for region, reader in data.readers.items()
    }
    selection = {
        "selection_split": "source_val",
        "all_source_validation_samples": True,
        "target_region_used": False,
        "regions": rows,
        "macro_global_mae": 1.0,
    }
    optimizer, scheduler = transfer_trainer.create_transfer_optimizer(model, config, 6)
    checkpoint = tmp_path / "best.pt"
    transfer_trainer.save_transfer_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        config=config,
        progress={"epoch": 30},
        selection_metric=selection,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    verify = {
        "status": "verified_transfer_source_checkpoint_complete_validation",
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_digest": payload["config_digest"],
        "source_files_digest": config["source_snapshot"]["files_digest"],
        "test_evaluated": False,
        "source_validation": selection,
        "source_statistics_identity": data.preprocessing.identity,
        "model_config_sha256": canonical_sha256(model.get_config()),
        "selection_metric_digest": canonical_sha256(selection),
    }
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(verify))
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "completed_epochs": 30,
                "config_digest": payload["config_digest"],
                "test_evaluated": False,
            }
        )
    )
    accepted = inspect_selected_model(
        checkpoint, verification, summary, kind="shared_graph", source_root=source_root
    )
    assert accepted["training_regions"] == ("Alameda", "ContraCosta")
    restored = rebuild_regional_model(accepted, "cpu")
    assert state_digest(restored.model) == state_digest(model)
    verify["source_statistics_identity"] = dict(
        verify["source_statistics_identity"]
    ) | {"target_statistics_used": True}
    verification.write_text(json.dumps(verify))
    with pytest.raises(PermissionError, match="does not bind source-only statistics"):
        inspect_selected_model(
            checkpoint,
            verification,
            summary,
            kind="shared_graph",
            source_root=source_root,
        )


def test_three_seed_cohort_and_independent_repeat_require_six_predeclared_slots(
    tmp_path,
):
    gate = {
        "region": "Orange",
        "max_test_evaluations": 6,
        "evaluation_ledger": str(tmp_path / "cohort_ledger.jsonl"),
        "gate_sha256": "a" * 64,
        "authorization_id": "all-three-seeds",
    }
    family = {
        "model_family": "ndr",
        "architecture": {"variant": "baseline"},
        "source_files_digest": "b" * 64,
    }
    cohort = {
        f"ndr_{seed}": {
            "family_identity": family,
            "training_seed": seed,
            "checkpoint_sha256": str(seed % 10) * 64,
        }
        for seed in (42, 1, 2)
    }
    first = RegionalEvaluationLedger(
        gate, cohort, tmp_path / "gate.json", tmp_path / "first"
    )
    assert first.reserve()["consumed_slots_after"] == 3
    first.finish("completed")
    second = RegionalEvaluationLedger(
        gate, cohort, tmp_path / "copied_gate.json", tmp_path / "repeat"
    )
    assert second.reserve()["consumed_slots_after"] == 6
    second.finish("failed", error_type="Interrupted")
    third = RegionalEvaluationLedger(
        gate, cohort, tmp_path / "different_gate.json", tmp_path / "extra"
    )
    with pytest.raises(PermissionError, match="budget is exhausted"):
        third.reserve()


def test_real_single_region_graph_writer_and_scope_enter_matched_regional_evaluation(
    tmp_path,
):
    from scripts import train_ndr_graph_indomain as graph_trainer
    from tests.ndr_research.test_graph_indomain_training import (
        fixture_data,
        fixture_config,
        new_model,
    )

    data = fixture_data()
    data.identity["train_end"] = data.region.train_end
    config = fixture_config(data, epochs=30)
    config["claim_scope"] = "validation_development_only"
    model = new_model()
    optimizer, scheduler = graph_trainer.create_optimizer(
        model, config, data.train.num_batch
    )
    selected = {
        "mae": 1.0,
        "global_masked_mae": 1.0,
        "valid_count": 12,
        "selection": {"all_validation_samples": True, "samples": data.val.size},
        "observed_targets_sha256": "b" * 64,
        "observation_masks_sha256": "c" * 64,
    }
    checkpoint = tmp_path / "best.pt"
    graph_trainer.save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data=data,
        config=config,
        progress={"epoch": 10},
        selection_metric=selected,
    )
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "status": "verified_graph_indomain_checkpoint_complete_validation",
                "config_digest": canonical_digest(config),
                "checkpoint_sha256": file_sha256(checkpoint),
                "source_files_digest": config["source_snapshot"]["files_digest"],
                "test_evaluated": False,
                "normalizer_identity": config["normalizer_identity"],
                "graph_identity": config["graph_identity"],
                "validation": selected,
            }
        )
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "completed_epochs": 30,
                "test_evaluated": False,
                "config_digest": canonical_digest(config),
            }
        )
    )
    admitted = inspect_selected_model(
        checkpoint, verification, summary, kind="graph_indomain"
    )
    assert admitted["training_regions"] == ("Alameda",)
    assert admitted["model_family"] == "graph_indomain_h64"
    rebuilt = rebuild_regional_model(admitted, "cpu")
    assert rebuilt.kind == "graph_indomain"
    assert rebuilt.graph_identity == config["graph_identity"]
    assert state_digest(rebuilt.model) == state_digest(model)
    assert not any(name == "sd_node" for name, _ in rebuilt.model.named_buffers())
    with pytest.raises(ValueError, match="schema"):
        inspect_selected_model(checkpoint, verification, summary, kind="shared_graph")
    current = json.loads(verification.read_text())
    current["normalizer_identity"]["fit_range"] = [0, data.region.train_end + 1]
    verification.write_text(json.dumps(current))
    with pytest.raises(ValueError, match="bind its train statistics"):
        inspect_selected_model(checkpoint, verification, summary, kind="graph_indomain")
