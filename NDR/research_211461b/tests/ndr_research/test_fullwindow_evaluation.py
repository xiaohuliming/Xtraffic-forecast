"""Synthetic H5 and checkpoint tests, never real target/test/GPU access."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import h5py
import pytest
import torch

from ndr_research.fullwindow_data import (
    FullWindowRegionReader,
    fit_fullwindow_statistics,
)
from ndr_research.fullwindow_evaluation import (
    GATE_SCHEMA,
    SEEDS,
    FullWindowEvaluationLedger,
    FullWindowForecastAdapter,
    FullWindowFrozenGrant,
    FullWindowFrozenReader,
    admit_fullwindow_gate,
    evaluate_fullwindow_models,
    inspect_fullwindow_model,
)
from ndr_research.fullwindow_training import (
    factory_from_data,
    make_training_data,
    rebuild_model,
)
from ndr_research.strong_references import make_strong_reference, reference_recipe
from ndr_research.transfer_data import PROTOCOL
from ndr_research.transfer_evaluation import SCENARIOS, canonical_sha256, file_sha256
from scripts import evaluate_ndr_fullwindow as cli
from scripts.train_ndr_transfer import (
    checkpoint_safe,
    configure_transfer_runtime,
    transfer_runtime_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def synthetic_material(tmp_path, steps=1500):
    path = Path(__file__).with_name("test_fullwindow_data.py")
    specification = importlib.util.spec_from_file_location(
        "_fw_frozen_synthetic_material", path
    )
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    data, roads, contract = helper.make_material(tmp_path, steps=steps)
    # Preserve a real causal cache fill at an originally missing history point.
    with h5py.File(data / "Alameda_traffic.h5", "r+") as handle:
        index = int(steps * 0.85) - 1
        handle["flow_mask"][index, 0, 0] = False
        handle["flow_series_imputed"][index, 0, 0] = handle["flow_series_imputed"][
            index - 1, 0, 0
        ]
        handle["flow_series_imputed"][index + 1, 1, 0] = 0
    contract = helper.freeze_material(data, roads, steps)
    source = FullWindowRegionReader("Alameda", data, roads, contract=contract)
    statistics = fit_fullwindow_statistics(source)
    return data, roads, contract, source, statistics


def grant_for(source, statistics, authorization="synthetic_frozen_only"):
    return FullWindowFrozenGrant(
        authorization,
        "a" * 64,
        "b" * 64,
        "Alameda",
        source.identity["identity_digest"],
        source.sampling_identity["sampling_digest"],
        statistics.identity["identity_digest"],
    )


def frozen_reader(data, roads, contract, source, statistics):
    grant = grant_for(source, statistics)
    reader = FullWindowFrozenReader(
        "Alameda",
        data,
        roads,
        grant=grant,
        statistics=statistics,
        source_reader_identity=source.identity,
        sampling_identity=source.sampling_identity,
        contract=contract,
    )
    return reader, grant


def write_synthetic_checkpoint(root, source, statistics, kind, seed):
    """Formal-shaped local test artifacts do not constitute trained results."""
    root.mkdir()
    data = make_training_data(
        source, statistics, batch_size=2, seed=seed, preload=False
    )
    method = "shared_graph" if kind == "fullwindow_ndr" else "staeformer"
    if kind == "fullwindow_ndr":
        factory = factory_from_data(method, data)
        model = rebuild_model(method, factory)
        schema, scope, summary_status = (
            "fullwindow_ndr_checkpoint_v1",
            "fullwindow_validation_development_only",
            "training_completed_validation_pending_independent_verification",
        )
    else:
        factory = {
            "method": "staeformer",
            "node_count": source.N,
            "flow_mean": statistics.flow_mean,
            "flow_std": statistics.flow_std,
            "bist_checkout": None,
        }
        model = make_strong_reference(**factory)
        schema, scope, summary_status = (
            "fullwindow_strong_checkpoint_v1",
            "fullwindow_validation_development_capped",
            "fullwindow_strong_training_completed_validation_pending_independent_verification",
        )
    files = {
        name: {
            "sha256": file_sha256(ROOT / name),
            "size_bytes": (ROOT / name).stat().st_size,
        }
        for name in (
            "ndr_research/fullwindow_data.py",
            "ndr_research/fullwindow_training.py",
            "scripts/train_ndr_fullwindow.py",
            "scripts/train_strong_fullwindow.py",
            "ndr_research/shared_graph.py",
            "ndr_research/strong_references.py",
        )
    }
    snapshot = {
        "files": files,
        "files_digest": canonical_sha256(files),
        "git": {"git_sha": "synthetic_only"},
    }
    config = {
        "schema_version": schema,
        "model_class": type(model).__name__,
        "model_factory": factory,
        "method": method,
        "region": "Alameda",
        "seed": seed,
        "epochs": 100,
        "max_train_batches": None,
        "max_val_batches": None,
        "sampling_protocol": "full_window",
        "claim_scope": scope,
        "test_evaluated": False,
        "test_numeric_api_available": False,
        "history_policy": source.history_policy,
        "source_snapshot": snapshot,
        "runtime": transfer_runtime_identity(torch.device("cpu")),
        "training_process_id": 1,
        "data_identity": dict(data.identity),
        "reader_identity": source.identity,
        "sampling_identity": source.sampling_identity,
        "statistics_identity": dict(statistics.identity),
        "graph_hidden_dim": 64,
        "training_recipe": reference_recipe(
            "staeformer", "native", region="Alameda", epochs=100
        )
        if kind == "fullwindow_strong"
        else {"loss": "synthetic_graph_ndr"},
    }
    from ndr_research.fullwindow_training import validation_identity

    selection = validation_identity(data)
    collection = {
        "sampling_protocol": "full_window",
        "split": "val",
        "sampling_digest": selection["sampling_digest"],
        "samples": selection["samples"],
        "forecast_origins_sha256": selection["forecast_origins_sha256"],
        "observed_targets_sha256": "c" * 64,
        "observation_masks_sha256": "d" * 64,
        "all_declared_split_samples": True,
    }
    selected = {
        "selection": selection,
        "collection": collection,
        "mae": 5.0,
        "global_masked_mae": 5.0,
        "valid_count": selection["samples"] * 24,
        "horizon_valid_counts": [selection["samples"] * 2] * 12,
        "observed_targets_sha256": "c" * 64,
        "observation_masks_sha256": "d" * 64,
    }
    checkpoint = {
        "schema_version": schema,
        "sampling_protocol": "full_window",
        "model_class": type(model).__name__,
        "model_factory": factory,
        "model": model.state_dict(),
        "config": config,
        "config_digest": canonical_sha256(config),
        "optimizer": {},
        "scheduler": {},
        "rng": {},
        "loaders": {},
        "progress": {"best_epoch": 2},
        "data_identity": dict(data.identity),
        "reader_identity": source.identity,
        "sampling_identity": source.sampling_identity,
        "statistics_state": statistics.state_dict(),
        "statistics_identity": dict(statistics.identity),
        "selection_metric": selected,
        "test_evaluated": False,
        "test_numeric_api_available": False,
    }
    if kind == "fullwindow_strong":
        checkpoint.update(
            native_model_class="STAEformer", model_upstream=model.upstream
        )
    checkpoint_path = root / "best.pt"
    torch.save(checkpoint_safe(checkpoint), checkpoint_path)
    (root / "best.pt.sha256.json").write_text(
        json.dumps(
            {
                "sha256": file_sha256(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "config_digest": checkpoint["config_digest"],
            }
        )
    )
    summary = {
        "status": summary_status,
        "completed_epochs": 100,
        "best_epoch": 2,
        "config_digest": checkpoint["config_digest"],
        "best_validation": selected,
        "test_evaluated": False,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary))
    verification = {
        "schema_version": 1 if kind == "fullwindow_ndr" else schema,
        "status": "verified_fullwindow_ndr_checkpoint_complete_validation"
        if kind == "fullwindow_ndr"
        else "verified_fullwindow_strong_complete_validation_capped",
        "claim_scope": scope,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_digest": checkpoint["config_digest"],
        "source_files_digest": snapshot["files_digest"],
        "model_class": type(model).__name__,
        "reader_identity": source.identity,
        "sampling_identity": source.sampling_identity,
        "statistics_identity": dict(statistics.identity),
        "sampling_protocol": "full_window",
        "test_evaluated": False,
        "training_process_id": 1,
        "verification_process_id": 2,
        "validation": selected,
        "summary_sha256": file_sha256(summary_path),
        "model_factory": factory,
        "data_identity": dict(data.identity),
        "data_identity_digest": data.identity["identity_digest"],
        "source_snapshot_files_digest": snapshot["files_digest"],
    }
    verification_path = root / "verification.json"
    verification_path.write_text(json.dumps(verification))
    return checkpoint_path, verification_path, summary_path


def synthetic_gate(tmp_path, source, statistics):
    models = []
    for kind in ("fullwindow_ndr", "fullwindow_strong"):
        for seed in SEEDS:
            paths = write_synthetic_checkpoint(
                tmp_path / f"{kind}_{seed}", source, statistics, kind, seed
            )
            spec = inspect_fullwindow_model(*paths, kind=kind)
            models.append(
                {
                    "model_id": f"{kind}_s{seed}",
                    "checkpoint_kind": kind,
                    **{
                        field: str(path)
                        for field, path in zip(
                            ("checkpoint", "verification", "summary"), paths
                        )
                    },
                    **{
                        field + "_sha256": file_sha256(path)
                        for field, path in zip(
                            ("checkpoint", "verification", "summary"), paths
                        )
                    },
                    **{
                        field: spec[field]
                        for field in (
                            "family_identity",
                            "training_seed",
                            "model_class",
                            "config_digest",
                            "validation_selection_sha256",
                            "model_state_sha256",
                        )
                    },
                }
            )
    gate = {
        "schema_version": GATE_SCHEMA,
        "sampling_protocol": "full_window",
        "protocol_version": PROTOCOL,
        "split": "test",
        "evaluation_role": "same_region_temporal_heldout",
        "region": "Alameda",
        "structure_development_closed": True,
        "selection_frozen": True,
        "test_used_for_structure_selection": False,
        "authorization_id": "synthetic_only_gate",
        "training_seeds": list(SEEDS),
        "max_test_evaluations": 6,
        "evaluation_ledger": str(tmp_path / "ledger.jsonl"),
        "scenarios": list(SCENARIOS),
        "perturbation_seed": 42,
        "models": models,
        "reader_identity": source.identity,
        "sampling_identity": source.sampling_identity,
        "statistics_identity": dict(statistics.identity),
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    return path, gate


def test_frozen_reader_exact_test_grant_whitelist_and_development_stays_closed(
    tmp_path, monkeypatch
):
    data, roads, contract, source, stats = synthetic_material(tmp_path)
    with pytest.raises(PermissionError):
        source.indices("test")
    with pytest.raises(PermissionError):
        source.get_batch([0], stats, split="test")
    with pytest.raises(PermissionError, match="admitted frozen grant"):
        FullWindowFrozenReader(
            "Alameda",
            data,
            roads,
            grant=None,
            statistics=stats,
            source_reader_identity=source.identity,
            sampling_identity=source.sampling_identity,
        )
    reads = []
    original = h5py.Dataset.__getitem__

    def record(dataset, selection):
        reads.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    reader, grant = frozen_reader(data, roads, contract, source, stats)
    indices = reader.indices()
    batch = reader.get_test_batch(indices[:2], grant=grant)
    assert (
        reader.role == "same_region_temporal_heldout"
        and reader.metadata.role == "target"
    )
    assert reader.metadata.identity["test_numeric_api_available"] is False
    assert reader.access_audit()["test_windows_read"] == 2
    assert all(
        name in {"/times_ns", "/region_idx", "/flow_series_imputed", "/flow_mask"}
        for name, _ in reads
    )
    assert all(
        selection[2] == 0 for name, selection in reads if name.startswith("/flow_")
    )
    assert not batch["history_mask"][0, 0, -1] and batch["flow_hist"][0, 0, -1] > 0
    with pytest.raises(PermissionError, match="another grant"):
        reader.get_test_batch(indices[:1], grant=grant_for(source, stats, "different"))
    with pytest.raises(PermissionError, match="exact declared test"):
        reader.get_test_batch(source.indices("val")[:1], grant=grant)
    with pytest.raises(PermissionError):
        reader.metadata._read_flow(reader.val_end, reader.T, purpose="window:test")


def test_actual_ndr_and_sta_models_share_one_fullwindow_collection_and_real_day_blocks(
    tmp_path,
):
    configure_transfer_runtime("cpu", 42, 1)
    data, roads, contract, source, stats = synthetic_material(tmp_path)
    training = make_training_data(source, stats, batch_size=2, seed=42, preload=False)
    graph = rebuild_model(
        "shared_graph", factory_from_data("shared_graph", training)
    ).eval()
    native = make_strong_reference(
        method="staeformer",
        node_count=source.N,
        flow_mean=stats.flow_mean,
        flow_std=stats.flow_std,
    ).eval()
    models = {
        "graph": FullWindowForecastAdapter(
            graph, kind="fullwindow_ndr", method="shared_graph"
        ).eval(),
        "sta": FullWindowForecastAdapter(
            native, kind="fullwindow_strong", method="staeformer"
        ).eval(),
    }
    reader, grant = frozen_reader(data, roads, contract, source, stats)
    result = evaluate_fullwindow_models(
        models, reader=reader, grant=grant, batch_size=16
    )
    assert reader.access_audit()["test_windows_read"] == len(reader.indices())
    assert (
        result["target_windows_read_once_for_all_models"]
        and result["collection"]["sampling_protocol"] == "full_window"
    )
    assert (
        result["collection"]["fullwindow_forecast_origins_legacy_sha256"]
        == source.sampling_identity["split_origin_sha256"]["test"]
    )
    for model in result["models"].values():
        for scenario in model["scenarios"].values():
            assert scenario["collection"] == result["collection"]
            assert len(scenario["day_blocks"]) == 2
            assert (
                sum(
                    sum(block["valid_count"])
                    for block in scenario["day_blocks"].values()
                )
                == scenario["valid_count"]
            )
            assert scenario["valid_count"] > scenario["percentage_valid_count"]


def test_fullwindow_gate_true_class_full_validation_pid_and_budget(tmp_path):
    configure_transfer_runtime("cpu", 42, 1)
    _, _, _, source, stats = synthetic_material(tmp_path)
    path, declaration = synthetic_gate(tmp_path, source, stats)
    gate, admitted = admit_fullwindow_gate(path)
    assert len(admitted) == 6 and {
        value["model_class"] for value in admitted.values()
    } == {"SharedGraphNDR", "StrongReference"}
    assert all(
        value["normalizer_scalar_dtype"] == "float64"
        and value["legacy27_scalar_dtype"] == "float32"
        for value in admitted.values()
    )
    ledger = FullWindowEvaluationLedger(gate, admitted, path, tmp_path / "first")
    assert ledger.reserve()["training_seed_inference_slots"] == 3
    ledger.finish("failed")
    renamed = {
        "renamed_" + name: dict(spec, checkpoint_sha256="a" * 64)
        for name, spec in admitted.items()
    }
    second = FullWindowEvaluationLedger(
        dict(gate, authorization_id="new"),
        renamed,
        tmp_path / "newgate.json",
        tmp_path / "second",
    )
    assert second.family_sha == ledger.family_sha
    assert second.reserve()["consumed_slots_after"] == 6
    second.finish("completed")
    with pytest.raises(PermissionError, match="budget is exhausted"):
        FullWindowEvaluationLedger(gate, admitted, path, tmp_path / "third").reserve()
    changed = copy.deepcopy(declaration)
    changed["sampling_protocol"] = "event"
    path.write_text(json.dumps(changed))
    with pytest.raises(PermissionError, match="frozen test gate"):
        admit_fullwindow_gate(path)
    row = declaration["models"][0]
    verify = Path(row["verification"])
    value = json.loads(verify.read_text())
    value["verification_process_id"] = value["training_process_id"]
    verify.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="independent process"):
        inspect_fullwindow_model(
            row["checkpoint"], verify, row["summary"], kind=row["checkpoint_kind"]
        )


def test_frozen_reader_detects_material_mutation_and_cli_gate_rejects_before_reader(
    tmp_path, monkeypatch
):
    configure_transfer_runtime("cpu", 42, 1)
    data, roads, contract, source, stats = synthetic_material(tmp_path)
    reader, grant = frozen_reader(data, roads, contract, source, stats)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as handle:
        handle["flow_series_imputed"][reader.T - 1, 0, 0] += 1
    with pytest.raises(RuntimeError, match="raw material changed"):
        reader.get_test_batch(reader.indices()[:1], grant=grant)
    path = tmp_path / "invalid_gate.json"
    path.write_text(json.dumps({"schema_version": GATE_SCHEMA}))

    def forbidden(*args, **kwargs):
        pytest.fail("failed gate created frozen test reader")

    monkeypatch.setattr(cli, "FullWindowFrozenReader", forbidden)
    args = cli.argument_parser().parse_args(
        ["--gate-json", str(path), "--output", str(tmp_path / "rejected")]
    )
    with pytest.raises(PermissionError):
        cli.run_evaluation(args)
    assert not (tmp_path / "rejected").exists()


def test_corruption_does_not_reuse_old_cache_and_clean_retains_original_imputation():
    class NativeSpy(torch.nn.Module):
        def forward(self, history, time):
            self.last_history = history.clone()
            return history[..., -1:].expand(-1, -1, 12)

    spy = NativeSpy()
    adapter = FullWindowForecastAdapter(
        spy, kind="fullwindow_strong", method="staeformer"
    ).eval()
    flow = torch.arange(12).float().reshape(1, 1, 12)
    flow[..., -1] = 999
    mask = torch.ones_like(flow, dtype=torch.bool)
    mask[..., -1] = False
    baseline = torch.full_like(flow, 40)
    calendar = torch.zeros(1, 12, 2)
    arguments = (
        flow,
        baseline,
        baseline,
        calendar,
        mask,
        torch.empty(2, 0, dtype=torch.long),
        torch.ones(1),
    )
    adapter(*arguments)
    assert spy.last_history[..., -1].item() == 999
    adapter.scenario = "last_missing_3steps"
    mask[..., -3:] = False
    adapter(*arguments)
    assert spy.last_history[..., -1].item() == 8


def test_fullwindow_cli_pairs_all_six_models_and_failed_reader_costs_three_slots(
    tmp_path, monkeypatch
):
    configure_transfer_runtime("cpu", 42, 1)
    data, roads, contract, source, stats = synthetic_material(tmp_path, steps=100)
    path, _ = synthetic_gate(tmp_path, source, stats)
    original_reader = cli.FullWindowFrozenReader
    created = []

    def synthetic_reader(*args, **kwargs):
        reader = original_reader(*args, **kwargs, contract=contract)
        created.append(reader)
        return reader

    monkeypatch.setattr(cli, "FullWindowFrozenReader", synthetic_reader)
    args = cli.argument_parser().parse_args(
        [
            "--gate-json",
            str(path),
            "--output",
            str(tmp_path / "report"),
            "--data-dir",
            str(data),
            "--graph-dir",
            str(roads),
            "--batch-size",
            "2",
        ]
    )
    result = cli.run_evaluation(args)
    assert result["status"] == "frozen_fullwindow_evaluation_complete"
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert len(report["models"]) == 6
    assert report["evaluation_role"] == "same_region_temporal_heldout"
    assert (
        report["sampling_protocol"] == "full_window"
        and report["scalar_fit_precision"] == "float64"
    )
    assert report["legacy27_scores_directly_comparable"] is False
    assert (
        created[0].access_audit()["test_windows_read"]
        == source.sampling_identity["counts"]["test"]
    )
    assert all(
        row["collection"] == report["collection"] for row in report["models"].values()
    )
    assert report["model_state_sha256"] == {
        name: row["model_state_sha256"] for name, row in report["models"].items()
    }

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic frozen reader unavailable")

    monkeypatch.setattr(cli, "FullWindowFrozenReader", unavailable)
    args.output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="unavailable"):
        cli.run_evaluation(args)
    records = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert records[-1]["status"] == "failed"
    assert (
        sum(
            record["training_seed_inference_slots"]
            for record in records
            if record["status"] == "reserved"
        )
        == 6
    )


def test_fullwindow_source_statistics_reader_and_sampling_changes_fail_before_test(
    tmp_path,
):
    configure_transfer_runtime("cpu", 42, 1)
    data, roads, contract, source, stats = synthetic_material(tmp_path)
    grant = grant_for(source, stats)
    changed = copy.deepcopy(source.identity)
    changed["history_policy"] = "observed_zero"
    with pytest.raises(ValueError, match="do not align"):
        FullWindowFrozenReader(
            "Alameda",
            data,
            roads,
            grant=grant,
            statistics=stats,
            source_reader_identity=changed,
            sampling_identity=source.sampling_identity,
            contract=contract,
        )
    reader, _ = frozen_reader(data, roads, contract, source, stats)
    with pytest.raises(AttributeError):
        reader.statistics = stats
    with pytest.raises(AttributeError):
        reader.grant = grant
