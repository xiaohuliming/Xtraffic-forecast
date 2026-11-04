from __future__ import annotations

import copy
import json

import h5py
import numpy as np
import pytest
import torch

from ndr_research.data import sha256_file
from ndr_research.fullwindow_training import (
    METHODS, batch_tensors, factory_from_data, load_training_data, objective_recipe,
    optimize_batch, rebuild_model, validation_identity,
)
from scripts import train_ndr_fullwindow as trainer
from tests.ndr_research.test_fullwindow_data import make_material


@pytest.fixture(autouse=True)
def configure_cpu():
    trainer.configure_transfer_runtime("cpu", 42, 1)


def fixture_data(tmp_path, *, batch_size=16, preload=True):
    data, graph, contract = make_material(tmp_path)
    return load_training_data(region="Alameda", data_dir=data, graph_dir=graph,
                              batch_size=batch_size, seed=42, preload=preload, contract=contract)


def fixture_config(data, method="baseline", *, epochs=2, max_val=None):
    teacher = "projected" if method == "projected_oracle" else "original"
    factory = factory_from_data(method, data)
    model = rebuild_model(method, factory)
    return {"schema_version": trainer.CHECKPOINT_SCHEMA, "method": method, "teacher": teacher,
            "graph_hidden_dim": 64, "model_factory": factory, "model_class": type(model).__name__,
            "epochs": epochs, "seed": 42, "batch_size": data.train.batch_size, "microbatch_size": 8,
            "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 5.0, "eta_min": 1e-5,
            "lambda_gate": 1.0, "lambda_specialization": 0.3, "max_train_batches": None,
            "max_val_batches": max_val, "log_interval": 100, "threads": 1, "region": "Alameda",
            "runtime": trainer.transfer_runtime_identity(torch.device("cpu")), "training_process_id": -1,
            "source_snapshot": trainer.source_snapshot(), "claim_scope": "smoke_only",
            "data_identity": dict(data.identity), "statistics_identity": dict(data.statistics.identity),
            "reader_identity": data.reader.identity, "sampling_identity": data.reader.sampling_identity,
            "objective_recipe": objective_recipe(method, teacher)}


def model_for(config):
    return rebuild_model(config["method"], config["model_factory"])


def test_training_data_has_own_fullwindow_sampling_authority_and_no_event_access(tmp_path, monkeypatch):
    read = []
    original = h5py.Dataset.__getitem__
    def track(dataset, selected):
        read.append(dataset.name)
        return original(dataset, selected)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    data = fixture_data(tmp_path)
    assert data.identity["sampling_protocol"] == "full_window"
    assert data.identity["reader_identity"]["material"]["authority_role"] == "raw_traffic_and_static_graph_only"
    assert data.identity["sampling_identity"]["counts"]["train"] == data.train.size == 47
    assert data.identity["sampling_identity"]["event_sample_identity_used"] is False
    assert not hasattr(data, "region") and not hasattr(data, "readers")
    assert set(read) <= {"/times_ns", "/region_idx", "/flow_series_imputed", "/flow_mask"}
    assert data.reader.access_audit()["test_numeric_values_read"] is False
    selection = validation_identity(data)
    assert selection["sampling_digest"] == data.reader.sampling_identity["sampling_digest"]
    assert selection["all_validation_samples"] and selection["samples"] == 4


@pytest.mark.parametrize("method", METHODS)
def test_real_method_dispatch_uses_actual_original_objective_and_runtime_graph_sigma(tmp_path, method):
    data = fixture_data(tmp_path)
    config = fixture_config(data, method)
    model = model_for(config)
    data.train.start_epoch(epoch=0)
    tensors = batch_tensors(data.train.get_next(), torch.device("cpu"))
    optimizer, scheduler = trainer.create_optimizer(model, config, data.train.num_batch)
    result = optimize_batch(model, optimizer, scheduler, tensors, config)
    assert result["updated"] and result["valid_count"] > 0
    assert set(result["parts"]) == {"mae", "gate", "spec"}
    assert all(np.isfinite(value) for value in result["parts"].values())
    if method == "shared_graph":
        np.testing.assert_array_equal(model.aux["node_scale"][0, :, 0].detach().numpy(), data.statistics.residual_std_by_node)
        assert not any(name == "sd_node" for name, _ in model.named_buffers())
    else:
        assert model.research_spec["variant"] == method
        np.testing.assert_array_equal(model.sd_node.numpy(), data.statistics.residual_std_by_node)
    assert scheduler.last_epoch == 1
    assert data.reader.access_audit()["test_numeric_values_read"] is False


class ZeroNativeForecast(torch.nn.Module):
    research_spec = {"variant": "baseline"}
    def forward(self, flow_hist, x_baseline, y_baseline, time_feat):
        return torch.zeros_like(y_baseline)


def test_complete_global_validation_keeps_tail_and_own_collection_hash(tmp_path):
    data = fixture_data(tmp_path, batch_size=3)
    result = trainer.evaluate(ZeroNativeForecast(), data, torch.device("cpu"), method="baseline", microbatch_size=2)
    batch = data.reader.get_batch(data.val.indices, data.statistics, split="val")
    assert result["global_masked_mae"] == pytest.approx(batch["target"][batch["target_mask"]].mean(dtype=np.float64))
    assert result["valid_count"] == int(batch["target_mask"].sum()) == 4 * 2 * 12
    assert result["selection"]["samples"] == 4 and result["batches"] == 2
    assert result["collection"]["all_declared_split_samples"] is True
    assert result["collection"]["sampling_protocol"] == "full_window"
    assert result["collection"]["forecast_origins_sha256"] == validation_identity(data)["forecast_origins_sha256"]
    assert data.reader.access_audit()["test_numeric_values_read"] is False


@pytest.mark.parametrize("method,budget,stage", [("baseline", 6, "train"), ("shared_graph", 9, "validate")])
def test_fullwindow_cursor_epoch_and_rng_resume_match_continuous(tmp_path, monkeypatch, method, budget, stage):
    material = tmp_path / "material"
    material.mkdir()
    data = fixture_data(material)
    config = fixture_config(data, method)
    trainer.configure_transfer_runtime("cpu", 42, 1)
    continuous = model_for(config)
    full = tmp_path / "full"
    full.mkdir()
    trainer.train_run(model=continuous, data=data, device=torch.device("cpu"), config=config, output=full)
    trainer.configure_transfer_runtime("cpu", 42, 1)
    model = model_for(config)
    restored_data = load_training_data(region="Alameda", data_dir=material/'data', graph_dir=material/'graph',
                                       batch_size=16, seed=42, contract=data.reader.contract)
    output = tmp_path / "resumed"
    output.mkdir()
    clock = iter(range(100))
    with monkeypatch.context() as bounded:
        bounded.setattr(trainer.time, "monotonic", lambda: float(next(clock)))
        stopped = trainer.train_run(model=model, data=restored_data, device=torch.device("cpu"), config=config,
                                    output=output, max_wallclock_seconds=budget)
    assert stopped["status"] == "stopped_resumable"
    checkpoint = trainer.load_checkpoint(output / "last.pt")
    assert checkpoint["progress"]["stage"] == stage
    assert checkpoint["loaders"]["train"]["schema_version"] == "fullwindow_loader_v1"
    assert checkpoint["loaders"]["train"]["epoch"] == checkpoint["progress"]["epoch"] == 0
    assert checkpoint["loaders"]["train"]["cursor"] == (32 if stage == "train" else 47)
    resumed = model_for(config)
    trainer.train_run(model=resumed, data=restored_data, device=torch.device("cpu"), config=config, output=output, checkpoint=checkpoint)
    for name, expected in continuous.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    verification = trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=restored_data, device=torch.device("cpu"),
                                              output=output, current_runtime=config["runtime"])
    assert verification["absolute_mae_difference"] == 0
    assert verification["status"] == "verified_fullwindow_ndr_smoke_complete_validation"
    assert verification["statistics_identity"] == dict(data.statistics.identity)
    assert verification["test_evaluated"] is False


def test_fullwindow_checkpoint_rejects_changed_sampling_schema_and_loader_epoch(tmp_path):
    data = fixture_data(tmp_path)
    config = fixture_config(data, "shared_graph", epochs=1)
    output = tmp_path / "run"
    output.mkdir()
    trainer.train_run(model=model_for(config), data=data, device=torch.device("cpu"), config=config, output=output)
    checkpoint = trainer.load_checkpoint(output / "last.pt")
    wrong = copy.deepcopy(checkpoint)
    wrong["progress"]["stage"] = "validate"
    wrong["progress"]["epoch"] = 0
    wrong["loaders"]["train"]["epoch"] = 1
    optimizer, scheduler = trainer.create_optimizer(model_for(config), config, data.train.num_batch)
    with pytest.raises(ValueError, match="cursor/epoch"):
        trainer.restore_training(wrong, model_for(config), optimizer, scheduler, data, config)
    checkpoint["schema_version"] = "graph_indomain_checkpoint_v1"
    torch.save(checkpoint, output / "last.pt")
    (output / "last.pt.sha256.json").write_text(json.dumps({"sha256": sha256_file(output/'last.pt'),
        "size_bytes": (output/'last.pt').stat().st_size, "config_digest": checkpoint["config_digest"]}))
    with pytest.raises(ValueError, match="independent full-window"):
        trainer.load_checkpoint(output / "last.pt")


@pytest.mark.parametrize("method", ["baseline", "shared_graph"])
def test_calibration_runs_full_effective_original_objective(tmp_path, method):
    data = fixture_data(tmp_path, batch_size=2)
    config = fixture_config(data, method)
    config["microbatch_size"] = 1
    report = trainer.calibrate(model_for(config), data, torch.device("cpu"), config, warmup=0, timed=1, output=tmp_path)
    assert report["splits"]["train"]["full_effective_batch_size"] == 2
    assert set(report["actual_train_objective_parts"][0]) == {"mae", "gate", "spec"}
    assert report["objective_recipe"]["normalizer_scope"].startswith("complete_effective_batch")
    assert report["test_evaluated"] is False


def test_cli_defaults_fixed_architectures_and_explicit_cpu_subsets():
    args = trainer.argument_parser().parse_args([])
    assert (args.batch_size, args.microbatch_size, args.epochs) == (16, 8, 30)
    args.device, args.teacher = "cpu", "original"
    with pytest.raises(ValueError, match="explicit bounded"):
        trainer.validate_arguments(args)
    with pytest.raises(ValueError, match="actual variant"):
        objective_recipe("baseline", "projected")
