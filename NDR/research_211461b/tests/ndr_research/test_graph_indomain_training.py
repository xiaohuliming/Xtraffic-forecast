from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from ndr_research.data import array_digest, canonical_digest, prepare_research_batch, sha256_file
from ndr_research.graph_indomain import batch_tensors, graph_identity, model_config, normalizer_identity
from ndr_research.shared_graph import SharedGraphNDR
from scripts import train_ndr_graph_indomain as trainer
from tests.ndr_research.test_training import tiny_data


@pytest.fixture(autouse=True)
def configure_cpu():
    trainer.configure_transfer_runtime("cpu", 42, 1)


def fixture_data(batch_size=2):
    data = tiny_data(batch_size=batch_size)
    data.region.edge_index = np.asarray([[0], [1]], dtype=np.int64)
    data.identity.update({
        "region": "Alameda", "protocol_version": "fixture_v3", "cache_schema_version": 3,
        "fit_range": [0, data.region.train_end], "fit_positions": "unique_raw_observed_training_timestamps",
        "weekly_climatology_sha256": array_digest(data.statistics.baseline),
        "node_residual_std_sha256": array_digest(data.statistics.residual_std_by_node),
        "valid_training_points": data.statistics.valid_training_points,
        "training_analysis_labels": False,
    })
    return data


def fixture_config(data, *, epochs=2, max_val=None):
    return {"region": "Alameda", "hidden_dim": 64, "model_class": "SharedGraphNDR",
            "model_config": model_config(64), "teacher": "original", "epochs": epochs,
            "seed": 42, "batch_size": data.train.bs, "microbatch_size": 1,
            "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 5.0,
            "eta_min": 1e-5, "lambda_gate": 1.0, "lambda_specialization": 0.3,
            "max_train_batches": None, "max_val_batches": max_val, "log_interval": 100,
            "claim_scope": "smoke_only", "source_snapshot": trainer.source_snapshot(),
            "runtime": trainer.transfer_runtime_identity(torch.device("cpu")),
            "data_identity": data.identity, "normalizer_identity": normalizer_identity(data),
            "graph_identity": graph_identity(data), "training_process_id": -1}


def new_model():
    return SharedGraphNDR(**model_config(64))


def test_bridge_preserves_original_event_inputs_and_adds_runtime_train_scales_and_edges():
    data = fixture_data()
    batch = data.val.next_batch()
    original = prepare_research_batch(batch, data.statistics)
    actual = batch_tensors(batch, data, torch.device("cpu"))
    for field in ("flow_hist", "x_baseline", "y_baseline", "time_feat", "history_mask", "target_mask"):
        np.testing.assert_array_equal(actual[field].numpy(), original[field])
    np.testing.assert_array_equal(actual["edge_index"], data.region.edge_index)
    np.testing.assert_array_equal(actual["node_scale"], [[4.0, 6.0], [4.0, 6.0]])
    assert set(actual) == set(trainer.forward_graph.__globals__["FORWARD_FIELDS"]) | {"target", "target_mask"}
    assert not any(name == "sd_node" for name, _ in new_model().named_buffers())
    assert "source_regions" not in fixture_config(data)
    with pytest.raises(ValueError, match="analysis labels"):
        batch_tensors(batch | {"affected_mask": np.zeros(1)}, data, torch.device("cpu"))


def test_train_normalizer_and_static_graph_bind_the_exact_region_and_train_prefix():
    data = fixture_data()
    identity = normalizer_identity(data)
    assert identity["fit_range"] == [0, 50]
    assert identity["node_residual_std_sha256"] == array_digest(np.asarray([4.0, 6.0], np.float32))
    assert graph_identity(data)["edge_index_sha256"] == array_digest(data.region.edge_index)
    data.identity["fit_range"] = [0, 100]
    with pytest.raises(ValueError, match="raw observed train"):
        normalizer_identity(data)
    data.region.edge_index = np.asarray([[0], [2]], dtype=np.int64)
    with pytest.raises(ValueError, match="node bounds"):
        graph_identity(data)


def test_fixed_architecture_and_cli_cpu_smoke_bounds():
    defaults = trainer.argument_parser().parse_args([])
    assert (defaults.batch_size, defaults.microbatch_size) == (16, 8)
    assert (defaults.lambda_gate, defaults.lambda_specialization) == (1.0, 0.3)
    assert model_config(240)["road_layers"] == 2 and model_config(240)["latent_tokens"] == 8
    with pytest.raises(ValueError, match="fixed hidden"):
        model_config(128)
    args = trainer.argument_parser().parse_args(["--device", "cpu"])
    with pytest.raises(ValueError, match="explicit bounded"):
        trainer.validate_arguments(args)
    args = trainer.argument_parser().parse_args(["--device", "cpu", "--max-train-batches", "1", "--max-val-batches", "1"])
    trainer.validate_arguments(args)


class ZeroForecast(torch.nn.Module):
    def forward(self, flow_hist, x_baseline, y_baseline, time_feat, history_mask, edge_index, node_scale):
        return torch.zeros_like(y_baseline)


def test_full_global_validation_retains_tail_and_weights_all_observed_points():
    data = fixture_data()
    actual = trainer.evaluate(ZeroForecast(), data, torch.device("cpu"), microbatch_size=1)
    targets, masks = [], []
    for index in data.val.indices:
        sample = data.region.get_sample(int(index))
        targets.append(sample["y_true"][..., 0])
        masks.append(sample["y_mask"][..., 0])
    targets, masks = np.stack(targets), np.stack(masks)
    assert actual["selection"]["all_validation_samples"] and actual["selection"]["samples"] == 3
    assert actual["valid_count"] == int(masks.sum())
    assert actual["global_masked_mae"] == pytest.approx(targets[masks].mean(dtype=np.float64))
    assert actual["mae"] != pytest.approx(actual["average_horizon_mae"])


@pytest.mark.parametrize("budget, stage", [(6, "train"), (9, "validate")])
def test_completed_effective_update_resume_matches_continuous_and_rejects_cursor_drift(tmp_path, monkeypatch, budget, stage):
    data = fixture_data()
    config = fixture_config(data)
    continuous = new_model()
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    trainer.train_run(model=continuous, data=data, device=torch.device("cpu"), config=config, output=full_dir)
    trainer.configure_transfer_runtime("cpu", 42, 1)
    interrupted = new_model()
    data = fixture_data()
    output = tmp_path / "resumed"
    output.mkdir()
    clock = iter(range(100))
    with monkeypatch.context() as limited:
        limited.setattr(trainer.time, "monotonic", lambda: float(next(clock)))
        result = trainer.train_run(model=interrupted, data=data, device=torch.device("cpu"),
                                   config=config, output=output, max_wallclock_seconds=budget)
    assert result["status"] == "stopped_resumable"
    if stage == "train":
        assert result["next_batch"] == 2
    else:
        assert result["next_stage"] == "validate"
    checkpoint = trainer.load_checkpoint(output / "last.pt")
    assert checkpoint["schema_version"] == "graph_indomain_checkpoint_v1"
    assert checkpoint["progress"]["stage"] == stage
    drift = copy.deepcopy(checkpoint)
    drift["progress"]["partial_epoch"]["batches"] = 0
    optimizer, scheduler = trainer.create_optimizer(new_model(), config, data.train.num_batch)
    with pytest.raises(ValueError, match="committed loader cursor"):
        trainer.restore_training(drift, new_model(), optimizer, scheduler, fixture_data(), config)
    restored = new_model()
    trainer.train_run(model=restored, data=fixture_data(), device=torch.device("cpu"),
                      config=config, output=output, checkpoint=checkpoint)
    for name, expected in continuous.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    actual = trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=fixture_data(),
                                       device=torch.device("cpu"), output=output, current_runtime=config["runtime"])
    assert actual["status"] == "verified_graph_indomain_smoke_complete_validation"
    assert actual["absolute_mae_difference"] == 0 and actual["test_evaluated"] is False


def test_checkpoint_rejects_schema_corruption_config_changes_and_new_graph(tmp_path):
    data = fixture_data()
    config = fixture_config(data, epochs=1)
    output = tmp_path / "run"
    output.mkdir()
    trainer.train_run(model=new_model(), data=data, device=torch.device("cpu"), config=config, output=output)
    cp = trainer.load_checkpoint(output / "best.pt")
    wrong = fixture_data()
    wrong.region.edge_index = np.asarray([[1], [0]], np.int64)
    with pytest.raises(ValueError, match="normalization or static graph"):
        trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=wrong, device=torch.device("cpu"),
                                  output=output, current_runtime=config["runtime"])
    optimizer, scheduler = trainer.create_optimizer(new_model(), config, data.train.num_batch)
    with pytest.raises(ValueError, match="code/config/cache"):
        trainer.restore_training(cp, new_model(), optimizer, scheduler, data, config | {"teacher": "projected"})
    cp["schema_version"] = 1
    torch.save(cp, output / "best.pt")
    (output / "best.pt.sha256.json").write_text(json.dumps({"sha256": sha256_file(output / "best.pt"),
             "size_bytes": (output / "best.pt").stat().st_size, "config_digest": cp["config_digest"]}))
    with pytest.raises(ValueError, match="graph indomain checkpoint"):
        trainer.load_checkpoint(output / "best.pt")


def test_subset_verification_is_explicit_and_same_process_is_rejected(tmp_path):
    data = fixture_data()
    config = fixture_config(data, epochs=1, max_val=1)
    output = tmp_path / "run"
    output.mkdir()
    trainer.train_run(model=new_model(), data=data, device=torch.device("cpu"), config=config, output=output)
    verified = trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=fixture_data(),
                                         device=torch.device("cpu"), output=output, current_runtime=config["runtime"])
    assert verified["status"] == "verified_graph_indomain_smoke_validation_subset"
    assert verified["validation"]["selection"]["all_validation_samples"] is False
    cp = trainer.load_checkpoint(output / "best.pt")
    cp["config"]["training_process_id"] = trainer.os.getpid()
    cp["config_digest"] = canonical_digest(cp["config"])
    torch.save(cp, output / "best.pt")
    (output / "best.pt.sha256.json").write_text(json.dumps({"sha256": sha256_file(output / "best.pt"),
             "size_bytes": (output / "best.pt").stat().st_size, "config_digest": cp["config_digest"]}))
    with pytest.raises(ValueError, match="independent process"):
        trainer.verify_checkpoint(checkpoint_path=output / "best.pt", data=fixture_data(),
                                  device=torch.device("cpu"), output=output, current_runtime=config["runtime"])


def test_calibration_times_complete_effective_batches_and_never_reads_test(tmp_path):
    data = fixture_data()
    config = fixture_config(data, epochs=1)
    result = trainer.calibrate(new_model(), data, torch.device("cpu"), config, warmup=0, timed=1, output=tmp_path)
    assert result["splits"]["train"]["full_effective_batch_size"] == 2
    assert result["splits"]["val"]["full_effective_batch_size"] == 2
    assert result["claim_scope"] == "calibration_only" and result["test_evaluated"] is False
