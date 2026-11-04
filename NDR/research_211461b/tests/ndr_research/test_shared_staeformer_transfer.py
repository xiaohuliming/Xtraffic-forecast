from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.STAEformer.STAEformer import STAEformer
from ndr_research.shared_staeformer import SHARED_STAEFORMER_CONFIG, SharedSTAEformer
from scripts import train_shared_staeformer_transfer as trainer


def fixture_data(batch_size=2):
    path = Path(__file__).with_name("test_transfer_training.py")
    specification = importlib.util.spec_from_file_location("_reference_source_fixture", path)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    data = helper.fixture_data(batch_size)
    state = data.preprocessing.to_state()
    for name in ("mode", "template", "residual_scale"):
        setattr(data.preprocessing, name, state[name])
    return data


def model_from_data(data):
    return SharedSTAEformer(**trainer.model_config_from_sources(data))


def fixture_config(data, epochs=2):
    return {"source_regions": list(data.readers), "heldout_region": "Orange", "epochs": epochs,
            "model_class": "SharedSTAEformer", "architecture": dict(SHARED_STAEFORMER_CONFIG),
            "seed": 42, "batch_size": 2, "microbatch_size": 1, "learning_rate": 0.001,
            "weight_decay": 0.0001, "gradient_clip": 5.0, "eta_min": 0.00001,
            "max_train_batches": None, "max_val_batches": None, "log_interval": 100,
            "threads": 1, "claim_scope": "smoke_only", "source_snapshot": trainer.source_snapshot(),
            "runtime": trainer.transfer_runtime_identity(torch.device("cpu"))}


def batch(nodes=5, batch_size=2):
    flow = torch.rand(batch_size, nodes, 12) * 40
    observed = torch.rand_like(flow) > 0.3
    calendar = torch.stack((torch.arange(12) / 288, torch.ones(12) / 7), dim=-1)
    return {"flow_hist": flow, "x_baseline": torch.full_like(flow, 40),
            "y_baseline": torch.full_like(flow, 40), "time_feat": calendar.repeat(batch_size, 1, 1),
            "history_mask": observed, "edge_index": torch.empty((2, 0), dtype=torch.long),
            "node_scale": torch.full((batch_size, nodes), 5.0)}


def test_native_temporal_tying_matches_dynamic_layout_and_strict_node_transfer():
    trainer.configure_transfer_runtime("cpu", 42, 1)
    model = SharedSTAEformer(flow_center=40, flow_scale=5).eval()
    assert model.native.adaptive_embedding.shape == (12, 1, 80)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1061172
    assert not hasattr(model.native, "node_emb")
    state_shapes = {name: value.shape for name, value in model.state_dict().items()}
    for nodes in (1, 5, 7):
        values = batch(nodes)
        native = STAEformer(num_nodes=nodes, **SHARED_STAEFORMER_CONFIG).eval()
        state = {name: value.clone() for name, value in model.native.state_dict().items()}
        state["adaptive_embedding"] = state["adaptive_embedding"].expand(12, nodes, 80).clone()
        native.load_state_dict(state, strict=True)
        inputs = model.build_inputs(values["flow_hist"], values["history_mask"], values["x_baseline"], values["time_feat"])
        expected = native(inputs).squeeze(-1).permute(0, 2, 1) * 5 + 40
        torch.testing.assert_close(model(**values), expected, rtol=0, atol=0)
        transferred = SharedSTAEformer(flow_center=40, flow_scale=5).eval()
        transferred.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(transferred(**values), expected, rtol=0, atol=0)
        assert {name: value.shape for name, value in transferred.state_dict().items()} == state_shapes


def test_permutation_equivariance_ignored_future_graph_and_sentinel_invariance():
    trainer.configure_transfer_runtime("cpu", 42, 1)
    model = SharedSTAEformer(flow_center=40, flow_scale=5).eval()
    values = batch()
    original = model(**values)
    permutation = torch.tensor([4, 1, 0, 3, 2])
    permuted = {name: value[:, permutation] if name in {"flow_hist", "x_baseline", "history_mask"} else value
                for name, value in values.items()}
    torch.testing.assert_close(model(**permuted), original[:, permutation], rtol=1e-5, atol=5e-6)
    changed = dict(values)
    changed["flow_hist"] = torch.where(values["history_mask"], values["flow_hist"], float("nan"))
    changed["y_baseline"] = torch.full_like(values["y_baseline"], float("nan"))
    changed["node_scale"] = torch.full_like(values["node_scale"], float("nan"))
    changed["edge_index"] = torch.tensor([[999], [999]])
    torch.testing.assert_close(model(**changed), original, rtol=0, atol=0)
    assert "target" not in inspect.signature(model.forward).parameters
    assert "affected_mask" not in inspect.signature(model.forward).parameters


def test_zero_observations_remain_valid_and_calendar_is_restored():
    model = SharedSTAEformer(flow_center=40, flow_scale=5)
    values = batch(1, 1)
    values["flow_hist"].zero_()
    values["history_mask"].fill_(True)
    values["time_feat"][0, :, 1] = torch.arange(12).remainder(7) / 7
    inputs = model.build_inputs(values["flow_hist"], values["history_mask"], values["x_baseline"], values["time_feat"])
    assert (inputs[..., 0] == -8).all()
    torch.testing.assert_close(inputs[0, :, 0, 2], torch.arange(12).remainder(7).float())
    with pytest.raises(ValueError, match="boolean"):
        model(**(values | {"history_mask": values["history_mask"].float()}))


def test_pure_mae_effective_denominator_and_microbatch_gradients():
    trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data(4)
    tensors = trainer.graph_batch_tensors(data.train_loaders["ContraCosta"].get_next(), torch.device("cpu"))
    tensors["target_mask"][0] = False
    tensors["target"][0] = 0
    full = model_from_data(data).eval()
    micro = copy.deepcopy(full)
    expected = trainer.backward_reference_batch(full, tensors, microbatch_size=4)
    actual = trainer.backward_reference_batch(micro, tensors, microbatch_size=1)
    assert expected["valid_count"] == actual["valid_count"]
    assert actual["objective"] == pytest.approx(expected["absolute_error_sum"] / expected["valid_count"], rel=2e-6)
    assert actual["objective"] == pytest.approx(expected["objective"], rel=2e-6)
    assert set(actual["parts"]) == {"mae"}
    for left, right in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=5e-5, atol=4e-5)


def test_two_source_validation_is_complete_and_region_balanced(monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(trainer, "forward_reference", lambda _, tensors: torch.zeros_like(tensors["target"]))
    result = trainer.evaluate_sources(model_from_data(data), data, torch.device("cpu"), microbatch_size=1)
    rows = list(result["regions"].values())
    assert result["all_source_validation_samples"]
    assert result["macro_global_mae"] == np.mean([row["global_masked_mae"] for row in rows])
    assert result["macro_global_mae"] != pytest.approx(sum(row["absolute_error_sum"] for row in rows) / sum(row["valid_count"] for row in rows))
    assert result["source_regions"] == ["Alameda", "ContraCosta"] and not result["target_region_used"]
    data.readers["Alameda"].role = "target"
    with pytest.raises(ValueError, match="source-role"):
        trainer.evaluate_sources(model_from_data(data), data, torch.device("cpu"), microbatch_size=1)


def test_interrupted_reference_restores_weights_optimizer_schedule_rng_and_loaders(tmp_path, monkeypatch):
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data()
    config = fixture_config(data)
    full = model_from_data(data)
    full_output = tmp_path / "full"
    full_output.mkdir()
    trainer.train_reference_run(model=full, data=data, device=device, config=config, output=full_output)
    trainer.configure_transfer_runtime("cpu", 42, 1)
    interrupted_data = fixture_data()
    interrupted = model_from_data(interrupted_data)
    output = tmp_path / "continued"
    output.mkdir()
    ticks = iter(range(100))
    with monkeypatch.context() as limited:
        limited.setattr(trainer.time, "monotonic", lambda: float(next(ticks)))
        stopped = trainer.train_reference_run(model=interrupted, data=interrupted_data, device=device,
                                              config=config, output=output, max_wallclock_seconds=3)
    assert stopped["status"] == "stopped_resumable" and stopped["plan_cursor"] == 2
    checkpoint = trainer.load_reference_checkpoint(output / "last.pt")
    resumed_data, resumed = fixture_data(), model_from_data(data)
    trainer.train_reference_run(model=resumed, data=resumed_data, device=device, config=config,
                                 output=output, checkpoint=checkpoint)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    full_checkpoint = trainer.load_reference_checkpoint(full_output / "last.pt")
    continued_checkpoint = trainer.load_reference_checkpoint(output / "last.pt")
    assert full_checkpoint["scheduler"] == continued_checkpoint["scheduler"]
    for number, state in full_checkpoint["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, continued_checkpoint["optimizer"]["state"][number][name], rtol=0, atol=0)
    for split in ("train", "val"):
        for region in data.readers:
            left, right = full_checkpoint["loaders"][split][region], continued_checkpoint["loaders"][split][region]
            assert left["cursor"] == right["cursor"]
            torch.testing.assert_close(left["permutation"], right["permutation"], rtol=0, atol=0)
    for key in ("torch_cpu",):
        torch.testing.assert_close(full_checkpoint["rng"][key], continued_checkpoint["rng"][key], rtol=0, atol=0)
    assert full_checkpoint["rng"]["python"] == continued_checkpoint["rng"]["python"]
    left_numpy, right_numpy = full_checkpoint["rng"]["numpy"], continued_checkpoint["rng"]["numpy"]
    for name, value in left_numpy.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, right_numpy[name], rtol=0, atol=0)
        else:
            assert value == right_numpy[name]
    report = trainer.verify_reference_checkpoint(checkpoint_path=output / "best.pt", data=resumed_data,
                                                  device=device, output=output,
                                                  current_runtime=trainer.transfer_runtime_identity(device))
    assert report["absolute_mae_difference"] == 0 and report["all_source_validation_samples"]
    assert report["model_config"] == trainer.model_config_from_sources(data)
    assert report["status"] == "verified_smoke_complete_source_validation"


def test_zero_valid_batch_skips_optimizer_and_nonfinite_prediction_fails_closed(monkeypatch):
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data()
    model = model_from_data(data)
    config = fixture_config(data)
    tensors = trainer.graph_batch_tensors(data.train_loaders["Alameda"].get_next(), device)
    tensors["target_mask"].zero_()
    tensors["target"].zero_()
    optimizer, scheduler = trainer.create_transfer_optimizer(model, config, 7)
    state = copy.deepcopy(model.state_dict())
    schedule = copy.deepcopy(scheduler.state_dict())
    result = trainer.optimize_reference_batch(model, optimizer, scheduler, tensors, config)
    assert not result["updated"] and result["objective"] == 0 and result["valid_count"] == 0
    assert scheduler.state_dict() == schedule
    assert optimizer.state_dict()["state"] == {}
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    monkeypatch.setattr(trainer, "forward_reference", lambda _, values: torch.full_like(values["target"], float("nan")))
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.backward_reference_batch(model, tensors, microbatch_size=1)


def test_fixed_cli_and_calibration_are_source_only(tmp_path):
    parser = trainer.argument_parser()
    args = parser.parse_args([])
    assert not hasattr(args, "teacher") and not hasattr(args, "hidden_dim")
    assert args.batch_size == 16 and args.microbatch_size == 2
    with pytest.raises(ValueError, match="CPU training is engineering smoke only"):
        trainer.validate_arguments(parser.parse_args(["--device", "cpu"]))
    with pytest.raises(ValueError, match="frozen source validation"):
        trainer.validate_arguments(parser.parse_args(["--mode", "verify", "--checkpoint", "best.pt", "--max-val-batches", "1"]))
    device = trainer.configure_transfer_runtime("cpu", 42, 1)
    data = fixture_data()
    result = trainer.calibrate_reference(model=model_from_data(data), data=data, device=device,
                                         config=fixture_config(data), output=tmp_path, warmup=0, timed=1)
    assert result["claim_scope"] == "calibration_only" and not result["target_data_read"]
    assert result["timing_includes_reader"]
    assert not (tmp_path / "best.pt").exists()
