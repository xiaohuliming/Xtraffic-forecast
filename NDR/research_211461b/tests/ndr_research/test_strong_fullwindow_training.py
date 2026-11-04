"""Synthetic full-window strong controls; no real target labels or GPU access."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ndr_research.fullwindow_training import load_training_data, validation_identity
from ndr_research.strong_references import make_strong_reference
from scripts import train_strong_fullwindow as runner
from scripts.train_ndr_research import configure_runtime, empty_partial, runtime_identity
from scripts.train_ndr_strong_reference import create_reference_optimizer, optimize_reference
from tests.ndr_research.test_fullwindow_data import make_material

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bist_checkout():
    for path in (ROOT / ".third_party/igstgnn_table4/pristine/bist_dd94adf",
                 Path("/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf")):
        if (path / "src/models/bist.py").is_file():
            return path
    pytest.skip("locked pristine BiST checkout is not installed")


def tiny_data(tmp_path, batch_size=3):
    data, graph, contract = make_material(tmp_path)
    return load_training_data(region="Alameda", data_dir=data, graph_dir=graph,
                              batch_size=batch_size, seed=42, preload=False, contract=contract)


def make_model(method, data, checkout=None):
    return make_strong_reference(method=method, node_count=data.reader.N,
                                 flow_mean=data.statistics.flow_mean, flow_std=data.statistics.flow_std,
                                 bist_checkout=checkout)


def make_config(data, method="staeformer", *, epochs=1, recipe="native", max_train=None, max_val=None):
    training_recipe = runner.fullwindow_recipe(method, recipe, region="Alameda", epochs=30,
                                               batch_size=data.train.bs, microbatch_size=data.train.bs)
    return {"schema_version": runner.SCHEMA, "sampling_protocol": "full_window", "model_class": "StrongReference",
            "method": method, "recipe": recipe, "region": "Alameda", "seed": 42, "epochs": epochs,
            "requested_epochs": 30, "batch_size": data.train.bs, "microbatch_size": data.train.bs,
            "training_recipe": training_recipe, "max_train_batches": max_train, "max_val_batches": max_val,
            "claim_scope": "smoke_only", "source_snapshot": runner.source_snapshot(), "runtime": runtime_identity(torch.device("cpu")),
            "training_process_id": 999999, "data_identity": data.identity}


def committed_progress(step, data):
    partial = empty_partial()
    partial.update(batches=1, error_sum=step["error_sum"], valid_count=step["valid_count"],
                   objective_sum=step["objective"], optimized_batches=int(step["updated"]))
    return {"epoch": 0, "stage": "train", "best_epoch": None, "best_val_mae": None,
            "updates": int(step["updated"]), "history": [], "partial_epoch": partial, "elapsed_seconds": 0.0}


@pytest.mark.parametrize("method", ["staeformer", "bist"])
def test_real_strong_controls_optimize_and_select_all_fullwindow_points(tmp_path, method, bist_checkout):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)
    config = make_config(data, method)
    model = make_model(method, data, bist_checkout)
    out = tmp_path / "run"
    out.mkdir()
    result = runner.train_strong(model=model, data=data, device=torch.device("cpu"), config=config, output=out)
    assert result["completed_epochs"] == 1 and result["test_evaluated"] is False
    assert result["best_validation"]["valid_count"] == 4 * 2 * 12
    assert result["best_validation"]["selection"] == validation_identity(data)
    assert result["best_validation"]["collection"]["all_declared_split_samples"]
    checkpoint = runner.load_checkpoint(out / "best.pt")
    assert checkpoint["schema_version"] == "fullwindow_strong_checkpoint_v1"
    assert checkpoint["sampling_protocol"] == "full_window"
    assert checkpoint["statistics_identity"] == dict(data.statistics.identity)
    assert checkpoint["reader_identity"] == data.reader.identity
    assert checkpoint["native_model_class"] == ("STAEformer" if method == "staeformer" else "BiST")
    assert result["updates"] == data.train.num_batch
    assert checkpoint["progress"]["history"][0]["train_batches"] == 16
    assert all(event["stop"] <= data.reader.val_end for event in data.reader.access_audit()["recent_access"] if event["kind"] == "flow_bounds")


def test_global_validation_is_observed_point_weighted_and_keeps_tail(tmp_path):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)

    class Zero(torch.nn.Module):
        def forward(self, flow_hist, time_feat):
            return torch.zeros_like(flow_hist)

    result = runner.evaluate_strong(Zero(), data, torch.device("cpu"), microbatch_size=2)
    direct_sum = direct_count = 0
    data.val.start_epoch(0)
    sizes = []
    for batch in data.val.get_iterator():
        sizes.append(len(batch["target"]))
        direct_sum += batch["target"][batch["target_mask"]].sum(dtype=np.float64)
        direct_count += int(batch["target_mask"].sum())
    assert sizes == [3, 1]
    assert result["global_masked_mae"] == pytest.approx(direct_sum / direct_count)
    assert result["valid_count"] == direct_count
    assert result["selection"]["samples"] == 4


def test_strict_checkpoint_restores_optimizer_scheduler_rng_and_committed_cursor(tmp_path):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)
    config = make_config(data, epochs=2)
    model = make_model("staeformer", data)
    optimizer, scheduler = create_reference_optimizer(model, config, data.train.num_batch)
    data.train.start_epoch(0)
    step = optimize_reference(model, optimizer, scheduler, runner.tensors_from_batch(data.train.get_next(), torch.device("cpu")), config)
    path = tmp_path / "last.pt"
    runner.save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config,
                           progress=committed_progress(step, data))
    checkpoint = runner.load_checkpoint(path)
    expected_step = optimize_reference(model, optimizer, scheduler, runner.tensors_from_batch(data.train.get_next(), torch.device("cpu")), config)
    expected = copy.deepcopy(model.state_dict())
    restored = make_model("staeformer", data)
    restored_optimizer, restored_scheduler = create_reference_optimizer(restored, config, data.train.num_batch)
    progress = runner.restore_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler, data, config)
    assert data.train.cursor == 3 and data.train.epoch == 0 and progress["updates"] == 1
    observed_step = optimize_reference(restored, restored_optimizer, restored_scheduler, runner.tensors_from_batch(data.train.get_next(), torch.device("cpu")), config)
    assert observed_step["objective"] == expected_step["objective"]
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    changed = copy.deepcopy(checkpoint)
    changed["progress"]["partial_epoch"]["batches"] = 2
    with pytest.raises(ValueError, match="committed optimization progress"):
        runner.restore_checkpoint(changed, restored, restored_optimizer, restored_scheduler, data, config)
    changed = copy.deepcopy(checkpoint)
    changed["model"]["flow_mean"].add_(1)
    with pytest.raises(ValueError, match="scalar buffers"):
        runner.restore_checkpoint(changed, restored, restored_optimizer, restored_scheduler, data, config)


def test_wallclock_stop_and_resume_complete_frozen_training(tmp_path):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)
    config, model = make_config(data, max_train=2, max_val=1), make_model("staeformer", data)
    output = tmp_path / "run"
    output.mkdir()
    stopped = runner.train_strong(model=model, data=data, device=torch.device("cpu"), config=config, output=output,
                                  max_wallclock_seconds=1e-12)
    assert stopped["status"] == "stopped_resumable_fullwindow"
    checkpoint = runner.load_checkpoint(output / "last.pt")
    finished = runner.train_strong(model=make_model("staeformer", data), data=data, device=torch.device("cpu"),
                                   config=config, output=output, checkpoint=checkpoint)
    assert finished["completed_epochs"] == 1 and finished["updates"] == 2


def test_recipe_caps_normalization_precision_and_native_bn_are_explicit():
    for method in ("staeformer", "bist"):
        for recipe in ("native", "matched-ndr"):
            value = runner.fullwindow_recipe(method, recipe, region="Alameda", epochs=100)
            assert value["sampling_protocol"] == "full_window"
            assert value["budget_label"] == "fullwindow_capped_100_epoch_development"
            assert value["native_convergence_reproduced"] is False
            assert value["legacy27_normalizer_difference"] == "legacy_scalar_mean_std_used_float32_reductions"
    with pytest.raises(ValueError, match="full effective"):
        runner.fullwindow_recipe("bist", "native", region="Alameda", epochs=30, batch_size=8, microbatch_size=2)
    with pytest.raises(ValueError, match="explicit declaration"):
        runner.fullwindow_recipe("bist", "matched-ndr", region="Alameda", epochs=30, batch_size=8, microbatch_size=2)
    value = runner.fullwindow_recipe("bist", "matched-ndr", region="Alameda", epochs=30,
                                     batch_size=8, microbatch_size=2, allow_bn_microbatch_override=True)
    assert value["batch_normalization_scope"] == "microbatch_override" and value["bn_microbatch_override_declared"]


def test_cpu_calibration_uses_real_masked_update_and_native_full_batch(tmp_path, bist_checkout):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)
    config = make_config(data, "bist")
    model = make_model("bist", data, bist_checkout)
    sizes = []
    handle = model.register_forward_pre_hook(lambda _module, args: sizes.append(args[0].shape[0]))
    before = [value.clone() for value in model.parameters()]
    result = runner.calibrate_strong(model=model, data=data, device=torch.device("cpu"), config=config,
                                     warmup_batches=1, timed_batches=2, output=tmp_path)
    handle.remove()
    assert result["cuda_budget_eligible"] is False
    assert result["status"] == "calibrated_fullwindow_cpu_engineering_only"
    assert sizes == [3] * 6
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    assert result["phases"]["val"]["timed_effective_batch_sizes"] == [3, 3]
    assert result["phases"]["val"]["skipped_retained_tails"] > 0
    assert len(result["phases"]["train"]["timed_batch_seconds"]) == 2


def test_cli_rejects_cpu_formal_subsets_and_resume_recipe_changes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="explicit engineering smoke"):
        runner.validate_arguments(runner.argument_parser().parse_args(["--device", "cpu"]))
    with pytest.raises(ValueError, match="smoke-only"):
        runner.validate_arguments(runner.argument_parser().parse_args(["--max-train-batches", "1"]))
    with pytest.raises(ValueError, match="last.pt"):
        runner.validate_arguments(runner.argument_parser().parse_args(["--resume", "best.pt"]))
    data = tiny_data(tmp_path)
    monkeypatch.setattr(runner, "load_training_data", lambda **_kwargs: data)
    output = tmp_path / "run"
    runner.main(["--smoke", "--device", "cpu", "--threads", "1", "--batch-size", "3",
                 "--max-train-batches", "1", "--max-val-batches", "1", "--output-dir", str(output)])
    with pytest.raises(ValueError, match="override --epochs"):
        runner.main(["--resume", str(output / "last.pt"), "--device", "cpu", "--epochs", "100"])


def test_fresh_process_cli_verifies_exact_saved_collection_without_target_data(tmp_path, monkeypatch):
    configure_runtime("cpu", 42, 1)
    data_dir, graph_dir, contract = make_material(tmp_path)
    real_loader = load_training_data
    monkeypatch.setattr(runner, "load_training_data", lambda **kwargs: real_loader(**(kwargs | {"contract": contract})))
    output = tmp_path / "run"
    runner.main(["--smoke", "--device", "cpu", "--threads", "1", "--batch-size", "3",
                 "--max-train-batches", "1", "--data-dir", str(data_dir), "--graph-dir", str(graph_dir),
                 "--lazy", "--output-dir", str(output)])
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(dataclasses.asdict(contract)))
    code = """import json,sys
from pathlib import Path
from scripts import train_strong_fullwindow as r
from ndr_research.fullwindow_data import FullWindowMaterialContract
original=r.load_training_data
contract=FullWindowMaterialContract(**json.loads(Path(sys.argv[1]).read_text()))
r.load_training_data=lambda **kwargs: original(**(kwargs|{'contract':contract}))
r.main(sys.argv[2:])
"""
    subprocess.run(["/opt/anaconda3/bin/python", "-c", code, str(contract_path), "--mode", "verify", "--device", "cpu",
                    "--checkpoint", str(output / "best.pt"), "--data-dir", str(data_dir), "--graph-dir", str(graph_dir)],
                   cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    verified = json.loads((output / "verification.json").read_text())
    assert verified["status"] == "verified_fullwindow_strong_smoke_only"
    assert verified["training_process_id"] != verified["verification_process_id"]
    assert verified["validation"]["selection"]["all_validation_samples"]
    assert verified["sampling_identity"]["counts"] == {"train": 47, "val": 4, "test": 4, "total": 55}
    assert verified["test_evaluated"] is False


def test_snapshot_contains_real_sampling_model_objective_and_locked_sources():
    files = runner.source_snapshot()["files"]
    required = {"scripts/train_strong_fullwindow.py", "ndr_research/fullwindow_data.py", "ndr_research/fullwindow_training.py",
                "ndr_research/strong_references.py", "scripts/train_ndr_strong_reference.py", "dist_net/protocol.py",
                "baselines/STAEformer/STAEformer.py", "scripts/reproduction/igstgnn_table4/upstreams.lock.json"}
    assert required <= set(files)
    assert "ndr_research/fullwindow_evaluation.py" not in files
    assert "scripts/evaluate_ndr_fullwindow.py" not in files


def test_checkpoint_hash_and_scope_changes_cannot_be_relabelled_event(tmp_path):
    configure_runtime("cpu", 42, 1)
    data = tiny_data(tmp_path)
    config, model = make_config(data), make_model("staeformer", data)
    optimizer, scheduler = create_reference_optimizer(model, config, data.train.num_batch)
    step = optimize_reference(model, optimizer, scheduler, runner.tensors_from_batch(data.train.get_next(), torch.device("cpu")), config)
    path = tmp_path / "last.pt"
    runner.save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config,
                           progress=committed_progress(step, data))
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="byte identity"):
        runner.load_checkpoint(path)
    path.write_bytes(original)
    checkpoint = runner.load_checkpoint(path)
    checkpoint["sampling_protocol"] = "event"
    torch.save(checkpoint, path)
    marker = path.with_suffix(".pt.sha256.json")
    payload = json.loads(marker.read_text())
    payload.update(sha256=runner.sha256_file(path), size_bytes=path.stat().st_size)
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independent full-window strong"):
        runner.load_checkpoint(path)
