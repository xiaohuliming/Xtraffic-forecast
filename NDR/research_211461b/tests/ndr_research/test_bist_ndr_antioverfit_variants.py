"""CPU checks for the anti-overfitting BiST-NDR plugin variants: anneal and regularize."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import pytest
import torch

from ndr_research.strong_references import (
    METHODS,
    PLUGIN_LOSS,
    PLUGIN_LOSS_ANNEAL,
    PLUGIN_METHODS,
    PLUGIN_REGULARIZATION,
    BiSTNDRHeadDropout,
    StrongReferenceNDR,
    _native_bist,
    make_strong_reference,
    reference_recipe,
)
from scripts import train_ndr_strong_reference as trainer
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_model import BiSTNDR
from tests.ndr_research.test_bist_ndr_strong_reference import deterministic_training_mode, tiny_data


@pytest.fixture
def bist_checkout():
    candidates = [Path(os.environ.get("XTRAFFIC_BIST_TEST_CHECKOUT", "/nonexistent")),
                  Path(__file__).resolve().parents[2] / ".third_party/igstgnn_table4/pristine/bist_dd94adf",
                  Path("/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf")]
    for path in candidates:
        if (path / "src/models/bist.py").is_file():
            return path
    pytest.skip("locked pristine BiST checkout is not installed for this environment")


def variant_model(checkout, method, seed=7):
    torch.manual_seed(seed)
    return make_strong_reference(method=method, node_count=2, flow_mean=40, flow_std=10, residual_std=5.0,
                                 residual_std_by_node=[5.0, 6.0], bist_checkout=checkout)


def config_for(data, method, epochs=1):
    value = reference_recipe(method, "native", region="Alameda", epochs=30, batch_size=2)
    return {"epochs": epochs, "batch_size": 2, "microbatch_size": 2, "training_recipe": value,
            "max_train_batches": None, "max_val_batches": None, "claim_scope": "smoke_only",
            "source_snapshot": trainer.source_snapshot(), "runtime": trainer.runtime_identity(torch.device("cpu")),
            "training_process_id": os.getpid()}


def next_train_batch(data):
    """Advance the tiny training loader, restarting its epoch when exhausted."""
    try:
        batch = data.train.next_batch()
    except StopIteration:
        data.train.start_epoch()
        batch = data.train.next_batch()
    return trainer.batch_tensors(batch, data, torch.device("cpu"))


def head_dropout_only_training_mode(model):
    """Training mode with every native dropout and batch-norm silenced, keeping the two head dropouts."""
    deterministic_training_mode(model)
    model.plugin.normal_head_dropout.p = PLUGIN_REGULARIZATION["head_input_dropout"]
    model.plugin.disturbance_head_dropout.p = PLUGIN_REGULARIZATION["head_input_dropout"]


def test_variant_methods_build_the_plugin_wrapper_with_the_same_parameters_as_bist_ndr(bist_checkout):
    assert set(PLUGIN_METHODS) == {"bist_ndr", "bist_ndr_anneal", "bist_ndr_regularize"} and set(PLUGIN_METHODS) <= set(METHODS)
    base = variant_model(bist_checkout, "bist_ndr")
    anneal = variant_model(bist_checkout, "bist_ndr_anneal")
    regularize = variant_model(bist_checkout, "bist_ndr_regularize")
    for model, method in ((anneal, "bist_ndr_anneal"), (regularize, "bist_ndr_regularize")):
        assert isinstance(model, StrongReferenceNDR) and model.method == method and trainer.is_plugin_model(model)
        assert model.upstream["plugin"]["variant"] == method and model.upstream["plugin"]["eval_forward"] == "identical_to_bist_ndr"
        assert model.upstream["plugin"]["added_parameters"] == base.upstream["plugin"]["added_parameters"] > 0
        assert list(model.state_dict()) == list(base.state_dict())
        assert model.factory_kwargs()["method"] == method and model.research_spec["plugin_variant"] == method
        with pytest.raises(ValueError, match="identity mismatch"):
            trainer._require_model_identity(model, {"model_upstream": base.upstream})
    assert type(anneal.plugin) is BiSTNDR and anneal.upstream["plugin"]["loss_schedule"] == PLUGIN_LOSS_ANNEAL
    assert isinstance(regularize.plugin, BiSTNDRHeadDropout) and regularize.upstream["plugin"]["head_input_dropout"] == 0.2
    assert base.upstream["plugin"]["variant"] == "bist_ndr" and "loss_schedule" not in base.upstream["plugin"]
    with pytest.raises(ValueError, match="plugin method"):
        StrongReferenceNDR(method="bist", node_count=2, flow_mean=40, flow_std=10, residual_std=5.0,
                           residual_std_by_node=[5.0, 6.0], bist_checkout=bist_checkout)


def test_anneal_weights_follow_the_linear_schedule_to_zero_at_half_of_the_planned_updates():
    recipe = reference_recipe("bist_ndr_anneal", "native", region="ContraCosta", epochs=100)
    expected = {0: (1.0, 0.3), 25: (0.75, 0.225), 50: (0.5, 0.15), 75: (0.25, 0.075), 100: (0.0, 0.0), 150: (0.0, 0.0)}
    for completed, (gate, specialization) in expected.items():
        weights = trainer.plugin_loss_weights(recipe, completed_updates=completed, planned_updates=200)
        assert weights["schedule"] == "linear_anneal_to_zero" and weights["anneal_end_updates"] == 100
        assert weights["planned_total_updates"] == 200 and weights["completed_updates"] == completed
        assert weights["lambda_gate"] == pytest.approx(gate, abs=1e-12)
        assert weights["lambda_specialization"] == pytest.approx(specialization, abs=1e-12)
        assert weights["anneal_fraction"] == pytest.approx(min(1.0, completed / 100))
    assert trainer.plugin_loss_weights(recipe, completed_updates=0, planned_updates=201)["anneal_end_updates"] == 101
    assert trainer.plugin_loss_weights(recipe, completed_updates=100, planned_updates=200)["lambda_gate"] == 0.0
    # ContraCosta and Alameda planned totals under the native batch 64 contract with retained tails.
    for batches, end in ((257, 12850), (501, 25050)):
        assert trainer.plugin_loss_weights(recipe, completed_updates=0, planned_updates=100 * batches)["anneal_end_updates"] == end
    for bad in ({"completed_updates": None, "planned_updates": 10}, {"completed_updates": 3, "planned_updates": None},
                {"completed_updates": -1, "planned_updates": 10}, {"completed_updates": 2.0, "planned_updates": 10}):
        with pytest.raises(ValueError, match="completed and planned"):
            trainer.plugin_loss_weights(recipe, **bad)
    constant = trainer.plugin_loss_weights(reference_recipe("bist_ndr", "native", region="ContraCosta", epochs=100))
    assert constant == {"schedule": "constant", "completed_updates": None, "lambda_gate": 1.0, "lambda_specialization": 0.3}
    assert trainer.plugin_loss_weights(reference_recipe("bist_ndr", "native", region="ContraCosta", epochs=100),
                                       completed_updates=999, planned_updates=10)["lambda_gate"] == 1.0


@pytest.mark.parametrize("epochs", [30, 100])
def test_variant_recipes_record_their_fields_and_keep_the_locked_native_optimizer(epochs):
    control = reference_recipe("bist", "native", region="ContraCosta", epochs=epochs)
    base = reference_recipe("bist_ndr", "native", region="ContraCosta", epochs=epochs)
    anneal = reference_recipe("bist_ndr_anneal", "native", region="ContraCosta", epochs=epochs)
    regularize = reference_recipe("bist_ndr_regularize", "native", region="ContraCosta", epochs=epochs)
    for value in (anneal, regularize):
        assert value["optimizer"] == control["optimizer"] == base["optimizer"]
        assert value["batch_size"] == value["microbatch_size"] == 64 and value["scheduler_step_unit"] == "completed_epoch"
        assert not value["early_stopping"] and value["requested_epoch_budget"] == epochs
        assert value["plugin_loss"]["lambda_gate"] == 1.0 and value["plugin_loss"]["lambda_specialization"] == 0.3
        assert value["plugin_loss"]["loss_source_sha256"] == base["plugin_loss"]["loss_source_sha256"]
    assert "schedule" not in base["plugin_loss"] and "plugin_variant" not in base and "plugin_regularization" not in base
    assert anneal["plugin_loss"]["schedule"] == PLUGIN_LOSS_ANNEAL
    assert anneal["plugin_loss"]["schedule"]["end_fraction_of_planned_updates"] == 0.5
    assert anneal["plugin_loss"]["schedule"]["lambda_gate_initial"] == PLUGIN_LOSS["lambda_gate"] == 1.0
    assert anneal["plugin_loss"]["schedule"]["lambda_specialization_initial"] == PLUGIN_LOSS["lambda_specialization"] == 0.3
    assert anneal["plugin_loss"]["schedule"]["lambda_gate_final"] == anneal["plugin_loss"]["schedule"]["lambda_specialization_final"] == 0.0
    assert "plugin_regularization" not in anneal and anneal["comparison_role"] == "plugin_candidate_loss_annealed"
    assert anneal["optimizer_authority"] == base["optimizer_authority"]
    assert regularize["plugin_loss"]["schedule"] == {"name": "constant"}
    assert regularize["plugin_regularization"]["head_input_dropout"] == 0.2
    assert regularize["plugin_regularization"]["parameter_groups"] == {"native_bist": {"weight_decay": 1e-5}, "ndr_plugin": {"weight_decay": 1e-3}}
    assert regularize["plugin_regularization"]["plugin_parameter_modules"] == ["disturbance_regression", "disturbance_decoder", "gate_mod"]
    assert regularize["comparison_role"] == "plugin_candidate_regularized"
    assert regularize["optimizer_authority"].endswith("_with_plugin_group_weight_decay_override")
    assert isinstance(anneal["plugin_variant"], str) and isinstance(regularize["plugin_variant"], str)


def test_anneal_objective_uses_the_scheduled_weights_and_equals_the_original_contract_at_zero_updates(bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    base = variant_model(bist_checkout, "bist_ndr")
    anneal = variant_model(bist_checkout, "bist_ndr_anneal")
    anneal.load_state_dict(base.state_dict(), strict=True)
    base_config, anneal_config = config_for(data, "bist_ndr"), config_for(data, "bist_ndr_anneal")
    base_optimizer, base_scheduler = trainer.create_reference_optimizer(base, base_config, 4)
    anneal_optimizer, anneal_scheduler = trainer.create_reference_optimizer(anneal, anneal_config, 4)
    data.train.start_epoch()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    deterministic_training_mode(base)
    deterministic_training_mode(anneal)
    with pytest.raises(ValueError, match="completed and planned"):
        trainer.optimize_reference(anneal, anneal_optimizer, anneal_scheduler, tensors, anneal_config)
    expected = trainer.optimize_reference(base, base_optimizer, base_scheduler, tensors, base_config)
    observed = trainer.optimize_reference(anneal, anneal_optimizer, anneal_scheduler, tensors, anneal_config,
                                          completed_updates=0, planned_updates=20)
    assert expected["plugin_loss_weights"]["schedule"] == "constant"
    assert observed["plugin_loss_weights"]["lambda_gate"] == 1.0 and observed["plugin_loss_weights"]["lambda_specialization"] == 0.3
    assert observed["objective"] == pytest.approx(expected["objective"], rel=1e-9)
    for key in ("mae", "gate", "spec"):
        assert observed["loss_parts"][key] == pytest.approx(expected["loss_parts"][key], rel=1e-9)
    for left, right in zip(base.parameters(), anneal.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Mid schedule and after the end, the objective is the primary MAE plus the scaled auxiliary parts.
    for completed, gate, specialization in ((5, 0.5, 0.15), (10, 0.0, 0.0), (17, 0.0, 0.0)):
        tensors = next_train_batch(data)
        step = trainer.optimize_reference(anneal, anneal_optimizer, anneal_scheduler, tensors, anneal_config,
                                          completed_updates=completed, planned_updates=20)
        weights = step["plugin_loss_weights"]
        assert weights["lambda_gate"] == pytest.approx(gate) and weights["lambda_specialization"] == pytest.approx(specialization)
        assert weights["anneal_end_updates"] == 10 and weights["completed_updates"] == completed
        parts = step["loss_parts"]
        assert step["objective"] == pytest.approx(parts["mae"] + gate * parts["gate"] + specialization * parts["spec"], rel=1e-6)
        assert step["updated"] and parts["gate"] > 0 and parts["spec"] > 0


def test_anneal_checkpoint_restores_the_schedule_from_the_completed_update_count(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = variant_model(bist_checkout, "bist_ndr_anneal")
    config = config_for(data, "bist_ndr_anneal", epochs=2)
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    planned = config["epochs"] * data.train.num_batch
    assert planned == 4
    data.train.start_epoch()
    progress = {"epoch": 0, "stage": "train", "updates": 0, "plugin_loss_schedule": None, "planned_total_updates": planned}
    for _ in range(2):
        tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
        step = trainer.optimize_reference(model, optimizer, scheduler, tensors, config,
                                          completed_updates=progress["updates"], planned_updates=planned)
        progress["updates"] += int(step["updated"])
        progress["plugin_loss_schedule"] = step["plugin_loss_weights"]
    assert progress["updates"] == 2 and progress["plugin_loss_schedule"]["completed_updates"] == 1
    assert progress["plugin_loss_schedule"]["lambda_gate"] == pytest.approx(0.5)
    assert progress["plugin_loss_schedule"]["anneal_end_updates"] == 2
    path = tmp_path / "last.pt"
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)
    checkpoint = trainer.load_reference_checkpoint(path)
    assert checkpoint["model_factory"]["method"] == "bist_ndr_anneal"
    assert checkpoint["config"]["training_recipe"]["plugin_loss"]["schedule"]["name"] == "linear_anneal_to_zero"
    assert checkpoint["progress"]["plugin_loss_schedule"]["lambda_specialization"] == pytest.approx(0.15)
    data.train.start_epoch()
    original_next = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    expected = trainer.optimize_reference(model, optimizer, scheduler, original_next, config,
                                          completed_updates=progress["updates"], planned_updates=planned)
    restored_data = tiny_data()
    restored = make_strong_reference(**(checkpoint["model_factory"] | {"bist_checkout": bist_checkout, "device": "cpu"}))
    trainer._require_model_identity(restored, checkpoint)
    restored_optimizer, restored_scheduler = trainer.create_reference_optimizer(restored, config, restored_data.train.num_batch)
    restored_progress = trainer.restore_reference(checkpoint, restored, restored_optimizer, restored_scheduler, restored_data, config)
    assert restored_progress["updates"] == 2 and restored_progress["plugin_loss_schedule"] == progress["plugin_loss_schedule"]
    restored_data.train.start_epoch()
    restored_next = trainer.batch_tensors(restored_data.train.next_batch(), restored_data, torch.device("cpu"))
    observed = trainer.optimize_reference(restored, restored_optimizer, restored_scheduler, restored_next, config,
                                          completed_updates=restored_progress["updates"], planned_updates=restored_progress["planned_total_updates"])
    assert observed["plugin_loss_weights"] == expected["plugin_loss_weights"]
    assert observed["plugin_loss_weights"]["lambda_gate"] == 0.0 and observed["plugin_loss_weights"]["completed_updates"] == 2
    assert observed["objective"] == pytest.approx(expected["objective"], rel=1e-9)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_anneal_training_loop_records_the_weights_per_epoch_and_in_the_summary(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = variant_model(bist_checkout, "bist_ndr_anneal")
    config = config_for(data, "bist_ndr_anneal", epochs=2)
    output = tmp_path / "anneal"
    output.mkdir()
    result = trainer.train_reference(model=model, data=data, device=torch.device("cpu"), config=config, output=output)
    assert result["status"] == "training_completed_validation_pending_independent_verification"
    assert result["updates"] == 4 and result["planned_total_updates"] == 4
    assert result["plugin_loss_schedule_last_update"]["lambda_gate"] == 0.0
    history = trainer.load_reference_checkpoint(output / "last.pt")["progress"]["history"]
    weights = [row["plugin_loss_weights_last_update"] for row in history]
    assert [row["lambda_gate"] for row in weights] == [pytest.approx(0.5), pytest.approx(0.0)]
    assert [row["completed_updates"] for row in weights] == [1, 3]
    assert all(row["anneal_end_updates"] == 2 and row["planned_total_updates"] == 4 for row in weights)
    calibration = trainer.calibrate_reference(model=variant_model(bist_checkout, "bist_ndr_anneal"), data=tiny_data(), device=torch.device("cpu"),
                                              config=config_for(tiny_data(), "bist_ndr_anneal"), warmup_batches=1, timed_batches=2, output=tmp_path)
    assert calibration["status"] == "calibrated_cpu_engineering_only" and calibration["cuda_budget_eligible"] is False


def test_regularize_parameter_groups_split_native_and_plugin_weight_decay(bist_checkout):
    data = tiny_data()
    model = variant_model(bist_checkout, "bist_ndr_regularize")
    groups = model.parameter_groups()
    native_ids = {id(parameter) for parameter in model.plugin.native.parameters()}
    plugin_ids = {id(parameter) for parameter in model.parameters()} - native_ids
    assert {id(parameter) for parameter in groups["native_bist"]} == native_ids
    assert {id(parameter) for parameter in groups["ndr_plugin"]} == plugin_ids
    assert sum(parameter.numel() for parameter in groups["ndr_plugin"]) == model.upstream["plugin"]["added_parameters"]
    plugin_names = [name for name, parameter in model.named_parameters() if id(parameter) in plugin_ids]
    assert plugin_names and all(name.startswith(("plugin.disturbance_regression.", "plugin.disturbance_decoder.", "plugin.gate_mod.")) for name in plugin_names)
    assert {name.split(".")[1] for name in plugin_names} == {"disturbance_regression", "disturbance_decoder", "gate_mod"}
    optimizer, scheduler = trainer.create_reference_optimizer(model, config_for(data, "bist_ndr_regularize"), data.train.num_batch)
    assert [group["name"] for group in optimizer.param_groups] == ["native_bist", "ndr_plugin"]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-5, 1e-3]
    assert all(group["lr"] == 0.002 and group["eps"] == 1e-8 for group in optimizer.param_groups)
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == native_ids
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == plugin_ids
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR) and scheduler.milestones == {30: 1, 60: 1, 80: 1, 100: 1}
    for method in ("bist_ndr", "bist_ndr_anneal"):
        single, _ = trainer.create_reference_optimizer(variant_model(bist_checkout, method), config_for(data, method), data.train.num_batch)
        assert len(single.param_groups) == 1 and single.param_groups[0]["weight_decay"] == 1e-5
    broken = config_for(data, "bist_ndr_regularize")
    broken["training_recipe"] = copy.deepcopy(broken["training_recipe"])
    broken["training_recipe"]["plugin_regularization"]["parameter_groups"]["native_bist"]["weight_decay"] = 1e-3
    with pytest.raises(ValueError, match="recipe weight decay"):
        trainer.create_reference_optimizer(model, broken, data.train.num_batch)


def test_regularize_dropout_only_acts_in_training_and_the_eval_forward_equals_bist_ndr_exactly(bist_checkout):
    model = variant_model(bist_checkout, "bist_ndr_regularize")
    assert isinstance(model.plugin.normal_head_dropout, torch.nn.Dropout) and model.plugin.normal_head_dropout.p == 0.2
    assert isinstance(model.plugin.disturbance_head_dropout, torch.nn.Dropout) and model.plugin.disturbance_head_dropout.p == 0.2
    direct = BiSTNDR(native=_native_bist(2, bist_checkout)[0], residual_std=5.0,
                     residual_std_by_node=model.plugin.sd_node.numpy()).eval()
    direct.load_state_dict(model.plugin.state_dict(), strict=True)
    data = tiny_data()
    tensors = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        expected = direct(tensors["flow_hist"], tensors["x_baseline"], tensors["y_baseline"], tensors["time_feat"])
        first = trainer.strong_forward(model, tensors)
        second = trainer.strong_forward(model, tensors)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    assert not model.plugin.normal_head_dropout.training and not model.plugin.disturbance_head_dropout.training
    head_dropout_only_training_mode(model)
    assert model.plugin.normal_head_dropout.training and model.plugin.disturbance_head_dropout.training
    with torch.no_grad():
        torch.manual_seed(1)
        masked_a = trainer.strong_forward(model, tensors)
        torch.manual_seed(2)
        masked_b = trainer.strong_forward(model, tensors)
    assert not torch.equal(masked_a, expected) and not torch.equal(masked_b, expected) and not torch.equal(masked_a, masked_b)
    model.plugin.normal_head_dropout.p = 0.0
    model.plugin.disturbance_head_dropout.p = 0.0
    with torch.no_grad():
        unmasked = trainer.strong_forward(model, tensors)
    torch.testing.assert_close(unmasked, expected, rtol=0, atol=0)
    # The anneal variant shares the exact adapter, so its eval forward is the same bytes too.
    anneal = variant_model(bist_checkout, "bist_ndr_anneal").eval()
    anneal.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(trainer.strong_forward(anneal, tensors), expected, rtol=0, atol=0)


def test_regularize_checkpoint_restores_the_two_group_optimizer_and_continues_exactly(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = variant_model(bist_checkout, "bist_ndr_regularize")
    config = config_for(data, "bist_ndr_regularize")
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    data.train.start_epoch()
    first = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    step = trainer.optimize_reference(model, optimizer, scheduler, first, config, completed_updates=0, planned_updates=4)
    assert step["updated"] and step["plugin_loss_weights"]["schedule"] == "constant"
    assert all(parameter.grad is not None for parameter in model.parameter_groups()["ndr_plugin"])
    path = tmp_path / "last.pt"
    progress = {"epoch": 0, "stage": "train", "updates": 1}
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config, progress=progress)
    checkpoint = trainer.load_reference_checkpoint(path)
    assert checkpoint["model_factory"]["method"] == "bist_ndr_regularize"
    assert [group["weight_decay"] for group in checkpoint["optimizer"]["param_groups"]] == [1e-5, 1e-3]
    next_original = trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu"))
    trainer.optimize_reference(model, optimizer, scheduler, next_original, config, completed_updates=1, planned_updates=4)
    expected = copy.deepcopy(model.state_dict())
    restored_data = tiny_data()
    restored = make_strong_reference(**(checkpoint["model_factory"] | {"bist_checkout": bist_checkout, "device": "cpu"}))
    assert isinstance(restored.plugin, BiSTNDRHeadDropout)
    trainer._require_model_identity(restored, checkpoint)
    restored_optimizer, restored_scheduler = trainer.create_reference_optimizer(restored, config, restored_data.train.num_batch)
    assert trainer.restore_reference(checkpoint, restored, restored_optimizer, restored_scheduler, restored_data, config) == progress
    assert [group["weight_decay"] for group in restored_optimizer.param_groups] == [1e-5, 1e-3]
    next_restored = trainer.batch_tensors(restored_data.train.next_batch(), restored_data, torch.device("cpu"))
    trainer.optimize_reference(restored, restored_optimizer, restored_scheduler, next_restored, config, completed_updates=1, planned_updates=4)
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_independent_verification_reloads_a_regularize_checkpoint_in_eval_mode(tmp_path, bist_checkout):
    trainer.configure_runtime("cpu", 42, 1)
    data = tiny_data()
    model = variant_model(bist_checkout, "bist_ndr_regularize")
    config = config_for(data, "bist_ndr_regularize")
    config["training_process_id"] = os.getpid() + 100000
    config["data_identity"] = data.identity
    optimizer, scheduler = trainer.create_reference_optimizer(model, config, data.train.num_batch)
    data.train.start_epoch()
    trainer.optimize_reference(model, optimizer, scheduler, trainer.batch_tensors(data.train.next_batch(), data, torch.device("cpu")), config,
                               completed_updates=0, planned_updates=4)
    validation = trainer.evaluate_reference(model, data, torch.device("cpu"), microbatch_size=2)
    path = tmp_path / "best.pt"
    trainer.save_reference_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler, data=data, config=config,
                                      progress={"epoch": 1, "stage": "start", "updates": 1}, selection_metric=validation)
    result = trainer.verify_reference(checkpoint_path=path, data=tiny_data(), device=torch.device("cpu"), output=tmp_path / "verify",
                                      runtime=trainer.runtime_identity(torch.device("cpu")), bist_checkout=bist_checkout)
    assert result["status"] == "verified_smoke_only" and result["validation"]["mae"] == validation["mae"]
    assert result["verification_process_id"] == os.getpid() != config["training_process_id"]


def test_cli_accepts_both_variant_methods():
    for method in ("bist_ndr_anneal", "bist_ndr_regularize"):
        args = trainer.argument_parser().parse_args(["--method", method, "--recipe", "native", "--epochs", "100"])
        assert args.method == method and args.epochs == 100
        trainer.validate_arguments(args)
    assert math.ceil(0.5 * 25700) == 12850 and math.ceil(0.5 * 50100) == 25050
