"""Raw STAEformer and locked native BiST controls, without training engines or labels."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from baselines.STAEformer.STAEformer import STAEformer
from ndr_research.data import canonical_digest, sha256_file
from scripts.reproduction.igstgnn_table4.bist_factory import _IMPORT_LOCK
from scripts.reproduction.igstgnn_table4.bist_ndr_pair_model import BiSTNDR
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set
from scripts.reproduction.igstgnn_table4.upstream_lock import verify_pristine_upstream_checkout


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config/reproduction/igstgnn_table4/profiles.public_material_v2.json"
DEFAULT_BIST_CHECKOUT = ROOT / ".third_party/igstgnn_table4/pristine/bist_dd94adf"
METHODS = ("staeformer", "bist", "bist_ndr", "bist_ndr_anneal", "bist_ndr_regularize")
NATIVE_CONTROL_METHODS = ("staeformer", "bist")
PLUGIN_METHODS = ("bist_ndr", "bist_ndr_anneal", "bist_ndr_regularize")
BIST_RECIPE_METHODS = ("bist",) + PLUGIN_METHODS
RECIPES = ("native", "matched-ndr")
# The NDR plugin loss weights are the original s0g values used by every verified NDR table.
PLUGIN_LOSS = {"lambda_gate": 1.0, "lambda_specialization": 0.3,
               "gate_teacher": "original_soft_distance_oracle_confidence_weighted_huber",
               "specialization": "oracle_weighted_dual_head_mae_batch_normalized_weights",
               "loss_source_sha256": None}
# Anti-overfitting plugin variants. Both keep the eval-mode forward of bist_ndr unchanged:
# anneal only reschedules the auxiliary loss weights, regularize only changes training-time
# dropout at the two residual head inputs and the weight decay of the added parameters.
PLUGIN_LOSS_ANNEAL = {
    "name": "linear_anneal_to_zero", "unit": "completed_effective_updates",
    "end_fraction_of_planned_updates": 0.5,
    "lambda_gate_initial": PLUGIN_LOSS["lambda_gate"], "lambda_gate_final": 0.0,
    "lambda_specialization_initial": PLUGIN_LOSS["lambda_specialization"], "lambda_specialization_final": 0.0,
    "planned_updates": "epochs_times_training_batches_including_retained_tail",
    "resume": "weights_follow_the_checkpointed_completed_update_count",
}
PLUGIN_REGULARIZATION = {
    "head_input_dropout": 0.2,
    "dropout_scope": "normal_and_disturbance_head_inputs_training_mode_only",
    "plugin_parameter_modules": ["disturbance_regression", "disturbance_decoder", "gate_mod"],
    "plugin_weight_decay": 0.001,
    "optimizer_override": "adam_weight_decay_by_parameter_group",
}
PLUGIN_VARIANT_LABELS = {
    "bist_ndr": "original_plugin_training_contract",
    "bist_ndr_anneal": "auxiliary_loss_weights_linearly_annealed_to_zero_by_half_of_planned_updates",
    "bist_ndr_regularize": "plugin_parameter_group_weight_decay_1e-3_and_head_input_dropout_0.2",
}
_BIST_IMPORTS: dict[str, Any] = {}
STAE_MODEL_CONFIG = {
    "in_steps": 12, "out_steps": 12, "steps_per_day": 288,
    "input_dim": 1, "output_dim": 1, "input_embedding_dim": 24,
    "tod_embedding_dim": 24, "dow_embedding_dim": 24,
    "spatial_embedding_dim": 0, "adaptive_embedding_dim": 80,
    "feed_forward_dim": 256, "num_heads": 4, "num_layers": 3,
    "dropout": 0.1, "use_mixed_proj": True,
}


def bist_profile() -> dict[str, Any]:
    profile = load_profile_set(PROFILE_PATH)["profiles"]["bist"]
    return copy.deepcopy(profile)


def _load_locked_bist(checkout: Path) -> tuple[Any, dict[str, Any]]:
    provenance = verify_pristine_upstream_checkout(source_id="bist_dd94adf", checkout=checkout)
    resolved = Path(provenance["checkout"])
    provenance["python_files_sha256"] = {
        path.relative_to(resolved).as_posix(): sha256_file(path)
        for path in sorted((resolved / "src").rglob("*.py"))
    }
    provenance["python_files_digest"] = canonical_digest(provenance["python_files_sha256"])
    with _IMPORT_LOCK:
        key = str(resolved)
        if key not in _BIST_IMPORTS:
            saved = {name: module for name, module in tuple(sys.modules.items()) if name == "src" or name.startswith("src.")}
            for name in saved:
                del sys.modules[name]
            package = ModuleType("src")
            package.__path__ = [str(resolved / "src")]
            package.__package__ = "src"
            sys.modules["src"] = package
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            try:
                module = importlib.import_module("src.models.bist")
                _BIST_IMPORTS[key] = SimpleNamespace(MLP=module.MLP, BiST=module.BiST)
            finally:
                for name in tuple(sys.modules):
                    if name == "src" or name.startswith("src."):
                        del sys.modules[name]
                sys.modules.update(saved)
                sys.dont_write_bytecode = previous
        return _BIST_IMPORTS[key], provenance


def _native_bist(node_count: int, checkout: Path) -> tuple[nn.Module, dict[str, Any]]:
    upstream, provenance = _load_locked_bist(checkout)
    effective = bist_profile()["effective"]
    architecture, inputs, adjacency = effective["model"], effective["input_contract"], effective["adjacency"]
    shallow = upstream.MLP(
        node_num=node_count, input_dim=inputs["input_dim"], output_dim=inputs["target_dim"],
        num_layer=architecture["layers"], model_dim=architecture["model_dim"],
        prompt_dim=architecture["prompt_dim"], tod_size=inputs["time_slots_per_day"],
        kernel_size=architecture["kernel_size"],
    )
    width = architecture["model_dim"] + 3 * architecture["prompt_dim"]
    arguments = {
        "extra_type": architecture["extra_type"], "same": architecture["same"],
        "hid_dim": architecture["hidden_dim"], "horizon": 12,
        "predefined_adj": int(adjacency["predefined"]), "adjs": [],
        "rp_layer": architecture["residual_propagation_layers"],
        "datadriven_adj": int(adjacency["data_driven"]), "datadriven_adj_dim": 32,
        "datadriven_adj_head": 0, "adaptive_adj": int(adjacency["adaptive"]),
        "adaptive_adj_dim": 10, "use_global_opt": False, "mrf": 1,
    }
    model = upstream.BiST(
        node_num=node_count, input_dim=inputs["input_dim"], output_dim=inputs["target_dim"],
        model_args=arguments, stmodel=shallow, dim=[width, width], core=architecture["core_dim"],
    )
    provenance.update({"profile_id": bist_profile()["profile_id"], "profile_effective": effective,
                       "profile_file_sha256": sha256_file(PROFILE_PATH),
                       "native_engine_created": False, "test_loader_created": False})
    return model, provenance


class StrongReference(nn.Module):
    """History-only indomain native model, returning predictions in raw flow units."""

    def __init__(
        self, *, method: str, node_count: int, flow_mean: float, flow_std: float,
        bist_checkout: str | Path | None = None,
    ) -> None:
        super().__init__()
        if method not in NATIVE_CONTROL_METHODS or type(node_count) is not int or node_count <= 0:
            raise ValueError("strong control requires a supported native model and positive node count")
        values = torch.tensor([flow_mean, flow_std], dtype=torch.float64)
        if not torch.isfinite(values).all() or flow_std <= 0:
            raise ValueError("normalization must contain finite train statistics and positive scale")
        self.method, self.node_count = method, node_count
        self.register_buffer("flow_mean", torch.tensor(float(flow_mean)))
        self.register_buffer("flow_std", torch.tensor(float(flow_std)))
        if method == "staeformer":
            self.native = STAEformer(num_nodes=node_count, **STAE_MODEL_CONFIG)
            self.upstream = {"source": "tracked_STAEformer_published_configuration",
                             "source_file_sha256": sha256_file(ROOT / "baselines/STAEformer/STAEformer.py"),
                             "model_config": dict(STAE_MODEL_CONFIG)}
            checkout = None
        else:
            checkout = Path(bist_checkout) if bist_checkout is not None else DEFAULT_BIST_CHECKOUT
            self.native, self.upstream = _native_bist(node_count, checkout)
            checkout = str(checkout.resolve())
        self.constructor = {"method": method, "node_count": node_count,
                            "flow_mean": float(flow_mean), "flow_std": float(flow_std),
                            "bist_checkout": checkout}
        self.research_spec = {
            "method": method, "claim_scope": "indomain_strong_control",
            "output_units": "raw_flow", "input_flow": "unique_raw_train_observation_zscore",
            "calendar": "tod_normalized_dow_raw_0_to_6" if method == "staeformer" else "native_normalized_tod_and_dow_96_tod_slots",
            "input_mask_handling": "original_v3_causally_filled_cache",
            "target_input": False, "incident_input": False,
        }

    def build_inputs(self, flow_hist: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        if flow_hist.ndim != 3 or flow_hist.shape[1:] != (self.node_count, 12):
            raise ValueError("strong control flow must have shape (B,N,12)")
        batch = flow_hist.shape[0]
        if time_feat.shape != (batch, 12, 2) or not torch.isfinite(time_feat).all():
            raise ValueError("strong control calendar must match (B,12,2)")
        if not torch.isfinite(flow_hist).all() or not ((time_feat >= 0) & (time_feat < 1)).all():
            raise FloatingPointError("strong control inputs must be finite with normalized historical calendar")
        signal = ((flow_hist - self.flow_mean) / self.flow_std).permute(0, 2, 1)
        tod = time_feat[..., 0].unsqueeze(-1).expand(batch, 12, self.node_count)
        dow = time_feat[..., 1]
        if self.method == "staeformer":
            dow = (dow * 7).round().clamp(0, 6)
        dow = dow.unsqueeze(-1).expand_as(tod)
        return torch.stack((signal, tod, dow), -1)

    def forward(self, flow_hist: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        prediction = self.native(self.build_inputs(flow_hist, time_feat))
        if prediction.shape != (flow_hist.shape[0], 12, self.node_count, 1):
            raise ValueError("native control returned an unexpected horizon shape")
        return prediction.squeeze(-1).permute(0, 2, 1) * self.flow_std + self.flow_mean

    def factory_kwargs(self) -> dict[str, Any]:
        return copy.deepcopy(self.constructor)


class BiSTNDRHeadDropout(BiSTNDR):
    """The verified pair adapter with dropout on both residual head inputs, active only in training.

    ``nn.Dropout`` returns its input unchanged in eval mode, so the eval forward equals ``BiSTNDR``
    exactly. The normal head is recomputed from the shared encoding with the native regression
    layer so that its input can carry a dropout mask; the transposed view fed to that layer has
    the strides of the tensor the native MLP convolves, so without dropout the result is the
    same convolution on the same bytes.
    """

    def __init__(self, *, native: nn.Module, residual_std: float, residual_std_by_node: np.ndarray,
                 head_input_dropout: float) -> None:
        super().__init__(native=native, residual_std=residual_std, residual_std_by_node=residual_std_by_node)
        if type(head_input_dropout) is not float or not 0.0 < head_input_dropout < 1.0:
            raise ValueError("head input dropout must be a float strictly inside (0, 1)")
        self.head_input_dropout = head_input_dropout
        self.normal_head_dropout = nn.Dropout(head_input_dropout)
        self.disturbance_head_dropout = nn.Dropout(head_input_dropout)

    def _residual_heads(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shallow, encoded, _ = self.native.stmodel(inputs)
        backcast = self.native.backcast(encoded)
        if self.native.same:
            residual_encoded = self.native.stmodel.module(shallow - backcast)
        else:
            residual_encoded = self.native.stmodel_detach.module(shallow - backcast)
        residual_encoded = self.native.residual_information_propagation(residual_encoded)
        normal = self.native.stmodel.regression_layer(
            self.normal_head_dropout(encoded).transpose(1, -1)
        ) + self.native.decoder(self.normal_head_dropout(residual_encoded)).transpose(1, -1)
        disturbance = self.disturbance_regression(
            self.disturbance_head_dropout(encoded).transpose(1, -1)
        ) + self.disturbance_decoder(self.disturbance_head_dropout(residual_encoded)).transpose(1, -1)
        return normal.squeeze(-1).permute(0, 2, 1), disturbance.squeeze(-1).permute(0, 2, 1)


class StrongReferenceNDR(nn.Module):
    """Locked native BiST backbone carrying the original NDR plugin, in raw flow units.

    The wrapped module is the very ``BiSTNDR`` adapter whose 18-task official-protocol
    pairing was analyzed on 2026-10-20: shared BiST encoding, duplicated normal and
    disturbance heads, the original horizon gate, and anchors on the train-only weekly
    climatology and the last observation. Only the training contract differs here.
    """

    def __init__(
        self, *, method: str, node_count: int, flow_mean: float, flow_std: float,
        residual_std: float, residual_std_by_node: Any,
        bist_checkout: str | Path | None = None,
    ) -> None:
        super().__init__()
        if method not in PLUGIN_METHODS or type(node_count) is not int or node_count <= 0:
            raise ValueError("the NDR plugin control requires a bist_ndr plugin method and a positive node count")
        values = torch.tensor([flow_mean, flow_std, residual_std], dtype=torch.float64)
        if not torch.isfinite(values).all() or flow_std <= 0 or residual_std <= 0:
            raise ValueError("plugin normalization must contain finite train statistics and positive scales")
        scales = np.asarray(residual_std_by_node, dtype=np.float32)
        if scales.shape != (node_count,) or not np.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("plugin node residual scales must be positive with shape (N,)")
        self.method, self.node_count = method, node_count
        self.register_buffer("flow_mean", torch.tensor(float(flow_mean)))
        self.register_buffer("flow_std", torch.tensor(float(flow_std)))
        checkout = Path(bist_checkout) if bist_checkout is not None else DEFAULT_BIST_CHECKOUT
        native, upstream = _native_bist(node_count, checkout)
        if method == "bist_ndr_regularize":
            self.plugin = BiSTNDRHeadDropout(native=native, residual_std=float(residual_std), residual_std_by_node=scales,
                                             head_input_dropout=PLUGIN_REGULARIZATION["head_input_dropout"])
        else:
            self.plugin = BiSTNDR(native=native, residual_std=float(residual_std), residual_std_by_node=scales)
        plugin_parameters = sum(p.numel() for p in self.plugin.parameters()) - sum(p.numel() for p in native.parameters())
        variant: dict[str, Any] = {"variant": method, "training_variant": PLUGIN_VARIANT_LABELS[method],
                                   "eval_forward": "identical_to_bist_ndr"}
        if method == "bist_ndr_anneal":
            variant["loss_schedule"] = dict(PLUGIN_LOSS_ANNEAL)
        if method == "bist_ndr_regularize":
            variant.update({"head_input_dropout": PLUGIN_REGULARIZATION["head_input_dropout"],
                            "head_dropout_adapter_class": f"{BiSTNDRHeadDropout.__module__}.{BiSTNDRHeadDropout.__name__}",
                            "plugin_weight_decay": PLUGIN_REGULARIZATION["plugin_weight_decay"]})
        self.upstream = {**upstream, "plugin": {
            "adapter_class": f"{BiSTNDR.__module__}.{BiSTNDR.__name__}",
            "adapter_source_sha256": sha256_file(ROOT / "scripts/reproduction/igstgnn_table4/bist_ndr_pair_model.py"),
            "loss_source_sha256": sha256_file(ROOT / "scripts/train_staeformer_deseason.py"),
            "head_initialization": "identical_copy_from_bist_native_heads",
            "normal_anchor": "training_only_weekly_median", "disturbance_anchor": "last_observation_persistence",
            "gate": "original_horizon_monotone_36_parameters", "added_parameters": int(plugin_parameters),
            **variant,
        }}
        if len(self.parameter_groups()["ndr_plugin"]) == 0:
            raise ValueError("the NDR plugin must add parameters beyond the native BiST")
        self.constructor = {"method": method, "node_count": node_count,
                            "flow_mean": float(flow_mean), "flow_std": float(flow_std),
                            "residual_std": float(residual_std), "residual_std_by_node": scales.tolist(),
                            "bist_checkout": str(checkout.resolve())}
        self.research_spec = {
            "method": method, "claim_scope": "indomain_strong_control_with_ndr_plugin",
            "output_units": "raw_flow", "input_flow": "train_only_weekly_climatology_residual_over_global_residual_std",
            "calendar": "native_normalized_tod_and_dow_96_tod_slots",
            "input_mask_handling": "original_v3_causally_filled_cache",
            "anchors": ("train_only_weekly_median", "last_observation_persistence"),
            "plugin_variant": method,
            "target_input": False, "incident_input": False,
        }

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Partition every parameter into the native BiST group and the added plugin group."""
        native_ids = {id(parameter) for parameter in self.plugin.native.parameters()}
        allowed = tuple(f"plugin.{name}." for name in PLUGIN_REGULARIZATION["plugin_parameter_modules"])
        groups: dict[str, list[nn.Parameter]] = {"native_bist": [], "ndr_plugin": []}
        for name, parameter in self.named_parameters():
            if id(parameter) in native_ids:
                groups["native_bist"].append(parameter)
            elif name.startswith(allowed):
                groups["ndr_plugin"].append(parameter)
            else:
                raise ValueError(f"unexpected non-native plugin parameter {name}")
        return groups

    def forward(self, flow_hist: torch.Tensor, time_feat: torch.Tensor,
                x_baseline: torch.Tensor, y_baseline: torch.Tensor) -> torch.Tensor:
        if flow_hist.ndim != 3 or flow_hist.shape[1:] != (self.node_count, 12):
            raise ValueError("plugin control flow must have shape (B,N,12)")
        batch = flow_hist.shape[0]
        if time_feat.shape != (batch, 12, 2) or not torch.isfinite(time_feat).all():
            raise ValueError("plugin control calendar must match (B,12,2)")
        if x_baseline.shape != flow_hist.shape or y_baseline.shape != flow_hist.shape:
            raise ValueError("plugin anchors must match the (B,N,12) history and horizon")
        if not torch.isfinite(flow_hist).all() or not ((time_feat >= 0) & (time_feat < 1)).all():
            raise FloatingPointError("plugin control inputs must be finite with normalized historical calendar")
        if not torch.isfinite(x_baseline).all() or not torch.isfinite(y_baseline).all():
            raise FloatingPointError("plugin anchors must be finite")
        prediction = self.plugin(flow_hist, x_baseline, y_baseline, time_feat)
        if prediction.shape != flow_hist.shape:
            raise ValueError("plugin control returned an unexpected horizon shape")
        return prediction

    def factory_kwargs(self) -> dict[str, Any]:
        return copy.deepcopy(self.constructor)


def make_strong_reference(*, device: str | torch.device = "cpu", **kwargs: Any) -> nn.Module:
    if kwargs.get("method") in PLUGIN_METHODS:
        return StrongReferenceNDR(**kwargs).to(device)
    return StrongReference(**kwargs).to(device)


def reference_recipe(method: str, recipe: str, *, region: str, epochs: int,
                     batch_size: int | None = None, microbatch_size: int | None = None) -> dict[str, Any]:
    if method not in METHODS or recipe not in RECIPES or epochs not in (30, 100):
        raise ValueError("strong controls require native/matched-ndr with an explicit 30/100 epoch budget")
    if region not in ("Alameda", "ContraCosta", "Orange"):
        raise ValueError("unknown strong-control region")
    original = bist_profile()["effective"] if method in BIST_RECIPE_METHODS else None
    if recipe == "native" and method in BIST_RECIPE_METHODS:
        optimizer = copy.deepcopy(original["optimizer"])
        native_batch = original["batch_size_by_region"]["Contra_Costa" if region == "ContraCosta" else region]
        scheduler_unit = "completed_epoch"
    else:
        optimizer = {
            "name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
            "epsilon": 1e-8, "gradient_clip": 5.0,
            "scheduler": {"name": "CosineAnnealingLR", "eta_min": 0.00001,
                          "t_max": "effective_epochs_times_training_batches"},
        }
        native_batch = 16
        scheduler_unit = "completed_effective_update"
    effective_batch = native_batch if batch_size is None else batch_size
    micro = effective_batch if microbatch_size is None else microbatch_size
    if type(effective_batch) is not int or type(micro) is not int or not 0 < micro <= effective_batch:
        raise ValueError("microbatch must be within the positive effective batch")
    plugin_loss = None
    if method in PLUGIN_METHODS:
        plugin_loss = {**PLUGIN_LOSS, "loss_source_sha256": sha256_file(ROOT / "scripts/train_staeformer_deseason.py")}
    if original and recipe == "native":
        authority = "locked_bist_public_material_profile" if method == "bist" else "locked_bist_public_material_profile_shared_by_bist_ndr_plugin"
        if method == "bist_ndr_regularize":
            authority += "_with_plugin_group_weight_decay_override"
    else:
        authority = "matched_ndr_recipe" if recipe == "matched-ndr" else "existing_xtraffic_raw_staeformer_recipe"
    value = {
        "method": method, "recipe": recipe, "requested_epoch_budget": epochs,
        "budget_label": "fixed_30_epoch_development" if epochs == 30 else "native_100_epoch_budget_supplement" if recipe == "native" else "matched_100_epoch_budget_supplement",
        "optimizer": optimizer, "scheduler_step_unit": scheduler_unit,
        "batch_size": effective_batch, "microbatch_size": micro,
        "profile_batch_size": native_batch, "batch_size_override": effective_batch != native_batch,
        "batch_normalization_scope": "microbatch_override" if method in BIST_RECIPE_METHODS and micro < effective_batch else "effective_batch" if method in BIST_RECIPE_METHODS else "no_batch_normalization",
        "native_profile_batch_size": original["batch_size_by_region"]["Contra_Costa" if region == "ContraCosta" else region] if original else 16,
        "native_epoch_cap": original["max_epochs"] if original else None,
        "native_patience": original["patience"] if original else None,
        "early_stopping": False, "partial_batch_train": "retain", "partial_batch_validation": "retain",
        "native_tail_override": "native_drop_to_v3_retain" if original else None,
        "native_seed_override": "fixed_recorded_research_seed" if original else None,
        "native_profile_sha256": sha256_file(PROFILE_PATH) if original else None,
        "optimizer_authority": authority,
        "staeformer_recipe_source_sha256": sha256_file(ROOT / "scripts/train_staeformer_xtraffic.py") if method == "staeformer" else None,
        "primary_loss": "raw_flow_masked_mae_full_effective_valid_point_denominator",
        "plugin_loss": plugin_loss,
        "comparison_role": {"staeformer": "raw_backbone_control", "bist": "backbone_control", "bist_ndr": "plugin_candidate",
                            "bist_ndr_anneal": "plugin_candidate_loss_annealed",
                            "bist_ndr_regularize": "plugin_candidate_regularized"}[method],
    }
    # The three original methods keep their recipe dictionaries byte for byte; only the two
    # anti-overfitting variants carry the extra schedule and regularization declarations.
    if method == "bist_ndr_anneal":
        value["plugin_loss"] = {**plugin_loss, "schedule": dict(PLUGIN_LOSS_ANNEAL)}
        value["plugin_variant"] = PLUGIN_VARIANT_LABELS[method]
    if method == "bist_ndr_regularize":
        value["plugin_loss"] = {**plugin_loss, "schedule": {"name": "constant"}}
        value["plugin_regularization"] = {
            **PLUGIN_REGULARIZATION,
            "parameter_groups": {"native_bist": {"weight_decay": optimizer["weight_decay"]},
                                 "ndr_plugin": {"weight_decay": PLUGIN_REGULARIZATION["plugin_weight_decay"]}},
        }
        value["plugin_variant"] = PLUGIN_VARIANT_LABELS[method]
    return value
