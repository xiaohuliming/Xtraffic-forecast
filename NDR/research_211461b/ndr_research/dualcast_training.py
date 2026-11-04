"""Locked author auxiliary equations and explicitly named v3 compatibility recipes."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ndr_research.data import array_digest, canonical_digest, sha256_file
from ndr_research.dualcast_adapter import (
    MODEL_RELATIVE_PATH, MODEL_SHA256, UPSTREAM_COMMIT, DualCastSReference,
    load_reference_class,
)


AUTHOR_FILES = {
    MODEL_RELATIVE_PATH: MODEL_SHA256,
    "DualCastS/model/train.py": "f288756a394eeb9774ffb1f1fa3f05b8106b82e7eb70b2a741eef9a0c25798a4",
    "DualCastS/main.py": "89fedc9c081457e2a277d0f9702ed38759d15490cfa944044ce5b01ee3e21d9e",
    "DualCastS/utils/utils_.py": "8ebe772f78ea3c06204b74768eab06fb9df62fdc19ffaa2a71b20bd56529d2f1",
}
RECIPES = ("v3_masked_native_equations", "v3_masked_defined_aux")
_AUTHOR_AUX: dict[str, Any] = {}


class UndefinedAuthorAuxiliary(FloatingPointError):
    """Native reciprocal/DBI equations are mathematically undefined for this batch."""


def locked_author_identity(checkout: Path | str) -> dict[str, Any]:
    checkout = Path(checkout).resolve()
    load_reference_class(checkout)
    status = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain=v1", "--untracked-files=all"], text=True, timeout=20)
    if status.strip():
        raise ValueError("DualCast author checkout must remain clean")
    for relative, expected in AUTHOR_FILES.items():
        path = checkout / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"locked DualCast author bytes changed: {relative}")
    return {"checkout": str(checkout), "upstream_commit": UPSTREAM_COMMIT,
            "author_files_sha256": dict(AUTHOR_FILES),
            "author_files_digest": canonical_digest(AUTHOR_FILES), "git_dirty": False,
            "license_file_present": (checkout / "LICENSE").exists(),
            "author_source_vendored": False}


def load_author_auxiliary(checkout: Path | str):
    """Compile the three unchanged author AST blocks, without executing their engine."""
    provenance = locked_author_identity(checkout)
    key = provenance["author_files_digest"]
    if key in _AUTHOR_AUX:
        return _AUTHOR_AUX[key]
    path = Path(checkout) / "DualCastS/model/train.py"
    syntax = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    blocks = {}
    batch_loop = next(node for node in ast.walk(function)
                      if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "batch_idx"
                      and isinstance(node.iter, ast.Call) and node.iter.args
                      and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == "train_num_batch")
    for node in batch_loop.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and isinstance(node.test.value, ast.Name) and node.test.value.id == "args" and node.test.attr in ("l_flt", "l_env", "l_dbi"):
            if node.test.attr in blocks:
                raise ValueError("author auxiliary AST is ambiguous")
            blocks[node.test.attr] = copy.deepcopy(node)
    if set(blocks) != {"l_flt", "l_env", "l_dbi"}:
        raise ValueError("author reciprocal KL/DBI blocks could not be extracted exactly")
    header = ast.parse("def author_loss(loss_batch, gfeati, gfeate, p_Z, anchors, args, kl_criterion, torch):\n    l_flt = loss_batch.new_zeros(())\n    l_env = loss_batch.new_zeros(())\n    l_dbi = loss_batch.new_zeros(())\n    return loss_batch, {'filter': l_flt, 'environment': l_env, 'dbi': l_dbi}\n")
    definition = header.body[0]
    definition.body[3:3] = [blocks[name] for name in ("l_flt", "l_env", "l_dbi")]
    ast.fix_missing_locations(header)
    namespace = {"__name__": "locked_dualcast_author_auxiliary", "F": F}
    exec(compile(header, str(path), "exec"), namespace)
    extracted = namespace["author_loss"]
    extracted.author_ast_digest = canonical_digest({name: ast.dump(blocks[name], include_attributes=False) for name in sorted(blocks)})
    _AUTHOR_AUX[key] = extracted
    return extracted


class _PermutationReplay:
    def __init__(self, permutation: torch.Tensor) -> None:
        self.permutation = permutation
        self.calls = 0

    def randperm(self, size: int) -> torch.Tensor:
        if size != self.permutation.numel() or self.calls:
            raise ValueError("author environment permutation replay is inconsistent")
        self.calls += 1
        return self.permutation

    def __getattr__(self, name: str):
        return getattr(torch, name)


class TrainableDualCastS(DualCastSReference):
    def __init__(self, **kwargs: Any) -> None:
        checkout = Path(kwargs["checkout"]).resolve()
        identity = locked_author_identity(checkout)
        super().__init__(**kwargs)
        self.author_loss = load_author_auxiliary(checkout)
        self.register_buffer("locked_adjacency", kwargs["adjacency"].detach().clone().to(self.flow_mean))
        self.upstream = identity | {"adapter_provenance": self.provenance,
                                    "author_aux_ast_digest": self.author_loss.author_ast_digest,
                                    "adjacency_sha256": array_digest(self.locked_adjacency.cpu().numpy())}
        self.constructor = {
            "checkout": str(checkout), "flow_mean": float(kwargs["flow_mean"]),
            "flow_std": float(kwargs["flow_std"]),
            "embed_dim": kwargs.get("embed_dim", 64), "layers": kwargs.get("layers", 1),
            "heads": kwargs.get("heads", 8), "forward_expansion": kwargs.get("forward_expansion", 4),
            "dropout": kwargs.get("dropout", 0.0),
        }

    def factory_kwargs(self) -> dict[str, Any]:
        return copy.deepcopy(self.constructor) | {"adjacency": self.locked_adjacency.detach().cpu().clone()}


def make_dualcast_model(*, device: str | torch.device = "cpu", **kwargs: Any) -> TrainableDualCastS:
    return TrainableDualCastS(**(kwargs | {"device": device}))


def regional_adjacency(region: Any) -> torch.Tensor:
    edges = np.asarray(region.edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("v3 DualCast adjacency requires the bound integer regional edge index")
    if edges.size and (edges.min() < 0 or edges.max() >= region.N):
        raise ValueError("regional edge indices are outside the node identity")
    adjacency = np.zeros((region.N, region.N), dtype=np.float32)
    adjacency[edges[0], edges[1]] = 1.0
    return torch.from_numpy(adjacency)


def dualcast_recipe(name: str, *, batch_size: int = 32, epochs: int = 100) -> dict[str, Any]:
    if name not in RECIPES or type(batch_size) is not int or batch_size <= 0 or epochs not in (30, 100):
        raise ValueError("DualCast v3 requires a named recipe, positive batch and 30/100 epoch budget")
    return {
        "name": name, "exact_native": False,
        "primary": "v3_raw_flow_masked_mae_all_effective_observations",
        "auxiliary_equations": "unchanged_locked_author_AST_reciprocal_batchmean_KL_and_detached_DBI",
        "undefined_auxiliary_policy": "fail_closed" if name == RECIPES[0] else "omit_only_mathematically_undefined_terms_with_recorded_reason",
        "kl_epsilon_added": False, "alternative_auxiliary_loss_substituted": False,
        "dbi_autograd": "author_torch_tensor_of_ratios_detaches_gradients",
        "alpha": 1.0, "beta": 0.01, "gamma": 0.05, "num_prototypes": 17,
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "epsilon": 1e-8, "gradient_clip": None},
        "scheduler": {"name": "StepLR", "step_size": 10, "gamma": 0.9, "step_unit": "completed_epoch"},
        "batch_size": batch_size, "microbatch_size": batch_size,
        "batch_normalization_scope": "full_effective_batch",
        "auxiliary_scope": "full_effective_batch_not_microbatch_average",
        "tail_policy": "retain_valid_main_points; native_equations_fail_closed_or_defined_aux_guards",
        "partial_batch_validation": "retain_all", "early_stopping": False,
        "requested_epoch_budget": epochs, "author_epoch_cap": 100, "author_patience": 100,
        "author_batch_size": 32,
        "batch_size_override": batch_size != 32,
        "calendar": "real_history_weekday_to_friday_origin_and_weekend_flag",
        "normalization": "unique_raw_observed_v3_training_timestamps",
        "adjacency": "v3_bound_regional_directed_binary_edges_no_extra_self_loops",
        "claim_scope": "same_protocol_regional_masked_compatibility_reference_not_exact_native_or_PeMS_score",
    }


def _kl_domain(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, str | None]:
    probabilities = F.softmax(target, dim=-1)
    other = F.softmax(logits, dim=-1)
    value = F.kl_div(F.log_softmax(logits, dim=-1), probabilities, reduction="batchmean")
    if torch.equal(other.detach(), probabilities.detach()):
        return value, "identical_distributions_zero_KL"
    if not torch.isfinite(value):
        return value, "nonfinite_KL"
    if value <= 0:
        return value, "nonpositive_floating_point_KL"
    return value, None


def _dbi_domain(members: list[list[torch.Tensor]], anchors: torch.Tensor) -> tuple[int, str | None]:
    flattened = anchors.reshape(len(members), -1)
    active = []
    with torch.no_grad():
        for index, group in enumerate(members):
            if not group:
                continue
            points = torch.stack(group).reshape(len(group), -1)
            spread = torch.norm(flattened[index] - points, p=2) / len(group)
            if not torch.isfinite(spread):
                return len(active), "nonfinite_prototype_spread"
            if spread != 0:
                active.append(index)
        if len(active) < 2:
            return len(active), "fewer_than_two_nonzero_valid_prototype_groups"
        for position, index in enumerate(active):
            for other in active[position + 1:]:
                distance = torch.norm(flattened[index] - flattened[other], p=2)
                if not torch.isfinite(distance) or distance <= 0:
                    return len(active), "zero_or_nonfinite_between_anchor_distance"
    return len(active), None


def compute_dualcast_loss(model: TrainableDualCastS, prediction: torch.Tensor,
                          target: torch.Tensor, mask: torch.Tensor, recipe: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    if recipe.get("name") not in RECIPES:
        raise ValueError("DualCast objective requires an explicitly named locked compatibility recipe")
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("DualCast v3 prediction/target/boolean observation mask must match")
    if not torch.isfinite(prediction).all() or not mask.any() or not torch.isfinite(target[mask]).all():
        raise FloatingPointError("DualCast v3 main objective requires finite forecasts and valid observed targets")
    aux = model.last_aux
    if aux is None:
        raise ValueError("DualCast auxiliary features are missing from the current forecast")
    intrinsic, environment, anchors = aux["intrinsic"], aux["environment"], aux["anchors"]
    if any(not torch.isfinite(value).all() for value in (intrinsic, environment, anchors)):
        raise FloatingPointError("DualCast internal auxiliary tensors must remain finite")
    primary = (prediction[mask] - target[mask]).abs().mean()
    main_value = float(primary.detach())
    # Draw once as the author does. The unchanged AST reuses this exact permutation.
    permutation = torch.randperm(environment.shape[0])
    filter_kl, filter_reason = _kl_domain(intrinsic, environment)
    environment_kl, environment_reason = _kl_domain(environment[permutation], environment)
    groups, dbi_reason = _dbi_domain(aux["prototype_members"], anchors)
    reasons = {name: reason for name, reason in (("filter", filter_reason), ("environment", environment_reason), ("dbi", dbi_reason)) if reason is not None}
    if recipe["name"] == RECIPES[0] and reasons:
        raise UndefinedAuthorAuxiliary("locked author equations undefined: " + json_reasons(reasons))
    arguments = SimpleNamespace(alpha=recipe["alpha"], beta=recipe["beta"], gamma=recipe["gamma"], num_prototypes=17,
                                l_flt=filter_reason is None, l_env=environment_reason is None, l_dbi=dbi_reason is None)
    replay = _PermutationReplay(permutation)
    loss, terms = model.author_loss(primary, intrinsic, environment, aux["prototype_members"], anchors,
                                    arguments, nn.KLDivLoss(reduction="batchmean"), replay)
    if not torch.isfinite(loss) or any(not torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError("defined DualCast author equations produced a nonfinite objective")
    parts = {
        "main": main_value, "filter": float(terms["filter"].detach()),
        "environment": float(terms["environment"].detach()), "dbi": float(terms["dbi"].detach()),
        "objective": float(loss.detach()), "omitted_undefined_auxiliary": reasons,
        "filter_kl": float(filter_kl.detach()) if torch.isfinite(filter_kl) else None,
        "environment_kl": float(environment_kl.detach()) if torch.isfinite(environment_kl) else None,
        "environment_permutation": permutation.tolist(), "active_dbi_groups": groups,
        "dbi_requires_grad": bool(terms["dbi"].requires_grad),
        "exact_native_claim": False,
    }
    return loss, parts


def json_reasons(reasons: Mapping[str, str]) -> str:
    return "; ".join(f"{name}={reason}" for name, reason in sorted(reasons.items()))
