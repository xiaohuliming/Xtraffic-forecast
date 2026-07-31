from __future__ import annotations

import copy
import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from .profiles import load_profile_set
from .upstream_lock import verify_pristine_upstream_checkout


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)
BIST_CHECKOUT = (
    PROJECT_ROOT / ".third_party" / "igstgnn_table4" / "pristine" / "bist_dd94adf"
)
BIST_SOURCE_ID = "bist_dd94adf"

_IMPORT_LOCK = threading.RLock()
_UPSTREAM: SimpleNamespace | None = None


@dataclass(frozen=True)
class BiSTFactoryResult:
    model: torch.nn.Module
    engine: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.MultiStepLR
    effective_config: dict[str, Any]
    provenance: dict[str, Any]


def _load_upstream() -> SimpleNamespace:
    global _UPSTREAM
    with _IMPORT_LOCK:
        if _UPSTREAM is not None:
            return _UPSTREAM
        saved = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "src" or name.startswith("src.")
        }
        for name in saved:
            del sys.modules[name]
        package = ModuleType("src")
        package.__path__ = [str(BIST_CHECKOUT / "src")]
        package.__package__ = "src"
        sys.modules["src"] = package
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            model_module = importlib.import_module("src.models.bist")
            engine_module = importlib.import_module("src.base.engine")
            metrics_module = importlib.import_module("src.utils.metrics")
            _UPSTREAM = SimpleNamespace(
                MLP=model_module.MLP,
                BiST=model_module.BiST,
                BaseEngine=engine_module.BaseEngine,
                masked_mae=metrics_module.masked_mae,
            )
        finally:
            for name in tuple(sys.modules):
                if name == "src" or name.startswith("src."):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.dont_write_bytecode = previous_dont_write_bytecode
        return _UPSTREAM


def _apply_upstream_runtime_seed(seed: int) -> None:
    torch.set_num_threads(3)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _validate_inputs(
    *,
    profile_effective: Mapping[str, Any],
    node_num: int,
    dataloader: Mapping[str, Any],
    scaler: Any,
    logger: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_set = load_profile_set(PROFILE_PATH)
    profile = profile_set["profiles"]["bist"]
    effective = copy.deepcopy(profile["effective"])
    if dict(profile_effective) != effective:
        raise ValueError("profile_effective does not match the locked BiST profile")
    if isinstance(node_num, bool) or not isinstance(node_num, int):
        raise TypeError("node_num must be an integer")
    if node_num <= 0:
        raise ValueError("node_num must be positive")
    for split in ("train_loader", "val_loader", "test_loader"):
        if split not in dataloader:
            raise ValueError(f"dataloader is missing {split}")
    if not callable(getattr(scaler, "inverse_transform", None)):
        raise TypeError("scaler must provide inverse_transform")
    if not callable(getattr(logger, "info", None)):
        raise TypeError("logger must provide info")
    return profile_set, effective


def build_bist_factory(
    *,
    profile_effective: Mapping[str, Any],
    manifest_frozen_seed: int,
    node_num: int,
    device: torch.device | str,
    dataloader: Mapping[str, Any],
    scaler: Any,
    logger: Any,
    output_dir: Path | str,
    max_epochs_override: int | None = None,
) -> BiSTFactoryResult:
    profile_set, effective = _validate_inputs(
        profile_effective=profile_effective,
        node_num=node_num,
        dataloader=dataloader,
        scaler=scaler,
        logger=logger,
    )
    if isinstance(manifest_frozen_seed, bool) or not isinstance(
        manifest_frozen_seed, int
    ):
        raise TypeError("manifest_frozen_seed must be an integer")
    if not 0 <= manifest_frozen_seed < 999999:
        raise ValueError("manifest_frozen_seed must be in [0, 999999)")
    if (
        effective["seed"] is not None
        or effective.get("seed_policy") != "upstream_runtime_random_draw"
    ):
        raise ValueError("Locked BiST profile seed policy is invalid")
    effective_overrides: dict[str, dict[str, int]] = {}
    runtime_effective = copy.deepcopy(effective)
    if max_epochs_override is not None:
        if (
            isinstance(max_epochs_override, bool)
            or not isinstance(max_epochs_override, int)
        ):
            raise TypeError("max_epochs_override must be an integer")
        if not 1 <= max_epochs_override <= effective["max_epochs"]:
            raise ValueError("max_epochs_override must be within the native cap")
        effective_overrides["max_epochs"] = {
            "from": effective["max_epochs"],
            "to": max_epochs_override,
        }
        runtime_effective["max_epochs"] = max_epochs_override
    upstream_provenance = verify_pristine_upstream_checkout(
        source_id=BIST_SOURCE_ID,
        checkout=BIST_CHECKOUT,
    )
    upstream = _load_upstream()
    _apply_upstream_runtime_seed(manifest_frozen_seed)

    input_contract = effective["input_contract"]
    model_config = effective["model"]
    adjacency = effective["adjacency"]
    optimizer_config = effective["optimizer"]
    device_value = torch.device(device)
    base = upstream.MLP(
        node_num=node_num,
        input_dim=input_contract["input_dim"],
        output_dim=input_contract["target_dim"],
        num_layer=model_config["layers"],
        model_dim=model_config["model_dim"],
        prompt_dim=model_config["prompt_dim"],
        tod_size=input_contract["time_slots_per_day"],
        kernel_size=model_config["kernel_size"],
    )
    hidden_embedding_dim = model_config["model_dim"] + 3 * model_config["prompt_dim"]
    model_args = {
        "extra_type": model_config["extra_type"],
        "same": model_config["same"],
        "hid_dim": model_config["hidden_dim"],
        "horizon": 12,
        "predefined_adj": int(adjacency["predefined"]),
        "adjs": [],
        "rp_layer": model_config["residual_propagation_layers"],
        "datadriven_adj": int(adjacency["data_driven"]),
        "datadriven_adj_dim": 32,
        "datadriven_adj_head": 0,
        "adaptive_adj": int(adjacency["adaptive"]),
        "adaptive_adj_dim": 10,
        "use_global_opt": False,
        "mrf": 1,
    }
    model = upstream.BiST(
        node_num=node_num,
        input_dim=input_contract["input_dim"],
        output_dim=input_contract["target_dim"],
        model_args=model_args,
        stmodel=base,
        dim=[hidden_embedding_dim, hidden_embedding_dim],
        core=model_config["core_dim"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        eps=optimizer_config["epsilon"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optimizer_config["scheduler"]["milestones"],
        gamma=optimizer_config["scheduler"]["gamma"],
    )
    engine_args = SimpleNamespace(
        core=model_config["core_dim"],
        horizon=12,
    )
    engine = upstream.BaseEngine(
        device=device_value,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=upstream.masked_mae,
        lrate=optimizer_config["learning_rate"],
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=optimizer_config["gradient_clip"],
        max_epochs=runtime_effective["max_epochs"],
        patience=effective["patience"],
        log_dir=str(Path(output_dir)),
        logger=logger,
        args=engine_args,
    )
    profile = profile_set["profiles"]["bist"]
    provenance = {
        **upstream_provenance,
        "profile_set_sha256": profile_set["profile_set_sha256"],
        "config_sha256": profile_set["config_hashes"]["bist"],
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "engine_class": (f"{engine.__class__.__module__}.{engine.__class__.__name__}"),
        "seed_policy": "upstream_runtime_random_draw_frozen_in_manifest",
        "manifest_frozen_seed": manifest_frozen_seed,
        "effective_overrides": effective_overrides,
        "runtime_contract": {
            "torch_num_threads": 3,
            "cuda_deterministic": False,
            "cudnn_benchmark": False,
        },
        "license_notice": (
            "The pinned BiST checkout has no repository license file and "
            "is executed in place without vendoring its source."
        ),
        "compatibility_patches_declared": list(profile["compatibility_patches"]),
        "compatibility_patches_applied_by_factory": [
            "bist_get_config_recursion_fix_v1",
            "bist_manifest_frozen_runtime_seed_v2",
            "bist_region_registry_and_tod_v1",
        ],
    }
    return BiSTFactoryResult(
        model=model,
        engine=engine,
        optimizer=optimizer,
        scheduler=scheduler,
        effective_config=runtime_effective,
        provenance=provenance,
    )
