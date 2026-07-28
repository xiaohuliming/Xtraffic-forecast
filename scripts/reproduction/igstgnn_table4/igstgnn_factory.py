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
IGSTGNN_CHECKOUT = (
    PROJECT_ROOT / ".third_party" / "igstgnn_table4" / "pristine" / "igstgnn_64c6e24"
)
IGSTGNN_SOURCE_ID = "igstgnn_64c6e24"

_SENSOR_DIMENSION_KEYS = (
    "sensor_type_size",
    "surface_size",
    "roadway_use_size",
)
_IMPORT_LOCK = threading.RLock()
_UPSTREAM: SimpleNamespace | None = None


@dataclass(frozen=True)
class IGSTGNNFactoryResult:
    model: torch.nn.Module
    engine: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.MultiStepLR
    effective_config: dict[str, Any]
    provenance: dict[str, Any]


class IGSTGNNDataParallel(torch.nn.DataParallel):
    @staticmethod
    def _to_replica_device(value: Any, device: torch.device) -> Any:
        if torch.is_tensor(value):
            return value.to(device=device)
        if isinstance(value, list):
            moved = [
                IGSTGNNDataParallel._to_replica_device(item, device)
                for item in value
            ]
            return (
                moved
                if any(new is not old for new, old in zip(moved, value, strict=True))
                else value
            )
        if isinstance(value, tuple):
            moved = tuple(
                IGSTGNNDataParallel._to_replica_device(item, device)
                for item in value
            )
            return (
                moved
                if any(new is not old for new, old in zip(moved, value, strict=True))
                else value
            )
        if isinstance(value, dict):
            moved = {
                key: IGSTGNNDataParallel._to_replica_device(item, device)
                for key, item in value.items()
            }
            if all(moved[key] is item for key, item in value.items()):
                return value
            copied = value.copy()
            copied.update(moved)
            return copied
        return value

    def replicate(
        self,
        module: torch.nn.Module,
        device_ids: list[int],
    ) -> list[torch.nn.Module]:
        replicas = super().replicate(module, device_ids)
        internal_names = {"_parameters", "_buffers", "_modules"}
        for replica, device_id in zip(replicas, device_ids, strict=True):
            device = torch.device("cuda", device_id)
            for child in replica.modules():
                for name, value in tuple(vars(child).items()):
                    if name in internal_names:
                        continue
                    moved = self._to_replica_device(value, device)
                    if moved is not value:
                        setattr(child, name, moved)
        return replicas

    @property
    def horizon(self) -> int:
        return self.module.horizon

    def param_num(self) -> int:
        return self.module.param_num()


def wrap_igstgnn_data_parallel(
    model: torch.nn.Module,
    *,
    device: torch.device,
    enabled: bool,
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    if not enabled:
        return model, None
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError(
            "IGSTGNN data parallel execution requires primary device cuda:0"
        )
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) < 2:
        raise ValueError(
            "IGSTGNN data parallel execution requires at least two visible GPUs"
        )
    wrapped = IGSTGNNDataParallel(
        model,
        device_ids=device_ids,
        output_device=0,
    )
    return wrapped, {
        "type": "torch_data_parallel",
        "device_ids": device_ids,
        "full_batch_preserved": True,
        "unregistered_tensor_policy": "replica_device_copy",
        "upstream_source_modified": False,
    }


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
        package.__path__ = [str(IGSTGNN_CHECKOUT / "src")]
        package.__package__ = "src"
        sys.modules["src"] = package
        try:
            model_module = importlib.import_module("src.models.igstgnn")
            engine_module = importlib.import_module("src.engines.igstgnn_engine")
            metrics_module = importlib.import_module("src.utils.metrics")
            graph_module = importlib.import_module("src.utils.graph_algo")
            _UPSTREAM = SimpleNamespace(
                IGSTGNN=model_module.IGSTGNN,
                IGSTGNN_Engine=engine_module.IGSTGNN_Engine,
                masked_mae=metrics_module.masked_mae,
                normalize_adj_mx=graph_module.normalize_adj_mx,
            )
        finally:
            for name in tuple(sys.modules):
                if name == "src" or name.startswith("src."):
                    del sys.modules[name]
            sys.modules.update(saved)
        return _UPSTREAM


def _coerce_adjacency(adjacency: Any, node_num: int) -> np.ndarray:
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.detach().cpu().numpy()
    value = np.asarray(adjacency)
    if value.shape != (node_num, node_num):
        raise ValueError(
            f"adjacency must have shape {(node_num, node_num)}, got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError("adjacency must be numeric")
    if not np.isfinite(value).all():
        raise ValueError("adjacency must contain only finite values")
    return np.array(value, dtype=np.float32, copy=True)


def _apply_upstream_seed(seed: int) -> None:
    torch.set_num_threads(3)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def _coerce_sensor_dimensions(sensor_dimensions: Any) -> dict[str, int]:
    if isinstance(sensor_dimensions, Mapping):
        if set(sensor_dimensions) != set(_SENSOR_DIMENSION_KEYS):
            raise ValueError(
                "sensor_dimensions must contain exactly "
                f"{list(_SENSOR_DIMENSION_KEYS)!r}"
            )
        values = {key: sensor_dimensions[key] for key in _SENSOR_DIMENSION_KEYS}
    else:
        try:
            values = {
                key: getattr(sensor_dimensions, key) for key in _SENSOR_DIMENSION_KEYS
            }
        except AttributeError as error:
            raise TypeError(
                "sensor_dimensions must provide the three category sizes"
            ) from error
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"sensor_dimensions.{key} must be an integer")
        if value <= 0:
            raise ValueError(f"sensor_dimensions.{key} must be positive")
    return values


def _validate_inputs(
    *,
    profile_effective: Mapping[str, Any],
    node_num: int,
    dataloader: Any,
    scaler: Any,
    sensor_dimensions: Any,
    logger: Any,
    dataset: str,
    data_path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], Path]:
    profile_set = load_profile_set(PROFILE_PATH)
    canonical = profile_set["profiles"]["igstgnn"]["effective"]
    if dict(profile_effective) != canonical:
        raise ValueError("profile_effective does not match the locked igstgnn profile")
    if isinstance(node_num, bool) or not isinstance(node_num, int):
        raise TypeError("node_num must be an integer")
    if node_num <= 0:
        raise ValueError("node_num must be positive")
    if not isinstance(dataset, str) or dataset not in canonical["batch_size_by_region"]:
        raise ValueError("dataset must be an official Table 4 region")
    for split in ("train_loader", "val_loader", "test_loader"):
        try:
            loader = dataloader[split]
        except (KeyError, TypeError) as error:
            raise ValueError(f"dataloader is missing {split}") from error
        if not callable(getattr(loader, "get_iterator", None)):
            raise TypeError(f"dataloader {split} must provide get_iterator")
    train_batches = getattr(dataloader["train_loader"], "num_batch", None)
    if (
        isinstance(train_batches, bool)
        or not isinstance(train_batches, int)
        or train_batches <= 0
    ):
        raise ValueError("train_loader.num_batch must be a positive integer")
    if not callable(getattr(dataloader["train_loader"], "shuffle", None)):
        raise TypeError("train_loader must provide shuffle")
    if not callable(getattr(scaler, "inverse_transform", None)):
        raise TypeError("scaler must provide inverse_transform")
    if not callable(getattr(logger, "info", None)):
        raise TypeError("logger must provide info")
    dimensions = _coerce_sensor_dimensions(sensor_dimensions)
    data_directory = Path(data_path)
    if not data_directory.is_dir():
        raise FileNotFoundError(f"IGSTGNN data_path not found: {data_directory}")
    for filename in ("desc_mapping.json", "type_mapping.json"):
        path = data_directory / filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"IGSTGNN mapping file not found: {path}")
    return (
        profile_set,
        copy.deepcopy(canonical),
        dimensions,
        data_directory,
    )


def build_igstgnn_factory(
    *,
    profile_effective: Mapping[str, Any],
    node_num: int,
    device: torch.device | str,
    adjacency: Any,
    dataloader: Any,
    scaler: Any,
    sensor_dimensions: Any,
    logger: Any,
    output_dir: Path | str,
    dataset: str,
    data_path: Path | str,
    data_parallel: bool = False,
) -> IGSTGNNFactoryResult:
    profile_set, effective, dimensions, data_directory = _validate_inputs(
        profile_effective=profile_effective,
        node_num=node_num,
        dataloader=dataloader,
        scaler=scaler,
        sensor_dimensions=sensor_dimensions,
        logger=logger,
        dataset=dataset,
        data_path=data_path,
    )
    source = verify_pristine_upstream_checkout(
        source_id=IGSTGNN_SOURCE_ID,
        checkout=IGSTGNN_CHECKOUT,
    )
    upstream = _load_upstream()
    _apply_upstream_seed(effective["seed"])
    device_value = torch.device(device)
    adjacency_value = _coerce_adjacency(adjacency, node_num)
    normalized_adjacency = upstream.normalize_adj_mx(
        adjacency_value,
        "doubletransition",
    )
    adjacency_tensors = [
        torch.as_tensor(
            np.asarray(value),
            dtype=torch.float32,
            device=device_value,
        )
        for value in normalized_adjacency
    ]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_contract = effective["input_contract"]
    model_config = effective["model"]
    model_args = {
        "num_feat": input_contract["flow_features"],
        "num_hidden": model_config["hidden_dim"],
        "node_hidden": model_config["node_embedding_dim"],
        "time_emb_dim": model_config["time_embedding_dim"],
        "layer": model_config["layers"],
        "k_t": model_config["temporal_kernel"],
        "k_s": model_config["spatial_kernel"],
        "gap": model_config["gap"],
        "cl_epoch": model_config["curriculum_interval"],
        "warm_epoch": model_config["warm_epochs"],
        "tpd": input_contract["time_slots_per_day"],
        "dropout": model_config["dropout"],
        "sigma_t": model_config["sigma_t"],
        "lambda_incident": model_config["lambda_incident"],
        "adjs": adjacency_tensors,
        "incident": True,
        "use_sensor_info": True,
        "dataset": dataset,
        "data_path": str(data_directory),
        **dimensions,
    }
    sequence = profile_set["shared"]["sequence"]
    model = upstream.IGSTGNN(
        node_num=node_num,
        input_dim=input_contract["input_dim"],
        output_dim=input_contract["target_dim"],
        seq_len=sequence["input_steps"],
        horizon=sequence["output_steps"],
        model_args=model_args,
        dataset=dataset,
        data_path=str(data_directory),
        use_sensor_info=True,
    )
    model, execution_adapter = wrap_igstgnn_data_parallel(
        model,
        device=device_value,
        enabled=data_parallel,
    )
    optimizer_config = effective["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        eps=optimizer_config["epsilon"],
    )
    scheduler_config = effective["scheduler"]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_config["milestones"],
        gamma=scheduler_config["gamma"],
    )
    train_batches = dataloader["train_loader"].num_batch
    engine = upstream.IGSTGNN_Engine(
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
        max_epochs=effective["max_epochs"],
        patience=effective["patience"],
        log_dir=str(output_path),
        logger=logger,
        seed=effective["seed"],
        cl_step=model_config["curriculum_interval"] * train_batches,
        warm_step=model_config["warm_epochs"] * train_batches,
        horizon=sequence["output_steps"],
        incident=True,
    )
    profile = profile_set["profiles"]["igstgnn"]
    provenance = {
        "factory_schema_version": 1,
        "source_id": IGSTGNN_SOURCE_ID,
        "commit": source["commit"],
        "tree": source["tree"],
        "checkout": source["checkout"],
        "lock_digest": source["lock_digest"],
        "profile_id": profile["profile_id"],
        "profile_config_hash": profile_set["config_hashes"]["igstgnn"],
        "seed_applied": effective["seed"],
        "runtime_contract": {
            "torch_num_threads": 3,
            "cuda_deterministic": False,
            "cudnn_benchmark": False,
        },
        "upstream_entrypoint": "experiments/IGSTGNN/main.py",
        "model_class": "src.models.igstgnn.IGSTGNN",
        "engine_class": ("src.engines.igstgnn_engine.IGSTGNN_Engine"),
        "incident": True,
        "use_sensor_info": True,
        "adjacency_normalization": "doubletransition",
        "upstream_source_modified": False,
        "compatibility_patches_declared": list(profile["compatibility_patches"]),
        "compatibility_patches_applied_by_factory": [],
    }
    if execution_adapter is not None:
        provenance["execution_adapter"] = execution_adapter
    return IGSTGNNFactoryResult(
        model=model,
        engine=engine,
        optimizer=optimizer,
        scheduler=scheduler,
        effective_config=effective,
        provenance=provenance,
    )
