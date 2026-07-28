from __future__ import annotations

import copy
import hashlib
import importlib
import json
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
LARGEST_CHECKOUT = (
    PROJECT_ROOT / ".third_party" / "igstgnn_table4" / "pristine" / "largest_b7c6ff5"
)
LARGEST_SOURCE_ID = "largest_b7c6ff5"

_IMPORT_LOCK = threading.RLock()
_UPSTREAM: SimpleNamespace | None = None


@dataclass(frozen=True)
class LargeSTFactoryResult:
    model: torch.nn.Module
    engine: Any
    optimizer: torch.optim.Optimizer | None
    scheduler: Any | None
    effective_config: dict[str, Any]
    factory_config: dict[str, Any]
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
        package.__path__ = [str(LARGEST_CHECKOUT / "src")]
        package.__package__ = "src"
        sys.modules["src"] = package
        try:
            hl_model = importlib.import_module("src.models.hl")
            lstm_model = importlib.import_module("src.models.lstm")
            dcrnn_model = importlib.import_module("src.models.dcrnn")
            agcrn_model = importlib.import_module("src.models.agcrn")
            stgcn_model = importlib.import_module("src.models.stgcn")
            gwnet_model = importlib.import_module("src.models.gwnet")
            astgcn_model = importlib.import_module("src.models.astgcn")
            sttn_model = importlib.import_module("src.models.sttn")
            dstagnn_model = importlib.import_module("src.models.dstagnn")
            dgcrn_model = importlib.import_module("src.models.dgcrn")
            d2stgnn_model = importlib.import_module("src.models.d2stgnn")
            base_engine = importlib.import_module("src.base.engine")
            hl_engine = importlib.import_module("src.engines.hl_engine")
            dcrnn_engine = importlib.import_module("src.engines.dcrnn_engine")
            agcrn_engine = importlib.import_module("src.engines.agcrn_engine")
            astgcn_engine = importlib.import_module("src.engines.astgcn_engine")
            dstagnn_engine = importlib.import_module("src.engines.dstagnn_engine")
            dgcrn_engine = importlib.import_module("src.engines.dgcrn_engine")
            d2stgnn_engine = importlib.import_module("src.engines.d2stgnn_engine")
            metrics = importlib.import_module("src.utils.metrics")
            graph_algo = importlib.import_module("src.utils.graph_algo")
            _UPSTREAM = SimpleNamespace(
                HL=hl_model.HL,
                LSTM=lstm_model.LSTM,
                DCRNN=dcrnn_model.DCRNN,
                AGCRN=agcrn_model.AGCRN,
                STGCN=stgcn_model.STGCN,
                GWNET=gwnet_model.GWNET,
                ASTGCN=astgcn_model.ASTGCN,
                STTN=sttn_model.STTN,
                DSTAGNN=dstagnn_model.DSTAGNN,
                DGCRN=dgcrn_model.DGCRN,
                D2STGNN=d2stgnn_model.D2STGNN,
                BaseEngine=base_engine.BaseEngine,
                HL_Engine=hl_engine.HL_Engine,
                DCRNN_Engine=dcrnn_engine.DCRNN_Engine,
                AGCRN_Engine=agcrn_engine.AGCRN_Engine,
                ASTGCN_Engine=astgcn_engine.ASTGCN_Engine,
                DSTAGNN_Engine=dstagnn_engine.DSTAGNN_Engine,
                DGCRN_Engine=dgcrn_engine.DGCRN_Engine,
                D2STGNN_Engine=d2stgnn_engine.D2STGNN_Engine,
                masked_mae=metrics.masked_mae,
                normalize_adj_mx=graph_algo.normalize_adj_mx,
                calculate_cheb_poly=graph_algo.calculate_cheb_poly,
            )
        finally:
            for name in tuple(sys.modules):
                if name == "src" or name.startswith("src."):
                    del sys.modules[name]
            sys.modules.update(saved)
        return _UPSTREAM


def _validate_common_inputs(
    *,
    method: str,
    profile_effective: Mapping[str, Any],
    node_num: int,
    dataloader: Mapping[str, Any],
    scaler: Any,
    logger: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_set = load_profile_set(PROFILE_PATH)
    profiles = profile_set["profiles"]
    if method not in profiles:
        raise ValueError(f"Unsupported LargeST Table 4 method: {method}")
    canonical = profiles[method]["effective"]
    if dict(profile_effective) != canonical:
        raise ValueError(
            f"profile_effective does not match the locked {method} profile"
        )
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
    return profile_set, copy.deepcopy(canonical)


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
    return np.array(value, copy=True)


def _adjacency_sha256(adjacency: Any, node_num: int) -> str:
    value = np.ascontiguousarray(_coerce_adjacency(adjacency, node_num))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _apply_upstream_runtime(seed: int | None) -> None:
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_largest_factory(
    *,
    method: str,
    profile_effective: Mapping[str, Any],
    node_num: int,
    device: torch.device | str,
    adjacency: Any,
    dataloader: Mapping[str, Any],
    scaler: Any,
    logger: Any,
    output_dir: Path | str,
) -> LargeSTFactoryResult:
    profile_set, effective = _validate_common_inputs(
        method=method,
        profile_effective=profile_effective,
        node_num=node_num,
        dataloader=dataloader,
        scaler=scaler,
        logger=logger,
    )
    if method not in {
        "hl",
        "lstm",
        "dcrnn",
        "agcrn",
        "stgcn",
        "gwnet",
        "astgcn",
        "sttn",
        "dstagnn",
        "dgcrn",
        "d2stgnn",
    }:
        raise ValueError(f"Unsupported LargeST Table 4 method: {method}")

    source = verify_pristine_upstream_checkout(
        source_id=LARGEST_SOURCE_ID,
        checkout=LARGEST_CHECKOUT,
    )
    upstream = _load_upstream()
    device_value = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _apply_upstream_runtime(effective["seed"])

    common_model = {
        "node_num": node_num,
        "input_dim": effective["input_contract"]["input_dim"],
        "output_dim": effective["input_contract"]["target_dim"],
        "seq_len": profile_set["shared"]["sequence"]["input_steps"],
        "horizon": profile_set["shared"]["sequence"]["output_steps"],
    }
    compatibility_patches: list[str] = []
    if method == "hl":
        model = upstream.HL(**common_model)
        optimizer = None
        scheduler = None
        engine_class = upstream.HL_Engine
        learning_rate = 0.0
        clip_grad_value = 0.0
        max_epochs = 100
        patience = 30
        model_class_name = "src.models.hl.HL"
        engine_class_name = "src.engines.hl_engine.HL_Engine"
        runtime_defaults = {"max_epochs": 100, "patience": 30}
    elif method == "lstm":
        model_config = effective["model"]
        model = upstream.LSTM(
            **common_model,
            init_dim=model_config["init_dim"],
            hid_dim=model_config["hidden_dim"],
            end_dim=model_config["end_dim"],
            layer=model_config["layers"],
            dropout=model_config["dropout"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.BaseEngine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.lstm.LSTM"
        engine_class_name = "src.base.engine.BaseEngine"
        runtime_defaults = {}
    elif method == "dcrnn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        model_config = effective["model"]
        model = upstream.DCRNN(
            **common_model,
            device=device_value,
            adj_mx=adjacency_value,
            n_filters=model_config["filter_dim"],
            max_diffusion_step=model_config["diffusion_steps"],
            filter_type="doubletransition",
            num_rnn_layers=model_config["rnn_layers"],
            cl_decay_steps=model_config["curriculum_decay_steps"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler_config = optimizer_config["scheduler"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config["gamma"],
        )
        engine_class = upstream.DCRNN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.dcrnn.DCRNN"
        engine_class_name = "src.engines.dcrnn_engine.DCRNN_Engine"
        compatibility_patches = ["dcrnn_scheduler_verbose_compatibility_v1"]
        runtime_defaults = {}
    elif method == "agcrn":
        model_config = effective["model"]
        model = upstream.AGCRN(
            **common_model,
            embed_dim=model_config["embedding_dim"],
            rnn_unit=model_config["rnn_units"],
            num_layer=model_config["layers"],
            cheb_k=model_config["chebyshev_order"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.AGCRN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = 0.0
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.agcrn.AGCRN"
        engine_class_name = "src.engines.agcrn_engine.AGCRN_Engine"
        runtime_defaults = {}
    elif method == "stgcn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        adjacency_value = adjacency_value - np.eye(node_num)
        gso = upstream.normalize_adj_mx(adjacency_value, "scalap")[0]
        gso = torch.as_tensor(np.asarray(gso), dtype=torch.float32, device=device_value)
        model_config = effective["model"]
        block_num = model_config["blocks"]
        temporal_kernel = model_config["temporal_kernel"]
        sequence_length = common_model["seq_len"]
        horizon = common_model["horizon"]
        output_steps = sequence_length - (temporal_kernel - 1) * 2 * block_num
        blocks = [[common_model["input_dim"]]]
        blocks.extend([[64, 16, 64] for _ in range(block_num)])
        blocks.append([128] if output_steps == 0 else [128, 128])
        blocks.append([horizon])
        model = upstream.STGCN(
            **common_model,
            gso=gso,
            blocks=blocks,
            Kt=temporal_kernel,
            Ks=model_config["spatial_kernel"],
            dropout=model_config["dropout"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler_config = optimizer_config["scheduler"]
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )
        engine_class = upstream.BaseEngine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = 0.0
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.stgcn.STGCN"
        engine_class_name = "src.base.engine.BaseEngine"
        runtime_defaults = {
            "block_channels": [[64, 16, 64]] * block_num,
            "output_channels": [128, 128],
        }
    elif method == "gwnet":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        normalized = upstream.normalize_adj_mx(adjacency_value, "doubletransition")
        supports = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in normalized
        ]
        model_config = effective["model"]
        model = upstream.GWNET(
            **common_model,
            supports=supports,
            adp_adj=model_config["adaptive_adjacency"],
            dropout=model_config["dropout"],
            residual_channels=model_config["init_dim"],
            dilation_channels=model_config["init_dim"],
            skip_channels=model_config["skip_dim"],
            end_channels=model_config["end_dim"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.BaseEngine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.gwnet.GWNET"
        engine_class_name = "src.base.engine.BaseEngine"
        runtime_defaults = {"kernel_size": 2, "blocks": 4, "layers": 2}
    elif method == "astgcn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        adjacency_value = adjacency_value - np.eye(node_num)
        binary_adjacency = np.zeros((node_num, node_num), dtype=np.float32)
        binary_adjacency[adjacency_value != 0] = 1.0
        laplacian = upstream.normalize_adj_mx(binary_adjacency, "scalap")[0]
        model_config = effective["model"]
        chebyshev = upstream.calculate_cheb_poly(
            laplacian, model_config["chebyshev_order"]
        )
        cheb_poly = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in chebyshev
        ]
        model = upstream.ASTGCN(
            **common_model,
            device=device_value,
            cheb_poly=cheb_poly,
            order=model_config["chebyshev_order"],
            nb_block=model_config["blocks"],
            nb_chev_filter=model_config["spatial_filters"],
            nb_time_filter=model_config["temporal_filters"],
            time_stride=model_config["stride"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler_config = optimizer_config["scheduler"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["t_max"],
            eta_min=scheduler_config["eta_min"],
        )
        engine_class = upstream.ASTGCN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.astgcn.ASTGCN"
        engine_class_name = "src.engines.astgcn_engine.ASTGCN_Engine"
        runtime_defaults = {}
    elif method == "sttn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        normalized = upstream.normalize_adj_mx(adjacency_value, "doubletransition")
        supports = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in normalized
        ]
        model_config = effective["model"]
        model = upstream.STTN(
            **common_model,
            device=device_value,
            supports=supports,
            blocks=model_config["blocks"],
            mlp_expand=model_config["mlp_expansion"],
            hidden_channels=model_config["hidden_dim"],
            end_channels=model_config["end_dim"],
            dropout=model_config["dropout"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.BaseEngine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.sttn.STTN"
        engine_class_name = "src.base.engine.BaseEngine"
        runtime_defaults = {"attention_heads": 4, "attention_depth": 1}
    elif method == "dstagnn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        adjacency_value = adjacency_value - np.eye(node_num)
        binary_adjacency = np.zeros((node_num, node_num), dtype=np.float32)
        binary_adjacency[adjacency_value != 0] = 1.0
        laplacian = upstream.normalize_adj_mx(binary_adjacency, "scalap")[0]
        model_config = effective["model"]
        chebyshev = upstream.calculate_cheb_poly(
            laplacian, model_config["chebyshev_order"]
        )
        cheb_poly = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in chebyshev
        ]
        adjacent_pattern = torch.as_tensor(
            binary_adjacency,
            dtype=torch.float32,
            device=device_value,
        )
        model = upstream.DSTAGNN(
            **common_model,
            device=device_value,
            cheb_poly=cheb_poly,
            order=model_config["chebyshev_order"],
            nb_block=model_config["blocks"],
            nb_chev_filter=model_config["spatial_filters"],
            nb_time_filter=model_config["temporal_filters"],
            time_stride=1,
            adj_pa=adjacent_pattern,
            d_model=model_config["d_model"],
            d_k=model_config["d_k"],
            d_v=model_config["d_k"],
            n_head=model_config["heads"],
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.DSTAGNN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = 0.0
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.dstagnn.DSTAGNN"
        engine_class_name = "src.engines.dstagnn_engine.DSTAGNN_Engine"
        runtime_defaults = {"time_stride": 1, "d_v": model_config["d_k"]}
    elif method == "dgcrn":
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        normalized = upstream.normalize_adj_mx(adjacency_value, "doubletransition")
        supports = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in normalized
        ]
        model_config = effective["model"]
        model = upstream.DGCRN(
            **common_model,
            device=device_value,
            predefined_adj=supports,
            gcn_depth=model_config["depth"],
            rnn_size=model_config["rnn_units"],
            hyperGNN_dim=model_config["hyper_dim"],
            node_dim=model_config["node_dim"],
            middle_dim=2,
            list_weight=[0.05, 0.95, 0.95],
            tpd=effective["input_contract"]["time_slots_per_day"],
            tanhalpha=model_config["tanh_alpha"],
            cl_decay_step=2000,
            dropout=0.3,
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
        scheduler = None
        engine_class = upstream.DGCRN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.dgcrn.DGCRN"
        engine_class_name = "src.engines.dgcrn_engine.DGCRN_Engine"
        runtime_defaults = {
            "middle_dim": 2,
            "list_weight": [0.05, 0.95, 0.95],
            "cl_decay_step": 2000,
            "dropout": 0.3,
        }
    else:
        adjacency_value = _coerce_adjacency(adjacency, node_num)
        normalized = upstream.normalize_adj_mx(adjacency_value, "doubletransition")
        supports = [
            torch.as_tensor(
                np.asarray(value),
                dtype=torch.float32,
                device=device_value,
            )
            for value in normalized
        ]
        model_config = effective["model"]
        input_contract = effective["input_contract"]
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
            "dropout": 0.1,
            "adjs": supports,
            "seq_len": common_model["seq_len"],
        }
        model = upstream.D2STGNN(
            **common_model,
            model_args=model_args,
        )
        optimizer_config = effective["optimizer"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
            eps=optimizer_config["epsilon"],
        )
        scheduler_config = optimizer_config["scheduler"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config["gamma"],
        )
        engine_class = upstream.D2STGNN_Engine
        learning_rate = optimizer_config["learning_rate"]
        clip_grad_value = optimizer_config["gradient_clip"]
        max_epochs = effective["max_epochs"]
        patience = effective["patience"]
        model_class_name = "src.models.d2stgnn.D2STGNN"
        engine_class_name = "src.engines.d2stgnn_engine.D2STGNN_Engine"
        runtime_defaults = {"dropout": 0.1}

    engine_extra: dict[str, Any] = {}
    if method == "dgcrn":
        engine_extra = {
            "step_size": effective["model"]["curriculum_step"],
            "horizon": common_model["horizon"],
        }
    elif method == "d2stgnn":
        train_loader = dataloader["train_loader"]
        num_batch = getattr(train_loader, "num_batch", None)
        if (
            isinstance(num_batch, bool)
            or not isinstance(num_batch, int)
            or num_batch <= 0
        ):
            raise ValueError(
                "d2stgnn train_loader.num_batch must be a positive integer"
            )
        engine_extra = {
            "cl_step": (effective["model"]["curriculum_interval"] * num_batch),
            "warm_step": effective["model"]["warm_epochs"] * num_batch,
            "horizon": common_model["horizon"],
        }
    runtime_contract = {
        "torch_num_threads": 3,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
    }
    graph_input_methods = {
        "dcrnn",
        "stgcn",
        "gwnet",
        "astgcn",
        "sttn",
        "dstagnn",
        "dgcrn",
        "d2stgnn",
    }
    adjacency_input_sha256 = (
        _adjacency_sha256(adjacency, node_num)
        if method in graph_input_methods
        else None
    )
    profile = profile_set["profiles"][method]
    factory_config = {
        "schema_version": 1,
        "method": method,
        "source": {
            "source_id": LARGEST_SOURCE_ID,
            "commit": source["commit"],
            "tree": source["tree"],
            "lock_digest": source["lock_digest"],
        },
        "profile": {
            "profile_id": profile["profile_id"],
            "profile_set_sha256": profile_set["profile_set_sha256"],
            "profile_config_hash": profile_set["config_hashes"][method],
            "effective": copy.deepcopy(effective),
        },
        "shared": {
            "sequence": copy.deepcopy(profile_set["shared"]["sequence"]),
            "cuda_deterministic": profile_set["shared"]["cuda_deterministic"],
        },
        "node_num": node_num,
        "device": str(device_value),
        "adjacency_input_sha256": adjacency_input_sha256,
        "runtime_defaults": copy.deepcopy(runtime_defaults),
        "runtime_contract": copy.deepcopy(runtime_contract),
        "engine_derived": copy.deepcopy(engine_extra),
        "compatibility_patches_applied": list(compatibility_patches),
    }
    factory_config_hash = _canonical_sha256(factory_config)
    engine = engine_class(
        device=device_value,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=upstream.masked_mae,
        lrate=learning_rate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_value=clip_grad_value,
        max_epochs=max_epochs,
        patience=patience,
        log_dir=str(output_path),
        logger=logger,
        seed=effective["seed"],
        **engine_extra,
    )
    provenance = {
        "factory_schema_version": 1,
        "source_id": LARGEST_SOURCE_ID,
        "commit": source["commit"],
        "tree": source["tree"],
        "checkout": source["checkout"],
        "lock_digest": source["lock_digest"],
        "source_verification_status": source["status"],
        "profile_id": profile["profile_id"],
        "profile_set_sha256": profile_set["profile_set_sha256"],
        "profile_set_canonical_sha256": profile_set["profile_set_canonical_sha256"],
        "profile_config_hash": profile_set["config_hashes"][method],
        "factory_config_hash": factory_config_hash,
        "seed_applied": effective["seed"],
        "runtime_contract": runtime_contract,
        "upstream_entrypoint": f"experiments/{method}/main.py",
        "model_class": model_class_name,
        "engine_class": engine_class_name,
        "compatibility_patches_applied_by_factory": compatibility_patches,
        "runtime_defaults": runtime_defaults,
    }
    return LargeSTFactoryResult(
        model=model,
        engine=engine,
        optimizer=optimizer,
        scheduler=scheduler,
        effective_config=effective,
        factory_config=factory_config,
        provenance=provenance,
    )
