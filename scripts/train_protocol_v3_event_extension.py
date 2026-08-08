#!/usr/bin/env python3
"""Train BiST, IGSTGNN, or NDR-Net on strict protocol-v3 event samples."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from dist_net.data import RegionData
from dist_net.training_runtime import TrainingRunContext
from scripts.protocol_v3_event_extension_data import (
    V3EventBatchLoader,
    fit_weekly_climatology,
    load_aligned_sensor_support,
    prepare_ndr_batch,
)
from scripts.reproduction.igstgnn_table4.bist_factory import build_bist_factory
from scripts.reproduction.igstgnn_table4.igstgnn_factory import build_igstgnn_factory
from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    _make_model as make_ndr_model,
    backward_ndr_batch,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set
from scripts.reproduction.igstgnn_table4.runtime_data import (
    IncidentStandardScaler,
    TupleLoaderView,
)
from scripts.train_baseline_zoo import flow_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config/reproduction/igstgnn_table4/profiles.public_material_v2.json"
)
NDR_PROFILE_PATH = (
    PROJECT_ROOT
    / "config/reproduction/igstgnn_table4/ndr_extension_profile_v1.json"
)
REGION_TO_SUPPORT = {
    "Alameda": "Alameda",
    "ContraCosta": "Contra_Costa",
    "Orange": "Orange",
}
REGION_CODE = {"Alameda": 0, "ContraCosta": 1, "Orange": 2}
METHODS = ("bist", "igstgnn", "ndr_net")


class FileLogger:
    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def info(self, value: Any, *args: Any) -> None:
        text = str(value) % args if args else str(value)
        self.handle.write(text.rstrip("\n") + "\n")
        self.handle.flush()

    warning = info
    error = info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--region", required=True, choices=tuple(REGION_TO_SUPPORT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-microbatch-size", type=int, default=0)
    parser.add_argument("--igstgnn-data-parallel", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _load_ndr_profile() -> dict[str, Any]:
    value = json.loads(NDR_PROFILE_PATH.read_text(encoding="utf-8"))
    if value.get("profile_id") != "ndr_net_table4_extension_v1":
        raise RuntimeError("NDR profile identity is invalid")
    return value


def _loaders(
    *,
    method: str,
    rdata: RegionData,
    flow_mean: float,
    flow_std: float,
    sensor_support: Any,
    seed: int,
) -> dict[str, V3EventBatchLoader]:
    profile_set = load_profile_set(PROFILE_PATH)
    support_region = REGION_TO_SUPPORT[rdata.region_name]
    if method in {"bist", "igstgnn"}:
        effective = profile_set["profiles"][method]["effective"]
        batch_size = int(effective["batch_size_by_region"][support_region])
        include_incident = method == "igstgnn"
    else:
        effective = _load_ndr_profile()["effective"]
        batch_size = int(effective["batch_size_by_region"][support_region])
        include_incident = False
    result = {}
    for split, code in (("train", 0), ("val", 1), ("test", 2)):
        result[split] = V3EventBatchLoader(
            region_data=rdata,
            indices=np.flatnonzero(rdata.split == code),
            batch_size=batch_size,
            flow_mean=flow_mean,
            flow_std=flow_std,
            sensor_support=sensor_support if method == "igstgnn" else None,
            include_incident=include_incident,
            shuffle=split == "train",
            drop_last=False,
            seed=seed,
        )
    return result


def _adjacency(path: Path, node_count: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"official adjacency is missing: {path}")
    value = np.load(path, allow_pickle=False)
    if (
        value.shape != (node_count, node_count)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
    ):
        raise ValueError("official adjacency is invalid")
    return np.asarray(value, dtype=np.float32)


def _build_model(
    *,
    method: str,
    rdata: RegionData,
    loaders: Mapping[str, V3EventBatchLoader],
    statistics: Any,
    sensor_support: Any,
    support_dir: Path,
    device: torch.device,
    seed: int,
    epochs: int,
    logger: FileLogger,
    output_dir: Path,
    igstgnn_data_parallel: bool,
) -> tuple[Any, Any, Any, dict[str, Any], int | None, Any | None]:
    profile_set = load_profile_set(PROFILE_PATH)
    scaler = IncidentStandardScaler(mean=statistics.flow_mean, std=statistics.flow_std)
    if method == "bist":
        effective = profile_set["profiles"]["bist"]["effective"]
        tuple_loaders = {
            f"{split}_loader": TupleLoaderView(loader)
            for split, loader in loaders.items()
        }
        factory = build_bist_factory(
            profile_effective=effective,
            manifest_frozen_seed=seed,
            node_num=rdata.N,
            device=device,
            dataloader=tuple_loaders,
            scaler=scaler,
            logger=logger,
            output_dir=output_dir,
            max_epochs_override=epochs,
        )
        return (
            factory.model.to(device), factory.optimizer, factory.scheduler,
            factory.provenance, int(effective["optimizer"]["gradient_clip"]), None,
        )
    if method == "igstgnn":
        effective = profile_set["profiles"]["igstgnn"]["effective"]
        dataloader = {f"{split}_loader": loader for split, loader in loaders.items()}
        factory = build_igstgnn_factory(
            profile_effective=effective,
            node_num=rdata.N,
            device=device,
            adjacency=_adjacency(support_dir / "adj_matrix.npy", rdata.N),
            dataloader=dataloader,
            scaler=scaler,
            sensor_dimensions=sensor_support.sensor_dimensions,
            logger=logger,
            output_dir=output_dir,
            dataset=REGION_TO_SUPPORT[rdata.region_name],
            data_path=support_dir,
            data_parallel=igstgnn_data_parallel,
            manifest_frozen_seed=seed,
            max_epochs_override=epochs,
        )
        clip = int(effective["optimizer"]["gradient_clip"])
        return (
            factory.model.to(device), factory.optimizer, factory.scheduler,
            factory.provenance, clip, factory.engine,
        )
    profile = _load_ndr_profile()
    effective = profile["effective"]
    model = make_ndr_model(
        node_count=rdata.N,
        statistics=statistics,
        device=device,
    )
    optimizer_config = effective["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * loaders["train"].num_batch),
        eta_min=optimizer_config["scheduler"]["eta_min"],
    )
    provenance = {
        "source_id": profile["source_id"],
        "profile_id": profile["profile_id"],
        "variant": profile["variant"],
        "effective_overrides": {"seeds": {"to": [seed]}},
    }
    return (
        model, optimizer, scheduler, provenance,
        int(optimizer_config["gradient_clip"]), None,
    )


def _standard_tensors(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key in {
            "x_data", "y_data", "y_mask", "incident_features",
            "incident_position", "incident_distances",
        }
    } | {
        "sensor_data": {
            key: torch.as_tensor(value, device=device)
            for key, value in batch.get("sensor_data", {}).items()
        }
    }


def _forward_standard(
    model: Any,
    method: str,
    tensors: Mapping[str, Any],
) -> torch.Tensor:
    if method == "bist":
        value = model(tensors["x_data"])
    else:
        incident = {
            "incident": tensors["incident_features"].float(),
            "position": tensors["incident_position"].long(),
            "distances": tensors["incident_distances"].float(),
        }
        value = model(
            tensors["x_data"].float(),
            incident_data=incident,
            sensor_data=tensors["sensor_data"],
        )
    if value.ndim != 4 or value.shape[-1] != 1:
        raise RuntimeError(f"{method} returned an invalid prediction shape")
    return value[..., 0]


def _backward_standard_batch(
    *,
    model: Any,
    method: str,
    batch: Mapping[str, Any],
    device: torch.device,
    mean: float,
    std: float,
    max_microbatch_size: int,
    curriculum_horizon: int | None = None,
) -> tuple[float, int]:
    tensors = _standard_tensors(batch, device)
    batch_size = int(tensors["x_data"].shape[0])
    micro = max_microbatch_size or batch_size
    full_mask = tensors["y_mask"][..., 0]
    if curriculum_horizon is not None:
        if not 1 <= curriculum_horizon <= full_mask.shape[1]:
            raise RuntimeError("curriculum horizon is outside the forecast range")
        full_mask = full_mask[:, :curriculum_horizon]
    valid_total = full_mask.sum().clamp(min=1).float()
    error_sum = 0.0
    valid_count = 0
    for start in range(0, batch_size, micro):
        stop = min(start + micro, batch_size)
        chunk = slice(start, stop)
        selected = {
            key: (
                {name: value[chunk] for name, value in item.items()}
                if key == "sensor_data"
                else item[chunk]
            )
            for key, item in tensors.items()
        }
        prediction = _forward_standard(model, method, selected)
        prediction_raw = prediction * float(std) + float(mean)
        target_raw = selected["y_data"][..., 0] * float(std) + float(mean)
        mask = selected["y_mask"][..., 0].bool()
        if curriculum_horizon is not None:
            prediction_raw = prediction_raw[:, :curriculum_horizon]
            target_raw = target_raw[:, :curriculum_horizon]
            mask = mask[:, :curriculum_horizon]
        error = torch.abs(prediction_raw - target_raw)
        (error[mask].sum() / valid_total).backward()
        error_sum += float(error[mask].detach().double().sum().cpu().item())
        valid_count += int(mask.sum().cpu().item())
    return error_sum, valid_count


def _evaluate(
    *,
    model: Any,
    method: str,
    loader: V3EventBatchLoader,
    statistics: Any,
    device: torch.device,
    capture: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    was_training = model.training
    model.eval()
    error_sum = np.float64(0)
    valid_count = 0
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader.get_iterator():
            if method == "ndr_net":
                prepared = prepare_ndr_batch(batch, statistics)
                tensors = {
                    key: torch.from_numpy(value).to(device)
                    for key, value in prepared.items()
                }
                prediction = model(
                    tensors["flow_hist"], tensors["x_baseline"],
                    tensors["y_baseline"], tensors["time_feat"],
                ).permute(0, 2, 1)
                target = tensors["target"].permute(0, 2, 1)
                mask = tensors["target_mask"].permute(0, 2, 1).bool()
            else:
                tensors = _standard_tensors(batch, device)
                prediction = _forward_standard(model, method, tensors)
                prediction = prediction * statistics.flow_std + statistics.flow_mean
                target = tensors["y_data"][..., 0] * statistics.flow_std + statistics.flow_mean
                mask = tensors["y_mask"][..., 0].bool()
            error_sum += np.float64(
                torch.abs(prediction - target)[mask].double().sum().cpu().item()
            )
            valid_count += int(mask.sum().cpu().item())
            if capture:
                predictions.append(prediction.float().cpu().numpy())
                actuals.append(target.float().cpu().numpy())
                masks.append(mask.cpu().numpy())
    if was_training:
        model.train()
    if valid_count <= 0:
        raise RuntimeError("evaluation split has no valid observations")
    return (
        float(error_sum / valid_count),
        np.concatenate(predictions) if capture else None,
        np.concatenate(actuals) if capture else None,
        np.concatenate(masks) if capture else None,
    )


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.epochs != 30 or args.seed != 42:
        raise ValueError("formal v3 extension table is fixed to seed 42 and 30 epochs")
    if args.patience != 20:
        raise ValueError("formal v3 extension patience is fixed to 20")
    if args.max_microbatch_size < 0:
        raise ValueError("max-microbatch-size must be nonnegative")
    device = _device(args.device)
    _set_seed(args.seed)
    output = args.out_dir / args.region
    run_context = TrainingRunContext.prepare(
        output,
        data_dir=args.data_dir,
        region=args.region,
        model=args.method,
        variant="v3_event_aligned",
        seed=args.seed,
        sampling_protocol="event",
    )
    output = run_context.output_dir
    rdata = RegionData(
        args.region,
        args.data_dir,
        args.graph_dir,
        lazy=True,
        require_rel_feat=True,
        include_analysis_labels=False,
        verify_artifact_hashes=True,
    )
    run_context.verify_region_cache(rdata)
    support_dir = args.support_root / REGION_TO_SUPPORT[args.region]
    sensor_support = load_aligned_sensor_support(
        sensors_path=support_dir / "sensors.csv",
        node_order_path=args.node_order,
        region_idx=rdata.region_idx,
    )
    for name in ("desc_mapping.json", "type_mapping.json"):
        path = support_dir / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"IGSTGNN mapping file is missing: {path}")
    statistics = fit_weekly_climatology(rdata)
    loaders = _loaders(
        method=args.method,
        rdata=rdata,
        flow_mean=statistics.flow_mean,
        flow_std=statistics.flow_std,
        sensor_support=sensor_support,
        seed=args.seed,
    )
    log_path = output / "train.log"
    checkpoint = output / "ckpt_best.pt"
    with log_path.open("x", encoding="utf-8") as handle:
        logger = FileLogger(handle)
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            logger.info(
                "method=%s region=%s protocol=protocol_v3_noleak_dedup seed=%d",
                args.method, args.region, args.seed,
            )
            logger.info(
                "train=%d val=%d test=%d train_valid_points=%d",
                loaders["train"].size, loaders["val"].size, loaders["test"].size,
                statistics.valid_training_points,
            )
            (
                model, optimizer, scheduler, model_provenance, gradient_clip,
                training_engine,
            ) = _build_model(
                method=args.method,
                rdata=rdata,
                loaders=loaders,
                statistics=statistics,
                sensor_support=sensor_support,
                support_dir=support_dir,
                device=device,
                seed=args.seed,
                epochs=args.epochs,
                logger=logger,
                output_dir=output,
                igstgnn_data_parallel=args.igstgnn_data_parallel,
            )
            best_val = float("inf")
            best_epoch = 0
            wait = 0
            ndr_profile = _load_ndr_profile()["effective"]
            loss_args = SimpleNamespace(
                lambda_gate=ndr_profile["loss"]["lambda_gate"],
                lambda_spec=ndr_profile["loss"]["lambda_specialization"],
            )
            for epoch in range(1, args.epochs + 1):
                model.train()
                loaders["train"].shuffle()
                error_sum = 0.0
                valid_count = 0
                for batch in loaders["train"].get_iterator():
                    optimizer.zero_grad(set_to_none=True)
                    if args.method == "ndr_net":
                        prepared = prepare_ndr_batch(batch, statistics)
                        tensors = {
                            key: torch.from_numpy(value).to(device)
                            for key, value in prepared.items()
                        }
                        loss_batch = {
                            "y_true": tensors["target"].unsqueeze(-1),
                            "y_mask": tensors["target_mask"].unsqueeze(-1),
                        }
                        batch_loss, _, _ = backward_ndr_batch(
                            model=model,
                            tensors=tensors,
                            loss_batch=loss_batch,
                            loss_args=loss_args,
                            device=device,
                            max_microbatch_size=(
                                args.max_microbatch_size
                                or (8 if args.region == "Orange" else loaders["train"].bs)
                            ),
                        )
                        error_sum += batch_loss
                        valid_count += int(tensors["target_mask"].sum().cpu().item())
                    else:
                        curriculum_horizon = None
                        if training_engine is not None:
                            training_engine._update_curriculum_length()
                            curriculum_horizon = int(training_engine._cl_len)
                        batch_error, batch_valid = _backward_standard_batch(
                            model=model,
                            method=args.method,
                            batch=batch,
                            device=device,
                            mean=statistics.flow_mean,
                            std=statistics.flow_std,
                            max_microbatch_size=args.max_microbatch_size,
                            curriculum_horizon=curriculum_horizon,
                        )
                        error_sum += batch_error
                        valid_count += batch_valid
                    if gradient_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    if args.method == "ndr_net":
                        scheduler.step()
                if args.method != "ndr_net" and scheduler is not None:
                    scheduler.step()
                val_mae, _, _, _ = _evaluate(
                    model=model,
                    method=args.method,
                    loader=loaders["val"],
                    statistics=statistics,
                    device=device,
                )
                logger.info(
                    "epoch=%03d train_objective=%.8f validation_mae=%.8f",
                    epoch, error_sum / max(valid_count, 1), val_mae,
                )
                if val_mae < best_val:
                    best_val = val_mae
                    best_epoch = epoch
                    wait = 0
                    _atomic_checkpoint(
                        checkpoint,
                        {
                            "model_state": model.state_dict(),
                            "best_validation_mae": best_val,
                            "best_epoch": best_epoch,
                            "method": args.method,
                            "region": args.region,
                            "seed": args.seed,
                        },
                    )
                else:
                    wait += 1
                if args.method != "ndr_net" and wait >= args.patience:
                    logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
                    break
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state"], strict=True)
            test_mae, prediction, actual, mask = _evaluate(
                model=model,
                method=args.method,
                loader=loaders["test"],
                statistics=statistics,
                device=device,
                capture=True,
            )
            logger.info("best_epoch=%d best_validation_mae=%.8f test_mae=%.8f", best_epoch, best_val, test_mae)

    assert prediction is not None and actual is not None and mask is not None
    test_indices = np.flatnonzero(rdata.split == 2)
    affected = np.stack(
        [rdata.analysis_labels(int(index))["affected_mask"] for index in test_indices]
    )
    sample_start = rdata.sample_start[test_indices].astype(np.int64, copy=False)
    region_code = np.full(test_indices.size, REGION_CODE[args.region], dtype=np.int64)
    np.savez_compressed(
        output / "test_predictions.npz",
        region_code=region_code,
        sample_start=sample_start,
        region_node_idx=rdata.region_idx.astype(np.int64, copy=False),
        pred_raw_flow=prediction.astype(np.float32, copy=False),
        actual_future_flow=actual.astype(np.float32, copy=False),
        y_mask_flow=mask.astype(np.bool_, copy=False),
        affected_mask=affected.astype(np.bool_, copy=False),
    )
    metrics = flow_metrics(prediction, actual, mask, affected)
    summary = {
        **metrics,
        "best_val_L_main": best_val,
        "best_epoch": best_epoch,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "adapter": {
            "id": "protocol_v3_event_extension_numeric_v1",
            "training_analysis_labels": False,
            "explicit_observation_masks": True,
            "checkpoint_metric": "global_validation_masked_mae_all_valid_points",
            "partial_batch": "retain_for_train_val_test",
            "early_stopping_patience": (
                None if args.method == "ndr_net" else args.patience
            ),
            "sensor_station_id_sha256": sensor_support.station_id_sha256,
            "region_idx_sha256": sensor_support.region_idx_sha256,
            "weekly_climatology_fit": "unique_raw_train_timestamps_only",
            "model_provenance": model_provenance,
            "upstream_curriculum_applied": args.method == "igstgnn",
        },
        **rdata.result_metadata(),
    }
    run_context.write_summary(
        summary,
        valid_count=int(mask.sum()),
        required_metrics=("all",),
    )


if __name__ == "__main__":
    main()
