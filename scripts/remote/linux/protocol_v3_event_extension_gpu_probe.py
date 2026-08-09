#!/usr/bin/env python3
"""Measure v3 extension batch loading and one real optimization step."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dist_net.data import RegionData
from scripts.protocol_v3_event_extension_data import (
    fit_weekly_climatology,
    load_aligned_sensor_support,
    prepare_ndr_batch,
)
from scripts.reproduction.igstgnn_table4.ndr_extension_runner import (
    backward_ndr_batch,
)
from scripts.train_protocol_v3_event_extension import (
    FileLogger,
    METHODS,
    REGION_TO_SUPPORT,
    _backward_standard_batch,
    _build_model,
    _load_ndr_profile,
    _loaders,
    _set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--region", required=True, choices=tuple(REGION_TO_SUPPORT))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--node-order", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-microbatch-size", type=int, default=0)
    parser.add_argument("--igstgnn-data-parallel", action="store_true")
    parser.add_argument(
        "--traffic-loading", choices=("eager", "lazy"), required=True
    )
    parser.add_argument("--loader-batches", type=int, default=20)
    return parser.parse_args()


def estimate_training_seconds(
    *,
    batch_build_seconds: float,
    optimization_seconds: float,
    train_batches: int,
    epochs: int,
) -> float:
    if batch_build_seconds < 0 or optimization_seconds <= 0:
        raise ValueError("probe timings must be nonnegative and optimization positive")
    if type(train_batches) is not int or train_batches <= 0:
        raise ValueError("train_batches must be positive")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be positive")
    return (
        float(batch_build_seconds) + float(optimization_seconds)
    ) * train_batches * epochs


def main() -> int:
    args = parse_args()
    if (
        args.seed != 42
        or args.max_microbatch_size < 0
        or args.loader_batches <= 0
    ):
        raise ValueError("probe seed must be 42 and microbatch must be nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("extension capacity probe requires CUDA")
    _set_seed(args.seed)
    rdata = RegionData(
        args.region,
        args.data_dir,
        args.graph_dir,
        lazy=args.traffic_loading == "lazy",
        require_rel_feat=True,
        include_analysis_labels=False,
        verify_artifact_hashes=True,
    )
    support_dir = args.support_root / REGION_TO_SUPPORT[args.region]
    support = load_aligned_sensor_support(
        sensors_path=support_dir / "sensors.csv",
        node_order_path=args.node_order,
        region_idx=rdata.region_idx,
    )
    statistics = fit_weekly_climatology(rdata)
    loaders = _loaders(
        method=args.method,
        rdata=rdata,
        flow_mean=statistics.flow_mean,
        flow_std=statistics.flow_std,
        sensor_support=support,
        seed=args.seed,
    )
    with tempfile.TemporaryDirectory(prefix="v3-extension-probe-") as directory:
        log_path = Path(directory) / "probe.log"
        with log_path.open("x", encoding="utf-8") as handle:
            logger = FileLogger(handle)
            model, optimizer, _, _, clip, training_engine = _build_model(
                method=args.method,
                rdata=rdata,
                loaders=loaders,
                statistics=statistics,
                sensor_support=support,
                support_dir=support_dir,
                device=device,
                seed=args.seed,
                epochs=30,
                logger=logger,
                output_dir=Path(directory),
                igstgnn_data_parallel=args.igstgnn_data_parallel,
            )
        loaders["train"].shuffle()
        loader_started = time.perf_counter()
        iterator = loaders["train"].get_iterator()
        measured_batches = min(args.loader_batches, loaders["train"].num_batch)
        batch = None
        for _ in range(measured_batches):
            batch = next(iterator)
        batch_build_seconds = (
            time.perf_counter() - loader_started
        ) / measured_batches
        if batch is None:
            raise RuntimeError("loader probe produced no training batch")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
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
            effective = _load_ndr_profile()["effective"]
            loss_args = SimpleNamespace(
                lambda_gate=effective["loss"]["lambda_gate"],
                lambda_spec=effective["loss"]["lambda_specialization"],
            )
            loss, shape, microbatches = backward_ndr_batch(
                model=model,
                tensors=tensors,
                loss_batch=loss_batch,
                loss_args=loss_args,
                device=device,
                max_microbatch_size=(args.max_microbatch_size or loaders["train"].bs),
            )
            valid_count = int(tensors["target_mask"].sum().cpu().item())
        else:
            curriculum_horizon = None
            if training_engine is not None:
                training_engine._update_curriculum_length()
                curriculum_horizon = int(training_engine._cl_len)
            loss_sum, valid_count = _backward_standard_batch(
                model=model,
                method=args.method,
                batch=batch,
                device=device,
                mean=statistics.flow_mean,
                std=statistics.flow_std,
                max_microbatch_size=args.max_microbatch_size,
                curriculum_horizon=curriculum_horizon,
            )
            loss = loss_sum / max(valid_count, 1)
            shape = (
                int(batch["x_data"].shape[0]), 12, int(rdata.N), 1
            )
            microbatches = (
                int(np.ceil(batch["x_data"].shape[0] / args.max_microbatch_size))
                if args.max_microbatch_size
                else 1
            )
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        torch.cuda.synchronize()
        optimization_seconds = time.perf_counter() - started
        estimated_task_seconds = estimate_training_seconds(
            batch_build_seconds=batch_build_seconds,
            optimization_seconds=optimization_seconds,
            train_batches=loaders["train"].num_batch,
            epochs=30,
        )
        result = {
            "status": "PROBE_OK",
            "method": args.method,
            "region": args.region,
            "batch_size": int(batch["x_data"].shape[0]),
            "train_batches": loaders["train"].num_batch,
            "max_microbatch_size": args.max_microbatch_size or int(batch["x_data"].shape[0]),
            "microbatches": microbatches,
            "prediction_shape": list(shape),
            "valid_count": valid_count,
            "loss": float(loss),
            "traffic_loading": args.traffic_loading,
            "loader_batches_measured": measured_batches,
            "batch_build_seconds": batch_build_seconds,
            "optimization_seconds": optimization_seconds,
            "elapsed_seconds": optimization_seconds,
            "estimated_task_seconds": estimated_task_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "visible_gpu_count": torch.cuda.device_count(),
        }
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
