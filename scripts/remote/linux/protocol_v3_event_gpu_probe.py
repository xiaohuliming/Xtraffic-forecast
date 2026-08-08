#!/usr/bin/env python3
"""Measure real per-batch cost and peak memory for v3 event baseline models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import Adam  # noqa: E402

from dist_net.data import MultiRegionDataset, RegionData, make_loader  # noqa: E402
from scripts.train_baseline_zoo import (  # noqa: E402
    IN_CH,
    MODELS,
    REGION_TO_GRAPH_KEY,
    backward_effective_masked_mae_batch,
    build_model,
    dense_adj,
    make_input,
    make_label,
    predict_normalized_in_microbatches,
    region_data_kwargs,
    train_flow_stats,
)


class ProbeError(RuntimeError):
    """Raised when a probe cannot produce a trustworthy measurement."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _Namespace(argparse.Namespace):
    pass


def _build_model_args(model: str, seed: int, device: torch.device) -> _Namespace:
    args = _Namespace()
    args.model = model
    args.seed = seed
    args.device = str(device)
    args.epochs = 30
    args.batch_size = 32
    return args


def probe_model(
    *,
    model_name: str,
    region: str,
    data_dir: Path,
    graph_dir: Path,
    device: torch.device,
    batch_size: int,
    microbatch_size: int,
    seed: int,
    warmup_batches: int,
    timed_batches: int,
    timed_val_batches: int,
) -> dict[str, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make(split: str) -> MultiRegionDataset:
        return MultiRegionDataset(
            region_names=[region],
            data_dir=data_dir,
            graph_dir=graph_dir,
            split=split,
            lazy=False,
            region_data_cls=RegionData,
            region_data_kwargs=region_data_kwargs(
                protocol="event",
                split=split,
                stride=1,
                train_frac=0.7,
                val_frac=0.15,
            ),
        )

    train_ds, val_ds = make("train"), make("val")
    rdata = train_ds.regions[region]
    node_count, _, horizon = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    mean, std = train_flow_stats(data_dir / f"{region}_traffic.h5", rdata.train_end)
    adjacency = dense_adj(rdata.edge_index, node_count)

    model_args = _build_model_args(model_name, seed, device)
    model, needs_label, cfg = build_model(
        model_name, node_count, device, adjacency, model_args
    )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    train_loader = make_loader(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
    train_batches = (len(train_ds) + batch_size - 1) // batch_size
    val_batches = (len(val_ds) + batch_size - 1) // batch_size

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    losses: list[float] = []
    iter_cnt = 0
    model.train()
    total_batches = warmup_batches + timed_batches
    for index, batch in enumerate(train_loader):
        if index >= total_batches:
            break
        x = make_input(batch, mean, std, device, IN_CH.get(model_name, 3))
        y_true = batch["y_true"][..., 0].to(device)
        y_mask = batch["y_mask"][..., 0].to(device)
        label = make_label(batch, mean, std, device) if needs_label else None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value = backward_effective_masked_mae_batch(
            model=model,
            name=model_name,
            x=x,
            label=label,
            y_true=y_true,
            y_mask=y_mask,
            iter_cnt=iter_cnt,
            horizon=horizon,
            node_count=node_count,
            device=device,
            mean=mean,
            std=std,
            max_microbatch_size=microbatch_size,
        )
        if cfg["clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        iter_cnt += 1
        if index >= warmup_batches:
            durations.append(elapsed)
            losses.append(float(loss_value))

    if len(durations) < max(1, timed_batches // 2):
        raise ProbeError(
            f"{region}/{model_name}: too few timed train batches to estimate cost"
        )
    if not all(np.isfinite(losses)):
        raise ProbeError(f"{region}/{model_name}: non-finite training loss")

    val_durations: list[float] = []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            if index >= timed_val_batches:
                break
            x = make_input(batch, mean, std, device, IN_CH.get(model_name, 3))
            label = make_label(batch, mean, std, device) if needs_label else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predict_normalized_in_microbatches(
                model,
                model_name,
                x,
                label,
                iter_cnt,
                horizon,
                node_count,
                device,
                microbatch_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            val_durations.append(time.perf_counter() - started)

    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    train_median = statistics.median(durations)
    val_median = statistics.median(val_durations) if val_durations else 0.0
    epoch_seconds = train_median * train_batches + val_median * val_batches
    return {
        "region": region,
        "model": model_name,
        "batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "node_count": node_count,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "timed_train_batches": len(durations),
        "timed_val_batches": len(val_durations),
        "train_batch_seconds_median": round(train_median, 6),
        "val_batch_seconds_median": round(val_median, 6),
        "epoch_seconds_estimate": round(epoch_seconds, 3),
        "first_loss": round(losses[0], 8),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def merge_calibration(
    base: dict[str, object],
    measurements: Sequence[dict[str, object]],
    *,
    calibration_id: str,
    reference_host: str,
) -> dict[str, object]:
    region_scale = dict(base["region_scale"])
    epoch_seconds = dict(base["epoch_seconds"])
    reference_region = str(base.get("reference_region", "Orange"))
    # measured is true only for what this host actually ran. Values carried over
    # from the base calibration stay false no matter how they were obtained
    # there, so a merged file can never overstate its own provenance.
    measured = {name: False for name in epoch_seconds}
    measured_on = {name: base.get("calibration_id") for name in epoch_seconds}
    probed: list[str] = []
    for item in measurements:
        region = str(item["region"])
        if region not in region_scale:
            raise ProbeError(f"probe region {region} has no calibration scale")
        model = str(item["model"])
        scale = float(region_scale[region]) / float(region_scale[reference_region])
        epoch_seconds[model] = round(
            float(item["epoch_seconds_estimate"]) / scale, 3
        )
        measured[model] = True
        measured_on[model] = calibration_id
        probed.append(model)
    inherited = sorted(name for name, flag in measured.items() if not flag)
    return {
        "calibration_version": 1,
        "calibration_id": calibration_id,
        "status": "measured" if not inherited else "partially_measured",
        "reference_host": reference_host,
        "reference_region": reference_region,
        "created_at": _utc_now(),
        "statistic": "median timed train batch extrapolated over one full epoch",
        "note": (
            f"Probed on this host: {sorted(probed)}. Inherited from "
            f"{base.get('calibration_id')} and still estimates: {inherited}."
        ),
        "inherited_from": base.get("calibration_id"),
        "probed_models": sorted(probed),
        "inherited_models": inherited,
        "epoch_seconds": epoch_seconds,
        "measured": measured,
        "measured_on": measured_on,
        "region_scale": region_scale,
        "region_scale_basis": base.get("region_scale_basis"),
        "measurements": list(measurements),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="Orange")
    parser.add_argument("--models", nargs="+", default=["dgcrn", "d2stgnn"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=0,
        help="0 keeps the full effective batch, which is what a 32 GB card should allow",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--timed-batches", type=int, default=10)
    parser.add_argument("--timed-val-batches", type=int, default=5)
    parser.add_argument(
        "--base-calibration",
        type=Path,
        default=(
            REPO_ROOT
            / "config"
            / "protocol_v3"
            / "event_baseline_gpu_calibration_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-id", default="event_baseline_5090_measured_v1")
    parser.add_argument(
        "--reference-host",
        default="RTX 5090 32 GB, native batch 32, one task per card",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.region not in REGION_TO_GRAPH_KEY:
        print(f"PROBE_FAILED unsupported region {args.region}", file=sys.stderr)
        return 2
    unknown = [name for name in args.models if name not in MODELS]
    if unknown:
        print(f"PROBE_FAILED unsupported models {unknown}", file=sys.stderr)
        return 2
    device = torch.device(args.device)
    microbatch = args.microbatch_size or args.batch_size
    if not 1 <= microbatch <= args.batch_size:
        print("PROBE_FAILED invalid microbatch size", file=sys.stderr)
        return 2

    measurements: list[dict[str, object]] = []
    for model_name in args.models:
        try:
            item = probe_model(
                model_name=model_name,
                region=args.region,
                data_dir=args.data_dir,
                graph_dir=args.graph_dir,
                device=device,
                batch_size=args.batch_size,
                microbatch_size=microbatch,
                seed=args.seed,
                warmup_batches=args.warmup_batches,
                timed_batches=args.timed_batches,
                timed_val_batches=args.timed_val_batches,
            )
        except torch.cuda.OutOfMemoryError as exc:
            print(
                f"PROBE_OOM region={args.region} model={model_name} "
                f"batch={args.batch_size} microbatch={microbatch} {exc}",
                file=sys.stderr,
            )
            return 3
        except ProbeError as exc:
            print(f"PROBE_FAILED {exc}", file=sys.stderr)
            return 1
        measurements.append(item)
        print(
            "PROBE_OK "
            f"region={item['region']} model={item['model']} "
            f"train_batch_s={item['train_batch_seconds_median']:.3f} "
            f"epoch_s={item['epoch_seconds_estimate']:.0f} "
            f"peak_allocated_bytes={item['peak_allocated_bytes']}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

    base = json.loads(args.base_calibration.read_text(encoding="utf-8"))
    merged = merge_calibration(
        base,
        measurements,
        calibration_id=args.calibration_id,
        reference_host=args.reference_host,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(f"PROBE_CALIBRATION_WRITTEN {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
