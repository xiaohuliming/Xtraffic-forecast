#!/usr/bin/env python3
# ruff: noqa: E402
"""Publication-grade FDN benchmark trainer for the phase-1 go/no-go matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.graph import bidirectional_random_walk_supports
from dist_net.protocol import build_full_window_index
from fourier_dual_net.model import FourierDualNet
from model import gwnet


PROTOCOL_VERSION = "fdn_benchmark_v1"
ARTIFACT_SCHEMA_VERSION = 1
T_H = 12
T_P = 12
VARIANTS = ("gwn", "dual_gwn", "fdn_fixed_k3", "fdn_learnable_k3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_write(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class BenchmarkData:
    dataset: str
    flow_raw: np.ndarray
    flow_normalized: np.ndarray
    observed_mask: np.ndarray
    edge_index: np.ndarray
    anchors_by_split: Mapping[str, np.ndarray]
    split_by_anchor: Mapping[int, int]
    train_end: int
    val_end: int
    mean: float
    std: float
    data_sha256: str
    adjacency_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.flow_raw.shape[1])


def _load_edges(path: Path, node_count: int) -> np.ndarray:
    pairs: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"from", "to"}.issubset(reader.fieldnames):
            raise ValueError(f"adjacency CSV lacks from/to columns: {path}")
        for row in reader:
            source = int(row["from"])
            target = int(row["to"])
            if not 0 <= source < node_count or not 0 <= target < node_count:
                raise ValueError(f"adjacency endpoint is outside the node range: {row}")
            pairs.add((source, target))
            pairs.add((target, source))
    if not pairs:
        raise ValueError(f"adjacency CSV contains no usable edges: {path}")
    return np.asarray(sorted(pairs), dtype=np.int64).T


def load_benchmark_data(
    dataset: str,
    data_dir: Path,
    *,
    expected_data_sha256: str | None = None,
    expected_adjacency_sha256: str | None = None,
) -> BenchmarkData:
    name = dataset.lower()
    if name not in {"pems04", "pems08"}:
        raise ValueError(f"unsupported phase-1 dataset: {dataset!r}")
    data_path = data_dir / f"{name.upper()}.npz"
    adjacency_path = data_dir / f"{name.upper()}_distance.csv"
    data_sha = sha256_file(data_path)
    adjacency_sha = sha256_file(adjacency_path)
    if expected_data_sha256 is not None and data_sha != expected_data_sha256:
        raise ValueError(f"{name} data SHA-256 mismatch")
    if expected_adjacency_sha256 is not None and adjacency_sha != expected_adjacency_sha256:
        raise ValueError(f"{name} adjacency SHA-256 mismatch")

    with np.load(data_path, allow_pickle=False) as archive:
        if set(archive.files) != {"data"}:
            raise ValueError(f"{data_path} must contain only the data array")
        raw = np.asarray(archive["data"])
    if raw.ndim != 3 or raw.shape[2] < 1 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{data_path} has an invalid traffic tensor")
    flow = np.asarray(raw[..., 0], dtype=np.float32)
    finite = np.isfinite(flow)
    observed = finite & (flow != 0.0)
    if not finite.all():
        flow = np.where(finite, flow, np.float32(0.0))

    total_steps = int(flow.shape[0])
    train_end = int(total_steps * 0.6)
    val_end = train_end + int(total_steps * 0.2)
    training_values = np.asarray(flow[:train_end][observed[:train_end]], dtype=np.float64)
    if training_values.size == 0 or not np.isfinite(training_values).all():
        raise ValueError(f"{name} has no finite observed training values")
    mean = float(training_values.mean(dtype=np.float64))
    std = float(training_values.std(dtype=np.float64))
    if not math.isfinite(std) or std <= 1e-6:
        raise ValueError(f"{name} training standard deviation is invalid")
    normalized = np.zeros_like(flow, dtype=np.float32)
    normalized[observed] = ((flow[observed] - mean) / std).astype(np.float32)

    anchors, split_codes = build_full_window_index(
        total_steps=total_steps,
        history=T_H,
        horizon=T_P,
        stride=1,
        train_end=train_end,
        val_end=val_end,
    )
    anchors_by_split = {
        "train": anchors[split_codes == 0],
        "val": anchors[split_codes == 1],
        "test": anchors[split_codes == 2],
    }
    if any(values.size == 0 for values in anchors_by_split.values()):
        raise ValueError(f"{name} produced an empty split")
    target_sets = {
        key: set(
            np.concatenate([np.arange(int(anchor) + 1, int(anchor) + 1 + T_P) for anchor in values])
        )
        for key, values in anchors_by_split.items()
    }
    if (
        target_sets["train"] & target_sets["val"]
        or target_sets["train"] & target_sets["test"]
        or target_sets["val"] & target_sets["test"]
    ):
        raise RuntimeError(f"{name} target timestamps overlap across splits")

    split_by_anchor = {
        int(anchor): int(code) for anchor, code in zip(anchors.tolist(), split_codes.tolist())
    }
    return BenchmarkData(
        dataset=name,
        flow_raw=flow,
        flow_normalized=normalized,
        observed_mask=observed,
        edge_index=_load_edges(adjacency_path, flow.shape[1]),
        anchors_by_split=anchors_by_split,
        split_by_anchor=split_by_anchor,
        train_end=train_end,
        val_end=val_end,
        mean=mean,
        std=std,
        data_sha256=data_sha,
        adjacency_sha256=adjacency_sha,
    )


def iter_batches(
    data: BenchmarkData,
    split: str,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
    anchors = np.asarray(data.anchors_by_split[split], dtype=np.int64)
    order = np.arange(anchors.size)
    if split == "train":
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(order)
    history_offsets = np.arange(-T_H + 1, 1, dtype=np.int64)
    target_offsets = np.arange(1, T_P + 1, dtype=np.int64)
    yielded = 0
    for start in range(0, order.size, batch_size):
        selected = anchors[order[start : start + batch_size]]
        history_index = selected[:, None] + history_offsets[None, :]
        target_index = selected[:, None] + target_offsets[None, :]
        x = data.flow_normalized[history_index].transpose(0, 2, 1)[..., None]
        y = data.flow_raw[target_index].transpose(0, 2, 1)
        mask = data.observed_mask[target_index].transpose(0, 2, 1)
        if not mask.any():
            continue
        yield (
            torch.from_numpy(np.ascontiguousarray(x)).to(device),
            torch.from_numpy(np.ascontiguousarray(y)).to(device),
            torch.from_numpy(np.ascontiguousarray(mask)).to(device),
            selected.copy(),
        )
        yielded += 1
        if max_batches is not None and yielded >= max_batches:
            break


def _make_gwn(node_count: int, supports: list[torch.Tensor], device: torch.device, nhid: int) -> nn.Module:
    return gwnet(
        device=device,
        num_nodes=node_count,
        dropout=0.3,
        supports=supports,
        gcn_bool=True,
        addaptadj=True,
        in_dim=1,
        out_dim=T_P,
        residual_channels=nhid,
        dilation_channels=nhid,
        skip_channels=nhid * 8,
        end_channels=nhid * 16,
        blocks=4,
        layers=2,
    )


class GWNFamily(nn.Module):
    def __init__(
        self,
        variant: str,
        node_count: int,
        supports: list[torch.Tensor],
        device: torch.device,
        nhid: int,
    ) -> None:
        super().__init__()
        if variant not in {"gwn", "dual_gwn"}:
            raise ValueError(f"invalid GWN family variant: {variant}")
        self.variant = variant
        self.first = _make_gwn(node_count, supports, device, nhid)
        self.second = (
            _make_gwn(node_count, supports, device, nhid)
            if variant == "dual_gwn"
            else None
        )

    @staticmethod
    def _forward_one(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        value = model(x.permute(0, 3, 1, 2).contiguous())
        return value.squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction = self._forward_one(self.first, x)
        if self.second is not None:
            prediction = prediction + self._forward_one(self.second, x)
        return prediction


def build_model(
    variant: str,
    *,
    node_count: int,
    supports: list[torch.Tensor],
    device: torch.device,
    nhid: int,
) -> nn.Module:
    if variant in {"gwn", "dual_gwn"}:
        return GWNFamily(variant, node_count, supports, device, nhid).to(device)
    if variant not in {"fdn_fixed_k3", "fdn_learnable_k3"}:
        raise ValueError(f"unsupported variant: {variant}")
    mode = "fixed_k" if variant == "fdn_fixed_k3" else "learnable"
    return FourierDualNet(
        num_nodes=node_count,
        supports=supports,
        T_h=T_H,
        T_p=T_P,
        K=3,
        decomp_mode=mode,
        in_dim_flow=1,
        nhid=nhid,
        dropout=0.3,
        device=device,
        main_blocks=4,
        main_layers=2,
        pert_blocks=4,
        pert_layers=2,
        use_time_emb=False,
    ).to(device)


@dataclass
class MetricAccumulator:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_percentage_error_sum: float = 0.0
    valid_count: int = 0
    horizon_absolute_error_sum: np.ndarray | None = None
    horizon_squared_error_sum: np.ndarray | None = None
    horizon_absolute_percentage_error_sum: np.ndarray | None = None
    horizon_valid_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.horizon_absolute_error_sum = np.zeros(T_P, dtype=np.float64)
        self.horizon_squared_error_sum = np.zeros(T_P, dtype=np.float64)
        self.horizon_absolute_percentage_error_sum = np.zeros(T_P, dtype=np.float64)
        self.horizon_valid_count = np.zeros(T_P, dtype=np.int64)

    def update(self, prediction: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor) -> None:
        if prediction.shape != actual.shape or mask.shape != actual.shape:
            raise ValueError("prediction, actual, and mask shapes must match")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("prediction contains non-finite values")
        valid = mask.bool()
        if not torch.isfinite(actual[valid]).all():
            raise FloatingPointError("actual contains non-finite valid values")
        error = prediction.double() - actual.double()
        for horizon in range(T_P):
            selected = valid[:, :, horizon]
            count = int(selected.sum().item())
            if count == 0:
                continue
            horizon_error = error[:, :, horizon][selected]
            horizon_actual = actual[:, :, horizon].double()[selected]
            absolute = horizon_error.abs()
            self.horizon_absolute_error_sum[horizon] += float(absolute.sum().item())
            self.horizon_squared_error_sum[horizon] += float(horizon_error.square().sum().item())
            self.horizon_absolute_percentage_error_sum[horizon] += float(
                (absolute / horizon_actual.abs().clamp_min(1e-6)).sum().item()
            )
            self.horizon_valid_count[horizon] += count
            self.absolute_error_sum += float(absolute.sum().item())
            self.squared_error_sum += float(horizon_error.square().sum().item())
            self.absolute_percentage_error_sum += float(
                (absolute / horizon_actual.abs().clamp_min(1e-6)).sum().item()
            )
            self.valid_count += count

    def compute(self) -> dict[str, object]:
        if self.valid_count <= 0 or np.any(self.horizon_valid_count <= 0):
            raise ValueError("metric accumulator has an empty evaluation scope")
        result: dict[str, object] = {
            "mae": self.absolute_error_sum / self.valid_count,
            "rmse": math.sqrt(self.squared_error_sum / self.valid_count),
            "mape_percent": 100.0 * self.absolute_percentage_error_sum / self.valid_count,
            "valid_count": int(self.valid_count),
        }
        for horizon in (2, 5, 11):
            label = f"h{horizon + 1}"
            count = int(self.horizon_valid_count[horizon])
            result[label] = {
                "mae": float(self.horizon_absolute_error_sum[horizon] / count),
                "rmse": float(math.sqrt(self.horizon_squared_error_sum[horizon] / count)),
                "mape_percent": float(
                    100.0 * self.horizon_absolute_percentage_error_sum[horizon] / count
                ),
                "valid_count": count,
            }
        return result


def evaluate(
    model: nn.Module,
    data: BenchmarkData,
    split: str,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    capture: bool,
    max_batches: int | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    was_training = model.training
    model.eval()
    accumulator = MetricAccumulator()
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, mask, selected in iter_batches(
            data,
            split,
            batch_size=batch_size,
            seed=seed,
            epoch=0,
            device=device,
            max_batches=max_batches,
        ):
            prediction = model(x) * data.std + data.mean
            accumulator.update(prediction, y, mask)
            if capture:
                predictions.append(prediction.detach().cpu().numpy().astype(np.float32))
                actuals.append(y.detach().cpu().numpy().astype(np.float32))
                masks.append(mask.detach().cpu().numpy().astype(np.bool_))
                anchors.append(selected.astype(np.int64))
    if was_training:
        model.train()
    artifact = None
    if capture:
        artifact = {
            "prediction": np.concatenate(predictions),
            "actual": np.concatenate(actuals),
            "mask": np.concatenate(masks),
            "forecast_origin": np.concatenate(anchors),
        }
    return accumulator.compute(), artifact


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pems04", "pems08"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-id")
    parser.add_argument("--code-sha")
    parser.add_argument("--config-hash")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-adjacency-sha256")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.nhid <= 0:
        raise ValueError("epochs, batch size, and hidden width must be positive")
    if args.seed not in {42, 1, 2}:
        raise ValueError("phase-1 seeds are fixed to 42, 1, and 2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.probe_only:
        required = {
            "output_dir": args.output_dir,
            "task_id": args.task_id,
            "code_sha": args.code_sha,
            "config_hash": args.config_hash,
            "expected_data_sha256": args.expected_data_sha256,
            "expected_adjacency_sha256": args.expected_adjacency_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("formal training is missing: " + ", ".join(missing))
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            if not output_dir.is_dir() or next(output_dir.iterdir(), None) is not None:
                raise FileExistsError(f"formal output directory is not fresh: {output_dir}")
        else:
            output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = None

    _set_seed(args.seed)
    data = load_benchmark_data(
        args.dataset,
        args.data_dir.resolve(),
        expected_data_sha256=args.expected_data_sha256,
        expected_adjacency_sha256=args.expected_adjacency_sha256,
    )
    supports = bidirectional_random_walk_supports(data.edge_index, data.node_count, device)
    model = build_model(
        args.variant,
        node_count=data.node_count,
        supports=supports,
        device=device,
        nhid=args.nhid,
    )
    parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(data.anchors_by_split["train"].size / args.batch_size)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * batches_per_epoch),
        eta_min=args.learning_rate * 1e-2,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.probe_only:
        model.train()
        x, y, mask, _ = next(
            iter_batches(
                data,
                "train",
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=1,
                device=device,
                max_batches=1,
            )
        )
        prediction = model(x) * data.std + data.mean
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("probe prediction is non-finite")
        loss = torch.abs(prediction[mask] - y[mask]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        print(
            json.dumps(
                {
                    "status": "probe_ok",
                    "dataset": args.dataset,
                    "variant": args.variant,
                    "seed": args.seed,
                    "loss": float(loss.detach().cpu().item()),
                    "parameters": parameter_count,
                    "peak_memory_bytes": peak,
                    "train_windows": int(data.anchors_by_split["train"].size),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    assert output_dir is not None
    config = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "task_id": args.task_id,
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "nhid": args.nhid,
        "device": args.device,
        "history_steps": T_H,
        "horizon_steps": T_P,
        "split": {"train_end": data.train_end, "val_end": data.val_end},
        "normalization": {
            "fit_scope": "unique_raw_training_timestamps_observed_nonzero_only",
            "missing_input_fill_after_normalization": 0.0,
            "mean": data.mean,
            "std": data.std,
        },
        "data_sha256": data.data_sha256,
        "adjacency_sha256": data.adjacency_sha256,
        "code_sha": args.code_sha,
        "config_hash": args.config_hash,
    }
    strict_json_write(output_dir / "config.json", config)
    diagnostics_path = output_dir / "training_diagnostics.jsonl"
    best_val = math.inf
    best_epoch = 0
    started = time.perf_counter()
    with diagnostics_path.open("x", encoding="utf-8") as diagnostics:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            train_error_sum = 0.0
            train_valid_count = 0
            for x, y, mask, _ in iter_batches(
                data,
                "train",
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
                device=device,
            ):
                prediction = model(x) * data.std + data.mean
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("training prediction is non-finite")
                valid_count = int(mask.sum().item())
                if valid_count == 0:
                    continue
                absolute_error = torch.abs(prediction[mask] - y[mask])
                loss = absolute_error.sum() / valid_count
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                train_error_sum += float(absolute_error.detach().double().sum().cpu().item())
                train_valid_count += valid_count
            if train_valid_count == 0:
                raise RuntimeError("training epoch has zero valid observations")
            validation, _ = evaluate(
                model,
                data,
                "val",
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
                capture=False,
            )
            val_mae = float(validation["mae"])
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                checkpoint = {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "task_id": args.task_id,
                    "code_sha": args.code_sha,
                    "config_hash": args.config_hash,
                    "epoch": epoch,
                    "best_val_mae": best_val,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    },
                }
                temporary = output_dir / ".ckpt.tmp"
                torch.save(checkpoint, temporary)
                os.replace(temporary, output_dir / "ckpt_best.pt")
            record = {
                "epoch": epoch,
                "train_mae": train_error_sum / train_valid_count,
                "val_mae": val_mae,
                "best_epoch": best_epoch,
                "best_val_mae": best_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "time_seconds": time.perf_counter() - epoch_started,
            }
            diagnostics.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            diagnostics.flush()
            print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint = torch.load(output_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    if checkpoint["task_id"] != args.task_id or checkpoint["config_hash"] != args.config_hash:
        raise RuntimeError("best checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, prediction_artifact = evaluate(
        model,
        data,
        "test",
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        capture=True,
    )
    assert prediction_artifact is not None
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        prediction=prediction_artifact["prediction"],
        actual=prediction_artifact["actual"],
        mask=prediction_artifact["mask"],
        forecast_origin=prediction_artifact["forecast_origin"],
    )
    decomp_mask = None
    if isinstance(model, FourierDualNet):
        decomp_mask = [float(value) for value in model.decomp.get_mask().detach().cpu().tolist()]
        strict_json_write(output_dir / "decomp_mask.json", {"mask": decomp_mask})
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "task_id": args.task_id,
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "code_sha": args.code_sha,
        "config_hash": args.config_hash,
        "data_sha256": data.data_sha256,
        "adjacency_sha256": data.adjacency_sha256,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "epoch_cap": args.epochs,
        "test_metrics": test_metrics,
        "test_windows": int(data.anchors_by_split["test"].size),
        "train_windows": int(data.anchors_by_split["train"].size),
        "val_windows": int(data.anchors_by_split["val"].size),
        "training_time_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
        "decomp_mask": decomp_mask,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    strict_json_write(output_dir / "summary.json", summary)
    print(json.dumps({"status": "completed", "summary": summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
