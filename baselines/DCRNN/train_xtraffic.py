#!/usr/bin/env python3
"""Train and evaluate DCRNN on the project's XTraffic 2023 protocol.

This entry point deliberately reuses ``dist_net.data`` so DCRNN sees exactly the
same windows, temporal split, imputation, target mask, and node order as NDR-Net.
Only historical traffic and the original time-in-day feature enter the model.
Incident features and affected-node labels are never model inputs; affected labels
are used after inference only to report the same grouped metrics as NDR-Net.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.optim import Adam

from baselines.DCRNN.graph import (
    build_bidirectional_supports,
    graph_diagnostics,
    slice_official_adjacency,
)
from baselines.DCRNN.model import DCRNN
from dist_net.data import (
    FullWindowRegionData,
    MultiRegionDataset,
    RegionData,
    make_loader,
)

REGIONS = ("Alameda", "ContraCosta", "Orange")

# Fixed values from the official METR-LA configuration. XTraffic changes only
# the observations, graph, masks, and sample membership; it does not retune the
# DCRNN architecture or optimization recipe.
ORIGINAL_DCRNN_CONFIG = {
    "history": 12,
    "horizon": 12,
    "input_dim": 2,  # traffic value + time in day
    "output_dim": 1,
    "hidden_dim": 64,
    "num_layers": 2,
    "max_diffusion_step": 2,
    "cl_decay_steps": 2000,
    "use_curriculum_learning": True,
    "batch_size": 64,
    "val_batch_size": 64,
    "test_batch_size": 64,
    "optimizer": "adam",
    "base_lr": 0.01,
    "adam_epsilon": 1.0e-3,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "l1_decay": 0.0,
    "epochs": 100,
    "lr_decay_ratio": 0.1,
    "lr_decay_steps": (20, 30, 40, 50),
    "min_lr": 2.0e-6,
    "grad_clip": 5.0,
    "patience": 50,
    "test_every_n_epochs": 10,
    "max_to_keep": 100,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCRNN baseline on the same XTraffic 2023 windows as NDR-Net"
    )
    parser.add_argument("--region", required=True, choices=REGIONS)
    parser.add_argument("--protocol", choices=("event", "full_window"), default="event")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "outputs" / "dist_net" / "region_data"
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=ROOT / "outputs" / "region_graphs"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=ROOT.parent / "data",
        help="Directory containing XTraffic adj_matrix.npy",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "baselines" / "dcrnn"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--lazy", action="store_true", help="Read traffic lazily from HDF5"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="full_window protocol only"
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7, help="full_window only"
    )
    parser.add_argument("--val-frac", type=float, default=0.1, help="full_window only")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch on tiny split subsets; still uses every node in the selected region",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_datasets(args: argparse.Namespace) -> tuple[MultiRegionDataset, ...]:
    if args.protocol == "full_window":
        region_cls = FullWindowRegionData
        region_kwargs = {
            "stride": args.stride,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
        }
    else:
        region_cls = RegionData
        region_kwargs = {}

    def make(split: str) -> MultiRegionDataset:
        return MultiRegionDataset(
            region_names=[args.region],
            data_dir=args.data_dir,
            graph_dir=args.graph_dir,
            split=split,
            lazy=args.lazy,
            region_data_cls=region_cls,
            region_data_kwargs=region_kwargs,
        )

    datasets = (make("train"), make("val"), make("test"))
    if args.smoke:
        batch_size = int(ORIGINAL_DCRNN_CONFIG["batch_size"])
        limits = (
            max(2 * batch_size, 4),
            max(batch_size, 2),
            max(batch_size, 2),
        )
        for dataset, limit in zip(datasets, limits):
            dataset.index_table = dataset.index_table[:limit]
    if any(len(dataset) == 0 for dataset in datasets):
        raise RuntimeError("one or more data splits are empty")
    return datasets


def split_fingerprint(dataset: MultiRegionDataset) -> dict[str, object]:
    """Fingerprint sample anchors so comparison runs can prove window identity."""
    starts = np.asarray(
        [
            dataset.regions[region].sample_start[local]
            for region, local in dataset.index_table
        ],
        dtype=np.int64,
    )
    return {
        "count": int(starts.size),
        "first": int(starts.min()),
        "last": int(starts.max()),
        "sample_start_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
    }


def compute_train_stats(rdata: RegionData) -> dict[str, float]:
    """Match ``x_train[..., 0].mean/std`` from the official data loader.

    Event windows can overlap, so timestamps are weighted by their number of
    appearances in training histories. XTraffic's validity mask is the sole data
    adaptation: missing entries do not contribute to either moment.
    """
    train_starts = rdata.sample_start[rdata.split == 0]
    if train_starts.size == 0:
        raise RuntimeError("no training sample starts found")
    difference = np.zeros(int(rdata.T) + 1, dtype=np.int64)
    history_low = train_starts.astype(np.int64) - int(rdata.T_h) + 1
    history_high = train_starts.astype(np.int64) + 1
    if history_low.min() < 0 or history_high.max() > int(rdata.T):
        raise RuntimeError("training history falls outside the traffic series")
    np.add.at(difference, history_low, 1)
    np.add.at(difference, history_high, -1)
    time_weights = np.cumsum(difference[:-1])

    total = 0.0
    total_square = 0.0
    count = 0.0
    for low in range(0, int(rdata.T), 2048):
        high = min(low + 2048, int(rdata.T))
        weights = time_weights[low:high]
        if not np.any(weights):
            continue
        if rdata.flow_series is not None:
            values = rdata.flow_series[low:high, :, 0]
            mask = rdata.flow_mask[low:high, :, 0].astype(bool)
        else:
            handle = rdata._traffic_handle()
            values = handle["flow_series_imputed"][low:high, :, 0]
            mask = handle["flow_mask"][low:high, :, 0].astype(bool)
        entry_weights = mask.astype(np.float64) * weights[:, None]
        values64 = np.asarray(values, dtype=np.float64)
        count += float(entry_weights.sum())
        total += float((values64 * entry_weights).sum())
        total_square += float((np.square(values64) * entry_weights).sum())

    if count == 0.0:
        raise RuntimeError("no valid training flow values")
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std == 0.0:
        raise RuntimeError("training flow standard deviation is zero")
    return {"mean": float(mean), "std": std}


def build_encoder_input(
    batch: dict[str, torch.Tensor],
    stats: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    """Build the original DCRNN input: standardized flow plus time in day."""
    flow = batch["x_hist"][..., 0:1].to(device=device, dtype=torch.float32)
    flow = ((flow - stats["mean"]) / stats["std"]).permute(0, 2, 1, 3)

    # Official generate_training_data.py appends a single time_in_day scalar and
    # leaves it unstandardized. XTraffic's first time feature is exactly tod/288.
    tod = batch["time_feat"][..., 0:1].to(device=device, dtype=torch.float32)
    tod = tod.unsqueeze(2).expand(-1, -1, flow.shape[2], -1)
    return torch.cat([flow, tod], dim=-1).contiguous()


def build_targets(
    batch: dict[str, torch.Tensor], stats: dict[str, float], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = (
        batch["y_true"][..., 0]
        .permute(0, 2, 1)
        .unsqueeze(-1)
        .to(device=device, dtype=torch.float32)
    )
    mask = (
        batch["y_mask"][..., 0].permute(0, 2, 1).unsqueeze(-1).to(device=device).bool()
    )
    flow_mean = stats["mean"]
    flow_std = stats["std"]
    standardized = (raw - flow_mean) / flow_std
    return raw, standardized, mask


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weight = mask.to(dtype=prediction.dtype)
    return ((prediction - target).abs() * weight).sum() / weight.sum().clamp(min=1.0)


def inverse_flow(prediction: torch.Tensor, stats: dict[str, float]) -> torch.Tensor:
    return prediction * stats["std"] + stats["mean"]


@torch.no_grad()
def evaluate(
    model: DCRNN,
    loader: torch.utils.data.DataLoader,
    stats: dict[str, float],
    device: torch.device,
) -> float:
    model.eval()
    absolute_sum = 0.0
    valid_count = 0
    for batch in loader:
        inputs = build_encoder_input(batch, stats, device)
        raw_target, _standard_target, mask = build_targets(batch, stats, device)
        prediction = inverse_flow(model(inputs), stats)
        absolute_sum += float(((prediction - raw_target).abs() * mask).sum().item())
        valid_count += int(mask.sum().item())
    return absolute_sum / max(valid_count, 1)


def safe_mean(values: np.ndarray, selection: np.ndarray) -> float:
    return float(values[selection].mean()) if selection.any() else float("nan")


def original_learning_rate(epoch: int) -> float:
    """Epoch schedule from the official DCRNN supervisor/configuration."""
    decays = sum(epoch >= step for step in ORIGINAL_DCRNN_CONFIG["lr_decay_steps"])
    learning_rate = ORIGINAL_DCRNN_CONFIG["base_lr"] * (
        ORIGINAL_DCRNN_CONFIG["lr_decay_ratio"] ** decays
    )
    return max(float(ORIGINAL_DCRNN_CONFIG["min_lr"]), float(learning_rate))


@torch.no_grad()
def test_inference(
    model: DCRNN,
    loader: torch.utils.data.DataLoader,
    dataset: MultiRegionDataset,
    rdata: RegionData,
    stats: dict[str, float],
    device: torch.device,
    out_dir: Path,
) -> dict[str, object]:
    model.eval()
    samples, horizon, nodes = len(dataset), int(rdata.T_p), int(rdata.N)
    pred_flow = np.empty((samples, horizon, nodes), dtype=np.float32)
    actual_flow = np.empty_like(pred_flow)
    y_mask_flow = np.empty((samples, horizon, nodes), dtype=bool)
    affected = np.empty((samples, nodes), dtype=bool)
    sample_start = np.empty(samples, dtype=np.int64)
    region_code = np.empty(samples, dtype=np.int64)
    cursor = 0
    for batch in loader:
        inputs = build_encoder_input(batch, stats, device)
        prediction = inverse_flow(model(inputs), stats).squeeze(-1).cpu().numpy()
        batch_size = prediction.shape[0]
        end = cursor + batch_size
        pred_flow[cursor:end] = prediction
        actual_flow[cursor:end] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
        y_mask_flow[cursor:end] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
        affected[cursor:end] = batch["affected_mask"].numpy()
        sample_start[cursor:end] = batch["sample_start"].numpy()
        region_code[cursor:end] = batch["region_code"].numpy()
        cursor = end
    if cursor != samples:
        raise RuntimeError(f"test inference wrote {cursor} samples, expected {samples}")

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        region_code=region_code,
        sample_start=sample_start,
        region_node_idx=rdata.region_idx.astype(np.int64),
        pred_raw_flow=pred_flow,
        actual_future_flow=actual_flow,
        y_mask_flow=y_mask_flow,
        affected_mask=affected,
    )

    error = np.abs(pred_flow - actual_flow)
    valid = y_mask_flow
    affected_3d = np.broadcast_to(affected[:, None, :], valid.shape)
    squared = np.square(pred_flow - actual_flow)
    per_horizon = []
    for step in range(horizon):
        step_valid = valid[:, step]
        per_horizon.append(
            {
                "horizon": step + 1,
                "mae": safe_mean(error[:, step], step_valid),
                "affected_mae": safe_mean(error[:, step], step_valid & affected),
                "unaffected_mae": safe_mean(error[:, step], step_valid & ~affected),
            }
        )
    return {
        "all": safe_mean(error, valid),
        "affected": safe_mean(error, valid & affected_3d),
        "unaffected": safe_mean(error, valid & ~affected_3d),
        "rmse": float(np.sqrt(squared[valid].mean())) if valid.any() else float("nan"),
        "per_horizon": per_horizon,
        "n_test_samples": samples,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    train_ds, val_ds, test_ds = make_datasets(args)
    rdata = train_ds.regions[args.region]
    if (int(rdata.T_h), int(rdata.T_p)) != (
        ORIGINAL_DCRNN_CONFIG["history"],
        ORIGINAL_DCRNN_CONFIG["horizon"],
    ):
        raise RuntimeError(
            "official DCRNN requires 12 history and 12 target steps; "
            f"cache provides {rdata.T_h}/{rdata.T_p}"
        )
    stats = compute_train_stats(rdata)

    adjacency = slice_official_adjacency(
        args.archive_dir / "adj_matrix.npy", rdata.region_idx
    )
    graph_info = graph_diagnostics(adjacency)
    supports = build_bidirectional_supports(adjacency)

    model = DCRNN(
        num_nodes=int(rdata.N),
        input_dim=ORIGINAL_DCRNN_CONFIG["input_dim"],
        output_dim=ORIGINAL_DCRNN_CONFIG["output_dim"],
        horizon=ORIGINAL_DCRNN_CONFIG["horizon"],
        supports=supports,
        hidden_dim=ORIGINAL_DCRNN_CONFIG["hidden_dim"],
        num_layers=ORIGINAL_DCRNN_CONFIG["num_layers"],
        max_diffusion_step=ORIGINAL_DCRNN_CONFIG["max_diffusion_step"],
        cl_decay_steps=ORIGINAL_DCRNN_CONFIG["cl_decay_steps"],
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = (
        args.out_dir
        / args.protocol
        / "official_graph"
        / "original_config"
        / args.region
        / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "graph_dir": str(args.graph_dir),
        "archive_dir": str(args.archive_dir),
        "out_dir": str(args.out_dir),
        "num_nodes": int(rdata.N),
        "implementation": "faithful_pytorch_port_of_official_dcrnn",
        "official_config": ORIGINAL_DCRNN_CONFIG,
        "input_dim": ORIGINAL_DCRNN_CONFIG["input_dim"],
        "history": int(rdata.T_h),
        "horizon": int(rdata.T_p),
        "parameters": parameter_count,
        "train_stats": stats,
        "graph": graph_info,
        "split_fingerprints": {
            "train": split_fingerprint(train_ds),
            "val": split_fingerprint(val_ds),
            "test": split_fingerprint(test_ds),
        },
        "data_contract": {
            "model_inputs": ["x_hist[..., flow]", "time_feat[..., time_in_day]"],
            "training_targets": ["y_true", "y_mask"],
            "evaluation_only": ["affected_mask", "sample_start", "region_code"],
            "forbidden_model_inputs": [
                "incident_feat",
                "incident_mask",
                "affected_mask",
                "x_baseline",
                "y_baseline",
            ],
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"device={device} region={args.region} protocol={args.protocol} "
        "input=flow+time_in_day graph=official_xtraffic",
        flush=True,
    )
    print(
        f"N={rdata.N} train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
        f"params={parameter_count:,} graph_edges={graph_info['num_edges']}",
        flush=True,
    )
    print(f"train stats={stats}", flush=True)

    batch_size = int(ORIGINAL_DCRNN_CONFIG["batch_size"])
    train_loader = make_loader(train_ds, batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size, shuffle=False)
    optimizer = Adam(
        model.parameters(),
        lr=ORIGINAL_DCRNN_CONFIG["base_lr"],
        eps=ORIGINAL_DCRNN_CONFIG["adam_epsilon"],
        weight_decay=ORIGINAL_DCRNN_CONFIG["weight_decay"],
    )
    batches_per_epoch = max(len(train_loader), 1)
    epochs = 1 if args.smoke else int(ORIGINAL_DCRNN_CONFIG["epochs"])

    best_val = float("inf")
    best_epoch = 0
    no_improvement = 0
    batches_seen = 0
    log_path = out_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_mae",
                "val_mae",
                "lr",
                "teacher_forcing",
                "seconds",
            ),
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            start = time.time()
            learning_rate = original_learning_rate(epoch)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            model.train()
            absolute_sum = 0.0
            valid_count = 0
            for batch_index, batch in enumerate(train_loader):
                inputs = build_encoder_input(batch, stats, device)
                raw_target, standard_target, mask = build_targets(batch, stats, device)
                prediction_z = model(
                    inputs,
                    labels=standard_target,
                    labels_mask=mask,
                    batches_seen=batches_seen,
                    use_curriculum_learning=ORIGINAL_DCRNN_CONFIG[
                        "use_curriculum_learning"
                    ],
                )
                prediction = inverse_flow(prediction_z, stats)
                loss = masked_mae(prediction, raw_target, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ORIGINAL_DCRNN_CONFIG["grad_clip"]
                )
                optimizer.step()
                absolute_sum += float(
                    ((prediction - raw_target).abs() * mask).sum().item()
                )
                valid_count += int(mask.sum().item())
                batches_seen += 1
                if batch_index % 100 == 0:
                    print(
                        f"  ep{epoch:02d} batch={batch_index}/{batches_per_epoch} "
                        f"mae={loss.item():.3f} tf={model.sampling_threshold(batches_seen):.3f}",
                        flush=True,
                    )
            train_mae = absolute_sum / max(valid_count, 1)
            val_mae = evaluate(model, val_loader, stats, device)
            elapsed = time.time() - start
            row = {
                "epoch": epoch,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "lr": learning_rate,
                "teacher_forcing": model.sampling_threshold(batches_seen),
                "seconds": elapsed,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"==> ep{epoch:02d} train={train_mae:.4f} val={val_mae:.4f} "
                f"tf={row['teacher_forcing']:.3f} ({elapsed:.0f}s)",
                flush=True,
            )
            if val_mae <= best_val:
                best_val = val_mae
                best_epoch = epoch
                no_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "batches_seen": batches_seen,
                    },
                    out_dir / "ckpt_best.pt",
                )
                print("    saved best", flush=True)
            else:
                no_improvement += 1
                if no_improvement > ORIGINAL_DCRNN_CONFIG["patience"]:
                    print(
                        "    early stop after official patience=50",
                        flush=True,
                    )
                    break

    checkpoint = torch.load(
        out_dir / "ckpt_best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    metrics = test_inference(model, test_loader, test_ds, rdata, stats, device, out_dir)
    summary = {
        **metrics,
        "region": args.region,
        "protocol": args.protocol,
        "graph_source": "official_xtraffic",
        "input_mode": "original_flow_time_in_day",
        "seed": args.seed,
        "parameters": parameter_count,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "comparison_contract": "same XTraffic cache/windows/split/target mask as NDR-Net",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"test MAE all={summary['all']:.3f} affected={summary['affected']:.3f} "
        f"unaffected={summary['unaffected']:.3f}",
        flush=True,
    )
    print(f"artifacts: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
