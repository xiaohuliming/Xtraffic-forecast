#!/usr/bin/env python3
"""Train DIST-Net.

Default behavior trains on all 3 regions joint, with region-bucketed batching
(each batch comes from a single region so N stays constant). For quick
verification, use --toy-n-per-region to limit training/val set sizes.

Logs:
  • per-batch losses + monitors → outputs/dist_net/runs/<name>/log.jsonl
  • per-epoch summary → outputs/dist_net/runs/<name>/summary.json
  • checkpoints → outputs/dist_net/runs/<name>/ckpt_*.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import (
    MultiRegionDataset,
    REGION_NAME_TO_CODE,
    collate,
    make_loader,
)
from dist_net.losses import compute_losses
from dist_net.model import DISTNet, DISTNetConfig, count_parameters
from dist_net.monitors import collect_all

REGION_NAME_BY_CODE = {v: k for k, v in REGION_NAME_TO_CODE.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--data-dir", type=Path,
                   default=Path("outputs/dist_net/region_data"))
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/dist_net/runs"))
    p.add_argument("--run-name", default=None,
                   help="Subdir under --out-dir; defaults to a timestamp")
    # data
    p.add_argument("--regions", nargs="+",
                   default=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--toy-n-per-region", type=int, default=0,
                   help=">0: keep only first N (train|val|test) samples per region")
    p.add_argument("--batch-size", type=int, default=8)
    # model
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--d-D", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-enc-layers", type=int, default=2)
    # loss weights
    p.add_argument("--alpha", type=float, default=1.0, help="L_main weight")
    p.add_argument("--beta", type=float, default=0.3, help="L_normal weight")
    p.add_argument("--gamma", type=float, default=0.5, help="L_incident weight")
    p.add_argument("--lam-aff", type=float, default=0.3,
                   help="Down-weight on L_normal at affected nodes")
    # optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup from 0 to --lr over the first N steps")
    # logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1, help="Validate every N epochs")
    # misc
    p.add_argument("--device", default=None, help="cpu | cuda | mps; auto if omitted")
    p.add_argument("--seed", type=int, default=42)
    # blind mode (label-free baseline)
    p.add_argument("--blind", action="store_true",
                   help="Zero incident_feat / incident_mask / rel_feat at both "
                        "training and inference. Tests label-free baseline. "
                        "When set, gamma (L_incident weight) is forced to 0.")
    # strict single-branch (no IncidentBranch / cross-attn / gate)
    p.add_argument("--single-branch", action="store_true",
                   help="Disable IncidentBranch, BidirectionalCrossAttention, AffectedGate. "
                        "Pure NormalBranch + prediction head. beta and gamma forced to 0.")
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def filter_index_table(ds: MultiRegionDataset, n_per_region: int) -> None:
    """In-place: keep only the first n_per_region samples for each region.
    Useful for toy runs."""
    if n_per_region <= 0:
        return
    kept: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for region_name, local_idx in ds.index_table:
        c = seen.get(region_name, 0)
        if c < n_per_region:
            kept.append((region_name, local_idx))
            seen[region_name] = c + 1
    ds.index_table = kept


def device_recursive(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: device_recursive(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(device_recursive(x, device) for x in obj)
    return obj


def edge_index_registry(ds: MultiRegionDataset, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for region_name, rdata in ds.regions.items():
        out[region_name] = torch.from_numpy(rdata.edge_index).to(device)
    return out


def maybe_blind_batch(batch, blind: bool) -> None:
    if not blind:
        return
    batch["incident_feat"] = torch.zeros_like(batch["incident_feat"])
    batch["incident_mask"] = torch.zeros_like(batch["incident_mask"])
    if "rel_feat" in batch:
        batch["rel_feat"] = torch.zeros_like(batch["rel_feat"])


def run_forward_loss(model, batch, edge_index, args):
    maybe_blind_batch(batch, args.blind or args.single_branch)
    gamma_eff = 0.0 if (args.blind or args.single_branch) else args.gamma
    beta_eff = 0.0 if args.single_branch else args.beta
    out = model(batch, edge_index=edge_index)
    losses = compute_losses(
        pred=out["pred"], pred_normal=out["pred_normal_final"],
        y_true=batch["y_true"], y_baseline=batch["y_baseline"],
        y_mask=batch["y_mask"], affected_mask=batch["affected_mask"],
        alpha=args.alpha, beta=beta_eff, gamma=gamma_eff, lam_aff=args.lam_aff,
    )
    return out, losses


@torch.no_grad()
def evaluate(model, val_ds, edge_index_lookup, args, device) -> dict[str, float]:
    model.eval()
    loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    totals = {"L_main": 0.0, "L_normal": 0.0, "L_incident": 0.0, "L_total": 0.0}
    n = 0
    for batch in loader:
        batch = device_recursive(batch, device)
        rc = int(batch["region_code"][0].item())
        edge_index = edge_index_lookup[REGION_NAME_BY_CODE[rc]]
        _out, losses = run_forward_loss(model, batch, edge_index, args)
        bs = batch["x_hist"].shape[0]
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        n += bs
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    if args.blind:
        print("*** BLIND MODE: incident_feat / incident_mask / rel_feat zeroed "
              "at every batch; gamma forced to 0 ***", flush=True)
    if args.single_branch:
        print("*** SINGLE-BRANCH MODE: IncidentBranch / cross-attn / gate "
              "removed; incident inputs zeroed; beta and gamma forced to 0 ***", flush=True)

    # Run dir
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    summary_path = run_dir / "summary.json"
    print(f"run dir: {run_dir}", flush=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2),
                                       encoding="utf-8")

    # Datasets
    t0 = time.time()
    train_ds = MultiRegionDataset(
        region_names=args.regions, data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="train", lazy=False,
    )
    val_ds = MultiRegionDataset(
        region_names=args.regions, data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="val", lazy=False,
    )
    print(f"dataset built in {time.time() - t0:.1f}s; "
          f"|train|={len(train_ds)}  |val|={len(val_ds)}", flush=True)
    if args.toy_n_per_region:
        filter_index_table(train_ds, args.toy_n_per_region)
        filter_index_table(val_ds, args.toy_n_per_region)
        print(f"TOY: per-region cap={args.toy_n_per_region}, "
              f"|train|={len(train_ds)}  |val|={len(val_ds)}")

    edge_index_lookup = edge_index_registry(train_ds, device)

    # Model
    cfg = DISTNetConfig(
        c_x=3,
        c_meta=train_ds.regions[args.regions[0]].C_meta,
        c_e=13, d_t=5,
        n_regions=len(REGION_NAME_TO_CODE),
        hidden_dim=args.hidden_dim,
        t_h=12, t_p=12,
        single_branch=args.single_branch,
    )
    model = DISTNet(cfg).to(device)
    pc = count_parameters(model)
    print("model parameter count:")
    for k, v in pc.items():
        if not k.startswith("_"):
            print(f"  {k:35s} {v:>10,}")
    print(f"  {'TOTAL':35s} {pc['_total']:>10,}", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_batches_per_epoch = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = max(1, args.epochs * n_batches_per_epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 1e-2)

    history: list[dict] = []
    best_val_L_main = float("inf")
    global_step = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loader = make_loader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, seed=args.seed + epoch)
            epoch_start = time.time()
            batch_times = []
            running = {"L_main": 0.0, "L_normal": 0.0, "L_incident": 0.0, "L_total": 0.0}
            running_n = 0

            for bi, batch in enumerate(loader):
                t_start = time.time()
                batch = device_recursive(batch, device)
                rc = int(batch["region_code"][0].item())
                region_name = REGION_NAME_BY_CODE[rc]
                edge_index = edge_index_lookup[region_name]

                out, losses = run_forward_loss(model, batch, edge_index, args)

                optimizer.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()

                bs = batch["x_hist"].shape[0]
                for k in running:
                    running[k] += float(losses[k].item()) * bs
                running_n += bs

                batch_times.append(time.time() - t_start)

                if (bi % args.log_every == 0) or (bi == n_batches_per_epoch - 1):
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "batch_in_epoch": bi,
                        "region": region_name,
                        "N": int(batch["x_hist"].shape[1]),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "L_main": float(losses["L_main"].item()),
                        "L_normal": float(losses["L_normal"].item()),
                        "L_incident": float(losses["L_incident"].item()),
                        "L_total": float(losses["L_total"].item()),
                    }
                    record.update(collect_all(out, batch, model))
                    logf.write(json.dumps(record) + "\n")
                    logf.flush()
                    if bi % (args.log_every * 5) == 0 or bi == n_batches_per_epoch - 1:
                        ms_per_batch = sum(batch_times[-min(50, len(batch_times)):]) / max(
                            1, len(batch_times[-min(50, len(batch_times)):])
                        ) * 1000
                        print(f"ep{epoch:02d} batch {bi:5d}/{n_batches_per_epoch:5d} "
                              f"({region_name:11s} N={int(batch['x_hist'].shape[1]):>4d})  "
                              f"L_main={record['L_main']:.3f} L_normal={record['L_normal']:.3f} "
                              f"L_incident={record['L_incident']:.3f}  lr={record['lr']:.2e}  "
                              f"branch_cos={record['monitor.branch_cosine']:.3f}  "
                              f"g.mean={record['gate.mean']:.3f}  "
                              f"{ms_per_batch:.0f}ms/batch",
                              flush=True)
                global_step += 1

            avg_losses = {k: v / max(running_n, 1) for k, v in running.items()}
            epoch_time = time.time() - epoch_start
            print(f"==> epoch {epoch} train avg: "
                  f"L_main={avg_losses['L_main']:.4f}  "
                  f"L_normal={avg_losses['L_normal']:.4f}  "
                  f"L_incident={avg_losses['L_incident']:.4f}  "
                  f"L_total={avg_losses['L_total']:.4f}  "
                  f"({epoch_time:.1f}s, {epoch_time/max(1,len(batch_times))*1000:.0f}ms/batch)",
                  flush=True)

            ep_summary = {"epoch": epoch, "train_avg": avg_losses,
                          "epoch_time_s": epoch_time}

            if epoch % args.val_every == 0:
                t_val = time.time()
                val_metrics = evaluate(model, val_ds, edge_index_lookup, args, device)
                print(f"    val: L_main={val_metrics['L_main']:.4f}  "
                      f"L_normal={val_metrics['L_normal']:.4f}  "
                      f"L_incident={val_metrics['L_incident']:.4f}  "
                      f"L_total={val_metrics['L_total']:.4f}  "
                      f"({time.time() - t_val:.1f}s)", flush=True)
                ep_summary["val"] = val_metrics
                if val_metrics["L_main"] < best_val_L_main:
                    best_val_L_main = val_metrics["L_main"]
                    ckpt_path = run_dir / "ckpt_best.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "config": vars(args),
                        "model_config": vars(cfg),
                        "val_metrics": val_metrics,
                    }, ckpt_path)
                    print(f"    saved best -> {ckpt_path}", flush=True)

            history.append(ep_summary)
            summary_path.write_text(json.dumps({
                "history": history,
                "best_val_L_main": best_val_L_main,
                "regions": args.regions,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "total_params": pc["_total"],
            }, indent=2), encoding="utf-8")

    final_ckpt = run_dir / "ckpt_final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "config": vars(args),
        "model_config": vars(cfg),
    }, final_ckpt)
    print(f"\ndone. final checkpoint: {final_ckpt}", flush=True)
    print(f"      best val L_main: {best_val_L_main:.4f}", flush=True)


if __name__ == "__main__":
    main()
