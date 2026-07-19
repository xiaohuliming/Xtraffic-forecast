#!/usr/bin/env python3
"""STAEformer +/- ICSF label-injection (decisive experiment 2).

Tests whether incident labels help the STRONGEST label-free model. ICSF is the
same faithful-in-spirit port of IGSTGNN's IncidentsIcsfModule used for the GWN
collision experiment (scripts/train_gwn_icsf.py), here injected into STAEformer's
normalized-flow channel at the last history timestep (input space). Event-anchored
protocol (labels exist only there). Same masked-MAE / artifact schema as
train_staeformer_xtraffic.py so STAEformer vs STAEformer+ICSF is a clean A/B.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "STAEformer"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, gather_primary_incident, make_loader
from dist_net.metrics import MaskedMAEAccumulator, masked_mae_loss
from dist_net.training_runtime import TrainingRunContext
from STAEformer import STAEformer

SIGMA_EUC, SIGMA_ROAD, CUT_EUC, CUT_ROAD = 1.5, 1.5, 4.0, 5.0


def build_x(batch, mu, sd, device):
    """-> (B, T_h, N, 3) = [z(flow), tod/288, dow0..6]."""
    x_hist = batch["x_hist"].to(device)              # (B,N,T_h,3)
    tf = batch["time_feat"].to(device)               # (B,T_h,2) = tod/288, dow/7
    B, N, T_h, _ = x_hist.shape
    flow = (x_hist[..., 0] - mu) / sd                # (B,N,T_h)
    flow = flow.permute(0, 2, 1)                     # (B,T_h,N)
    tod = tf[:, :, 0].unsqueeze(-1).expand(B, T_h, N)
    dow = (tf[:, :, 1] * 7.0).round().clamp(0, 6).unsqueeze(-1).expand(B, T_h, N)
    return torch.stack([flow, tod, dow], dim=-1)     # (B,T_h,N,3)


def primary_incident(batch, device):
    """Return the anchor incident's type, elapsed time and distances."""
    feat, rel_p, has_inc = gather_primary_incident(batch, device)
    B = feat.size(0)
    ar = torch.arange(B, device=device)

    type_idx = feat[:, :8].argmax(dim=1)
    elapsed = feat[:, 8:9]
    log_euc, log_road, up_down = rel_p[..., 0], rel_p[..., 1], rel_p[..., 2]

    def gk(d, sigma, cut):
        v = torch.exp(-(d ** 2) / (2.0 * sigma ** 2))
        v = torch.where(d > cut, torch.zeros_like(v), v)
        v = torch.where(d <= 1e-6, torch.zeros_like(v), v)
        return v

    euc_k = gk(log_euc, SIGMA_EUC, CUT_EUC)
    road_k = gk(log_road, SIGMA_ROAD, CUT_ROAD)
    down = (up_down > 0.5).float()
    dist = torch.stack([euc_k, road_k, down], dim=-1)     # (B, N, 3)
    closest = (log_euc + (log_euc <= 1e-6).float() * 1e9).argmin(dim=1)
    dist[ar, closest] = 0.0
    dist = dist * has_inc[:, None, None]
    return type_idx, elapsed, dist


class ICSFLite(nn.Module):
    """Port of IncidentsIcsfModule. c_in=1: operates on STAEformer's flow channel."""

    def __init__(self, c_in=1, d=32):
        super().__init__()
        self.type_emb = nn.Embedding(8, 8)
        self.dur_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 4))
        self.incident_fusion = nn.Sequential(nn.Linear(8 + 4 + 2, 64), nn.ReLU(), nn.Linear(64, d))
        self.distance_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, d))
        self.q_proj = nn.Linear(c_in, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * d, 64), nn.ReLU(),
                                        nn.Dropout(0.1), nn.Linear(64, d))
        self.in_proj = nn.Linear(d, c_in)

    def forward(self, x_last, type_idx, elapsed, dist, time_last):
        # x_last (B,N,c_in), dist (B,N,3), time_last (B,2)
        emb = torch.cat([self.type_emb(type_idx), self.dur_mlp(elapsed), time_last], dim=-1)
        inc = self.incident_fusion(emb)                            # (B, d)
        dmask = (dist.sum(-1) > 0).float().unsqueeze(-1)           # (B, N, 1)
        dist_attn = F.softmax(self.distance_encoder(dist), dim=1)  # (B, N, d)
        K = self.k_proj(inc).unsqueeze(1) * dmask
        V = self.v_proj(inc).unsqueeze(1).expand(-1, x_last.size(1), -1)
        Q = self.q_proj(x_last)
        a1 = F.softmax((Q * K).masked_fill(dmask == 0, -1e7), dim=1)
        attn = F.softmax(self.fusion_mlp(torch.cat([a1, dist_attn], -1))
                         .masked_fill(dmask == 0, -1e7), dim=1)
        return attn * V * dmask                                    # (B, N, d)


class STAEformerICSF(nn.Module):
    def __init__(self, N, T_h, T_p, use_icsf=True, d_icsf=32):
        super().__init__()
        self.use_icsf = use_icsf
        self.backbone = STAEformer(num_nodes=N, in_steps=T_h, out_steps=T_p,
                                   steps_per_day=288, input_dim=1, output_dim=1)
        if use_icsf:
            self.icsf = ICSFLite(c_in=1, d=d_icsf)

    def forward(self, x, type_idx, elapsed, dist, time_last):
        # x (B,T_h,N,3) = [z(flow), tod, dow]; inject into flow channel last step
        if self.use_icsf:
            flow_last = x[:, -1, :, 0:1]                           # (B,N,1)
            effect = self.icsf(flow_last, type_idx, elapsed, dist, time_last)
            x = x.clone()
            x[:, -1, :, 0:1] = x[:, -1, :, 0:1] + self.icsf.in_proj(effect)
        return self.backbone(x)                                   # (B,T_p,N,1)


def masked_mae(pred, target, mask):
    return masked_mae_loss(pred, target, mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data_v3_noleak_dedup")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/protocol_v3_noleak_dedup/baselines/staeformer_icsf"),
    )
    p.add_argument("--use_icsf", action="store_true")
    p.add_argument("--d_icsf", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "icsf" if args.use_icsf else "base"
    out_dir = args.out_dir / args.region / tag
    run_context = TrainingRunContext.prepare(
        out_dir,
        data_dir=args.data_dir,
        region=args.region,
        model="staeformer_icsf",
        variant=tag,
        seed=args.seed,
        sampling_protocol="event",
    )
    out_dir = run_context.output_dir
    print(f"device: {device}  region: {args.region}  seed: {args.seed}  use_icsf: {args.use_icsf}",
          flush=True)

    rel_kwargs = {"require_rel_feat": True}
    train_ds = MultiRegionDataset(
        [args.region],
        args.data_dir,
        args.graph_dir,
        split="train",
        region_data_kwargs=rel_kwargs,
    )
    val_ds = MultiRegionDataset(
        [args.region],
        args.data_dir,
        args.graph_dir,
        split="val",
        region_data_kwargs=rel_kwargs,
    )
    test_ds = MultiRegionDataset(
        [args.region],
        args.data_dir,
        args.graph_dir,
        split="test",
        region_data_kwargs={
            "require_rel_feat": True,
            "include_analysis_labels": True,
        },
    )
    rdata = train_ds.regions[args.region]
    run_context.verify_region_cache(rdata)
    if not rdata.has_rel_feat:
        raise RuntimeError(
            "ICSF requires the v3 event relation cache; "
            "run scripts/build_event_rel_feat.py first"
        )
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)

    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    hi = rdata.train_end
    seg = flows[:hi]; segm = fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f}", flush=True)

    model = STAEformerICSF(N, T_h, T_p, use_icsf=args.use_icsf, d_icsf=args.d_icsf).to(device)
    print(f"params: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def fwd(batch):
        x = build_x(batch, mu, sd, device)
        type_idx, elapsed, dist = primary_incident(batch, device)
        time_last = batch["time_feat"][:, -1, :].to(device)       # (B,2)
        pred = model(x, type_idx, elapsed, dist, time_last)      # (B,T_p,N,1)
        return pred.squeeze(-1).permute(0, 2, 1) * sd + mu, x.size(0)  # (B,N,T_p)

    def ev(loader):
        model.eval()
        metric = MaskedMAEAccumulator()
        with torch.no_grad():
            for batch in loader:
                pred, _bs = fwd(batch)
                y = batch["y_true"][..., 0].to(device)
                m = batch["y_mask"][..., 0].to(device)
                metric.update(pred, y, m)
        model.train()
        return metric.compute()

    best = float("inf")
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        for bi, batch in enumerate(train_loader):
            pred, bs = fwd(batch)
            y = batch["y_true"][..., 0].to(device)
            m = batch["y_mask"][..., 0].to(device)
            loss = masked_mae(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * bs
            n += bs
            if bi % 300 == 0:
                print(f"  ep{ep:02d} batch {bi}/{nb} L={loss.item():.3f}", flush=True)
        v = ev(val_loader)
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f}  val L={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "mu": mu, "sd": sd, "N": N, "T_h": T_h, "T_p": T_p}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)
        else:
            no_improve += 1
            if args.patience and no_improve >= args.patience:
                print(f"    early stop: no val improvement for {args.patience} epochs", flush=True)
                break

    print(f"\nbest val: {best:.4f}\n=== test inference ===", flush=True)
    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"])
    model.eval()
    S = len(test_ds)
    pred_flow = np.empty((S, T_p, N), dtype=np.float32)
    actual_flow = np.empty((S, T_p, N), dtype=np.float32)
    y_mask_flow = np.empty((S, T_p, N), dtype=bool)
    affected = np.empty((S, N), dtype=bool)
    sample_start = np.empty((S,), dtype=np.int64)
    region_code = np.empty((S,), dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            pred, bs = fwd(batch)
            pred = pred.permute(0, 2, 1).cpu().numpy()            # (B,T_p,N)
            pred_flow[cursor:cursor + bs] = pred
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            sample_start[cursor:cursor + bs] = batch["sample_start"].numpy()
            region_code[cursor:cursor + bs] = batch["region_code"].numpy()
            cursor += bs

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
    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best, "seed": args.seed, "use_icsf": args.use_icsf,
           "params": sum(q.numel() for q in model.parameters())}
    res.update(rdata.result_metadata())
    print(f"\ntest MAE: all={res['all']:.3f}  affected={res['affected']:.3f}  "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    run_context.write_summary(
        res,
        valid_count=int(y_mask_flow.sum()),
        required_metrics=("all", "affected", "unaffected"),
    )


if __name__ == "__main__":
    main()
