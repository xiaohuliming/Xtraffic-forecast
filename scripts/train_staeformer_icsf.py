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

from dist_net.data import MultiRegionDataset, make_loader
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
    """type_idx (B,), duration (B,1), distances (B,N,3) for the first active slot."""
    inc_feat = batch["incident_feat"].to(device)          # (B, M, C_e)
    inc_mask = batch["incident_mask"].to(device)          # (B, M)
    rel = batch["rel_feat"].to(device)                    # (B, M, N, 4)
    B = inc_feat.size(0)
    slot = inc_mask.float().argmax(dim=1)
    ar = torch.arange(B, device=device)
    feat = inc_feat[ar, slot]                             # (B, C_e)
    rel_p = rel[ar, slot]                                 # (B, N, 4)
    has_inc = inc_mask.any(dim=1).float()                 # (B,)

    type_idx = feat[:, :8].argmax(dim=1)
    duration = feat[:, 8:9]
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
    return type_idx, duration, dist


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

    def forward(self, x_last, type_idx, duration, dist, time_last):
        # x_last (B,N,c_in), dist (B,N,3), time_last (B,2)
        emb = torch.cat([self.type_emb(type_idx), self.dur_mlp(duration), time_last], dim=-1)
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

    def forward(self, x, type_idx, duration, dist, time_last):
        # x (B,T_h,N,3) = [z(flow), tod, dow]; inject into flow channel last step
        if self.use_icsf:
            flow_last = x[:, -1, :, 0:1]                           # (B,N,1)
            effect = self.icsf(flow_last, type_idx, duration, dist, time_last)
            x = x.clone()
            x[:, -1, :, 0:1] = x[:, -1, :, 0:1] + self.icsf.in_proj(effect)
        return self.backbone(x)                                   # (B,T_p,N,1)


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/baselines/staeformer_icsf"))
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
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  region: {args.region}  seed: {args.seed}  use_icsf: {args.use_icsf}",
          flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)

    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
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
        type_idx, duration, dist = primary_incident(batch, device)
        time_last = batch["time_feat"][:, -1, :].to(device)       # (B,2)
        pred = model(x, type_idx, duration, dist, time_last)      # (B,T_p,N,1)
        return pred.squeeze(-1).permute(0, 2, 1) * sd + mu, x.size(0)  # (B,N,T_p)

    def ev(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                pred, bs = fwd(batch)
                y = batch["y_true"][..., 0].to(device)
                m = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(pred, y, m).item()) * bs
                n += bs
        model.train()
        return tot / max(n, 1)

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
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            pred, bs = fwd(batch)
            pred = pred.permute(0, 2, 1).cpu().numpy()            # (B,T_p,N)
            pred_flow[cursor:cursor + bs] = pred
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            cursor += bs

    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best, "seed": args.seed, "use_icsf": args.use_icsf,
           "params": sum(q.numel() for q in model.parameters())}
    print(f"\ntest MAE: all={res['all']:.3f}  affected={res['affected']:.3f}  "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
