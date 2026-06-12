#!/usr/bin/env python3
"""GWN +/- ICSF/TIID collision experiment (audit item 2).

Faithful-in-spirit port of IGSTGNN's IncidentsIcsfModule onto GraphWaveNet in OUR
pipeline (see docs/superpowers/specs/2026-06-12-gwn-icsf-collision-design.md).
Per sample, the PRIMARY active incident (first active slot) provides:
  type (one-hot argmax), duration (hours), distances (N,3) = gaussian kernels of
  log euc/road distance + downstream flag (adapter parity: sigma 1.5/1.5, cutoff 4/5,
  closest sensor zeroed).
--use_icsf  adds incident-attention correction to the LAST history timestep (input space)
--use_tiid  adds exp(-t^2/2) decayed incident effect to the output (their sigma_t=1.0)
Artifacts: <out_dir>/<region>/{ckpt_best.pt, test_predictions.npz, summary.json}.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, make_loader
from model import gwnet

SIGMA_EUC, SIGMA_ROAD, CUT_EUC, CUT_ROAD = 1.5, 1.5, 4.0, 5.0


def build_adj_supports(edge_index, N, device):
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return [torch.from_numpy(deg_inv[:, None] * A).to(device),
            torch.from_numpy((deg_inv[:, None] * A.T)).to(device)]


def primary_incident(batch, device):
    """Return type_idx (B,), duration (B,1), distances (B,N,3) for the first active slot."""
    inc_feat = batch["incident_feat"].to(device)          # (B, M, C_e)
    inc_mask = batch["incident_mask"].to(device)          # (B, M)
    rel = batch["rel_feat"].to(device)                    # (B, M, N, 4)
    B = inc_feat.size(0)
    slot = inc_mask.float().argmax(dim=1)                 # first active (0 if none)
    ar = torch.arange(B, device=device)
    feat = inc_feat[ar, slot]                             # (B, C_e)
    rel_p = rel[ar, slot]                                 # (B, N, 4)
    has_inc = inc_mask.any(dim=1).float()                 # (B,)

    type_idx = feat[:, :8].argmax(dim=1)                  # (B,)
    duration = feat[:, 8:9]                               # (B, 1) hours

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
    closest = (log_euc + (log_euc <= 1e-6).float() * 1e9).argmin(dim=1)  # (B,)
    dist[ar, closest] = 0.0
    dist = dist * has_inc[:, None, None]
    return type_idx, duration, dist


class ICSFLite(nn.Module):
    """Port of IncidentsIcsfModule: incident embedding + distance attention fusion."""

    def __init__(self, c_in=3, d=32):
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
        self.out_proj = nn.Linear(d, 1)

    def forward(self, x_last, type_idx, duration, dist, time_last):
        # x_last (B,N,C), dist (B,N,3), time_last (B,2)
        emb = torch.cat([self.type_emb(type_idx), self.dur_mlp(duration), time_last], dim=-1)
        inc = self.incident_fusion(emb)                            # (B, d)
        dmask = (dist.sum(-1) > 0).float().unsqueeze(-1)           # (B, N, 1)
        dist_attn = F.softmax(self.distance_encoder(dist), dim=1)  # (B, N, d)
        K = self.k_proj(inc).unsqueeze(1) * dmask                  # (B, N, d)
        V = self.v_proj(inc).unsqueeze(1).expand(-1, x_last.size(1), -1)
        Q = self.q_proj(x_last)                                    # (B, N, d)
        a1 = F.softmax((Q * K).masked_fill(dmask == 0, -1e7), dim=1)
        attn = F.softmax(self.fusion_mlp(torch.cat([a1, dist_attn], -1))
                         .masked_fill(dmask == 0, -1e7), dim=1)
        effect = attn * V * dmask                                  # (B, N, d)
        return effect


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


class GWNICSF(nn.Module):
    def __init__(self, N, T_p, supports, device, nhid=32, dropout=0.3,
                 use_icsf=True, use_tiid=False, d_icsf=32):
        super().__init__()
        self.use_icsf, self.use_tiid = use_icsf, use_tiid
        self.backbone = gwnet(device=device, num_nodes=N, dropout=dropout,
                              supports=supports, gcn_bool=True, addaptadj=True,
                              in_dim=3, out_dim=T_p,
                              residual_channels=nhid, dilation_channels=nhid,
                              skip_channels=nhid * 8, end_channels=nhid * 16)
        if use_icsf or use_tiid:
            self.icsf = ICSFLite(c_in=3, d=d_icsf)
        decay = torch.exp(-torch.arange(1, T_p + 1, dtype=torch.float32) ** 2 / 2.0)
        self.register_buffer("decay", decay)               # sigma_t = 1.0, theirs

    def forward(self, x_hist, type_idx, duration, dist, time_last):
        # x_hist (B,N,T,3) -> effect on last step + optional output decay
        effect = None
        if self.use_icsf or self.use_tiid:
            effect = self.icsf(x_hist[:, :, -1, :], type_idx, duration, dist, time_last)
        x = x_hist
        if self.use_icsf and effect is not None:
            x = x_hist.clone()
            x[:, :, -1, :] = x[:, :, -1, :] + self.icsf.in_proj(effect)
        out = self.backbone(x.permute(0, 3, 1, 2).contiguous())   # (B,T_p,N,1)
        y = out.squeeze(-1).permute(0, 2, 1).contiguous()         # (B,N,T_p)
        if self.use_tiid and effect is not None:
            y = y + self.icsf.out_proj(effect) * self.decay.view(1, 1, -1)
        return y


def run_batch(model, batch, device):
    x = batch["x_hist"].to(device)
    type_idx, duration, dist = primary_incident(batch, device)
    time_last = batch["time_feat"][:, -1, :].to(device)
    return model(x, type_idx, duration, dist, time_last)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/baselines/gwn_icsf"))
    p.add_argument("--use_icsf", action="store_true")
    p.add_argument("--use_tiid", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--nhid", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  region: {args.region}  seed: {args.seed}  "
          f"icsf={args.use_icsf}  tiid={args.use_tiid}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    rdata = train_ds.regions[args.region]
    N, T_p = int(rdata.N), int(rdata.T_p)
    supports = build_adj_supports(rdata.edge_index, N, device)
    model = GWNICSF(N, T_p, supports, device, nhid=args.nhid, dropout=args.dropout,
                    use_icsf=args.use_icsf, use_tiid=args.use_tiid).to(device)
    print(f"params: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(n_batches, 1),
                              eta_min=args.lr * 1e-2)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        s, n = 0.0, 0
        model.train()
        for bi, batch in enumerate(train_loader):
            y = batch["y_true"][..., 0].to(device)
            m = batch["y_mask"][..., 0].to(device)
            loss = masked_mae(run_batch(model, batch, device), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            s += float(loss.item()) * y.size(0)
            n += y.size(0)
            if bi % 300 == 0:
                print(f"  ep{epoch:02d} batch {bi}/{n_batches} L={loss.item():.3f}", flush=True)
        model.eval()
        tot, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y_true"][..., 0].to(device)
                m = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(run_batch(model, batch, device), y, m).item()) * y.size(0)
                nv += y.size(0)
        val = tot / max(nv, 1)
        print(f"==> ep{epoch:02d} train L={s/max(n,1):.4f}  val L={val:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val < best_val:
            best_val = val
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val": val, "config": vars(args)}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    print(f"\nbest val: {best_val:.4f}\n=== test inference ===", flush=True)
    best = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
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
            pred = run_batch(model, batch, device).permute(0, 2, 1).cpu().numpy()
            bs = pred.shape[0]
            pred_flow[cursor:cursor + bs] = pred
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            sample_start[cursor:cursor + bs] = batch["sample_start"].numpy()
            region_code[cursor:cursor + bs] = batch["region_code"].numpy()
            cursor += bs

    np.savez_compressed(out_dir / "test_predictions.npz",
                        region_code=region_code, sample_start=sample_start,
                        region_node_idx=rdata.region_idx.astype(np.int64),
                        pred_raw_flow=pred_flow, actual_future_flow=actual_flow,
                        y_mask_flow=y_mask_flow, affected_mask=affected)
    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best_val, "use_icsf": args.use_icsf, "use_tiid": args.use_tiid,
           "seed": args.seed}
    print(f"\ntest MAE: all={res['all']:.3f}  affected={res['affected']:.3f}  "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
