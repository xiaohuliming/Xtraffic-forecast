#!/usr/bin/env python3
"""STAEformer + adaptive de-seasonalization wrapper (route B, backbone-agnostic test).

Swaps the GWN backbone of the RGDN de-seasonalization framework for STAEformer to
test whether the v0c gain transfers to the strongest label-free backbone. Variants:
  s0a  raw STAEformer (= existing baseline, control)
  s0b  + de-seasonalization (predict deviation from cache climatology, add baseline back)
  s0c  + adaptive alpha (anchor = alpha*baseline + (1-alpha)*persistence, alpha from
        recent residual magnitude)  <- the key one
  s0d  + constant alpha (learned scalar blend, no r-dependence; ablation)

Channel 0 of the STAEformer input is the de-seasonalized residual; tod/dow channels are
kept so STAEformer's time embeddings still fire (this is the point: does de-season help
a backbone that ALREADY has time embeddings?). Same masked-MAE / artifact schema as
train_staeformer_xtraffic.py so numbers compare directly to the GWN-based v0a-d.
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
from fourier_dual_net.rgdn import AdaptiveAlpha
from fourier_dual_net.deseason import train_residual_std
from STAEformer import STAEformer

VARIANTS = {
    "s0a": dict(deseason=False, adaptive=False, const_alpha=False),
    "s0b": dict(deseason=True,  adaptive=False, const_alpha=False),
    "s0c": dict(deseason=True,  adaptive=True,  const_alpha=False),
    "s0d": dict(deseason=True,  adaptive=False, const_alpha=True),
}


class STAEformerDeseason(nn.Module):
    def __init__(self, N, T_h, T_p, deseason=True, adaptive=False, const_alpha=False):
        super().__init__()
        self.deseason = bool(deseason)
        self.adaptive = bool(adaptive) and self.deseason
        self.const_alpha = bool(const_alpha) and self.deseason and not self.adaptive
        self.backbone = STAEformer(num_nodes=N, in_steps=T_h, out_steps=T_p,
                                   steps_per_day=288, input_dim=1, output_dim=1)
        self.register_buffer("sd_res", torch.tensor(1.0))
        self.register_buffer("flow_mu", torch.tensor(0.0))
        self.register_buffer("flow_sd", torch.tensor(1.0))
        self.last_alpha = None
        if self.adaptive:
            self.alpha_mod = AdaptiveAlpha()
        if self.const_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(2.2))   # sigmoid ~ 0.90 init

    def forward(self, flow_hist, x_baseline, y_baseline, time_feat):
        # flow_hist (B,N,T_h); x_baseline (B,N,T_h); y_baseline (B,N,T_p); time_feat (B,T_h,2)
        B, N, T_h = flow_hist.shape
        if self.deseason:
            sig = (flow_hist - x_baseline) / self.sd_res
        else:
            sig = (flow_hist - self.flow_mu) / self.flow_sd
        tod = time_feat[:, :, 0].unsqueeze(-1).expand(B, T_h, N)
        dow = (time_feat[:, :, 1] * 7.0).round().clamp(0, 6).unsqueeze(-1).expand(B, T_h, N)
        x = torch.stack([sig.permute(0, 2, 1), tod, dow], dim=-1)        # (B,T_h,N,3)
        out = self.backbone(x).squeeze(-1).permute(0, 2, 1)             # (B,N,T_p)
        if not self.deseason:
            return out * self.flow_sd + self.flow_mu
        if self.adaptive:
            r = (flow_hist - x_baseline).abs().mean(dim=2) / self.sd_res
            alpha = self.alpha_mod(r)
            self.last_alpha = alpha.detach()
            persist = flow_hist[:, :, -1:]
            anchor = alpha.unsqueeze(-1) * y_baseline + (1.0 - alpha).unsqueeze(-1) * persist
            return anchor + out * self.sd_res
        if self.const_alpha:
            alpha = torch.sigmoid(self.alpha_logit)
            self.last_alpha = alpha.detach().expand(B, N)
            persist = flow_hist[:, :, -1:]
            anchor = alpha * y_baseline + (1.0 - alpha) * persist
            return anchor + out * self.sd_res
        return y_baseline + out * self.sd_res


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/baselines/staeformer_deseason"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} region {args.region} variant {args.variant} seed {args.seed}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)

    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
    seg, segm = flows[:hi], fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    sd_res = train_residual_std(rdata.flow_series, rdata.flow_mask,
                                rdata.baseline_median, rdata.day_kind, rdata.tod, hi, ch=0)
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f} sd_res={sd_res:.3f}", flush=True)

    def make_model(variant):
        m = STAEformerDeseason(N, T_h, T_p, **VARIANTS[variant]).to(device)
        m.sd_res.fill_(sd_res); m.flow_mu.fill_(mu); m.flow_sd.fill_(sd)
        return m

    def forward_batch(model, batch):
        flow_hist = batch["x_hist"][..., 0].to(device)          # (B,N,T_h)
        x_base = batch["x_baseline"][..., 0].to(device)         # (B,N,T_h)
        y_base = batch["y_baseline"][..., 0].to(device)         # (B,N,T_p)
        tf = batch["time_feat"].to(device)                      # (B,T_h,2)
        return model(flow_hist, x_base, y_base, tf)             # (B,N,T_p)

    if args.smoke:
        loader = make_loader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        for v in VARIANTS:
            m = make_model(v)
            np_ = sum(q.numel() for q in m.parameters() if q.requires_grad)
            y = forward_batch(m, batch)
            loss = masked_mae(y, batch["y_true"][..., 0].to(device), batch["y_mask"][..., 0].to(device))
            loss.backward()
            print(f"  {v} params={np_:,} out={tuple(y.shape)} loss={loss.item():.3f} "
                  f"finite={bool(torch.isfinite(y).all())}", flush=True)
        print("SMOKE_OK", flush=True)
        return

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    out_dir = args.out_dir / args.region / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.variant)
    nparam = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"variant {args.variant} params={nparam:,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed,
                               num_workers=args.num_workers)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def ev(loader):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                y = forward_batch(model, batch)
                t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(y, t, msk).item()) * y.size(0); n += y.size(0)
        model.train(); return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for batch in train_loader:
            y = forward_batch(model, batch)
            t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
            loss = masked_mae(y, t, msk)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            s += float(loss.item()) * y.size(0); n += y.size(0)
        v = ev(val_loader)
        print(f"==> ep{ep:02d} train L={s/max(n,1):.4f} val L={v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best = v
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "mu": mu, "sd": sd, "sd_res": sd_res}, out_dir / "ckpt_best.pt")
            print("    saved best", flush=True)

    print(f"\nbest val {best:.4f}\n=== test ===", flush=True)
    st = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"]); model.eval()
    S = len(test_ds)
    pred_flow = np.empty((S, T_p, N), dtype=np.float32)
    actual_flow = np.empty((S, T_p, N), dtype=np.float32)
    y_mask_flow = np.empty((S, T_p, N), dtype=bool)
    affected = np.empty((S, N), dtype=bool)
    alpha_all = np.empty((S, N), dtype=np.float32) if (model.adaptive or model.const_alpha) else None
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            y = forward_batch(model, batch).permute(0, 2, 1).cpu().numpy()    # (B,T_p,N)
            bs = y.shape[0]
            pred_flow[cursor:cursor + bs] = y
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            if alpha_all is not None:
                alpha_all[cursor:cursor + bs] = model.last_alpha.cpu().numpy()
            cursor += bs

    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best, "seed": args.seed, "variant": args.variant, "params": nparam}
    if alpha_all is not None:
        res["alpha_affected"] = float(alpha_all[affected].mean()) if affected.any() else float("nan")
        res["alpha_unaffected"] = float(alpha_all[~affected].mean()) if (~affected).any() else float("nan")
        if model.adaptive:
            res["alpha_r0"] = float(model.alpha_mod.r0.item())
            res["alpha_tau"] = float(F.softplus(model.alpha_mod.raw_tau).item() + 1e-3)
        elif model.const_alpha:
            res["alpha_const"] = float(torch.sigmoid(model.alpha_logit).item())
        print(f"alpha: aff={res['alpha_affected']:.3f} unaff={res['alpha_unaffected']:.3f}", flush=True)
    print(f"\ntest MAE all={res['all']:.3f} affected={res['affected']:.3f} "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
