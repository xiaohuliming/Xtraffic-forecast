#!/usr/bin/env python3
"""STAEformer + adaptive de-seasonalization wrapper (route B, backbone-agnostic test).

Swaps the GWN backbone of the RGDN de-seasonalization framework for STAEformer to
test whether the v0c gain transfers to the strongest label-free backbone. Variants:
  s0a  raw STAEformer (= existing baseline, control)
  s0b  + de-seasonalization (predict deviation from cache climatology, add baseline back)
  s0c  + adaptive alpha (anchor = alpha*baseline + (1-alpha)*persistence, alpha from
        recent residual magnitude)  <- the key one
  s0d  + constant alpha (learned scalar blend, no r-dependence; ablation)

RADS-O / NDR ladder (bounded bet, 2026-07-03; strict stop lines, see rads notes):
  s0e  A0: horizon-wise gate g[n,h]=sigmoid(a_h - softplus(b_h)*d_n - softplus(c_h)*u_n),
       no oracle supervision (ablation rung, backfill if A1 wins)
  s0f  A1 "RADS-O": s0e + train-time oracle-gate distillation. Soft target
       g* = |y-P|/(|y-B|+|y-P|), confidence-weighted Huber, flow-unit calibrated.
  s0g  B "NDR-v0": s0f + dual residual heads (backbone output_dim=2): normal head on the
       periodic anchor, disturbance head on the persistence anchor, fused by g;
       oracle-supervised specialization loss with batch-normalized weights.

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

from dist_net.metrics import masked_mae_loss
from fourier_dual_net.rgdn import AdaptiveAlpha
from STAEformer import STAEformer

VARIANTS = {
    "s0a": dict(deseason=False, adaptive=False, const_alpha=False),
    "s0b": dict(deseason=True,  adaptive=False, const_alpha=False),
    "s0c": dict(deseason=True,  adaptive=True,  const_alpha=False),
    "s0d": dict(deseason=True,  adaptive=False, const_alpha=True),
    "s0e": dict(deseason=True,  hgate=True,  oracle=False, dual=False),
    "s0f": dict(deseason=True,  hgate=True,  oracle=True,  dual=False),
    "s0g": dict(deseason=True,  hgate=True,  oracle=True,  dual=True),
    "s0h": dict(deseason=True,  abank=True,  oracle=True),
}

ANCHOR_NAMES = ["base", "persist", "up", "down", "decay", "daylag"]   # K=6 anchor bank (s0h)


def build_road_graph(rdata, feat_names, K=8):
    """Directed local road adjacency from static_meta (Fwy id, Abs PM order, direction one-hot).
    A_up = K nearest lower-PM same-fwy-same-dir neighbors; A_down = higher-PM. Row-normalized."""
    sm = rdata.static_meta
    fi = {n: i for i, n in enumerate(feat_names)}
    N = sm.shape[0]
    fwy = sm[:, fi["Fwy"]] if "Fwy" in fi else np.zeros(N, np.float32)
    pm = sm[:, fi["Abs PM"]] if "Abs PM" in fi else np.arange(N, dtype=np.float32)
    dir_cols = [fi[c] for c in ("dir_N", "dir_E", "dir_S", "dir_W") if c in fi]
    direction = sm[:, dir_cols].argmax(1) if dir_cols else np.zeros(N, np.int64)
    A_up = np.zeros((N, N), np.float32)
    A_down = np.zeros((N, N), np.float32)
    groups = {}
    for i in range(N):
        groups.setdefault((round(float(fwy[i]), 4), int(direction[i])), []).append(i)
    for members in groups.values():
        idx = np.array(sorted(members, key=lambda j: pm[j]))
        for rank, n in enumerate(idx):
            for k in range(1, K + 1):
                if rank - k >= 0:
                    A_up[n, idx[rank - k]] = 1.0
                if rank + k < len(idx):
                    A_down[n, idx[rank + k]] = 1.0
    for A in (A_up, A_down):
        s = A.sum(1, keepdims=True)
        np.divide(A, s, out=A, where=s > 0)
    return A_up, A_down


class HorizonGate(nn.Module):
    """Per-node per-horizon regime gate: g = sigmoid(a_h - softplus(b_h)*d - softplus(c_h)*u).
    Monotone in deviation d and slope u by construction; ~3*T_p params."""

    def __init__(self, T_p: int):
        super().__init__()
        self.a = nn.Parameter(torch.full((T_p,), 2.0))
        self.b = nn.Parameter(torch.zeros(T_p))
        self.c = nn.Parameter(torch.zeros(T_p))

    def forward(self, d, u):                                             # d,u (B,N)
        a = self.a.view(1, 1, -1)
        bb = F.softplus(self.b).view(1, 1, -1)
        cc = F.softplus(self.c).view(1, 1, -1)
        return torch.sigmoid(a - bb * d.unsqueeze(-1) - cc * u.unsqueeze(-1))  # (B,N,T_p)


class STAEformerDeseason(nn.Module):
    def __init__(self, N, T_h, T_p, deseason=True, adaptive=False, const_alpha=False,
                 hgate=False, oracle=False, dual=False, abank=False,
                 road_up=None, road_down=None):
        super().__init__()
        self.deseason = bool(deseason)
        self.abank = bool(abank) and self.deseason
        self.hgate = bool(hgate) and self.deseason and not self.abank
        self.adaptive = bool(adaptive) and self.deseason and not self.hgate and not self.abank
        self.const_alpha = bool(const_alpha) and self.deseason and not self.adaptive and not self.hgate
        self.T_p = T_p
        self.oracle_sup = bool(oracle) and (self.hgate or self.abank)
        self.dual = bool(dual) and self.hgate
        self.backbone = STAEformer(num_nodes=N, in_steps=T_h, out_steps=T_p,
                                   steps_per_day=288, input_dim=1,
                                   output_dim=2 if self.dual else 1)
        self.register_buffer("sd_res", torch.tensor(1.0))
        self.register_buffer("flow_mu", torch.tensor(0.0))
        self.register_buffer("flow_sd", torch.tensor(1.0))
        self.register_buffer("sd_node", torch.ones(N))          # per-node residual std (confidence)
        self.last_alpha = None
        self.last_d = None
        self.aux = None                                          # training-loss extras (hgate only)
        if self.adaptive:
            self.alpha_mod = AdaptiveAlpha()
        if self.const_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(2.2))   # sigmoid ~ 0.90 init
        if self.hgate:
            self.gate_mod = HorizonGate(T_p)
        if self.abank:
            K = len(ANCHOR_NAMES)
            # per-(anchor,horizon) linear-in-(d,u) router logits; oracle distillation trains it
            w0 = torch.full((K, T_p), -0.5); w0[0] = 1.5; w0[1] = 0.5
            wd = torch.zeros(K, T_p); wd[0] = -1.0; wd[1] = 0.3
            self.w0 = nn.Parameter(w0)
            self.wd = nn.Parameter(wd)
            self.wu = nn.Parameter(torch.zeros(K, T_p))
            self.raw_lam = nn.Parameter(torch.full((N,), 6.0))      # softplus ~ 6 steps = 30min
            assert road_up is not None and road_down is not None
            self.register_buffer("road_up", road_up.float())
            self.register_buffer("road_down", road_down.float())
            self.register_buffer("up_deg", (road_up > 0).sum(1).float())
            self.register_buffer("down_deg", (road_down > 0).sum(1).float())
            self.last_router = None

    def forward(self, flow_hist, x_baseline, y_baseline, time_feat, y_daylag=None):
        # flow_hist (B,N,T_h); x_baseline (B,N,T_h); y_baseline (B,N,T_p); time_feat (B,T_h,2)
        B, N, T_h = flow_hist.shape
        if self.deseason:
            sig = (flow_hist - x_baseline) / self.sd_res
        else:
            sig = (flow_hist - self.flow_mu) / self.flow_sd
        tod = time_feat[:, :, 0].unsqueeze(-1).expand(B, T_h, N)
        dow = (time_feat[:, :, 1] * 7.0).round().clamp(0, 6).unsqueeze(-1).expand(B, T_h, N)
        x = torch.stack([sig.permute(0, 2, 1), tod, dow], dim=-1)        # (B,T_h,N,3)
        raw = self.backbone(x)                                           # (B,T_p,N,out_dim)
        if not self.deseason:
            return raw.squeeze(-1).permute(0, 2, 1) * self.flow_sd + self.flow_mu
        if self.abank:
            r = (flow_hist - x_baseline) / self.sd_res                   # (B,N,T_h)
            d = r.abs().mean(dim=2)                                      # (B,N)
            u = (r[:, :, -1] - r[:, :, -2]).abs()                        # (B,N)
            last = flow_hist[:, :, -1]                                   # (B,N)
            Tp = self.T_p
            persist = last.unsqueeze(-1).expand(-1, -1, Tp)
            up = torch.einsum("nm,bm->bn", self.road_up, last)
            up = torch.where(self.up_deg.view(1, -1) > 0, up, last).unsqueeze(-1).expand(-1, -1, Tp)
            down = torch.einsum("nm,bm->bn", self.road_down, last)
            down = torch.where(self.down_deg.view(1, -1) > 0, down, last).unsqueeze(-1).expand(-1, -1, Tp)
            lam = F.softplus(self.raw_lam) + 0.5                         # (N,) relaxation steps
            steps = torch.arange(1, Tp + 1, device=flow_hist.device, dtype=flow_hist.dtype)
            decay = torch.exp(-steps.view(1, 1, -1) / lam.view(1, -1, 1))            # (1,N,Tp)
            a_decay = y_baseline + (last - x_baseline[:, :, -1]).unsqueeze(-1) * decay
            A = torch.stack([y_baseline, persist, up, down, a_decay, y_daylag], dim=-1)  # (B,N,Tp,K)
            logits = (self.w0.t().view(1, 1, Tp, -1)
                      + d.unsqueeze(-1).unsqueeze(-1) * self.wd.t().view(1, 1, Tp, -1)
                      + u.unsqueeze(-1).unsqueeze(-1) * self.wu.t().view(1, 1, Tp, -1))
            g = torch.softmax(logits, dim=-1)                            # (B,N,Tp,K)
            out = raw.squeeze(-1).permute(0, 2, 1)
            yhat = (g * A).sum(-1) + out * self.sd_res
            self.last_alpha = g[..., 0].detach().mean(-1)                # baseline-anchor weight
            self.last_d = d.detach()
            self.last_router = g.detach().mean(dim=2)                    # (B,N,K)
            self.aux = {"g": g, "A": A}
            return yhat
        if self.hgate:
            r = (flow_hist - x_baseline) / self.sd_res                   # (B,N,T_h) signed
            d = r.abs().mean(dim=2)                                      # (B,N) deviation strength
            u = (r[:, :, -1] - r[:, :, -2]).abs()                        # (B,N) recent slope
            g = self.gate_mod(d, u)                                      # (B,N,T_p)
            self.last_alpha = g.detach().mean(-1)
            self.last_d = d.detach()
            persist = flow_hist[:, :, -1:]                               # (B,N,1)
            if self.dual:
                Rn = raw[..., 0].permute(0, 2, 1)                        # (B,N,T_p)
                Rd = raw[..., 1].permute(0, 2, 1)
                yn = y_baseline + Rn * self.sd_res
                yd = persist + Rd * self.sd_res
                self.aux = {"g": g, "B": y_baseline, "P": persist, "yn": yn, "yd": yd}
                return g * yn + (1.0 - g) * yd
            out = raw.squeeze(-1).permute(0, 2, 1)
            anchor = g * y_baseline + (1.0 - g) * persist
            self.aux = {"g": g, "B": y_baseline, "P": persist}
            return anchor + out * self.sd_res
        out = raw.squeeze(-1).permute(0, 2, 1)                           # (B,N,T_p)
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
    return masked_mae_loss(pred, target, mask)


def flow_metrics(pred: np.ndarray, actual: np.ndarray, msk: np.ndarray,
                 aff: np.ndarray) -> dict:
    """Masked MAE/RMSE/MAPE over all/affected/unaffected + per-horizon lists.
    MAPE follows LargeST (null_val=0): only entries with actual != 0 count."""
    S, T_p, N = pred.shape
    diff = np.abs(pred - actual)
    a_abs = np.abs(actual)
    aff3 = np.broadcast_to(aff[:, None, :], (S, T_p, N))

    def stats(d, a, sel):
        dd = d[sel].astype(np.float64)
        p = sel & (a > 1e-4)
        rr = (d[p] / a[p]).astype(np.float64)
        return (float(dd.mean()) if dd.size else float("nan"),
                float(np.sqrt((dd ** 2).mean())) if dd.size else float("nan"),
                float(rr.mean() * 100.0) if rr.size else float("nan"))

    mae_all, rmse_all, mape_all = stats(diff, a_abs, msk)
    mae_aff, rmse_aff, mape_aff = stats(diff, a_abs, msk & aff3)
    mae_un, rmse_un, mape_un = stats(diff, a_abs, msk & ~aff3)
    mae_h, rmse_h, mape_h = [], [], []
    for h in range(T_p):
        m, r, p = stats(diff[:, h], a_abs[:, h], msk[:, h])
        mae_h.append(m); rmse_h.append(r); mape_h.append(p)
    return {"all": mae_all, "affected": mae_aff, "unaffected": mae_un,
            "rmse": rmse_all, "rmse_affected": rmse_aff, "rmse_unaffected": rmse_un,
            "mape": mape_all, "mape_affected": mape_aff, "mape_unaffected": mape_un,
            "h3": mae_h[2], "h6": mae_h[5], "h12": mae_h[T_p - 1],
            "mae_h": mae_h, "rmse_h": rmse_h, "mape_h": mape_h}


def per_node_residual_std(rdata, hi, ch=0, floor=1e-6):
    flow = rdata.flow_series[:hi, :, ch]
    mask = rdata.flow_mask[:hi, :, ch].astype(bool)
    base = rdata.baseline_median[rdata.day_kind[:hi], rdata.tod[:hi]][:, :, ch]
    res = np.where(mask, flow - base, np.nan)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(res, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > floor), sd, 1.0)
    return sd.astype(np.float32)                                 # (N,)


def compute_loss(model, y, batch, args, device):
    """Fused MAE + (oracle-gate distillation, specialization) for the hgate variants.
    Gate loss is flow-unit calibrated (x sd_node): without it the oracle gradient on the
    ~36 gate params is two orders below the main-loss gradient and the supervision is inert."""
    t = batch["y_true"][..., 0].to(device)
    msk = batch["y_mask"][..., 0].to(device).float()
    loss = masked_mae(y, t, msk)
    parts = {"mae": float(loss.item())}
    aux = model.aux
    if aux is not None and model.oracle_sup and model.training and getattr(model, "abank", False):
        g, A = aux["g"], aux["A"]                                # (B,N,Tp,K)
        e = (t.unsqueeze(-1) - A).abs()
        inv = 1.0 / (e + 0.02 * model.sd_node.view(1, -1, 1, 1))
        gstar = inv / inv.sum(-1, keepdim=True)                  # inverse-error soft oracle
        spread = ((e.max(-1).values - e.min(-1).values) / (e.mean(-1) + 1e-6)).clamp(0, 1)
        arange_ = ((A.max(-1).values - A.min(-1).values) / model.sd_node.view(1, -1, 1)).clamp(0, 1)
        conf = spread * arange_ * msk
        huber = F.smooth_l1_loss(g, gstar, reduction="none", beta=0.25).sum(-1)
        lg = (conf * model.sd_node.view(1, -1, 1) * huber).sum() / (conf.sum() + 1e-6)
        loss = loss + args.lambda_gate * lg
        parts["gate"] = float(lg.item())
        return loss, parts
    if aux is not None and model.oracle_sup and model.training:
        g, Bq, P = aux["g"], aux["B"], aux["P"]
        eB = (t - Bq).abs()
        eP = (t - P).abs()
        gstar = eP / (eB + eP + 1e-6)                            # y closer to B -> gstar -> 1
        conf = ((eB - eP).abs() / (eB + eP + 1e-6)) \
            * ((Bq - P).abs() / model.sd_node.view(1, -1, 1)).clamp(0, 1) * msk
        huber = F.smooth_l1_loss(g, gstar, reduction="none", beta=0.25)
        lg = (conf * model.sd_node.view(1, -1, 1) * huber).sum() / (conf.sum() + 1e-6)
        loss = loss + args.lambda_gate * lg
        parts["gate"] = float(lg.item())
        if model.dual:
            mg = (gstar * msk).sum() / (msk.sum() + 1e-6)
            wN = gstar / (mg + 1e-6)
            wD = (1.0 - gstar) / (1.0 - mg + 1e-6)
            ls = ((wN * (aux["yn"] - t).abs() + wD * (aux["yd"] - t).abs()) * msk).sum() \
                / (msk.sum() + 1e-6)
            loss = loss + args.lambda_spec * ls
            parts["spec"] = float(ls.item())
    return loss, parts


def main():
    from dist_net.data import MultiRegionDataset, make_loader
    from dist_net.metrics import MaskedMAEAccumulator
    from dist_net.training_runtime import TrainingRunContext
    from fourier_dual_net.deseason import train_residual_std

    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--data_dir", default="outputs/dist_net/region_data_v3_noleak_dedup")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "outputs/protocol_v3_noleak_dedup/baselines/staeformer_deseason"
        ),
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lambda_gate", type=float, default=1.0)
    p.add_argument("--lambda_spec", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} region {args.region} variant {args.variant} seed {args.seed}", flush=True)

    out_dir = args.out_dir / args.region / f"{args.variant}_seed{args.seed}"
    run_context = TrainingRunContext.prepare(
        out_dir,
        data_dir=args.data_dir,
        region=args.region,
        model="staeformer_deseason",
        variant=args.variant,
        seed=args.seed,
        sampling_protocol="event",
    )
    out_dir = run_context.output_dir

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    rdata = train_ds.regions[args.region]
    run_context.verify_region_cache(rdata)
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)

    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    hi = rdata.train_end
    seg, segm = flows[:hi], fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    sd_res = train_residual_std(rdata.flow_series, rdata.flow_mask,
                                rdata.baseline_median, rdata.day_kind, rdata.tod, hi, ch=0)
    sd_node = per_node_residual_std(rdata, hi)
    import h5py
    with h5py.File(Path(args.data_dir) / f"{args.region}_traffic.h5", "r") as hf:
        feat_names = [x.decode() for x in hf.attrs["meta_feature_names"]]
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f} sd_res={sd_res:.3f}", flush=True)

    road_cache = {}

    def make_model(variant):
        kw = dict(VARIANTS[variant])
        if kw.get("abank"):
            if "up" not in road_cache:
                au, ad = build_road_graph(rdata, feat_names, K=8)
                road_cache["up"], road_cache["down"] = torch.from_numpy(au), torch.from_numpy(ad)
                print(f"road graph: up nnz={int((au > 0).sum())} down nnz={int((ad > 0).sum())}", flush=True)
            kw["road_up"], kw["road_down"] = road_cache["up"], road_cache["down"]
        m = STAEformerDeseason(N, T_h, T_p, **kw).to(device)
        m.sd_res.fill_(sd_res); m.flow_mu.fill_(mu); m.flow_sd.fill_(sd)
        m.sd_node.copy_(torch.from_numpy(sd_node).to(device))
        return m

    def forward_batch(model, batch):
        flow_hist = batch["x_hist"][..., 0].to(device)          # (B,N,T_h)
        x_base = batch["x_baseline"][..., 0].to(device)         # (B,N,T_h)
        y_base = batch["y_baseline"][..., 0].to(device)         # (B,N,T_p)
        tf = batch["time_feat"].to(device)                      # (B,T_h,2)
        ydl = batch["y_daylag"].to(device) if getattr(model, "abank", False) else None
        return model(flow_hist, x_base, y_base, tf, y_daylag=ydl)   # (B,N,T_p)

    if args.smoke:
        loader = make_loader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        for v in VARIANTS:
            m = make_model(v)
            np_ = sum(q.numel() for q in m.parameters() if q.requires_grad)
            y = forward_batch(m, batch)
            loss, parts = compute_loss(m, y, batch, args, device)
            loss.backward()
            print(f"  {v} params={np_:,} out={tuple(y.shape)} loss={loss.item():.3f} "
                  f"parts={parts} finite={bool(torch.isfinite(y).all())}", flush=True)
        print("SMOKE_OK", flush=True)
        return

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset(
        [args.region],
        args.data_dir,
        args.graph_dir,
        split="test",
        region_data_kwargs={"include_analysis_labels": True},
    )
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
        model.eval()
        metric = MaskedMAEAccumulator()
        with torch.no_grad():
            for batch in loader:
                y = forward_batch(model, batch)
                t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
                metric.update(y, t, msk)
        model.train()
        return metric.compute()

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for batch in train_loader:
            y = forward_batch(model, batch)
            loss, _parts = compute_loss(model, y, batch, args, device)
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
    sample_start = np.empty((S,), dtype=np.int64)
    region_code = np.empty((S,), dtype=np.int64)
    alpha_all = np.empty((S, N), dtype=np.float32) \
        if (model.adaptive or model.const_alpha or model.hgate or model.abank) else None
    d_all = np.empty((S, N), dtype=np.float32) if (model.hgate or model.abank) else None
    router_all = np.empty((S, N, len(ANCHOR_NAMES)), dtype=np.float32) if model.abank else None
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            y = forward_batch(model, batch).permute(0, 2, 1).cpu().numpy()    # (B,T_p,N)
            bs = y.shape[0]
            pred_flow[cursor:cursor + bs] = y
            actual_flow[cursor:cursor + bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor + bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor + bs] = batch["affected_mask"].numpy()
            sample_start[cursor:cursor + bs] = batch["sample_start"].numpy()
            region_code[cursor:cursor + bs] = batch["region_code"].numpy()
            if alpha_all is not None:
                alpha_all[cursor:cursor + bs] = model.last_alpha.cpu().numpy()
            if d_all is not None:
                d_all[cursor:cursor + bs] = model.last_d.cpu().numpy()
            if router_all is not None:
                router_all[cursor:cursor + bs] = model.last_router.cpu().numpy()
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
    res = {**flow_metrics(pred_flow, actual_flow, y_mask_flow, affected),
           "best_val": best, "seed": args.seed, "variant": args.variant, "params": nparam}
    res.update(rdata.result_metadata())
    if alpha_all is not None:
        res["alpha_affected"] = float(alpha_all[affected].mean()) if affected.any() else float("nan")
        res["alpha_unaffected"] = float(alpha_all[~affected].mean()) if (~affected).any() else float("nan")
        if model.adaptive:
            res["alpha_r0"] = float(model.alpha_mod.r0.item())
            res["alpha_tau"] = float(F.softplus(model.alpha_mod.raw_tau).item() + 1e-3)
        elif model.const_alpha:
            res["alpha_const"] = float(torch.sigmoid(model.alpha_logit).item())
        elif model.abank:
            for k, nm in enumerate(ANCHOR_NAMES):
                res[f"router_{nm}_aff"] = float(router_all[..., k][affected].mean()) if affected.any() else float("nan")
                res[f"router_{nm}_unaff"] = float(router_all[..., k][~affected].mean())
            print("router aff/unaff: " + " ".join(
                f"{nm}={res[f'router_{nm}_aff']:.2f}/{res[f'router_{nm}_unaff']:.2f}" for nm in ANCHOR_NAMES), flush=True)
        elif model.hgate:
            res["gate_a_mean"] = float(model.gate_mod.a.mean().item())
            res["gate_b_sp_mean"] = float(F.softplus(model.gate_mod.b).mean().item())
            res["gate_c_sp_mean"] = float(F.softplus(model.gate_mod.c).mean().item())
        print(f"alpha: aff={res['alpha_affected']:.3f} unaff={res['alpha_unaffected']:.3f}", flush=True)
    if d_all is not None:
        # deviation-bucket MAE: d is input-derived, so edges are model-independent and the
        # same buckets can be recomputed for any baseline later for paired comparison
        edges = np.quantile(d_all.reshape(-1), [0.2, 0.4, 0.6, 0.8])
        d3 = np.broadcast_to(d_all[:, None, :], (S, T_p, N))
        bmae, lo = [], -np.inf
        for hi_edge in list(edges) + [np.inf]:
            sel = y_mask_flow & (d3 > lo) & (d3 <= hi_edge)
            bmae.append(float(diff[sel].mean()) if sel.any() else None)
            lo = hi_edge
        res["bucket_edges"] = [float(x) for x in edges]
        res["bucket_mae"] = bmae
        print(
            "bucket MAE (low->high deviation): "
            f"{[round(b, 3) if b is not None else None for b in bmae]}",
            flush=True,
        )
        np.savez_compressed(out_dir / "gate_diag.npz", d_all=d_all, gate_mean=alpha_all,
                            affected_mask=affected)
    print(f"\ntest MAE all={res['all']:.3f} affected={res['affected']:.3f} "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    run_context.write_summary(
        res,
        valid_count=int(y_mask_flow.sum()),
        required_metrics=("all", "affected", "unaffected"),
    )


if __name__ == "__main__":
    main()
