#!/usr/bin/env python3
"""L1+L2 cheap performance pass: seed-ensemble + per-(node,horizon) bias correction fitted on
the VALIDATION split only (leakage-free), applied to test. Loads existing ckpt_best.pt files;
no training. Reports per-seed base/debiased and 3-seed ensemble base/debiased masked MAE."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "baselines" / "STAEformer"))

import numpy as np
import torch

from dist_net.data import MultiRegionDataset, make_loader
from train_staeformer_deseason import STAEformerDeseason, VARIANTS  # noqa

def masked_mae(pred, y, m):
    return float(np.abs(pred - y)[m].mean())

def infer(model, loader, device, S, T_p, N, need_truth=True):
    pred = np.empty((S, T_p, N), np.float32)
    y = np.empty((S, T_p, N), np.float32) if need_truth else None
    msk = np.empty((S, T_p, N), bool)
    aff = np.empty((S, N), bool)
    cur = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x_hist"][..., 0].to(device), batch["x_baseline"][..., 0].to(device),
                        batch["y_baseline"][..., 0].to(device), batch["time_feat"].to(device))
            b = out.size(0)
            pred[cur:cur+b] = out.permute(0, 2, 1).cpu().numpy()
            if need_truth: y[cur:cur+b] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            msk[cur:cur+b] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            aff[cur:cur+b] = batch["affected_mask"].numpy()
            cur += b
    return pred, y, msk, aff

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--data_dir", default="outputs/dist_net/region_data_v2_noleak")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument(
        "--ckpt_root",
        default="outputs/protocol_v2_noleak/baselines/staeformer_deseason",
    )
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    rdata = val_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    vl = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    tl = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    Sv, St = len(val_ds), len(test_ds)

    ens_base = np.zeros((St, T_p, N), np.float64)
    ens_deb = np.zeros((St, T_p, N), np.float64)
    res = {"region": args.region, "variant": args.variant, "per_seed": {}}
    yt = mt = at = None
    for seed in args.seeds:
        ck = torch.load(f"{args.ckpt_root}/{args.region}/{args.variant}_seed{seed}/ckpt_best.pt",
                        map_location=device, weights_only=False)
        model = STAEformerDeseason(N, T_h, T_p, **VARIANTS[args.variant]).to(device)
        model.sd_res.fill_(ck["sd_res"]); model.flow_mu.fill_(ck["mu"]); model.flow_sd.fill_(ck["sd"])
        missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
        assert not unexpected, unexpected
        assert all(k == "sd_node" for k in missing), missing   # old ckpts predate the sd_node buffer (unused at inference)
        model.eval()

        pv, yv, mv, _ = infer(model, vl, device, Sv, T_p, N)
        rsum = np.where(mv, pv - yv, 0.0).sum(axis=0)
        rcnt = mv.sum(axis=0).clip(min=1)
        bias = (rsum / rcnt).astype(np.float32)                    # (T_p,N) per-horizon per-node

        pt, yt, mt, at = infer(model, tl, device, St, T_p, N)
        base = masked_mae(pt, yt, mt)
        ptd = pt - bias[None]
        deb = masked_mae(ptd, yt, mt)
        aff3 = np.broadcast_to(at[:, None, :], yt.shape)
        res["per_seed"][seed] = {
            "base_all": base, "debias_all": deb,
            "base_aff": masked_mae(pt, yt, mt & aff3), "debias_aff": masked_mae(ptd, yt, mt & aff3),
            "base_unaff": masked_mae(pt, yt, mt & ~aff3), "debias_unaff": masked_mae(ptd, yt, mt & ~aff3),
            "mean_abs_bias": float(np.abs(bias).mean())}
        print(f"seed {seed}: base={base:.3f} debias={deb:.3f} (Δ{deb-base:+.3f}) "
              f"|bias|={np.abs(bias).mean():.3f}", flush=True)
        ens_base += pt; ens_deb += ptd
    k = len(args.seeds)
    aff3 = np.broadcast_to(at[:, None, :], yt.shape)
    res["ensemble"] = {
        "base_all": masked_mae(ens_base / k, yt, mt), "debias_all": masked_mae(ens_deb / k, yt, mt),
        "base_aff": masked_mae(ens_base / k, yt, mt & aff3), "debias_aff": masked_mae(ens_deb / k, yt, mt & aff3),
        "base_unaff": masked_mae(ens_base / k, yt, mt & ~aff3), "debias_unaff": masked_mae(ens_deb / k, yt, mt & ~aff3)}
    print(f"\nENSEMBLE({k}): base={res['ensemble']['base_all']:.3f} debias={res['ensemble']['debias_all']:.3f}")
    print(f"  aff: base={res['ensemble']['base_aff']:.3f} debias={res['ensemble']['debias_aff']:.3f}")
    out = Path(f"outputs/diagnostics/l1l2_{args.region}_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")

if __name__ == "__main__":
    main()
