#!/usr/bin/env python3
"""Diagnose why the neighbor-residual injection hurts RGDN v1 vs v2 (inference-only).

D1 test MAE with injection vs injection zeroed: does the learned injection hurt at inference.
D2 adaptive-adjacency structure: is adp learned or near-uniform noise.
D3 injection feature scale vs the de-seasonalized residual signal.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))
from dist_net.data import MultiRegionDataset, make_loader
from train_rgdn import build_adj_supports, masked_mae, forward_batch, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--ckpt", default="outputs/rgdn/Alameda/v1_seed42/ckpt_best.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = argparse.Namespace(**st["config"])
    test_ds = MultiRegionDataset([args.region], cfg.data_dir, cfg.graph_dir, split="test")
    rdata = test_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    supports = build_adj_supports(rdata.edge_index, N, device)

    model = make_model("v1", N, supports, T_h, T_p, device, cfg)
    model.load_state_dict(st["model_state"])
    model.sd_res.fill_(st["sd_res"]); model.flow_mu.fill_(st["mu"]); model.flow_sd.fill_(st["sd"])
    model.eval()
    c_inj = int(cfg.c_inject)
    test_loader = make_loader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # D2 adp structure
    with torch.no_grad():
        adp = torch.softmax(torch.relu(model.inject_mod.e1 @ model.inject_mod.e2), dim=1)
    row_max = adp.max(dim=1).values.mean().item()
    ent = (-(adp * (adp + 1e-12).log()).sum(dim=1)).mean().item()
    uni_ent = float(torch.log(torch.tensor(float(N))))
    print(f"[D2 adp] N={N} mean_row_max={row_max:.4f} (uniform={1.0/N:.4f}) "
          f"mean_row_entropy={ent:.3f}/{uni_ent:.3f} ratio={ent/uni_ent:.3f} "
          f"(ratio~1 => near-uniform averaging/noise, ~0 => peaked/learned)", flush=True)

    # D3 injection scale on one batch
    with torch.no_grad():
        batch = next(iter(test_loader))
        x_hist = batch["x_hist"].to(device); x_base = batch["x_baseline"].to(device)
        sig = (x_hist[..., 0] - x_base[..., 0]) / model.sd_res
        inj = model.inject_mod(sig)
        print(f"[D3 scale] sig_std={sig.std().item():.3f} injection_std={inj.std().item():.3f} "
              f"inj/sig={inj.std().item()/(sig.std().item()+1e-9):.3f}", flush=True)

    def eval_mae():
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in test_loader:
                y = forward_batch(model, b, device)
                t = b["y_true"][..., 0].to(device); m = b["y_mask"][..., 0].to(device)
                tot += float(masked_mae(y, t, m).item()) * y.size(0); n += y.size(0)
        return tot / max(n, 1)

    mae_norm = eval_mae()

    class Zero(torch.nn.Module):
        def forward(self, res):
            return torch.zeros(res.size(0), c_inj, res.size(1), res.size(2), device=res.device)

    orig = model.inject_mod
    model.inject_mod = Zero().to(device)
    mae_zero = eval_mae()
    model.inject_mod = orig
    print(f"[D1 ablation] v1_normal={mae_norm:.3f} injection_zeroed={mae_zero:.3f} "
          f"delta={mae_zero - mae_norm:+.3f}  (ref v2=11.779 v0b=11.711)", flush=True)
    print("[interpretation] zeroed << normal => learned injection is net-harmful at inference",
          flush=True)


if __name__ == "__main__":
    main()
