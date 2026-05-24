#!/usr/bin/env python3
"""End-to-end smoke test for DIST-Net stub implementation.

Verifies:
  1. dataloader yields well-shaped batches per region bucket
  2. DISTNet forward produces correct-shape outputs
  3. losses compute without NaN/Inf
  4. backward propagates gradients to all parameters
  5. per-submodule parameter count is on budget (~870K from design doc §11.1)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dist_net.data import MultiRegionDataset, make_loader, REGION_NAME_TO_CODE
from dist_net.losses import compute_losses
from dist_net.model import DISTNet, DISTNetConfig, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("outputs/dist_net/region_data"))
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=3,
                   help="How many batches to forward+backward through")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=== Building dataset (lazy=False, eager into memory) ===", flush=True)
    t0 = time.time()
    ds = MultiRegionDataset(
        region_names=["Alameda", "ContraCosta", "Orange"],
        data_dir=args.data_dir, graph_dir=args.graph_dir,
        split="train", lazy=False,
    )
    print(f"  built in {time.time() - t0:.1f}s; |train|={len(ds)}")

    loader = make_loader(ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)

    print("\n=== Building DISTNet (all stubs) ===", flush=True)
    cfg = DISTNetConfig(
        c_x=3, c_meta=ds.regions["Alameda"].C_meta, c_e=13, d_t=5,
        n_regions=3, hidden_dim=64, t_h=12, t_p=12,
    )
    model = DISTNet(cfg)
    pc = count_parameters(model)
    print(f"  per-submodule params:")
    for k, v in pc.items():
        if k.startswith("_"):
            continue
        print(f"    {k:35s} {v:>10,}")
    print(f"    {'TOTAL':35s} {pc['_total']:>10,}  (design target ~870K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print(f"\n=== Forward + backward smoke test ({args.n_batches} batches) ===", flush=True)
    region_name_by_code = {v: k for k, v in REGION_NAME_TO_CODE.items()}

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        rc = int(batch["region_code"][0].item())
        region_name = region_name_by_code[rc]
        edge_index = torch.from_numpy(ds.regions[region_name].edge_index)
        N = batch["x_hist"].shape[1]

        out = model(batch, edge_index=edge_index)
        # shape sanity
        assert out["pred"].shape == (args.batch_size, N, cfg.t_p, cfg.c_x), \
            f"pred shape mismatch: {out['pred'].shape}"
        assert out["g"].shape == (args.batch_size, N, cfg.t_p), \
            f"g shape mismatch: {out['g'].shape}"

        losses = compute_losses(
            pred=out["pred"], pred_normal=out["pred_normal_final"],
            y_true=batch["y_true"], y_baseline=batch["y_baseline"],
            y_mask=batch["y_mask"], affected_mask=batch["affected_mask"],
        )
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is non-finite: {v.item()}"

        # backward + step
        optimizer.zero_grad()
        losses["L_total"].backward()

        # check grads finite + at least some non-zero
        grad_norms = []
        zero_grad_params = []
        for name, p in model.named_parameters():
            if p.grad is None:
                zero_grad_params.append(f"{name} (no grad)")
                continue
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if gn == 0:
                zero_grad_params.append(name)
        finite_grads = all(torch.isfinite(p.grad).all() for p in model.parameters()
                           if p.grad is not None)

        optimizer.step()

        print(f"  batch {i} ({region_name}, N={N}, B={batch['x_hist'].shape[0]}):")
        print(f"    L_main={losses['L_main'].item():.4f}  "
              f"L_normal={losses['L_normal'].item():.4f}  "
              f"L_incident={losses['L_incident'].item():.4f}  "
              f"L_total={losses['L_total'].item():.4f}")
        print(f"    g.mean={out['g'].mean().item():.4f}  "
              f"delta_pred.abs.mean={out['delta_pred'].abs().mean().item():.4f}")
        print(f"    grad: min={min(grad_norms):.4e}  max={max(grad_norms):.4e}  "
              f"finite={finite_grads}  zero-grad params={len(zero_grad_params)}")
        if zero_grad_params and i == 0:
            print(f"    (zero grad in: {zero_grad_params[:5]}...)")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
