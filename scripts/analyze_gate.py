"""Analyze D3 gated fusion: record alpha on the test set + run sanity checks.

Rebuilds the model from the checkpoint's saved config + dims, re-runs test inference
with return_components=True to capture alpha, then computes the sanity checks from the
design spec. All in-loop tensors use (B, N, T_p) order (model-native).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))
from fourier_dual_net.model import FourierDualNet
from dist_net.data import MultiRegionDataset, make_loader


def build_adj_supports(edge_index, N, device):
    # Verbatim from scripts/train_fourier_dual_net.py:77-87 (row-norm random walk, 2 supports)
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    A_fwd = deg_inv[:, None] * A
    A_bwd = deg_inv[:, None] * A.T
    return [torch.from_numpy(A_fwd).to(device), torch.from_numpy(A_bwd).to(device)]


def load_baseline_node_mae(region):
    p = ROOT / "outputs" / "diagnostics" / f"per_node_{region}.npz"
    return np.load(p)["node_fdn"] if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)        # ckpt_best.pt
    ap.add_argument("--region", type=str, required=True)
    ap.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--graph_dir", type=Path, default=Path("data/graphs"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    N, C_x, T_h, T_p = state["N"], state["C_x"], state["T_h"], state["T_p"]

    test_ds = MultiRegionDataset(region_names=[args.region], data_dir=args.data_dir,
                                 graph_dir=args.graph_dir, split="test", lazy=False)
    rdata = test_ds.regions[args.region]
    supports = build_adj_supports(rdata.edge_index, N, device)
    tod_slots = np.asarray(rdata.tod)        # (T,) per-timestep tod slot 0..287

    model = FourierDualNet(
        num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
        K=cfg.get("K", 3), decomp_mode=cfg.get("decomp_mode", "learnable"),
        in_dim_flow=C_x, nhid=cfg.get("nhid", 32), dropout=cfg.get("dropout", 0.3),
        device=device,
        main_blocks=cfg.get("main_blocks", 4), main_layers=cfg.get("main_layers", 2),
        pert_blocks=cfg.get("pert_blocks", 4), pert_layers=cfg.get("pert_layers", 2),
        use_time_emb=cfg.get("use_time_emb", False),
        use_cross_attn=cfg.get("use_cross_attn", False),
        use_gated_fusion=True,
        gate_d_node=cfg.get("gate_d_node", 8), gate_hidden=cfg.get("gate_hidden", 32),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    A, SS, M, P, Y, ER = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x_hist"].to(device)
            tf = batch["time_feat"].to(device)
            assert tf.dim() == 3 and tf.size(-1) == 2, f"time_feat shape {tf.shape}"
            y, comp = model(x, time_feat=tf, return_components=True)
            A.append(comp["alpha"].cpu().numpy())            # (B,N,T_p)
            P.append(y.cpu().numpy())                         # (B,N,T_p)
            Y.append(batch["y_true"][..., 0].numpy())        # (B,N,T_p)
            M.append(batch["y_mask"][..., 0].numpy())        # (B,N,T_p)
            SS.append(batch["sample_start"].numpy())         # (B,)
            ER.append(model.gated_fusion._energy_ratio(comp["x_main"], comp["x_pert"]).cpu().numpy())

    alpha = np.concatenate(A);  pred = np.concatenate(P);  actual = np.concatenate(Y)
    mask = np.concatenate(M);   ss = np.concatenate(SS);   energy = np.concatenate(ER)
    np.savez(args.out_dir / "alpha_test.npz", alpha=alpha, sample_start=ss, energy_ratio=energy)

    tod = tod_slots[ss + 1]                  # first predicted step's tod slot (exact)
    midday = (tod >= 10 * 12) & (tod < 15 * 12)
    rush = ((tod >= 6 * 12) & (tod < 10 * 12)) | ((tod >= 15 * 12) & (tod < 19 * 12))

    diff = np.abs(pred - actual) * mask                      # (S,N,T_p)
    cnt = mask.sum(axis=(0, 2)).astype(np.float64)           # per node
    node_mae = np.where(cnt > 0, diff.sum(axis=(0, 2)) / np.maximum(cnt, 1), np.nan)
    alpha_node = alpha.mean(axis=(0, 2))                     # (N,)
    valid = ~np.isnan(node_mae)

    L = [f"# Sanity checks — {args.region}",
         f"output_scale = {float(model.gated_fusion.output_scale):.4f}"]
    a_mean, a_std = float(alpha.mean()), float(alpha.std())
    L.append(f"[{'PASS' if (abs(a_mean-0.5)<0.3 and a_std>0.1) else 'FAIL'}] C1 variance: mean={a_mean:.3f} std={a_std:.3f}")
    corr_node = float(np.corrcoef(node_mae[valid], alpha_node[valid])[0, 1])
    L.append(f"[{'PASS' if corr_node < -0.2 else 'FAIL'}] C2a corr(node_MAE, alpha) = {corr_node:+.3f} (want <-0.2)")
    a_mid = float(alpha[midday].mean()) if midday.any() else float('nan')
    a_rush = float(alpha[rush].mean()) if rush.any() else float('nan')
    L.append(f"[{'PASS' if a_mid < a_rush else 'FAIL'}] C2b alpha_midday={a_mid:.3f} < alpha_rush={a_rush:.3f}")
    corr_e = float(np.corrcoef(energy.ravel(), alpha.mean(axis=2).ravel())[0, 1])
    L.append(f"[{'PASS' if corr_e > 0.2 else 'FAIL'}] C2c corr(energy, alpha) = {corr_e:+.3f} (want >0.2)")
    a_h1, a_hL = float(alpha[..., 0].mean()), float(alpha[..., -1].mean())
    L.append(f"[{'PASS' if a_hL > a_h1 else 'FAIL'}] C3 alpha[h12]={a_hL:.3f} > alpha[h1]={a_h1:.3f}")

    base = load_baseline_node_mae(args.region)
    if base is not None:
        bv = base[~np.isnan(base)]
        p99b, p99n = float(np.quantile(bv, 0.99)), float(np.quantile(node_mae[valid], 0.99))
        L.append(f"[{'PASS' if p99n < p99b - 0.5 else 'FAIL'}] C4 p99 node MAE {p99n:.2f} vs baseline {p99b:.2f}")
    else:
        L.append("[SKIP] C4 baseline diagnostics not found (run scripts/diagnose_fdn_failures.py first)")
    if midday.any():
        md = (np.abs(pred[midday]-actual[midday]) * mask[midday]).sum() / max(mask[midday].sum(), 1)
        L.append(f"[INFO] midday MAE (D3) = {float(md):.3f}")

    npass = sum(s.startswith('[PASS]') for s in L)
    ntot = sum(s.startswith('[PASS]') or s.startswith('[FAIL]') for s in L)
    L.append(f"\nSummary: {npass}/{ntot} pass.")
    report = "\n".join(L)
    (args.out_dir / "sanity_checks.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
