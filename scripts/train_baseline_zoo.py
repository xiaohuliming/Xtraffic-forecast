#!/usr/bin/env python3
"""Train LargeST-zoo baselines per region on our HDF5 cache.

Reruns the IGSTGNN Table 4 baseline list under our unified pipeline
(event-anchored or full-window protocol). Model code vendored from
LargeST (baselines/largest_zoo, MIT) without logic changes.

Input convention follows LargeST: channels [z-scored flow, tod fraction,
dow fraction]; models predict normalized flow, predictions are
inverse-transformed before the masked-MAE loss (train stats, scalar).

Deviations from LargeST engines (documented, uniform across models):
  - 30-epoch budget with best-val checkpoint (theirs: 100 ep, patience 30);
    LR milestones scaled accordingly (dcrnn, d2stgnn).
  - DGCRN horizon curriculum disabled (task_level=12 from epoch 1);
    its teacher-forcing schedule still decays via cl_decay_step.
  - D2STGNN warm-up/curriculum disabled (full-horizon loss from epoch 1).

Writes:
  outputs/protocol_v2_noleak/baselines/zoo_<model>/<region>/
    ckpt_best.pt  test_predictions.npz  summary.json  train.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines" / "largest_zoo"))

import h5py
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from dist_net.data import MultiRegionDataset, FullWindowRegionData, RegionData, make_loader
from graph_algo import normalize_adj_mx, calculate_cheb_poly

REGION_TO_GRAPH_KEY = {
    "Alameda":     "alameda",
    "ContraCosta": "contra_costa",
    "Orange":      "orange",
}
MODELS = ["hl", "lstm", "stgcn", "agcrn", "astgcn", "sttn",
          "dcrnn", "dgcrn", "d2stgnn", "dstagnn"]
TPD = 288  # 5-min steps per day (LargeST used 96 on 15-min data)
# DSTAGNN's block residual only supports single-channel input (original paper
# uses flow only); every other model follows LargeST's [flow, tod, dow].
IN_CH = {"dstagnn": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--region", required=True, choices=list(REGION_TO_GRAPH_KEY.keys()))
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/dist_net/region_data_v2_noleak"),
    )
    p.add_argument("--graph-dir", type=Path, default=Path("outputs/region_graphs"))
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default outputs/protocol_v2_noleak/baselines/zoo_<model>")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--protocol", choices=["event", "full_window"], default="event")
    p.add_argument("--stride", type=int, default=1, help="full_window only")
    p.add_argument("--train_frac", type=float, default=0.7, help="full_window only")
    p.add_argument("--val_frac", type=float, default=0.1, help="full_window only")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="2 train batches + 2 val/test batches, 1 epoch (shape/loss check)")
    return p.parse_args()


def pick_device(name: str | None) -> torch.device:
    if name is not None:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dense_adj(edge_index: np.ndarray, N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def train_flow_stats(traffic_path: Path, train_end: int) -> tuple[float, float]:
    """Scalar mean/std of train-window flow over valid sensors (LargeST-style scaler)."""
    with h5py.File(traffic_path, "r") as f:
        flow = f["flow_series_imputed"][:train_end, :, 0]
        mask = f["flow_mask"][:train_end, :, 0].astype(bool)
    vals = flow[mask]
    return float(vals.mean()), float(vals.std())


def build_model(name: str, N: int, device: torch.device, A: np.ndarray,
                args: argparse.Namespace):
    """Returns (model, needs_label, optim_cfg). Hyperparameters = LargeST
    experiment defaults; LR milestones scaled from their 100-epoch budget."""
    T_h, T_p = 12, 12
    C_in = IN_CH.get(name, 3)
    base = dict(node_num=N, input_dim=C_in, output_dim=1, seq_len=T_h, horizon=T_p)
    cfg = {"lr": 1e-3, "wd": 1e-4, "clip": 5.0, "sched": None}

    if name == "lstm":
        from lstm import LSTM
        model = LSTM(init_dim=32, hid_dim=64, end_dim=512, layer=2, dropout=0.1, **base)
        return model, False, cfg
    if name == "stgcn":
        from stgcn import STGCN
        Ab = A.copy()
        np.fill_diagonal(Ab, 0.0)
        gso = torch.tensor(normalize_adj_mx(Ab, "scalap")[0], dtype=torch.float32).to(device)
        Kt, Ks, block_num = 3, 3, 2
        blocks = [[C_in]] + [[64, 16, 64]] * block_num
        Ko = T_h - (Kt - 1) * 2 * block_num
        blocks += [[128, 128]] if Ko > 0 else [[128]]
        blocks += [[T_p]]
        model = STGCN(gso=gso, blocks=blocks, Kt=Kt, Ks=Ks, dropout=0.5, **base)
        cfg.update(wd=5e-4, clip=0.0, sched=lambda opt: StepLR(opt, step_size=10, gamma=0.95))
        return model, False, cfg
    if name == "agcrn":
        from agcrn import AGCRN
        model = AGCRN(embed_dim=10, rnn_unit=64, num_layer=2, cheb_k=2, **base)
        # LargeST's AGCRN_Engine re-inits all params (weights_pool/bias_pool are
        # torch.FloatTensor allocations, i.e. uninitialized memory)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)
        cfg.update(wd=0.0, clip=0.0)
        return model, False, cfg
    if name == "astgcn":
        from astgcn import ASTGCN
        Ab = (A > 0).astype(np.float32)
        np.fill_diagonal(Ab, 0.0)
        L_tilde = normalize_adj_mx(Ab, "scalap")[0]
        cheb = [torch.from_numpy(m).float().to(device) for m in calculate_cheb_poly(L_tilde, 3)]
        model = ASTGCN(device=device, cheb_poly=cheb, order=3, nb_block=2,
                       nb_chev_filter=64, nb_time_filter=64, time_stride=1, **base)
        return model, False, cfg
    if name == "sttn":
        from sttn import STTN
        supports = [torch.tensor(m, dtype=torch.float32).to(device)
                    for m in normalize_adj_mx(A, "doubletransition")]
        model = STTN(device=device, supports=supports, blocks=2, mlp_expand=2,
                     hidden_channels=32, end_channels=512, dropout=0.1, **base)
        return model, False, cfg
    if name == "dcrnn":
        from dcrnn import DCRNN
        model = DCRNN(device=device, adj_mx=A, n_filters=64, max_diffusion_step=2,
                      filter_type="doubletransition", num_rnn_layers=2,
                      cl_decay_steps=2000, **base)
        cfg.update(lr=1e-2, wd=0.0,
                   sched=lambda opt: MultiStepLR(opt, milestones=[10, 20, 27], gamma=0.1))
        return model, True, cfg
    if name == "dgcrn":
        from dgcrn import DGCRN
        supports = [torch.tensor(m, dtype=torch.float32).to(device)
                    for m in normalize_adj_mx(A, "doubletransition")]
        model = DGCRN(device=device, predefined_adj=supports, gcn_depth=2, rnn_size=64,
                      hyperGNN_dim=16, node_dim=40, middle_dim=2,
                      list_weight=[0.05, 0.95, 0.95], tpd=TPD, tanhalpha=3,
                      cl_decay_step=2000, dropout=0.3, **base)
        return model, True, cfg
    if name == "d2stgnn":
        from d2stgnn import D2STGNN
        adjs = [torch.tensor(m, dtype=torch.float32).to(device)
                for m in normalize_adj_mx(A, "doubletransition")]
        margs = dict(num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
                     layer=5, k_t=3, k_s=2, gap=3, tpd=TPD, dropout=0.1,
                     adjs=adjs, seq_len=T_h, horizon=T_p, node_num=N)
        model = D2STGNN(model_args=margs, **base)
        cfg.update(lr=2e-3, wd=1e-5,
                   sched=lambda opt: MultiStepLR(opt, milestones=[1, 11, 14, 17, 20, 23, 26],
                                                 gamma=0.5))
        return model, False, cfg
    if name == "dstagnn":
        from dstagnn import DSTAGNN
        Ab = (A > 0).astype(np.float32)
        np.fill_diagonal(Ab, 0.0)
        L_tilde = normalize_adj_mx(Ab, "scalap")[0]
        cheb = [torch.from_numpy(m).float().to(device) for m in calculate_cheb_poly(L_tilde, 2)]
        adj_pa = torch.tensor(Ab).to(device)
        model = DSTAGNN(device=device, cheb_poly=cheb, order=2, nb_block=2,
                        nb_chev_filter=32, nb_time_filter=32, time_stride=1,
                        adj_pa=adj_pa, d_model=512, d_k=32, d_v=32, n_head=3, **base)
        cfg.update(lr=1e-4, wd=0.0, clip=0.0)
        return model, False, cfg
    raise ValueError(name)


def make_input(batch: dict, mean: float, std: float, device: torch.device,
               in_ch: int = 3) -> torch.Tensor:
    """(B,N,T_h,3) raw + (B,T_h,2) time_feat -> (B,T_h,N,in_ch) [flow_z, tod, dow]."""
    flow = batch["x_hist"][..., 0].to(device)                 # (B, N, T_h)
    flow = ((flow - mean) / std).permute(0, 2, 1)             # (B, T_h, N)
    if in_ch == 1:
        return flow.unsqueeze(-1)
    tf = batch["time_feat"].to(device)                        # (B, T_h, 2)
    tf = tf[:, :, None, :].expand(-1, -1, flow.size(2), -1)   # (B, T_h, N, 2)
    return torch.cat([flow.unsqueeze(-1), tf], dim=-1)


def make_label(batch: dict, mean: float, std: float, device: torch.device) -> torch.Tensor:
    """(B,N,T_p) raw flow -> (B,T_p,N,1) normalized, for teacher forcing."""
    y = batch["y_true"][..., 0].to(device)
    return ((y - mean) / std).permute(0, 2, 1).unsqueeze(-1)


def forward_model(model, name: str, x: torch.Tensor, label: torch.Tensor | None,
                  iter_cnt: int, T_p: int, N: int, device: torch.device) -> torch.Tensor:
    """Returns (B, N, T_p) normalized prediction."""
    if label is None:
        label = torch.zeros(x.size(0), T_p, N, 1, device=device)
    if name == "dcrnn":
        out = model(x, label, iter_cnt)
    elif name == "dgcrn":
        out = model(x, label, iter_cnt, task_level=T_p)
    else:
        out = model(x)
    return out.squeeze(-1).permute(0, 2, 1).contiguous()


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    num = (pred - target).abs() * mask
    return num.sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, name, loader, device, mean, std, T_p, N, limit=0) -> float:
    model.eval()
    total, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if limit and bi >= limit:
            break
        x = make_input(batch, mean, std, device, IN_CH.get(name, 3))
        y_true = batch["y_true"][..., 0].to(device)
        y_mask = batch["y_mask"][..., 0].to(device)
        pred_n = forward_model(model, name, x, None, 0, T_p, N, device)
        pred = pred_n * std + mean
        total += float(masked_mae(pred, y_true, y_mask).item()) * x.size(0)
        n += x.size(0)
    model.train()
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}\nmodel: {args.model}\nregion: {args.region}", flush=True)

    if args.protocol == "full_window":
        rd_cls = FullWindowRegionData
        rd_kw = {"stride": args.stride, "train_frac": args.train_frac, "val_frac": args.val_frac}
        suffix = "_fullwindow"
    else:
        rd_cls, rd_kw, suffix = RegionData, {}, ""
    out_base = args.out_dir or Path(
        f"outputs/protocol_v2_noleak/baselines/zoo_{args.model}{suffix}"
    )
    out_dir = out_base / args.region
    out_dir.mkdir(parents=True, exist_ok=True)

    def mk(split):
        return MultiRegionDataset(
            region_names=[args.region], data_dir=args.data_dir,
            graph_dir=args.graph_dir, split=split, lazy=False,
            region_data_cls=rd_cls, region_data_kwargs=rd_kw,
        )
    train_ds, val_ds, test_ds = mk("train"), mk("val"), mk("test")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    print(f"N={N}  T_h={T_h}  T_p={T_p}", flush=True)
    print(f"|train|={len(train_ds)}  |val|={len(val_ds)}  |test|={len(test_ds)}", flush=True)

    mean, std = train_flow_stats(
        args.data_dir / f"{args.region}_traffic.h5", rdata.train_end
    )
    print(f"flow scaler: mean={mean:.3f} std={std:.3f}", flush=True)

    A = dense_adj(rdata.edge_index, N)

    if args.model == "hl":
        run_hl(args, test_ds, rdata, out_dir, N, T_p)
        return

    model, needs_label, cfg = build_model(args.model, N, device, A, args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model} params: {n_params:,}", flush=True)

    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = cfg["sched"](optimizer) if cfg["sched"] else None
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size

    epochs = 1 if args.smoke else args.epochs
    best_val = float("inf")
    no_improve = 0
    iter_cnt = 0
    history = []
    with open(out_dir / "train.log", "w", encoding="utf-8") as lf:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            sum_loss, n = 0.0, 0
            for bi, batch in enumerate(train_loader):
                if args.smoke and bi >= 2:
                    break
                x = make_input(batch, mean, std, device, IN_CH.get(args.model, 3))
                y_true = batch["y_true"][..., 0].to(device)
                y_mask = batch["y_mask"][..., 0].to(device)
                label = make_label(batch, mean, std, device) if needs_label else None
                pred_n = forward_model(model, args.model, x, label, iter_cnt, T_p, N, device)
                pred = pred_n * std + mean
                loss = masked_mae(pred, y_true, y_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                optimizer.step()
                iter_cnt += 1
                sum_loss += float(loss.item()) * x.size(0)
                n += x.size(0)
                if bi % 100 == 0:
                    print(f"  ep{epoch:02d} batch {bi}/{n_batches} L={loss.item():.3f}",
                          flush=True)
            if scheduler is not None:
                scheduler.step()
            train_avg = sum_loss / max(n, 1)
            val = evaluate(model, args.model, val_loader, device, mean, std, T_p, N,
                           limit=2 if args.smoke else 0)
            t = time.time() - t0
            print(f"==> ep{epoch:02d} train L={train_avg:.4f}  val L={val:.4f}  ({t:.0f}s)",
                  flush=True)
            history.append({"epoch": epoch, "train_L_main": train_avg,
                            "val_L_main": val, "time_s": t})
            lf.write(json.dumps(history[-1]) + "\n")
            lf.flush()
            if val < best_val:
                best_val = val
                no_improve = 0
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_L_main": val, "config": vars(args) | {"n_params": n_params},
                            "N": N, "T_h": T_h, "T_p": T_p,
                            "flow_mean": mean, "flow_std": std},
                           out_dir / "ckpt_best.pt")
                print("    saved best", flush=True)
            else:
                no_improve += 1
                if args.patience and no_improve >= args.patience:
                    print(f"    early stop after {args.patience} stale epochs", flush=True)
                    break

    print(f"\nbest val L_main: {best_val:.4f}", flush=True)

    print("\n=== test inference ===", flush=True)
    best = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()
    write_test_outputs(model, args.model, test_loader, test_ds, rdata, device,
                       mean, std, T_p, N, out_dir, best_val, n_params,
                       limit=2 if args.smoke else 0)


def run_hl(args, test_ds, rdata, out_dir, N, T_p) -> None:
    """History Last: repeat the last observed frame for all horizons. No training."""
    loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    S = len(test_ds)
    pred = np.empty((S, T_p, N), dtype=np.float32)
    actual = np.empty((S, T_p, N), dtype=np.float32)
    msk = np.empty((S, T_p, N), dtype=bool)
    aff = np.empty((S, N), dtype=bool)
    ss = np.empty((S,), dtype=np.int64)
    rc = np.empty((S,), dtype=np.int64)
    cur = 0
    for batch in loader:
        last = batch["x_hist"][..., -1, 0].numpy()            # (B, N) raw flow at t
        b = last.shape[0]
        pred[cur:cur + b] = np.repeat(last[:, None, :], T_p, axis=1)
        actual[cur:cur + b] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
        msk[cur:cur + b] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
        aff[cur:cur + b] = batch["affected_mask"].numpy()
        ss[cur:cur + b] = batch["sample_start"].numpy()
        rc[cur:cur + b] = batch["region_code"].numpy()
        cur += b
    save_outputs(out_dir, rc, ss, rdata, pred, actual, msk, aff,
                 best_val=float("nan"), n_params=0)


def write_test_outputs(model, name, test_loader, test_ds, rdata, device,
                       mean, std, T_p, N, out_dir, best_val, n_params, limit=0) -> None:
    S = len(test_ds)
    pred = np.empty((S, T_p, N), dtype=np.float32)
    actual = np.empty((S, T_p, N), dtype=np.float32)
    msk = np.empty((S, T_p, N), dtype=bool)
    aff = np.empty((S, N), dtype=bool)
    ss = np.empty((S,), dtype=np.int64)
    rc = np.empty((S,), dtype=np.int64)
    cur = 0
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if limit and bi >= limit:
                S = cur
                break
            x = make_input(batch, mean, std, device, IN_CH.get(name, 3))
            pred_n = forward_model(model, name, x, None, 0, T_p, N, device)
            p = (pred_n * std + mean).permute(0, 2, 1).cpu().numpy()  # (B, T_p, N)
            b = p.shape[0]
            pred[cur:cur + b] = p
            actual[cur:cur + b] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            msk[cur:cur + b] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            aff[cur:cur + b] = batch["affected_mask"].numpy()
            ss[cur:cur + b] = batch["sample_start"].numpy()
            rc[cur:cur + b] = batch["region_code"].numpy()
            cur += b
    save_outputs(out_dir, rc[:S], ss[:S], rdata, pred[:S], actual[:S], msk[:S], aff[:S],
                 best_val, n_params)


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


def save_outputs(out_dir, rc, ss, rdata, pred, actual, msk, aff, best_val, n_params) -> None:
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        region_code=rc, sample_start=ss,
        region_node_idx=rdata.region_idx.astype(np.int64),
        pred_raw_flow=pred, actual_future_flow=actual,
        y_mask_flow=msk, affected_mask=aff,
    )
    summary = {**flow_metrics(pred, actual, msk, aff),
               "best_val_L_main": best_val, "n_params": n_params,
               **rdata.result_metadata()}
    print(f"\ntest MAE: all={summary['all']:.3f}  affected={summary['affected']:.3f}  "
          f"unaffected={summary['unaffected']:.3f}  "
          f"h3={summary['h3']:.3f} h6={summary['h6']:.3f} h12={summary['h12']:.3f}\n"
          f"test RMSE={summary['rmse']:.3f}  MAPE={summary['mape']:.2f}%",
          flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
