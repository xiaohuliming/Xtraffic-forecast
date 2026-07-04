#!/usr/bin/env python3
"""Ablation B-new: v0c plus a reported-incident gated correction branch.

This replaces the previous logistic-regression label branch. The main v0c path
is kept unchanged:

    main_pred = adaptive_anchor(alpha_base) + GraphWaveNet(residual_history)

The new branch reads only incident records that have happened at or before the
forecast start time. It does not use future affected labels as inputs, and it
does not modify alpha_base. Instead, it learns a gated correction:

    final_pred = main_pred + incident_gate * incident_delta

The affected_mask is still used only for evaluation grouping.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from A_v0c_original_2024 import (
    CHANNEL_FLOW,
    HERE,
    MatchedIncident,
    RegionBundle,
    V0COriginal,
    XTraffic2024Dataset,
    add_common_args,
    build_adj_supports,
    collate,
    compute_sd_res,
    jsonable_args,
    load_incidents_2024,
    make_gwn,
    masked_mae,
    match_incidents,
    prepare_bundle,
    set_seed,
)


INC_ACTIVE = 0
INC_RECENCY = 1
INC_COUNT = 2
INC_ANCHOR = 3
INCIDENT_DIM = 4


def build_reported_incident_features(
    matches: list[MatchedIncident],
    total_steps: int,
    num_nodes: int,
    lookback_steps: int,
    tau_steps: float,
) -> np.ndarray:
    """Create causal incident-record features with shape (T, N, 4).

    Feature channels:
      0 active: 1 if a reported incident has been near the sensor recently.
      1 recency: exp(-age/tau), larger for more recent reported incidents.
      2 count: clipped local incident count in the recent window.
      3 anchor: 1 for the nearest matched sensor of the reported incident.

    The tensor is written only at times t >= incident_start, so a sample at time
    t can never see incident records from future time steps.
    """
    if lookback_steps < 0:
        raise ValueError("lookback_steps must be non-negative")
    tau_steps = max(float(tau_steps), 1e-6)
    features = np.zeros((total_steps, num_nodes, INCIDENT_DIM), dtype=np.float32)
    count_scale = 1.0 / 3.0

    for m in matches:
        start = int(m.start_idx)
        if start < 0 or start >= total_steps:
            continue
        end = min(total_steps, start + int(lookback_steps) + 1)
        cand = np.asarray(m.candidate_idx, dtype=np.int64)
        cand = cand[(0 <= cand) & (cand < num_nodes)]
        if cand.size == 0:
            continue
        anchor = int(m.anchor_idx)
        anchor_ok = 0 <= anchor < num_nodes
        for tt in range(start, end):
            age = tt - start
            recency = math.exp(-float(age) / tau_steps)
            features[tt, cand, INC_ACTIVE] = 1.0
            features[tt, cand, INC_RECENCY] = np.maximum(features[tt, cand, INC_RECENCY], recency)
            features[tt, cand, INC_COUNT] += count_scale
            if anchor_ok:
                features[tt, anchor, INC_ANCHOR] = 1.0

    features[..., INC_COUNT] = np.clip(features[..., INC_COUNT], 0.0, 1.0)
    return features


def load_matched_incidents_for_features(args: argparse.Namespace, bundle: RegionBundle) -> list[MatchedIncident]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    inc = load_incidents_2024(data_dir)
    matches = match_incidents(
        inc,
        bundle.region_meta,
        bundle.traffic.shape[0],
        candidate_pm_radius=args.candidate_pm_radius,
        anchor_pm_radius=args.anchor_pm_radius,
    )
    if args.max_events and args.max_events > 0:
        matches = matches[:args.max_events]
    return matches


class XTrafficIncident2024Dataset(XTraffic2024Dataset):
    """Same samples as A, with causal incident-record history added."""

    def __init__(
        self,
        bundle: RegionBundle,
        incident_features: np.ndarray,
        split: str,
        history: int,
        horizon: int,
        max_samples: int = 0,
        seed: int = 42,
    ):
        super().__init__(bundle, split, history, horizon, max_samples, seed)
        self.incident_features = incident_features
        if incident_features.shape[:2] != bundle.traffic.shape[:2]:
            raise ValueError(
                f"Incident features shape {incident_features.shape[:2]} does not match traffic "
                f"{bundle.traffic.shape[:2]}"
            )

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = super().__getitem__(idx)
        t = int(item["sample_start"])
        hist = np.arange(t - self.history + 1, t + 1)
        inc_hist = self.incident_features[hist].transpose(1, 0, 2).astype(np.float32)
        item["incident_features"] = inc_hist
        return item


class V0CIncidentGate(V0COriginal):
    """v0c with a reported-incident branch fused by prediction-level gating."""

    def __init__(
        self,
        num_nodes: int,
        supports: list[torch.Tensor],
        horizon: int,
        device: torch.device,
        nhid: int = 32,
        dropout: float = 0.3,
        incident_nhid: int = 16,
        incident_dropout: float = 0.2,
    ):
        super().__init__(num_nodes, supports, horizon, device, nhid, dropout)
        self.incident_delta = make_gwn(
            num_nodes, supports, device, INCIDENT_DIM, horizon, incident_nhid, incident_dropout
        )
        self.incident_gate = make_gwn(
            num_nodes, supports, device, INCIDENT_DIM, horizon, incident_nhid, incident_dropout
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_baseline: torch.Tensor,
        y_baseline: torch.Tensor,
        incident_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        main_pred, details = super().forward(x_hist, x_baseline, y_baseline)

        inc_in = self._to_gwnet(incident_features)
        delta_z = self._from_gwnet(self.incident_delta(inc_in))
        gate_logits = self._from_gwnet(self.incident_gate(inc_in))

        # Keep the correction silent for nodes with no reported incident in the
        # available history window. This is causal and uses only the input branch.
        active_now = incident_features[..., INC_ACTIVE].amax(dim=2).unsqueeze(-1)
        incident_gate = torch.sigmoid(gate_logits) * active_now
        incident_delta = delta_z * self.sd_res
        pred = main_pred + incident_gate * incident_delta

        details.update(
            {
                "main_pred": main_pred,
                "incident_gate": incident_gate,
                "incident_delta": incident_delta,
                "incident_active": active_now,
            }
        )
        return pred, details


def _accumulate_mae(
    diff: torch.Tensor,
    mask: torch.Tensor,
    sums: dict[str, float],
    counts: dict[str, float],
    name: str,
) -> None:
    mf = mask.float()
    sums[name] += float((diff * mf).sum().item())
    counts[name] += float(mf.sum().item())


def evaluate_incident_gate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums = {
        "all": 0.0,
        "affected": 0.0,
        "unaffected": 0.0,
        "main_all": 0.0,
        "main_affected": 0.0,
        "main_unaffected": 0.0,
    }
    counts = {k: 0.0 for k in sums}
    alpha_aff, alpha_unaff = [], []
    gate_aff, gate_unaff = [], []
    delta_aff, delta_unaff = [], []
    active_aff, active_unaff = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_hist"].to(device)
            xb = batch["x_baseline"].to(device)
            yb = batch["y_baseline"].to(device)
            inc = batch["incident_features"].to(device)
            target = batch["y_true"][..., CHANNEL_FLOW].to(device)
            mask = batch["y_mask_flow"].to(device)
            aff = batch["affected_mask"].to(device)

            pred, details = model(x, xb, yb, inc)
            main_pred = details["main_pred"]
            diff = (pred - target).abs()
            main_diff = (main_pred - target).abs()
            aff3 = aff.unsqueeze(-1).expand_as(mask)
            unaff3 = ~aff3

            _accumulate_mae(diff, mask, sums, counts, "all")
            _accumulate_mae(diff, mask & aff3, sums, counts, "affected")
            _accumulate_mae(diff, mask & unaff3, sums, counts, "unaffected")
            _accumulate_mae(main_diff, mask, sums, counts, "main_all")
            _accumulate_mae(main_diff, mask & aff3, sums, counts, "main_affected")
            _accumulate_mae(main_diff, mask & unaff3, sums, counts, "main_unaffected")

            alpha = details.get("alpha")
            gate = details["incident_gate"]
            delta_abs = details["incident_delta"].abs()
            active = details["incident_active"]
            if aff.any():
                alpha_aff.append(alpha[aff].detach().cpu())
                gate_aff.append(gate[aff].detach().cpu())
                delta_aff.append(delta_abs[aff].detach().cpu())
                active_aff.append(active[aff].detach().cpu())
            if (~aff).any():
                alpha_unaff.append(alpha[~aff].detach().cpu())
                gate_unaff.append(gate[~aff].detach().cpu())
                delta_unaff.append(delta_abs[~aff].detach().cpu())
                active_unaff.append(active[~aff].detach().cpu())

    out = {k: sums[k] / max(counts[k], 1.0) for k in sums}
    if alpha_aff:
        out["alpha_affected"] = float(torch.cat(alpha_aff).mean().item())
        out["incident_gate_affected"] = float(torch.cat(gate_aff).mean().item())
        out["incident_delta_abs_affected"] = float(torch.cat(delta_aff).mean().item())
        out["incident_active_affected"] = float(torch.cat(active_aff).mean().item())
    if alpha_unaff:
        out["alpha_unaffected"] = float(torch.cat(alpha_unaff).mean().item())
        out["incident_gate_unaffected"] = float(torch.cat(gate_unaff).mean().item())
        out["incident_delta_abs_unaffected"] = float(torch.cat(delta_unaff).mean().item())
        out["incident_active_unaffected"] = float(torch.cat(active_unaff).mean().item())
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(out_dir=HERE / "outputs" / "B_v0c_incident_gate_2024")
    p.add_argument(
        "--incident-lookback-steps",
        type=int,
        default=24,
        help="How many previous 5-minute steps a reported incident remains visible to the branch.",
    )
    p.add_argument(
        "--incident-tau-steps",
        type=float,
        default=12.0,
        help="Exponential recency decay tau, in 5-minute steps.",
    )
    p.add_argument("--incident-nhid", type=int, default=16)
    p.add_argument("--incident-dropout", type=float, default=0.2)
    p.add_argument(
        "--gate-l1-weight",
        type=float,
        default=0.0,
        help="Optional sparsity penalty on the incident gate. Default keeps training comparable to A.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_bundle(args)
    matches = load_matched_incidents_for_features(args, bundle)
    print(
        f"[data] building causal reported-incident features "
        f"matches={len(matches)} lookback={args.incident_lookback_steps} tau={args.incident_tau_steps}",
        flush=True,
    )
    incident_features = build_reported_incident_features(
        matches,
        total_steps=bundle.traffic.shape[0],
        num_nodes=bundle.traffic.shape[1],
        lookback_steps=args.incident_lookback_steps,
        tau_steps=args.incident_tau_steps,
    )
    active_rate = float((incident_features[..., INC_ACTIVE] > 0).mean())
    print(f"[data] incident feature tensor={incident_features.shape} active_rate={active_rate:.6f}", flush=True)

    train_ds = XTrafficIncident2024Dataset(
        bundle, incident_features, "train", args.history, args.horizon, args.max_train_samples, args.seed
    )
    val_ds = XTrafficIncident2024Dataset(
        bundle, incident_features, "val", args.history, args.horizon, args.max_val_samples, args.seed
    )
    test_ds = XTrafficIncident2024Dataset(
        bundle, incident_features, "test", args.history, args.horizon, args.max_test_samples, args.seed
    )
    if not train_ds or not val_ds or not test_ds:
        raise RuntimeError("Empty split. Try a different region, offsets, or max-events setting.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )

    supports = build_adj_supports(bundle.edge_index, bundle.traffic.shape[1], device)
    model = V0CIncidentGate(
        bundle.traffic.shape[1],
        supports,
        args.horizon,
        device,
        args.nhid,
        args.dropout,
        args.incident_nhid,
        args.incident_dropout,
    ).to(device)
    sd_res = compute_sd_res(bundle, args.train_frac)
    model.sd_res.fill_(sd_res)
    print(
        f"[model] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
        f"sd_res={sd_res:.3f} incident_nhid={args.incident_nhid} gate_l1={args.gate_l1_weight:.4f}",
        flush=True,
    )

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs * len(train_loader), 1), eta_min=args.lr * 1e-2)
    best = float("inf")
    best_path = args.out_dir / f"{args.region}_best.pt"
    log_rows = []

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        flow_total, gate_total, total_loss, n = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch["x_hist"].to(device)
            xb = batch["x_baseline"].to(device)
            yb = batch["y_baseline"].to(device)
            inc = batch["incident_features"].to(device)
            target = batch["y_true"][..., CHANNEL_FLOW].to(device)
            mask = batch["y_mask_flow"].to(device)

            pred, details = model(x, xb, yb, inc)
            flow_loss = masked_mae(pred, target, mask)
            gate_penalty = details["incident_gate"].mean()
            loss = flow_loss + args.gate_l1_weight * gate_penalty

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            bs = x.size(0)
            flow_total += float(flow_loss.item()) * bs
            gate_total += float(gate_penalty.item()) * bs
            total_loss += float(loss.item()) * bs
            n += bs

        val = evaluate_incident_gate(model, val_loader, device)
        row = {
            "epoch": ep,
            "train_loss": total_loss / max(n, 1),
            "train_flow_mae": flow_total / max(n, 1),
            "train_gate_mean": gate_total / max(n, 1),
            **{f"val_{k}": v for k, v in val.items()},
        }
        log_rows.append(row)
        print(
            f"[ep {ep:02d}] flow={row['train_flow_mae']:.4f} "
            f"gate={row['train_gate_mean']:.4f} val_all={val['all']:.4f} "
            f"main_val_all={val['main_all']:.4f} time={time.time()-t0:.1f}s",
            flush=True,
        )
        if val["all"] < best:
            best = val["all"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": jsonable_args(args),
                    "sd_res": sd_res,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate_incident_gate(model, test_loader, device)
    summary = {
        "variant": "B_v0c_incident_gate_2024",
        "region": args.region,
        "nodes": int(bundle.traffic.shape[1]),
        "samples": {s: int(sum(x["split"] == s for x in bundle.samples)) for s in ["train", "val", "test"]},
        "best_val_all": best,
        "test": test,
        "incident_features": {
            "matched_incidents": int(len(matches)),
            "dim": INCIDENT_DIM,
            "active_rate": active_rate,
            "lookback_steps": int(args.incident_lookback_steps),
            "tau_steps": float(args.incident_tau_steps),
            "channels": ["active", "recency", "count", "anchor"],
        },
        "args": jsonable_args(args),
    }
    (args.out_dir / f"{args.region}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(args.out_dir / f"{args.region}_training_log.csv", index=False)
    print("[test]", json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
