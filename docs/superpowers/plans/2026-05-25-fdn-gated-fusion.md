# FDN Gated Fusion (D3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-(node, horizon) learned gate on top of FourierDualNet's two-branch output, replacing unconditional `y_main + y_pert` with `2 * (α·y_main + (1-α)·y_pert)`, where α is conditioned on spectral energy (flow channel only), node identity, and time-of-day.

**Architecture:** New `GatedFusion` module in `fourier_dual_net/model.py`, toggled by `use_gated_fusion: bool` (default False → exact baseline). Init makes ep0 output bit-for-bit identical to the additive baseline. Trained from scratch, **without** `--use_time_emb` (to isolate the gate from the previously-tested +A factor).

**Tech Stack:** PyTorch, GraphWaveNet (vendored under `baselines/GraphWaveNet/`), numpy, pandas. No new deps.

**Spec:** [docs/superpowers/specs/2026-05-25-fdn-gated-fusion-design.md](../specs/2026-05-25-fdn-gated-fusion-design.md)

---

## Verified codebase contracts (read first — confirmed against source, do NOT trust memory)

An adversarial review flagged that the first plan draft got several of these wrong; one review claim was *itself* wrong and was re-verified directly against source. Ground truth:

- **Data class**: `from dist_net.data import MultiRegionDataset, make_loader` (the class is `MultiRegionDataset`, NOT `RegionDataset`). There is **no `NUM_TOD_SLOTS` / `slot_per_day` constant** — the value 288 is hardcoded inside `get_sample` (`dist_net/data.py:160`). For time-of-day bucketing, index the per-timestep slot array `rdata.tod` directly (see below) instead of recomputing `(t+1) % 288`.
- **Region object**: `rdata = ds.regions[name]` is a `RegionData`. Access fields as **attributes** (it is NOT subscriptable): `rdata.N`, `rdata.C_meta`, `rdata.T_h` (=12), `rdata.T_p` (=12), `rdata.edge_index`, `rdata.region_idx`, `rdata.static_meta`, `rdata.tod` (int16, shape (T,), each timestep's tod slot 0..287), `rdata.dow`. There is **no `C_x`** field (train hardcodes `C_x = 3`). Source: `dist_net/data.py:42-107`. The per-timestep slot is `rdata.tod[t]`; sample's first predicted step is at absolute index `sample_start + 1`, so its tod slot is `rdata.tod[sample_start + 1]` (always in-range: `sample_start + T_p < T`).
- **supports** (verified at `scripts/train_fourier_dual_net.py:77-87`): row-normalized random walk, **TWO** supports:
  ```python
  A = np.zeros((N, N), dtype=np.float32)
  A[edge_index[0], edge_index[1]] = 1.0
  np.fill_diagonal(A, 1.0)
  deg = A.sum(axis=1)
  deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
  A_fwd = deg_inv[:, None] * A
  A_bwd = deg_inv[:, None] * A.T
  return [torch.from_numpy(A_fwd).to(device), torch.from_numpy(A_bwd).to(device)]
  ```
  (NOT symmetric-normalized, NOT single-support — a review agent guessed that wrong.) Not importable; copy verbatim.
- **Batch keys** (from `RegionData.get_sample`, AST-verified): `x_hist, x_hist_mask, y_true, y_mask, y_baseline, time_enc, time_feat, static_meta, region_code, incident_feat, incident_mask, affected_mask, rel_feat, n_active_incidents, sample_start, sample_idx` (note: **no `rel_mask`**). Flow target = `batch["y_true"][..., 0]`; mask = `batch["y_mask"][..., 0]`; `time_feat` is `(B, T_h, 2)`. The names `actual_future_flow/y_mask_flow/pred_raw_flow` exist ONLY in the saved npz.
- **In-loop tensor order**: model output is `(B, N, T_p)`; `batch["y_true"][...,0]` is `(B, N, T_p)`. The npz saves `(S, T_p, N)` after `.permute(0,2,1)`. analyze_gate.py keeps the in-loop `(B,N,T_p)` order throughout.
- **Checkpoint**: file `ckpt_best.pt` (NOT `best.pt`); state under key `"model_state"` (NOT `"model"`); also `"config"` (=`vars(args)`), `"N"`, `"C_x"`, `"T_h"`, `"T_p"`, `"val_L_main"`, `"epoch"`. Saved only when val improves. Source: `scripts/train_fourier_dual_net.py:235-239`.
- **`--smoke`** sets val=NaN → `NaN < inf` is False → **no checkpoint saved**. For a smoke that yields a checkpoint, use a real `--epochs 1` run.
- **`need_time`** appears in THREE places: `evaluate()` (L102), train loop (L202), test loop (L267).
- **`make_loader(dataset, batch_size, shuffle=False, seed=None)`**.
- **Train defaults**: `--data_dir data/processed`, `--graph_dir data/graphs`, batch_size=48, lr=1e-3, weight_decay=1e-4, seed=42, nhid=32, blocks/layers main=4/2 pert=4/2; optimizer Adam; `CosineAnnealingLR`.

---

## File Plan

| File | Action | Responsibility |
|---|---|---|
| `fourier_dual_net/model.py` | MODIFY | Add `GatedFusion`. Modify `FourierDualNet.__init__`/`forward`; cache `self._last_alpha`. |
| `tests/fourier_dual_net/__init__.py` | CREATE | Empty package marker. |
| `tests/fourier_dual_net/test_gated_fusion.py` | CREATE | Unit tests: shape, exact init==baseline, gradient, flow-only energy, FDN integration. |
| `scripts/train_fourier_dual_net.py` | MODIFY | Add `--use_gated_fusion/--gate_d_node/--gate_hidden/--gate_l1_lambda`; fix all 3 `need_time`; L1 term. |
| `scripts/analyze_gate.py` | CREATE | Rebuild model from ckpt config, record α, run sanity checks. |

---

## Task 1: GatedFusion module + unit tests

**Files:**
- Modify: `fourier_dual_net/model.py` (append `GatedFusion` after `CrossBranchAttention`, before `FourierDualNet`)
- Create: `tests/fourier_dual_net/__init__.py`, `tests/fourier_dual_net/test_gated_fusion.py`

- [ ] **Step 1.1: Test package marker**

```bash
mkdir -p tests/fourier_dual_net && touch tests/fourier_dual_net/__init__.py
```

- [ ] **Step 1.2: Write failing tests** — create `tests/fourier_dual_net/test_gated_fusion.py`:

```python
"""Unit tests for GatedFusion (D3)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))
from fourier_dual_net.model import GatedFusion, FourierDualNet


def _inputs(B=2, N=5, T_p=12, T_h=12, C=3):
    return (torch.randn(B, N, T_h, C), torch.randn(B, N, T_h, C),
            torch.randn(B, N, T_p), torch.randn(B, N, T_p),
            torch.randn(B, T_h, 2))


def test_forward_shape():
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    xm, xp, ym, yp, tf = _inputs()
    y, alpha = g(xm, xp, ym, yp, tf)
    assert y.shape == (2, 5, 12)
    assert alpha.shape == (2, 5, 12)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_init_preserves_baseline_exactly():
    torch.manual_seed(0)
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    xm, xp, ym, yp, tf = _inputs()
    y, alpha = g(xm, xp, ym, yp, tf)
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-6)
    assert torch.allclose(y, ym + yp, atol=1e-4), \
        f"init deviates by max {(y - (ym+yp)).abs().max().item():.4g}"


def test_gradient_flows():
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    xm, xp, ym, yp, tf = _inputs()
    y, _ = g(xm, xp, ym, yp, tf)
    y.sum().backward()
    assert g.gate_mlp[-1].weight.grad is not None
    assert g.gate_mlp[-1].weight.grad.abs().sum().item() > 0   # = ReLU acts, nonzero on step 1
    assert g.output_scale.grad is not None and g.output_scale.grad.abs().item() > 0


def test_energy_ratio_uses_flow_channel_only():
    g = GatedFusion(num_nodes=3, T_p=12, d_node=4)
    B, N, T_h = 2, 3, 12
    xm = torch.zeros(B, N, T_h, 3); xp = torch.zeros(B, N, T_h, 3)
    xm[..., 0] = 1.0; xp[..., 0] = 1.0
    er = g._energy_ratio(xm, xp)
    assert er.shape == (B, N)
    xm2 = xm.clone(); xm2[..., 1] = 1000.0   # huge energy in occupancy only
    assert torch.allclose(er, g._energy_ratio(xm2, xp)), "energy_ratio leaked non-flow channels"


def test_fdn_gated_init_matches_baseline():
    N, T_h, T_p, C = 5, 12, 12, 3
    supports = [torch.eye(N)]
    dev = torch.device("cpu")
    torch.manual_seed(42)
    m_base = FourierDualNet(num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
                            in_dim_flow=C, nhid=8, device=dev, use_time_emb=False)
    torch.manual_seed(42)
    m_gate = FourierDualNet(num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
                            in_dim_flow=C, nhid=8, device=dev, use_time_emb=False,
                            use_gated_fusion=True)
    x = torch.randn(2, N, T_h, C); tf = torch.randn(2, T_h, 2)
    m_base.eval(); m_gate.eval()
    with torch.no_grad():
        md = (m_base(x, time_feat=tf) - m_gate(x, time_feat=tf)).abs().max().item()
    assert md < 1e-4, f"init FDN+gate deviates by {md:.4g}"


def test_fdn_gated_requires_time_feat():
    N, T_h, T_p, C = 5, 12, 12, 3
    m = FourierDualNet(num_nodes=N, supports=[torch.eye(N)], T_h=T_h, T_p=T_p,
                       in_dim_flow=C, nhid=8, device=torch.device("cpu"),
                       use_time_emb=False, use_gated_fusion=True)
    with pytest.raises(AssertionError):
        m(torch.randn(2, N, T_h, C), time_feat=None)
```

- [ ] **Step 1.3: Run, confirm fail**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```
Expected: ImportError — `GatedFusion` not defined.

- [ ] **Step 1.4: Implement GatedFusion** — append to `fourier_dual_net/model.py` (after `CrossBranchAttention`, before `FourierDualNet`):

```python
class GatedFusion(nn.Module):
    """Gated fusion: y = output_scale * (alpha * y_main + (1-alpha) * y_pert).

    alpha (B, N, T_p) conditioned on:
      - spectral_energy_ratio (flow channel only, index 0),
      - node_emb (zero-init), time_avg (ToD/DoW mean over history).
    Init: last-layer weight AND bias = 0 -> alpha == 0.5 exactly; node_emb = 0;
    output_scale = 2.0 -> y == y_main + y_pert at ep0 (bit-for-bit).
    """

    def __init__(self, num_nodes: int, T_p: int = 12, d_node: int = 8, hidden: int = 32):
        super().__init__()
        self.T_p = T_p
        self.d_node = d_node
        self.node_emb = nn.Embedding(num_nodes, d_node)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1 + d_node + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, T_p),
        )
        self.output_scale = nn.Parameter(torch.tensor(2.0))
        with torch.no_grad():
            self.gate_mlp[-1].bias.zero_()
            self.gate_mlp[-1].weight.zero_()
            self.node_emb.weight.zero_()

    @staticmethod
    def _energy_ratio(x_main: torch.Tensor, x_pert: torch.Tensor) -> torch.Tensor:
        e_main = (x_main[..., 0:1] ** 2).mean(dim=(2, 3))   # flow only; (B, N)
        e_pert = (x_pert[..., 0:1] ** 2).mean(dim=(2, 3))
        return e_main / (e_main + e_pert + 1e-6)

    def forward(self, x_main, x_pert, y_main, y_pert, time_feat):
        B, N, T_p = y_main.shape
        assert T_p == self.T_p
        energy = self._energy_ratio(x_main, x_pert)                       # (B, N)
        node_e = self.node_emb.weight.unsqueeze(0).expand(B, N, self.d_node)
        time_avg = time_feat.mean(dim=1).unsqueeze(1).expand(B, N, 2)
        gate_in = torch.cat([energy.unsqueeze(-1), node_e, time_avg], dim=-1)
        alpha = torch.sigmoid(self.gate_mlp(gate_in))                     # (B, N, T_p)
        y = self.output_scale * (alpha * y_main + (1.0 - alpha) * y_pert)
        return y, alpha
```

- [ ] **Step 1.5: Run, confirm partial pass**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```
Expected: the 4 GatedFusion-only tests PASS; the two `test_fdn_*` still fail (FDN not wired — Task 2).

- [ ] **Step 1.6: Commit**

```bash
git add fourier_dual_net/model.py tests/fourier_dual_net/
git commit -m "Add GatedFusion module (zero-init, flow-only energy) + unit tests"
```

---

## Task 2: Integrate GatedFusion into FourierDualNet

**Files:** Modify `fourier_dual_net/model.py` (`FourierDualNet`).

- [ ] **Step 2.1: Confirm patch targets** — read the current `FourierDualNet.__init__` signature tail and `forward` bottom:

```bash
sed -n '140,250p' fourier_dual_net/model.py
```
Confirm signature ends `sensor_meta_dim: int | None = None):` and forward ends with the `y_main`/`y_pert`/`y = y_main + y_pert`/`return_components` block (matches the version read during planning). If different, adapt the edits below.

- [ ] **Step 2.2: Run, confirm fail**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py::test_fdn_gated_init_matches_baseline -v
```
Expected: `TypeError: ... unexpected keyword argument 'use_gated_fusion'`.

- [ ] **Step 2.3: Add constructor kwargs** — change the `__init__` signature tail:

```python
                 sensor_meta_dim: int | None = None,
                 use_gated_fusion: bool = False,
                 gate_d_node: int = 8,
                 gate_hidden: int = 32):
```
After the `self.pert_branch = gwnet(...)` block, append:

```python
        self.use_gated_fusion = use_gated_fusion
        self._last_alpha = None     # cached live tensor for optional L1 penalty
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                num_nodes=num_nodes, T_p=T_p,
                d_node=gate_d_node, hidden=gate_hidden,
            )
```

- [ ] **Step 2.4: Modify forward** — replace the `y_main`/`y_pert`/`return_components` block at the bottom of `FourierDualNet.forward` with:

```python
        y_main = self._from_gwnet(self.main_branch(self._to_gwnet(x_main_input)))
        y_pert = self._from_gwnet(self.pert_branch(self._to_gwnet(x_pert_input)))

        alpha = None
        if self.use_gated_fusion:
            assert time_feat is not None, "use_gated_fusion=True requires time_feat"
            y, alpha = self.gated_fusion(x_main_flow, x_pert_flow, y_main, y_pert, time_feat)
        else:
            y = y_main + y_pert
        self._last_alpha = alpha     # live tensor (or None) for the L1 fallback

        if return_components:
            comp = {"y_main": y_main, "y_pert": y_pert,
                    "x_main": x_main_flow, "x_pert": x_pert_flow,
                    "mask": self.decomp.get_mask()}
            if alpha is not None:
                comp["alpha"] = alpha
            return y, comp
        return y
```

- [ ] **Step 2.5: Run, confirm all pass**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```
Expected: 6/6 pass.

- [ ] **Step 2.6: Commit**

```bash
git add fourier_dual_net/model.py
git commit -m "Wire GatedFusion into FourierDualNet (use_gated_fusion flag, _last_alpha cache)"
```

---

## Task 3: Training-script integration (CLI + all 3 need_time + L1 hook)

**Files:** Modify `scripts/train_fourier_dual_net.py`.

- [ ] **Step 3.1: Add CLI flags** — after `p.add_argument("--cross_attn_heads", ...)`:

```python
    p.add_argument("--use_gated_fusion", action="store_true",
                   help="Enable D3 gated fusion (per-node, per-horizon alpha)")
    p.add_argument("--gate_d_node", type=int, default=8)
    p.add_argument("--gate_hidden", type=int, default=32)
    p.add_argument("--gate_l1_lambda", type=float, default=0.0,
                   help="L1 penalty lambda*mean|alpha-0.5| (fallback; 0 disables)")
```

- [ ] **Step 3.2: Pass flags to the constructor** — in the `FourierDualNet(...)` call, after `cross_attn_heads=args.cross_attn_heads,` and before `sensor_meta_dim=sensor_meta_dim,`:

```python
        use_gated_fusion=args.use_gated_fusion,
        gate_d_node=args.gate_d_node,
        gate_hidden=args.gate_hidden,
```

- [ ] **Step 3.3: Fix `need_time` in ALL THREE locations** — replace each `need_time = model.use_time_emb or model.requires_sensor_meta` (in `evaluate()` L102, train loop L202, test loop L267) with:

```python
        need_time = (model.use_time_emb or model.requires_sensor_meta
                     or getattr(model, "use_gated_fusion", False))
```

- [ ] **Step 3.4: Add the optional L1 term** — in the train loop, immediately after `loss = masked_mae(pred, y_true, y_mask)`:

```python
                if args.gate_l1_lambda > 0 and getattr(model, "_last_alpha", None) is not None:
                    loss = loss + args.gate_l1_lambda * (model._last_alpha - 0.5).abs().mean()
```

- [ ] **Step 3.5: Smoke-test flags**

```bash
python scripts/train_fourier_dual_net.py --help 2>&1 | grep -E "gated_fusion|gate_d_node|gate_hidden|gate_l1_lambda"
```
Expected: four matching lines.

- [ ] **Step 3.6: Commit**

```bash
git add scripts/train_fourier_dual_net.py
git commit -m "Add gated-fusion CLI flags, fix need_time in all 3 loops, L1 hook"
```

---

## Task 4: Gate analysis script

**Files:** Create `scripts/analyze_gate.py`.

- [ ] **Step 4.1: Create the script**

```python
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
```

- [ ] **Step 4.2: Smoke test --help**

```bash
python scripts/analyze_gate.py --help 2>&1 | head -15
```
Expected: argparse usage, no ImportError (validates `MultiRegionDataset`/`NUM_TOD_SLOTS` import path).

- [ ] **Step 4.3: Commit**

```bash
git add scripts/analyze_gate.py
git commit -m "Add analyze_gate.py (rebuild-from-ckpt, real data API, dual-support graph)"
```

---

## Task 5: Local end-to-end smoke (real 1 epoch — NOT --smoke)

`--smoke` saves no checkpoint (val=NaN). Use a real `--epochs 1` with tiny `nhid`.

- [ ] **Step 5.1: 1-epoch run** (on the 5080 if CPU is too slow — Alameda has 521 nodes)

```bash
python scripts/train_fourier_dual_net.py \
    --region Alameda --decomp_mode learnable --K 3 \
    --use_gated_fusion --nhid 8 --epochs 1 --batch_size 8 \
    --tag smoke_d3 --out_dir outputs/fourier_dual_net --device cpu 2>&1 | tail -30
```
Expected: 1 epoch, no NaN, writes `outputs/fourier_dual_net/smoke_d3/Alameda/ckpt_best.pt`.

- [ ] **Step 5.2: Run analyze on the smoke checkpoint**

```bash
python scripts/analyze_gate.py \
    --ckpt outputs/fourier_dual_net/smoke_d3/Alameda/ckpt_best.pt \
    --region Alameda \
    --out_dir outputs/fourier_dual_net/smoke_d3/Alameda/analysis --device cpu 2>&1 | tail -25
```
Expected: prints sanity-check lines (most FAIL after 1 epoch — fine) + `output_scale`. Validates the full analysis path against real data/ckpt/graph contracts. **This step is the real test of the Task 4 contract fixes.**

- [ ] **Step 5.3: Clean up**

```bash
rm -rf outputs/fourier_dual_net/smoke_d3/
```

---

## Task 6: Stage S1 — Alameda from-scratch on 5080 (NO --use_time_emb)

- [ ] **Step 6.1: Tarball + push + extract**

```bash
tar -czf /tmp/fdn_d3.tar.gz fourier_dual_net/model.py scripts/train_fourier_dual_net.py scripts/analyze_gate.py tests/fourier_dual_net/
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no /tmp/fdn_d3.tar.gz asus@100.126.189.30:C:/Users/asus/traffic_fourier/fdn_d3.tar.gz
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "cd C:/Users/asus/traffic_fourier && tar -xzf fdn_d3.tar.gz && ls fourier_dual_net/model.py"
```

- [ ] **Step 6.2: Launch S1 (no --use_time_emb — confound control)**

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "cd C:/Users/asus/traffic_fourier && nohup python scripts/train_fourier_dual_net.py --region Alameda --decomp_mode learnable --K 3 --use_gated_fusion --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_Alameda_D3.log 2>&1 &"
```

- [ ] **Step 6.3: Monitor (covers progress AND crashes)** — Monitor tool:

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "powershell -Command \"Get-Content C:/Users/asus/traffic_fourier/train_Alameda_D3.log -Wait -Tail 0\"" | grep -E --line-buffered "==> ep|test MAE|Traceback|Error|OOM|nan|saved best"
```

- [ ] **Step 6.4: Pull artifacts (ckpt_best.pt)**

```bash
mkdir -p outputs/fourier_dual_net/fdn_d3_gate/Alameda
for f in test_predictions.npz ckpt_best.pt summary.json; do
  sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/Alameda/$f outputs/fourier_dual_net/fdn_d3_gate/Alameda/$f
done
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no asus@100.126.189.30:C:/Users/asus/traffic_fourier/train_Alameda_D3.log outputs/fourier_dual_net/fdn_d3_gate/Alameda/train.log
```

- [ ] **Step 6.5: Compute S1 MAE + decision gate**

```bash
python -c "
import numpy as np
d = np.load('outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz')
diff = np.abs(d['pred_raw_flow'] - d['actual_future_flow']); m = d['y_mask_flow']
aff = np.broadcast_to(d['affected_mask'][:,None,:], diff.shape)
print(f\"all={diff[m].mean():.3f}  affected={diff[m&aff].mean():.3f}  unaffected={diff[m&~aff].mean():.3f}\")
"
```
**Decision gate**: baseline Alameda all-MAE ≈ 11.98. If D3 all-MAE ≥ 11.83 (drop < 0.15) → HALT → Task 9. A single seed cannot prove significance; if the drop is in (0, 0.3), require the seed-robustness runs (Task 8.5) before claiming a win.

---

## Task 7: Stage S2 — sanity checks on Alameda

- [ ] **Step 7.1: Run analyze_gate.py on ckpt_best.pt**

```bash
python scripts/analyze_gate.py \
    --ckpt outputs/fourier_dual_net/fdn_d3_gate/Alameda/ckpt_best.pt \
    --region Alameda \
    --out_dir outputs/fourier_dual_net/fdn_d3_gate/Alameda/analysis --device cpu
```

- [ ] **Step 7.2: Decision gate** — read `analysis/sanity_checks.txt`:
  - ≥3/4 PASS → real architectural contribution → S3 (Task 8).
  - 1-2 PASS but overall MAE down ≥0.15 → gain is real but not the gate's intended mechanism (check `output_scale` + run the Task 8.6 control). Report honestly; ask user before S3.
  - 0 PASS and MAE not down ≥0.15 → D3 failed → Task 9.

- [ ] **Step 7.3: Commit**

```bash
git add outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz \
        outputs/fourier_dual_net/fdn_d3_gate/Alameda/analysis/sanity_checks.txt \
        outputs/fourier_dual_net/fdn_d3_gate/Alameda/train.log
git commit -m "S1+S2 D3 gated fusion results on Alameda + sanity checks"
```

---

## Task 8: Stage S3 — CC + Orange + controls (only if S2 passes)

- [ ] **Step 8.1: Launch CC + Orange (no --use_time_emb)**

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "cd C:/Users/asus/traffic_fourier && nohup bash -c 'python scripts/train_fourier_dual_net.py --region ContraCosta --decomp_mode learnable --K 3 --use_gated_fusion --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_CC_D3.log 2>&1 && python scripts/train_fourier_dual_net.py --region Orange --decomp_mode learnable --K 3 --use_gated_fusion --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_Orange_D3.log 2>&1 && touch D3_ALL_DONE' &"
```

- [ ] **Step 8.2: Wait (single notification)** — background Bash:

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "until [ -f C:/Users/asus/traffic_fourier/D3_ALL_DONE ]; do sleep 60; done; echo DONE"
```

- [ ] **Step 8.3: Pull artifacts**

```bash
for region in ContraCosta Orange; do
  mkdir -p outputs/fourier_dual_net/fdn_d3_gate/$region
  for f in test_predictions.npz ckpt_best.pt summary.json; do
    sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/$region/$f outputs/fourier_dual_net/fdn_d3_gate/$region/$f
  done
done
```

- [ ] **Step 8.4: Sanity checks per region**

```bash
for region in ContraCosta Orange; do
  python scripts/analyze_gate.py --ckpt outputs/fourier_dual_net/fdn_d3_gate/$region/ckpt_best.pt --region $region --out_dir outputs/fourier_dual_net/fdn_d3_gate/$region/analysis --device cpu
done
```

- [ ] **Step 8.5: Seed robustness (noise band)** — baseline + D3 on 2 extra seeds for Alameda:

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "cd C:/Users/asus/traffic_fourier && nohup bash -c 'for s in 1 2; do python scripts/train_fourier_dual_net.py --region Alameda --decomp_mode learnable --K 3 --epochs 30 --seed \$s --tag base_seed\$s --out_dir outputs/fourier_dual_net --device cuda > base_s\$s.log 2>&1; python scripts/train_fourier_dual_net.py --region Alameda --decomp_mode learnable --K 3 --use_gated_fusion --epochs 30 --seed \$s --tag d3_seed\$s --out_dir outputs/fourier_dual_net --device cuda > d3_s\$s.log 2>&1; done; touch SEEDS_DONE' &"
```
Report baseline mean±std vs D3 mean±std; the S1 gap must exceed the band.

- [ ] **Step 8.6: Rescale control (attribution)** — measure how much of the gain is a trivial global rescale. Compare the `output_scale` logged by analyze across runs; if it drifts far from 2.0, run a frozen-gate control (gate α=0.5 fixed, only `output_scale`+backbones train — needs a `--gate_frozen` flag, defer to writing-plans if required). Report honestly.

- [ ] **Step 8.7: Aggregate + commit**

```bash
python -c "
import numpy as np, os
for region in ['Alameda','ContraCosta','Orange']:
    p=f'outputs/fourier_dual_net/fdn_d3_gate/{region}/test_predictions.npz'
    if not os.path.exists(p): continue
    d=np.load(p); diff=np.abs(d['pred_raw_flow']-d['actual_future_flow']); m=d['y_mask_flow']
    aff=np.broadcast_to(d['affected_mask'][:,None,:],diff.shape)
    print(f'{region:12s} all={diff[m].mean():.3f} aff={diff[m&aff].mean():.3f} un={diff[m&~aff].mean():.3f}')
"
git add outputs/fourier_dual_net/fdn_d3_gate/
git commit -m "S3 D3 gated fusion results on CC + Orange + seed/rescale controls"
```
Decision: ≥2/3 regions improve ≥0.15 (beyond the measured seed band) → generalizable win.

---

## Task 9: Fallback — if S1 fails the decision gate

Execute only if Task 6.5 shows Alameda drop < 0.15. L1 plumbing is already in place (Task 3.4 + `_last_alpha` from Task 2). Sequence:

1. **Fallback A** (gate polarization — C1 shows α near 0/1): retrain S1 with `--gate_l1_lambda 0.01`.
2. **Fallback B** (weak gate inputs): strip the gate to energy-ratio only (drop `node_emb` + `time_avg`); a new GatedFusion variant flag is needed — return to writing-plans.
3. **Abandon D3** → pivot to D1 (per-node K from autocorrelation features); new design spec.

When this task triggers, return to brainstorming/writing-plans rather than guessing which fallback applies.

---

## Self-Review

**1. Spec coverage**:
- §Goal (gate fusion) → Tasks 1+2 ✓
- §Gate inputs (flow-only energy, node_emb, time_avg) → Task 1.4 + `_energy_ratio` test ✓
- §Initialization (zero_(), exact baseline, output_scale, _last_alpha) → Task 1.4 + 2.3/2.4 + tests ✓
- §Design decisions (per-(node,horizon), learnable scale, no reg default, no dropout) → Task 1 ✓
- §Training plan — confound control (no --use_time_emb) → Tasks 6.2, 8.1 ✓
- §Training plan — seed robustness → Task 8.5 ✓
- §Training plan — rescale control → Task 8.6 ✓
- §Sanity checks (tightened thresholds, cross-region sign) → Task 4 + 7 + 8.4 ✓
- §Risks / Fallback (L1 via _last_alpha) → Task 3.4 + 9 ✓
- §Files to modify → matches File Plan ✓

**2. Placeholder scan**: no TBD/TODO. All code blocks complete. Task 8.6 frozen-gate control + Task 9 fallback B deliberately deferred (depend on observed failure mode) — flagged explicitly.

**3. Type/contract consistency** (re-verified against source, not memory):
- `MultiRegionDataset` + `NUM_TOD_SLOTS` import; `rdata.N/.C_meta/.T_h/.T_p/.edge_index` attributes ✓
- `build_adj_supports` copied verbatim — row-norm random walk, TWO supports (corrected from a wrong review guess) ✓
- batch keys `y_true[...,0]`, `y_mask[...,0]`, `time_feat (B,T_h,2)`, `sample_start` ✓
- checkpoint `ckpt_best.pt`, key `model_state`, dims from `state["N"/"C_x"/"T_h"/"T_p"]`, config from `state["config"]` ✓
- `make_loader(dataset, batch_size, shuffle=False)` ✓
- `GatedFusion(num_nodes, T_p, d_node, hidden)` ⇄ `FourierDualNet(..., gate_d_node, gate_hidden)` ⇄ CLI `--gate_d_node/--gate_hidden` ✓
- `GatedFusion.forward(x_main,x_pert,y_main,y_pert,time_feat) -> (y, alpha)` used identically in FDN.forward + tests ✓
- `_last_alpha` set in FDN.forward, read in train-loop L1 term ✓
- all 3 `need_time` lines fixed ✓
- axis order (B,N,T_p) consistent across analyze_gate.py reductions ✓
