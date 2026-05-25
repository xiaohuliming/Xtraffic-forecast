# FDN Gated Fusion (D3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-(node, horizon) learned gate on top of FourierDualNet's two-branch output, replacing unconditional `y_main + y_pert` with `2 * (α * y_main + (1-α) * y_pert)`, where α is conditioned on spectral energy, node identity, and time-of-day.

**Architecture:** New `GatedFusion` module added to `fourier_dual_net/model.py` alongside the existing `FourierDecomp` / `CrossBranchAttention` / `FourierDualNet` classes. Toggleable via constructor flag `use_gated_fusion: bool` (default False, preserves baseline). Initialized so that ep0 output is mathematically identical to the additive baseline. Trained from scratch with the same recipe as baseline FDN.

**Tech Stack:** PyTorch, GraphWaveNet (already vendored under `baselines/GraphWaveNet/`), numpy, pandas. No new dependencies.

**Spec reference:** [docs/superpowers/specs/2026-05-25-fdn-gated-fusion-design.md](../specs/2026-05-25-fdn-gated-fusion-design.md)

---

## File Plan

| File | Action | Responsibility |
|---|---|---|
| `fourier_dual_net/model.py` | MODIFY | Add `GatedFusion` class. Modify `FourierDualNet.__init__` and `forward` to optionally use it. |
| `tests/fourier_dual_net/test_gated_fusion.py` | CREATE | Unit tests: shapes, init-preserves-baseline, gradient flow. |
| `tests/fourier_dual_net/__init__.py` | CREATE | Empty, marks test dir as package. |
| `scripts/train_fourier_dual_net.py` | MODIFY | Add `--use_gated_fusion` CLI flag, pass to model constructor. |
| `scripts/analyze_gate.py` | CREATE | Load checkpoint + test set, record α tensor, compute 4 sanity checks from spec §Sanity checks. |

No new dependencies. No new directories besides `tests/fourier_dual_net/`.

---

## Task 1: GatedFusion module + unit tests

**Files:**
- Modify: `fourier_dual_net/model.py` (append new class)
- Create: `tests/fourier_dual_net/__init__.py`
- Create: `tests/fourier_dual_net/test_gated_fusion.py`

- [ ] **Step 1.1: Create test package marker**

```bash
mkdir -p tests/fourier_dual_net
touch tests/fourier_dual_net/__init__.py
```

- [ ] **Step 1.2: Write failing tests for GatedFusion**

Create `tests/fourier_dual_net/test_gated_fusion.py`:

```python
"""Unit tests for GatedFusion module (D3 design)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fourier_dual_net.model import GatedFusion


def _make_inputs(B=2, N=5, T_p=12, T_h=12, C=3, d_node=8):
    x_main = torch.randn(B, N, T_h, C)
    x_pert = torch.randn(B, N, T_h, C)
    y_main = torch.randn(B, N, T_p)
    y_pert = torch.randn(B, N, T_p)
    time_feat = torch.randn(B, T_h, 2)
    return x_main, x_pert, y_main, y_pert, time_feat


def test_forward_shape():
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    x_main, x_pert, y_main, y_pert, time_feat = _make_inputs()
    y, alpha = g(x_main, x_pert, y_main, y_pert, time_feat)
    assert y.shape == (2, 5, 12)
    assert alpha.shape == (2, 5, 12)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_init_preserves_baseline():
    """At init, output must equal y_main + y_pert (baseline additive fusion)."""
    torch.manual_seed(0)
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    x_main, x_pert, y_main, y_pert, time_feat = _make_inputs()
    y, alpha = g(x_main, x_pert, y_main, y_pert, time_feat)
    expected = y_main + y_pert
    # alpha ≈ 0.5, output_scale = 2.0 → y = 2 * (0.5*y_main + 0.5*y_pert) = y_main + y_pert
    assert torch.allclose(y, expected, atol=1e-4), \
        f"init output deviates from baseline by max {(y - expected).abs().max().item():.4g}"
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-2)


def test_gradient_flows_to_gate():
    g = GatedFusion(num_nodes=5, T_p=12, d_node=8)
    x_main, x_pert, y_main, y_pert, time_feat = _make_inputs()
    y, _ = g(x_main, x_pert, y_main, y_pert, time_feat)
    loss = y.sum()
    loss.backward()
    assert g.gate_mlp[-1].weight.grad is not None
    assert g.gate_mlp[-1].weight.grad.abs().sum().item() > 0
    assert g.output_scale.grad is not None
    assert g.node_emb.weight.grad is not None
```

- [ ] **Step 1.3: Run tests, confirm failure**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```

Expected: ImportError — `GatedFusion` not yet defined in `fourier_dual_net.model`.

- [ ] **Step 1.4: Implement GatedFusion in model.py**

Append to `fourier_dual_net/model.py` (after `CrossBranchAttention`, before `FourierDualNet`):

```python
class GatedFusion(nn.Module):
    """Gated fusion of Main + Pert branch outputs.

    Replaces `y = y_main + y_pert` with `y = output_scale * (alpha * y_main + (1-alpha) * y_pert)`,
    where alpha is conditioned on:
      - spectral_energy_ratio: ||x_main||^2 / (||x_main||^2 + ||x_pert||^2) per (sample, node)
      - node_emb: learnable per-node embedding (initialized to zero)
      - time_avg: ToD/DoW averaged over the history window

    Initialization (critical):
      - gate_mlp last layer: bias=0, weight scaled 0.01x  → alpha ≈ 0.5
      - node_emb: zeros
      - output_scale: learnable scalar, init=2.0
    Together these guarantee y == y_main + y_pert at ep0.
    """

    def __init__(self, num_nodes: int, T_p: int = 12, d_node: int = 8, hidden: int = 32):
        super().__init__()
        self.T_p = T_p
        self.d_node = d_node
        self.node_emb = nn.Embedding(num_nodes, d_node)
        gate_in_dim = 1 + d_node + 2   # spectral_energy + node_emb + time_avg
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, T_p),
        )
        self.output_scale = nn.Parameter(torch.tensor(2.0))

        with torch.no_grad():
            self.gate_mlp[-1].bias.zero_()
            self.gate_mlp[-1].weight.mul_(0.01)
            self.node_emb.weight.zero_()

    def forward(self, x_main: torch.Tensor, x_pert: torch.Tensor,
                y_main: torch.Tensor, y_pert: torch.Tensor,
                time_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_main, x_pert: (B, N, T_h, C) — decomposed inputs (for energy ratio)
        y_main, y_pert: (B, N, T_p)   — branch predictions
        time_feat:      (B, T_h, 2)   — ToD/DoW per history step
        returns: (y, alpha) where y has shape (B, N, T_p) and alpha (B, N, T_p)
        """
        B, N, T_p = y_main.shape
        assert T_p == self.T_p

        # spectral energy ratio per (sample, node)
        e_main = (x_main ** 2).mean(dim=(2, 3))                    # (B, N)
        e_pert = (x_pert ** 2).mean(dim=(2, 3))                    # (B, N)
        energy_ratio = e_main / (e_main + e_pert + 1e-6)           # (B, N)

        node_e = self.node_emb.weight.unsqueeze(0).expand(B, N, self.d_node)   # (B, N, d_node)
        time_avg = time_feat.mean(dim=1).unsqueeze(1).expand(B, N, 2)          # (B, N, 2)

        gate_in = torch.cat([
            energy_ratio.unsqueeze(-1),   # (B, N, 1)
            node_e,                       # (B, N, d_node)
            time_avg,                     # (B, N, 2)
        ], dim=-1)                        # (B, N, 1+d_node+2)

        alpha_logits = self.gate_mlp(gate_in)        # (B, N, T_p)
        alpha = torch.sigmoid(alpha_logits)          # (B, N, T_p)

        y = self.output_scale * (alpha * y_main + (1.0 - alpha) * y_pert)
        return y, alpha
```

- [ ] **Step 1.5: Run tests, confirm pass**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add fourier_dual_net/model.py tests/fourier_dual_net/__init__.py tests/fourier_dual_net/test_gated_fusion.py
git commit -m "Add GatedFusion module + unit tests (D3 design)"
```

---

## Task 2: Integrate GatedFusion into FourierDualNet

**Files:**
- Modify: `fourier_dual_net/model.py` (`FourierDualNet` class)
- Modify: `tests/fourier_dual_net/test_gated_fusion.py` (add integration test)

- [ ] **Step 2.1: Write failing integration test**

Append to `tests/fourier_dual_net/test_gated_fusion.py`:

```python
def test_fdn_with_gated_fusion_init_matches_baseline():
    """With use_gated_fusion=True at init, output must equal use_gated_fusion=False."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "GraphWaveNet"))
    from fourier_dual_net.model import FourierDualNet

    torch.manual_seed(42)
    N, T_h, T_p, C = 5, 12, 12, 3
    supports = [torch.eye(N)]
    device = torch.device("cpu")

    # Baseline
    torch.manual_seed(42)
    m_base = FourierDualNet(num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
                            in_dim_flow=C, nhid=8, device=device, use_time_emb=False)
    # Gated, same seed → backbones init identically
    torch.manual_seed(42)
    m_gate = FourierDualNet(num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
                            in_dim_flow=C, nhid=8, device=device, use_time_emb=False,
                            use_gated_fusion=True)

    x = torch.randn(2, N, T_h, C)
    tf = torch.randn(2, T_h, 2)
    m_base.eval(); m_gate.eval()
    with torch.no_grad():
        y_base = m_base(x, time_feat=tf)
        y_gate = m_gate(x, time_feat=tf)
    max_diff = (y_base - y_gate).abs().max().item()
    assert max_diff < 1e-4, f"init FDN+gate deviates from baseline by {max_diff:.4g}"


def test_fdn_gated_fusion_requires_time_feat():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "GraphWaveNet"))
    from fourier_dual_net.model import FourierDualNet

    N, T_h, T_p, C = 5, 12, 12, 3
    supports = [torch.eye(N)]
    m = FourierDualNet(num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
                       in_dim_flow=C, nhid=8, device=torch.device("cpu"),
                       use_time_emb=False, use_gated_fusion=True)
    x = torch.randn(2, N, T_h, C)
    with pytest.raises(AssertionError):
        m(x, time_feat=None)
```

- [ ] **Step 2.2: Run, confirm fail**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py::test_fdn_with_gated_fusion_init_matches_baseline -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'use_gated_fusion'`.

- [ ] **Step 2.3: Add use_gated_fusion to FourierDualNet.__init__**

In `fourier_dual_net/model.py`, find `class FourierDualNet` and modify `__init__` signature. Add new kwarg after `sensor_meta_dim`:

```python
                 sensor_meta_dim: int | None = None,
                 use_gated_fusion: bool = False,
                 gate_d_node: int = 8,
                 gate_hidden: int = 32):
```

Then inside `__init__`, after the `self.pert_branch = gwnet(...)` block, append:

```python
        self.use_gated_fusion = use_gated_fusion
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                num_nodes=num_nodes, T_p=T_p,
                d_node=gate_d_node, hidden=gate_hidden,
            )
```

- [ ] **Step 2.4: Modify forward to use the gate**

In `fourier_dual_net/model.py`, replace the bottom of `FourierDualNet.forward` (starting at the line that computes `y_main` and `y_pert`):

```python
        y_main = self._from_gwnet(self.main_branch(self._to_gwnet(x_main_input)))
        y_pert = self._from_gwnet(self.pert_branch(self._to_gwnet(x_pert_input)))

        alpha = None
        if self.use_gated_fusion:
            assert time_feat is not None, "use_gated_fusion=True requires time_feat"
            y, alpha = self.gated_fusion(x_main_flow, x_pert_flow, y_main, y_pert, time_feat)
        else:
            y = y_main + y_pert

        if return_components:
            comp = {"y_main": y_main, "y_pert": y_pert,
                    "x_main": x_main_flow, "x_pert": x_pert_flow,
                    "mask": self.decomp.get_mask()}
            if alpha is not None:
                comp["alpha"] = alpha
            return y, comp
        return y
```

- [ ] **Step 2.5: Run tests, confirm pass**

```bash
pytest tests/fourier_dual_net/test_gated_fusion.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add fourier_dual_net/model.py tests/fourier_dual_net/test_gated_fusion.py
git commit -m "Wire GatedFusion into FourierDualNet via use_gated_fusion flag"
```

---

## Task 3: Add CLI flag to training script

**Files:**
- Modify: `scripts/train_fourier_dual_net.py`

- [ ] **Step 3.1: Add CLI argument**

In `scripts/train_fourier_dual_net.py`, find the argparse block (look for `p.add_argument("--use_cross_attn", ...)` around line 54) and add after it:

```python
    p.add_argument("--use_gated_fusion", action="store_true",
                   help="Enable D3 gated fusion of Main/Pert outputs (per-node, per-horizon alpha)")
    p.add_argument("--gate_d_node", type=int, default=8)
    p.add_argument("--gate_hidden", type=int, default=32)
```

- [ ] **Step 3.2: Pass new flags to model constructor**

In `scripts/train_fourier_dual_net.py`, find the model construction block (around lines 152-163, where it passes `use_cross_attn` etc.) and add after `cross_attn_heads=args.cross_attn_heads,`:

```python
    cross_attn_heads=args.cross_attn_heads,
    use_gated_fusion=args.use_gated_fusion,
    gate_d_node=args.gate_d_node,
    gate_hidden=args.gate_hidden,
```

- [ ] **Step 3.3: Ensure time_feat is fetched when gated fusion is on**

In `scripts/train_fourier_dual_net.py` find the line `need_time = model.use_time_emb or model.requires_sensor_meta` (around line 102). Replace with:

```python
    need_time = model.use_time_emb or model.requires_sensor_meta or getattr(model, "use_gated_fusion", False)
```

- [ ] **Step 3.4: Smoke-test the script with --help**

```bash
python scripts/train_fourier_dual_net.py --help 2>&1 | grep -E "gated_fusion|gate_d_node|gate_hidden"
```

Expected: three matching lines confirming the flags are registered.

- [ ] **Step 3.5: Commit**

```bash
git add scripts/train_fourier_dual_net.py
git commit -m "Add --use_gated_fusion CLI flag to FDN training"
```

---

## Task 4: Gate analysis script

**Files:**
- Create: `scripts/analyze_gate.py`

This script runs inference on the test set of a trained `use_gated_fusion=True` checkpoint, dumps the `alpha` tensor, and runs the 4 sanity checks from the spec.

- [ ] **Step 4.1: Create analyze_gate.py**

```python
"""Analyze D3 gated fusion: record alpha on test set + run 4 sanity checks.

Inputs:
  --ckpt    Path to best.pt
  --region  Region name (Alameda/ContraCosta/Orange)
  --out_dir Where to write alpha tensor + sanity check report

Outputs (under out_dir):
  alpha_test.npz        — alpha (S, N, T_p) + per-node test MAE arrays for correlations
  sanity_checks.txt     — human-readable pass/fail for the 4 checks from spec §3.2

Sanity checks (from design spec):
  1. alpha has variance       (mean ≈ 0.5, std > 0.1)
  2. alpha reflects failure modes
       - corr(per_node_MAE, alpha_per_node.mean()) < 0       (high-MAE nodes get low alpha)
       - alpha at midday < alpha at rush hour
       - corr(spectral_energy_ratio, alpha) > 0
  3. alpha increases with horizon (alpha[..., -1].mean() > alpha[..., 0].mean())
  4. Targeted MAE reductions vs baseline:
       - p99 node MAE
       - midday MAE
       - Orange-affected-long-horizon MAE
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))
from fourier_dual_net.model import FourierDualNet
from dist_net.data import RegionDataset, make_loader


def load_baseline_test_mae(region: str) -> dict:
    """Load baseline (learnable_K3) test MAE for the 3 targeted breakdowns."""
    path = ROOT / "outputs" / "diagnostics" / f"per_node_{region}.npz"
    if not path.exists():
        return {}
    d = np.load(path)
    return {"node_fdn_mae": d["node_fdn"], "aff_freq": d["aff_freq"]}


def midday_mask_from_sample_start(sample_start: np.ndarray, slot_per_day: int = 288) -> np.ndarray:
    tod = (sample_start + 1) % slot_per_day
    return (tod >= 10 * 12) & (tod < 15 * 12)


def rush_mask_from_sample_start(sample_start: np.ndarray, slot_per_day: int = 288) -> np.ndarray:
    tod = (sample_start + 1) % slot_per_day
    am = (tod >= 6 * 12) & (tod < 10 * 12)
    pm = (tod >= 15 * 12) & (tod < 19 * 12)
    return am | pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--region", type=str, required=True)
    ap.add_argument("--data_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--graph_dir", type=Path, default=Path("data/graphs"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    state = torch.load(args.ckpt, map_location=device)
    cfg = state.get("config", {})

    test_ds = RegionDataset(
        region_names=[args.region], data_dir=args.data_dir,
        graph_dir=args.graph_dir, split="test", lazy=False,
    )
    rdata = test_ds.regions[args.region]
    N, T_h, T_p = rdata["N"], rdata["T_h"], rdata["T_p"]
    supports = [s.to(device) for s in rdata["supports"]]
    C_x = rdata["C_x"]

    model = FourierDualNet(
        num_nodes=N, supports=supports, T_h=T_h, T_p=T_p,
        K=cfg.get("K", 3), decomp_mode=cfg.get("decomp_mode", "learnable"),
        in_dim_flow=C_x, nhid=cfg.get("nhid", 32),
        dropout=cfg.get("dropout", 0.3), device=device,
        main_blocks=cfg.get("main_blocks", 4), main_layers=cfg.get("main_layers", 2),
        pert_blocks=cfg.get("pert_blocks", 4), pert_layers=cfg.get("pert_layers", 2),
        use_time_emb=cfg.get("use_time_emb", False),
        use_cross_attn=cfg.get("use_cross_attn", False),
        use_gated_fusion=True,
        gate_d_node=cfg.get("gate_d_node", 8),
        gate_hidden=cfg.get("gate_hidden", 32),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)

    alpha_list, ss_list, mask_list, pred_list, actual_list, aff_list = [], [], [], [], [], []
    energy_ratio_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_hist"].to(device)
            tf = batch["time_feat"].to(device)
            y, comp = model(x, time_feat=tf, return_components=True)
            alpha_list.append(comp["alpha"].cpu().numpy())
            ss_list.append(batch["sample_start"].numpy())
            mask_list.append(batch["y_mask_flow"].numpy())
            pred_list.append(y.cpu().numpy())
            actual_list.append(batch["actual_future_flow"].numpy())
            aff_list.append(batch["affected_mask"].numpy())
            # spectral_energy_ratio post-decomp
            x_main, x_pert = comp["x_main"], comp["x_pert"]
            e_main = (x_main ** 2).mean(dim=(2, 3)).cpu().numpy()
            e_pert = (x_pert ** 2).mean(dim=(2, 3)).cpu().numpy()
            energy_ratio_list.append(e_main / (e_main + e_pert + 1e-6))

    alpha = np.concatenate(alpha_list, axis=0)              # (S, N, T_p)
    ss = np.concatenate(ss_list, axis=0)                    # (S,)
    msk = np.concatenate(mask_list, axis=0)                 # (S, T_p, N)
    pred = np.concatenate(pred_list, axis=0)                # (S, T_p, N)
    actual = np.concatenate(actual_list, axis=0)
    aff = np.concatenate(aff_list, axis=0)                  # (S, N)
    energy = np.concatenate(energy_ratio_list, axis=0)      # (S, N)

    np.savez(args.out_dir / "alpha_test.npz",
             alpha=alpha, sample_start=ss, energy_ratio=energy, aff=aff)

    # ---- Sanity checks ----
    lines = []
    lines.append(f"# Sanity checks for {args.region}\n")

    # Check 1: variance
    a_mean = float(alpha.mean())
    a_std = float(alpha.std())
    c1 = abs(a_mean - 0.5) < 0.3 and a_std > 0.05
    lines.append(f"[{'PASS' if c1 else 'FAIL'}] Check 1: variance — mean={a_mean:.3f} std={a_std:.3f}")

    # Check 2a: high-MAE nodes get low alpha
    diff = np.abs(pred - actual) * msk
    cnt = msk.sum(axis=(0, 1)).astype(np.float64)
    node_mae = np.where(cnt > 0, diff.sum(axis=(0, 1)) / np.maximum(cnt, 1), np.nan)
    alpha_per_node = alpha.mean(axis=(0, 2))             # (N,)
    valid = ~np.isnan(node_mae)
    corr_node = float(np.corrcoef(node_mae[valid], alpha_per_node[valid])[0, 1])
    c2a = corr_node < -0.05
    lines.append(f"[{'PASS' if c2a else 'FAIL'}] Check 2a: corr(node_MAE, alpha_per_node) = {corr_node:+.3f} (want <0)")

    # Check 2b: midday < rush
    midday = midday_mask_from_sample_start(ss)
    rush = rush_mask_from_sample_start(ss)
    a_midday = float(alpha[midday].mean()) if midday.any() else float("nan")
    a_rush = float(alpha[rush].mean()) if rush.any() else float("nan")
    c2b = a_midday < a_rush
    lines.append(f"[{'PASS' if c2b else 'FAIL'}] Check 2b: alpha_midday={a_midday:.3f} < alpha_rush={a_rush:.3f}")

    # Check 2c: corr(energy, alpha) positive
    alpha_per_sample_node = alpha.mean(axis=2)           # (S, N)
    corr_energy = float(np.corrcoef(energy.ravel(), alpha_per_sample_node.ravel())[0, 1])
    c2c = corr_energy > 0.05
    lines.append(f"[{'PASS' if c2c else 'FAIL'}] Check 2c: corr(energy_ratio, alpha) = {corr_energy:+.3f} (want >0)")

    # Check 3: alpha at long horizon > short horizon
    a_h1 = float(alpha[..., 0].mean())
    a_hL = float(alpha[..., -1].mean())
    c3 = a_hL > a_h1
    lines.append(f"[{'PASS' if c3 else 'FAIL'}] Check 3: alpha[h12]={a_hL:.3f} > alpha[h1]={a_h1:.3f}")

    # Check 4: p99 node MAE, midday MAE vs baseline
    baseline = load_baseline_test_mae(args.region)
    if baseline:
        base_node = baseline["node_fdn_mae"]
        base_node_valid = base_node[~np.isnan(base_node)]
        node_mae_valid = node_mae[valid]
        p99_base = float(np.quantile(base_node_valid, 0.99))
        p99_new = float(np.quantile(node_mae_valid, 0.99))
        c4a = p99_new < p99_base - 0.5
        lines.append(f"[{'PASS' if c4a else 'FAIL'}] Check 4a: p99 node MAE {p99_new:.2f} vs baseline {p99_base:.2f}")
    else:
        lines.append(f"[SKIP] Check 4: baseline diagnostics not found at outputs/diagnostics/per_node_{args.region}.npz")

    # Midday MAE breakdown
    if midday.any():
        diff_md = np.abs(pred[midday] - actual[midday]) * msk[midday]
        n_md = msk[midday].sum()
        midday_mae = float(diff_md.sum() / max(n_md, 1))
        lines.append(f"[INFO]  Midday overall MAE (D3) = {midday_mae:.3f}")

    n_pass = sum(1 for line in lines if line.startswith("[PASS]"))
    n_total = sum(1 for line in lines if line.startswith("[PASS]") or line.startswith("[FAIL]"))
    lines.append(f"\nSummary: {n_pass}/{n_total} checks pass.")

    report = "\n".join(lines)
    (args.out_dir / "sanity_checks.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Smoke test the script's --help works**

```bash
python scripts/analyze_gate.py --help 2>&1 | head -15
```

Expected: argparse usage printed without import errors.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/analyze_gate.py
git commit -m "Add analyze_gate.py: record alpha + run 4 sanity checks"
```

---

## Task 5: Local end-to-end smoke test (1 epoch, tiny config)

Before pushing to remote, run a real training loop locally for 1 epoch on Alameda with tiny dimensions to confirm the gate trains without NaN / shape errors.

**Files:** No file changes — execution only.

- [ ] **Step 5.1: Run 1-epoch smoke training**

```bash
python scripts/train_fourier_dual_net.py \
    --region Alameda \
    --decomp_mode learnable --K 3 \
    --use_gated_fusion \
    --nhid 8 \
    --epochs 1 \
    --batch_size 8 \
    --tag smoke_d3 \
    --out_dir outputs/fourier_dual_net \
    --device cpu \
    2>&1 | tail -30
```

Expected: 1 epoch completes without errors. No NaN in train/val loss. Saves `outputs/fourier_dual_net/smoke_d3/Alameda/best.pt`.

- [ ] **Step 5.2: Run analyze_gate.py on the smoke checkpoint**

```bash
python scripts/analyze_gate.py \
    --ckpt outputs/fourier_dual_net/smoke_d3/Alameda/best.pt \
    --region Alameda \
    --out_dir outputs/fourier_dual_net/smoke_d3/Alameda/analysis \
    --device cpu \
    2>&1 | tail -30
```

Expected: prints 4 sanity check lines. Most will FAIL (only 1 epoch trained), but the script runs to completion. This validates the analysis path.

- [ ] **Step 5.3: Clean up smoke artifacts**

```bash
rm -rf outputs/fourier_dual_net/smoke_d3/
```

No commit needed — this step produces no source changes.

---

## Task 6: Stage S1 — Alameda from-scratch training on 5080

**Files:** None — remote execution.

- [ ] **Step 6.1: Tarball changed source files**

```bash
tar -czf /tmp/fdn_d3.tar.gz \
    fourier_dual_net/model.py \
    scripts/train_fourier_dual_net.py \
    scripts/analyze_gate.py \
    tests/fourier_dual_net/
ls -lh /tmp/fdn_d3.tar.gz
```

- [ ] **Step 6.2: Push tarball to 5080 + extract**

```bash
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
    /tmp/fdn_d3.tar.gz \
    asus@100.126.189.30:C:/Users/asus/traffic_fourier/fdn_d3.tar.gz
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no \
    asus@100.126.189.30 \
    "cd C:/Users/asus/traffic_fourier && tar -xzf fdn_d3.tar.gz && ls fourier_dual_net/model.py"
```

Expected: `fourier_dual_net/model.py` listed.

- [ ] **Step 6.3: Launch S1 Alameda training (nohup, background)**

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no \
    asus@100.126.189.30 \
    "cd C:/Users/asus/traffic_fourier && nohup python scripts/train_fourier_dual_net.py --region Alameda --decomp_mode learnable --K 3 --use_gated_fusion --use_time_emb --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_Alameda_D3.log 2>&1 &"
```

- [ ] **Step 6.4: Monitor training**

Use Monitor with a filter that catches success, progress, and any error/crash:

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 \
    "powershell -Command \"Get-Content C:/Users/asus/traffic_fourier/train_Alameda_D3.log -Wait -Tail 0\"" \
    | grep -E --line-buffered "==> ep|test MAE|Traceback|Error|OOM|nan|saved best"
```

Expected: ~30 `==> ep` lines arriving over ~2.5 hours, ending in `test MAE` line.

- [ ] **Step 6.5: Pull artifacts back**

```bash
mkdir -p outputs/fourier_dual_net/fdn_d3_gate/Alameda
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
    asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz \
    outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
    asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/Alameda/best.pt \
    outputs/fourier_dual_net/fdn_d3_gate/Alameda/best.pt
sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
    asus@100.126.189.30:C:/Users/asus/traffic_fourier/train_Alameda_D3.log \
    outputs/fourier_dual_net/fdn_d3_gate/Alameda/train.log
```

- [ ] **Step 6.6: Compute S1 overall MAE**

```bash
python -c "
import numpy as np
d = np.load('outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz')
diff = np.abs(d['pred_raw_flow'] - d['actual_future_flow'])
m = d['y_mask_flow']
aff_TpN = np.broadcast_to(d['affected_mask'][:, None, :], diff.shape)
print(f\"all       = {diff[m].mean():.3f}\")
print(f\"affected  = {diff[m & aff_TpN].mean():.3f}\")
print(f\"unaffected= {diff[m & ~aff_TpN].mean():.3f}\")
"
```

**Decision gate** (from spec): baseline FDN Alameda all-MAE ≈ 11.98. If D3 all-MAE drops by < 0.15 (i.e. >= 11.83), HALT and escalate to fallback (Task 8).

---

## Task 7: Stage S2 — Sanity checks on Alameda S1

**Files:** None — analysis only.

- [ ] **Step 7.1: Run analyze_gate.py on the S1 checkpoint**

```bash
python scripts/analyze_gate.py \
    --ckpt outputs/fourier_dual_net/fdn_d3_gate/Alameda/best.pt \
    --region Alameda \
    --out_dir outputs/fourier_dual_net/fdn_d3_gate/Alameda/analysis \
    --device cpu
```

Expected: report printed showing pass/fail for all 4 sanity checks.

- [ ] **Step 7.2: Decision gate**

Read `outputs/fourier_dual_net/fdn_d3_gate/Alameda/analysis/sanity_checks.txt`.

- If **≥ 3 of 4 checks PASS** → D3 has measurable architectural contribution → proceed to S3 (Task 8 with happy path).
- If **only 1-2 PASS but overall MAE dropped ≥ 0.15** → fusion works but for non-spec reasons (backbone co-adaptation absorbed the gain). Report this honestly; defer S3 decision to user.
- If **0 PASS AND overall MAE didn't drop ≥ 0.15** → D3 failed → fallback (Task 9).

- [ ] **Step 7.3: Commit Alameda S1 results**

```bash
git add outputs/fourier_dual_net/fdn_d3_gate/Alameda/test_predictions.npz \
        outputs/fourier_dual_net/fdn_d3_gate/Alameda/analysis/sanity_checks.txt \
        outputs/fourier_dual_net/fdn_d3_gate/Alameda/train.log
git commit -m "S1 D3 gated fusion results on Alameda + sanity checks"
```

---

## Task 8: Stage S3 — CC + Orange from-scratch (only if S2 passes)

**Files:** None — remote execution.

- [ ] **Step 8.1: Launch CC + Orange sequentially on 5080**

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 "cd C:/Users/asus/traffic_fourier && nohup bash -c 'python scripts/train_fourier_dual_net.py --region ContraCosta --decomp_mode learnable --K 3 --use_gated_fusion --use_time_emb --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_CC_D3.log 2>&1 && python scripts/train_fourier_dual_net.py --region Orange --decomp_mode learnable --K 3 --use_gated_fusion --use_time_emb --epochs 30 --tag fdn_d3_gate --out_dir outputs/fourier_dual_net --device cuda > train_Orange_D3.log 2>&1 && touch D3_ALL_DONE' &"
```

- [ ] **Step 8.2: Monitor with Monitor tool until D3_ALL_DONE appears**

```bash
sshpass -p 'Hzj050916' ssh -o StrictHostKeyChecking=no asus@100.126.189.30 \
    "until [ -f C:/Users/asus/traffic_fourier/D3_ALL_DONE ]; do sleep 60; done; echo DONE"
```

(Run as background Bash with `run_in_background=true` — single notification when done.)

- [ ] **Step 8.3: Pull CC + Orange artifacts back**

```bash
for region in ContraCosta Orange; do
    mkdir -p outputs/fourier_dual_net/fdn_d3_gate/$region
    sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
        asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/$region/test_predictions.npz \
        outputs/fourier_dual_net/fdn_d3_gate/$region/test_predictions.npz
    sshpass -p 'Hzj050916' scp -o StrictHostKeyChecking=no \
        asus@100.126.189.30:C:/Users/asus/traffic_fourier/outputs/fourier_dual_net/fdn_d3_gate/$region/best.pt \
        outputs/fourier_dual_net/fdn_d3_gate/$region/best.pt
done
```

- [ ] **Step 8.4: Run sanity checks on each region**

```bash
for region in ContraCosta Orange; do
    python scripts/analyze_gate.py \
        --ckpt outputs/fourier_dual_net/fdn_d3_gate/$region/best.pt \
        --region $region \
        --out_dir outputs/fourier_dual_net/fdn_d3_gate/$region/analysis \
        --device cpu
done
```

- [ ] **Step 8.5: Aggregate result table**

```bash
python -c "
import numpy as np
from pathlib import Path
for region in ['Alameda', 'ContraCosta', 'Orange']:
    p = Path(f'outputs/fourier_dual_net/fdn_d3_gate/{region}/test_predictions.npz')
    if not p.exists(): continue
    d = np.load(p)
    diff = np.abs(d['pred_raw_flow'] - d['actual_future_flow'])
    m = d['y_mask_flow']
    aff_TpN = np.broadcast_to(d['affected_mask'][:, None, :], diff.shape)
    print(f'{region:12s}  all={diff[m].mean():.3f}  aff={diff[m & aff_TpN].mean():.3f}  un={diff[m & ~aff_TpN].mean():.3f}')
"
```

Compare against baseline FDN learnable_K3 numbers from existing `outputs/fourier_dual_net/learnable_K3/<region>/test_predictions.npz`. Decision: 2/3 regions must improve by ≥ 0.15 to call D3 a generalizable win.

- [ ] **Step 8.6: Commit + write findings memo**

```bash
git add outputs/fourier_dual_net/fdn_d3_gate/
git commit -m "S3 D3 gated fusion results on CC + Orange"
```

---

## Task 9: Fallback — if S1 fails the decision gate

Only execute if Task 6.6 shows Alameda MAE drop < 0.15.

**Files:**
- Modify: `fourier_dual_net/model.py` (add optional L1 reg in `GatedFusion`)
- Modify: `scripts/train_fourier_dual_net.py` (pass `--gate_l1_lambda` to model and into loss)

This task is intentionally deferred and not pre-specified in code — it depends on the failure mode observed in S1 sanity checks. The spec's fallback sequence is:

1. **Fallback A** (gate polarization): if Check 1 fails with std too high (α near 0 or 1), add L1 `λ * |α - 0.5|` with λ=0.01 and retrain.
2. **Fallback B** (weak inputs): strip gate inputs to only `spectral_energy_ratio` (drop node_emb + time_avg), retrain.
3. **Abandon D3**: pivot to D1 (per-node K from autocorrelation features) — new design spec needed.

When this task triggers, return to brainstorming/writing-plans rather than guessing which fallback to apply.

---

## Self-Review

**1. Spec coverage**: walked through the spec sections:
- §Goal (gate fusion) → Tasks 1+2 ✓
- §Architecture data flow + Gate inputs → Task 1 ✓
- §Initialization (α=0.5 + scale=2 + node_emb=0) → Task 1.4 + test 1.2 ✓
- §Design decisions (per-(node,horizon), learnable scale, no reg, no dropout) → Task 1 ✓
- §Training plan (from scratch, 30 ep) → Tasks 6, 8 ✓
- §Sanity checks (4 checks) → Task 4 + 7 ✓
- §Risks / Fallback → Task 9 ✓
- §Files to modify → matches File Plan ✓

**2. Placeholder scan**: no TBD/TODO. Every code block is complete. Fallback (Task 9) is deliberately unspecified because the choice depends on failure mode — flagged explicitly.

**3. Type consistency**:
- `GatedFusion.__init__(num_nodes, T_p, d_node, hidden)` matches `FourierDualNet.__init__` passing `num_nodes=num_nodes, T_p=T_p, d_node=gate_d_node, hidden=gate_hidden` ✓
- `forward(x_main, x_pert, y_main, y_pert, time_feat)` returns `(y, alpha)` — used identically in `FourierDualNet.forward` and tests ✓
- CLI flags `--use_gated_fusion --gate_d_node --gate_hidden` match constructor kwargs in Task 3.2 ✓

No bugs found.
