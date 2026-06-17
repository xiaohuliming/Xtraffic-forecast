# RGDN 去季节化残差引导双支网络 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 RGDN 并在参数对齐下跑通 V0a/V0b/V1/V2 决定性消融,回答"去季节化加残差引导的非对称双支,能否在等参数下超过单 GWN"。

**Architecture:** 复用缓存的 train-only 事故-masked 气候态基线 `baseline_median[day_kind,tod]` 做去季节化,网络只预测偏离量。残差支线用 GWN 图卷做空间扰动传播,主支线用 GWN 关图卷的纯 TCN 做节点局部修正,并经一步自适应邻接图卷接收邻居残差摘要。`ŷ = y_baseline + (Δ̂_local + Δ̂_spatial)·sd_res`。

**Tech Stack:** Python 3, numpy 本地可跑;torch + GWN backbone 在 5080。模型与训练脚本复用 `baselines/GraphWaveNet/model.py` 的 `gwnet`、`dist_net/data.py`、`scripts/train_staeformer_xtraffic.py` 的管线与 npz schema。

**Spec:** `docs/superpowers/specs/2026-06-17-rgdn-deseasonalized-residual-guided-design.md`

---

## 执行环境约定

- 本地 Mac:有 numpy,无 torch、无 h5py。纯 numpy 文件可本地跑 pytest;含 torch 的文件本地只能
  `python3 -m py_compile` 做语法检查,行为验证在 5080。
- 5080:LAN `asus@192.168.31.13` 密码 `Hzj050916`,代码在 `C:/Users/asus/traffic_fourier`,
  python `C:\Python313\python.exe`,torch 在 user-site。长训练按 CLAUDE.md 的隐藏计划任务手册
  跑,smoke 这种 <2 分钟的可直接 ssh 前台跑。
- 每个 task 末尾 commit。提交到当前分支 `feat/d3-gated-fusion`。commit 信息英文,结尾带
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `fourier_dual_net/deseason.py` | 纯 numpy:基线查表 + 训练段残差标准差 | 新建 |
| `tests/fourier_dual_net/test_deseason.py` | deseason 的本地 numpy 单测 | 新建 |
| `dist_net/data.py` | get_sample 增补 `x_baseline` 历史基线 key | 改 |
| `fourier_dual_net/rgdn.py` | InjectionGraphConv + RGDN 模型,变体开关 | 新建 |
| `tests/fourier_dual_net/test_rgdn.py` | RGDN 的 torch 单测,5080 跑 | 新建 |
| `scripts/train_rgdn.py` | 训练/评测,`--variant`,sd_res 统计,`--smoke`,npz 产出 | 新建 |

---

## Task 1: 去季节化纯函数 deseason.py（本地完整 TDD）

**Files:**
- Create: `fourier_dual_net/deseason.py`
- Test: `tests/fourier_dual_net/test_deseason.py`

- [ ] **Step 1: 写失败测试** [LOCAL]

写 `tests/fourier_dual_net/test_deseason.py`:

```python
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fourier_dual_net.deseason import lookup_baseline, train_residual_std


def test_lookup_baseline_picks_right_bins():
    N, C = 3, 2
    bm = np.arange(2 * 288 * N * C, dtype=np.float32).reshape(2, 288, N, C)
    T = 10
    day_kind = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    tod = np.arange(T, dtype=np.int64)
    out = lookup_baseline(bm, day_kind, tod, 2, 6)        # steps 2..5
    assert out.shape == (4, N, C)
    assert np.allclose(out[0], bm[1, 2])                  # step2: dk=1 tod=2
    assert np.allclose(out[3], bm[0, 5])                  # step5: dk=0 tod=5


def test_train_residual_std_train_only_and_masked():
    N, T, hi = 4, 20, 10
    bm = np.zeros((2, 288, N, 2), dtype=np.float32)       # baseline 0 -> res == flow
    day_kind = np.zeros(T, dtype=np.int64)
    tod = np.zeros(T, dtype=np.int64)
    flow = np.ones((T, N, 2), dtype=np.float32)
    mask = np.ones((T, N, 2), dtype=bool)
    flow[hi:] = 1e6                                       # post-train poison, must be ignored
    flow[0, 0, 0] = 1e6; mask[0, 0, 0] = False            # masked-out poison, must be ignored
    sd = train_residual_std(flow, mask, bm, day_kind, tod, hi, ch=0)
    assert sd < 1e-3, sd                                  # all valid train residuals == 1.0 -> std ~ 0
```

- [ ] **Step 2: 跑测试确认失败** [LOCAL]

Run: `python3 -m pytest tests/fourier_dual_net/test_deseason.py -q`
Expected: FAIL,`ModuleNotFoundError` 或 `ImportError: cannot import name 'lookup_baseline'`。

- [ ] **Step 3: 实现 deseason.py** [LOCAL]

写 `fourier_dual_net/deseason.py`:

```python
"""Numpy-only de-seasonalization helpers for RGDN. No torch import so it runs locally."""
from __future__ import annotations

import numpy as np


def lookup_baseline(baseline_median: np.ndarray, day_kind: np.ndarray,
                    tod: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """baseline_median (2,288,N,C) -> (hi-lo, N, C) for time steps [lo, hi)."""
    return baseline_median[day_kind[lo:hi], tod[lo:hi]]


def train_residual_std(flow_series: np.ndarray, flow_mask: np.ndarray,
                       baseline_median: np.ndarray, day_kind: np.ndarray,
                       tod: np.ndarray, hi: int, ch: int = 0,
                       floor: float = 1e-6) -> float:
    """Masked std of (flow - baseline) over train steps [0, hi), channel ch."""
    flow = flow_series[:hi, :, ch]                                  # (hi, N)
    mask = flow_mask[:hi, :, ch].astype(bool)
    base = baseline_median[day_kind[:hi], tod[:hi]][:, :, ch]       # (hi, N)
    res = flow - base
    vals = res[mask]
    if vals.size == 0:
        return 1.0
    return float(vals.std() + floor)
```

- [ ] **Step 4: 跑测试确认通过** [LOCAL]

Run: `python3 -m pytest tests/fourier_dual_net/test_deseason.py -q`
Expected: PASS,2 passed。

- [ ] **Step 5: commit**

```bash
git add fourier_dual_net/deseason.py tests/fourier_dual_net/test_deseason.py
git commit -m "feat: numpy de-seasonalization helpers + local tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: get_sample 增补历史基线 x_baseline

**Files:**
- Modify: `dist_net/data.py`（get_sample 内,y_baseline 计算之后;返回 dict 增 key）

- [ ] **Step 1: 加历史基线查表** [LOCAL]

在 `dist_net/data.py` 的 `get_sample` 里,`y_baseline = np.transpose(...)` 那几行之后、
`time_enc = ...` 之前,插入:

```python
        # x_baseline lookup via (day_kind, tod) for each history step (mirror y_baseline)
        hist_idx = np.arange(hist_lo, hist_hi)
        x_baseline_t = self.baseline_median[self.day_kind[hist_idx], self.tod[hist_idx]]  # (T_h,N,3)
        x_baseline = np.transpose(x_baseline_t, (1, 0, 2)).astype(np.float32)             # (N,T_h,3)
```

- [ ] **Step 2: 加进返回 dict** [LOCAL]

在 `get_sample` 返回的 dict 里,`"y_baseline": y_baseline,` 这一行后面加一行:

```python
            "x_baseline":     x_baseline,                        # (N, T_h, 3)
```

- [ ] **Step 3: 语法检查** [LOCAL]

Run: `python3 -m py_compile dist_net/data.py`
Expected: 无输出即通过。行为正确性在 Task 5 的 5080 smoke 里验证 `x_baseline` 形状与查表一致。

- [ ] **Step 4: commit**

```bash
git add dist_net/data.py
git commit -m "feat: expose history baseline x_baseline in get_sample

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: RGDN 模型 rgdn.py

**Files:**
- Create: `fourier_dual_net/rgdn.py`
- Test: `tests/fourier_dual_net/test_rgdn.py`（5080 跑）

- [ ] **Step 1: 写 rgdn.py** [LOCAL 写 + py_compile]

写 `fourier_dual_net/rgdn.py`:

```python
"""RGDN: de-seasonalized residual-guided dual-branch forecaster.

Decomposition reuses the cache climatology (baseline added back as a known skeleton);
the network predicts only the standardized deviation. Variants are flag-driven:
  deseason: subtract baseline & predict deviation (else z-scored raw flow)
  dual:     two branches (else single GWN baseline)
  main_gcn: main branch uses graph conv (else node-local TCN)
  inject:   feed neighbor-residual summary into the main branch
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines" / "GraphWaveNet"))
from model import gwnet  # noqa: E402


def _make_gwn(num_nodes, supports, device, in_dim, T_p, nhid, gcn_bool, dropout):
    return gwnet(device=device, num_nodes=num_nodes, dropout=dropout, supports=supports,
                 gcn_bool=gcn_bool, addaptadj=gcn_bool, in_dim=in_dim, out_dim=T_p,
                 residual_channels=nhid, dilation_channels=nhid,
                 skip_channels=nhid * 8, end_channels=nhid * 16, blocks=4, layers=2)


class InjectionGraphConv(nn.Module):
    """One adaptive-adjacency diffusion step over residuals -> per-node neighbor summary.

    res (B,N,T) standardized flow residual -> (B, c_out, N, T) in gwnet channel layout.
    """

    def __init__(self, num_nodes: int, c_out: int = 4, d_emb: int = 10):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, d_emb) * 0.05)
        self.e2 = nn.Parameter(torch.randn(d_emb, num_nodes) * 0.05)
        self.proj = nn.Conv2d(1, c_out, kernel_size=(1, 1))

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)   # (N,N) rows sum to 1
        summary = torch.einsum("nm,bmt->bnt", adp, res)             # (B,N,T) neighbor-weighted res
        return self.proj(summary.unsqueeze(1))                      # (B,c_out,N,T)


class RGDN(nn.Module):
    def __init__(self, num_nodes, supports, T_h, T_p, device=None,
                 deseason=True, dual=True, main_gcn=False, inject=True,
                 nhid_single=32, nhid_main=26, nhid_res=22, c_inject=4, dropout=0.3):
        super().__init__()
        self.deseason = bool(deseason)
        self.dual = bool(dual)
        self.inject = bool(inject) and self.dual
        self.T_p = T_p
        # train statistics set from the script before training
        self.register_buffer("sd_res", torch.tensor(1.0))
        self.register_buffer("flow_mu", torch.tensor(0.0))
        self.register_buffer("flow_sd", torch.tensor(1.0))

        if not self.dual:
            self.single = _make_gwn(num_nodes, supports, device, 1, T_p, nhid_single, True, dropout)
            return

        self.inject_mod = InjectionGraphConv(num_nodes, c_out=c_inject) if self.inject else None
        main_in = 1 + 2 + (c_inject if self.inject else 0)          # res + tod/dow + injection
        self.main_branch = _make_gwn(num_nodes, supports, device, main_in, T_p, nhid_main,
                                     main_gcn, dropout)
        self.res_branch = _make_gwn(num_nodes, supports, device, 1, T_p, nhid_res, True, dropout)

    @staticmethod
    def _to_gwnet(x):   # (B,N,T,C)->(B,C,N,T)
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _from_gwnet(out):   # (B,T_p,N,1)->(B,N,T_p)
        return out.squeeze(-1).permute(0, 2, 1).contiguous()

    def _signal(self, x_hist):
        flow = x_hist[..., 0]                                       # (B,N,T_h)
        if self.deseason:
            return flow, None                                      # baseline subtracted in forward
        return (flow - self.flow_mu) / self.flow_sd, None

    def forward(self, x_hist, x_baseline, y_baseline, time_feat):
        flow = x_hist[..., 0]                                       # (B,N,T_h)
        if self.deseason:
            sig = (flow - x_baseline[..., 0]) / self.sd_res         # standardized residual
        else:
            sig = (flow - self.flow_mu) / self.flow_sd

        if not self.dual:
            out = self._from_gwnet(self.single(self._to_gwnet(sig.unsqueeze(-1))))
            return self._reseason(out, y_baseline)

        B, N, T_h = sig.shape
        y_res = self._from_gwnet(self.res_branch(self._to_gwnet(sig.unsqueeze(-1))))
        tf = time_feat.unsqueeze(1).expand(B, N, T_h, 2)
        feats = [sig.unsqueeze(-1), tf]
        if self.inject:
            inj = self.inject_mod(sig).permute(0, 2, 3, 1)          # (B,N,T_h,c_inject)
            feats.append(inj)
        y_main = self._from_gwnet(self.main_branch(self._to_gwnet(torch.cat(feats, dim=-1))))
        return self._reseason(y_main + y_res, y_baseline)

    def _reseason(self, out, y_baseline):
        if self.deseason:
            return y_baseline[..., 0] + out * self.sd_res
        return out * self.flow_sd + self.flow_mu
```

- [ ] **Step 2: 语法检查** [LOCAL]

Run: `python3 -m py_compile fourier_dual_net/rgdn.py`
Expected: 无输出即通过。

- [ ] **Step 3: 写 torch 单测** [LOCAL 写]

写 `tests/fourier_dual_net/test_rgdn.py`:

```python
"""RGDN torch tests. Run on the 5080 (needs torch)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))
from fourier_dual_net.rgdn import RGDN, InjectionGraphConv

N, T_h, T_p, B = 6, 12, 12, 2


def _supports(device):
    A = torch.eye(N, device=device)
    return [A, A]


def _batch(device):
    return (torch.randn(B, N, T_h, 3, device=device),    # x_hist
            torch.randn(B, N, T_h, 3, device=device),    # x_baseline
            torch.randn(B, N, T_p, 3, device=device),    # y_baseline
            torch.rand(B, T_h, 2, device=device))        # time_feat


def _flags(variant):
    return {
        "v0a": dict(deseason=False, dual=False),
        "v0b": dict(deseason=True, dual=False),
        "v1":  dict(deseason=True, dual=True, main_gcn=False, inject=True),
        "v2":  dict(deseason=True, dual=True, main_gcn=False, inject=False),
        "v3":  dict(deseason=True, dual=True, main_gcn=True, inject=False),
        "v4":  dict(deseason=False, dual=True, main_gcn=False, inject=True),
    }[variant]


def test_injection_adp_rows_sum_to_one():
    inj = InjectionGraphConv(N, c_out=4)
    res = torch.randn(B, N, T_h)
    out = inj(res)
    assert out.shape == (B, 4, N, T_h)
    adp = torch.softmax(torch.relu(inj.e1 @ inj.e2), dim=1)
    assert torch.allclose(adp.sum(dim=1), torch.ones(N), atol=1e-5)


def test_all_variants_forward_shape_and_finite():
    dev = torch.device("cpu")
    for v in ["v0a", "v0b", "v1", "v2", "v3", "v4"]:
        m = RGDN(N, _supports(dev), T_h, T_p, device=dev, **_flags(v))
        y = m(*_batch(dev))
        assert y.shape == (B, N, T_p), (v, y.shape)
        assert torch.isfinite(y).all(), v


def test_main_branch_has_no_graph_params_when_gcn_off():
    dev = torch.device("cpu")
    m = RGDN(N, _supports(dev), T_h, T_p, device=dev, **_flags("v1"))
    assert not hasattr(m.main_branch, "nodevec1")     # gcn_bool=False -> no adaptive adjacency
    assert len(m.res_branch.gconv) > 0                # residual branch keeps graph conv


def test_gradient_flows_v1():
    dev = torch.device("cpu")
    m = RGDN(N, _supports(dev), T_h, T_p, device=dev, **_flags("v1"))
    y = m(*_batch(dev))
    y.sum().backward()
    g = m.inject_mod.e1.grad
    assert g is not None and torch.isfinite(g).all()
```

- [ ] **Step 4: commit（行为验证在 Task 5）**

```bash
git add fourier_dual_net/rgdn.py tests/fourier_dual_net/test_rgdn.py
git commit -m "feat: RGDN model + injection graph conv, variant-flag driven

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 训练脚本 train_rgdn.py

**Files:**
- Create: `scripts/train_rgdn.py`

- [ ] **Step 1: 写 train_rgdn.py** [LOCAL 写 + py_compile]

写 `scripts/train_rgdn.py`:

```python
#!/usr/bin/env python3
"""Train RGDN variants on one XTraffic region. Same pipeline / masked-MAE / npz schema
as train_staeformer_xtraffic.py so numbers compare directly to FDN/GWN/STAEformer.

Variants: v0a single GWN raw | v0b single GWN de-seasonalized | v1 RGDN |
v2 RGDN no-inject | v3 dual main-gcn-on | v4 RGDN no-deseason.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "GraphWaveNet"))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dist_net.data import MultiRegionDataset, make_loader
from fourier_dual_net.rgdn import RGDN
from fourier_dual_net.deseason import train_residual_std

VARIANTS = {
    "v0a": dict(deseason=False, dual=False, main_gcn=False, inject=False),
    "v0b": dict(deseason=True,  dual=False, main_gcn=False, inject=False),
    "v1":  dict(deseason=True,  dual=True,  main_gcn=False, inject=True),
    "v2":  dict(deseason=True,  dual=True,  main_gcn=False, inject=False),
    "v3":  dict(deseason=True,  dual=True,  main_gcn=True,  inject=False),
    "v4":  dict(deseason=False, dual=True,  main_gcn=False, inject=True),
}


def build_adj_supports(edge_index, N, device):
    A = np.zeros((N, N), dtype=np.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    np.fill_diagonal(A, 1.0)
    deg = A.sum(axis=1)
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    return [torch.from_numpy(deg_inv[:, None] * A).to(device),
            torch.from_numpy(deg_inv[:, None] * A.T).to(device)]


def masked_mae(pred, target, mask):
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def make_model(variant, N, supports, T_h, T_p, device, args):
    m = RGDN(N, supports, T_h, T_p, device=device,
             nhid_single=args.nhid_single, nhid_main=args.nhid_main,
             nhid_res=args.nhid_res, c_inject=args.c_inject, dropout=args.dropout,
             **VARIANTS[variant]).to(device)
    return m


def train_stats(rdata, T_h, T_p):
    flows = rdata.flow_series[:, :, 0]
    fmask = rdata.flow_mask[:, :, 0].astype(bool)
    tr_ss = rdata.sample_start[rdata.split == 0]
    hi = int(tr_ss.max()) + T_h + T_p
    seg, segm = flows[:hi], fmask[:hi]
    mu, sd = float(seg[segm].mean()), float(seg[segm].std() + 1e-6)
    sd_res = train_residual_std(rdata.flow_series, rdata.flow_mask,
                                rdata.baseline_median, rdata.day_kind, rdata.tod, hi, ch=0)
    return mu, sd, sd_res


def forward_batch(model, batch, device):
    x_hist = batch["x_hist"].to(device)
    x_base = batch["x_baseline"].to(device)
    y_base = batch["y_baseline"].to(device)
    tf = batch["time_feat"].to(device)
    return model(x_hist, x_base, y_base, tf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda", choices=["Alameda", "ContraCosta", "Orange"])
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--graph_dir", default="outputs/region_graphs")
    p.add_argument("--out_dir", type=Path, default=Path("outputs/rgdn"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--nhid_single", type=int, default=32)
    p.add_argument("--nhid_main", type=int, default=26)
    p.add_argument("--nhid_res", type=int, default=22)
    p.add_argument("--c_inject", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true", help="build all variants, print params, 1 fwd/bwd, exit")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} region {args.region} variant {args.variant} seed {args.seed}", flush=True)

    train_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="train")
    rdata = train_ds.regions[args.region]
    N, T_h, T_p = int(rdata.N), int(rdata.T_h), int(rdata.T_p)
    supports = build_adj_supports(rdata.edge_index, N, device)
    mu, sd, sd_res = train_stats(rdata, T_h, T_p)
    print(f"N={N} T_h={T_h} T_p={T_p} flow mu={mu:.2f} sd={sd:.2f} sd_res={sd_res:.3f}", flush=True)

    if args.smoke:
        sample = make_loader(train_ds, batch_size=4, shuffle=False)
        batch = next(iter(sample))
        assert batch["x_baseline"].shape == batch["x_hist"].shape, "x_baseline shape mismatch"
        for v in VARIANTS:
            m = make_model(v, N, supports, T_h, T_p, device, args)
            m.sd_res.fill_(sd_res); m.flow_mu.fill_(mu); m.flow_sd.fill_(sd)
            nparam = sum(q.numel() for q in m.parameters() if q.requires_grad)
            y = forward_batch(m, batch, device)
            loss = masked_mae(y, batch["y_true"][..., 0].to(device), batch["y_mask"][..., 0].to(device))
            loss.backward()
            print(f"  {v:4s} params={nparam:,} out={tuple(y.shape)} loss={loss.item():.3f} finite={bool(torch.isfinite(y).all())}", flush=True)
        print("SMOKE_OK", flush=True)
        return

    val_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="val")
    test_ds = MultiRegionDataset([args.region], args.data_dir, args.graph_dir, split="test")
    out_dir = args.out_dir / args.region / f"{args.variant}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(args.variant, N, supports, T_h, T_p, device, args)
    model.sd_res.fill_(sd_res); model.flow_mu.fill_(mu); model.flow_sd.fill_(sd)
    nparam = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"variant {args.variant} params={nparam:,}", flush=True)

    opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False)
    nb = (len(train_ds) + args.batch_size - 1) // args.batch_size
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1) * max(nb, 1), eta_min=args.lr * 1e-2)

    def ev(loader):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                y = forward_batch(model, batch, device)
                t = batch["y_true"][..., 0].to(device); msk = batch["y_mask"][..., 0].to(device)
                tot += float(masked_mae(y, t, msk).item()) * y.size(0); n += y.size(0)
        model.train(); return tot / max(n, 1)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); s, n = 0.0, 0
        for batch in train_loader:
            y = forward_batch(model, batch, device)
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
    sample_start = np.empty((S,), dtype=np.int64)
    region_code = np.empty((S,), dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            y = forward_batch(model, batch, device).permute(0, 2, 1).cpu().numpy()   # (B,T_p,N)
            bs = y.shape[0]
            pred_flow[cursor:cursor+bs] = y
            actual_flow[cursor:cursor+bs] = batch["y_true"][..., 0].permute(0, 2, 1).numpy()
            y_mask_flow[cursor:cursor+bs] = batch["y_mask"][..., 0].permute(0, 2, 1).numpy()
            affected[cursor:cursor+bs] = batch["affected_mask"].numpy()
            sample_start[cursor:cursor+bs] = batch["sample_start"].numpy()
            region_code[cursor:cursor+bs] = batch["region_code"].numpy()
            cursor += bs

    np.savez_compressed(out_dir / "test_predictions.npz",
                        region_code=region_code, sample_start=sample_start,
                        region_node_idx=rdata.region_idx.astype(np.int64),
                        pred_raw_flow=pred_flow, actual_future_flow=actual_flow,
                        y_mask_flow=y_mask_flow, affected_mask=affected)
    diff = np.abs(pred_flow - actual_flow)
    aff3 = np.broadcast_to(affected[:, None, :], (S, T_p, N))
    res = {"all": float(diff[y_mask_flow].mean()),
           "affected": float(diff[y_mask_flow & aff3].mean()),
           "unaffected": float(diff[y_mask_flow & ~aff3].mean()),
           "best_val": best, "seed": args.seed, "variant": args.variant, "params": nparam}
    print(f"\ntest MAE all={res['all']:.3f} affected={res['affected']:.3f} "
          f"unaffected={res['unaffected']:.3f}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查** [LOCAL]

Run: `python3 -m py_compile scripts/train_rgdn.py`
Expected: 无输出即通过。

- [ ] **Step 3: commit**

```bash
git add scripts/train_rgdn.py
git commit -m "feat: train_rgdn.py with variant map, sd_res stats, smoke mode

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 5080 pre-flight（smoke + 参数对齐校准 + torch 单测）

**Files:** 无新增。在 5080 上验证并据结果回头微调 `--nhid_main/--nhid_res`。

- [ ] **Step 1: 同步代码到 5080** [5080]

把本地已提交的改动推到 5080。优先 git:在 5080 的 `C:/Users/asus/traffic_fourier` 上
`git pull`(或 push 本地分支后 pull)。若无 git 远端,scp 这几个文件:
`fourier_dual_net/deseason.py`、`fourier_dual_net/rgdn.py`、`scripts/train_rgdn.py`、
`dist_net/data.py`、`tests/fourier_dual_net/test_deseason.py`、`tests/fourier_dual_net/test_rgdn.py`。

- [ ] **Step 2: 跑 torch 单测** [5080]

Run（5080,直接前台,<1 分钟）:
`C:\Python313\python.exe -X utf8 -m pytest tests/fourier_dual_net/test_rgdn.py tests/fourier_dual_net/test_deseason.py -q`
Expected: 全部 PASS。重点过 `test_main_branch_has_no_graph_params_when_gcn_off` 与
`test_all_variants_forward_shape_and_finite`。

- [ ] **Step 3: 跑 smoke,核对参数与前向** [5080]

Run（5080,<2 分钟）:
`C:\Python313\python.exe -X utf8 scripts/train_rgdn.py --region Alameda --variant v1 --smoke`
Expected: 末行 `SMOKE_OK`,且打印每个变体 `params=...`。验证三件事:
1. `x_baseline shape mismatch` 断言不触发,即 Task 2 的 x_baseline 正确进 batch。
2. 六个变体 out 均为 `(4, 521, 12)`、finite=True。
3. 记录各变体参数,准备校准。

- [ ] **Step 4: 参数对齐校准** [5080]

目标:V1/V2/V3/V4 的总参数都落在 V0a/V0b 单 GWN 的 P 的 ±3% 内。
读 smoke 打印:若 dual 变体明显低于 P,调大 `--nhid_main`(主支无图卷,最省参,优先动它);
若高于 P,调小。重复 smoke 直到对齐。把最终 `--nhid_main/--nhid_res` 写进 Step 5 的训练命令。
注意 V1 与 V2/V3 因注入与图卷开关参数略有差异,以"四个 dual 变体都在 P 的 ±3%"为准,
跑前打印留底。

- [ ] **Step 5: commit 校准后的默认值（如改了脚本默认）**

若把校准好的 nhid 写回 `scripts/train_rgdn.py` 的 argparse 默认:

```bash
git add scripts/train_rgdn.py
git commit -m "chore: param-matched nhid defaults for RGDN dual variants

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 第一轮决定性实验（Alameda 种子 42:V0a/V0b/V1/V2）

**Files:** 产物 `outputs/rgdn/Alameda/{variant}_seed42/summary.json` 与 `test_predictions.npz`。

- [ ] **Step 1: 排队四个 run** [5080]

按 CLAUDE.md 的隐藏计划任务手册,写一个 .bat 队列(CRLF) + VBS 隐藏包装,顺序跑:

```
C:\Python313\python.exe -X utf8 scripts\train_rgdn.py --region Alameda --variant v0a --seed 42 --device cuda
C:\Python313\python.exe -X utf8 scripts\train_rgdn.py --region Alameda --variant v0b --seed 42 --device cuda
C:\Python313\python.exe -X utf8 scripts\train_rgdn.py --region Alameda --variant v1 --seed 42 --device cuda
C:\Python313\python.exe -X utf8 scripts\train_rgdn.py --region Alameda --variant v2 --seed 42 --device cuda
```

schtasks 创建 → run → 确认日志出现首个 `ep01` 后**立即删除任务**(避免占位触发二次发火)。
每 run <1h,四个串行约 2-3h。

- [ ] **Step 2: 监控** [5080]

Monitor 过滤词至少含:`test MAE|val L|Traceback|forrtl|aborting|CUDA out of memory|SMOKE_OK`。
读远端日志用 `python -X utf8` + base64 经 ssh 管道,避免 GBK 乱码。

- [ ] **Step 3: 取回结果** [5080->本地]

把四个 `summary.json` 与 `test_predictions.npz` scp 回本地 `outputs/rgdn/Alameda/`。

- [ ] **Step 4: 判读** [LOCAL]

读四个 summary.json 的 all/affected/unaffected。判据(对照已知种子噪声带约 0.04-0.08):
- V0b vs V0a:去季节化是否本身有用。
- V1 vs V0b:完整机制是否在等参数下超过去季节化单 GWN,超过 >噪声带才算机制成立。
- V1 vs V2:注入的净效应。
落在噪声带内就是诚实负结果,照常记录。

---

## Task 7: 落盘结论与同步状态

**Files:**
- Create: `outputs/diagnostics/rgdn_round1_results.txt`
- Modify: `CLAUDE.md`（一句话现状 + 下一步）
- Modify: 记忆 `project_fdn_architecture_attempts` 或新建 `project_rgdn`

- [ ] **Step 1: 写结果文件** [LOCAL]

把四个变体的 all/affected/unaffected、参数数、种子、判读结论写进
`outputs/diagnostics/rgdn_round1_results.txt`,数字全部出自 summary.json,严禁估算。

- [ ] **Step 2: 更新 CLAUDE.md 现状** [LOCAL]

把"一句话现状"更新为 RGDN 第一轮结果与下一步(机制成立→补 V3/V4 + 多种子 + 扩区;
不成立→记录负结果,回到 XTraffic 应用论文主线)。

- [ ] **Step 3: 更新记忆** [LOCAL]

按 RGDN 结果更新或新建记忆文件,写清是正向还是负结果、参数是否对齐、下一步。

- [ ] **Step 4: commit**

```bash
git add outputs/diagnostics/rgdn_round1_results.txt CLAUDE.md
git commit -m "results: RGDN round-1 Alameda V0a/V0b/V1/V2 ablation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 后续（不在本 plan,机制成立后再做）

- 补 V3/V4,补种子 42/1/2 出噪声带,扩 ContraCosta + Orange。
- 与 STAEformer 同台;显著性用 `significance_tests.py`。
- 异构骨干第二阶段:主支换 STID/Transformer。per-bin baseline_scale 标准化。完整 dow 基线消融。
