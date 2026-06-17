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
