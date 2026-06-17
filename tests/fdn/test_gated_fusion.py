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
    assert g.gate_mlp[-1].weight.grad.abs().sum().item() > 0
    assert g.output_scale.grad is not None and g.output_scale.grad.abs().item() > 0


def test_energy_ratio_uses_flow_channel_only():
    g = GatedFusion(num_nodes=3, T_p=12, d_node=4)
    B, N, T_h = 2, 3, 12
    xm = torch.zeros(B, N, T_h, 3); xp = torch.zeros(B, N, T_h, 3)
    xm[..., 0] = 1.0; xp[..., 0] = 1.0
    er = g._energy_ratio(xm, xp)
    assert er.shape == (B, N)
    xm2 = xm.clone(); xm2[..., 1] = 1000.0
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
