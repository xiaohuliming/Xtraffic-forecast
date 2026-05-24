"""Mode-collapse and training-health monitors per design doc §15.3.

Computed once per (logged) batch on the model output dict. Cheap operations
(< 1 ms typical). Returns a flat dict of scalars suitable for JSON-lining.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def branch_cosine_sim(z_normal: torch.Tensor, z_incident: torch.Tensor) -> float:
    """Mean cosine between two branches' per-node representations.
    Healthy 0.2-0.6, alert >0.85 (representations collapsing to same function).
    """
    zn = F.normalize(z_normal, dim=-1)
    zi = F.normalize(z_incident, dim=-1)
    cos = (zn * zi).sum(dim=-1)                                 # (B, N)
    return float(cos.mean())


@torch.no_grad()
def gate_stats(g: torch.Tensor, affected_mask: torch.Tensor) -> dict[str, float]:
    """g: (B, N, T_p), affected_mask: (B, N) bool.
    g.mean ~ overall gate activity; ratio = affected/unaffected.
    """
    g_node = g.mean(dim=-1)                                     # (B, N)
    g_mean = float(g_node.mean())
    g_std = float(g_node.std())
    aff = affected_mask.bool()
    any_aff = aff.any().item()
    if any_aff:
        g_aff = float(g_node[aff].mean())
    else:
        g_aff = float("nan")
    any_un = (~aff).any().item()
    if any_un:
        g_un = float(g_node[~aff].mean())
    else:
        g_un = float("nan")
    ratio = g_aff / max(g_un, 1e-6) if any_aff and any_un else float("nan")
    return {
        "gate.mean": g_mean,
        "gate.std": g_std,
        "gate.affected_mean": g_aff,
        "gate.unaffected_mean": g_un,
        "gate.aff_over_un": ratio,
    }


@torch.no_grad()
def over_smoothing_sim(z: torch.Tensor, n_pairs: int = 256) -> float:
    """Random-pair cosine sim of node representations within each sample.
    High value (>0.85) at deep layers = potential over-smoothing.
    """
    B, N, _ = z.shape
    zn = F.normalize(z, dim=-1)
    g = torch.Generator(device=z.device).manual_seed(0)
    a = torch.randint(0, N, (n_pairs,), generator=g, device=z.device)
    b = torch.randint(0, N, (n_pairs,), generator=g, device=z.device)
    cos = (zn[:, a] * zn[:, b]).sum(dim=-1).mean()
    return float(cos)


@torch.no_grad()
def delta_stats(delta_pred: torch.Tensor) -> dict[str, float]:
    return {
        "delta_pred.abs_mean": float(delta_pred.abs().mean()),
        "delta_pred.std": float(delta_pred.std()),
    }


@torch.no_grad()
def scale_mixing_weights(model) -> dict[str, float]:
    """Read the current softmax mixing weights of multi-scale patching from
    both branches. Tells us if a scale is collapsing to single."""
    out: dict[str, float] = {}
    for branch_name in ("normal", "incident"):
        try:
            patch = getattr(getattr(model, branch_name), "patching")
            α = patch.mixing_weights().detach().cpu().tolist()
            for i, name in enumerate(["long", "mid", "short"]):
                out[f"alpha.{branch_name}.{name}"] = float(α[i])
        except AttributeError:
            pass
    return out


def collect_all(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                model) -> dict[str, float]:
    """Single-call monitor collection. Cheap; safe to run every logged batch."""
    out: dict[str, float] = {
        "monitor.branch_cosine": branch_cosine_sim(output["z_normal"], output["z_incident"]),
        "monitor.os_z_normal": over_smoothing_sim(output["z_normal"]),
        "monitor.os_z_incident": over_smoothing_sim(output["z_incident"]),
    }
    out.update(gate_stats(output["g"], batch["affected_mask"]))
    out.update(delta_stats(output["delta_pred"]))
    out.update(scale_mixing_weights(model))
    return out
