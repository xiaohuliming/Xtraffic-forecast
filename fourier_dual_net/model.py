from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines" / "GraphWaveNet"))
from model import gwnet  # noqa: E402


class FourierDecomp(nn.Module):
    """Split flow signal into low-freq (Main) and high-freq (Pert) via FFT mask.

    Three modes:
      - fixed_k:      hard cutoff. Bins [0, K) → Main; [K, T/2+1) → Pert.
      - learnable:    7 global sigmoid weights, initialised as step at K.
      - conditioned:  per-(sample, sensor) mask via MLP from sensor_meta + ToD/DoW.
                      Mask shape (B, N, n_bins). Direction E.

    Forward signature: x (B, N, T, C). Optional sensor_meta (N, C_meta), time_feat (B, T, 2).
    """

    def __init__(self, T_h: int, K: int = 3, mode: str = "fixed_k",
                 sensor_meta_dim: int | None = None,
                 cond_hidden: int = 32):
        super().__init__()
        assert mode in ("fixed_k", "learnable", "conditioned")
        self.T_h = T_h
        self.K = K
        self.mode = mode
        self.n_bins = T_h // 2 + 1

        if mode == "learnable":
            init = torch.zeros(self.n_bins)
            init[:K] = 2.0
            init[K:] = -2.0
            self.bin_logit = nn.Parameter(init)
        elif mode == "conditioned":
            assert sensor_meta_dim is not None, "conditioned mode requires sensor_meta_dim"
            # Input: sensor static meta + time features (averaged over T window) = sensor_meta_dim + 2
            # Output: n_bins logits
            self.cond_mlp = nn.Sequential(
                nn.Linear(sensor_meta_dim + 2, cond_hidden),
                nn.ReLU(),
                nn.Linear(cond_hidden, self.n_bins),
            )
            # Initialize bias to step at K (so conditioned mask starts ≈ learnable default)
            with torch.no_grad():
                bias = torch.zeros(self.n_bins)
                bias[:K] = 2.0
                bias[K:] = -2.0
                self.cond_mlp[-1].bias.copy_(bias)
                # Small init weights so output ≈ bias initially
                self.cond_mlp[-1].weight.mul_(0.01)
        else:
            self.register_buffer("bin_logit", torch.zeros(self.n_bins))

    def get_mask(self, sensor_meta: torch.Tensor | None = None,
                 time_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Return mask. Shape depends on mode:
        - fixed_k / learnable: (n_bins,)  global
        - conditioned: (B, N, n_bins)
        """
        if self.mode == "learnable":
            return torch.sigmoid(self.bin_logit)
        elif self.mode == "conditioned":
            assert sensor_meta is not None and time_feat is not None
            B, T, _ = time_feat.shape
            N, C_meta = sensor_meta.shape
            time_avg = time_feat.mean(dim=1)              # (B, 2) — average ToD/DoW over window
            time_exp = time_avg.unsqueeze(1).expand(B, N, 2)        # (B, N, 2)
            meta_exp = sensor_meta.unsqueeze(0).expand(B, N, C_meta)  # (B, N, C_meta)
            inp = torch.cat([meta_exp, time_exp], dim=-1)    # (B, N, C_meta+2)
            return torch.sigmoid(self.cond_mlp(inp))         # (B, N, n_bins)
        else:
            m = torch.zeros(self.n_bins, device=self.bin_logit.device)
            m[: self.K] = 1.0
            return m

    def forward(self, x: torch.Tensor,
                sensor_meta: torch.Tensor | None = None,
                time_feat: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, T, C = x.shape
        assert T == self.T_h, f"expected T_h={self.T_h}, got {T}"

        x_perm = x.permute(0, 1, 3, 2).reshape(B * N * C, T)
        Fx = torch.fft.rfft(x_perm, dim=-1)                       # (B*N*C, n_bins)

        if self.mode == "conditioned":
            mask = self.get_mask(sensor_meta, time_feat)          # (B, N, n_bins)
            # broadcast to (B, N, C, n_bins) then flatten to match Fx
            mask_b = mask.unsqueeze(2).expand(B, N, C, self.n_bins).reshape(B * N * C, self.n_bins)
            mask_b = mask_b.to(Fx.dtype)
            F_main = Fx * mask_b
            F_pert = Fx * (1.0 - mask_b)
        else:
            mask = self.get_mask().to(Fx.dtype)                   # (n_bins,)
            F_main = Fx * mask
            F_pert = Fx * (1.0 - mask)

        x_main = torch.fft.irfft(F_main, n=T, dim=-1)
        x_pert = torch.fft.irfft(F_pert, n=T, dim=-1)
        x_main = x_main.reshape(B, N, C, T).permute(0, 1, 3, 2).contiguous()
        x_pert = x_pert.reshape(B, N, C, T).permute(0, 1, 3, 2).contiguous()
        return x_main, x_pert


class CrossBranchAttention(nn.Module):
    """Pert branch queries Main branch's signal at input level.

    For each (B, N) pair, attention along time dim T:
        Q = proj(x_pert)
        K, V = proj(x_main)
    Output is projected back to C dims and used as context for Pert branch.
    """

    def __init__(self, c_in: int, d_attn: int = 16, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj_q = nn.Linear(c_in, d_attn)
        self.proj_k = nn.Linear(c_in, d_attn)
        self.proj_v = nn.Linear(c_in, d_attn)
        self.attn = nn.MultiheadAttention(d_attn, n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(d_attn, c_in)
        self.layer_norm = nn.LayerNorm(c_in)

    def forward(self, x_pert: torch.Tensor, x_main: torch.Tensor) -> torch.Tensor:
        # x_pert, x_main: (B, N, T, C). Treat (B, N) as batch, attend over T.
        B, N, T, C = x_pert.shape
        q = self.proj_q(x_pert).reshape(B * N, T, -1)
        k = self.proj_k(x_main).reshape(B * N, T, -1)
        v = self.proj_v(x_main).reshape(B * N, T, -1)
        attn_out, _ = self.attn(q, k, v)
        ctx = self.out_proj(attn_out).reshape(B, N, T, C)
        return self.layer_norm(x_pert + ctx)   # residual + LN


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
        e_main = (x_main[..., 0:1] ** 2).mean(dim=(2, 3))
        e_pert = (x_pert[..., 0:1] ** 2).mean(dim=(2, 3))
        return e_main / (e_main + e_pert + 1e-6)

    def forward(self, x_main, x_pert, y_main, y_pert, time_feat):
        B, N, T_p = y_main.shape
        assert T_p == self.T_p
        energy = self._energy_ratio(x_main, x_pert)
        node_e = self.node_emb.weight.unsqueeze(0).expand(B, N, self.d_node)
        time_avg = time_feat.mean(dim=1).unsqueeze(1).expand(B, N, 2)
        gate_in = torch.cat([energy.unsqueeze(-1), node_e, time_avg], dim=-1)
        alpha = torch.sigmoid(self.gate_mlp(gate_in))
        y = self.output_scale * (alpha * y_main + (1.0 - alpha) * y_pert)
        return y, alpha


class FourierDualNet(nn.Module):
    """Dual-branch traffic forecaster with specialised backbones.

    Main branch (periodic signal):
      - sees flow + ToD/DoW time features
      - larger dilation pattern (long receptive field)
    Pert branch (perturbation signal):
      - sees flow only (FFT high-freq residual)
      - smaller dilation pattern (short receptive field)

    Both produce (B, N, T_p) flow predictions, summed.
    """

    def __init__(self, num_nodes: int, supports: list[torch.Tensor],
                 T_h: int, T_p: int, K: int = 3,
                 decomp_mode: str = "fixed_k",
                 in_dim_flow: int = 3, nhid: int = 32, dropout: float = 0.3,
                 device: torch.device | None = None,
                 main_blocks: int = 2, main_layers: int = 4,
                 pert_blocks: int = 4, pert_layers: int = 2,
                 use_time_emb: bool = True,
                 use_cross_attn: bool = False,
                 cross_attn_dim: int = 16, cross_attn_heads: int = 2,
                 sensor_meta_dim: int | None = None):
        super().__init__()
        self.use_time_emb = use_time_emb
        self.use_cross_attn = use_cross_attn
        self.in_dim_flow = in_dim_flow
        self.T_h = T_h
        self.decomp = FourierDecomp(
            T_h=T_h, K=K, mode=decomp_mode,
            sensor_meta_dim=sensor_meta_dim,
        )
        self.requires_sensor_meta = (decomp_mode == "conditioned")

        if use_cross_attn:
            self.cross_attn = CrossBranchAttention(
                c_in=in_dim_flow, d_attn=cross_attn_dim,
                n_heads=cross_attn_heads, dropout=dropout,
            )

        # Main branch sees flow + ToD + DoW (5 channels if use_time_emb)
        main_in_dim = in_dim_flow + (2 if use_time_emb else 0)
        self.main_branch = gwnet(
            device=device, num_nodes=num_nodes, dropout=dropout,
            supports=supports, gcn_bool=True, addaptadj=True,
            in_dim=main_in_dim, out_dim=T_p,
            residual_channels=nhid, dilation_channels=nhid,
            skip_channels=nhid * 8, end_channels=nhid * 16,
            blocks=main_blocks, layers=main_layers,
        )

        # Pert branch sees flow only
        self.pert_branch = gwnet(
            device=device, num_nodes=num_nodes, dropout=dropout,
            supports=supports, gcn_bool=True, addaptadj=True,
            in_dim=in_dim_flow, out_dim=T_p,
            residual_channels=nhid, dilation_channels=nhid,
            skip_channels=nhid * 8, end_channels=nhid * 16,
            blocks=pert_blocks, layers=pert_layers,
        )

    @staticmethod
    def _to_gwnet(x: torch.Tensor) -> torch.Tensor:
        # (B, N, T, C) → (B, C, N, T)
        return x.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _from_gwnet(out: torch.Tensor) -> torch.Tensor:
        # (B, T_p, N, 1) → (B, N, T_p)
        return out.squeeze(-1).permute(0, 2, 1).contiguous()

    def forward(self, x_hist: torch.Tensor, time_feat: torch.Tensor | None = None,
                sensor_meta: torch.Tensor | None = None,
                return_components: bool = False):
        """
        x_hist:       (B, N, T_h, C_flow=3)
        time_feat:    (B, T_h, 2) — [tod/288, dow/7] per timestep.
                      Required when use_time_emb=True or decomp_mode='conditioned'.
        sensor_meta:  (N, C_meta) — static sensor features.
                      Required when decomp_mode='conditioned'.
        """
        # Pass time_feat / sensor_meta to decomp (used by conditioned mode, ignored otherwise)
        x_main_flow, x_pert_flow = self.decomp(
            x_hist, sensor_meta=sensor_meta, time_feat=time_feat,
        )

        if self.use_time_emb:
            assert time_feat is not None, "use_time_emb=True requires time_feat"
            B, N, T, _ = x_main_flow.shape
            tf = time_feat.unsqueeze(1).expand(B, N, T, 2)        # (B, N, T, 2)
            x_main_input = torch.cat([x_main_flow, tf], dim=-1)   # (B, N, T, C_flow+2)
        else:
            x_main_input = x_main_flow

        # Cross-branch attention: Pert queries Main, gets refined input
        if self.use_cross_attn:
            x_pert_input = self.cross_attn(x_pert_flow, x_main_flow)  # (B, N, T, C_flow)
        else:
            x_pert_input = x_pert_flow

        y_main = self._from_gwnet(self.main_branch(self._to_gwnet(x_main_input)))
        y_pert = self._from_gwnet(self.pert_branch(self._to_gwnet(x_pert_input)))
        y = y_main + y_pert

        if return_components:
            return y, {"y_main": y_main, "y_pert": y_pert,
                       "x_main": x_main_flow, "x_pert": x_pert_flow,
                       "mask": self.decomp.get_mask()}
        return y
