# FDN++ Gated Fusion (D3) — Design Spec

**Date**: 2026-05-25
**Status**: Approved, ready for implementation plan
**Author**: brainstorming session with user

## Context

`FourierDualNet` baseline (learnable_K3) is current SOTA on the XTraffic pipeline:
- Beats GraphWaveNet (label-free baseline) on Alameda + ContraCosta
- Beats IGSTGNN (KDD'26 incident-aware SOTA) on Alameda + Orange under matched-window comparison

A prior FDN++ ablation on Alameda explored 3 architectural additions:
- **+A** (time_emb + asymmetric dilations 2×4/4×2): -0.07 MAE (noise)
- **+B** (cross-attention Pert←Main at input level): +0.28 MAE (regression)
- **+E** (input-conditioned spectral mask): +0.34 MAE (regression)

Result: no clear architectural innovation yet. Advisor explicitly asked for novelty
beyond "two GWNs running in parallel".

A diagnostic over baseline predictions identified 3 concrete failure modes:

1. **Long-tail nodes collapse** — p99 node MAE is 35-38, max 50-72.
   ~10% of sensors have MAE 3× the median.
2. **FDN loses to GraphWaveNet at midday** in ContraCosta (+0.24) and Orange (+0.35),
   despite winning at night / rush hour. Midday lacks strong periodicity, so the
   FFT-based "low-freq = main signal" assumption breaks.
3. **Orange-wide degradation** — 44% of nodes are *worse* under FDN than GWN
   (vs 18-19% for Alameda/CC). Large 990-node graph likely violates the global-mask
   assumption.

All three failure modes share one root cause: **unconditional additive fusion
(`y = y_main + y_pert`) cannot suppress the Main branch when its low-frequency
assumption breaks for a particular (node, time) combination.**

## Goal

Replace the additive fusion `y = y_main + y_pert` with a **learned per-(node, horizon)
gate** that decides how much to trust each branch. The gate's input includes signals
that directly correspond to the diagnosed failure modes.

## Non-goals

- No changes to the FFT decomposition (keep `learnable_K3` mode).
- No new branches (still 2-branch Main/Pert).
- No new training data / no incident labels (label-free constraint).
- No backbone architecture change (still GraphWaveNet).

## Architecture

### Data flow

```
x_hist (B,N,12,3) ─┬─→ FFT_mask(K=3, learnable) ─→ x_main ─→ GWN_main ─→ y_main ──┐
                   ├─→ (1-mask)                  ─→ x_pert ─→ GWN_pert ─→ y_pert ──┤
                   ├─→ spectral_energy_ratio (B,N) ─┐                              │
node_emb (N, 8) ─────────────────────────────────────┼─→ Gate MLP ─→ α (B,N,12) ───┤
time_feat (B,12,2) ─→ time_avg (B, 2) ─────────────┘                              │
                                                                                   ▼
                                              y = 2 * (α * y_main + (1-α) * y_pert)
                                                  (the 2× preserves additive baseline scale)
```

### Gate inputs

Three signals, each motivated by one failure mode:

| Input | Shape | Source | Targets failure mode |
|---|---|---|---|
| `spectral_energy_ratio = ‖x_main_flow‖² / (‖x_main_flow‖² + ‖x_pert_flow‖²)` | (B, N) | post-FFT, **flow channel only (index 0)** | midday (when low-freq energy drops, gate down-weights Main) |
| `node_emb` | (N, 8) | learnable embedding | long-tail nodes, Orange 44% (per-node identity) |
| `time_avg = time_feat.mean(dim=1)` | (B, 2) | ToD/DoW averaged over hist window | midday vs rush hour discrimination |

> **Energy ratio uses the flow channel only.** `x_main`/`x_pert` carry 3 channels (flow, occupancy, speed). Averaging squared signal over all 3 dilutes the periodicity cue the gate needs — use `x_main[..., 0:1]`. The input is z-scored upstream (same tensor the backbones consume), so the ratio is scale-consistent.
>
> **`time_avg` is NOT redundant.** The primary D3 run trains **without** `--use_time_emb` (see §Training plan — confound control), so the Main branch never sees time; the gate's `time_avg` is the only time-of-day signal in the model. Redundancy would arise only if `--use_time_emb` were also on, which the primary experiment deliberately avoids.

Concatenated to `(B, N, 11)`, fed to a small MLP:

```python
gate_mlp = nn.Sequential(
    nn.Linear(11, 32),
    nn.ReLU(),
    nn.Linear(32, T_p=12),   # one logit per horizon
)
alpha = torch.sigmoid(gate_mlp(gate_in))   # (B, N, 12)
```

Parameter cost: `11*32 + 32*12 + 8*N ≈ 760 + 8N` (negligible vs ~100k backbone params).

### Initialization (critical — must preserve baseline at ep0)

```python
# (1) Gate MLP last layer: bias=0 AND weight=0 → alpha_logits == 0 exactly → alpha == 0.5 exactly
nn.init.zeros_(gate_mlp[-1].bias)
nn.init.zeros_(gate_mlp[-1].weight)        # EXACT, not 0.01x — see note

# (2) node_emb starts at zero (no per-node bias initially)
nn.init.zeros_(node_emb.weight)

# (3) Output scaled by learnable scalar (init=2.0) so y = 2 * 0.5 * (y_main + y_pert) = baseline
output_scale = nn.Parameter(torch.tensor(2.0))
y = output_scale * (alpha * y_main + (1 - alpha) * y_pert)
```

At ep0 this is **bit-for-bit identical** (deviation < 1e-6) to baseline FDN — guaranteeing
no regression vs baseline by construction.

> **Why `zero_()` and not `mul_(0.01)`** (verified numerically in review): with `mul_(0.01)` the
> last-layer weights stay small-but-nonzero, so `alpha_logits ≈ ±3e-3`, `|α−0.5| ≈ 7e-4`, and the
> output deviates `≈ 2·(α−0.5)·(y_main−y_pert) ≈ 3e-3` — which **fails** the `atol=1e-4` baseline
> unit test. With `zero_()`, `alpha_logits == 0` for every input → `α == 0.5` exactly. Gradient still
> flows: `∂loss/∂W_last` equals the (nonzero) ReLU activations, so `W_last` leaves zero on step 1;
> `W_first` + `node_emb` get gradient from step 2 (one-step delay, negligible over 30 epochs).

> **Trivial-rescale guard** (review finding): a learnable `output_scale` could absorb a global gain and
> masquerade as a gate win. Attribution control (S4): freeze gate at α=0.5, train only `output_scale`
> + backbones. If that alone captures most of the gain, the win is rescaling, not gating. Always log
> the final `output_scale`.

> **L1 fallback hook** (review finding): the fallback regularizer `λ·mean|α−0.5|` needs the loss to
> see `α`. `FourierDualNet.forward` caches the live (grad-carrying) tensor as `self._last_alpha`
> (`None` when gating is off) so the train loop can add the penalty without changing the forward signature.

### Design decisions (all locked)

| Decision | Choice | Rationale |
|---|---|---|
| Gate granularity | per-(node, horizon), shape (B, N, 12) | Long horizons need different trust than short |
| Output scale | learnable scalar, init=2.0 | Allows model to break ensemble assumption if useful |
| Regularization | none initially; L1 push α→0.5 with λ=0.01 as fallback if α collapses | Don't over-constrain; observe first |
| Dropout in gate | none | Gate is tiny, not at risk of overfitting |

## Training plan

**Recipe**: from-scratch, same hyperparameters as baseline FDN (30 epochs, Adam,
cosine schedule, batch_size=48, lr=1e-3, wd=1e-4 — the real train-script defaults). No warm-start.

**Rationale**: warm-start would give an unfair advantage and prevent the backbone
from co-adapting with the gate. From-scratch is the publishable comparison.

**Confound control (verified review finding)**: the baseline FDN `learnable_K3` (Alameda 11.98) was
trained **without** `--use_time_emb`. The primary D3 run MUST therefore also be **without**
`--use_time_emb`, so the only difference vs baseline is the gate. (The gate keeps its own internal
`time_avg` input — that is distinct from the Main-branch time embedding.) Studying time embedding is a
*separate* ablation row: `D3 + --use_time_emb` compared against `+A` (which has time_emb), never against
the no-time_emb baseline.

**Seed robustness (verified review finding)**: a single seed cannot distinguish a 0.15 MAE move from
noise — the prior `+A` run's −0.07 was dismissed as noise with no measured band. Before trusting the
S1 gate, establish the band: re-run the **baseline** on ≥2 extra seeds (cheap, reuses config) and/or
run D3 on ≥3 seeds; report mean ± std. The init==baseline property reduces but does not eliminate seed
variance (from-scratch still randomizes backbone init + data order).

**Compute budget**: 5080 GPU via Tailscale, ~30 min/epoch × 30 epoch × 3 region ≈
**4.5 hours sequential** per seed (nohup, run overnight).

### Stages

| Stage | Region | Config | Goal |
|---|---|---|---|
| **S1** | Alameda | D3 from scratch, 30 ep | First signal — is overall MAE down ≥ 0.15? |
| **S2** | Alameda | Run 4 sanity checks (no training) | Diagnose what the gate actually learned |
| **S3** | CC + Orange | D3 from scratch, 30 ep each | Generalization — 2/3 regions improve ≥ 0.15? |
| **S4** | Alameda | Ablation: (a) drop one gate input at a time; (b) **rescale control** — gate frozen α=0.5, only `output_scale`+backbones train; (c) `output_scale` fixed at 2.0, gate active | Which input matters? Is the gain real gating or a global rescale? |

### Decision gates

- After **S1**: if Alameda MAE drops < 0.15 → halt, escalate to fallback plan.
- After **S2**: if α distribution doesn't differentiate (std < 0.1, no node/time
  variation) → MAE drop is from backbone tuning, not gate → D3 abandoned.
- After **S3**: if only 1/3 region improves → D3 has region-specific value;
  document and move on without claiming general improvement.

## Sanity checks (post-training, S2)

Record `alpha` on the full test set, then (thresholds tightened per review — the original
std>0.05 / |corr|>0.05 could pass by chance). Treat these as **descriptive diagnostics**, and
require **cross-region sign-consistency** for any directional claim:

1. **α has variance** — `alpha.mean() ≈ 0.5`, `alpha.std() > 0.1`. If not, gate didn't learn.
2. **α reflects diagnosed failure modes** (must hold with the same sign in all 3 regions to be claimed):
   - `corr(per_node_test_MAE, alpha_per_node.mean()) < −0.2` (high-MAE long-tail nodes get low α).
   - `alpha[midday] < alpha[rush_hour]`.
   - `corr(spectral_energy_ratio, alpha) > 0.2` (high low-freq energy → trust Main more).
3. **α increases with horizon** — `alpha[..., -1].mean() > alpha[..., 0].mean()`.
   Pert branch's predictions decay at long horizon; Main should be trusted more.
4. **Targeted MAE reductions**: p99 node MAE, midday MAE, and Orange affected-long-horizon
   MAE should all improve. If only overall MAE drops but these don't → D3 took
   credit for backbone improvement, not its own contribution.

If 3 of 4 sanity checks pass (with cross-region sign-consistency on check 2) → publishable contribution.

## Risks + fallback

| Risk | Probability | Mitigation |
|---|---|---|
| α collapses to 0 or 1 (polarization) | medium | Add L1 `λ * |α - 0.5|` with λ=0.01, retrain |
| Inconsistent across regions (like +A) | high | Sanity check 2 validates whether gate learned signal, even if MAE drop is small |
| Trained model worse than baseline | low (init=baseline) | Use `ckpt_best.pt` by val loss, not last epoch |
| Orange node_emb fails to converge | medium | Bump `d_node` to 16 or 32 if needed |
| Compute overrun | low | nohup overnight on 5080 |

**Fallback sequence** if D3 fails on S1:
1. Add L1 regularization, retry S1 once.
2. Strip gate inputs to only `spectral_energy_ratio`, retry S1.
3. If still no signal → abandon D3, pivot to D1 (per-node K from autocorrelation statistics).

## Files to modify

- `fourier_dual_net/model.py` — add `GatedFusion` module + integrate into `FourierDualNet`.
  New constructor flag `use_gated_fusion: bool`.
- `scripts/train_fourier_dual_net.py` — add `--use_gated_fusion` (+ `--gate_d_node`, `--gate_hidden`,
  `--gate_l1_lambda`) flags; fix `need_time` in **all three** loops (evaluate / train / test); add the L1 term.
- `scripts/analyze_gate.py` — NEW. Rebuilds the model **from the checkpoint's saved `config`/dims**
  (`ckpt_best.pt`, key `model_state`), re-runs test inference with `return_components=True` to record
  `alpha`, computes the sanity checks. Must use the real data API: `MultiRegionDataset`; attribute access
  `rdata.N/.T_h/.T_p/.edge_index`; the module constant `NUM_TOD_SLOTS` (=288) for time-of-day bucketing
  (there is no `slot_per_day` attribute); a **copied** `build_adj_supports` (row-normalized random-walk,
  **two** supports A_fwd + A_bwd — matching the train script exactly); real batch keys
  `batch["y_true"][...,0]` / `batch["y_mask"][...,0]` (NOT the npz-only `actual_future_flow`/`y_mask_flow`).
  Default `--data_dir data/processed`.

## Out of scope (saved for later)

- D1: per-node K with autocorrelation features (next candidate if D3 fails or is
  insufficient as a paper contribution).
- D2: learnable basis instead of FFT.
- D4: 3rd "unstructured" branch for long-tail nodes.
- D5: spectral-domain loss + long-horizon weighting.
