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
| `spectral_energy_ratio = ||x_main||² / (||x_main||² + ||x_pert||²)` | (B, N) | post-FFT, no extra cost | midday (when low-freq energy drops, gate down-weights Main) |
| `node_emb` | (N, 8) | learnable embedding | long-tail nodes, Orange 44% (per-node identity) |
| `time_avg = time_feat.mean(dim=1)` | (B, 2) | ToD/DoW averaged over hist window | midday vs rush hour discrimination |

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
# (1) Gate MLP last layer: bias=0, weight scaled 0.01x → alpha_logits ≈ 0 → alpha ≈ 0.5
nn.init.zeros_(gate_mlp[-1].bias)
gate_mlp[-1].weight.data.mul_(0.01)

# (2) node_emb starts at zero (no per-node bias initially)
nn.init.zeros_(node_emb.weight)

# (3) Output scaled by learnable scalar (init=2.0) so y = 2 * 0.5 * (y_main + y_pert) = baseline
output_scale = nn.Parameter(torch.tensor(2.0))
y = output_scale * (alpha * y_main + (1 - alpha) * y_pert)
```

At ep0 this is **mathematically identical** to baseline FDN — guaranteeing we
don't regress relative to baseline by construction.

### Design decisions (all locked)

| Decision | Choice | Rationale |
|---|---|---|
| Gate granularity | per-(node, horizon), shape (B, N, 12) | Long horizons need different trust than short |
| Output scale | learnable scalar, init=2.0 | Allows model to break ensemble assumption if useful |
| Regularization | none initially; L1 push α→0.5 with λ=0.01 as fallback if α collapses | Don't over-constrain; observe first |
| Dropout in gate | none | Gate is tiny, not at risk of overfitting |

## Training plan

**Recipe**: from-scratch, same hyperparameters as baseline FDN (30 epochs, AdamW,
cosine schedule, batch_size from existing config). No warm-start.

**Rationale**: warm-start would give an unfair advantage and prevent the backbone
from co-adapting with the gate. From-scratch is the publishable comparison.

**Compute budget**: 5080 GPU via Tailscale, ~30 min/epoch × 30 epoch × 3 region ≈
**4.5 hours sequential** (nohup, run overnight).

### Stages

| Stage | Region | Config | Goal |
|---|---|---|---|
| **S1** | Alameda | D3 from scratch, 30 ep | First signal — is overall MAE down ≥ 0.15? |
| **S2** | Alameda | Run 4 sanity checks (no training) | Diagnose what the gate actually learned |
| **S3** | CC + Orange | D3 from scratch, 30 ep each | Generalization — 2/3 regions improve ≥ 0.15? |
| **S4** | Alameda | Ablation: drop one gate input at a time (3 runs) | Which input matters? |

### Decision gates

- After **S1**: if Alameda MAE drops < 0.15 → halt, escalate to fallback plan.
- After **S2**: if α distribution doesn't differentiate (std < 0.1, no node/time
  variation) → MAE drop is from backbone tuning, not gate → D3 abandoned.
- After **S3**: if only 1/3 region improves → D3 has region-specific value;
  document and move on without claiming general improvement.

## Sanity checks (post-training, S2)

Record `alpha` on the full test set, then:

1. **α has variance** — `alpha.mean() ≈ 0.5`, `alpha.std() > 0.1`. If not, gate didn't learn.
2. **α reflects diagnosed failure modes**:
   - `corr(per_node_test_MAE, alpha_per_node.mean())` should be *negative*
     (high-MAE long-tail nodes get low α).
   - `alpha[midday] < alpha[rush_hour]` should hold.
   - `corr(spectral_energy_ratio, alpha)` should be *positive*
     (high low-freq energy → trust Main more).
3. **α increases with horizon** — `alpha[..., -1].mean() > alpha[..., 0].mean()`.
   Pert branch's predictions decay at long horizon; Main should be trusted more.
4. **Targeted MAE reductions**: p99 node MAE, midday MAE, and Orange affected-long-horizon
   MAE should all improve. If only overall MAE drops but these don't → D3 took
   credit for backbone improvement, not its own contribution.

If 3 of 4 sanity checks pass → publishable contribution.

## Risks + fallback

| Risk | Probability | Mitigation |
|---|---|---|
| α collapses to 0 or 1 (polarization) | medium | Add L1 `λ * |α - 0.5|` with λ=0.01, retrain |
| Inconsistent across regions (like +A) | high | Sanity check 2 validates whether gate learned signal, even if MAE drop is small |
| Trained model worse than baseline | low (init=baseline) | Use `best.pt` by val loss, not last epoch |
| Orange node_emb fails to converge | medium | Bump `d_node` to 16 or 32 if needed |
| Compute overrun | low | nohup overnight on 5080 |

**Fallback sequence** if D3 fails on S1:
1. Add L1 regularization, retry S1 once.
2. Strip gate inputs to only `spectral_energy_ratio`, retry S1.
3. If still no signal → abandon D3, pivot to D1 (per-node K from autocorrelation statistics).

## Files to modify

- `fourier_dual_net/model.py` — add `GatedFusion` module + integrate into `FourierDualNet`.
  New constructor flag `use_gated_fusion: bool`.
- `scripts/train_fourier_dual_net.py` — add `--use_gated_fusion` CLI flag.
- `scripts/analyze_gate.py` — NEW. Loads test_predictions.npz + records alpha values
  during inference, computes the 4 sanity checks.

## Out of scope (saved for later)

- D1: per-node K with autocorrelation features (next candidate if D3 fails or is
  insufficient as a paper contribution).
- D2: learnable basis instead of FFT.
- D4: 3rd "unstructured" branch for long-tail nodes.
- D5: spectral-domain loss + long-horizon weighting.
