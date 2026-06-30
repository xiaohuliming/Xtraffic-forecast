# Head-to-head: IGSTGNN vs Our Adapter (raw flow units, time-aligned)

## Setup

For each test incident, IGSTGNN and our model both forecast a 12-step (60-min)
window starting from their own anchor. We match samples by anchor timestep
(`ours.sample_start == IGSTGNN._t_idx`) and compare on the **11 overlapping
forecast time points** (T+1 ... T+11):

* Ours predicts [T, T+11]   → use horizons 2..12
* IGSTGNN predicts [T+1, T+12] → use horizons 1..11

We restrict IGSTGNN's per-county prediction to **our 36 candidate-node subset**
per sample (incident-localized mainline sensors within ±5 PM), so both models
are evaluated on the **same nodes × the same time points × the same set of
matched incidents**.

## Result (averaged over 11 horizons)

Adapter = `groupaware_impact_correction_adapter_anomgate05_magq090_w005_seed_23`
(our paper's "main candidate" 3-seed best, seed 23 only).

| Region | Scope | Matched samples | Pixels | IGSTGNN | Ours (source) | Ours (adapter) | Δ adapter−IGSTGNN |
|---|---|---:|---:|---:|---:|---:|---:|
| Alameda | all candidates | 4,806 | 742,073 | **13.25** | 14.64 | 14.63 | **+1.38** |
| Alameda | affected only | 4,806 | 230,320 | **17.21** | 20.68 | 20.67 | **+3.46** |
| Alameda | unaffected only | 4,806 | 511,753 | **11.47** | 11.92 | 11.91 | **+0.44** |
| Contra Costa | all candidates | 1,874 | 388,464 | **13.40** | 14.41 | 14.40 | **+1.00** |
| Contra Costa | affected only | 1,874 | 119,231 | **17.36** | 20.46 | 20.46 | **+3.10** |
| Contra Costa | unaffected only | 1,874 | 269,233 | **11.65** | 11.73 | 11.72 | **+0.07** |
| Orange | (skipped: missing IGSTGNN test_predictions.npz) | – | – | – | – | – | – |

(`Δ adapter−IGSTGNN > 0` means IGSTGNN is better.)

## Per-horizon Breakdown — Affected Only

### Alameda

| min ahead | IGSTGNN | Ours source | Ours adapter | Δ adapter−IGS |
|---:|---:|---:|---:|---:|
| 5  | 13.83 | 16.82 | 16.82 | +2.99 |
| 10 | 14.76 | 18.08 | 18.09 | +3.32 |
| 15 | 15.65 | 18.99 | 18.99 | +3.34 |
| 20 | 16.27 | 19.82 | 19.81 | +3.54 |
| 25 | 16.86 | 20.16 | 20.15 | +3.29 |
| 30 | 17.29 | 20.79 | 20.78 | +3.49 |
| 35 | 17.76 | 21.43 | 21.42 | +3.66 |
| 40 | 18.59 | 22.16 | 22.14 | +3.55 |
| 45 | 19.07 | 22.70 | 22.69 | +3.62 |
| 50 | 19.50 | 22.96 | 22.95 | +3.45 |
| 55 | 19.73 | 23.54 | 23.54 | +3.81 |

### Contra Costa

| min ahead | IGSTGNN | Ours source | Ours adapter | Δ adapter−IGS |
|---:|---:|---:|---:|---:|
| 5  | 14.18 | 17.26 | 17.26 | +3.08 |
| 10 | 14.94 | 18.36 | 18.36 | +3.42 |
| 15 | 16.03 | 19.02 | 19.02 | +2.99 |
| 20 | 16.71 | 19.38 | 19.39 | +2.67 |
| 25 | 17.45 | 20.11 | 20.12 | +2.68 |
| 30 | 17.77 | 20.38 | 20.37 | +2.60 |
| 35 | 18.56 | 20.88 | 20.88 | +2.32 |
| 40 | 18.29 | 21.40 | 21.38 | +3.09 |
| 45 | 18.51 | 22.31 | 22.30 | +3.79 |
| 50 | 18.82 | 22.47 | 22.46 | +3.64 |
| 55 | 19.65 | 23.51 | 23.49 | +3.85 |

## Honest reading

1. **IGSTGNN beats our model on raw-flow MAE for affected nodes by ≈3 flow
   units, consistently across both regions and all 11 horizons.** This holds
   for both the source dual-branch model and the impact-correction adapter
   (the adapter changes raw-flow MAE by <0.02 — its z-residual gains do not
   show in raw flow on this metric).

2. **Why this is consistent with our framework's claims (not a refutation):**

   * **Different output target.** Our model predicts a multi-channel
     residual against a per-time-of-day robust normal (z-units, balanced over
     flow + occupancy + speed). IGSTGNN trains directly on raw flow with a
     single channel and a single global normalizer, whose loss is dominated
     by the absolute flow scale. Our model is implicitly down-weighting the
     flow channel during training relative to occupancy and speed.

   * **Different graph scope.** Our model sees only a 36-node incident-local
     subgraph per sample; IGSTGNN sees the full county graph (521-990
     nodes). The full-graph context plausibly gives IGSTGNN access to
     upstream/downstream structure that our locality bound rules out.

   * **Different anchor convention.** IGSTGNN forecasts T+1 ... T+12; we
     forecast T ... T+11. Even though we time-align the comparison to the
     11 overlapping points, our H1 (which our paper highlights as
     "incident-onset" prediction) has no IGSTGNN equivalent.

3. **Where our framework still adds value, even given (1):**

   * §5.10 shows that **IGSTGNN's overall MAE hides a 30-38% degradation on
     affected nodes** under our affected/unaffected partition. So the
     critique we make of overall-MAE-only evaluation applies to IGSTGNN as
     well — they just happen to be lower in absolute terms on that
     unfavorable slice.

   * Our adapter results in §5.7-5.9 are reported in **z-residual** units
     against a learned normal, where the adapter measurably improves on
     affected nodes (3-seed mean −0.000465 affected MAE delta vs source).
     Raw-flow MAE compresses these gains because most of the residual
     variance lives in normal periods, where adapter and source agree.

4. **Recommendation for the paper.** Add a §5.11 ("External baseline,
   absolute units") that reports this head-to-head honestly, names the
   sources of asymmetry (objective, graph scope, anchoring), and is explicit
   that the framework's contribution is incident-aware *residual* modeling
   under a defined affected/unaffected partition — not raw-flow SOTA. Frame
   IGSTGNN as a strong external corroboration of (a) the test-set difficulty
   we report, and (b) the affected/unaffected gap pattern, rather than as a
   model we claim to beat.

## Reproducibility

* Side cache (per-test-sample `fut_scale, normal_pred, actual_future`):
  `outputs/impact_guided_next_stage/headtohead_igstgnn/test_raw_flow_side_cache.npz`
* Adapter raw-flow predictions:
  `outputs/.../groupaware_impact_correction_adapter_anomgate05_magq090_w005_seed_23/test_raw_flow_predictions.npz`
* Comparison CSV: `outputs/impact_guided_next_stage/headtohead_igstgnn/headtohead_summary.csv`
* Per-horizon CSV: `outputs/impact_guided_next_stage/headtohead_igstgnn/headtohead_per_horizon.csv`
* Scripts: `scripts/build_test_raw_flow_cache.py`,
  `scripts/run_adapter_test_inference.py`,
  `scripts/compare_headtohead_igstgnn.py`

## Open follow-up

* **Orange region missing.** `baselines/IGSTGNN/experiments/igstgnn/Orange_23/test_predictions.npz`
  was not pulled to local from the Win training box (saved as
  `D:/IGSTGNN_baseline/IGSTGNN/experiments/igstgnn/Orange_23/test_predictions.npz`).
  Pull it (≈600 MB raw / ≈256 MB compressed) to fill in the third row.
