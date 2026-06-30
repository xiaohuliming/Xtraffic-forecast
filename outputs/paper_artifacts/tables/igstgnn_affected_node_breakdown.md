# IGSTGNN on Our Affected/Unaffected Node Partition

This breakdown takes IGSTGNN's per-county test predictions (raw flow units) and
splits them by the user's `node_affected` definition (per-incident binary label
in `outputs/impact_labels/<region>/node_labels.csv`). It exposes the gap that
IGSTGNN's overall MAE conceals.

## Headline (average over horizons 1-12)

| Region | Test samples<br>(matched) | MAE all | MAE affected | MAE unaffected | Gap (aff − unaff) |
|---|---:|---:|---:|---:|---:|
| Alameda | 4,282 / 4,512 | 12.87 | **17.09** | 12.83 | **+4.26 (+33.2%)** |
| Contra Costa | 2,701 / 3,096 | 13.36 | **17.33** | 13.31 | **+4.02 (+30.2%)** |
| Orange | 6,007 / 6,344 | 13.55 | **18.63** | 13.52 | **+5.11 (+37.8%)** |

Affected nodes per incident (mean): Alameda 4.4, Contra Costa 5.2, Orange 5.0 —
roughly 1% of county nodes per sample, yet they drive a 30-38% MAE penalty that
the overall average completely smooths over.

## Per-horizon affected MAE growth

### Alameda

| H | All | Affected | Unaffected | Aff/Unaff |
|---|---:|---:|---:|---:|
| 1 | 10.70 | 13.63 | 10.67 | +28% |
| 3 | 11.72 | 15.49 | 11.68 | +33% |
| 6 | 12.78 | 16.84 | 12.74 | +32% |
| 9 | 13.74 | 18.58 | 13.69 | +36% |
| 12 | 14.50 | **19.71** | 14.45 | **+36%** |

### Contra Costa

| H | All | Affected | Unaffected | Aff/Unaff |
|---|---:|---:|---:|---:|
| 1 | 11.25 | 14.00 | 11.22 | +25% |
| 3 | 12.34 | 15.96 | 12.29 | +30% |
| 6 | 13.38 | 17.51 | 13.33 | +31% |
| 9 | 14.12 | 18.20 | 14.07 | +29% |
| 12 | 14.95 | **19.78** | 14.88 | **+33%** |

### Orange

| H | All | Affected | Unaffected | Aff/Unaff |
|---|---:|---:|---:|---:|
| 1 | 11.45 | 14.71 | 11.43 | +29% |
| 3 | 12.52 | 16.70 | 12.49 | +34% |
| 6 | 13.58 | 18.75 | 13.55 | +38% |
| 9 | 14.29 | 19.96 | 14.25 | +40% |
| 12 | 15.16 | **21.30** | 15.12 | **+41%** |

## Observations for the paper

1. **IGSTGNN's reported MAE is dominated by unaffected sensors.** On Alameda
   they report 12.69 average MAE; that figure essentially equals the
   unaffected-only MAE (12.83). The 4.4 affected sensors per incident drag the
   average up by less than 0.1 because they're outnumbered ~120-to-1 by
   unaffected sensors.

2. **Affected-node MAE degrades much faster with horizon.** All-node MAE grows
   ~35% from H1 to H12 (10.7 → 14.5 on Alameda). Affected-node MAE grows ~45%
   (13.6 → 19.7). At H12 the affected MAE is 36% above the unaffected MAE.
   This is the regime where incident-aware modeling matters most.

3. **All three regions show the same pattern.** Gap is 30-38% (Orange worst at
   38%, Alameda 33%, Contra Costa 30%). Orange's H12 affected MAE reaches
   21.30 vs 15.12 unaffected — a 41% degradation. Consistent across regions
   means this is a structural property of overall-MAE evaluation under sparse
   incident impact, not a region-specific artifact.

4. **Implication for our framework.** Our paper evaluates explicitly on the
   affected candidate subset (residual MAE 1.12 affected vs 0.71 all). That
   choice is now externally validated: even SOTA IGSTGNN trained on the same
   data shows a 30%+ overall→affected gap. Reporting only overall MAE — as
   most published baselines do — hides where the model actually fails during
   incidents.

## Reproducibility

- IGSTGNN test predictions: `baselines/IGSTGNN/experiments/igstgnn/<region>_23/test_predictions.npz`
- Affected labels: `outputs/impact_labels/<region>/node_labels.csv` (`incident_id × region_node_idx → affected ∈ {0,1}`)
- Script: `scripts/evaluate_igstgnn_affected_breakdown.py` (matches IGSTGNN test sample's `_t_idx` to `event_labels.csv start_idx`, then joins through `incident_id` to node-level affected mask)
- Per-region JSON outputs: `baselines/IGSTGNN/experiments/igstgnn/<region>_23/affected_breakdown.json`
