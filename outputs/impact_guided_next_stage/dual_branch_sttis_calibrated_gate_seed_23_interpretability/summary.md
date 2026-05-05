# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_calibrated_gate_no_aux_seed_23`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `0.90`

## Branch ablation

- Learned gate all-candidate MAE: `0.7446`; fixed 0.5 gate: `0.7500`.
- Learned gate affected-candidate MAE: `1.1616`; fixed 0.5 gate: `1.1719`.
- Affected branch-only MAE: normal-style `1.2052`, incident-graph `1.2324`.

## Gate behavior

- Mean gate on affected elements: `0.4482`.
- Mean gate on unaffected elements: `0.4596`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4595`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4381`.
- Correlation between gate and absolute target residual on all valid elements: `0.0772`.

## Files

- `branch_ablation_metrics.csv`
- `branch_ablation_by_horizon.csv`
- `gate_summary.csv`
- `gate_by_horizon.csv`
- `gate_selection_alignment.csv`
- `gate_correlations.csv`
- `event_group_gate_metrics.csv`
- `branch_ablation_mae.png`
- `gate_by_horizon.png`
- `gate_selection_alignment.png`
- `gate_by_event_group.png`

## Event group gate means

Severity:
| group   |   gate_mean |
|:--------|------------:|
| high    |      0.4444 |
| low     |      0.4621 |
| mid     |      0.4473 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4428 |
| mid     |      0.4498 |
| short   |      0.4613 |
