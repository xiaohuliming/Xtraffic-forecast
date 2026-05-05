# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `1.10`

## Branch ablation

- Learned gate all-candidate MAE: `0.7093`; fixed 0.5 gate: `0.7177`.
- Learned gate affected-candidate MAE: `1.1066`; fixed 0.5 gate: `1.1264`.
- Affected branch-only MAE: normal-style `1.2108`, incident-graph `1.1389`.

## Gate behavior

- Mean gate on affected elements: `0.5655`.
- Mean gate on unaffected elements: `0.5623`.
- On affected elements where the incident branch has lower local error, mean gate is `0.5764`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.5537`.
- Correlation between gate and absolute target residual on all valid elements: `0.2519`.

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
| high    |      0.5678 |
| low     |      0.5658 |
| mid     |      0.5620 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.5665 |
| mid     |      0.5619 |
| short   |      0.5689 |
