# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_gate_full_no_aux`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `1.05`

## Branch ablation

- Learned gate all-candidate MAE: `0.7181`; fixed 0.5 gate: `0.7442`.
- Learned gate affected-candidate MAE: `1.1234`; fixed 0.5 gate: `1.1706`.
- Affected branch-only MAE: normal-style `1.2478`, incident-graph `1.3562`.

## Gate behavior

- Mean gate on affected elements: `0.3686`.
- Mean gate on unaffected elements: `0.3642`.
- On affected elements where the incident branch has lower local error, mean gate is `0.3921`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.3511`.
- Correlation between gate and absolute target residual on all valid elements: `0.2198`.

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
| high    |      0.3657 |
| low     |      0.3761 |
| mid     |      0.3693 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.3613 |
| mid     |      0.3837 |
| short   |      0.3641 |
