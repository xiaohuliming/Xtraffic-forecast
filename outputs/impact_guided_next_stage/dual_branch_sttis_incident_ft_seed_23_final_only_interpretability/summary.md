# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_only`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `1.00`

## Branch ablation

- Learned gate all-candidate MAE: `0.7082`; fixed 0.5 gate: `0.7255`.
- Learned gate affected-candidate MAE: `1.1052`; fixed 0.5 gate: `1.1379`.
- Affected branch-only MAE: normal-style `1.2133`, incident-graph `1.2648`.

## Gate behavior

- Mean gate on affected elements: `0.4269`.
- Mean gate on unaffected elements: `0.4352`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4491`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4078`.
- Correlation between gate and absolute target residual on all valid elements: `0.1844`.

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
| high    |      0.4257 |
| low     |      0.4382 |
| mid     |      0.4234 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4234 |
| mid     |      0.4240 |
| short   |      0.4422 |
