# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_convexgate`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `0.95`

## Branch ablation

- Learned gate all-candidate MAE: `0.7079`; fixed 0.5 gate: `0.7245`.
- Learned gate affected-candidate MAE: `1.1050`; fixed 0.5 gate: `1.1363`.
- Affected branch-only MAE: normal-style `1.2150`, incident-graph `1.2524`.

## Gate behavior

- Mean gate on affected elements: `0.4256`.
- Mean gate on unaffected elements: `0.4341`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4472`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4068`.
- Correlation between gate and absolute target residual on all valid elements: `0.1870`.

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
| high    |      0.4243 |
| low     |      0.4359 |
| mid     |      0.4229 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4225 |
| mid     |      0.4226 |
| short   |      0.4402 |
