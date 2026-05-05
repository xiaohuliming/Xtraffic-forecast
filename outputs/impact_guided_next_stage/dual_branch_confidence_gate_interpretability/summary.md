# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_confidence_gate_no_aux`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `1.05`

## Branch ablation

- Learned gate all-candidate MAE: `0.7186`; fixed 0.5 gate: `0.7325`.
- Learned gate affected-candidate MAE: `1.1254`; fixed 0.5 gate: `1.1587`.
- Affected branch-only MAE: normal-style `1.2476`, incident-graph `1.1723`.

## Gate behavior

- Mean gate on affected elements: `0.5034`.
- Mean gate on unaffected elements: `0.4830`.
- On affected elements where the incident branch has lower local error, mean gate is `0.5200`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4865`.
- Correlation between gate and absolute target residual on all valid elements: `0.1967`.

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
| high    |      0.5101 |
| low     |      0.4893 |
| mid     |      0.5001 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.5086 |
| mid     |      0.5060 |
| short   |      0.4834 |
