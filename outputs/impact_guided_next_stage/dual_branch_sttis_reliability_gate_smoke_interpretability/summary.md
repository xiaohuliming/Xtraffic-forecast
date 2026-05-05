# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_reliability_gate_smoke`
- split: `test`
- evaluated samples: `256`
- residual_beta: `1.00`

## Branch ablation

- Learned gate all-candidate MAE: `0.7210`; fixed 0.5 gate: `0.7337`.
- Learned gate affected-candidate MAE: `1.1356`; fixed 0.5 gate: `1.1640`.
- Affected branch-only MAE: normal-style `1.2372`, incident-graph `1.2546`.

## Gate behavior

- Mean gate on affected elements: `0.4547`.
- Mean gate on unaffected elements: `0.4652`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4734`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4382`.
- Correlation between gate and absolute target residual on all valid elements: `0.1784`.

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
| high    |      0.4617 |
| low     |      0.4377 |
| mid     |      0.4534 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4601 |
| mid     |      0.4454 |
| short   |      0.4534 |
