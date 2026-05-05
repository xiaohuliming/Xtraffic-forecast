# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_delta_gate_seed_23`
- split: `test`
- evaluated samples: `4096`
- residual_beta: `1.05`

## Branch ablation

- Learned gate all-candidate MAE: `0.7105`; fixed 0.5 gate: `0.7252`.
- Learned gate affected-candidate MAE: `1.1205`; fixed 0.5 gate: `1.1498`.
- Affected branch-only MAE: normal-style `1.2217`, incident-graph `1.2653`.

## Gate behavior

- Mean gate on affected elements: `0.4680`.
- Mean gate on unaffected elements: `0.4819`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4879`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4509`.
- Correlation between gate and absolute target residual on all valid elements: `0.1646`.

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
| high    |      0.4666 |
| low     |      0.4780 |
| mid     |      0.4655 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4636 |
| mid     |      0.4661 |
| short   |      0.4832 |
