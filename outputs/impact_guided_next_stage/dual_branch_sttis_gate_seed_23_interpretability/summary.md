# Dual-Branch Gate Interpretability

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux_seed_23`
- split: `test`
- evaluated samples: `27499`
- residual_beta: `1.00`

## Branch ablation

- Learned gate all-candidate MAE: `0.7135`; fixed 0.5 gate: `0.7244`.
- Learned gate affected-candidate MAE: `1.1217`; fixed 0.5 gate: `1.1439`.
- Affected branch-only MAE: normal-style `1.2133`, incident-graph `1.2443`.

## Gate behavior

- Mean gate on affected elements: `0.4327`.
- Mean gate on unaffected elements: `0.4322`.
- On affected elements where the incident branch has lower local error, mean gate is `0.4502`.
- On affected elements where the normal-style branch has lower local error, mean gate is `0.4172`.
- Correlation between gate and absolute target residual on all valid elements: `0.1998`.

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
| high    |      0.4324 |
| low     |      0.4420 |
| mid     |      0.4289 |

Recovery:
| group   |   gate_mean |
|:--------|------------:|
| long    |      0.4294 |
| mid     |      0.4299 |
| short   |      0.4474 |
