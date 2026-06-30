# Table 6. Gate and branch interpretability on the test split

## Branch Ablation

| Fusion / branch setting | All MAE | Affected MAE | Unaffected MAE |
|---|---:|---:|---:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.8057 | 1.2478 | 0.5983 |
| Incident-graph residual only | 0.9012 | 1.3562 | 0.6876 |
| Fixed gate = 0.5 | 0.7442 | 1.1706 | 0.5442 |
| Learned gate | **0.7181** | **1.1234** | **0.5279** |

## Gate Selection Alignment

| Subset | Local branch condition | Mean incident-branch gate |
|---|---|---:|
| All | incident branch has lower local error | 0.3821 |
| All | normal-style branch has lower local error | 0.3528 |
| Affected | incident branch has lower local error | 0.3921 |
| Affected | normal-style branch has lower local error | 0.3511 |
| Unaffected | incident branch has lower local error | 0.3777 |
| Unaffected | normal-style branch has lower local error | 0.3535 |

## Gate Correlation With Residual Magnitude

| Subset | Gate vs. absolute target residual | Gate vs. absolute normal_delta |
|---|---:|---:|
| All | 0.2198 | 0.2150 |
| Affected | 0.2985 | 0.2676 |
| Unaffected | 0.0943 | 0.1465 |

Note: The learned gate is evaluated without retraining, using the same checkpoint and residual scaling coefficient as the main no-aux dual-branch model. The gate value is the incident-branch weight, so larger values indicate stronger reliance on incident-graph residual correction.
