# ST-TIS Incident Branch Fine-Tune

This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.

## Test Result

- source all candidates robust MAE: `0.7082`
- source affected candidates robust MAE: `1.1052`
- incident-ft all candidates robust MAE: `0.7012`
- incident-ft affected candidates robust MAE: `1.0724`
- residual_beta: 1.00

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_only`
- train_gate: True
- epochs: 1
- lr: 0.0001
- affected_weight: 4.0
- incident_loss_weight: 0.0
- gate_loss_weight: 0.0
- normal_better_gate_loss_weight: 0.0
- normal_better_margin: 0.1
- normal_better_min_gate: 0.0
- convex_gate_loss_weight: 0.05
- convex_gate_min_gap: 0.05
- trainable parameters: 319440

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |            512 |
| val     |     29210 |            512 |
| test    |     27499 |            512 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| val     |                            0.7209 |                                 1.1852 |                                   0.5299 |
| test    |                            0.7012 |                                 1.0724 |                                   0.5252 |

## Training

- best_epoch: 1
- best_val_loss: 0.6451
