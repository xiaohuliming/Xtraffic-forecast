# ST-TIS Incident Branch Fine-Tune

This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.

## Test Result

- source all candidates robust MAE: `0.7147`
- source affected candidates robust MAE: `1.1233`
- incident-ft all candidates robust MAE: `0.7119`
- incident-ft affected candidates robust MAE: `1.1121`
- residual_beta: 1.10

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_7`
- train_gate: True
- epochs: 3
- lr: 0.0001
- affected_weight: 4.0
- incident_loss_weight: 0.35
- gate_loss_weight: 0.05
- trainable parameters: 319440

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |         138528 |
| val     |     29210 |          29210 |
| test    |     27499 |          27499 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| train   |                            0.6600 |                                 1.0064 |                                   0.5180 |
| val     |                            0.7184 |                                 1.1628 |                                   0.5331 |
| test    |                            0.7119 |                                 1.1121 |                                   0.5241 |

## Training

- best_epoch: 3
- best_val_loss: 0.9724
