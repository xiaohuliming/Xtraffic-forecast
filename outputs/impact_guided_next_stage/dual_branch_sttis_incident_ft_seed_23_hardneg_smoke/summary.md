# ST-TIS Incident Branch Fine-Tune

This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.

## Test Result

- source all candidates robust MAE: `0.7113`
- source affected candidates robust MAE: `1.1182`
- incident-ft all candidates robust MAE: `0.7183`
- incident-ft affected candidates robust MAE: `1.0921`
- residual_beta: 1.00

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_23`
- train_gate: True
- epochs: 1
- lr: 0.0001
- affected_weight: 4.0
- incident_loss_weight: 0.0
- gate_loss_weight: 0.0
- normal_better_gate_loss_weight: 0.05
- normal_better_margin: 0.1
- normal_better_min_gate: 0.35
- trainable parameters: 319440

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |            256 |
| val     |     29210 |            256 |
| test    |     27499 |            256 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| val     |                            0.7131 |                                 1.1485 |                                   0.5351 |
| test    |                            0.7183 |                                 1.0921 |                                   0.5317 |

## Training

- best_epoch: 1
- best_val_loss: 0.6559
