# ST-TIS Incident Branch Fine-Tune

This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.

## Test Result

- source all candidates robust MAE: `0.7113`
- source affected candidates robust MAE: `1.1182`
- incident-ft all candidates robust MAE: `0.7082`
- incident-ft affected candidates robust MAE: `1.1050`
- residual_beta: 1.00

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_23`
- train_gate: True
- epochs: 3
- lr: 0.0001
- affected_weight: 4.0
- incident_loss_weight: 0.0
- gate_loss_weight: 0.0
- normal_better_gate_loss_weight: 0.005
- normal_better_margin: 0.1
- normal_better_min_gate: 0.35
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
| val     |                            0.7149 |                                 1.1575 |                                   0.5305 |
| test    |                            0.7082 |                                 1.1050 |                                   0.5221 |

## Training

- best_epoch: 3
- best_val_loss: 0.6222
