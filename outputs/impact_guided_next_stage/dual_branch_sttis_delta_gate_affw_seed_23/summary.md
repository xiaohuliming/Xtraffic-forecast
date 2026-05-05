# ST-TIS Bidirectional Gate Delta

This variant freezes the proposal-aware model and trains only a bounded bidirectional gate-delta adapter.

## Test Result

- source all candidates robust MAE: `0.7113`
- source affected candidates robust MAE: `1.1182`
- delta-gate all candidates robust MAE: `0.7113`
- delta-gate affected candidates robust MAE: `1.1169`
- delta_scale: 1.00
- residual_beta: 1.00

## Selection

- selection_metric: `affected_candidates_model_robust_mae`
- all_val_tolerance: `0.002`
- sweep_scales: `0.0,0.25,0.5,0.75,1.0,1.25,1.5`
- sweep_betas: `0.95,1.0,1.05,1.1`

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_23`
- epochs: 3
- lr: 0.0002
- gate_loss_weight: 0.2
- hard_residual_weight: 0.08
- delta_l2_weight: 0.0005
- hard_margin: 0.1
- affected_weight: 4.0
- delta_max: 2.0
- trainable delta parameters: 49740

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |         138528 |
| val     |     29210 |          29210 |
| test    |     27499 |          27499 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| train   |                            0.6577 |                                 1.0081 |                                   0.5142 |
| val     |                            0.7167 |                                 1.1657 |                                   0.5296 |
| test    |                            0.7113 |                                 1.1169 |                                   0.5210 |

## Training

- best_epoch: 3
- best_val_loss: 0.8405
