# ST-TIS Bidirectional Gate Delta

This variant freezes the proposal-aware model and trains only a bounded bidirectional gate-delta adapter.

## Test Result

- source all candidates robust MAE: `0.7093`
- source affected candidates robust MAE: `1.1066`
- delta-gate all candidates robust MAE: `0.7149`
- delta-gate affected candidates robust MAE: `1.0791`
- delta_scale: 0.00
- residual_beta: 1.05

## Selection

- selection_metric: `affected_candidates_model_robust_mae`
- all_val_tolerance: `0.002`
- sweep_scales: `0.0,0.5,1.0`
- sweep_betas: `1.0,1.05,1.1`

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23`
- epochs: 1
- lr: 0.0002
- gate_loss_weight: 0.1
- hard_residual_weight: 0.05
- delta_l2_weight: 0.001
- hard_margin: 0.1
- affected_weight: 3.0
- delta_max: 2.0
- trainable delta parameters: 49740

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |            256 |
| val     |     29210 |            256 |
| test    |     27499 |            256 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| train   |                            0.6579 |                                 0.9895 |                                   0.5157 |
| val     |                            0.7126 |                                 1.1449 |                                   0.5359 |
| test    |                            0.7149 |                                 1.0791 |                                   0.5331 |

## Training

- best_epoch: 1
- best_val_loss: 0.7093
