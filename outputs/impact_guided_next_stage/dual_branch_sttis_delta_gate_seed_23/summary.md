# ST-TIS Bidirectional Gate Delta

This variant freezes the proposal-aware model and trains only a bounded bidirectional gate-delta adapter.

## Test Result

- source all candidates robust MAE: `0.7113`
- source affected candidates robust MAE: `1.1182`
- delta-gate all candidates robust MAE: `0.7110`
- delta-gate affected candidates robust MAE: `1.1168`
- delta_scale: 0.75
- residual_beta: 1.05

## Selection

- selection_metric: `affected_candidates_model_robust_mae`
- all_val_tolerance: `0.002`
- sweep_scales: `0.0,0.25,0.5,0.75,1.0,1.25,1.5`
- sweep_betas: `0.95,1.0,1.05,1.1`

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_proposal_gate_seed_23`
- epochs: 3
- lr: 0.0002
- gate_loss_weight: 0.1
- hard_residual_weight: 0.05
- delta_l2_weight: 0.001
- hard_margin: 0.1
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
| train   |                            0.6575 |                                 1.0084 |                                   0.5137 |
| val     |                            0.7166 |                                 1.1661 |                                   0.5293 |
| test    |                            0.7110 |                                 1.1168 |                                   0.5206 |

## Training

- best_epoch: 3
- best_val_loss: 0.5132
