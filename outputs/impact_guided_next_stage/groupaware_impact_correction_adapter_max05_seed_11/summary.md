# Impact Correction Adapter

This variant freezes the source dual-branch model and learns a small local correction for incident-impact magnitude.

## Test Result

- source all candidates robust MAE: `0.709216`
- source affected candidates robust MAE: `1.104708`
- adapter all candidates robust MAE: `0.703569`
- adapter affected candidates robust MAE: `1.090669`
- adapter unaffected candidates robust MAE: `0.522153`

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_11_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
- base_beta: 1.1
- epochs: 5
- lr: 0.001
- max_correction: 0.5
- affected_weight: 0.0
- severity_high_weight: 2.0
- recovery_long_weight: 2.0
- high_long_weight: 3.0
- correction_l1_weight: 0.0
- unaffected_correction_weight: 0.05
- correction_target_weight: 1.0
- correction_target_tail_only: True
- non_tail_affected_correction_weight: 0.05
- selection_loss_key: loss
- trainable parameters: 13229

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |           3000 |
| val     |     29210 |           3000 |
| test    |     27499 |           3000 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| val     |                          0.722106 |                               1.160845 |                                 0.533591 |
| test    |                          0.703569 |                               1.090669 |                                 0.522153 |

## Training

- best_epoch: 1
- best_selection_loss: 0.745920
- best_val_loss: 0.745920
