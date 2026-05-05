# Impact Correction Adapter

This variant freezes the source dual-branch model and learns a small local correction for incident-impact magnitude.

## Test Result

- source all candidates robust MAE: `0.707528`
- source affected candidates robust MAE: `1.102129`
- adapter all candidates robust MAE: `0.706488`
- adapter affected candidates robust MAE: `1.101437`
- adapter unaffected candidates robust MAE: `0.521468`

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
- base_beta: 1.1
- epochs: 5
- lr: 0.001
- max_correction: 0.8
- affected_weight: 0.0
- severity_high_weight: 4.0
- recovery_long_weight: 4.0
- high_long_weight: 6.0
- correction_l1_weight: 0.0
- unaffected_correction_weight: 0.05
- correction_target_weight: 1.0
- correction_target_tail_only: True
- non_tail_affected_correction_weight: 0.05
- correction_node_gate_mode: none
- correction_node_gate_floor: 0.0
- correction_node_gate_temperature: 1.0
- correction_anomaly_gate_mode: branch_delta_abs
- correction_anomaly_gate_threshold: 0.5
- correction_anomaly_gate_floor: 0.25
- correction_anomaly_gate_temperature: 0.25
- selection_loss_key: final_loss
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
| val     |                          0.706160 |                               1.144449 |                                 0.527395 |
| test    |                          0.706488 |                               1.101437 |                                 0.521468 |

## Training

- best_epoch: 5
- best_selection_loss: 0.693800
- best_val_loss: 0.876837
