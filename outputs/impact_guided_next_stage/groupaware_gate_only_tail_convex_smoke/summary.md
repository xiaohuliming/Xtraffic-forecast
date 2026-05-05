# ST-TIS Incident Branch Fine-Tune

This variant freezes the normal branch and emphasizes affected-node residual learning in the incident branch.

## Test Result

- source all candidates robust MAE: `0.7075`
- source affected candidates robust MAE: `1.1021`
- incident-ft all candidates robust MAE: `0.7071`
- incident-ft affected candidates robust MAE: `1.1027`
- residual_beta: 1.00

## Settings

- model_dir: `outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
- train_incident: False
- train_gate: True
- epochs: 2
- lr: 5e-05
- affected_weight: 4.0
- incident_loss_weight: 0.0
- gate_loss_weight: 0.0
- normal_better_gate_loss_weight: 0.0
- normal_better_margin: 0.1
- normal_better_min_gate: 0.0
- convex_gate_loss_weight: 0.1
- convex_gate_min_gap: 0.05
- severity_high_weight: 0.75
- recovery_long_weight: 0.75
- high_long_weight: 1.0
- tail_weight_max: 2.0
- severity_high_z_threshold: 0.3522920310497284
- recovery_long_z_threshold: 0.24938054382801056
- trainable parameters: 46572

## Split Counts

| split   |   samples |   eval_samples |
|:--------|----------:|---------------:|
| train   |    138528 |           3000 |
| val     |     29210 |           3000 |
| test    |     27499 |           3000 |

## Metrics

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| val     |                            0.7067 |                                 1.1456 |                                   0.5277 |
| test    |                            0.7071 |                                 1.1027 |                                   0.5218 |

## Training

- best_epoch: 2
- best_val_loss: 0.7585
