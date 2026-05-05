# Impact Correction Adapter Evaluation

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_seed_11`
- cache_path: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5`
- config_overrides: `{'correction_anomaly_gate_threshold': 0.7, 'correction_anomaly_gate_floor': 0.25}`

Negative delta means the adapter is better.

## Delta

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                         -0.000629 |                              -0.000472 |                                -0.000703 |

## Adapter

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.708580 |                               1.104223 |                                 0.522953 |

## Source

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.709210 |                               1.104695 |                                 0.523656 |
