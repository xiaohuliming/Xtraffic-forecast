# Impact Correction Adapter Evaluation

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_seed_11`
- cache_path: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5`
- config_overrides: `{'correction_anomaly_gate_threshold': 0.3, 'correction_anomaly_gate_floor': 0.25}`

Negative delta means the adapter is better.

## Delta

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                         -0.000739 |                              -0.000558 |                                -0.000823 |

## Adapter

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.708471 |                               1.104137 |                                 0.522833 |

## Source

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.709210 |                               1.104695 |                                 0.523656 |
