# Impact Correction Adapter Evaluation

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_max05_seed_11`
- cache_path: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5`

Negative delta means the adapter is better.

## Delta

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                         -0.000090 |                              -0.000056 |                                -0.000105 |

## Adapter

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.709127 |                               1.104653 |                                 0.523554 |

## Source

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.709216 |                               1.104708 |                                 0.523659 |
