# Impact Correction Adapter Evaluation

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_smoke_seed_23`
- cache_path: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5`

Negative delta means the adapter is better.

## Delta

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                         -0.000217 |                              -0.000253 |                                -0.000199 |

## Adapter

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.704957 |                               1.096914 |                                 0.521761 |

## Source

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.705173 |                               1.097167 |                                 0.521960 |
