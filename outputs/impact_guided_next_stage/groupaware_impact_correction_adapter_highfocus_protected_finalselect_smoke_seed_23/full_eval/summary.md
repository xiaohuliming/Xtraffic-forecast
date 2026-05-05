# Impact Correction Adapter Evaluation

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_protected_finalselect_smoke_seed_23`
- cache_path: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5`

Negative delta means the adapter is better.

## Delta

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                         -0.000006 |                              -0.000009 |                                -0.000005 |

## Adapter

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.707522 |                               1.102120 |                                 0.522385 |

## Source

| split   |   all_candidates_model_robust_mae |   affected_candidates_model_robust_mae |   unaffected_candidates_model_robust_mae |
|:--------|----------------------------------:|---------------------------------------:|-----------------------------------------:|
| test    |                          0.707528 |                               1.102129 |                                 0.522390 |
