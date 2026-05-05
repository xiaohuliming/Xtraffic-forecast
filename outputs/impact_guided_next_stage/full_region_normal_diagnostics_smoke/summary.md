# Full-Region Normal Inference Diagnostic

This compares candidate-subgraph learned-normal inference against full-region inference sliced back to the same candidate nodes.

| region   |    train |      val |     test |   all_normal_diff_robust_mae |   all_local_target_robust_mae |   all_full_target_robust_mae |   all_target_change_pct |   affected_normal_diff_robust_mae |   affected_local_target_robust_mae |   affected_full_target_robust_mae |   affected_target_change_pct |
|:---------|---------:|---------:|---------:|-----------------------------:|------------------------------:|-----------------------------:|------------------------:|----------------------------------:|-----------------------------------:|----------------------------------:|-----------------------------:|
| Alameda  |  16.0000 |  16.0000 |  16.0000 |                       0.0663 |                        1.0750 |                       1.0616 |                 -1.2462 |                            0.0853 |                             1.3680 |                            1.3649 |                      -0.2213 |
| weighted | nan      | nan      | nan      |                       0.0663 |                        1.0750 |                       1.0616 |                 -1.2462 |                            0.0853 |                             1.3680 |                            1.3649 |                      -0.2213 |

Interpretation:
- `normal_diff_robust_mae` measures how much the normal forecast changes when using the full region graph.
- `target_change_pct` measures how the incident residual target magnitude changes under full-region normal inference.