# Full-Region Normal Inference Diagnostic

This compares candidate-subgraph learned-normal inference against full-region inference sliced back to the same candidate nodes.

| region      |    train |      val |     test |   all_normal_diff_robust_mae |   all_local_target_robust_mae |   all_full_target_robust_mae |   all_target_change_pct |   affected_normal_diff_robust_mae |   affected_local_target_robust_mae |   affected_full_target_robust_mae |   affected_target_change_pct |
|:------------|---------:|---------:|---------:|-----------------------------:|------------------------------:|-----------------------------:|------------------------:|----------------------------------:|-----------------------------------:|----------------------------------:|-----------------------------:|
| Alameda     | 256.0000 | 256.0000 | 256.0000 |                       0.0545 |                        0.8295 |                       0.8214 |                 -0.9812 |                            0.0645 |                             1.3147 |                            1.3106 |                      -0.3125 |
| ContraCosta | 256.0000 | 256.0000 | 256.0000 |                       0.0622 |                        0.9173 |                       0.9109 |                 -0.6949 |                            0.0780 |                             1.4811 |                            1.4733 |                      -0.5245 |
| Orange      | 256.0000 | 256.0000 | 256.0000 |                       0.0696 |                        0.8974 |                       0.8935 |                 -0.4271 |                            0.0827 |                             1.3882 |                            1.3863 |                      -0.1339 |
| weighted    | nan      | nan      | nan      |                       0.0626 |                        0.8868 |                       0.8808 |                 -0.6764 |                            0.0761 |                             1.4056 |                            1.4008 |                      -0.3397 |

Interpretation:
- `normal_diff_robust_mae` measures how much the normal forecast changes when using the full region graph.
- `target_change_pct` measures how the incident residual target magnitude changes under full-region normal inference.