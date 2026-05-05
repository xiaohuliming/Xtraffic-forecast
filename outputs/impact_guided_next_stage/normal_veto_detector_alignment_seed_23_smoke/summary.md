# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model               | group                  | subset     |         count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:--------------------|:-----------------------|:-----------|--------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| element_normal_veto | overall                | affected   |  87720.000000 |        0.291632 |     0.042580 |         0.050575 |         0.039288 |            0.011287 |               -0.099789 |          0.000657 |            0.501379 |   0.642733 |
| element_normal_veto | overall                | unaffected | 186335.000000 |        0.251042 |     0.035125 |         0.045065 |         0.031793 |            0.013272 |               -0.049683 |          0.000926 |            0.529766 |   0.679913 |
| element_normal_veto | severity_high          | affected   |  43024.000000 |        0.298113 |     0.042787 |         0.050446 |         0.039534 |            0.010912 |               -0.128340 |          0.000416 |            0.498745 |   0.638781 |
| element_normal_veto | severity_high          | unaffected |  66254.000000 |        0.261871 |     0.036410 |         0.045268 |         0.033267 |            0.012000 |               -0.051489 |          0.000958 |            0.534202 |   0.662659 |
| element_normal_veto | recovery_long_ge90     | affected   |  51604.000000 |        0.294938 |     0.042952 |         0.050443 |         0.039819 |            0.010624 |               -0.129572 |          0.000359 |            0.494923 |   0.635745 |
| element_normal_veto | recovery_long_ge90     | unaffected |  84358.000000 |        0.259157 |     0.036292 |         0.045257 |         0.033156 |            0.012102 |               -0.054266 |          0.000958 |            0.529624 |   0.665048 |
| element_normal_veto | severity_high_and_long | affected   |  38428.000000 |        0.294915 |     0.042757 |         0.050370 |         0.039572 |            0.010798 |               -0.138950 |          0.000350 |            0.495576 |   0.637828 |
| element_normal_veto | severity_high_and_long | unaffected |  58826.000000 |        0.261500 |     0.036368 |         0.045378 |         0.033178 |            0.012201 |               -0.050703 |          0.001094 |            0.533336 |   0.665631 |
| focused_impact_veto | overall                | affected   |  87720.000000 |        0.291632 |     0.046138 |         0.053809 |         0.042979 |            0.010830 |               -0.099789 |          0.000851 |            0.501277 |   0.636592 |
| focused_impact_veto | overall                | unaffected | 186335.000000 |        0.251042 |     0.038397 |         0.047637 |         0.035299 |            0.012337 |               -0.049683 |          0.000924 |            0.529546 |   0.667513 |
| focused_impact_veto | severity_high          | affected   |  43024.000000 |        0.298113 |     0.046330 |         0.053815 |         0.043152 |            0.010663 |               -0.128340 |          0.000773 |            0.498652 |   0.635089 |
| focused_impact_veto | severity_high          | unaffected |  66254.000000 |        0.261871 |     0.040093 |         0.048493 |         0.037113 |            0.011380 |               -0.051489 |          0.000960 |            0.533900 |   0.652365 |
| focused_impact_veto | recovery_long_ge90     | affected   |  51604.000000 |        0.294938 |     0.046373 |         0.053744 |         0.043289 |            0.010455 |               -0.129572 |          0.000703 |            0.494865 |   0.633305 |
| focused_impact_veto | recovery_long_ge90     | unaffected |  84358.000000 |        0.259157 |     0.039885 |         0.048263 |         0.036955 |            0.011308 |               -0.054266 |          0.000952 |            0.529493 |   0.652368 |
| focused_impact_veto | severity_high_and_long | affected   |  38428.000000 |        0.294915 |     0.046171 |         0.053681 |         0.043031 |            0.010650 |               -0.138950 |          0.000726 |            0.495472 |   0.636341 |
| focused_impact_veto | severity_high_and_long | unaffected |  58826.000000 |        0.261500 |     0.039989 |         0.048420 |         0.037004 |            0.011416 |               -0.050703 |          0.001089 |            0.533115 |   0.653523 |

## Best Thresholds

| model               | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| element_normal_veto | overall                | affected |            0.027500 |  0.497211 |            0.352548 |         0.843210 |                0.001063 |       0.130000 |       0.750000 |    0.000235 |           0.044239 |           0.000091 |
| element_normal_veto | severity_high          | affected |            0.027500 |  0.498121 |            0.353669 |         0.842040 |                0.000845 |       0.120000 |       0.555556 |    0.002729 |           0.055538 |           0.001464 |
| element_normal_veto | recovery_long_ge90     | affected |            0.027500 |  0.495168 |            0.349906 |         0.846649 |                0.000663 |       0.127500 |       0.666667 |    0.000394 |           0.075659 |           0.000174 |
| element_normal_veto | severity_high_and_long | affected |            0.027500 |  0.494443 |            0.349636 |         0.843995 |                0.000784 |       0.117500 |       0.552239 |    0.003265 |           0.053097 |           0.001744 |
| focused_impact_veto | overall                | affected |            0.035000 |  0.491878 |            0.355771 |         0.796654 |                0.001718 |       0.115000 |       0.560847 |    0.004144 |           0.051173 |           0.002155 |
| focused_impact_veto | severity_high          | affected |            0.035000 |  0.493192 |            0.356911 |         0.797833 |                0.002146 |       0.105000 |       0.550351 |    0.018322 |           0.040060 |           0.009925 |
| focused_impact_veto | recovery_long_ge90     | affected |            0.035000 |  0.491317 |            0.354653 |         0.799343 |                0.001859 |       0.107500 |       0.553055 |    0.011301 |           0.036437 |           0.006027 |
| focused_impact_veto | severity_high_and_long | affected |            0.035000 |  0.490444 |            0.353968 |         0.798200 |                0.002206 |       0.107500 |       0.600000 |    0.012706 |           0.049559 |           0.006245 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
