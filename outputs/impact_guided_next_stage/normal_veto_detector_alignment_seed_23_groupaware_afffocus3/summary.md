# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model                     | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:--------------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| hierarchical_conservative | overall                | affected   | 4641126.000000 |        0.292067 |     0.046573 |         0.055017 |         0.043089 |            0.011927 |               -0.108690 |          0.000594 |            0.508236 |   0.636932 |
| hierarchical_conservative | overall                | unaffected | 9892006.000000 |        0.248973 |     0.039022 |         0.050010 |         0.035379 |            0.014631 |               -0.050459 |          0.001060 |            0.524958 |   0.677653 |
| hierarchical_conservative | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046992 |         0.055031 |         0.043588 |            0.011444 |               -0.143850 |          0.000558 |            0.505835 |   0.631904 |
| hierarchical_conservative | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.040171 |         0.050321 |         0.036650 |            0.013671 |               -0.055503 |          0.001229 |            0.525247 |   0.665718 |
| hierarchical_conservative | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046830 |         0.054872 |         0.043448 |            0.011424 |               -0.133164 |          0.000619 |            0.506106 |   0.632439 |
| hierarchical_conservative | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.040173 |         0.050260 |         0.036672 |            0.013588 |               -0.054811 |          0.001167 |            0.525254 |   0.665158 |
| hierarchical_conservative | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046944 |         0.054874 |         0.043587 |            0.011287 |               -0.150364 |          0.000544 |            0.505136 |   0.630723 |
| hierarchical_conservative | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040373 |         0.050384 |         0.036869 |            0.013515 |               -0.056081 |          0.001256 |            0.525681 |   0.663871 |
| afffocus3_groupaware      | overall                | affected   | 4641126.000000 |        0.300034 |     0.099771 |         0.118523 |         0.091733 |            0.026790 |               -0.106755 |          0.001868 |            0.510301 |   0.640977 |
| afffocus3_groupaware      | overall                | unaffected | 9892006.000000 |        0.257246 |     0.079353 |         0.102093 |         0.071478 |            0.030615 |               -0.049225 |          0.002364 |            0.527207 |   0.678379 |
| afffocus3_groupaware      | severity_high          | affected   | 2318560.000000 |        0.305487 |     0.102000 |         0.120144 |         0.094019 |            0.026125 |               -0.141470 |          0.002290 |            0.508269 |   0.636845 |
| afffocus3_groupaware      | severity_high          | unaffected | 3498399.000000 |        0.265853 |     0.082320 |         0.103497 |         0.074651 |            0.028846 |               -0.053884 |          0.002827 |            0.527728 |   0.666653 |
| afffocus3_groupaware      | recovery_long_ge90     | affected   | 2840138.000000 |        0.304111 |     0.101693 |         0.119829 |         0.093767 |            0.026063 |               -0.130721 |          0.002299 |            0.508574 |   0.636945 |
| afffocus3_groupaware      | recovery_long_ge90     | unaffected | 4494774.000000 |        0.265952 |     0.082339 |         0.103368 |         0.074720 |            0.028649 |               -0.053308 |          0.002670 |            0.527565 |   0.665930 |
| afffocus3_groupaware      | severity_high_and_long | affected   | 2119837.000000 |        0.305465 |     0.102275 |         0.120247 |         0.094371 |            0.025875 |               -0.147846 |          0.002338 |            0.507699 |   0.635768 |
| afffocus3_groupaware      | severity_high_and_long | unaffected | 3087421.000000 |        0.267516 |     0.082928 |         0.103850 |         0.075287 |            0.028563 |               -0.054408 |          0.002897 |            0.528160 |   0.664754 |

## Best Thresholds

| model                     | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| hierarchical_conservative | overall                | affected |            0.030000 |  0.493834 |            0.350973 |         0.832829 |                0.001035 |       0.170000 |       0.562500 |    0.000007 |           0.033560 |           0.000003 |
| hierarchical_conservative | severity_high          | affected |            0.030000 |  0.494652 |            0.351578 |         0.834079 |                0.001203 |       0.152500 |       0.532000 |    0.000193 |           0.023607 |           0.000108 |
| hierarchical_conservative | recovery_long_ge90     | affected |            0.030000 |  0.494081 |            0.350958 |         0.834325 |                0.001214 |       0.167500 |       0.555556 |    0.000018 |           0.013294 |           0.000010 |
| hierarchical_conservative | severity_high_and_long | affected |            0.030000 |  0.493794 |            0.350782 |         0.833681 |                0.001218 |       0.152500 |       0.519824 |    0.000187 |           0.022737 |           0.000107 |
| afffocus3_groupaware      | overall                | affected |            0.062500 |  0.507513 |            0.363919 |         0.838280 |                0.003037 |       0.347500 |       0.561644 |    0.000059 |           0.109217 |           0.000031 |
| afffocus3_groupaware      | severity_high          | affected |            0.062500 |  0.508837 |            0.363484 |         0.847908 |                0.004026 |       0.340000 |       0.556886 |    0.000131 |           0.107132 |           0.000072 |
| afffocus3_groupaware      | recovery_long_ge90     | affected |            0.062500 |  0.508059 |            0.362766 |         0.847491 |                0.003857 |       0.342500 |       0.554878 |    0.000105 |           0.102794 |           0.000058 |
| afffocus3_groupaware      | severity_high_and_long | affected |            0.065000 |  0.507974 |            0.364904 |         0.835589 |                0.004301 |       0.340000 |       0.554140 |    0.000134 |           0.114770 |           0.000074 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
