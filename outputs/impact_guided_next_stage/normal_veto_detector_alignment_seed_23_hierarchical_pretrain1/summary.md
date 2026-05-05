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
| hierarchical_pretrain1    | overall                | affected   | 4641126.000000 |        0.292067 |     0.047626 |         0.056187 |         0.044094 |            0.012093 |               -0.108690 |          0.000690 |            0.508083 |   0.639769 |
| hierarchical_pretrain1    | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038439 |         0.048932 |         0.034961 |            0.013971 |               -0.050459 |          0.001047 |            0.525019 |   0.677073 |
| hierarchical_pretrain1    | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.048294 |         0.056565 |         0.044792 |            0.011773 |               -0.143850 |          0.000755 |            0.505640 |   0.635807 |
| hierarchical_pretrain1    | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039518 |         0.049216 |         0.036153 |            0.013062 |               -0.055503 |          0.001219 |            0.525316 |   0.664995 |
| hierarchical_pretrain1    | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.048171 |         0.056431 |         0.044697 |            0.011734 |               -0.133164 |          0.000790 |            0.505910 |   0.635908 |
| hierarchical_pretrain1    | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039545 |         0.049179 |         0.036201 |            0.012978 |               -0.054811 |          0.001159 |            0.525319 |   0.664350 |
| hierarchical_pretrain1    | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.048345 |         0.056529 |         0.044881 |            0.011648 |               -0.150364 |          0.000751 |            0.504926 |   0.634697 |
| hierarchical_pretrain1    | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.039750 |         0.049314 |         0.036403 |            0.012911 |               -0.056081 |          0.001246 |            0.525743 |   0.663010 |

## Best Thresholds

| model                     | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| hierarchical_conservative | overall                | affected |            0.030000 |  0.493834 |            0.350973 |         0.832829 |                0.001035 |       0.170000 |       0.562500 |    0.000007 |           0.033560 |           0.000003 |
| hierarchical_conservative | severity_high          | affected |            0.030000 |  0.494652 |            0.351578 |         0.834079 |                0.001203 |       0.152500 |       0.532000 |    0.000193 |           0.023607 |           0.000108 |
| hierarchical_conservative | recovery_long_ge90     | affected |            0.030000 |  0.494081 |            0.350958 |         0.834325 |                0.001214 |       0.167500 |       0.555556 |    0.000018 |           0.013294 |           0.000010 |
| hierarchical_conservative | severity_high_and_long | affected |            0.030000 |  0.493794 |            0.350782 |         0.833681 |                0.001218 |       0.152500 |       0.519824 |    0.000187 |           0.022737 |           0.000107 |
| hierarchical_pretrain1    | overall                | affected |            0.032500 |  0.495968 |            0.355628 |         0.819275 |                0.001279 |       0.167500 |       0.565217 |    0.000048 |           0.041464 |           0.000025 |
| hierarchical_pretrain1    | severity_high          | affected |            0.032500 |  0.497289 |            0.355877 |         0.825185 |                0.001653 |       0.155000 |       0.561886 |    0.000415 |           0.045075 |           0.000220 |
| hierarchical_pretrain1    | recovery_long_ge90     | affected |            0.032500 |  0.496438 |            0.355013 |         0.825149 |                0.001591 |       0.167500 |       0.555556 |    0.000054 |           0.039545 |           0.000029 |
| hierarchical_pretrain1    | severity_high_and_long | affected |            0.032500 |  0.496352 |            0.354748 |         0.826106 |                0.001679 |       0.155000 |       0.550420 |    0.000416 |           0.044712 |           0.000225 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
