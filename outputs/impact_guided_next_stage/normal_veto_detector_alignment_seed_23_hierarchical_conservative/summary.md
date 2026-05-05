# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model                     | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:--------------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| focused_impact_veto       | overall                | affected   | 4641126.000000 |        0.292067 |     0.045863 |         0.053025 |         0.042908 |            0.010117 |               -0.108690 |          0.000587 |            0.508352 |   0.629189 |
| focused_impact_veto       | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038423 |         0.047818 |         0.035308 |            0.012509 |               -0.050459 |          0.000999 |            0.525205 |   0.666986 |
| focused_impact_veto       | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046416 |         0.053213 |         0.043538 |            0.009676 |               -0.143850 |          0.000557 |            0.505904 |   0.624401 |
| focused_impact_veto       | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039753 |         0.048430 |         0.036743 |            0.011688 |               -0.055503 |          0.001147 |            0.525495 |   0.655009 |
| focused_impact_veto       | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046306 |         0.053127 |         0.043438 |            0.009688 |               -0.133164 |          0.000628 |            0.506185 |   0.624703 |
| focused_impact_veto       | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039746 |         0.048351 |         0.036759 |            0.011592 |               -0.054811 |          0.001095 |            0.525506 |   0.653979 |
| focused_impact_veto       | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046447 |         0.053162 |         0.043604 |            0.009558 |               -0.150364 |          0.000552 |            0.505196 |   0.623314 |
| focused_impact_veto       | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040001 |         0.048545 |         0.037011 |            0.011534 |               -0.056081 |          0.001172 |            0.525930 |   0.652884 |
| hierarchical_conservative | overall                | affected   | 4641126.000000 |        0.292067 |     0.046573 |         0.055017 |         0.043089 |            0.011927 |               -0.108690 |          0.000594 |            0.508236 |   0.636932 |
| hierarchical_conservative | overall                | unaffected | 9892006.000000 |        0.248973 |     0.039022 |         0.050010 |         0.035379 |            0.014631 |               -0.050459 |          0.001060 |            0.524958 |   0.677653 |
| hierarchical_conservative | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046992 |         0.055031 |         0.043588 |            0.011444 |               -0.143850 |          0.000558 |            0.505835 |   0.631904 |
| hierarchical_conservative | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.040171 |         0.050321 |         0.036650 |            0.013671 |               -0.055503 |          0.001229 |            0.525247 |   0.665718 |
| hierarchical_conservative | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046830 |         0.054872 |         0.043448 |            0.011424 |               -0.133164 |          0.000619 |            0.506106 |   0.632439 |
| hierarchical_conservative | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.040173 |         0.050260 |         0.036672 |            0.013588 |               -0.054811 |          0.001167 |            0.525254 |   0.665158 |
| hierarchical_conservative | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046944 |         0.054874 |         0.043587 |            0.011287 |               -0.150364 |          0.000544 |            0.505136 |   0.630723 |
| hierarchical_conservative | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040373 |         0.050384 |         0.036869 |            0.013515 |               -0.056081 |          0.001256 |            0.525681 |   0.663871 |

## Best Thresholds

| model                     | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| focused_impact_veto       | overall                | affected |            0.030000 |  0.488041 |            0.341650 |         0.853936 |                0.001001 |       0.130000 |       0.456944 |    0.000243 |           0.010490 |           0.000155 |
| focused_impact_veto       | severity_high          | affected |            0.030000 |  0.489333 |            0.342053 |         0.859347 |                0.001178 |       0.140000 |       0.550000 |    0.000032 |           0.025183 |           0.000017 |
| focused_impact_veto       | recovery_long_ge90     | affected |            0.030000 |  0.488376 |            0.341291 |         0.858257 |                0.001190 |       0.140000 |       0.500000 |    0.000036 |           0.007093 |           0.000021 |
| focused_impact_veto       | severity_high_and_long | affected |            0.030000 |  0.488262 |            0.340978 |         0.859533 |                0.001200 |       0.142500 |       0.550000 |    0.000017 |           0.032278 |           0.000009 |
| hierarchical_conservative | overall                | affected |            0.030000 |  0.493834 |            0.350973 |         0.832829 |                0.001035 |       0.170000 |       0.562500 |    0.000007 |           0.033560 |           0.000003 |
| hierarchical_conservative | severity_high          | affected |            0.030000 |  0.494652 |            0.351578 |         0.834079 |                0.001203 |       0.152500 |       0.532000 |    0.000193 |           0.023607 |           0.000108 |
| hierarchical_conservative | recovery_long_ge90     | affected |            0.030000 |  0.494081 |            0.350958 |         0.834325 |                0.001214 |       0.167500 |       0.555556 |    0.000018 |           0.013294 |           0.000010 |
| hierarchical_conservative | severity_high_and_long | affected |            0.030000 |  0.493794 |            0.350782 |         0.833681 |                0.001218 |       0.152500 |       0.519824 |    0.000187 |           0.022737 |           0.000107 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
