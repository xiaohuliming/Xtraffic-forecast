# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model               | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:--------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| focused_impact_veto | overall                | affected   | 4641126.000000 |        0.292067 |     0.045863 |         0.053025 |         0.042908 |            0.010117 |               -0.108690 |          0.000587 |            0.508352 |   0.629189 |
| focused_impact_veto | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038423 |         0.047818 |         0.035308 |            0.012509 |               -0.050459 |          0.000999 |            0.525205 |   0.666986 |
| focused_impact_veto | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046416 |         0.053213 |         0.043538 |            0.009676 |               -0.143850 |          0.000557 |            0.505904 |   0.624401 |
| focused_impact_veto | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039753 |         0.048430 |         0.036743 |            0.011688 |               -0.055503 |          0.001147 |            0.525495 |   0.655009 |
| focused_impact_veto | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046306 |         0.053127 |         0.043438 |            0.009688 |               -0.133164 |          0.000628 |            0.506185 |   0.624703 |
| focused_impact_veto | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039746 |         0.048351 |         0.036759 |            0.011592 |               -0.054811 |          0.001095 |            0.525506 |   0.653979 |
| focused_impact_veto | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046447 |         0.053162 |         0.043604 |            0.009558 |               -0.150364 |          0.000552 |            0.505196 |   0.623314 |
| focused_impact_veto | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040001 |         0.048545 |         0.037011 |            0.011534 |               -0.056081 |          0.001172 |            0.525930 |   0.652884 |
| lowsparse_rankalign | overall                | affected   | 4641126.000000 |        0.292067 |     0.045934 |         0.054222 |         0.042514 |            0.011707 |               -0.108690 |          0.000579 |            0.508282 |   0.644515 |
| lowsparse_rankalign | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038059 |         0.049142 |         0.034385 |            0.014757 |               -0.050459 |          0.001046 |            0.525159 |   0.691176 |
| lowsparse_rankalign | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046785 |         0.054631 |         0.043463 |            0.011169 |               -0.143850 |          0.000553 |            0.505839 |   0.638328 |
| lowsparse_rankalign | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039790 |         0.050242 |         0.036164 |            0.014078 |               -0.055503 |          0.001229 |            0.525404 |   0.680081 |
| lowsparse_rankalign | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046663 |         0.054550 |         0.043347 |            0.011203 |               -0.133164 |          0.000615 |            0.506111 |   0.639029 |
| lowsparse_rankalign | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039743 |         0.050115 |         0.036142 |            0.013972 |               -0.054811 |          0.001169 |            0.525415 |   0.679257 |
| lowsparse_rankalign | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046850 |         0.054596 |         0.043571 |            0.011025 |               -0.150364 |          0.000546 |            0.505129 |   0.637078 |
| lowsparse_rankalign | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040066 |         0.050405 |         0.036448 |            0.013957 |               -0.056081 |          0.001260 |            0.525840 |   0.678299 |

## Best Thresholds

| model               | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| focused_impact_veto | overall                | affected |            0.030000 |  0.488041 |            0.341650 |         0.853936 |                0.001001 |       0.130000 |       0.456944 |    0.000243 |           0.010490 |           0.000155 |
| focused_impact_veto | severity_high          | affected |            0.030000 |  0.489333 |            0.342053 |         0.859347 |                0.001178 |       0.140000 |       0.550000 |    0.000032 |           0.025183 |           0.000017 |
| focused_impact_veto | recovery_long_ge90     | affected |            0.030000 |  0.488376 |            0.341291 |         0.858257 |                0.001190 |       0.140000 |       0.500000 |    0.000036 |           0.007093 |           0.000021 |
| focused_impact_veto | severity_high_and_long | affected |            0.030000 |  0.488262 |            0.340978 |         0.859533 |                0.001200 |       0.142500 |       0.550000 |    0.000017 |           0.032278 |           0.000009 |
| lowsparse_rankalign | overall                | affected |            0.032500 |  0.500037 |            0.357837 |         0.829784 |                0.001117 |       0.152500 |       0.500000 |    0.000004 |           0.057461 |           0.000003 |
| lowsparse_rankalign | severity_high          | affected |            0.032500 |  0.500298 |            0.356913 |         0.836251 |                0.001365 |       0.140000 |       0.515789 |    0.000142 |           0.030131 |           0.000082 |
| lowsparse_rankalign | recovery_long_ge90     | affected |            0.032500 |  0.499803 |            0.356458 |         0.835982 |                0.001340 |       0.132500 |       0.452381 |    0.000565 |           0.012050 |           0.000370 |
| lowsparse_rankalign | severity_high_and_long | affected |            0.032500 |  0.499250 |            0.355779 |         0.836627 |                0.001396 |       0.132500 |       0.470968 |    0.000579 |           0.017516 |           0.000366 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
