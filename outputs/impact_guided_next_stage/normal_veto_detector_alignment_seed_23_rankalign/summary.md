# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model                 | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:----------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| focused_impact_veto   | overall                | affected   | 4641126.000000 |        0.292067 |     0.045863 |         0.053025 |         0.042908 |            0.010117 |               -0.108690 |          0.000587 |            0.508352 |   0.629189 |
| focused_impact_veto   | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038423 |         0.047818 |         0.035308 |            0.012509 |               -0.050459 |          0.000999 |            0.525205 |   0.666986 |
| focused_impact_veto   | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046416 |         0.053213 |         0.043538 |            0.009676 |               -0.143850 |          0.000557 |            0.505904 |   0.624401 |
| focused_impact_veto   | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039753 |         0.048430 |         0.036743 |            0.011688 |               -0.055503 |          0.001147 |            0.525495 |   0.655009 |
| focused_impact_veto   | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046306 |         0.053127 |         0.043438 |            0.009688 |               -0.133164 |          0.000628 |            0.506185 |   0.624703 |
| focused_impact_veto   | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039746 |         0.048351 |         0.036759 |            0.011592 |               -0.054811 |          0.001095 |            0.525506 |   0.653979 |
| focused_impact_veto   | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046447 |         0.053162 |         0.043604 |            0.009558 |               -0.150364 |          0.000552 |            0.505196 |   0.623314 |
| focused_impact_veto   | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040001 |         0.048545 |         0.037011 |            0.011534 |               -0.056081 |          0.001172 |            0.525930 |   0.652884 |
| rankalign_impact_veto | overall                | affected   | 4641126.000000 |        0.292067 |     0.044896 |         0.053113 |         0.041506 |            0.011607 |               -0.108690 |          0.000568 |            0.508365 |   0.644820 |
| rankalign_impact_veto | overall                | unaffected | 9892006.000000 |        0.248973 |     0.037174 |         0.048189 |         0.033522 |            0.014667 |               -0.050459 |          0.001033 |            0.525237 |   0.691481 |
| rankalign_impact_veto | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.045790 |         0.053562 |         0.042499 |            0.011063 |               -0.143850 |          0.000536 |            0.505914 |   0.638380 |
| rankalign_impact_veto | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.038953 |         0.049371 |         0.035339 |            0.014031 |               -0.055503 |          0.001215 |            0.525478 |   0.680398 |
| rankalign_impact_veto | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.045659 |         0.053485 |         0.042368 |            0.011117 |               -0.133164 |          0.000602 |            0.506188 |   0.639336 |
| rankalign_impact_veto | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.038901 |         0.049236 |         0.035314 |            0.013923 |               -0.054811 |          0.001155 |            0.525490 |   0.679594 |
| rankalign_impact_veto | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.045875 |         0.053557 |         0.042624 |            0.010933 |               -0.150364 |          0.000530 |            0.505200 |   0.637209 |
| rankalign_impact_veto | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.039253 |         0.049563 |         0.035645 |            0.013918 |               -0.056081 |          0.001245 |            0.525912 |   0.678643 |

## Best Thresholds

| model                 | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:----------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| focused_impact_veto   | overall                | affected |            0.030000 |  0.488041 |            0.341650 |         0.853936 |                0.001001 |       0.130000 |       0.456944 |    0.000243 |           0.010490 |           0.000155 |
| focused_impact_veto   | severity_high          | affected |            0.030000 |  0.489333 |            0.342053 |         0.859347 |                0.001178 |       0.140000 |       0.550000 |    0.000032 |           0.025183 |           0.000017 |
| focused_impact_veto   | recovery_long_ge90     | affected |            0.030000 |  0.488376 |            0.341291 |         0.858257 |                0.001190 |       0.140000 |       0.500000 |    0.000036 |           0.007093 |           0.000021 |
| focused_impact_veto   | severity_high_and_long | affected |            0.030000 |  0.488262 |            0.340978 |         0.859533 |                0.001200 |       0.142500 |       0.550000 |    0.000017 |           0.032278 |           0.000009 |
| rankalign_impact_veto | overall                | affected |            0.030000 |  0.500288 |            0.354558 |         0.849409 |                0.001043 |       0.130000 |       0.419979 |    0.000577 |           0.010185 |           0.000401 |
| rankalign_impact_veto | severity_high          | affected |            0.032500 |  0.500213 |            0.359536 |         0.821734 |                0.001392 |       0.130000 |       0.448571 |    0.000683 |           0.017804 |           0.000453 |
| rankalign_impact_veto | recovery_long_ge90     | affected |            0.032500 |  0.499831 |            0.359181 |         0.821532 |                0.001365 |       0.150000 |       0.666667 |    0.000002 |           0.205455 |           0.000001 |
| rankalign_impact_veto | severity_high_and_long | affected |            0.032500 |  0.499171 |            0.358374 |         0.822191 |                0.001425 |       0.130000 |       0.443299 |    0.000682 |           0.015895 |           0.000458 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
