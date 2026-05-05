# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model                    | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:-------------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| focused_impact_veto      | overall                | affected   | 4641126.000000 |        0.292067 |     0.045863 |         0.053025 |         0.042908 |            0.010117 |               -0.108690 |          0.000587 |            0.508352 |   0.629189 |
| focused_impact_veto      | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038423 |         0.047818 |         0.035308 |            0.012509 |               -0.050459 |          0.000999 |            0.525205 |   0.666986 |
| focused_impact_veto      | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046416 |         0.053213 |         0.043538 |            0.009676 |               -0.143850 |          0.000557 |            0.505904 |   0.624401 |
| focused_impact_veto      | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039753 |         0.048430 |         0.036743 |            0.011688 |               -0.055503 |          0.001147 |            0.525495 |   0.655009 |
| focused_impact_veto      | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046306 |         0.053127 |         0.043438 |            0.009688 |               -0.133164 |          0.000628 |            0.506185 |   0.624703 |
| focused_impact_veto      | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039746 |         0.048351 |         0.036759 |            0.011592 |               -0.054811 |          0.001095 |            0.525506 |   0.653979 |
| focused_impact_veto      | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046447 |         0.053162 |         0.043604 |            0.009558 |               -0.150364 |          0.000552 |            0.505196 |   0.623314 |
| focused_impact_veto      | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040001 |         0.048545 |         0.037011 |            0.011534 |               -0.056081 |          0.001172 |            0.525930 |   0.652884 |
| hierarchical_impact_veto | overall                | affected   | 4641126.000000 |        0.292067 |     0.047655 |         0.055385 |         0.044466 |            0.010919 |               -0.108690 |          0.000570 |            0.508244 |   0.630731 |
| hierarchical_impact_veto | overall                | unaffected | 9892006.000000 |        0.248973 |     0.040448 |         0.050744 |         0.037034 |            0.013710 |               -0.050459 |          0.001049 |            0.524950 |   0.670262 |
| hierarchical_impact_veto | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.048110 |         0.055505 |         0.044979 |            0.010527 |               -0.143850 |          0.000513 |            0.505833 |   0.626335 |
| hierarchical_impact_veto | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.041762 |         0.051261 |         0.038466 |            0.012795 |               -0.055503 |          0.001209 |            0.525217 |   0.658317 |
| hierarchical_impact_veto | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.047932 |         0.055321 |         0.044825 |            0.010496 |               -0.133164 |          0.000584 |            0.506113 |   0.626693 |
| hierarchical_impact_veto | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.041740 |         0.051180 |         0.038463 |            0.012717 |               -0.054811 |          0.001153 |            0.525229 |   0.657866 |
| hierarchical_impact_veto | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.048055 |         0.055359 |         0.044964 |            0.010395 |               -0.150364 |          0.000500 |            0.505135 |   0.625274 |
| hierarchical_impact_veto | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.041978 |         0.051343 |         0.038701 |            0.012642 |               -0.056081 |          0.001234 |            0.525653 |   0.656517 |

## Best Thresholds

| model                    | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:-------------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| focused_impact_veto      | overall                | affected |            0.030000 |  0.488041 |            0.341650 |         0.853936 |                0.001001 |       0.130000 |       0.456944 |    0.000243 |           0.010490 |           0.000155 |
| focused_impact_veto      | severity_high          | affected |            0.030000 |  0.489333 |            0.342053 |         0.859347 |                0.001178 |       0.140000 |       0.550000 |    0.000032 |           0.025183 |           0.000017 |
| focused_impact_veto      | recovery_long_ge90     | affected |            0.030000 |  0.488376 |            0.341291 |         0.858257 |                0.001190 |       0.140000 |       0.500000 |    0.000036 |           0.007093 |           0.000021 |
| focused_impact_veto      | severity_high_and_long | affected |            0.030000 |  0.488262 |            0.340978 |         0.859533 |                0.001200 |       0.142500 |       0.550000 |    0.000017 |           0.032278 |           0.000009 |
| hierarchical_impact_veto | overall                | affected |            0.032500 |  0.488932 |            0.347465 |         0.824699 |                0.001036 |       0.167500 |       0.555556 |    0.000004 |          -0.005638 |           0.000002 |
| hierarchical_impact_veto | severity_high          | affected |            0.032500 |  0.490592 |            0.348935 |         0.825869 |                0.001244 |       0.147500 |       0.532787 |    0.000188 |           0.013642 |           0.000105 |
| hierarchical_impact_veto | recovery_long_ge90     | affected |            0.032500 |  0.489758 |            0.348126 |         0.825673 |                0.001241 |       0.165000 |       0.583333 |    0.000008 |           0.000654 |           0.000004 |
| hierarchical_impact_veto | severity_high_and_long | affected |            0.032500 |  0.489703 |            0.348213 |         0.824878 |                0.001271 |       0.147500 |       0.524664 |    0.000186 |           0.010675 |           0.000105 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
