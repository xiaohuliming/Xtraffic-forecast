# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model               | group                  | subset     |          count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:--------------------|:-----------------------|:-----------|---------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| element_normal_veto | overall                | affected   | 4641126.000000 |        0.292067 |     0.042515 |         0.050106 |         0.039383 |            0.010723 |               -0.108690 |          0.000477 |            0.508470 |   0.635533 |
| element_normal_veto | overall                | unaffected | 9892006.000000 |        0.248973 |     0.035215 |         0.045205 |         0.031903 |            0.013302 |               -0.050459 |          0.001000 |            0.525384 |   0.677388 |
| element_normal_veto | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.043094 |         0.050065 |         0.040142 |            0.009924 |               -0.143850 |          0.000236 |            0.506010 |   0.626780 |
| element_normal_veto | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.036178 |         0.045394 |         0.032981 |            0.012412 |               -0.055503 |          0.001157 |            0.525706 |   0.665910 |
| element_normal_veto | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.043039 |         0.050121 |         0.040061 |            0.010060 |               -0.133164 |          0.000379 |            0.506289 |   0.628579 |
| element_normal_veto | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.036180 |         0.045325 |         0.033006 |            0.012319 |               -0.054811 |          0.001104 |            0.525714 |   0.664926 |
| element_normal_veto | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.043194 |         0.050065 |         0.040286 |            0.009779 |               -0.150364 |          0.000209 |            0.505294 |   0.625322 |
| element_normal_veto | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.036389 |         0.045473 |         0.033210 |            0.012263 |               -0.056081 |          0.001185 |            0.526143 |   0.663943 |
| focused_impact_veto | overall                | affected   | 4641126.000000 |        0.292067 |     0.045863 |         0.053025 |         0.042908 |            0.010117 |               -0.108690 |          0.000587 |            0.508352 |   0.629189 |
| focused_impact_veto | overall                | unaffected | 9892006.000000 |        0.248973 |     0.038423 |         0.047818 |         0.035308 |            0.012509 |               -0.050459 |          0.000999 |            0.525205 |   0.666986 |
| focused_impact_veto | severity_high          | affected   | 2318560.000000 |        0.297464 |     0.046416 |         0.053213 |         0.043538 |            0.009676 |               -0.143850 |          0.000557 |            0.505904 |   0.624401 |
| focused_impact_veto | severity_high          | unaffected | 3498399.000000 |        0.257548 |     0.039753 |         0.048430 |         0.036743 |            0.011688 |               -0.055503 |          0.001147 |            0.525495 |   0.655009 |
| focused_impact_veto | recovery_long_ge90     | affected   | 2840138.000000 |        0.296026 |     0.046306 |         0.053127 |         0.043438 |            0.009688 |               -0.133164 |          0.000628 |            0.506185 |   0.624703 |
| focused_impact_veto | recovery_long_ge90     | unaffected | 4494774.000000 |        0.257688 |     0.039746 |         0.048351 |         0.036759 |            0.011592 |               -0.054811 |          0.001095 |            0.525506 |   0.653979 |
| focused_impact_veto | severity_high_and_long | affected   | 2119837.000000 |        0.297403 |     0.046447 |         0.053162 |         0.043604 |            0.009558 |               -0.150364 |          0.000552 |            0.505196 |   0.623314 |
| focused_impact_veto | severity_high_and_long | unaffected | 3087421.000000 |        0.259234 |     0.040001 |         0.048545 |         0.037011 |            0.011534 |               -0.056081 |          0.001172 |            0.525930 |   0.652884 |

## Best Thresholds

| model               | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:--------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| element_normal_veto | overall                | affected |            0.027500 |  0.494097 |            0.350706 |         0.835845 |                0.000784 |       0.072500 |       0.399434 |    0.171276 |           0.002336 |           0.125237 |
| element_normal_veto | severity_high          | affected |            0.027500 |  0.493686 |            0.349626 |         0.839658 |                0.000530 |       0.150000 |       0.600000 |    0.000004 |           0.067469 |           0.000002 |
| element_normal_veto | recovery_long_ge90     | affected |            0.027500 |  0.493630 |            0.349424 |         0.840500 |                0.000702 |       0.150000 |       0.454545 |    0.000006 |           0.006845 |           0.000004 |
| element_normal_veto | severity_high_and_long | affected |            0.027500 |  0.492602 |            0.348367 |         0.840662 |                0.000509 |       0.150000 |       0.600000 |    0.000005 |           0.067469 |           0.000002 |
| focused_impact_veto | overall                | affected |            0.030000 |  0.488041 |            0.341650 |         0.853936 |                0.001001 |       0.130000 |       0.456944 |    0.000243 |           0.010490 |           0.000155 |
| focused_impact_veto | severity_high          | affected |            0.030000 |  0.489333 |            0.342053 |         0.859347 |                0.001178 |       0.140000 |       0.550000 |    0.000032 |           0.025183 |           0.000017 |
| focused_impact_veto | recovery_long_ge90     | affected |            0.030000 |  0.488376 |            0.341291 |         0.858257 |                0.001190 |       0.140000 |       0.500000 |    0.000036 |           0.007093 |           0.000021 |
| focused_impact_veto | severity_high_and_long | affected |            0.030000 |  0.488262 |            0.340978 |         0.859533 |                0.001200 |       0.142500 |       0.550000 |    0.000017 |           0.032278 |           0.000009 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
