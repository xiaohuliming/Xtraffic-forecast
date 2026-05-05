# Normal-Veto Detector Alignment

- split: `test`
- positive target: `base_abs - normal_abs > 0.1`

## Summary

| model              | group                  | subset     |        count |   positive_rate |   score_mean |   score_pos_mean |   score_neg_mean |   score_pos_neg_gap |   normal_advantage_mean |   final_gain_mean |   final_better_rate |   hist_auc |
|:-------------------|:-----------------------|:-----------|-------------:|----------------:|-------------:|-----------------:|-----------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------:|
| hierarchical_smoke | overall                | affected   | 46100.000000 |        0.288308 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.115921 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | overall                | unaffected | 97610.000000 |        0.243950 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.050867 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | severity_high          | affected   | 22430.000000 |        0.304637 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.174433 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | severity_high          | unaffected | 32834.000000 |        0.254157 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.051700 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | recovery_long_ge90     | affected   | 28894.000000 |        0.298228 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.146560 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | recovery_long_ge90     | unaffected | 47752.000000 |        0.253288 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.047356 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | severity_high_and_long | affected   | 20152.000000 |        0.306322 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.182340 |          0.000000 |            0.000000 |   0.500000 |
| hierarchical_smoke | severity_high_and_long | unaffected | 28298.000000 |        0.258676 |     0.000000 |         0.000000 |         0.000000 |            0.000000 |               -0.048706 |          0.000000 |            0.000000 |   0.500000 |

## Best Thresholds

| model              | group                  | subset   |   best_f1_threshold |   best_f1 |   best_f1_precision |   best_f1_recall |   best_f1_selected_gain |   hp_threshold |   hp_precision |   hp_recall |   hp_selected_gain |   hp_selected_rate |
|:-------------------|:-----------------------|:---------|--------------------:|----------:|--------------------:|-----------------:|------------------------:|---------------:|---------------:|------------:|-------------------:|-------------------:|
| hierarchical_smoke | overall                | affected |            0.000000 |  0.447576 |            0.288308 |         1.000000 |                0.000000 |       0.000000 |       0.288308 |    1.000000 |           0.000000 |           1.000000 |
| hierarchical_smoke | severity_high          | affected |            0.000000 |  0.467006 |            0.304637 |         1.000000 |                0.000000 |       0.000000 |       0.304637 |    1.000000 |           0.000000 |           1.000000 |
| hierarchical_smoke | recovery_long_ge90     | affected |            0.000000 |  0.459439 |            0.298228 |         1.000000 |                0.000000 |       0.000000 |       0.298228 |    1.000000 |           0.000000 |           1.000000 |
| hierarchical_smoke | severity_high_and_long | affected |            0.000000 |  0.468984 |            0.306322 |         1.000000 |                0.000000 |       0.000000 |       0.306322 |    1.000000 |           0.000000 |           1.000000 |

## Interpretation Notes

- `score_pos_neg_gap` measures whether the veto score is higher where normal branch is actually better.
- `hist_auc` is a histogram approximation of detector AUC; 0.5 means no ranking signal.
- `selected_final_gain_mean` is positive when high-score positions improve the final fused residual over the base fused proposal.
