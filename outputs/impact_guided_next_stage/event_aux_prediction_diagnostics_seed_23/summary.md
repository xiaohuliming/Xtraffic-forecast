# Event-Aux Prediction Diagnostics

- split: `test`
- `pred_event_aux` is the event-level auxiliary output used by event-conditioned veto probes.
- `std_*` compares standardized targets; `semantic_*` compares severity/recovery/spread after undoing normalization.

## Continuous Alignment

| model                     | feature   |   std_mae |   std_pearson |   std_spearman |   semantic_mae |   relu_pred_std_mean |
|:--------------------------|:----------|----------:|--------------:|---------------:|---------------:|---------------------:|
| hierarchical_conservative | severity  |  0.606218 |      0.610486 |       0.630580 |       0.915626 |             0.221880 |
| hierarchical_conservative | recovery  |  0.691095 |      0.615180 |       0.649050 |      52.151394 |             0.159736 |
| afffocus3_groupaware      | severity  |  0.643170 |      0.565111 |       0.580470 |       0.963386 |             0.154545 |
| afffocus3_groupaware      | recovery  |  0.743530 |      0.559425 |       0.587980 |      57.660690 |             0.175592 |

## Group Discrimination

| model                     | target             |   positive_rate |      auc |   score_gap |
|:--------------------------|:-------------------|----------------:|---------:|------------:|
| hierarchical_conservative | severity_high      |        0.333285 | 0.790873 |    0.551951 |
| hierarchical_conservative | recovery_long_ge90 |        0.420233 | 0.834438 |    0.630925 |
| afffocus3_groupaware      | severity_high      |        0.333285 | 0.767182 |    0.424125 |
| afffocus3_groupaware      | recovery_long_ge90 |        0.420233 | 0.801394 |    0.461311 |

## Predicted Signal By True Group

| model                     | group               |   samples |   true_severity_mean |   pred_severity_std_mean |   relu_pred_severity_std_mean |   true_recovery_mean |   pred_recovery_std_mean |   relu_pred_recovery_std_mean |
|:--------------------------|:--------------------|----------:|---------------------:|-------------------------:|------------------------------:|---------------------:|-------------------------:|------------------------------:|
| hierarchical_conservative | severity_low        |      9167 |             1.113532 |                -0.414258 |                      0.043165 |             6.951565 |                -0.550945 |                      0.022380 |
| hierarchical_conservative | severity_high       |      9165 |             4.282304 |                 0.371906 |                      0.428984 |           155.914886 |                 0.232125 |                      0.327993 |
| hierarchical_conservative | recovery_short_lt30 |     12563 |             1.465276 |                -0.343542 |                      0.057086 |             3.534586 |                -0.479838 |                      0.032119 |
| hierarchical_conservative | recovery_long_ge90  |     11556 |             3.714641 |                 0.369772 |                      0.415701 |           167.533752 |                 0.231629 |                      0.314608 |
| afffocus3_groupaware      | severity_low        |      9167 |             1.113532 |                -0.387638 |                      0.035558 |             6.951565 |                -0.320655 |                      0.045284 |
| afffocus3_groupaware      | severity_high       |      9165 |             4.282304 |                 0.220932 |                      0.290651 |           155.914886 |                 0.261553 |                      0.318399 |
| afffocus3_groupaware      | recovery_short_lt30 |     12563 |             1.465276 |                -0.326907 |                      0.047693 |             3.534586 |                -0.261176 |                      0.061361 |
| afffocus3_groupaware      | recovery_long_ge90  |     11556 |             3.714641 |                 0.217098 |                      0.281444 |           167.533752 |                 0.259583 |                      0.310233 |

## Interpretation Notes

- A useful dynamic event-conditioned gate should give clearly larger predicted severity/recovery scores on the high/long groups.
- AUC close to 0.5 means the predicted event signal is weak for ranking high-impact events.
