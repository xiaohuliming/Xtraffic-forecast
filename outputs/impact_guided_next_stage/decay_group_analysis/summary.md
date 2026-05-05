# Decay Group Analysis

This compares the no-decay disagreement-proxy model against the temporal-decay-head model on the shared test split.

## Overall

| group        |   samples |   no_decay_all_model_robust_mae |   decay_all_model_robust_mae |   all_decay_gain_pct |   no_decay_affected_model_robust_mae |   decay_affected_model_robust_mae |   affected_decay_gain_pct |   h06_affected_decay_gain_pct |   h12_affected_decay_gain_pct |
|:-------------|----------:|--------------------------------:|-----------------------------:|---------------------:|-------------------------------------:|----------------------------------:|--------------------------:|------------------------------:|------------------------------:|
| overall_test |     27499 |                          0.7248 |                       0.7239 |               0.1172 |                               1.1381 |                            1.1308 |                    0.6394 |                        0.6698 |                        0.7182 |

## Severity Groups

| group         |   samples |   no_decay_all_model_robust_mae |   decay_all_model_robust_mae |   all_decay_gain_pct |   no_decay_affected_model_robust_mae |   decay_affected_model_robust_mae |   affected_decay_gain_pct |   h06_affected_decay_gain_pct |   h12_affected_decay_gain_pct |
|:--------------|----------:|--------------------------------:|-----------------------------:|---------------------:|-------------------------------------:|----------------------------------:|--------------------------:|------------------------------:|------------------------------:|
| severity_low  |      9167 |                          0.5789 |                       0.5808 |              -0.3367 |                               0.8539 |                            0.8553 |                   -0.1559 |                       -0.1607 |                       -0.2963 |
| severity_mid  |      9167 |                          0.6752 |                       0.6760 |              -0.1203 |                               0.9826 |                            0.9811 |                    0.1540 |                        0.1574 |                        0.0607 |
| severity_high |      9165 |                          0.8600 |                       0.8560 |               0.4720 |                               1.3345 |                            1.3206 |                    1.0454 |                        1.1004 |                        1.2163 |

## Recovery Groups

| group               |   samples |   no_decay_all_model_robust_mae |   decay_all_model_robust_mae |   all_decay_gain_pct |   no_decay_affected_model_robust_mae |   decay_affected_model_robust_mae |   affected_decay_gain_pct |   h06_affected_decay_gain_pct |   h12_affected_decay_gain_pct |
|:--------------------|----------:|--------------------------------:|-----------------------------:|---------------------:|-------------------------------------:|----------------------------------:|--------------------------:|------------------------------:|------------------------------:|
| recovery_short_lt30 |     12563 |                          0.6174 |                       0.6181 |              -0.1155 |                               0.9591 |                            0.9572 |                    0.1902 |                        0.1555 |                        0.2701 |
| recovery_mid_30_90  |      3380 |                          0.6819 |                       0.6847 |              -0.4230 |                               1.0520 |                            1.0545 |                   -0.2362 |                       -0.1588 |                       -0.5135 |
| recovery_long_ge90  |     11556 |                          0.8140 |                       0.8111 |               0.3596 |                               1.2339 |                            1.2224 |                    0.9365 |                        0.9828 |                        1.0535 |

## Horizon Comparison

|   horizon |   no_decay_all_model_robust_mae |   decay_all_model_robust_mae |   all_decay_gain_pct |   no_decay_affected_model_robust_mae |   decay_affected_model_robust_mae |   affected_decay_gain_pct |
|----------:|--------------------------------:|-----------------------------:|---------------------:|-------------------------------------:|----------------------------------:|--------------------------:|
|    1.0000 |                          0.5335 |                       0.5333 |               0.0397 |                               0.7487 |                            0.7465 |                    0.2893 |
|    2.0000 |                          0.5965 |                       0.5964 |               0.0150 |                               0.8738 |                            0.8707 |                    0.3511 |
|    3.0000 |                          0.6449 |                       0.6448 |               0.0210 |                               0.9711 |                            0.9664 |                    0.4833 |
|    4.0000 |                          0.6791 |                       0.6787 |               0.0642 |                               1.0415 |                            1.0358 |                    0.5498 |
|    5.0000 |                          0.7054 |                       0.7043 |               0.1579 |                               1.0976 |                            1.0899 |                    0.6994 |
|    6.0000 |                          0.7298 |                       0.7287 |               0.1529 |                               1.1475 |                            1.1398 |                    0.6698 |
|    7.0000 |                          0.7533 |                       0.7521 |               0.1579 |                               1.1953 |                            1.1869 |                    0.7019 |
|    8.0000 |                          0.7745 |                       0.7738 |               0.0919 |                               1.2401 |                            1.2324 |                    0.6184 |
|    9.0000 |                          0.7948 |                       0.7933 |               0.1906 |                               1.2817 |                            1.2715 |                    0.8013 |
|   10.0000 |                          0.8126 |                       0.8116 |               0.1204 |                               1.3216 |                            1.3121 |                    0.7183 |
|   11.0000 |                          0.8278 |                       0.8263 |               0.1731 |                               1.3523 |                            1.3418 |                    0.7793 |
|   12.0000 |                          0.8456 |                       0.8444 |               0.1503 |                               1.3866 |                            1.3766 |                    0.7182 |

Interpretation:
- Negative `decay_delta` means the temporal decay head has lower robust MAE.
- Positive `decay_gain_pct` means the temporal decay head improves over no-decay.