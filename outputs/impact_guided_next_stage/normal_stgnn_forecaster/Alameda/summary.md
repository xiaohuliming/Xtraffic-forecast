# Normal STGNN Forecaster: Alameda

## Samples

| split   |   samples |
|:--------|----------:|
| train   |      3282 |
| val     |       602 |
| test    |       752 |

## Metrics

| split   |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:--------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| train   |      4.5807 |      5.5492 |               17.4530 |      12.8285 |      15.4498 |      16.7972 |      17.7560 |             0.6650 |             0.7796 |                  14.7008 |  42013385.0000 |
| val     |      4.9314 |      5.8991 |               16.4032 |      15.0726 |      17.6663 |      17.4613 |      18.2728 |             0.8064 |             0.9209 |                  12.4328 |   7794993.0000 |
| test    |      5.1335 |      5.9262 |               13.3761 |      14.7793 |      16.9381 |      20.3843 |      21.0609 |             0.8262 |             0.9297 |                  11.1366 |   9700733.0000 |

## Notes

- The blend baseline is the same transparent normal predictor used by the current residual branch.
- The learned model predicts a correction over that blend baseline, so zero correction recovers the baseline.
- Training and evaluation losses are masked by incident-free node-time labels.