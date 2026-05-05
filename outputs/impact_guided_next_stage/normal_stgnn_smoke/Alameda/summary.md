# Normal STGNN Forecaster: Alameda

## Samples

| split   |   samples |
|:--------|----------:|
| train   |        64 |
| val     |        23 |
| test    |        30 |

## Metrics

| split   |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:--------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| train   |      3.4674 |      3.5135 |                1.3128 |      10.0704 |      10.2622 |      18.3699 |      17.3652 |             0.6338 |             0.6264 |                  -1.1859 |    821765.0000 |
| val     |      4.4575 |      4.6100 |                3.3078 |      13.6861 |      13.9796 |      21.3145 |      21.2896 |             0.8416 |             0.8544 |                   1.4977 |    301631.0000 |
| test    |      9.5295 |      9.3022 |               -2.4427 |      24.5429 |      24.2993 |      23.6303 |      23.1599 |             1.3549 |             1.3122 |                  -3.2544 |    382549.0000 |

## Notes

- The blend baseline is the same transparent normal predictor used by the current residual branch.
- The learned model predicts a correction over that blend baseline, so zero correction recovers the baseline.
- Training and evaluation losses are masked by incident-free node-time labels.