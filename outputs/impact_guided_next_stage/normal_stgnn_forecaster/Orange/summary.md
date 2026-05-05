# Normal STGNN Forecaster: Orange

## Samples

| split   |   samples |
|:--------|----------:|
| train   |      5496 |
| val     |      1114 |
| test    |      1289 |

## Metrics

| split   |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:--------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| train   |      4.5758 |      5.6433 |               18.9162 |      13.4254 |      16.3184 |      17.1537 |      17.8787 |             0.6542 |             0.7784 |                  15.9630 | 133344028.0000 |
| val     |      4.7784 |      6.0097 |               20.4878 |      13.6988 |      16.9899 |      17.5677 |      18.3967 |             0.7479 |             0.8895 |                  15.9220 |  27326570.0000 |
| test    |      5.0972 |      5.9262 |               13.9880 |      14.6859 |      16.7712 |      19.4228 |      19.7069 |             0.7645 |             0.8607 |                  11.1839 |  31328844.0000 |

## Notes

- The blend baseline is the same transparent normal predictor used by the current residual branch.
- The learned model predicts a correction over that blend baseline, so zero correction recovers the baseline.
- Training and evaluation losses are masked by incident-free node-time labels.