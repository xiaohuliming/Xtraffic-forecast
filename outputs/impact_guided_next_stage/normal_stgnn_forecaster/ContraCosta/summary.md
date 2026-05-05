# Normal STGNN Forecaster: ContraCosta

## Samples

| split   |   samples |
|:--------|----------:|
| train   |      4689 |
| val     |       754 |
| test    |      1033 |

## Metrics

| split   |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:--------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| train   |      4.8688 |      5.5364 |               12.0594 |      13.2048 |      15.0518 |      17.7615 |      17.3956 |             0.6771 |             0.7512 |                   9.8682 |  58435916.0000 |
| val     |      5.6787 |      6.4565 |               12.0474 |      14.8966 |      16.9388 |      17.5051 |      17.3316 |             0.9419 |             1.0397 |                   9.4037 |   9407176.0000 |
| test    |      5.7441 |      6.1948 |                7.2757 |      15.6801 |      16.8025 |      21.7657 |      20.7370 |             0.9214 |             0.9859 |                   6.5336 |  12884386.0000 |

## Notes

- The blend baseline is the same transparent normal predictor used by the current residual branch.
- The learned model predicts a correction over that blend baseline, so zero correction recovers the baseline.
- Training and evaluation losses are masked by incident-free node-time labels.