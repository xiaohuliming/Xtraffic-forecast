# Learned Normal STGNN Summary

This run trains a learned normal-traffic branch on mostly non-incident windows.
The next integration step is to use these normal forecasts as the baseline for incident residual learning.

## Test Metrics

| region   |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:---------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| Alameda  |      9.5295 |      9.3022 |               -2.4427 |      24.5429 |      24.2993 |      23.6303 |      23.1599 |             1.3549 |             1.3122 |                  -3.2544 |    382549.0000 |

## Region Outputs

- Alameda: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_smoke/Alameda`