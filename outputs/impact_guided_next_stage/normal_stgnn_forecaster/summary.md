# Learned Normal STGNN Summary

This run trains a learned normal-traffic branch on mostly non-incident windows.
The next integration step is to use these normal forecasts as the baseline for incident residual learning.

## Test Metrics

| region      |   model_mae |   blend_mae |   mae_improvement_pct |   model_rmse |   blend_rmse |   model_mape |   blend_mape |   model_robust_mae |   blend_robust_mae |   robust_improvement_pct |   valid_values |
|:------------|------------:|------------:|----------------------:|-------------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-------------------------:|---------------:|
| Alameda     |      5.1335 |      5.9262 |               13.3761 |      14.7793 |      16.9381 |      20.3843 |      21.0609 |             0.8262 |             0.9297 |                  11.1366 |   9700733.0000 |
| ContraCosta |      5.7441 |      6.1948 |                7.2757 |      15.6801 |      16.8025 |      21.7657 |      20.7370 |             0.9214 |             0.9859 |                   6.5336 |  12884386.0000 |
| Orange      |      5.0972 |      5.9262 |               13.9880 |      14.6859 |      16.7712 |      19.4228 |      19.7069 |             0.7645 |             0.8607 |                  11.1839 |  31328844.0000 |

## Region Outputs

- Alameda: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_forecaster/Alameda`
- ContraCosta: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_forecaster/ContraCosta`
- Orange: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_forecaster/Orange`