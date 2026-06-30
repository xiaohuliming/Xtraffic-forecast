# Dual-Branch Gate Case Studies

- split: `test`
- selection: mixed cases with up to `2` success, neutral, and failure samples based on affected MAE gain over fixed gate=0.5

## Selected Cases

|   rank | category   |   sample_idx |   region_code |   affected_nodes |   recovery_min |   learned_affected_mae |   fixed_affected_mae |   learned_vs_fixed_affected_gain |   gate_mean_affected |
|-------:|:-----------|-------------:|--------------:|-----------------:|---------------:|-----------------------:|---------------------:|---------------------------------:|---------------------:|
|      1 | success    |       192208 |             2 |           2.0000 |       180.0000 |                 3.9825 |               5.4605 |                           1.4780 |               0.5014 |
|      2 | success    |        56226 |             0 |           2.0000 |       180.0000 |                 3.8845 |               5.3330 |                           1.4485 |               0.6267 |
|      3 | neutral    |       195028 |             2 |           3.0000 |         0.0000 |                 0.4735 |               0.4735 |                          -0.0000 |               0.3665 |
|      4 | neutral    |       187753 |             2 |           7.0000 |       155.0000 |                 0.6391 |               0.6391 |                           0.0000 |               0.4466 |
|      5 | failure    |        88134 |             1 |           1.0000 |        25.0000 |                 7.0882 |               4.9337 |                          -2.1545 |               0.5612 |
|      6 | failure    |        60576 |             0 |           2.0000 |        15.0000 |                 2.4142 |               1.0681 |                          -1.3461 |               0.4935 |

## Figures

- `case_01_success_sample_192208.png`
- `case_02_success_sample_56226.png`
- `case_03_neutral_sample_195028.png`
- `case_04_neutral_sample_187753.png`
- `case_05_failure_sample_88134.png`
- `case_06_failure_sample_60576.png`

## Reading the Heatmaps

- The first strip marks affected candidate sensors.
- The gate heatmap shows the learned incident-branch weight. Larger values mean stronger reliance on the incident-graph residual branch.
- The target residual heatmap shows where the normal counterfactual forecast is most wrong.
- The fixed-minus-learned error heatmap is positive where learned gate improves over a fixed 0.5 fusion.
- The normal-minus-incident branch error heatmap is positive where the incident branch is locally more accurate.
