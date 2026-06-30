# Dual-Branch Gate Case Studies

- split: `test`
- selection: top `4` samples by affected MAE improvement over fixed gate=0.5

## Selected Cases

|   rank |   sample_idx |   region_code |   affected_nodes |   recovery_min |   learned_affected_mae |   fixed_affected_mae |   learned_vs_fixed_affected_gain |   gate_mean_affected |
|-------:|-------------:|--------------:|-----------------:|---------------:|-----------------------:|---------------------:|---------------------------------:|---------------------:|
| 1.0000 |  192208.0000 |        2.0000 |           2.0000 |       180.0000 |                 3.9825 |               5.4605 |                           1.4780 |               0.5014 |
| 2.0000 |   56226.0000 |        0.0000 |           2.0000 |       180.0000 |                 3.8845 |               5.3330 |                           1.4485 |               0.6267 |
| 3.0000 |  184513.0000 |        2.0000 |           5.0000 |       180.0000 |                 3.5177 |               4.9165 |                           1.3987 |               0.3773 |
| 4.0000 |  184542.0000 |        2.0000 |           5.0000 |       180.0000 |                 2.6550 |               4.0505 |                           1.3955 |               0.3682 |

## Figures

- `case_01_sample_192208.png`
- `case_02_sample_56226.png`
- `case_03_sample_184513.png`
- `case_04_sample_184542.png`

## Reading the Heatmaps

- The first strip marks affected candidate sensors.
- The gate heatmap shows the learned incident-branch weight. Larger values mean stronger reliance on the incident-graph residual branch.
- The target residual heatmap shows where the normal counterfactual forecast is most wrong.
- The fixed-minus-learned error heatmap is positive where learned gate improves over a fixed 0.5 fusion.
- The normal-minus-incident branch error heatmap is positive where the incident branch is locally more accurate.
