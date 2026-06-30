# Table 8. Mixed gate case studies

| Rank | Category | Sample index | Region code | Affected nodes | Recovery min | Learned affected MAE | Fixed gate affected MAE | Gain over fixed | Normal-only affected MAE | Incident-only affected MAE | Mean affected gate |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | success | 192208 | 2 | 2 | 180.0 | 3.9825 | 5.4605 | 1.4780 | 6.0167 | 5.4870 | 0.5014 |
| 2 | success | 56226 | 0 | 2 | 180.0 | 3.8845 | 5.3330 | 1.4485 | 5.2091 | 5.9619 | 0.6267 |
| 3 | neutral | 195028 | 2 | 3 | 0.0 | 0.4735 | 0.4735 | -0.0000 | 0.4866 | 0.4844 | 0.3665 |
| 4 | neutral | 187753 | 2 | 7 | 155.0 | 0.6391 | 0.6391 | 0.0000 | 0.6765 | 0.7222 | 0.4466 |
| 5 | failure | 88134 | 1 | 1 | 25.0 | 7.0882 | 4.9337 | -2.1545 | 1.9051 | 9.6011 | 0.5612 |
| 6 | failure | 60576 | 0 | 2 | 15.0 | 2.4142 | 1.0681 | -1.3461 | 2.9640 | 3.5559 | 0.4935 |

The mixed set selects success, neutral, and failure samples from the same full test split according to affected-candidate MAE gain over fixed gate = 0.5. This table is intended to expose both the useful behavior and the boundary cases of the learned gate.
