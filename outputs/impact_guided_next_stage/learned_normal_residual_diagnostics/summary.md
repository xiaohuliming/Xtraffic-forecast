# Learned Normal Residual Cache Diagnostic

## 一句话结论

learned normal cache 确实降低了事故窗口 residual target 的整体幅度，test 全候选 mean |residual| 从 0.8735 降到 0.8328，下降 4.66%。

但历史输入 `hist_residual` 与旧 cache 完全一致，说明当前 residual branch 的输入仍是统计 baseline residual，而预测目标已经变成 learned-normal residual；这支持“输入/目标不对齐”这个诊断。

## Cache 一致性检查

|   old_samples |   new_samples | same_sample_count   |   split_mismatch_count |   region_mismatch_count |   node_valid_mismatch_count |   node_affected_mismatch_count |   hist_residual_max_abs_diff |   hist_residual_mean_abs_diff |
|--------------:|--------------:|:--------------------|-----------------------:|------------------------:|----------------------------:|-------------------------------:|-----------------------------:|------------------------------:|
|        195237 |        195237 | True                |                      0 |                       0 |                           0 |                              0 |                     0.000000 |                      0.000000 |

## Test Split 关键分布

| split   | node_class   |    count |   old_abs_mean |   new_abs_mean |   target_reduction_pct |   normal_delta_abs_mean |   normal_delta_to_old_abs_pct |   new_abs_lower_rate_pct |   sign_flip_rate_pct |   old_abs_p95 |   new_abs_p95 |
|:--------|:-------------|---------:|---------------:|---------------:|-----------------------:|------------------------:|------------------------------:|-------------------------:|---------------------:|--------------:|--------------:|
| test    | affected     |  4641126 |         1.3888 |         1.2938 |                 6.8353 |                  0.4282 |                       30.8343 |                  55.9517 |               8.1490 |        4.4500 |        4.1500 |
| test    | all          | 14533132 |         0.8735 |         0.8328 |                 4.6612 |                  0.3581 |                       40.9889 |                  53.3521 |              10.2670 |        2.5500 |        2.4500 |
| test    | unaffected   |  9892006 |         0.6318 |         0.6165 |                 2.4190 |                  0.3251 |                       51.4616 |                  52.1325 |              11.2607 |        1.8500 |        1.8500 |

## Region 对比

| region      | node_class   |   old_abs_mean |   new_abs_mean |   target_reduction_pct |   normal_delta_abs_mean |   sign_flip_rate_pct |
|:------------|:-------------|---------------:|---------------:|-----------------------:|------------------------:|---------------------:|
| Alameda     | affected     |         1.4750 |         1.3749 |                 6.7811 |                  0.4120 |               7.4211 |
| Alameda     | all          |         0.8917 |         0.8466 |                 5.0660 |                  0.3279 |               9.3257 |
| ContraCosta | affected     |         1.3481 |         1.2469 |                 7.5088 |                  0.4347 |               8.0105 |
| ContraCosta | all          |         0.8677 |         0.8162 |                 5.9357 |                  0.3483 |               9.5053 |
| Orange      | affected     |         1.3548 |         1.2647 |                 6.6546 |                  0.4350 |               8.5886 |
| Orange      | all          |         0.8652 |         0.8305 |                 4.0136 |                  0.3781 |              11.0414 |

## Channel 对比

| channel   | node_class   |   old_abs_mean |   new_abs_mean |   target_reduction_pct |   normal_delta_abs_mean |   sign_flip_rate_pct |
|:----------|:-------------|---------------:|---------------:|-----------------------:|------------------------:|---------------------:|
| flow      | affected     |         1.4694 |         1.3548 |                 7.8023 |                  0.5333 |               9.0541 |
| flow      | all          |         0.9912 |         0.9529 |                 3.8560 |                  0.4658 |              11.8954 |
| occupancy | affected     |         1.5729 |         1.4680 |                 6.6695 |                  0.4663 |               9.3393 |
| occupancy | all          |         0.9325 |         0.8752 |                 6.1388 |                  0.3778 |              12.0666 |
| speed     | affected     |         1.0723 |         1.0129 |                 5.5309 |                  0.2565 |               5.6413 |
| speed     | all          |         0.6476 |         0.6239 |                 3.6629 |                  0.1930 |               5.9330 |

## Horizon 对比

|   horizon | node_class   |   old_abs_mean |   new_abs_mean |   target_reduction_pct |   normal_delta_abs_mean |
|----------:|:-------------|---------------:|---------------:|-----------------------:|------------------------:|
|         1 | affected     |         0.8414 |         0.8144 |                 3.2064 |                  0.2715 |
|         1 | all          |         0.5946 |         0.5912 |                 0.5612 |                  0.2313 |
|         2 | affected     |         1.0096 |         0.9615 |                 4.7675 |                  0.3116 |
|         2 | all          |         0.6812 |         0.6647 |                 2.4185 |                  0.2603 |
|         3 | affected     |         1.1422 |         1.0793 |                 5.5104 |                  0.3618 |
|         3 | all          |         0.7494 |         0.7256 |                 3.1771 |                  0.2985 |
|         4 | affected     |         1.2450 |         1.1694 |                 6.0696 |                  0.3973 |
|         4 | all          |         0.8019 |         0.7720 |                 3.7289 |                  0.3258 |
|         5 | affected     |         1.3289 |         1.2415 |                 6.5788 |                  0.4178 |
|         5 | all          |         0.8444 |         0.8070 |                 4.4355 |                  0.3441 |
|         6 | affected     |         1.4076 |         1.3077 |                 7.1003 |                  0.4306 |
|         6 | all          |         0.8833 |         0.8396 |                 4.9450 |                  0.3610 |
|         7 | affected     |         1.4783 |         1.3689 |                 7.3985 |                  0.4422 |
|         7 | all          |         0.9199 |         0.8704 |                 5.3876 |                  0.3743 |
|         8 | affected     |         1.5426 |         1.4254 |                 7.5982 |                  0.4636 |
|         8 | all          |         0.9519 |         0.8983 |                 5.6325 |                  0.3908 |
|         9 | affected     |         1.5990 |         1.4758 |                 7.7057 |                  0.4889 |
|         9 | all          |         0.9808 |         0.9251 |                 5.6787 |                  0.4087 |
|        10 | affected     |         1.6517 |         1.5255 |                 7.6428 |                  0.5064 |
|        10 | all          |         1.0053 |         0.9489 |                 5.6088 |                  0.4232 |
|        11 | affected     |         1.6884 |         1.5590 |                 7.6640 |                  0.5236 |
|        11 | all          |         1.0248 |         0.9667 |                 5.6750 |                  0.4387 |
|        12 | affected     |         1.7312 |         1.5984 |                 7.6700 |                  0.5235 |
|        12 | all          |         1.0460 |         0.9853 |                 5.7996 |                  0.4402 |

## 输入目标对齐

| split   | node_class   |   hist_last_abs_mean |   new_cache_hist_last_abs_mean |   old_target_step1_abs_mean |   new_target_step1_abs_mean |   hist_to_new_step1_ratio |   hist_cache_abs_diff |
|:--------|:-------------|---------------------:|-------------------------------:|----------------------------:|----------------------------:|--------------------------:|----------------------:|
| test    | all          |               0.8726 |                         0.8726 |                      0.5946 |                      0.5912 |                    1.4759 |                0.0000 |
| test    | affected     |               1.9872 |                         1.9872 |                      0.8414 |                      0.8144 |                    2.4400 |                0.0000 |
| test    | unaffected   |               0.5014 |                         0.5014 |                      0.4789 |                      0.4866 |                    1.0303 |                0.0000 |

## 模型指标对照

| split   | metric                                    |   old_statistical_normal |   new_learned_normal |   new_vs_old_reduction_pct |
|:--------|:------------------------------------------|-------------------------:|---------------------:|---------------------------:|
| test    | all_candidates_baseline_robust_mae        |                   0.8735 |               0.8328 |                     4.6612 |
| test    | all_candidates_model_robust_mae           |                   0.7378 |               0.7579 |                    -2.7300 |
| test    | affected_candidates_baseline_robust_mae   |                   1.3888 |               1.2938 |                     6.8353 |
| test    | affected_candidates_model_robust_mae      |                   1.1659 |               1.1646 |                     0.1149 |
| test    | unaffected_candidates_baseline_robust_mae |                   0.6318 |               0.6165 |                     2.4190 |
| test    | unaffected_candidates_model_robust_mae    |                   0.5369 |               0.5672 |                    -5.6284 |

## 解释

- 全候选 target 下降 4.66%，受影响节点下降 6.84%，非受影响节点下降 2.42%。
- 这说明 learned normal 没有只修正常规节点；它也吸收了一部分事故相关偏差，使 residual target 更小。
- 当前模型输入的历史 residual 仍然来自统计 baseline，因此模型看到的历史异常尺度和未来 target 的尺度不一致。
- 下一步应把 learned-normal 信息显式加入 residual branch 输入，例如 `normal_delta = Y_normal_stgnn - Y_blend` 和 normal uncertainty proxy。

## 输出文件

- `by_split_nodeclass.csv`
- `by_region_nodeclass.csv`
- `by_channel_nodeclass.csv`
- `by_horizon_nodeclass.csv`
- `input_target_alignment.csv`
- `model_metric_comparison.csv`
- `plots/test_region_target_magnitude.png`
- `plots/test_horizon_target_magnitude.png`
- `plots/test_channel_target_magnitude.png`

## Inputs

- old_cache: `outputs/full_candidate_stgnn_heatmap_model/first_pass/full_candidate_samples.h5`
- new_cache: `outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal/full_candidate_samples.h5`