# 完整候选集合 STGNN + 节点时间影响热力图

## 一句话结论

这一版使用 HDF5 磁盘缓存支持完整候选节点集合，使用不区分上下游方向的 undirected distance graph propagation，并可加入节点-时间 impact heatmap 辅助监督。

测试集上，全候选邻域 robust MAE 从 `0.8328` 降到 `0.7579`，相对提升 `8.99%`。

只看受影响候选节点，robust MAE 从 `1.2938` 降到 `1.1646`，相对提升 `9.99%`。

## 关键变化

- 候选节点集合上限设为 36；主实验建议使用完整候选集合上限 36。
- 训练数据使用 HDF5 缓存，不再一次性把所有大张量放进内存。
- 新增节点-时间 impact heatmap 辅助目标，比事件级 severity/recovery/spread 更贴近模型输出。

## 实验设置

- 区域: Alameda, ContraCosta, Orange
- input_steps: 12
- horizon_steps: 12
- max_candidate_nodes: 36
- candidate_pm_radius: 5.0
- hidden_dim: 96
- graph_layers: 2
- graph_mode: undirected
- graph_sigma: 3.0
- heatmap_aux_weight: 0.0
- event_aux_weight: 0.05
- node_aux_weight: 0.03
- normal_model_dir: /Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_forecaster
- residual_beta: 1.00

## 样本数量

| split   |   samples |
|:--------|----------:|
| train   |    138528 |
| val     |     29210 |
| test    |     27499 |

## 预测指标

| split   |   all_candidates_model_robust_mae |   all_candidates_baseline_robust_mae |   all_candidates_improvement_pct |   affected_candidates_model_robust_mae |   affected_candidates_baseline_robust_mae |   affected_candidates_improvement_pct |   affected_node_rate |
|:--------|----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------------:|------------------------------------------:|--------------------------------------:|---------------------:|
| train   |                            0.7045 |                               0.7568 |                           6.9172 |                                 1.0520 |                                    1.1387 |                                7.6085 |               0.2275 |
| val     |                            0.7609 |                               0.8273 |                           8.0207 |                                 1.2082 |                                    1.3241 |                                8.7534 |               0.2343 |
| test    |                            0.7579 |                               0.8328 |                           8.9903 |                                 1.1646 |                                    1.2938 |                                9.9905 |               0.2498 |

## 各地区测试集指标

| region      |   samples |   all_candidates_model_robust_mae |   all_candidates_baseline_robust_mae |   all_candidates_improvement_pct |   affected_candidates_model_robust_mae |   affected_candidates_baseline_robust_mae |   affected_candidates_improvement_pct |   unaffected_candidates_model_robust_mae |   unaffected_candidates_baseline_robust_mae |   unaffected_candidates_improvement_pct |   affected_node_rate |
|:------------|----------:|----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------------:|------------------------------------------:|--------------------------------------:|-----------------------------------------:|--------------------------------------------:|----------------------------------------:|---------------------:|
| Alameda     |      9256 |                            0.7546 |                               0.8466 |                          10.8621 |                                 1.2037 |                                    1.3749 |                               12.4517 |                                   0.5496 |                                      0.6054 |                                  9.2149 |               0.2498 |
| ContraCosta |      4002 |                            0.7467 |                               0.8162 |                           8.5229 |                                 1.1234 |                                    1.2469 |                                9.9009 |                                   0.5715 |                                      0.6160 |                                  7.2261 |               0.2491 |
| Orange      |     14241 |                            0.7635 |                               0.8305 |                           8.0684 |                                 1.1564 |                                    1.2647 |                                8.5640 |                                   0.5757 |                                      0.6230 |                                  7.5877 |               0.2501 |

## 训练情况

- 最佳轮数 best_epoch: 5
- 选择指标 best_metric: val_loss
- 最佳验证损失 best_val_loss: 0.4654

## 仍然存在的限制

- 局部图仍然主要基于 signed postmile 距离，还没有融合完整路网拓扑。
- impact heatmap 辅助是否真的带来增益，需要继续跑 no-heatmap 消融验证。
- learned normal branch 在事故 cache 中使用候选子图近似 full-region normal STGNN，仍需和完整路网推理对照。
