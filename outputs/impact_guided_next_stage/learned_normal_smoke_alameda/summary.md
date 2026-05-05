# 完整候选集合 STGNN + 节点时间影响热力图

## 一句话结论

这一版使用 HDF5 磁盘缓存支持完整候选节点集合，使用不区分上下游方向的 undirected distance graph propagation，并可加入节点-时间 impact heatmap 辅助监督。

测试集上，全候选邻域 robust MAE 从 `0.8370` 降到 `0.8376`，相对提升 `-0.08%`。

只看受影响候选节点，robust MAE 从 `1.2929` 降到 `1.2941`，相对提升 `-0.09%`。

## 关键变化

- 候选节点集合上限设为 16；主实验建议使用完整候选集合上限 36。
- 训练数据使用 HDF5 缓存，不再一次性把所有大张量放进内存。
- 新增节点-时间 impact heatmap 辅助目标，比事件级 severity/recovery/spread 更贴近模型输出。

## 实验设置

- 区域: Alameda
- input_steps: 12
- horizon_steps: 12
- max_candidate_nodes: 16
- candidate_pm_radius: 5.0
- hidden_dim: 48
- graph_layers: 1
- graph_mode: undirected
- graph_sigma: 1.0
- heatmap_aux_weight: 0.0
- event_aux_weight: 0.02
- node_aux_weight: 0.02
- normal_model_dir: /Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/normal_stgnn_forecaster
- residual_beta: 1.00

## 样本数量

| split   |   samples |
|:--------|----------:|
| train   |       192 |
| val     |       192 |
| test    |       192 |

## 预测指标

| split   |   all_candidates_model_robust_mae |   all_candidates_baseline_robust_mae |   all_candidates_improvement_pct |   affected_candidates_model_robust_mae |   affected_candidates_baseline_robust_mae |   affected_candidates_improvement_pct |   affected_node_rate |
|:--------|----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------------:|------------------------------------------:|--------------------------------------:|---------------------:|
| train   |                            0.8685 |                               0.8718 |                           0.3724 |                                 1.1459 |                                    1.1498 |                                0.3453 |               0.3354 |
| val     |                            0.9696 |                               0.9703 |                           0.0740 |                                 1.6068 |                                    1.6070 |                                0.0110 |               0.2838 |
| test    |                            0.8376 |                               0.8370 |                          -0.0795 |                                 1.2941 |                                    1.2929 |                               -0.0914 |               0.2447 |

## 各地区测试集指标

| region   |   samples |   all_candidates_model_robust_mae |   all_candidates_baseline_robust_mae |   all_candidates_improvement_pct |   affected_candidates_model_robust_mae |   affected_candidates_baseline_robust_mae |   affected_candidates_improvement_pct |   unaffected_candidates_model_robust_mae |   unaffected_candidates_baseline_robust_mae |   unaffected_candidates_improvement_pct |   affected_node_rate |
|:---------|----------:|----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------------:|------------------------------------------:|--------------------------------------:|-----------------------------------------:|--------------------------------------------:|----------------------------------------:|---------------------:|
| Alameda  |       192 |                            0.8376 |                               0.8370 |                          -0.0795 |                                 1.2941 |                                    1.2929 |                               -0.0914 |                                   0.6548 |                                      0.6544 |                                 -0.0701 |               0.2447 |

## 训练情况

- 最佳轮数 best_epoch: 1
- 选择指标 best_metric: val_loss
- 最佳验证损失 best_val_loss: 0.6547

## 仍然存在的限制

- 局部图仍然主要基于 signed postmile 距离，还没有融合完整路网拓扑。
- impact heatmap 辅助是否真的带来增益，需要继续跑 no-heatmap 消融验证。
- learned normal branch 在事故 cache 中使用候选子图近似 full-region normal STGNN，仍需和完整路网推理对照。
