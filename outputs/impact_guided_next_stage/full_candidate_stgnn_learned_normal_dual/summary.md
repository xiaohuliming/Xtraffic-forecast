# 完整候选集合 STGNN + 节点时间影响热力图

## 一句话结论

这一版使用 HDF5 磁盘缓存支持完整候选节点集合，使用不区分上下游方向的 undirected distance graph propagation，并可加入节点-时间 impact heatmap 辅助监督。

测试集上，全候选邻域 robust MAE 从 `0.8328` 降到 `0.7254`，相对提升 `12.90%`。

只看受影响候选节点，robust MAE 从 `1.2938` 降到 `1.1380`，相对提升 `12.04%`。

## 关键变化

- 候选节点集合上限设为 36；主实验建议使用完整候选集合上限 36。
- 训练数据使用 HDF5 缓存，不再一次性把所有大张量放进内存。
- 新增节点-时间 impact heatmap 辅助目标，比事件级 severity/recovery/spread 更贴近模型输出。
- 新增 normal_delta 输入：把 learned normal 相对统计 blend 的修正量作为 future known covariate。
- 新增 dual historical residual 输入：同时编码统计 residual 和 learned-normal historical residual。

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
- use_normal_delta: True
- use_dual_hist_residual: True
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
| train   |                            0.6711 |                               0.7568 |                          11.3206 |                                 1.0262 |                                    1.1387 |                                9.8745 |               0.2275 |
| val     |                            0.7300 |                               0.8273 |                          11.7611 |                                 1.1845 |                                    1.3241 |                               10.5389 |               0.2343 |
| test    |                            0.7254 |                               0.8328 |                          12.9028 |                                 1.1380 |                                    1.2938 |                               12.0442 |               0.2498 |

## 各地区测试集指标

| region      |   samples |   all_candidates_model_robust_mae |   all_candidates_baseline_robust_mae |   all_candidates_improvement_pct |   affected_candidates_model_robust_mae |   affected_candidates_baseline_robust_mae |   affected_candidates_improvement_pct |   unaffected_candidates_model_robust_mae |   unaffected_candidates_baseline_robust_mae |   unaffected_candidates_improvement_pct |   affected_node_rate |
|:------------|----------:|----------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------------:|------------------------------------------:|--------------------------------------:|-----------------------------------------:|--------------------------------------------:|----------------------------------------:|---------------------:|
| Alameda     |      9256 |                            0.7226 |                               0.8466 |                          14.6391 |                                 1.1690 |                                    1.3749 |                               14.9747 |                                   0.5189 |                                      0.6054 |                                 14.2912 |               0.2498 |
| ContraCosta |      4002 |                            0.7191 |                               0.8162 |                          11.8971 |                                 1.1027 |                                    1.2469 |                               11.5656 |                                   0.5408 |                                      0.6160 |                                 12.2090 |               0.2491 |
| Orange      |     14241 |                            0.7289 |                               0.8305 |                          12.2296 |                                 1.1324 |                                    1.2647 |                               10.4620 |                                   0.5361 |                                      0.6230 |                                 13.9445 |               0.2501 |

## 训练情况

- 最佳轮数 best_epoch: 5
- 选择指标 best_metric: val_loss
- 最佳验证损失 best_val_loss: 0.4412

## 仍然存在的限制

- 局部图仍然主要基于 signed postmile 距离，还没有融合完整路网拓扑。
- impact heatmap 辅助是否真的带来增益，需要继续跑 no-heatmap 消融验证。
- learned normal branch 在事故 cache 中使用候选子图近似 full-region normal STGNN，仍需和完整路网推理对照。
