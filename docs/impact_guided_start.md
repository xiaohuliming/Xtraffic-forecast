# 事故影响引导交通流预测启动文档

## 1. 研究问题

当前研究不再把核心任务设为事故类型识别，而是设为：

```text
给定历史交通状态、道路/传感器属性和事故记录，模型预测未来交通状态，
同时显式学习事故对交通状态造成的影响强度、持续时间、空间扩散和上下游差异。
```

推荐题目：

```text
Impact-Guided Spatio-Temporal Traffic Forecasting under Incidents
```

或更强调模型结构：

```text
Normal-Impact Decomposed STGNN for Incident-Affected Traffic Forecasting
```

## 2. 核心判断

XTraffic 原文说明，用交通状态反推事故类型的准确率并不高。这不代表交通状态中没有事故信号，而是说明事故类型不是交通影响的好监督标签。

同一类事故可能几乎不影响交通，也可能造成长时间拥堵；不同事故类型也可能产生相似的流量下降、速度下降和占有率上升。因此，更稳的路线是让模型学习事故造成的实际影响。

当前文件夹中的实验已经支持这个判断：

- 事故类型对 severity / recovery / spread 的解释力很弱。
- 影响标签与事故窗口预测误差高度相关。
- 事故 residual learning 能稳定降低事故窗口预测误差。
- 完整候选集合 STGNN residual model 已经比 MLP 和截断候选集合更强。

## 3. 当前最稳贡献

目前最稳的论文主线是：

```text
事故影响应建模为事故中心候选节点图上的时空残差传播。
```

可以稳定写的贡献：

1. 构造事故影响标签，把事故从离散类型转化为 severity、recovery、spread、directionality 和 node-time heatmap。
2. 用 normal forecast + incident residual 的方式解耦常规交通模式和事故扰动。
3. 在不使用标签挑选 top-k 受影响节点的情况下，基于事故位置构造完整候选节点图，并用 STGNN 学习事故 residual propagation。
4. 证明影响标签比事故类型更能解释预测误差，支持从 incident type learning 转向 incident impact learning。

暂时不要作为强贡献写的点：

- 当前 heatmap auxiliary supervision 没有明显提升预测。
- 当前 event-level severity / recovery / spread auxiliary supervision 增益也不稳定。
- 简单 signed-postmile directional graph 没有明显优于 undirected graph。

这些可以作为探索结果或后续改进，而不是主贡献。

## 4. 模型草图

整体形式：

```text
Y_hat = Y_normal + Delta_incident
```

### 4.1 Normal Branch

目标是学习无事故或弱事故情况下的常规交通状态：

- 周期性：小时、星期、通勤高峰。
- 趋势性：日内变化。
- 常规空间依赖：相邻路段、上下游传播。

当前原型里使用的是统计 normal baseline + last-observation blend。下一阶段应替换为更强 normal backbone，例如 GraphWaveNet、D2STGNN，或先实现一个轻量 normal STGNN。

### 4.2 Incident Impact Branch

目标是学习事故造成的未来残差：

```text
Delta_incident = Y_actual - Y_normal
```

输入包括：

- 历史 residual 序列。
- 事故上下文：类型、描述、持续时间、时间位置、hour/day 特征。
- 候选节点上下文：signed postmile、距离、anchor 标记、valid mask、affected supervision。
- 候选节点图：同 freeway / direction / postmile radius 内的传感器集合。

旧的统计 normal baseline 最好原型是：

```text
outputs/full_candidate_stgnn_heatmap_model/ablation_sigma_3_00_undirected
```

测试集结果：

- 全候选 robust MAE: 0.8735 -> 0.7378，提升 15.54%。
- 受影响节点 robust MAE: 1.3888 -> 1.1659，提升 16.05%。

当前 learned normal 路线的主模型原型是：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual
```

测试集结果：

- 全候选 robust MAE: 0.8328 -> 0.7254，提升 12.90%。
- 受影响节点 robust MAE: 1.2938 -> 1.1380，提升 12.04%。
- 相比旧统计 baseline best，全候选从 0.7378 进一步降到 0.7254，受影响节点从 1.1659 降到 1.1380。

在此基础上，加入 `abs(normal_delta)` 作为 normal branch disagreement / uncertainty proxy 后，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty
```

测试集结果：

- 全候选 robust MAE: 0.8328 -> 0.7248，提升 12.97%，略优于 dual historical residual 的 0.7254。
- 受影响节点 robust MAE: 1.2938 -> 1.1381，和 dual historical residual 的 1.1380 基本持平。
- 结论：disagreement proxy 是一个轻量小幅增强，但主要收益仍来自 `normal_delta` 和 dual historical residual 的输入对齐。

### 4.3 Impact Supervision

保留以下监督信号：

- event severity。
- recovery time。
- spread nodes / spread pm。
- upstream/downstream directionality。
- node affected。
- node-time impact heatmap。

但下一阶段要把辅助监督改得更贴近模型输出，优先考虑：

```text
节点-时间 residual curve supervision
```

而不是只用事件级 severity / recovery / spread。

### 4.4 Temporal Decay

北航 IGSTGNN 的 TIID 使用固定 Gaussian decay：

```text
omega_tau = exp(-tau^2 / (2 sigma_t^2))
```

我们可以借鉴“显式时间衰减”的思想，但不要照搬固定高斯。XTraffic 中事故恢复时间有长尾，建议后续改为：

- mixture Gaussian decay。
- exponential + long-tail decay。
- GRU decay conditioned on incident context。
- recovery survival head。

## 5. 下一阶段实验顺序

### Stage A: 固定当前证据链

目标：把当前结果变成可复现、可汇报的一组实验。

需要保留：

- impact label quality report。
- forecast error vs impact validation。
- candidate MLP vs candidate STGNN。
- full-candidate STGNN 消融。

### Stage B: 固定评估集合的候选半径实验

当前 radius 消融会改变参与评估的节点集合，因此只能看趋势。下一步应做 fixed evaluation set：

- 统一用最大半径构造评估集合。
- 比较 2.5 / 5.0 / 7.5 radius 的训练输入。
- 在同一批事件和节点上计算指标。

这一步能验证“更远候选节点是否真的有用”。

### Stage C: 强 normal backbone

当前 normal baseline 是统计型，论文模型应替换成 learned normal forecaster。

已经开始实现轻量版本：

- 用非事故窗口训练 normal STGNN。
- 以统计 blend baseline 为起点，学习正常交通 correction。
- 输出未来 12 步 flow / occupancy / speed 的 normal forecast。
- 在事故窗口中冻结 normal branch，仅训练 incident residual branch。

当前入口：

```bash
bash scripts/run_impact_guided_next_stage.sh normal-smoke
bash scripts/run_impact_guided_next_stage.sh normal
```

对应脚本：

```text
scripts/train_normal_stgnn_forecaster.py
```

`normal-smoke` 已验证 Alameda 小样本链路能跑通，输出位于：

```text
outputs/impact_guided_next_stage/normal_stgnn_smoke
```

`normal` 正式三区域实验也已跑通，输出位于：

```text
outputs/impact_guided_next_stage/normal_stgnn_forecaster
```

按 test 集有效值加权后，learned normal STGNN 相比统计 blend baseline：

- MAE: 5.9904 -> 5.2583，提升 12.22%。
- robust MAE: 0.9030 -> 0.8131，提升 9.96%。

分地区 test robust MAE 提升：

- Alameda: 0.9297 -> 0.8262，提升 11.14%。
- ContraCosta: 0.9859 -> 0.9214，提升 6.53%。
- Orange: 0.8607 -> 0.7645，提升 11.18%。

下一步已经在 full-candidate residual 脚本中加入可选入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-smoke
bash scripts/run_impact_guided_next_stage.sh learned-normal-full
```

该入口会用 learned normal branch 重建事故 residual cache，使 residual target 从 `Y_actual - Y_blend` 变为 `Y_actual - Y_normal_stgnn`。当前实现为了计算可承受，先在事故候选子图上复用 normal STGNN 权重，是一个可跑通的近似版本；后续可再和 full-region normal 推理对照。

`learned-normal-smoke` 已验证 Alameda 小样本链路能跑通，输出位于：

```text
outputs/impact_guided_next_stage/learned_normal_smoke_alameda
```

该 smoke 使用 train / val / test 各 192 个事故样本，只训练 1 个 epoch；结果基本持平，不能作为性能结论，只说明 learned-normal cache 重建、residual target 重算、STGNN 训练和报告生成链路都已经打通。

`learned-normal-full` 正式三区域实验已跑通，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal
```

该实验使用 learned normal STGNN 重建全量事故 residual cache，共 195237 个样本：

- train: 138528
- val: 29210
- test: 27499

测试集结果：

- 全候选 normal baseline robust MAE: 0.8328。
- 全候选 residual model robust MAE: 0.7579，较 learned-normal baseline 提升 8.99%。
- 受影响节点 normal baseline robust MAE: 1.2938。
- 受影响节点 residual model robust MAE: 1.1646，较 learned-normal baseline 提升 9.99%。

和旧的统计 normal baseline full-candidate 最优结果相比：

- 全候选 baseline 更好：0.8735 -> 0.8328，说明 learned normal branch 确实改善了事故窗口的 counterfactual normal forecast。
- 全候选最终 model 略差：0.7378 -> 0.7579，说明 residual branch 在 learned normal 下可学习残差变小、分布改变后，当前 residual 结构未完全吃到 normal 分支收益。
- 受影响节点最终 model 基本持平略好：1.1659 -> 1.1646，说明 learned normal 对真正受事故影响节点没有破坏主结论。

当前判断：learned normal branch 可以作为论文方向的一部分，但主结果仍应保留统计 baseline full-candidate STGNN 作为当前 best model；learned-normal full 更适合作为“更强 normal branch 的初步探索”和后续改进入口。下一步要做的是让 residual branch 输入与 learned normal 对齐，例如加入 learned-normal 历史 residual、normal uncertainty，或端到端联合微调。

随后已完成 residual cache 诊断，输出位于：

```text
outputs/impact_guided_next_stage/learned_normal_residual_diagnostics
```

诊断命令：

```bash
bash scripts/run_impact_guided_next_stage.sh diagnostics
```

关键发现：

- 两个 cache 样本完全对齐：split、region、node_valid、node_affected 全部 0 mismatch。
- 历史输入 `hist_residual` 完全一致：max abs diff = 0，mean abs diff = 0。
- 但 future target 已经改变：test 全候选 mean |residual| 从 0.8735 降到 0.8328，下降 4.66%。
- 受影响节点 target 下降更明显：1.3888 -> 1.2938，下降 6.84%。
- 非受影响节点 target 也下降：0.6318 -> 0.6165，下降 2.42%。
- 输入目标尺度不匹配非常明显：test affected 节点上，历史最后一步 |residual| 平均为 1.9872，而 learned-normal 未来第 1 步 target 平均为 0.8144，比例约 2.44。

解释：learned normal branch 不只是修正常规交通，也吸收了一部分事故相关偏差；同时 residual branch 的历史输入仍是统计 baseline residual，导致输入异常尺度和预测目标尺度不一致。这解释了为什么 learned-normal baseline 更好，但 residual model 的全候选最终指标没有超过旧 best。

下一步优先改造 residual branch 输入，而不是继续盲目加大模型：

- 加入 `normal_delta = Y_normal_stgnn - Y_blend`，让模型知道 normal branch 在哪些节点和 horizon 做了修正。
- 加入 normal uncertainty proxy，例如 learned normal correction magnitude、历史 correction variance、或 blend-vs-learned disagreement。
- 将历史输入从统计 residual 改为 learned-normal historical residual，或同时保留两路 residual。

`normal_delta` 输入增强已实现并完成正式三区域实验，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_delta
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-delta-smoke
bash scripts/run_impact_guided_next_stage.sh learned-normal-delta-full
```

实现方式：

- cache 新增 `normal_delta` dataset，形状为 `[horizon, candidate_node, channel]`。
- 数值定义为 `(Y_normal_stgnn - Y_blend) / scale`。
- residual STGNN 将每个候选节点的 `normal_delta` flatten 后拼接到 node hidden 上，作为 future known covariate 预测事故 residual。

正式 test 结果：

- learned-normal baseline 全候选 robust MAE: 0.8328。
- learned-normal + normal_delta model 全候选 robust MAE: 0.7434，较 learned-normal model 的 0.7579 进一步下降 1.92%。
- learned-normal + normal_delta 受影响节点 robust MAE: 1.1620，优于 learned-normal model 的 1.1646，也略优于旧统计 baseline best 的 1.1659。
- 和旧统计 baseline best 相比，全候选仍略差：0.7378 -> 0.7434，差约 0.76%；但受影响节点已经略好：1.1659 -> 1.1620。

当前更新判断：`normal_delta` 验证了“输入目标对齐”方向是有效的。它基本补回了 learned normal 分支带来的全候选退化，并让真正受事故影响节点达到阶段性最好结果。下一步应继续围绕 learned normal 信息增强 residual branch，而不是简单加深 STGNN。

随后已完成 learned-normal historical residual 双路输入实验，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-dual-smoke
bash scripts/run_impact_guided_next_stage.sh learned-normal-dual-full
```

实现方式：

- cache 新增 `hist_residual_normal` dataset，形状为 `[input_steps, candidate_node, channel]`。
- 数值定义为历史窗口中的 `(Y_actual - Y_normal_stgnn) / scale`。
- residual STGNN 同时编码统计 residual 和 learned-normal historical residual，并继续保留 `normal_delta` future known covariate。

正式 test 结果：

- learned-normal baseline 全候选 robust MAE: 0.8328。
- learned-normal + normal_delta + dual historical residual model 全候选 robust MAE: 0.7254，较 normal_delta 版的 0.7434 继续下降 2.43%。
- 受影响节点 robust MAE: 1.1380，较 normal_delta 版的 1.1620 继续下降 2.07%。
- 相比旧统计 baseline best，全候选 0.7378 -> 0.7254，受影响节点 1.1659 -> 1.1380，说明 learned normal branch 经过输入对齐后已经成为当前最佳主线。

`normal_delta` disagreement proxy 也已完成实验，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_uncertainty
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-smoke
bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-full
```

实现方式：

- 不重建 cache，复用 dual historical residual cache。
- 在 signed `normal_delta` 之外，额外拼接 `abs(normal_delta)`，表示 learned normal 与统计 blend 的 disagreement 强度。

正式 test 结果：

- 全候选 robust MAE: 0.7254 -> 0.7248，小幅改善。
- 受影响节点 robust MAE: 1.1380 -> 1.1381，基本持平。
- 结论：可以在最终表中作为轻量增强或 uncertainty proxy 消融，但不要把它作为主要创新点。

当前 learned-normal 消融表已生成：

```text
outputs/impact_guided_next_stage/ablation_summary/learned_normal_ablation_table.md
outputs/impact_guided_next_stage/ablation_summary/learned_normal_ablation_table.csv
```

生成脚本：

```text
scripts/summarize_learned_normal_ablation.py
```

seed robustness 也已完成，使用 seed 7 / 11 / 23，输出位于：

```text
outputs/impact_guided_next_stage/seed_robustness/seed_robustness_summary.md
outputs/impact_guided_next_stage/seed_robustness/seed_robustness_runs.csv
outputs/impact_guided_next_stage/seed_robustness/seed_robustness_summary.csv
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-uncertainty-seeds
python3 scripts/summarize_seed_robustness.py
```

三组 seed 测试集结果：

- seed 7: 全候选 0.7248，受影响节点 1.1381。
- seed 11: 全候选 0.7268，受影响节点 1.1408。
- seed 23: 全候选 0.7250，受影响节点 1.1382。

均值和标准差：

- 全候选 robust MAE: 0.7255 ± 0.0011。
- 受影响节点 robust MAE: 1.1391 ± 0.0015。
- 非受影响节点 robust MAE: 0.5315 ± 0.0009。

结论：当前 learned normal + normal_delta + dual history + disagreement proxy 结果对随机种子非常稳定；相较旧统计 normal best 的 0.7378 / 1.1659，优势不是单 seed 偶然。

full-region normal 推理对照也已完成，用来验证当前“候选子图复用 normal STGNN 权重”的近似误差。输出位于：

```text
outputs/impact_guided_next_stage/full_region_normal_diagnostics/summary.md
outputs/impact_guided_next_stage/full_region_normal_diagnostics/full_region_normal_diagnostics.csv
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-fullregion-diagnostics
```

同时，full-region normal cache 构建链路已通过小样本 smoke：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-fullregion-smoke
```

实现方式：

- 在 residual cache 脚本中新增 `--normal-inference-scope {local, full}`。
- `local` 是当前主实验做法：只在事故候选子图上复用 normal STGNN 权重。
- `full` 是对照做法：先在完整区域图上跑 normal STGNN，再把预测切回事故候选节点。

诊断采样设置：

- 三个区域 Alameda / ContraCosta / Orange。
- 每个区域 train / val / test 各采样 256 个事故窗口，共每区 768 个样本。
- 比较 local normal prediction 和 full-region normal prediction 在同一候选节点上的差异。

weighted 结果：

- 全候选 normal prediction 差异 robust MAE: 0.0626。
- 受影响节点 normal prediction 差异 robust MAE: 0.0761。
- 全候选 residual target 从 0.8868 降到 0.8808，只变化 -0.68%。
- 受影响节点 residual target 从 1.4056 降到 1.4008，只变化 -0.34%。

结论：full-region normal inference 确实和候选子图近似不完全相同，但差异较小，不足以解释当前主模型的大幅提升。当前论文主实验可以继续使用 local candidate-subgraph normal inference；full-region normal 更适合作为方法严谨性的对照或附录实验，而不是必须立刻重跑全量 195k cache 的主实验。

impact-aware temporal decay head 第一版也已完成，输出位于：

```text
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/summary.md
outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay/metrics.json
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-smoke
bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-full
bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-seeds
```

实现方式：

- 在 residual decoder 后加入节点级 horizon gate。
- gate 形状为 `[node, horizon]`，由事故影响分支隐藏状态预测。
- gate 使用 `2 * sigmoid(logit)`，最后一层零初始化，因此初始值为 1，刚开始等价于旧模型。
- 训练后 gate 可以显式放大或减弱不同 horizon 的事故 residual，形成可学习的持续/衰减形状。

正式 test 结果：

- learned normal + dual history: 全候选 0.7254，受影响节点 1.1380。
- + disagreement proxy: 全候选 0.7248，受影响节点 1.1381。
- + temporal decay head: 全候选 0.7239，受影响节点 1.1308。

该结果说明 temporal decay head 对全候选是小幅提升，但对受事故影响节点提升更明显，符合“重点改善事故影响/恢复过程”的预期。

horizon-wise 结果也已加入新 metrics，decay head 在测试集上的代表性 horizon：

| horizon | all model | all baseline | all improve | affected model | affected baseline | affected improve |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5333 | 0.5912 | 9.79% | 0.7465 | 0.8144 | 8.34% |
| 3 | 0.6448 | 0.7256 | 11.14% | 0.9664 | 1.0793 | 10.46% |
| 6 | 0.7287 | 0.8396 | 13.21% | 1.1398 | 1.3077 | 12.84% |
| 9 | 0.7933 | 0.9251 | 14.24% | 1.2715 | 1.4758 | 13.85% |
| 12 | 0.8444 | 0.9853 | 14.31% | 1.3766 | 1.5984 | 13.87% |

当前 learned-normal 消融表已更新，包含 temporal decay head：

```text
outputs/impact_guided_next_stage/ablation_summary/learned_normal_ablation_table.md
```

temporal decay head 的 seed robustness 也已完成，输出位于：

```text
outputs/impact_guided_next_stage/decay_seed_robustness/seed_robustness_summary.md
outputs/impact_guided_next_stage/decay_seed_robustness/seed_robustness_runs.csv
outputs/impact_guided_next_stage/decay_seed_robustness/seed_robustness_summary.csv
```

三组 seed 测试集结果：

- seed 7: 全候选 0.7239，受影响节点 1.1308，horizon-6 affected 1.1398，horizon-12 affected 1.3766。
- seed 11: 全候选 0.7260，受影响节点 1.1388，horizon-6 affected 1.1485，horizon-12 affected 1.3885。
- seed 23: 全候选 0.7228，受影响节点 1.1330，horizon-6 affected 1.1425，horizon-12 affected 1.3802。

均值和标准差：

- 全候选 robust MAE: 0.7242 ± 0.0016。
- 受影响节点 robust MAE: 1.1342 ± 0.0041。
- 非受影响节点 robust MAE: 0.5319 ± 0.0014。
- horizon-6 受影响节点 robust MAE: 1.1436 ± 0.0045。
- horizon-12 受影响节点 robust MAE: 1.3818 ± 0.0061。

和无 decay 的 disagreement proxy seed robustness 相比：

- 全候选均值: 0.7255 -> 0.7242。
- 受影响节点均值: 1.1391 -> 1.1342。

结论：temporal decay head 的收益不是单 seed 偶然，尤其在受事故影响节点上保持稳定增益；它可以作为当前主模型的一部分。

temporal decay head 的 severity / recovery / horizon 分组评估也已完成，输出位于：

```text
outputs/impact_guided_next_stage/decay_group_analysis/summary.md
outputs/impact_guided_next_stage/decay_group_analysis/severity_group_metrics.csv
outputs/impact_guided_next_stage/decay_group_analysis/recovery_group_metrics.csv
outputs/impact_guided_next_stage/decay_group_analysis/horizon_comparison.csv
```

命令入口：

```bash
bash scripts/run_impact_guided_next_stage.sh learned-normal-decay-groups
```

实现方式：

- 对比 no-decay disagreement-proxy model 和 temporal-decay-head model。
- 两者共用同一个 test split 和同一个 residual cache。
- severity 分组按 test 集 `severity_any_z_auc_topk` 三分位切分。
- recovery 分组使用 `<30min`、`30-90min`、`>=90min`。
- 同时输出 overall horizon 1-12 对比。

关键结论：

- overall affected gain: 0.64%。
- high-severity affected gain: 1.05%，horizon-6 gain: 1.10%，horizon-12 gain: 1.22%。
- long-recovery affected gain: 0.94%，horizon-6 gain: 0.98%，horizon-12 gain: 1.05%。
- low-severity 组略有退化，说明 decay head 的收益主要集中在真正强事故影响场景，而不是均匀提升所有样本。

分组结果：

| group | samples | affected no-decay | affected decay | affected gain |
|---|---:|---:|---:|---:|
| severity_low | 9167 | 0.8539 | 0.8553 | -0.16% |
| severity_mid | 9167 | 0.9826 | 0.9811 | 0.15% |
| severity_high | 9165 | 1.3345 | 1.3206 | 1.05% |
| recovery_short_lt30 | 12563 | 0.9591 | 0.9572 | 0.19% |
| recovery_mid_30_90 | 3380 | 1.0520 | 1.0545 | -0.24% |
| recovery_long_ge90 | 11556 | 1.2339 | 1.2224 | 0.94% |

horizon 结果显示，decay head 对 affected 节点的增益从短 horizon 到长 horizon 基本持续为正：

| horizon | affected no-decay | affected decay | affected gain |
|---:|---:|---:|---:|
| 1 | 0.7487 | 0.7465 | 0.29% |
| 3 | 0.9711 | 0.9664 | 0.48% |
| 6 | 1.1475 | 1.1398 | 0.67% |
| 9 | 1.2817 | 1.2715 | 0.80% |
| 12 | 1.3866 | 1.3766 | 0.72% |

论文表述上可以写成：

```text
The temporal decay head yields the largest gains under high-severity and long-recovery incidents, indicating that explicitly modeling temporal persistence helps capture prolonged incident effects.
```

当前最优下一步：

- 固定 learned normal + normal_delta + dual history + temporal decay head 作为当前主模型。
- 整理论文实验表：主消融表、seed robustness、severity/recovery 分组表、horizon-wise 表。
- 之后再考虑 recovery-conditioned gate 或 survival-style recovery head。

之后再接入更强 backbone：

- GraphWaveNet。
- D2STGNN。
- STTN / ST-TIS style lightweight transformer。

### Stage D: Impact-Aware Decay Head

第一版已经跑通：在 full-candidate residual STGNN 后加入可学习时间衰减：

```text
Delta_incident(i, tau) = spatial_impact(i) * temporal_decay(tau) * residual_context(i, tau)
```

重点验证：

- severe incident。
- long recovery incident。
- upstream-dominant incident。
- horizon 6 / horizon 12。

## 6. 近期命令入口

下一阶段实验入口放在：

```text
scripts/run_impact_guided_next_stage.sh
```

推荐先跑 smoke，确认环境和路径：

```bash
bash scripts/run_impact_guided_next_stage.sh smoke
```

复用已有 HDF5 cache 跑 full-candidate STGNN 主实验：

```bash
bash scripts/run_impact_guided_next_stage.sh full
```

跑当前最重要的图结构和辅助监督消融：

```bash
bash scripts/run_impact_guided_next_stage.sh ablations
```

## 7. 论文表述底稿

英文：

```text
Instead of treating incidents as categorical labels, we formulate incident effects as spatiotemporal residual propagation over an incident-centered candidate graph. A counterfactual normal forecast is first estimated, and an impact branch learns the residual deviation caused by incidents. We further construct interpretable impact labels, including severity, recovery, spatial spread, directionality, and node-time heatmaps, to analyze and supervise incident effects.
```

中文：

```text
本文不再将事故作为离散类别进行建模，而是将事故影响表示为事故中心候选节点图上的时空残差传播。模型首先估计无事故情况下的正常交通预测，再由事故影响分支学习事故造成的残差偏离。我们进一步构造严重程度、恢复时间、空间扩散、上下游方向性和节点-时间影响热力图等可解释影响标签，用于分析和监督事故影响。
```
