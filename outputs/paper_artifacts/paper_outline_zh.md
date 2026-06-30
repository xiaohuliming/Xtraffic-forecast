# 论文写作骨架：事故影响感知交通流预测

## 1. 题目建议

优先题目：

```text
Latent-Incident Mediated Spatio-Temporal Traffic Forecasting under Incidents
```

备选题目：

```text
Impact-Guided Spatio-Temporal Traffic Forecasting under Traffic Incidents
Normal-Impact Decomposed Traffic Forecasting under Incident Perturbations
```

推荐使用第一个题目，因为它能突出本文不是直接依赖事故类型判断，而是通过潜在事故影响残差来改善预测。

## 2. 核心问题

现有交通流预测模型通常假设交通状态主要由正常时空依赖、周期性和局部传播规律决定。当道路发生事故时，未来交通状态会出现额外扰动，例如流量下降、速度下降、占有率上升，以及影响从事故点向上下游传播。此时，单纯学习正常交通模式的模型容易出现误差放大。

XTraffic 原论文指出，从交通状态中反推事故类型的准确度并不高。这说明事故类型标签并不一定是交通预测任务中最有效的监督信号。同一种事故类型可能对应完全不同的交通影响程度，不同事故类型也可能产生相似的交通状态扰动。因此，本文将重点从“事故识别”转向“事故影响建模”。

## 3. 研究假设

本文的核心假设是：

```text
事故对交通预测的关键影响，可以表示为正常交通预测之外的时空残差传播。
```

因此，未来交通状态被分解为：

```text
Y_hat = Y_normal + beta * G_decay * Delta_incident
```

其中：

- `Y_normal` 表示正常交通模式下的反事实预测。
- `Delta_incident` 表示事故诱导的交通残差影响。
- `G_decay` 表示随预测步长变化的事故影响持续性门控。
- `beta` 是验证集选择的残差融合系数。

## 4. 模型架构

模型由四个主要部分组成：

1. **Normal STGNN Branch**  
   使用正常或弱事故窗口训练轻量时空图神经网络，预测未来 12 个时间步的正常交通状态。该分支提供事故发生时的反事实正常预测。

2. **Residual Construction Module**  
   构造事故残差学习信号，包括：
   - learned-normal future residual；
   - statistical residual；
   - `normal_delta`，即 learned normal 与 statistical normal 的差异；
   - `abs(normal_delta)`，作为 normal branch disagreement / uncertainty proxy。

3. **Incident Residual STGNN Branch**  
   在事故位置周围构造 full candidate sensor graph，不使用标签挑选 top-k 受影响节点。模型在完整候选节点集合上学习事故残差的空间传播和时间演化。

4. **Temporal Decay Head**  
   为每个候选节点学习 horizon-level gate：

   ```text
   G_decay = 2 * sigmoid(phi(h))
   ```

   初始化时 gate 接近 1，训练后可以自适应调整不同预测步长下事故影响的持续程度。

## 5. 当前实验结论

### 5.1 主结果

最佳模型为：

```text
learned normal + normal_delta + dual historical residual + disagreement proxy + temporal decay head
```

测试集结果：

- all candidates robust MAE: `0.8328 -> 0.7239`，提升 `13.07%`。
- affected candidates robust MAE: `1.2938 -> 1.1308`，提升 `12.60%`。

对应表格：

- `outputs/paper_artifacts/tables/main_result_table.md`
- `outputs/paper_artifacts/tables/main_result_table.tex`

### 5.2 消融实验

learned-normal 架构内部的消融趋势为：

- 加入 `normal_delta` 后，全候选误差明显下降。
- 加入 dual historical residual 后，affected candidates 误差进一步下降。
- `abs(normal_delta)` 作为 disagreement proxy 带来轻微提升。
- temporal decay head 对 affected candidates 最有效。

对应表格：

- `outputs/paper_artifacts/tables/component_ablation_table.md`
- `outputs/paper_artifacts/tables/component_ablation_table.tex`

### 5.3 Seed 鲁棒性

temporal decay model 在 seeds `7 / 11 / 23` 上稳定：

- all candidates robust MAE: `0.7242 +/- 0.0016`
- affected candidates robust MAE: `1.1342 +/- 0.0041`

对应表格：

- `outputs/paper_artifacts/tables/seed_robustness_table.md`
- `outputs/paper_artifacts/tables/seed_robustness_table.tex`

### 5.4 分组分析

时间衰减头的收益主要集中在：

- high-severity incidents；
- long-recovery incidents；
- 中后期预测 horizon。

这点非常重要，因为它支持本文的核心解释：temporal decay head 捕捉的不是普通残差修正，而是事故影响的持续性和恢复动态。

对应表格和图：

- `outputs/paper_artifacts/tables/temporal_decay_group_table.md`
- `outputs/paper_artifacts/tables/horizon_decay_table.md`
- `outputs/paper_artifacts/figures/severity_recovery_decay_gain.png`
- `outputs/paper_artifacts/figures/horizon_decay_gain_pct.png`

## 6. 图表安排

建议论文中这样安排：

| 位置 | 内容 | 文件 |
| --- | --- | --- |
| Figure 1 | 模型架构图 | `outputs/paper_artifacts/method_diagram.md` |
| Figure 2 | 消融实验 affected MAE | `outputs/paper_artifacts/figures/ablation_affected_mae.png` |
| Figure 3 | horizon-wise affected MAE | `outputs/paper_artifacts/figures/horizon_affected_mae.png` |
| Figure 4 | severity / recovery 分组提升 | `outputs/paper_artifacts/figures/severity_recovery_decay_gain.png` |
| Table 1 | 主结果 | `outputs/paper_artifacts/tables/main_result_table.tex` |
| Table 2 | 组件消融 | `outputs/paper_artifacts/tables/component_ablation_table.tex` |
| Table 3 | seed 鲁棒性 | `outputs/paper_artifacts/tables/seed_robustness_table.tex` |
| Table 4 | temporal decay 分组分析 | `outputs/paper_artifacts/tables/temporal_decay_group_table.tex` |

## 7. 当前可写贡献点

1. 提出从事故类型识别转向事故影响残差建模的预测框架。
2. 构造 normal-impact decomposition，把未来交通状态分解为正常反事实预测和事故诱导残差。
3. 在事故中心 full candidate graph 上学习事故残差传播，避免依赖标签筛选受影响节点。
4. 引入 learned-normal disagreement 和 dual historical residual，增强事故残差分支对异常扰动的感知。
5. 设计 temporal decay head，对事故影响随预测 horizon 的持续性进行自适应建模，并证明其收益集中在高严重度和长恢复事故中。

## 8. 当前风险与写法

需要谨慎表述的地方：

- temporal decay head 总体提升不大，但 affected candidates 和严重/长恢复事故上有清晰收益。
- low-severity group 有轻微退化，因此不要写成所有事故都受益。
- heatmap auxiliary、event-level auxiliary、directional graph 当前不是最强贡献，可以放在 appendix 或 negative findings。

推荐写法：

```text
The temporal decay head yields modest overall gains but consistently improves affected-candidate forecasting under high-severity and long-recovery incidents, indicating that explicit temporal persistence modeling is most beneficial when incident impact is strong and prolonged.
```

## 9. 下一步实验优先级

目前结果已经足够支撑一版论文初稿。若继续补实验，优先级如下：

1. 固定 evaluation set 的 radius 消融，证明 full candidate graph 的必要性。
2. 与标准交通预测 backbone 的事故窗口表现对比。
3. recovery-conditioned gate 或 confidence-aware decay，缓解 low-severity group 退化。
4. 选 2-3 个事故案例图，展示预测曲线和 residual correction。

