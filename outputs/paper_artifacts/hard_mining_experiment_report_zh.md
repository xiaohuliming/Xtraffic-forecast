# Hard-example reweighting 探索报告

## 目的

Mixed case study 显示，当前 dual-branch gate 在少数样本上会出现局部失败。继 confidence-aware gate 后，本轮尝试更简单的 hard-example reweighting：不改变模型结构，只在训练损失中对较难的有效 residual elements 额外加权，观察是否能改善 affected candidates 和失败样本。

该方法不使用 affected node label，因此仍属于 no-aux 设置。

## 新增脚本

- `scripts/train_dual_branch_gate_hard_mining.py`

训练目标为：

\[
L = L_{residual} + \lambda_{hard} L_{hard}
\]

其中 \(L_{hard}\) 是每个 batch 内 top fraction hard elements 上的 residual loss。尝试了两种 hard signal：

- `prediction_error`：按当前模型 detached prediction error 选择 hard elements；
- `target_residual`：按目标残差幅度 \(|Y-\hat{Y}^{normal}|\) 选择 hard elements。

## 主结果对比

| Model | Beta | All MAE | Affected MAE | Unaffected MAE | All gain | Affected gain |
|:--|--:|--:|--:|--:|--:|--:|
| dual_branch_gate_no_aux | 1.05 | **0.7181** | **1.1234** | 0.5279 | **13.78%** | **13.17%** |
| confidence_gate_no_aux | 1.05 | 0.7186 | 1.1254 | 0.5278 | 13.71% | 13.02% |
| hard_pred_no_aux | 1.05 | 0.7201 | 1.1250 | 0.5301 | 13.53% | 13.05% |
| hard_target_no_aux | 0.95 | 0.7199 | 1.1260 | 0.5293 | 13.56% | 12.97% |

## 结论

1. `prediction_error` hard mining 没有提升最终效果，All MAE 和 Unaffected MAE 明显变差，Affected MAE 也没有超过原模型。
2. `target_residual` hard mining 更稳定一些，但仍然没有超过原模型。
3. 简单 hard-example reweighting 会让模型更关注高误差或高残差元素，但这不等于更好地学习事故影响。它可能会过度追逐噪声、极端残差或当前模型尚未稳定的局部误差。
4. 当前主模型仍应保持为 `dual_branch_gate_no_aux`。

## 对后续方向的影响

Confidence-aware gate 和 hard-example reweighting 都没有超过原始 no-aux gate，说明当前瓶颈不太可能通过简单训练权重或简单 branch error head 解决。下一步更合理的方向是提升 incident branch 的表达能力，例如将当前轻量 STGNN incident branch 替换为 ST-TIS 风格模块，增强时空信息融合，而不是继续在同一轻量分支上调 loss。

## 可引用文件

- `outputs/impact_guided_next_stage/dual_branch_gate_hard_pred_no_aux/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_gate_hard_target_no_aux/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_gate_hard_pred_no_aux/summary.md`
- `outputs/impact_guided_next_stage/dual_branch_gate_hard_target_no_aux/summary.md`
