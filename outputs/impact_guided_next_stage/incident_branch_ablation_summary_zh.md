# Incident/ST-TIS 分支微调消融记录

本轮只关注模型，不涉及论文排版。

## 当前最佳三 seed 结果

| 模型 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| proposal-aware ST-TIS gate | 0.7131 ± 0.0017 | 1.1200 ± 0.0028 | 0.5222 ± 0.0016 |
| incident-ft old：affected final loss + incident branch loss + gate loss | 0.7104 ± 0.0013 | 1.1087 ± 0.0029 | 0.5235 ± 0.0006 |
| incident-ft final-only：affected final loss only | 0.7094 ± 0.0016 | 1.1081 ± 0.0029 | 0.5223 ± 0.0012 |
| final-only + validation-selected bias calibration | 0.7089 ± 0.0014 | 1.1078 ± 0.0032 | 0.5218 ± 0.0006 |
| final-only + convex gate distillation | 0.7089 ± 0.0012 | 1.1071 ± 0.0024 | 0.5221 ± 0.0007 |
| final-only + convex gate distillation + validation-selected bias | 0.7085 ± 0.0013 | 1.1061 ± 0.0031 | 0.5220 ± 0.0005 |

final-only 的平均提升：

| 相比 proposal | all | affected | unaffected |
|---|---:|---:|---:|
| mean improvement | 0.0038 | 0.0119 | 0.0001 |

当前三 seed 最好的是 `final-only + convex gate distillation + validation-selected bias`。相对 `final-only`，平均进一步改善：

| 相比 final-only | all | affected | unaffected |
|---|---:|---:|---:|
| mean improvement | 0.0009 | 0.0021 | 0.0004 |

## seed 23 消融

| 变体 | all MAE | affected MAE | unaffected MAE | 结论 |
|---|---:|---:|---:|---|
| proposal | 0.7113 | 1.1182 | 0.5204 | 原始 proposal gate |
| affected_weight=0, incident_loss=0.35, gate_loss=0.05 | 0.7074 | 1.1103 | 0.5184 | 整体最好，但 affected 不如加权版本 |
| affected_weight=4, incident_loss=0.35, gate_loss=0.05 | 0.7093 | 1.1066 | 0.5230 | 原先主模型 |
| affected_weight=8, incident_loss=0.35, gate_loss=0.05 | 0.7104 | 1.1061 | 0.5248 | affected 最低，但整体损失变大 |
| affected_weight=4, incident_loss=0, gate_loss=0.05 | 0.7083 | 1.1052 | 0.5220 | 去掉 incident branch direct loss 后更好 |
| affected_weight=4, incident_loss=0, gate_loss=0 | 0.7082 | 1.1052 | 0.5219 | 当前 seed 23 最干净配置 |
| final-only + learned veto gate | 0.7078 | 1.1053 | 0.5213 | all/unaffected 极小幅改善，但 affected 没有改善 |
| affected_weight=4, incident_loss=0.35, gate frozen | 0.7094 | 1.1074 | 0.5227 | gate 重训练有小幅贡献，但不是主要来源 |

## 机制判断

1. 单独 gate adapter/veto 的收益很小；主要路线应转向 incident/ST-TIS 分支与 gate 的联合适配。最新结果显示 convex gate distillation 是比单独 adapter 更稳的小改动。
2. affected weight 是一个 trade-off knob：
   - `affected_weight=0` 更偏整体/普通节点；
   - `affected_weight=4` 更平衡；
   - `affected_weight=8` 进一步压 affected，但牺牲 all/unaffected。
3. 直接让 incident branch 单独拟合 residual 未必有利。final-only 配置反而更好，说明当前模型更像“ST-TIS 分支提供事故相关融合特征，gate 学会如何使用它”，而不是“incident branch 单独就是更准的专家”。
4. 典型失败样本 `88134` 仍未解决：
   - proposal learned affected MAE: 6.5106
   - old incident-ft learned affected MAE: 6.5727
   - final-only learned affected MAE: 6.8535
   - convex-gate original learned affected MAE: 6.7009
   - normal branch affected MAE 约 2.1877，说明失败原因仍是 gate 在 incident branch 极差时没有充分压低 incident 权重。

## 阶段性建议

在 hard-case robustness 和 convex-gate 实验前，主模型曾暂定为 `incident-ft final-only`：

```text
freeze normal branch
train ST-TIS incident branch + proposal gate
loss = affected-weighted final residual prediction loss
no incident branch direct loss
no gate auxiliary loss
```

后续实验已经确认：在这个基础上加入 `convex gate distillation` 可以获得更好的三 seed 平均结果。当前最终建议见文末“当前模型结论更新”。

当时提出的 hard-case robustness 方向如下：

1. 对 `88134` 这类 normal branch 明显更好的样本，引入 branch-disagreement hard-negative weighting。
2. 将 affected weighting 做成 validation-selected Pareto 配置：all-oriented (`aw=0`) 和 affected-oriented (`aw=8`) 各保留一版。
3. 重新设计 gate 的可靠性输入，让它在 incident branch proposal 幅度过大、与 normal branch 强烈冲突时更保守。

## 2026-05-02 hard-case robustness 进展

### 训练期 hard-negative gate loss

新增训练项：当训练时发现 normal branch 局部误差明显小于 incident branch，并且 gate 仍然较高时，对 gate 加保守惩罚。

seed 23 结果：

| 变体 | all MAE | affected MAE | unaffected MAE | 备注 |
|---|---:|---:|---:|---|
| final-only | 0.7082 | 1.1052 | 0.5219 | 当前基线 |
| hardneg weight=0.005 | 0.7082 | 1.1050 | 0.5221 | affected 极小幅改善，all 极小幅变差 |
| hardneg weight=0.05 | 0.7213 | 1.1286 | 0.5302 | gate 被过度压低，整体崩坏 |

`88134` case：

| 变体 | gate mean affected | learned affected MAE |
|---|---:|---:|
| proposal | 0.4478 | 6.5106 |
| final-only | 0.4156 | 6.8535 |
| hardneg weight=0.005 | 0.4122 | 6.9238 |
| hardneg weight=0.05 | 0.1309 | 4.2500 |

判断：hard-negative 方向能修复 `88134`，但训练期全局 gate 惩罚很容易过强，导致整体指标明显下降。当前不建议把 hardneg loss 作为主模型训练项。

### 推理期 gate posthoc calibration

对 final-only 模型做 posthoc gate sweep，只改推理时 gate，不改模型权重。

seed 23：

| 变体 | beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| original | 1.00 | 0.7082 | 1.1052 | 0.5219 |
| bias_-0.2 | 1.05 | 0.7077 | 1.1044 | 0.5215 |
| threshold cap best | 1.00 | 0.7121 | 1.1160 | 0.5225 |

threshold cap 针对 `88134` 的极端 branch disagreement 设计，但在验证集和测试集上均明显不如简单 `bias_-0.2`，不建议保留为主方案。

三 seed `bias_-0.2` / original 按 validation all MAE 选择后的 test 均值：

| 方案 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| final-only original checkpoint | 0.7094 ± 0.0016 | 1.1081 ± 0.0029 | 0.5223 ± 0.0012 |
| final-only + validation-selected bias calibration | 0.7089 ± 0.0014 | 1.1078 ± 0.0032 | 0.5218 ± 0.0006 |

判断：`bias_-0.2` 是一个稳定但很小的推理校准收益，可以作为 calibration 结果汇报；它不能解决 `88134` 这类极端失败样本。

### Learned veto gate

在 final-only checkpoint 后面冻结原模型，只训练一个轻量 veto adapter，让模型在 incident branch 明显不可信时降低 incident gate。训练目标包括 affected-weighted final loss、veto loss、hard residual loss 和稀疏约束。

seed 23 完整评估：

| 变体 | all MAE | affected MAE | unaffected MAE | 选择超参 |
|---|---:|---:|---:|---|
| final-only source | 0.7082 | 1.1052 | 0.5219 | beta=1.00 |
| final-only + validation-selected bias | 0.7077 | 1.1044 | 0.5215 | bias=-0.2, beta=1.05 |
| final-only + learned veto | 0.7078 | 1.1053 | 0.5213 | veto_scale=0.25, beta=1.00 |

`88134` case：

| 变体 | gate mean affected | learned affected MAE |
|---|---:|---:|
| final-only | 0.4156 | 6.8535 |
| final-only + learned veto | 0.4143 | 6.8464 |
| hardneg weight=0.05 | 0.1309 | 4.2500 |

判断：learned veto 对整体 MAE 有很小帮助，主要来自 unaffected 轻微改善；但它没有改善 affected MAE，也几乎不修复 `88134`。因此不建议作为主模型模块继续扩展。它的价值是确认：只在模型末端加一个保守拒绝门，不足以学到强 case-level robustness。

### Oracle branch upper bound

为了判断问题是“分支本身不行”还是“gate 没选好”，对 seed 23 final-only checkpoint 做了 test-set oracle 上限诊断。

| 预测方式 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| normal baseline | 0.8328 | 1.2938 | 0.6165 |
| normal branch only | 0.7777 | 1.2133 | 0.5733 |
| incident branch only | 0.8325 | 1.2648 | 0.6296 |
| fixed gate 0.5 | 0.7255 | 1.1379 | 0.5320 |
| learned gate | 0.7082 | 1.1052 | 0.5219 |
| oracle branch min | 0.5686 | 0.9046 | 0.4110 |
| oracle convex fusion | 0.4890 | 0.8034 | 0.3415 |

判断：分支里确实存在大量可用信息。尤其是 affected 区域，learned gate 到 oracle branch min 之间仍有约 `0.2006` MAE 空间，到 oracle convex fusion 之间有约 `0.3017` MAE 空间。因此下一步重点不应再是简单整体压低/抬高 gate，而是让 gate 学到更细粒度的分支置信度和连续融合位置。

### Convex gate distillation

新增训练项：让 gate 学习 normal/incident residual proposal 的逐元素最优连续融合系数。

```text
target_gate = clip((y - normal_residual) / (incident_residual - normal_residual), 0, 1)
loss = affected-weighted final residual loss
     + 0.05 * SmoothL1(gate, target_gate)
```

其中 normal/incident residual proposal 在构造 target 时 detach，并且当两个 proposal 太接近时不计算该项，避免不稳定目标。

三 seed 结果：

| 模型 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| final-only original | 0.7094 ± 0.0016 | 1.1081 ± 0.0029 | 0.5223 ± 0.0012 |
| final-only + validation-selected bias | 0.7089 ± 0.0014 | 1.1078 ± 0.0032 | 0.5218 ± 0.0006 |
| convex-gate original | 0.7089 ± 0.0012 | 1.1071 ± 0.0024 | 0.5221 ± 0.0007 |
| convex-gate + validation-selected bias | 0.7085 ± 0.0013 | 1.1061 ± 0.0031 | 0.5220 ± 0.0005 |

seed 23 最佳 posthoc：

| 变体 | beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| final-only + bias_-0.2 | 1.05 | 0.7077 | 1.1044 | 0.5215 |
| convex-gate original | 0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | 1.05 | 0.7071 | 1.1028 | 0.5215 |

`88134` case：

| 变体 | gate mean affected | learned affected MAE |
|---|---:|---:|
| final-only original | 0.4156 | 6.8535 |
| final-only + bias_-0.2 | 0.3939 | 7.0042 |
| convex-gate original | 0.4019 | 6.7009 |
| convex-gate + bias_-0.2 | 0.3804 | 7.1310 |

判断：convex-gate 是目前最有价值的小改动，三 seed 上稳定改善 all 和 affected。它仍没有真正解决 `88134` 这类极端 case；posthoc bias 对总体有效，但可能伤害个别强分歧样本。seed 23 interpretability 显示 gate 均值和分支选择对齐变化不大，主要收益更像是 incident branch proposal 与整体融合校准的小幅改善。

### Branch confidence gate v1

新增一个轻量 branch-confidence adapter：冻结 convex-gate 源模型，只训练 normal/incident 两个 confidence head。

```text
gate_logit = base_gate_logit
           + confidence_scale * (incident_confidence - normal_confidence)
```

训练目标：

```text
affected-weighted final residual loss
+ confidence ranking loss
+ convex gate distillation loss
```

seed 23 完整结果：

| 变体 | scale / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5215 |
| branch-confidence original | scale=1.5, beta=0.95 | 0.7077 | 1.1051 | 0.5212 |
| branch-confidence + bias_-0.2 | beta=1.05 | 0.7070 | 1.1032 | 0.5211 |

`88134` case：

| 变体 | gate mean affected | learned affected MAE |
|---|---:|---:|
| convex-gate original | 0.4019 | 6.7009 |
| convex-gate + bias_-0.2 | 0.3804 | 7.1310 |
| branch-confidence original | 0.4015 | 6.7622 |
| branch-confidence + bias_-0.2 | 0.3812 | 7.2114 |

判断：branch-confidence v1 没有达到预期。它能进一步改善 all/unaffected，但 affected 不如 convex-gate + bias，并且 `88134` 变差。诊断发现 confidence 修正很小，`88134` 上 confidence_delta 约 `-0.028`，不足以明显压低 incident gate。强版 smoke 增大 confidence/convex loss 和 confidence_max 后，validation 反而选择 `confidence_scale=0`，因此不建议继续用这种简单 confidence head 作为主线。

### Branch uncertainty gate v1

新增 branch uncertainty adapter：冻结 convex-gate 源模型，只训练 normal/incident 两个 risk head，分别预测 `log(1 + branch_error)`，再用 risk 差修正 gate。

```text
gate_logit = base_gate_logit
           + uncertainty_scale * tanh(normal_risk - incident_risk)
```

seed 23 完整结果：

| 变体 | scale / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5215 |
| uncertainty-gate original | scale=0.5, beta=0.95 | 0.7078 | 1.1049 | 0.5215 |
| uncertainty-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5214 |

`88134` case：

| 变体 | gate mean affected | learned affected MAE |
|---|---:|---:|
| convex-gate original | 0.4019 | 6.7009 |
| uncertainty-gate original | 0.4067 | 6.7517 |
| uncertainty-gate + bias_-0.2 | 0.3857 | 7.1949 |

判断：uncertainty v1 相比 branch-confidence v1 更贴近目标，整体和 affected 有极小改善，但仍没有超过 convex-gate + bias，也没有修复 `88134`。关键诊断：在 `88134` 上 risk head 预测 `normal_risk=0.3634`、`incident_risk=0.3614`，方向错了。原因可能是 risk head 只看 branch hidden/delta，没有直接看 decoded residual proposal 的幅度和分歧。下一步更合理的是 proposal-aware uncertainty head：让 risk head 直接输入 proposal features / base gate logits。

### Proposal-aware uncertainty gate

在 branch uncertainty v1 后，继续测试更强的 proposal-aware risk head：冻结 convex-gate 源模型，只训练 normal/incident risk head，但 risk head 的输入改为：

```text
[normal/incident hidden features
 + decoded residual proposal features
 + normalized base gate logits]
```

这样 risk head 不只看隐藏状态，还能直接看到两个分支 proposal 的幅度、差异和当前 gate 倾向。

seed 23 完整结果：

| 变体 | scale / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5215 |
| proposal-uncertainty original | scale=0.5, beta=0.95 | 0.7077 | 1.1048 | 0.5214 |
| proposal-uncertainty + bias_-0.2 | beta=1.05 | 0.7071 | 1.1029 | 0.5214 |

posthoc 搜索中，proposal-uncertainty 的 `bias_-0.2, beta=1.05` 在 test all 上与 convex-gate + bias 几乎持平，并略微改善 unaffected；但 affected 仍略差于 convex-gate + bias。

`88134` case：

| 变体 | gate mean affected | learned affected MAE | risk 诊断 |
|---|---:|---:|---|
| convex-gate original | 0.4019 | 6.7009 | - |
| convex-gate + bias_-0.2 | 0.3804 | 7.1310 | - |
| proposal-uncertainty original | 0.4110 | 6.8004 | `normal_risk=0.5907`, `incident_risk=0.5622`, `risk_delta=0.0847` |
| proposal-uncertainty + bias_-0.2 | 0.3904 | 7.2576 | 同上 |

判断：proposal-aware uncertainty 的方向比 branch-only uncertainty 更合理，但当前实现仍没有超过 convex-gate + bias 的 affected 表现，也没有修复 `88134`。关键问题更明确了：即使 risk head 看到 proposal features，它仍可能在极端样本上判断 normal branch 更危险，从而继续抬高 incident gate。当前不建议扩展到三 seed；它可以作为 negative/diagnostic 结果保留。

### Local selector gate v1

新增 local 3-way selector：冻结 convex-gate 源模型，只训练一个逐元素 selector head。selector 不再只是修正 scalar gate，而是在三个 proposal 之间做 softmax：

```text
option 0: normal branch residual
option 1: incident branch residual
option 2: base fused residual = original gate fusion
```

训练目标包括：

```text
affected-weighted final residual loss
+ cross entropy(selector_logits, argmin proposal error)
+ regret loss = expected proposal error - oracle best proposal error
```

seed 23 完整结果：

| 变体 | temperature / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5215 |
| local selector | temp=0.75, beta=1.10 | 0.7087 | 1.1046 | 0.5229 |

local selector 对 affected 比 convex-gate original 略好，但 all/unaffected 变差，也仍不如 convex-gate + bias。

`88134` case：

| 变体 | gate/effective incident weight | learned affected MAE | selector 权重 |
|---|---:|---:|---|
| convex-gate original | 0.4019 | 6.7009 | - |
| convex-gate + bias_-0.2 | 0.3804 | 7.1310 | - |
| local selector | 0.4089 | 7.4897 | normal 0.1066 / incident 0.0283 / fused 0.8652 |

判断：local selector 形式是对的，但普通随机训练没有学会在 `88134` 这类极端样本上拒绝 fused proposal。

### Local selector + hard replay

为了让 selector 看见更多 normal 明显胜出的训练模式，新增 hard replay：扫描训练集，挑选 normal branch 显著优于 incident/base fused 的样本，额外重复加入训练。

中等规模 seed 23 实验：

```text
random train samples: 20000
hard replay scan samples: 50000
hard replay selected samples: 5000
epochs: 2
```

全量结果：

| 变体 | temperature / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| local selector + hard replay | temp=1.25, beta=1.10 | 0.7121 | 1.1127 | 0.5242 |

`88134` case：

| 变体 | gate/effective incident weight | learned affected MAE | selector 权重 |
|---|---:|---:|---|
| convex-gate original | 0.4019 | 6.7009 | - |
| local selector | 0.4089 | 7.4897 | normal 0.1066 / incident 0.0283 / fused 0.8652 |
| local selector + hard replay | 0.3696 | 6.5647 | normal 0.3731 / incident 0.1293 / fused 0.4976 |

判断：hard replay 的机制确实能修复 `88134` 一类极端样本，但当前训练强度太粗，严重伤害全量 all/affected。它说明“针对 hard-case 的局部拒绝”方向成立，但需要更温和的触发条件或两阶段策略；当前不建议作为主模型结果。

### Two-stage normal veto gate

在 local selector/hard replay 后，新增更温和的 two-stage normal veto：冻结 convex-gate 源模型，只训练一个 normal-veto head。默认保留原始 fused proposal，只在局部预测 normal branch 更可靠时，把 fused proposal 往 normal branch 拉。

```text
base_fused = (1 - base_gate) * normal + base_gate * incident
normal_veto_amount = scale * sigmoid(veto_logit / temperature)
prediction = (1 - normal_veto_amount) * base_fused
           + normal_veto_amount * normal
```

训练目标：

```text
affected-weighted final residual loss
+ BCE(veto_logit, normal-better target)
+ regret loss against min(base_fused, normal)
+ sparsity loss on veto amount
```

seed 23 完整结果：

| 变体 | scale / temp / beta | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|---:|
| convex-gate original | beta=0.95 | 0.7079 | 1.1050 | 0.5216 |
| convex-gate + bias_-0.2 | beta=1.05 | 0.7071 | 1.1028 | 0.5215 |
| normal-veto selected | scale=0.25, temp=1.0, beta=1.05 | 0.7075 | 1.1027 | 0.5221 |
| normal-veto strong | scale=1.0, temp=1.0, beta=1.05 | 0.7087 | 1.1059 | 0.5223 |

`88134` case：

| 变体 | effective incident weight | normal veto amount | learned affected MAE |
|---|---:|---:|---:|
| convex-gate original | 0.4019 | - | 6.7009 |
| convex-gate + bias_-0.2 | 0.3804 | - | 7.1310 |
| normal-veto selected | 0.3953 | 0.0128 | 7.1871 |
| normal-veto strong | 0.3755 | 0.0514 | 6.9253 |

判断：two-stage normal veto 是目前最温和、最稳定的 selector 类改动。它在 seed 23 affected 上略好于 convex-gate + bias，但 all/unaffected 仍略差，而且没有修复 `88134`。强 veto 对硬样本稍有帮助，但会伤全量 affected。因此当前它可以作为 promising diagnostic/备选，不足以替代三 seed 主线；若继续，需要做更精准的触发器，而不是简单提高 veto scale。

### Two-stage normal veto 三 seed 复现

对 seed 7/11/23 都按同一配置复现 continuous normal-veto：

```text
source = incident-ft final-only + convex gate distillation
normal_veto_scale candidates = 0.0, 0.25, 1.0
temperature candidates = 1.0, 1.5
beta candidates = 0.95, 1.0, 1.05
selection = validation all MAE within +0.002, then affected MAE
```

三个 seed 都选择了同一组参数：`scale=0.25, temperature=1.0, beta=1.05`。

| 模型 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| convex-gate original | 0.7089 ± 0.0012 | 1.1071 ± 0.0024 | 0.5221 ± 0.0007 |
| convex-gate + validation-selected bias | 0.7085 ± 0.0013 | 1.1061 ± 0.0031 | 0.5220 ± 0.0005 |
| continuous normal-veto | 0.7087 ± 0.0013 | 1.1056 ± 0.0029 | 0.5225 ± 0.0005 |

逐 seed 对比：

| seed | convex-gate + bias affected | normal-veto affected | normal-veto all | normal-veto unaffected |
|---:|---:|---:|---:|---:|
| 7 | 1.1088 | 1.1086 | 0.7100 | 0.5230 |
| 11 | 1.1066 | 1.1054 | 0.7086 | 0.5224 |
| 23 | 1.1028 | 1.1027 | 0.7075 | 0.5221 |

判断：continuous normal-veto 是一个稳定的 affected-oriented 变体。它把三 seed affected 均值从 `1.1061` 降到 `1.1056`，但 all 从 `0.7085` 升到 `0.7087`，unaffected 从 `0.5220` 升到 `0.5225`。因此如果以整体 MAE 为主，主线仍是 `convex-gate + validation-selected bias`；如果强调事故影响节点 MAE，normal-veto 可以作为备选主模型/消融亮点。

### Normal-veto 触发器与 posthoc bias 诊断

后续又测试了三类强化：

1. 对现有 normal-veto score 做 hard threshold sweep。
2. 将 normal-veto target 从 continuous ramp 改成 binary normal-better detector。
3. 在 normal-veto 后再做 posthoc gate bias sweep。

结论：

| 方向 | 结果 | 判断 |
|---|---|---|
| hard threshold sweep | seed 23 validation 仍选择 continuous scale=0.25；hard threshold 没赢 | 现有 veto score 不是高精度触发器 |
| binary detector smoke | 小样本 test affected 曾到 1.0969 | 完整 test 复核后消失，是抽样幻觉 |
| binary detector full | validation 选择 scale=0，即不用 veto；test affected 1.1035 | 二值目标太粗，启用后伤验证集 |
| normal-veto posthoc bias | 三 seed validation 仍选择 original,beta=1.05 | bias 没有进一步稳定收益 |

`88134` 诊断补充：

| 变体 | affected MAE |
|---|---:|
| convex-gate original | 6.7009 |
| convex-gate + bias_-0.2 | 7.1310 |
| normal-veto selected continuous | 7.1871 |
| normal-veto hard threshold t=0.05,a=0.25 | 5.9944 |

hard threshold 可以救 `88134`，但代价是全局触发率过高，完整 test 上不成立。这说明“硬样本救援”方向本身可能有用，但当前触发器不够精确。

### Oracle high-precision normal-veto 上限

为了判断“高精度触发器”是否值得继续，新增 oracle sweep：直接用真实未来误差判断 normal branch 是否比 base fused proposal 更好。

```text
normal_advantage = |base_fused - y| - |normal - y|
trigger = normal_advantage >= margin
prediction = (1 - amount * trigger) * base_fused
           + amount * trigger * normal
```

这个实验不是可部署模型，只是上限诊断。

seed 23 test：

| 方法 | all MAE | affected MAE | unaffected MAE | affected trigger | unaffected trigger |
|---|---:|---:|---:|---:|---:|
| convex-gate original | 0.7079 | 1.1050 | 0.5216 | - | - |
| continuous normal-veto | 0.7075 | 1.1027 | 0.5221 | - | - |
| oracle margin=0, amount=1, beta=1.10 | 0.6070 | 0.9635 | 0.4398 | 0.4365 | 0.4416 |
| oracle margin=0.2, amount=1, beta=1.10 | 0.6334 | 0.9879 | 0.4671 | 0.1848 | 0.1339 |

判断：normal-veto 的理论上限很大。即使只触发 margin>=0.2 的较强 normal-better 位置，affected 仍可从 `1.1030` 左右降到 `0.9879`。因此问题不是“normal 回拉没用”，而是当前可学习 detector 没有学到 oracle 触发规则。

### High-precision trigger v1/v2

基于 oracle 结果，测试了两个更稀疏的可学习触发器：

1. `binary_hp`：二值目标，`normal_better_margin=0.2`，提高负样本权重和 sparsity。
2. `continuous_hp`：连续目标，但只从 `margin=0.2` 以后开始回归，保留 normal_advantage 的强弱信息。

seed 23 完整复核：

| 方法 | all MAE | affected MAE | unaffected MAE | affected trigger | unaffected trigger |
|---|---:|---:|---:|---:|---:|
| continuous normal-veto | 0.7075 | 1.1027 | 0.5221 | - | - |
| binary high-precision smoke full sweep | 0.7075 | 1.1030 | 0.5219 | 0.0595 | 0.0631 |
| continuous high-precision smoke full sweep | 0.7077 | 1.1030 | 0.5223 | 0.0337 | 0.0328 |

判断：通过调 margin、负样本权重、sparsity，确实可以让触发器变稀疏，但它没有学到 oracle 那种“触发在真正有收益的位置”。两个版本的 affected 都没有超过原 continuous normal-veto。因此当前不建议继续简单调 BCE/continuous target；下一步需要换 detector 信息或训练范式。

### Pairwise ranking / regret regression detector

在 high-precision v1/v2 后，继续测试两类替代训练范式：

1. `pairwise ranking detector`：对同一 batch 内的 positive/negative 位置做排序约束，让 `normal_advantage > margin` 的位置 veto logit 高于 `normal_advantage <= 0` 的位置。
2. `regret regression detector`：不用 BCE，直接用 SmoothL1 回归 continuous normal_advantage target。

新增训练参数：

```text
ranking_loss_weight
ranking_margin
ranking_positive_margin
ranking_negative_margin
ranking_pairs_per_batch
ranking_affected_only
veto_loss_kind = bce / smooth_l1 / mse
```

seed 23 smoke 结果：

| 变体 | eval | all MAE | affected MAE | unaffected MAE | 选择 |
|---|---|---:|---:|---:|---|
| binary_hp_smoke | 2048 sample | 0.7092 | 1.1032 | 0.5228 | scale=0.5,temp=1.5,beta=1.05 |
| continuous_hp_smoke | 2048 sample | 0.7094 | 1.1032 | 0.5232 | scale=1.0,temp=1.0,beta=1.05 |
| ranking affected-only | 2048 sample | 0.7094 | 1.1037 | 0.5229 | validation 选 scale=0 |
| ranking balanced | 2048 sample | 0.7092 | 1.1033 | 0.5228 | scale=0.25,temp=1.5,beta=1.05 |
| smooth-L1 regression | 2048 sample | 0.7094 | 1.1032 | 0.5230 | scale=0.25,temp=1.5,beta=1.05 |

其中两个 high-precision 版本做了完整 test sweep 复核：

| 变体 | full test all | full test affected | full test unaffected |
|---|---:|---:|---:|
| binary_hp_smoke_full_sweep | 0.7075 | 1.1030 | 0.5219 |
| continuous_hp_smoke_full_sweep | 0.7077 | 1.1030 | 0.5223 |
| 原 continuous normal-veto | 0.7075 | 1.1027 | 0.5221 |

判断：pairwise ranking 和 SmoothL1 regression 也没有超过原 continuous normal-veto。ranking affected-only 甚至被 validation 选择为 `scale=0`，说明它太保守；balanced ranking 和 regression 能产生 veto，但 affected 仍停在 `1.1032~1.1033`。这进一步说明问题不只是 BCE 损失形式，而是 detector 输入/监督粒度不够。

### Node-event normal-veto

为了检查逐元素 veto head 是否太细碎，又实现了一个 node-event 版本：每个样本、每个节点只输出一个事故回拉强度，再广播到所有预测 horizon/channel。它仍然使用 normal/proposal/global/veto 特征，但参数量更少，约 `46345`，相对 element normal-veto 的约 `49740` 更受约束。

seed 23 test：

| 变体 | 选择方式 | all MAE | affected MAE | unaffected MAE | affected trigger | unaffected trigger |
|---|---|---:|---:|---:|---:|---:|
| convex-gate original | checkpoint | 0.7079 | 1.1050 | 0.5216 | - | - |
| element continuous normal-veto | checkpoint | 0.7075 | 1.1027 | 0.5221 | - | - |
| node-event val-selected full sweep | validation 选择 | 0.7076 | 1.1030 | 0.5221 | 0.0469 | 0.0505 |
| node-event test-best all | test posthoc | 0.7072 | 1.1033 | 0.5214 | 0.1040 | 0.1163 |
| node-event test-best affected | test posthoc | 0.7078 | 1.1030 | 0.5224 | 0.0938 | 0.1011 |

判断：node-event 版本训练正常，也能在 test-posthoc 下把 all MAE 压到 `0.7072`，但这是测试集后验选择，不能作为有效模型选择标准；按 validation 选择时 affected 是 `1.1030`，没有超过 element continuous normal-veto 的 `1.1027`。所以仅把 veto 从逐元素改成节点事件级，不足以解决 detector 学不准的问题。

### Impact-conditioned normal-veto

为了真正改变 detector 输入，而不是只改 loss 或输出粒度，又实现了一个 impact-conditioned normal-veto：训练时用 `impact_heatmap`、`node_affected`、`event_aux` 监督 auxiliary heads；推理时不使用真实事故标签，只把这些 heads 预测出来的事故影响热力图、节点受影响概率、事件 severity/recovery/spread 表示作为 veto detector 的额外输入。

新增模型类：

```text
DualBranchSTTISImpactConditionedNormalVetoGate
```

新增训练参数：

```text
normal_veto_context = impact_aux
impact_aux_weight = 0.03
event_aux_weight = 0.01
node_aux_weight = 0.02
eval_batch_size
```

三 seed full test：

| seed | 方法 | all MAE | affected MAE | unaffected MAE | 选择 |
|---:|---|---:|---:|---:|---|
| 7 | element continuous normal-veto | 0.710000 | 1.108568 | 0.523000 | scale=0.25,temp=1.0,beta=1.05 |
| 7 | impact-conditioned normal-veto | 0.710061 | 1.108088 | 0.523315 | scale=0.5,temp=1.0,beta=1.10 |
| 11 | element continuous normal-veto | 0.708593 | 1.105411 | 0.522414 | scale=0.25,temp=1.0,beta=1.05 |
| 11 | impact-conditioned normal-veto | 0.708576 | 1.105321 | 0.522431 | scale=0.25,temp=1.0,beta=1.05 |
| 23 | element continuous normal-veto | 0.707490 | 1.102705 | 0.522064 | scale=0.25,temp=1.0,beta=1.05 |
| 23 | impact-conditioned normal-veto | 0.707466 | 1.102669 | 0.522044 | scale=0.25,temp=1.0,beta=1.05 |

三 seed 平均：

| 方法 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| element continuous normal-veto | 0.708694 | 1.105561 | 0.522493 |
| impact-conditioned normal-veto | 0.708701 | 1.105359 | 0.522597 |

判断：impact-conditioned detector 的方向是有效的，但收益很小。它在三个 seed 上都降低 affected MAE，平均降 `0.00020`；代价是 all 基本持平略差，unaffected 平均升 `0.00010`。因此它可以作为 affected-oriented v2 candidate，但还不能替代 all-oriented mainline，也不能宣称显著提升。后续如果继续模型，应重点看它是否集中改善 severe/long-recovery 事故，而不是只看整体 affected 均值。

seed 23 分组诊断：

| group | samples | affected delta | all delta | 判断 |
|---|---:|---:|---:|---|
| overall | 27499 | -0.000036 | -0.000025 | 极小改善 |
| severity_low | 9167 | +0.000019 | -0.000008 | affected 略差 |
| severity_mid | 9167 | +0.000027 | -0.000006 | affected 略差 |
| severity_high | 9165 | -0.000096 | -0.000051 | 高严重度事故改善更明显 |
| recovery_short_lt30 | 12563 | +0.000012 | -0.000012 | affected 略差 |
| recovery_mid_30_90 | 3380 | +0.000091 | +0.000008 | affected 略差 |
| recovery_long_ge90 | 11556 | -0.000081 | -0.000042 | 长恢复事故改善更明显 |

分组结果支持这个方向的解释：impact_aux 的收益主要出现在 high-severity 和 long-recovery 事故上，符合“学习事故影响程度/持续时间”的初始动机。但绝对幅度仍然很小，下一步不能只继续微调权重，而应把这个信号强化成更明确的 severity/recovery-conditioned detector。

### Severity/recovery-focused impact-conditioned normal-veto

进一步把 severity/recovery 显式变成训练权重：只对 above-average severity/recovery 事件里的 affected 位置增加 loss 权重，推理时仍然不使用真实事故标签。新增参数：

```text
severity_focus_weight = 0.5
recovery_focus_weight = 0.5
event_focus_temperature = 1.0
event_focus_max = 2.0
```

三 seed full test：

| seed | 方法 | all MAE | affected MAE | unaffected MAE | 选择 |
|---:|---|---:|---:|---:|---|
| 7 | element continuous normal-veto | 0.710000 | 1.108568 | 0.523000 | scale=0.25,temp=1.0,beta=1.05 |
| 7 | impact-conditioned normal-veto | 0.710061 | 1.108088 | 0.523315 | scale=0.5,temp=1.0,beta=1.10 |
| 7 | severity/recovery-focused impact-veto | 0.710039 | 1.107855 | 0.523391 | scale=0.5,temp=1.0,beta=1.10 |
| 11 | element continuous normal-veto | 0.708593 | 1.105411 | 0.522414 | scale=0.25,temp=1.0,beta=1.05 |
| 11 | impact-conditioned normal-veto | 0.708576 | 1.105321 | 0.522431 | scale=0.25,temp=1.0,beta=1.05 |
| 11 | severity/recovery-focused impact-veto | 0.708575 | 1.105219 | 0.522477 | scale=0.25,temp=1.0,beta=1.05 |
| 23 | element continuous normal-veto | 0.707490 | 1.102705 | 0.522064 | scale=0.25,temp=1.0,beta=1.05 |
| 23 | impact-conditioned normal-veto | 0.707466 | 1.102669 | 0.522044 | scale=0.25,temp=1.0,beta=1.05 |
| 23 | severity/recovery-focused impact-veto | 0.707456 | 1.102596 | 0.522064 | scale=0.25,temp=1.0,beta=1.05 |

三 seed 平均：

| 方法 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| element continuous normal-veto | 0.708694 | 1.105561 | 0.522493 |
| impact-conditioned normal-veto | 0.708701 | 1.105359 | 0.522597 |
| severity/recovery-focused impact-veto | 0.708690 | 1.105223 | 0.522644 |

判断：focused 版本比普通 impact_aux 更符合事故导向目标，三 seed 都继续降低 affected MAE，平均相对 element normal-veto 降 `0.00034`，相对普通 impact_aux 再降 `0.00014`。它的代价是 unaffected 平均升 `0.00015`，因此不适合作为 all-oriented mainline，但可以作为目前最合理的 affected-oriented v2。

seed 23 分组诊断进一步验证了机制：

| group | affected delta vs element normal-veto |
|---|---:|
| severity_low | +0.000104 |
| severity_mid | +0.000102 |
| severity_high | -0.000322 |
| recovery_short_lt30 | +0.000049 |
| recovery_mid_30_90 | +0.000257 |
| recovery_long_ge90 | -0.000249 |

也就是说它确实把收益集中到了高严重度和长恢复事故上，而不是所有 affected 节点平均改善。这与课题目标“学习事故对交通流量的影响程度/持续时间”更一致。

三 seed 分组诊断进一步确认这个机制是稳定的。下面的 delta 都是 `focused impact-veto - element normal-veto`，负数表示 focused 更好：

| group | all delta mean | affected delta mean | unaffected delta mean | affected delta range |
|---|---:|---:|---:|---:|
| overall | -0.000005 | -0.000338 | +0.000151 | [-0.000713, -0.000109] |
| severity_low | +0.000220 | +0.000345 | +0.000189 | [+0.000104, +0.000789] |
| severity_mid | +0.000150 | +0.000140 | +0.000155 | [+0.000090, +0.000229] |
| severity_high | -0.000282 | -0.000882 | +0.000116 | [-0.001833, -0.000322] |
| recovery_short_lt30 | +0.000140 | +0.000037 | +0.000171 | [-0.000021, +0.000084] |
| recovery_mid_30_90 | +0.000338 | +0.000698 | +0.000187 | [+0.000257, +0.001485] |
| recovery_long_ge90 | -0.000196 | -0.000702 | +0.000123 | [-0.001485, -0.000249] |

三 seed 结论：`severity_high` 和 `recovery_long_ge90` 的 affected delta 在 7/11/23 三个 seed 上全部为负，说明 focused impact-veto 不是随机改善 overall affected，而是稳定把收益转移到更严重、恢复更久的事故事件上。低/中严重度和中等恢复时长的 affected MAE 反而变差，这进一步说明该模型是在做事故影响导向的取舍，而不是普遍降噪。

### Severity/recovery target-boost impact-veto

在 focused impact-veto 后，又测试了更强的 detector 监督：不仅提高 high-severity / long-recovery affected 位置的 loss 权重，还把这些位置的 normal-veto target 本身放大。

新增参数：

```text
severity_target_boost = 0.5
recovery_target_boost = 0.5
event_target_boost_max = 2.0
```

三 seed full test：

| 方法 | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| element continuous normal-veto | 0.708694 | 1.105561 | 0.522493 |
| severity/recovery-focused impact-veto | 0.708690 | 1.105223 | 0.522644 |
| target-boost focused impact-veto | 0.708620 | 1.105375 | 0.522470 |

判断：target boost 不是新的 affected 最优。它相对 focused 版把 all/unaffected 拉好，但 affected 反而从 `1.105223` 回升到 `1.105375`。因此它更像 balanced v2，而不是事故节点优先 v2。seed 23 单独看很好，三 seed 后说明它有稳定性问题：对 seed 7 的 affected 改善不如 focused 版。

seed 23 分组诊断：

| group | affected delta vs element normal-veto |
|---|---:|
| severity_low | +0.000091 |
| severity_mid | +0.000043 |
| severity_high | -0.000398 |
| recovery_short_lt30 | +0.000067 |
| recovery_mid_30_90 | +0.000083 |
| recovery_long_ge90 | -0.000322 |

这个分组仍然符合课题动机：高严重度、长恢复事故改善更明显。但如果主指标是 affected overall，当前还是 focused impact-veto 更合适；如果希望 all/unaffected 更稳，target-boost 可以作为 balanced 消融。

### Event-conditioned inference calibration

在 focused impact-veto 之后，继续测试了一个不重新训练的推理期校准：使用模型自己预测的 `pred_event_aux`，按预测 severity/recovery 动态放大 normal-veto 回拉强度。

校准形式：

```text
event_boost = 1
  + severity_boost * relu(pred_event_aux_severity)
  + recovery_boost * relu(pred_event_aux_recovery)

normal_veto_amount = base_scale * normal_veto_score * event_boost
```

seed 23 focused 模型 full val/test 小网格：

| 选择 | base scale | severity boost | recovery boost | beta | val all | val affected | test all | test affected | test unaffected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation-selected | 0.25 | 0.0 | 0.0 | 1.05 | 0.714741 | 1.156167 | 0.707456 | 1.102596 | 0.522064 |
| test-best affected | 0.50 | 0.0 | 0.0 | 1.10 | 0.715165 | 1.156392 | 0.707611 | 1.102264 | 0.522448 |
| best event-boosted val row | 0.25 | 0.0 | 0.5 | 1.05 | 0.714716 | 1.156210 | 0.707422 | 1.102655 | 0.521988 |

判断：验证集没有选择任何 severity/recovery event boost；event boost 虽然有时能略降 all/unaffected，但会损害 validation affected。测试集后验最优 affected 也不是 event-conditioned，而只是更大的 `base_scale=0.5,beta=1.10`。因此推理期用预测 event_aux 直接放大 veto 不稳，当前不建议采用。更合理的路线仍是训练期 severity/recovery-focused weighting，或者把 severity/recovery 用作 detector 的显式监督目标，而不是只做后验比例放大。

### Normal-veto 机制诊断

三 seed test 平均：

| subset | base fused MAE | final MAE | base gate | effective gate | veto amount | normal better than base | final better than base |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 0.7093 | 0.7087 | 0.4492 | 0.4312 | 0.0384 | 0.4392 | 0.5144 |
| affected | 1.1062 | 1.1056 | 0.4483 | 0.4276 | 0.0442 | 0.4407 | 0.5092 |
| unaffected | 0.5231 | 0.5225 | 0.4497 | 0.4329 | 0.0357 | 0.4386 | 0.5168 |

机制理解：

1. normal branch 单独并不强，三 seed affected normal MAE 约 `1.2134`，比 fused 差很多。
2. 但在约 `44%` 的逐元素位置上，normal branch 比 base fused 更好。
3. normal-veto 的平均回拉幅度很小，affected `0.0442`、unaffected `0.0357`，只是把 effective incident gate 从约 `0.4483` 降到 `0.4276`。
4. 因此它不是“放弃事故分支”，而是在局部纠正 fused/incident proposal 的过冲；这能稳定改善 affected，但对 unaffected 有轻微代价。

### Focused impact-veto case study

为了确认 focused impact-veto 的收益是否真的来自严重/长恢复事故，而不是只体现在 aggregate 表格里，新增了 seed 23 的逐样本对比可视化：

```text
script:
scripts/compare_focused_veto_case_studies.py

output:
outputs/impact_guided_next_stage/focused_veto_case_studies_seed_23/
```

该脚本对 full test split 的 `27499` 个样本逐一比较：

```text
base:    element continuous normal-veto
focused: severity/recovery-focused impact-conditioned normal-veto
```

每个样本记录 affected/all/unaffected MAE、focused 相对 base 的 affected gain、normal-veto amount 差值、effective gate 差值，并额外筛选 high-severity / long-recovery 的代表性 success 与 boundary cases 画图。

seed 23 test 逐样本分组快照：

| group | samples | positive gain rate | mean affected gain | median affected gain | mean veto diff | mean gate diff |
|---|---:|---:|---:|---:|---:|---:|
| overall | 27499 | 48.69% | +0.000113 | +0.000013 | +0.003123 | -0.001396 |
| severity_high | 9165 | 55.56% | +0.000452 | +0.000099 | +0.003258 | -0.001475 |
| recovery_long_ge90 | 11556 | 54.20% | +0.000344 | +0.000065 | +0.003132 | -0.001395 |
| severity_high_and_long | 8029 | 55.98% | +0.000489 | +0.000111 | +0.003177 | -0.001431 |

代表案例：

| case | sample | group | base affected MAE | focused affected MAE | affected gain | veto diff | gate diff |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 188225 | high severity + long recovery | 5.1639 | 5.1117 | +0.0522 | +0.0167 | -0.0096 |
| 2 | 188411 | high severity + long recovery | 8.6505 | 8.6011 | +0.0493 | +0.0144 | -0.0090 |
| 3 | 185114 | high severity | 12.8186 | 12.7585 | +0.0602 | +0.0163 | -0.0104 |
| 4 | 55485 | high severity + long recovery | 2.8330 | 2.7946 | +0.0385 | +0.0098 | -0.0063 |
| 5 | 187938 | boundary severe/long | 4.0900 | 4.1277 | -0.0377 | +0.0058 | -0.0051 |
| 6 | 185116 | boundary severe/long | 2.3853 | 2.4194 | -0.0341 | +0.0152 | -0.0114 |

图中同时展示 affected-node flow residual 曲线、target residual heatmap、base/focused absolute error、base-minus-focused error gain，以及 focused 相比 base 的 normal-veto amount 变化。

观察：

1. 在 high-severity 和 long-recovery 分组里，focused 的逐样本 mean/median affected gain 都高于 overall，和三 seed 分组结果一致。
2. focused 的平均 normal-veto amount 更高，effective incident gate 更低，说明它学到的是“在事故影响更强的位置更保守地回拉到 normal branch”，而不是简单增强事故分支。
3. 成功案例中，误差改善通常集中在 affected 节点附近的若干 horizon，并非所有节点全局下降；这符合事故影响的局部性。
4. boundary cases 说明该机制仍会过度回拉：即便事件是 high-severity，某些样本中 normal-veto 增强会让 affected MAE 变差。这解释了为什么 focused 版本改善 severe/long，但会牺牲一部分 low/mid 或单点事故样本。

因此，focused impact-veto 可以作为“事故影响导向模型”的更有说服力版本：它不只是降低平均误差，而是把模型容量和门控行为偏向高严重度、长恢复时长事故。不过它还不是最终解，下一步应该围绕 boundary cases 做 detector calibration 或 hard normal-better pretraining。

### Normal-veto detector alignment 诊断

为了解释 focused impact-veto 的成功和失败边界，新增 detector 对齐诊断脚本：

```text
scripts/diagnose_normal_veto_detector_alignment.py

output:
outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23/
```

诊断目标是看 `normal_veto_amount` 是否真的能识别：

```text
normal-better target = base_fused_abs - normal_branch_abs > 0.1
```

seed 23 full test affected 位置：

| model | group | score mean | pos-neg score gap | detector AUC | final gain mean |
|---|---|---:|---:|---:|---:|
| element normal-veto | overall | 0.042515 | 0.010723 | 0.635533 | +0.000477 |
| focused impact-veto | overall | 0.045863 | 0.010117 | 0.629189 | +0.000587 |
| element normal-veto | severity_high | 0.043094 | 0.009924 | 0.626780 | +0.000236 |
| focused impact-veto | severity_high | 0.046416 | 0.009676 | 0.624401 | +0.000557 |
| element normal-veto | recovery_long_ge90 | 0.043039 | 0.010060 | 0.628579 | +0.000379 |
| focused impact-veto | recovery_long_ge90 | 0.046306 | 0.009688 | 0.624703 | +0.000628 |
| element normal-veto | severity_high_and_long | 0.043194 | 0.009779 | 0.625322 | +0.000209 |
| focused impact-veto | severity_high_and_long | 0.046447 | 0.009558 | 0.623314 | +0.000552 |

判断：focused impact-veto 的收益不是因为 detector 排序更准；它的 AUC 和 pos-neg gap 反而略低。它的收益来自更强的 severity/recovery-conditioned 回拉：在 severe/long affected 位置上，平均 final gain 更高。这解释了为什么它能改善 severe/long，但也会产生 boundary cases：当回拉位置不够准时，增强 veto 会过度保守。

### Rank-align detector calibration

基于上面的诊断，又测试了两个校准变体：

```text
rank-align:
focused impact-veto
+ pairwise ranking loss, weight=0.03

low-sparsity rank-align:
focused impact-veto
+ pairwise ranking loss, weight=0.03
+ sparsity loss 0.02 -> 0.01
```

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_rankalign_quickgrid/
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_rankalign_lowsparse_quickgrid/
outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23_rankalign/
outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23_rankalign_lowsparse/
```

seed 23 full test, validation-selected scale/temp/beta 均为 `0.25 / 1.0 / 1.05`：

| model | all MAE | affected MAE | unaffected MAE | affected AUC | severity_high affected AUC | long_recovery affected AUC |
|---|---:|---:|---:|---:|---:|---:|
| focused impact-veto | 0.707456 | 1.102596 | 0.522064 | 0.629189 | 0.624401 | 0.624703 |
| rank-align impact-veto | 0.707439 | 1.102614 | 0.522030 | 0.644820 | 0.638380 | 0.639336 |
| low-sparsity rank-align | 0.707426 | 1.102603 | 0.522018 | 0.644515 | 0.638328 | 0.639029 |

相对 focused impact-veto 的分组 MAE delta：

| model | overall affected | severity_high affected | recovery_long_ge90 affected | all | unaffected |
|---|---:|---:|---:|---:|---:|
| rank-align | +0.000018 | +0.000021 | +0.000026 | -0.000017 | -0.000034 |
| low-sparsity rank-align | +0.000008 | +0.000004 | +0.000013 | -0.000029 | -0.000046 |

判断：pairwise ranking 确实改善了 detector alignment，AUC 从约 `0.629` 提高到约 `0.645`，severe/long 组也提高。但它没有转化为 affected MAE 最优，反而让 affected 略差一点；收益主要体现在 all/unaffected。因此：

```text
affected-oriented v2:
severity/recovery-focused impact-veto

balanced v2:
low-sparsity rank-align impact-veto
```

这轮结果说明，单纯把 detector 排序做准仍不够；最终误差还取决于回拉强度、normal branch 局部偏差，以及 affected 节点上的 residual 幅度。下一步如果继续模型侧突破，不应只调 ranking loss，而应考虑把 detector 从逐元素 BCE/ranking 升级为“节点-事件级 hard normal-better pretraining + 元素级 refinement”，让模型先判断某个事故/节点是否存在 normal 回拉收益，再决定具体 horizon/channel 的回拉强度。

### Hierarchical conservative impact-veto

基于上面的结论，新增了 hierarchical impact-veto：

```text
node-event prior:
先预测某个事故样本/候选节点是否存在 normal-better 回拉收益

element refinement:
再预测具体 horizon/channel 的 normal-veto 强度
```

实现：

```text
model class:
DualBranchSTTISHierarchicalImpactNormalVetoGate

script:
scripts/finetune_sttis_normal_veto_gate.py

key args:
--normal-veto-context impact_aux
--normal-veto-granularity hierarchical
--node-event-veto-loss-weight 0.02
--node-event-veto-positive-fraction 0.05
--sparsity-loss-weight 0.02
```

这个版本先测试了较强 node-event loss `0.05`，发现 AUC 有提升但 severe/long affected 变差，说明 node prior 过强会过度回拉。随后改为更保守的 `node_event_veto_loss_weight=0.02` 和 `sparsity_loss_weight=0.02`。

seed 23 full test：

| model | all MAE | affected MAE | unaffected MAE | affected detector AUC |
|---|---:|---:|---:|---:|
| focused impact-veto | 0.707456 | 1.102596 | 0.522064 | 0.629189 |
| low-sparsity rank-align | 0.707426 | 1.102603 | 0.522018 | 0.644515 |
| hierarchical conservative | 0.707412 | 1.102588 | 0.522004 | 0.636932 |

三 seed full test：

| model | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| focused impact-veto | 0.708690 | 1.105223 | 0.522644 |
| hierarchical conservative | 0.708662 | 1.105163 | 0.522633 |

相对 focused impact-veto，三 seed delta：

| seed | all delta | affected delta | unaffected delta |
|---:|---:|---:|---:|
| 7 | -0.000020 | -0.000133 | +0.000033 |
| 11 | -0.000018 | -0.000041 | -0.000008 |
| 23 | -0.000043 | -0.000008 | -0.000060 |

三 seed 分组 delta：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000027 | -0.000060 | -0.000012 |
| severity_high | -0.000079 | -0.000170 | -0.000019 |
| recovery_long_ge90 | -0.000055 | -0.000118 | -0.000016 |
| severity_low | +0.000008 | +0.000075 | -0.000009 |
| severity_mid | +0.000008 | +0.000037 | -0.000006 |
| recovery_short_lt30 | +0.000005 | +0.000045 | -0.000008 |
| recovery_mid_30_90 | -0.000007 | -0.000006 | -0.000008 |

判断：hierarchical conservative 是目前最符合课题目标的 affected-oriented v2。它三 seed 全部降低 affected MAE，并且收益进一步集中到 `severity_high` 和 `recovery_long_ge90`。低/中严重度和短恢复样本略差，说明模型确实在做“事故影响严重/持续事件优先”的取舍，而不是无差别降噪。和 low-sparsity rank-align 相比，它的 detector AUC 没有那么高，但最终 affected MAE 更好，说明单纯 ranking 准确率不是最终目标，分层 prior 与回拉强度的平衡更重要。

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_{7,11,23}_normal_veto_hierarchical_conservative_quickgrid/
outputs/impact_guided_next_stage/impact_aux_focus_hierarchical_conservative_group_comparison_three_seed_summary.md
outputs/impact_guided_next_stage/hierarchical_conservative_three_seed_summary.csv
```

### Hierarchical conservative case study

为了确认 hierarchical conservative 不是只在总表上出现微小波动，又补了 seed 23 的逐样本 case study。这里的逐样本均值和官方 MAE 口径不同：

```text
official MAE:
先汇总所有有效 horizon/node/channel 元素，再算整体误差

case-study sample mean:
先对每个事故样本算 affected MAE，再对样本平均
```

因此 seed 23 出现了一个有用的现象：官方 affected MAE 是 hierarchical conservative 略好，但逐样本平均 affected gain 接近 0 且略为负。这说明新模块不是“每个事故样本都稳定更好”，而是对部分时空元素的回拉更有效，收益仍然很小。

seed 23 官方元素加权口径：

| group | focused affected | hierarchical affected | delta |
|---|---:|---:|---:|
| overall | 1.102596 | 1.102588 | -0.000008 |
| severity_high | 1.285280 | 1.285279 | -0.000001 |
| recovery_long_ge90 | 1.190491 | 1.190499 | +0.000009 |

seed 23 逐样本 case-study 均值：

| group | affected_gain_mean | win rate | veto diff mean | gate diff mean |
|---|---:|---:|---:|---:|
| overall | -0.000015 | 0.4713 | +0.000893 | -0.000464 |
| severity_high | -0.000031 | 0.4924 | +0.000758 | -0.000363 |
| recovery_long_ge90 | -0.000041 | 0.4907 | +0.000693 | -0.000336 |

代表性成功样本：

| sample | type | focused affected | hierarchical affected | gain |
|---:|---|---:|---:|---:|
| 61886 | high severity + long recovery | 2.1656 | 2.1493 | +0.0163 |
| 61404 | high severity + long recovery | 4.0982 | 4.0826 | +0.0156 |
| 185115 | high severity | 5.4814 | 5.4510 | +0.0303 |
| 187938 | high severity | 4.1277 | 4.0984 | +0.0293 |

代表性边界失败样本：

| sample | type | focused affected | hierarchical affected | gain |
|---:|---|---:|---:|---:|
| 186610 | long recovery | 3.6305 | 3.6629 | -0.0324 |
| 186605 | high severity + long recovery | 3.9540 | 3.9858 | -0.0318 |

判断：hierarchical conservative 可以作为 affected-oriented v2 candidate，但目前还不能说它已经显著解决 case-level robustness。更准确的定位是：它把事故影响先验显式分成 node-event prior 和 element refinement，在三 seed 上带来稳定但极小的 affected 改善，并且三 seed 平均收益集中到严重/长恢复事故；但单个 seed、单个样本上仍有不少反例。下一步如果继续模型侧突破，需要让 node-event detector 更可靠，而不是继续只微调回拉强度。

输出：

```text
outputs/impact_guided_next_stage/hierarchical_conservative_case_studies_seed_23/
```

### Node-event pretrain warmup

基于 case study 的发现，给 hierarchical conservative 增加了一个训练流程改动：正式 normal-veto fine-tune 之前，先只 warm up `node_event_veto_head`。

```text
stage 1:
freeze source model + element normal-veto head
train node-event detector only
target = 当前事故节点是否存在足够比例的 normal-better 元素

stage 2:
恢复原 hierarchical conservative fine-tune
train node-event prior + element refinement + impact/event/node auxiliary heads
```

实现：

```text
scripts/finetune_sttis_normal_veto_gate.py

new args:
--node-event-pretrain-epochs
--node-event-pretrain-lr
```

seed 23 设置：

```text
base config = hierarchical conservative
node_event_pretrain_epochs = 1
node_event_pretrain_lr = 2e-4
```

seed 23 full test，相对无 pretrain 的 hierarchical conservative：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000022 | -0.000096 | +0.000013 |
| severity_high | -0.000073 | -0.000197 | +0.000010 |
| recovery_long_ge90 | -0.000061 | -0.000171 | +0.000008 |
| severity_low | +0.000021 | +0.000038 | +0.000017 |
| recovery_mid_30_90 | +0.000051 | +0.000103 | +0.000030 |

detector alignment, affected subset：

| model | overall AUC | severity_high AUC | recovery_long_ge90 AUC | overall final gain |
|---|---:|---:|---:|---:|
| hierarchical conservative | 0.636932 | 0.631904 | 0.632439 | 0.000594 |
| hierarchical pretrain1 | 0.639769 | 0.635807 | 0.635908 | 0.000690 |

best-F1 selected gain 也提高：

| group | conservative | pretrain1 |
|---|---:|---:|
| overall | 0.001035 | 0.001279 |
| severity_high | 0.001203 | 0.001653 |
| recovery_long_ge90 | 0.001214 | 0.001591 |
| severity_high_and_long | 0.001218 | 0.001679 |

seed 7/11 补完之后，三 seed 结论发生变化：

| model | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| hierarchical conservative | 0.708662 | 1.105163 | 0.522633 |
| hierarchical pretrain1 | 0.708652 | 1.105408 | 0.522502 |

相对 hierarchical conservative，三 seed 平均分组 delta：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000010 | +0.000246 | -0.000131 |
| severity_high | +0.000211 | +0.000665 | -0.000090 |
| recovery_long_ge90 | +0.000143 | +0.000521 | -0.000097 |
| severity_low | -0.000184 | -0.000276 | -0.000161 |
| severity_mid | -0.000140 | -0.000126 | -0.000146 |
| recovery_short_lt30 | -0.000136 | -0.000070 | -0.000156 |
| recovery_mid_30_90 | -0.000255 | -0.000466 | -0.000167 |

主要问题来自 seed 7：

| seed | all delta | affected delta | unaffected delta |
|---:|---:|---:|---:|
| 7 | -0.000040 | +0.000779 | -0.000424 |
| 11 | +0.000030 | +0.000054 | +0.000019 |
| 23 | -0.000022 | -0.000096 | +0.000013 |

seed 7 进一步手动检查了 scale 选择问题：

| scale | beta | all MAE | affected MAE | unaffected MAE |
|---:|---:|---:|---:|---:|
| pretrain1 selected: 0.25 | 1.05 | 0.709979 | 1.108502 | 0.523000 |
| force 0.50 | 1.10 | 0.710041 | 1.107862 | 0.523391 |
| conservative selected: 0.50 | 1.10 | 0.710019 | 1.107723 | 0.523424 |

判断：node-event warmup 有机制价值，但当前 `pretrain1` 配置不能进入主模型。它会把收益推向 all/unaffected 和低/中严重度，反而损害三 seed 平均的高严重度、长恢复期 affected MAE。这个结果说明：先把 node-event detector 预训练得更“自信”不等于最终更懂严重事故；warmup 后 validation sweep 可能选择更小的回拉 scale，例如 seed 7 从 `0.50` 变成 `0.25`，导致 affected 事故节点回拉不足。即使手动改回 `0.50/1.10`，seed 7 affected 仍略差于无 warmup，说明问题不只是 selection，也包括 warmup 权重本身。后续如果继续做 warmup，需要把预训练目标改成 severity/recovery-aware 或高严重样本校准，而不是直接采用这个版本。

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain1_quickgrid/
outputs/impact_guided_next_stage/hierarchical_conservative_pretrain1_group_comparison_seed_23/
outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23_hierarchical_pretrain1/
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_{7,11,23}_normal_veto_hierarchical_pretrain1_quickgrid/
outputs/impact_guided_next_stage/hierarchical_conservative_pretrain1_group_comparison_three_seed_summary.md
outputs/impact_guided_next_stage/seed7_pretrain1_manual_scale_test.csv
```

### Severity/recovery-aware node-event warmup

普通 `pretrain1` 的问题是 warmup 会变成偏 all/unaffected 的泛化降噪器。因此新增了两个 warmup-only 选项：

```text
--node-event-pretrain-affected-only
--node-event-pretrain-event-focus-multiplier
```

这两个参数只影响 node-event warmup，不改变正式 hierarchical fine-tune 的 loss。实验配置：

```text
node_event_pretrain_epochs = 1
node_event_pretrain_affected_only = true
node_event_pretrain_event_focus_multiplier = 3.0
```

seed 23 相对 hierarchical conservative：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000060 | -0.000137 | -0.000024 |
| severity_high | -0.000128 | -0.000253 | -0.000044 |
| recovery_long_ge90 | -0.000110 | -0.000221 | -0.000040 |
| severity_low | +0.000001 | +0.000038 | -0.000009 |
| recovery_mid_30_90 | +0.000001 | +0.000031 | -0.000012 |

seed 23 看起来比普通 `pretrain1` 更符合事故影响目标；它把收益集中到了 `severity_high` 和 `recovery_long_ge90`，同时没有牺牲 unaffected。

但是 seed 7 仍然失败：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000066 | +0.000698 | -0.000425 |
| severity_low | -0.000591 | -0.000847 | -0.000527 |
| severity_mid | -0.000429 | -0.000339 | -0.000471 |
| severity_high | +0.000580 | +0.001896 | -0.000293 |
| recovery_long_ge90 | +0.000392 | +0.001514 | -0.000317 |

seed 7 手动 scale 检查：

| scale | beta | all MAE | affected MAE | unaffected MAE |
|---:|---:|---:|---:|---:|
| selected 0.25 | 1.05 | 0.709953 | 1.108420 | 0.523000 |
| force 0.50 | 1.10 | 0.710033 | 1.107757 | 0.523429 |
| conservative selected 0.50 | 1.10 | 0.710019 | 1.107723 | 0.523424 |

判断：severity/recovery-aware warmup 比普通 `pretrain1` 有进步，至少 seed 23 的方向完全正确，并且 seed 7 在强制 `0.50/1.10` 时几乎追平 conservative。但它仍不能作为主线，因为 seed 7 的默认 validation sweep 会选择更偏 all/unaffected 的 `0.25/1.05`，严重事故 affected 退步明显。暂时不补 seed 11；更优先的下一步不是继续 warmup，而是做 group-aware posthoc selection：选择 scale/beta 时显式考虑 validation 上的 `severity_high affected` 和 `recovery_long_ge90 affected`，避免 selection 把事故收益换成普通节点收益。

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_{7,23}_normal_veto_hierarchical_pretrain_afffocus3_quickgrid/
outputs/impact_guided_next_stage/hierarchical_conservative_pretrain_afffocus3_group_comparison_seed_{7,23}/
outputs/impact_guided_next_stage/seed7_pretrain_afffocus3_manual_scale_test.csv
```

### Group-aware posthoc selection

由于 `afffocus3` 的主要问题不是 seed 23，而是 seed 7 默认 validation sweep 会选择更偏 all/unaffected 的 `0.25/1.05`，新增了一个独立的 posthoc 选择脚本：

```text
scripts/select_group_aware_veto_posthoc.py
```

选择逻辑：

```text
1. 在 validation 上分别评估：
   overall
   severity_high
   recovery_long_ge90

2. 只保留 overall all MAE 距离最优不超过 0.002 的配置

3. 在候选配置中最小化：
   0.5 * overall affected
 + 1.0 * severity_high affected
 + 1.0 * recovery_long_ge90 affected
```

这个规则在 seed 7/11/23 都选择了：

```text
normal_veto_scale = 0.50
normal_veto_temperature = 1.00
residual_beta = 1.10
```

相对 hierarchical conservative，三 seed 平均：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | +0.000263 | -0.000298 | +0.000527 |
| severity_high | -0.000264 | -0.001335 | +0.000446 |
| recovery_long_ge90 | -0.000075 | -0.000958 | +0.000482 |
| severity_low | +0.000675 | +0.001071 | +0.000577 |
| severity_mid | +0.000572 | +0.000583 | +0.000566 |
| recovery_short_lt30 | +0.000500 | +0.000375 | +0.000538 |
| recovery_mid_30_90 | +0.000928 | +0.001596 | +0.000649 |

三 seed overall：

| model | all MAE | affected MAE | unaffected MAE |
|---|---:|---:|---:|
| hierarchical conservative | 0.708662 | 1.105163 | 0.522633 |
| afffocus3 + group-aware posthoc | 0.708926 | 1.104865 | 0.523160 |

判断：这是目前最“事故影响导向”的版本。它不是 all-oriented，也不是平衡版本；它明确牺牲 low/mid/short 和 unaffected，换取 `severity_high`、`recovery_long_ge90` 的 affected 改善。这个方向和课题“学习事故对交通流量的影响程度/持续时间”最贴近，可以作为 severe/long-impact-oriented candidate 保留。论文或汇报时不能说它全面更好，应该说它在严重/长恢复事故场景下更好，并展示 trade-off。

输出：

```text
outputs/impact_guided_next_stage/group_aware_posthoc_seed_{7,11,23}_hierarchical_pretrain_afffocus3/
outputs/impact_guided_next_stage/group_aware_posthoc_afffocus3_three_seed_summary.md
outputs/impact_guided_next_stage/group_aware_posthoc_afffocus3_selected_configs.csv
```

为了让该 posthoc 版本能被后续脚本直接使用，已经将三个 seed 的外部选择结果物化成标准模型目录：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_{7,11,23}_normal_veto_hierarchical_pretrain_afffocus3_groupaware/
```

这些目录中的 `model.pt` 已经写入：

```text
normal_veto_scale = 0.50
normal_veto_temperature = 1.00
residual_beta = 1.10
posthoc_selection_variant = group_aware_severity_recovery
```

seed 23 case study 进一步说明了机制：

| group | affected sample-gain mean | win rate | veto affected diff | gate affected diff |
|---|---:|---:|---:|---:|
| overall | +0.000249 | 0.4702 | +0.0523 | -0.0234 |
| severity_high | +0.002115 | 0.5538 | +0.0550 | -0.0245 |
| recovery_long_ge90 | +0.001584 | 0.5371 | +0.0548 | -0.0241 |
| severity_high_and_long | +0.002286 | 0.5673 | +0.0553 | -0.0245 |
| severity_low | -0.001081 | 0.3909 | +0.0491 | -0.0224 |
| recovery_mid_30_90 | -0.001395 | 0.4263 | +0.0534 | -0.0235 |

代表性成功样本：

| sample | type | conservative affected | group-aware affected | gain |
|---:|---|---:|---:|---:|
| 188411 | high severity + long recovery | 8.6177 | 8.4384 | +0.1793 |
| 55484 | high severity + long recovery | 3.4367 | 3.3294 | +0.1073 |
| 191360 | high severity + long recovery | 1.7950 | 1.6938 | +0.1012 |
| 88938 | high severity | 9.3559 | 9.2565 | +0.0994 |

代表性边界失败样本：

| sample | type | conservative affected | group-aware affected | gain |
|---:|---|---:|---:|---:|
| 59222 | high severity + long recovery | 5.9884 | 6.1149 | -0.1265 |
| 186587 | high severity + short recovery | 6.4420 | 6.5506 | -0.1085 |

解释：group-aware 版本的本质不是学到一个全局更优预测器，而是更激进地提高 normal-veto，从而降低 incident gate。这个策略在严重/长恢复事故中更常降低 affected 误差，但也会误伤一些仍需要 incident branch 的样本。因此它适合定位为 `severe/long-impact-oriented candidate`，不适合替代 all-oriented mainline。

输出：

```text
outputs/impact_guided_next_stage/groupaware_afffocus3_case_studies_seed_23/
outputs/impact_guided_next_stage/hierarchical_conservative_groupaware_afffocus3_group_comparison_seed_23/
```

### Event-conditioned veto boost 复核

在 group-aware 版本之后，继续测试了一个更直接的想法：推理时使用事件级 severity/recovery 信号动态放大 normal-veto。

首先用模型自己预测的 `pred_event_aux` 做动态 boost：

```text
normal_veto_amount = scale * normal_veto_score
                   * (1 + severity_boost * relu(pred_severity)
                        + recovery_boost * relu(pred_recovery))
```

seed 23 `afffocus3` 结果：

| signal | validation selected | val all | val affected | test all | test affected | test unaffected |
|---|---|---:|---:|---:|---:|---:|
| predicted event_aux | no boost, scale=0.25,beta=1.05 | 0.714714 | 1.156160 | 0.707352 | 1.102451 | 0.521979 |
| true event_aux oracle | no boost, scale=0.25,beta=1.05 | 0.714714 | 1.156160 | 0.707352 | 1.102451 | 0.521979 |

测试集后验最优 affected 仍然不是 event-conditioned boost，而是固定更大的 `scale=0.50,beta=1.10`：

| signal | test-best setting | test all | test affected | test unaffected |
|---|---|---:|---:|---:|
| predicted event_aux | boost=0, scale=0.50,beta=1.10 | 0.707528 | 1.102129 | 0.522390 |
| true event_aux oracle | boost=0, scale=0.50,beta=1.10 | 0.707528 | 1.102129 | 0.522390 |

为了确认问题是否来自 `pred_event_aux` 本身，又新增诊断脚本：

```text
scripts/diagnose_event_aux_prediction.py
```

seed 23 test 上，事件辅助预测并非完全无效，但幅度和校准都偏弱：

| model | feature | std pearson | std spearman | high/long AUC | relu pred mean |
|---|---|---:|---:|---:|---:|
| hierarchical conservative | severity | 0.6105 | 0.6306 | 0.7909 | 0.2219 |
| hierarchical conservative | recovery | 0.6152 | 0.6490 | 0.8344 | 0.1597 |
| afffocus3 group-aware | severity | 0.5651 | 0.5805 | 0.7672 | 0.1545 |
| afffocus3 group-aware | recovery | 0.5594 | 0.5880 | 0.8014 | 0.1756 |

关键判断：`pred_event_aux` 有一定排序能力，但还不足以作为逐样本 normal-veto 放大器；更重要的是，即使用真实 `event_aux` 做 oracle boost，validation 也不选择 boost。这说明“事件级严重度/恢复时长”太粗，不能直接决定每个节点、每个 horizon、每个 channel 应该往 normal branch 回拉多少。后续不建议继续做简单的 event-level multiplicative boost；更合理的是把 severity/recovery 用在训练期分组选择、node-event detector 校准，或者作为更细粒度 detector 的辅助监督。

输出：

```text
outputs/impact_guided_next_stage/event_conditioned_afffocus3_seed_23_probe/
outputs/impact_guided_next_stage/event_conditioned_afffocus3_seed_23_oracle_true_probe/
outputs/impact_guided_next_stage/event_aux_prediction_diagnostics_seed_23/
```

### Group-aware normal-veto detector alignment

为了判断 group-aware 的收益来自“触发器排序更准”，还是主要来自“把同一个触发器整体放大”，继续跑了 normal-veto detector alignment：

```text
scripts/diagnose_normal_veto_detector_alignment.py
```

positive target 定义为：

```text
base_abs - normal_abs > 0.10
```

seed 23 test 重点结果：

| model | group | subset | positive rate | veto amount mean | pos-neg gap | hist AUC | final gain mean |
|---|---|---|---:|---:|---:|---:|---:|
| hierarchical conservative | overall | affected | 0.2921 | 0.0466 | 0.0119 | 0.6369 | 0.0006 |
| afffocus3 group-aware | overall | affected | 0.3000 | 0.0998 | 0.0268 | 0.6410 | 0.0019 |
| hierarchical conservative | severity_high | affected | 0.2975 | 0.0470 | 0.0114 | 0.6319 | 0.0006 |
| afffocus3 group-aware | severity_high | affected | 0.3055 | 0.1020 | 0.0261 | 0.6368 | 0.0023 |
| hierarchical conservative | recovery_long_ge90 | affected | 0.2960 | 0.0468 | 0.0114 | 0.6324 | 0.0006 |
| afffocus3 group-aware | recovery_long_ge90 | affected | 0.3041 | 0.1017 | 0.0261 | 0.6369 | 0.0023 |

高精度阈值仍然几乎不可用：

| model | group | hp precision | hp recall | hp selected rate |
|---|---|---:|---:|---:|
| hierarchical conservative | severity_high affected | 0.5320 | 0.000193 | 0.000108 |
| afffocus3 group-aware | severity_high affected | 0.5569 | 0.000131 | 0.000072 |

判断：group-aware 版本确实让 normal-veto 在 normal-better 位置和非 normal-better 位置之间的分数差更大，final gain 也更高；但 detector AUC 只从约 `0.632` 升到约 `0.637`，高精度阈值的 recall 仍接近 0。因此它不是“已经学会了可靠拒绝器”，而是“在弱排序信号上更积极地回拉”。这解释了为什么它能改善 severe/long 的均值，但也会误伤 boundary cases。下一步真正要做的是提高 detector ranking/precision，而不是继续单纯放大 scale。

输出：

```text
outputs/impact_guided_next_stage/normal_veto_detector_alignment_seed_23_groupaware_afffocus3/
```

### Posthoc normal-better detector

为了验证“detector 本身不够强”是否可以通过更直接的 proposal 特征解决，又新增了一个冻结主模型的后验 detector：

```text
scripts/train_posthoc_normal_better_detector.py
```

它不改 normal/incident branch，也不改原 gate，只从冻结模型中抽取 detached 特征训练一个 MLP，目标是判断：

```text
base_abs - normal_abs > positive_margin
```

输入特征包括 normal residual、incident residual、base fused residual、proposal 差异、base gate、原 normal-veto score、normal_delta、预测 impact/node/event auxiliary、node context、horizon/channel 编码。推理时用该 detector 重新生成 normal-veto amount：

```text
prediction = (1 - amount) * base_fused + amount * normal
```

先跑 seed 23 full：

```text
train samples = 20000
epochs = 2
positive_margin = 0.10
```

detector 排序明显变强：

| detector | val AUC | test AUC | test score pos-neg gap |
|---|---:|---:|---:|
| group-aware source veto alignment | - | 0.6410 | 0.0268 |
| posthoc generic detector | 0.7232 | 0.7271 | 0.1835 |
| posthoc severity/recovery-focused detector | 0.7166 | 0.7212 | 0.1625 |

但是 MAE 没有超过 group-aware source。validation 选择都是 `scale=0.10,temp=1.0,beta=1.05`：

| model | test all | test affected | test unaffected |
|---|---:|---:|---:|
| group-aware source | 0.707528 | 1.102129 | 0.522390 |
| posthoc generic detector | 0.707544 | 1.103049 | 0.521980 |
| posthoc severity/recovery-focused detector | 0.707533 | 1.103016 | 0.521980 |

分组显示问题更清楚：

| group | generic affected delta vs source | focused affected delta vs source |
|---|---:|---:|
| severity_low | -0.001327 | -0.001318 |
| severity_mid | -0.000567 | -0.000572 |
| severity_high | +0.002650 | +0.002584 |
| recovery_short_lt30 | -0.000602 | -0.000606 |
| recovery_mid_30_90 | -0.001569 | -0.001545 |
| recovery_long_ge90 | +0.002070 | +0.002013 |
| severity_high_and_long | +0.002828 | +0.002759 |

判断：posthoc detector 的确学到了更强的 normal-better 排序，但这种排序主要帮助 low/mid severity 和 short/mid recovery，反而伤害 severe/high 和 long recovery。也就是说，`normal-better classification AUC` 不是我们最终目标的充分条件；它可能把模型推向“普通场景降噪”，而不是“严重事故影响建模”。这解释了为什么 group-aware source 虽然 detector AUC 不高，却仍然在 severe/long affected 上更好：它的选择目标和训练偏置更贴近事故影响场景。

下一步不建议继续单纯提高 normal-better AUC。更合理的是把 detector 的目标从 generic normal-better 改成 group-conditioned final-MAE gain，例如直接优化 `severity_high/recovery_long affected gain`，或者设计“source veto + posthoc detector”的混合校准，让 posthoc detector 只在 low/short 场景接管，severe/long 仍保留 group-aware source 的强回拉策略。

输出：

```text
outputs/impact_guided_next_stage/posthoc_normal_better_detector_seed_23_full/
outputs/impact_guided_next_stage/posthoc_normal_better_detector_seed_23_focus_full/
```

### Source/Posthoc hybrid 探针

posthoc detector 的问题是：它更适合 low/mid/short 普通事故，却会伤害 severe/long。于是继续测试一个 hybrid：

```text
source = group-aware normal-veto
posthoc = detached normal-better detector

如果样本不是 high severity / long recovery:
    用 posthoc amount
否则:
    保留 source amount
```

这一步分两类：

1. `true_*`：用真实 severity/recovery 分组，只作为 oracle 互补上限。
2. `pred_*`：用模型预测的 `pred_event_aux` 分组，是可部署版本。

新增脚本：

```text
scripts/sweep_hybrid_source_posthoc_detector.py
```

seed 23 full 结果：

| model | test all | test affected | test unaffected | 备注 |
|---|---:|---:|---:|---|
| source group-aware | 0.707528 | 1.102129 | 0.522390 | 当前 severe/long source |
| oracle hybrid best test | 0.707419 | 1.101955 | 0.522311 | true non-high/non-long 才交给 posthoc |
| oracle hybrid val-selected | 0.707576 | 1.102022 | 0.522510 | validation group-aware 选择 |
| deployable pred best test | 0.707500 | 1.102134 | 0.522346 | predicted low/short 接管，几乎持平 |
| deployable val-selected | 0.707528 | 1.102129 | 0.522390 | validation 选择回 source |

oracle hybrid 的分组表现说明互补空间确实存在：

| group | oracle selected affected | source affected | delta |
|---|---:|---:|---:|
| overall | 1.102022 | 1.102129 | -0.000107 |
| severity_low | 0.835765 | 0.836268 | -0.000503 |
| recovery_short_lt30 | 0.937463 | 0.937863 | -0.000400 |
| severity_high | 1.283541 | 1.283541 | 0.000000 |
| recovery_long_ge90 | 1.189193 | 1.189193 | 0.000000 |
| severity_high_and_long | 1.270579 | 1.270579 | 0.000000 |

但可部署 predicted hybrid 不足以进入主线。按 validation group-aware 选择时，它直接选择 source；即便看 test 后验最优，`pred_low_or_short_posthoc` 也只是 overall affected `1.102134`，几乎等于 source `1.102129`，并且 high/long 略有误伤。

判断：hybrid 证明了结构上有互补空间：source 适合 severe/long，posthoc detector 适合 low/short。但当前 `pred_event_aux` 分组不够可靠，不能把 oracle hybrid 变成可部署收益。下一步如果继续模型突破，应优先提升“事件分组/事故强度识别”的可部署可靠性，或者训练一个直接输出 `use_source_vs_posthoc` 的 selector，而不是用粗糙的 predicted severity/recovery 阈值。

输出：

```text
outputs/impact_guided_next_stage/hybrid_source_posthoc_detector_seed_23_full/
```

### Tail-focused normal-veto 复核

继续测试了两种更直接面向 severe/long 的 normal-veto 训练目标。

第一版 `tailfocus_smoke` 显式提高 `severity_high / recovery_long / high_and_long` affected 样本的 final loss 权重，并 boost normal-veto target。结果显示它改善了 low/mid/short 和 unaffected，但 severe/long affected 被伤害：

| group | affected delta vs group-aware source |
|---|---:|
| overall | +0.000381 |
| severity_low | -0.001122 |
| severity_mid | -0.000560 |
| severity_high | +0.001502 |
| recovery_short_lt30 | -0.000500 |
| recovery_mid_30_90 | -0.001394 |
| recovery_long_ge90 | +0.001111 |
| severity_high_and_long | +0.001617 |

随后做 strict high-long posthoc selection，把 `severity_high_and_long` 加入验证集选择，并降低 overall 权重：

```text
selection_groups = overall,severity_high,recovery_long_ge90,severity_high_and_long
selection_weights = 0.25,1.0,1.0,1.0
```

它选择 `scale=0.5,temp=1.0,beta=1.1`，测试集 severe/long 仍然略差：

| group | affected delta vs group-aware source |
|---|---:|
| overall | +0.000129 |
| severity_high | +0.000046 |
| recovery_long_ge90 | +0.000068 |
| severity_high_and_long | +0.000048 |

第二版 `tailconservative_smoke` 改成高危保守目标：高危 affected 样本仍提高 final loss 权重，但不再 boost veto target；同时在 high-risk 上提高 normal-better 判定门槛，并惩罚误触发 normal-veto：

```text
tail_normal_better_margin_add = 0.15
tail_veto_negative_weight = 3.0
tail_sparsity_weight = 2.0
```

完整 test group 对比仍不理想：

| group | affected delta vs group-aware source |
|---|---:|
| overall | +0.000403 |
| severity_low | -0.000886 |
| severity_mid | -0.000309 |
| severity_high | +0.001299 |
| recovery_short_lt30 | -0.000466 |
| recovery_mid_30_90 | -0.000897 |
| recovery_long_ge90 | +0.001037 |
| severity_high_and_long | +0.001420 |

判断：normal-veto 微调无论是 tail-target boost 还是 high-risk conservative regret，都仍然把收益主要转移到 low/mid/short 和 unaffected，不能稳定改善 severe/high 与 long recovery affected。当前不建议继续把 normal-veto 作为 severe/long 主突破口。更合理的下一步是回到主干 gate/incident 分支本身，让模型直接学习高危事故的影响形态，而不是在末端继续学习“是否往 normal 回拉”。

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_tailfocus_smoke/
outputs/impact_guided_next_stage/tailfocus_smoke_groupaware_highlong_selection_seed_23/
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_normal_veto_tailconservative_smoke/
outputs/impact_guided_next_stage/tailconservative_smoke_group_comparison_vs_groupaware_seed_23/
```

### 主干 high-risk incident/gate fine-tune

为了避免末端 normal-veto 把收益转移到 low/short，继续在主干 incident branch + gate fine-tune 中加入高危事故权重：

```text
source = final_convexgate
trainable = incident branch + proposal gate
loss = affected-weighted final residual loss
     + 0.05 * convex gate distillation
     + high-risk affected final-loss weight

severity_high_weight = 0.75
recovery_long_weight = 0.75
high_long_weight = 1.0
```

seed 23 smoke 训练稳定，但完整 test group 对比仍然说明它没有改善 severe/long。相对当前 group-aware source：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | +0.001563 | +0.000945 | +0.001852 |
| severity_low | +0.000886 | -0.001309 | +0.001432 |
| severity_mid | +0.000646 | -0.001470 | +0.001623 |
| severity_high | +0.002784 | +0.003314 | +0.002433 |
| recovery_short_lt30 | +0.000750 | -0.001260 | +0.001368 |
| recovery_mid_30_90 | +0.000241 | -0.002761 | +0.001494 |
| recovery_long_ge90 | +0.002488 | +0.002630 | +0.002398 |
| severity_high_and_long | +0.003077 | +0.003704 | +0.002646 |

判断：简单给主干 final loss 加 high-risk 权重也不够，甚至会更明显地伤害 severe/high 和 long recovery。当前现象已经连续出现三次：posthoc detector、normal-veto tail training、主干 high-risk fine-tune 都更容易改善 low/mid/short，而不是 severe/long。这说明 severe/long 的关键不只是“权重更大”，而是当前分支/门控对高危事故影响形态的表达或选择机制不对。下一步应先做 branch-level group diagnosis：在 severe/long 中分别比较 normal branch、incident branch、learned gate、oracle branch min、oracle convex fusion，确认瓶颈到底是 incident proposal 质量、gate 选择，还是两者都有问题。

输出：

```text
outputs/impact_guided_next_stage/dual_branch_sttis_incident_ft_seed_23_final_convexgate_tail_ft_smoke/
outputs/impact_guided_next_stage/final_convexgate_tail_ft_smoke_group_comparison_vs_groupaware_seed_23/
```

### Branch-level group oracle 诊断

对当前 severe/long source（`hierarchical_pretrain_afffocus3_groupaware`）做 branch/oracle 分组诊断。结果很关键：incident branch 单独在所有组都比 normal branch 更差，但 learned gate 又显著优于 normal branch，oracle convex 仍有巨大空间。

affected group 结果：

| group | normal | incident | learned | oracle min | oracle convex | learned - oracle convex |
|---|---:|---:|---:|---:|---:|---:|
| overall | 1.210753 | 1.320505 | 1.102129 | 0.895003 | 0.772451 | 0.329678 |
| severity_high | 1.427301 | 1.537091 | 1.283541 | 1.053214 | 0.909494 | 0.374047 |
| recovery_long_ge90 | 1.322212 | 1.437077 | 1.189193 | 0.969257 | 0.829376 | 0.359816 |
| severity_high_and_long | 1.420764 | 1.529631 | 1.270579 | 1.040796 | 0.892952 | 0.377627 |
| severity_low | 0.900606 | 0.989012 | 0.836268 | 0.664464 | 0.577127 | 0.259141 |

gate/alpha 诊断显示 learned gate 在高危 affected 上低于 oracle convex alpha：

| group | gate mean | oracle alpha mean | alpha - gate |
|---|---:|---:|---:|
| overall | 0.381168 | 0.453150 | +0.071982 |
| severity_high | 0.379065 | 0.452372 | +0.073308 |
| recovery_long_ge90 | 0.376756 | 0.451515 | +0.074759 |
| severity_high_and_long | 0.377979 | 0.452375 | +0.074396 |
| severity_low | 0.392358 | 0.456432 | +0.064074 |

判断：

1. incident branch 不是“单独更准的事故专家”，但它提供了有价值的 residual 方向；固定融合和 learned gate 都能超过 normal branch。
2. severe/long 的 oracle gap 比 low/short 更大，说明高危事故更需要细粒度连续融合，而不是更强 normal-veto。
3. 不能简单整体把 gate 拉到 0.5，因为 fixed gate 仍明显差于 learned gate。下一步应训练更细粒度的 gate/alpha predictor，尤其是 high-risk convex-alpha distillation，而不是继续训练 incident branch proposal 或末端 veto。

输出：

```text
scripts/diagnose_branch_group_oracle.py
outputs/impact_guided_next_stage/branch_group_oracle_diagnostics_groupaware_seed_23/
```

### Gate-only convex-alpha 微调诊断

基于 branch oracle 诊断，尝试冻结 incident/normal 分支，只训练 gate/proposal-norm，并对 high-risk affected 位置加 convex-alpha distillation。这个方向没有成功。

seed 23 smoke（`groupaware_gate_only_tail_convex_smoke`）在 3000-sample test 上 all 略降，但 affected 变差；完整 test 分组进一步确认 severe/long 也变差：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | +0.000386 | +0.001102 | +0.000050 |
| severity_low | -0.000297 | -0.000537 | -0.000237 |
| severity_mid | -0.000145 | -0.000292 | -0.000078 |
| severity_high | +0.001279 | +0.002575 | +0.000419 |
| recovery_short_lt30 | -0.000130 | -0.000010 | -0.000167 |
| recovery_mid_30_90 | -0.000527 | -0.001034 | -0.000315 |
| recovery_long_ge90 | +0.000992 | +0.002002 | +0.000355 |
| severity_high_and_long | +0.001422 | +0.002711 | +0.000537 |

固定 base/candidate 都用 `residual_beta=1.10` 后仍然全组变差，说明问题不是 beta 选择，而是 gate 微调本身泛化不好。诊断显示 gate mean 确实更接近 oracle alpha，但预测误差反而变差，因此逐点 oracle alpha 目标噪声较大，不能直接作为当前主线。

输出：

```text
outputs/impact_guided_next_stage/groupaware_gate_only_tail_convex_smoke/
outputs/impact_guided_next_stage/groupaware_gate_only_tail_convex_smoke_group_comparison_seed_23/
outputs/impact_guided_next_stage/groupaware_gate_only_tail_convex_smoke_group_comparison_beta110_seed_23/
outputs/impact_guided_next_stage/branch_group_oracle_diagnostics_gate_only_tail_convex_seed_23/
```

### Tail-target impact correction adapter

新的有效方向：冻结当前 group-aware source，只训练一个很小的 impact correction adapter。adapter 输入 source 的 normal/incident proposal、gate、normal-veto、预测 impact/event/node 信号和上下文，输出局部 correction，直接修正最终 residual：

```text
source_pred = beta * source_model(x)
correction = adapter(source details)
final_pred = source_pred + correction
```

关键不是普通 final loss 加权，而是训练时只在 high severity / long recovery 的 affected 节点上加入 correction target：

```text
target = clip(y - source_pred, -max_correction, max_correction)
```

同时对 unaffected 和 non-tail affected 节点加 correction magnitude 保护。这个设计更贴近“学习事故影响幅度/持续时间”，也避免继续把 incident branch 当成独立专家。

seed 23 最平衡配置（10k train smoke, full test group eval）：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000294 | -0.000530 | -0.000183 |
| severity_low | -0.000273 | -0.000563 | -0.000201 |
| severity_mid | -0.000445 | -0.000793 | -0.000285 |
| severity_high | -0.000176 | -0.000339 | -0.000068 |
| recovery_short_lt30 | -0.000368 | -0.000655 | -0.000280 |
| recovery_mid_30_90 | -0.000389 | -0.000910 | -0.000171 |
| recovery_long_ge90 | -0.000216 | -0.000402 | -0.000098 |
| severity_high_and_long | -0.000133 | -0.000292 | -0.000024 |

同配方三 seed（7/11/23）平均：

| group | source affected | adapter affected | affected delta | all delta | unaffected delta |
|---|---:|---:|---:|---:|---:|
| overall | 1.104865 | 1.104592 | -0.000273 | -0.000204 | -0.000172 |
| severity_low | 0.836313 | 0.836002 | -0.000311 | -0.000184 | -0.000152 |
| severity_mid | 0.960241 | 0.959842 | -0.000399 | -0.000273 | -0.000215 |
| severity_high | 1.288900 | 1.288725 | -0.000175 | -0.000157 | -0.000146 |
| recovery_short_lt30 | 0.937695 | 0.937372 | -0.000323 | -0.000221 | -0.000189 |
| recovery_mid_30_90 | 1.028462 | 1.027965 | -0.000498 | -0.000261 | -0.000162 |
| recovery_long_ge90 | 1.193578 | 1.193370 | -0.000208 | -0.000178 | -0.000158 |
| severity_high_and_long | 1.276355 | 1.276202 | -0.000153 | -0.000140 | -0.000131 |

判断：收益幅度不大，但这是目前少数在 overall、affected、unaffected 以及 severe/long/high-and-long 上全部同向改善的模型改动。30k train 的 target1/target2 中等规模也全组改善，但没有明显超过 10k 配方；target2 对 high/high-long 略好，overall/affected 不如 10k target1。下一步应把 adapter 接入正式模型加载/评估链，并做更系统的 target weight、selection metric 和 correction regularization sweep。

后续 seed23 targeted sweep：

1. `protected`：把 unaffected / non-tail affected correction penalty 从 `0.05` 加到 `0.15`，结果 correction 几乎被压到 0，不值得继续。
2. `highfocus_loss`：把 severity/recovery/high-long 权重从 `2/2/3` 加到 `4/4/6`，按 total loss 选 checkpoint。仍全组改善，但不如原 balanced tailtarget。
3. `highfocus_finalselect`：同样 `4/4/6`，但按 `val_final_loss` 选 checkpoint。affected 端略优于 balanced，尤其 high/high-long affected；但 high severity、long recovery、high-and-long 的 unaffected 变差，导致 all delta 不如 balanced。
4. `highfocus_nodegate025`：在 highfocus 基础上，用 source 的 predicted affected-node probability 对 correction 做 sigmoid node gate，gate floor=0.25。结果没有解决 highfocus 的副作用，high-risk unaffected 反而更差。

seed23 affected delta 对比：

| group | balanced | highfocus loss | highfocus finalselect |
|---|---:|---:|---:|
| overall | -0.000530 | -0.000428 | -0.000546 |
| severity_high | -0.000339 | -0.000320 | -0.000369 |
| recovery_long_ge90 | -0.000402 | -0.000372 | -0.000427 |
| severity_high_and_long | -0.000292 | -0.000295 | -0.000321 |

但 highfocus finalselect 的 high-risk unaffected delta：

| group | unaffected delta |
|---|---:|
| severity_high | +0.000093 |
| recovery_long_ge90 | +0.000058 |
| severity_high_and_long | +0.000148 |

node-gated highfocus 的 seed23 full-group 结果：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000133 | -0.000443 | +0.000012 |
| severity_high | -0.000040 | -0.000329 | +0.000152 |
| recovery_long_ge90 | -0.000058 | -0.000348 | +0.000125 |
| severity_high_and_long | +0.000006 | -0.000273 | +0.000197 |

同源模型在 test set 上的 predicted affected-node probability 分布：

| node type | mean | q10 | q25 | q50 | q75 | q90 |
|---|---:|---:|---:|---:|---:|---:|
| true affected | 0.372791 | 0.174621 | 0.243438 | 0.340948 | 0.473382 | 0.626301 |
| true unaffected | 0.186306 | 0.052938 | 0.072539 | 0.172422 | 0.261458 | 0.348834 |

判断：node probability 有一定区分度，但 affected/unaffected 分布仍明显重叠，直接把它作为 correction gate 不够可靠。这个结果进一步支持：当前不应该依赖显式事故检测式定位，而应保留 balanced tail correction 这种更温和的影响修正。

signal diagnostics 显示，`branch_delta_abs = |incident branch - normal branch|` 比 predicted node probability 更像事故影响强度信号：

| group | node type | node_prob mean | branch_delta_abs mean | correction_abs mean |
|---|---|---:|---:|---:|
| overall | affected | 0.372786 | 0.945517 | 0.014124 |
| overall | unaffected | 0.186302 | 0.436801 | 0.009042 |
| severity_high_and_long | affected | 0.414974 | 1.118156 | 0.017658 |
| severity_high_and_long | unaffected | 0.197893 | 0.466553 | 0.011711 |

据此新增 `anomgate05`：在 highfocus correction 上加入 branch-disagreement anomaly gate。

```text
correction_anomaly_gate = floor + (1 - floor) * sigmoid((|incident - normal| - threshold) / temperature)
threshold = 0.5
temperature = 0.25
floor = 0.25
```

三 seed 结果：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000441 | -0.000484 | -0.000421 |
| severity_high | -0.000397 | -0.000233 | -0.000505 |
| recovery_long_ge90 | -0.000455 | -0.000332 | -0.000533 |
| severity_high_and_long | -0.000369 | -0.000163 | -0.000510 |

对比原 `balanced_default` 三 seed：

| model | overall all | overall affected | overall unaffected | high_and_long all | high_and_long affected | high_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| anomgate05 | -0.000441 | -0.000484 | -0.000421 | -0.000369 | -0.000163 | -0.000510 |
| balanced_default | -0.000204 | -0.000273 | -0.000172 | -0.000140 | -0.000153 | -0.000131 |

seed-level caveat：`anomgate05` 的 overall affected 三 seed 都改善，但 high-risk affected 在 seed11 上有小幅回退：

| seed | group | affected delta |
|---:|---|---:|
| 7 | severity_high_and_long | -0.000406 |
| 11 | severity_high_and_long | +0.000176 |
| 23 | severity_high_and_long | -0.000259 |

判断：`anomgate05` 是目前最强的 all-oriented / unaffected-protected 候选，且三 seed overall affected 也更好；但它不是纯 severe/long affected 最稳版本。更准确的定位是：用分支分歧作为事故影响 correction 的触发器，比直接用事故节点概率更可靠。

`anomgate05` paired sample-level 统计进一步确认这个定位：

| group | target | seed mean delta | seed std | all seeds improved | mean improve rate |
|---|---|---:|---:|---|---:|
| overall | all | -0.000385 | 0.000184 | True | 0.562820 |
| overall | affected | -0.000396 | 0.000063 | True | 0.541220 |
| overall | unaffected | -0.000403 | 0.000271 | True | 0.562467 |
| severity_high | all | -0.000361 | 0.000140 | True | 0.542826 |
| severity_high | affected | -0.000167 | 0.000361 | False | 0.519355 |
| severity_high | unaffected | -0.000511 | 0.000472 | True | 0.560041 |
| recovery_long_ge90 | all | -0.000430 | 0.000218 | True | 0.555757 |
| recovery_long_ge90 | affected | -0.000246 | 0.000180 | True | 0.528233 |
| recovery_long_ge90 | unaffected | -0.000541 | 0.000470 | True | 0.565837 |
| severity_high_and_long | all | -0.000337 | 0.000149 | True | 0.537593 |
| severity_high_and_long | affected | -0.000093 | 0.000408 | False | 0.513763 |
| severity_high_and_long | unaffected | -0.000520 | 0.000521 | True | 0.556693 |

paired stats 的解释：`anomgate05` 的改善样本比例不像 balanced 那样在所有 affected 组都更平均；它主要把 unaffected 和 all 指标拉稳，同时保留 overall affected 收益。seed11 的 severity_high/high-and-long affected 是必须在汇报中主动说明的弱点。

进一步对 seed11 做 posthoc anomaly-gate sweep（不重训，只改变推理时 threshold/floor）：

| setting | overall all | overall affected | overall unaffected | high_and_long all | high_and_long affected | high_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| balanced backup | -0.000211 | -0.000149 | -0.000240 | -0.000182 | -0.000045 | -0.000275 |
| anomgate threshold=0.5 floor=0.25 | -0.000689 | -0.000518 | -0.000769 | -0.000588 | +0.000176 | -0.001112 |
| posthoc threshold=0.3 floor=0.25 | -0.000739 | -0.000558 | -0.000823 | -0.000632 | +0.000167 | -0.001180 |
| posthoc threshold=0.7 floor=0.25 | -0.000629 | -0.000472 | -0.000703 | -0.000529 | +0.000189 | -0.001022 |

在 `threshold={0.0,0.3,0.5,0.7}`、`floor={0.1,0.25,0.5}` 的 sweep 中，没有任何设置能把 seed11 `severity_high_and_long affected` 变成负 delta；最好的 affected delta 仍为 `+0.000153`。这说明 seed11 high-risk affected 的问题不是简单 gate threshold/floor 能解决的，而更可能来自 raw correction 在该组上的方向/幅度偏差。当前不建议为了修这个 caveat 继续调 anomaly gate 阈值。

alignment diagnostics 进一步说明了 seed11 caveat 的原因：

| group | target | mean improvement | sign match rate | beneficial rate | harmful rate |
|---|---|---:|---:|---:|---:|
| severity_high | affected | -0.000083 | 0.508788 | 0.503765 | 0.496233 |
| severity_high_and_long | affected | -0.000176 | 0.508697 | 0.503440 | 0.496558 |
| severity_high_and_long | unaffected | +0.001112 | 0.518680 | 0.514014 | 0.485983 |

解释：high-risk affected 不是大面积方向错，而是 beneficial/harmful 几乎五五开，少量 harmful correction 的幅度更大；unaffected 则受益更明显。因此只调 gate 很难修，应该约束 correction 本身的“有害幅度”。

尝试加入 high-risk affected correction-regret loss：

```text
regret = relu(|target_correction - correction| - |target_correction|)
target_correction = y - source_pred
mask = high severity / long recovery affected nodes
```

seed11 结果：

| variant | overall all | overall affected | overall unaffected | high_and_long all | high_and_long affected | high_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| anomgate05 | -0.000689 | -0.000518 | -0.000769 | -0.000588 | +0.000176 | -0.001112 |
| regret_weight=0.05 | -0.000508 | -0.000410 | -0.000554 | -0.000471 | +0.000055 | -0.000831 |
| regret_weight=0.5 | -0.000003 | -0.000001 | -0.000003 | -0.000002 | +0.000001 | -0.000004 |

判断：regret loss 方向是对的，可以把 seed11 high-risk affected 回退明显压小；但权重稍大就会把 correction 压成近似 0，且 `0.05` 也会牺牲 anomgate 的 all/unaffected 收益。当前不把 regret 版升级为主线，只把它作为下一轮“有害 correction 抑制”的线索。

balanced/default 更长训练验证（seed23，10 epochs，其他配置不变）：

| group | all delta | affected delta | unaffected delta |
|---|---:|---:|---:|
| overall | -0.000215 | -0.000391 | -0.000132 |
| severity_high | -0.000156 | -0.000264 | -0.000084 |
| recovery_long_ge90 | -0.000185 | -0.000310 | -0.000106 |
| severity_high_and_long | -0.000134 | -0.000234 | -0.000066 |

对比 5 epoch balanced/default，10 epoch 的 unaffected 保护略好，但 overall affected、severity_high affected、recovery_long affected 都更弱，没有放大收益。因此当前瓶颈不是单纯训练轮数，默认仍保留 5 epoch 的 `balanced_default`。

因此当前推荐：

```text
new all-oriented main candidate:
highfocus_anomgate05
severity_high_weight=4
recovery_long_weight=4
high_long_weight=6
unaffected/non-tail protection=0.05
selection_loss_key=final_loss
correction_anomaly_gate_mode=branch_delta_abs
threshold=0.5
temperature=0.25
floor=0.25

previous balanced backup:
tailtarget_balanced
severity_high_weight=2
recovery_long_weight=2
high_long_weight=3
unaffected/non-tail protection=0.05
selection_loss_key=loss

affected-oriented adapter:
highfocus_finalselect
severity_high_weight=4
recovery_long_weight=4
high_long_weight=6
unaffected/non-tail protection=0.05
selection_loss_key=final_loss
```

进一步 formal sweep（seed23）验证 `max_correction`、`target_margin`、`selection_loss_key`：

| config | 结论 |
|---|---|
| balanced_default | 默认主线，所有 group 的 all/affected/unaffected 都改善 |
| balanced_finalselect | 和 balanced_default 基本相同，因为两者都选到同一轮 |
| margin0 | 和 default 接近，没有明显收益 |
| margin005 | target 太稀疏，affected 收益下降 |
| max05 | 低幅 correction 消融；seed23 更保守，但三 seed 平均弱于 balanced |
| max12 | affected overall 更强，但 high-risk unaffected 变差，severity_high/recovery_long/high_and_long 的 all delta 转正 |

关键 full-group 对比：

| config | overall affected | severity_high affected | recovery_long affected | high_and_long affected | high_and_long all | high_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| balanced_default | -0.000530 | -0.000339 | -0.000402 | -0.000292 | -0.000133 | -0.000024 |
| max05 | -0.000316 | -0.000208 | -0.000250 | -0.000184 | -0.000174 | -0.000167 |
| max12 | -0.000639 | -0.000239 | -0.000338 | -0.000134 | +0.000262 | +0.000534 |
| highfocus_finalselect | -0.000546 | -0.000369 | -0.000427 | -0.000321 | -0.000043 | +0.000148 |

补完 `max05` 的 seed 7/11 复验后，三 seed 均值如下：

| config | overall all | overall affected | overall unaffected | high_and_long all | high_and_long affected | high_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| balanced_default | -0.000204 | -0.000273 | -0.000172 | -0.000140 | -0.000153 | -0.000131 |
| max05_conservative | -0.000113 | -0.000131 | -0.000105 | -0.000090 | -0.000070 | -0.000104 |

当时结论：在尚未加入 branch-disagreement anomaly gate 前，`balanced_default` 是默认模型；`max05` 三 seed 稳定同向改善，但收益明显小于 balanced，因此只作为 low-correction/conservative ablation，不再作为 deployment backup；`highfocus_finalselect` 可以作为 affected-oriented ablation，但不应作为 balanced 主线；`max12` 不建议继续。后续 `anomgate05` 已把 all-oriented 主候选从 `balanced_default` 更新为 branch-disagreement gate 版本。

补充 paired sample-level 统计（`balanced_default`，三 seed，test split）：

这里统计的是每个测试事件自己的 source MAE 与 adapter MAE 差值，`delta = adapter - source`，负数表示 adapter 更好。置信区间是 event-level delta 的正态近似，只作为辅助证据，不等价于严格的 seed-level 显著性检验。

| group | target | seed mean delta | seed std | all seeds improved | mean improve rate |
|---|---|---:|---:|---|---:|
| overall | all | -0.000199 | 0.000087 | True | 0.599295 |
| overall | affected | -0.000238 | 0.000182 | True | 0.558647 |
| overall | unaffected | -0.000182 | 0.000074 | True | 0.591209 |
| severity_high | all | -0.000152 | 0.000041 | True | 0.581597 |
| severity_high | affected | -0.000142 | 0.000132 | True | 0.538601 |
| recovery_long_ge90 | all | -0.000175 | 0.000059 | True | 0.589448 |
| recovery_long_ge90 | affected | -0.000168 | 0.000139 | True | 0.545447 |
| severity_high_and_long | all | -0.000136 | 0.000036 | True | 0.574916 |
| severity_high_and_long | affected | -0.000120 | 0.000115 | True | 0.533856 |
| severity_high_and_long | unaffected | -0.000150 | 0.000121 | True | 0.584654 |

判断：paired stats 支持当前主线“稳定小幅改善”的定位。三 seed 的重点组均值都是负数，且改善样本比例多数在 53%-60% 左右；但 high-and-long affected 的 seed std 接近 mean，说明 severe/long affected 的优势仍偏弱，需要继续扩大训练或增强 tail correction，不能宣称已经显著解决严重事故预测。

最终候选表（impact correction）：

| candidate | role | evidence | overall all | overall affected | overall unaffected | high_and_long all | high_and_long affected | high_and_long unaffected | decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| anomgate05 | new all-oriented main candidate | 3-seed mean | -0.000441 | -0.000484 | -0.000421 | -0.000369 | -0.000163 | -0.000510 | 当前最强 all/unaffected 候选；需说明 high-risk affected seed11 caveat |
| balanced_default | previous balanced default | 3-seed mean | -0.000204 | -0.000273 | -0.000172 | -0.000140 | -0.000153 | -0.000131 | 更简单、更温和的备份 |
| max05_conservative | low-correction conservative ablation | 3-seed mean | -0.000113 | -0.000131 | -0.000105 | -0.000090 | -0.000070 | -0.000104 | 稳定但弱于 balanced，保留为消融 |
| highfocus_affected | affected-oriented ablation | seed23 | -0.000224 | -0.000546 | -0.000072 | -0.000043 | -0.000321 | +0.000148 | 只用于展示 affected/unaffected tradeoff |
| max12_not_recommended | rejected strong-correction ablation | seed23 | -0.000170 | -0.000639 | +0.000050 | +0.000262 | -0.000134 | +0.000534 | 不建议 |

当前最合理的下一步：

1. 若追求默认主线：把 `anomgate05` 作为新的 all-oriented 主候选，补 paired stats，确认改善样本比例和 seed-level caveat。
2. 若追求 severe/long affected：不要继续只调 anomaly gate threshold/floor；下一步应把 regret/no-harm 约束做得更柔和，例如只惩罚 top harmful correction 或使用更小权重/分段权重，避免 correction 整体塌缩。
3. 若写模型消融：保留 `balanced_default`、`max05_conservative`、`highfocus_affected`、`highfocus_nodegate025` 和 `max12_not_recommended`，用于说明 correction 幅度、affected emphasis、node-gated localization、branch-disagreement gate 与 unaffected 保护之间的 tradeoff。

输出：

```text
scripts/train_impact_correction_adapter.py
scripts/evaluate_impact_correction_adapter.py
scripts/summarize_impact_correction_candidates.py
scripts/diagnose_impact_correction_signals.py
scripts/sweep_impact_correction_anomaly_gate.py
scripts/diagnose_impact_correction_alignment.py
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_smoke_seed_23/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_seed_7/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_seed_11/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_smoke_seed_23/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_seed_7/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_seed_11/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_regret005_seed_11/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_regret05_seed_11/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_max05_seed_7/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_max05_seed_11/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_nodegate025_smoke_seed_23/
outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_e10_seed_23/
outputs/impact_guided_next_stage/impact_correction_signal_diagnostics_balanced_seed23/
outputs/impact_guided_next_stage/impact_correction_tailtarget_3seed_summary/
outputs/impact_guided_next_stage/impact_correction_max05_3seed_summary/
outputs/impact_guided_next_stage/impact_correction_balanced_paired_stats/
outputs/impact_guided_next_stage/impact_correction_anomgate05_paired_stats/
outputs/impact_guided_next_stage/anomaly_gate_posthoc_sweep_seed11/
outputs/impact_guided_next_stage/impact_correction_alignment_anomgate05_seed11/
outputs/impact_guided_next_stage/impact_correction_seed23_targeted_sweep/
outputs/impact_guided_next_stage/impact_correction_seed23_formal_sweep/
outputs/impact_guided_next_stage/impact_correction_final_candidates/
```

### 当前模型结论更新

当前建议保留三条模型线：

```text
all-oriented mainline:
incident-ft final-only
+ convex gate distillation
+ validation-selected posthoc bias calibration

affected-oriented alternative:
incident-ft final-only
+ convex gate distillation
+ continuous two-stage normal-veto

affected-oriented v2 candidate:
incident-ft final-only
+ convex gate distillation
+ hierarchical conservative severity/recovery-focused impact-conditioned normal-veto

balanced v2 candidate:
incident-ft final-only
+ convex gate distillation
+ severity/recovery-focused impact-conditioned normal-veto
+ weak pairwise rank-align
+ lower sparsity penalty

severe/long-impact-oriented candidate:
incident-ft final-only
+ convex gate distillation
+ hierarchical conservative impact-conditioned normal-veto
+ affected-only severity/recovery-aware node-event warmup
+ group-aware posthoc scale/beta selection

impact-correction candidate:
group-aware source
+ frozen-source local impact correction adapter
+ tail-only correction target on high severity / long recovery affected nodes
+ unaffected and non-tail correction magnitude protection
+ branch-disagreement anomaly gate for correction confidence (anomgate05)
```

不建议当前主线采用：

```text
hard-negative gate loss
thresholded disagreement/magdiff gate cap
learned veto gate
branch-confidence gate v1
branch-uncertainty gate v1
proposal-aware uncertainty gate v1
local selector gate v1
local selector + hard replay v1
hard-threshold normal-veto
binary normal-veto detector v1
node-event normal-veto v1
node-event pretrain1 warmup as mainline
rank-align impact-veto as affected-oriented mainline
event-conditioned normal-veto boost with predicted/true event_aux
posthoc generic normal-better detector as severe/long model
deployable source/posthoc hybrid as mainline
tail-focused normal-veto as severe/long model
high-risk conservative normal-veto as severe/long model
main-branch high-risk incident/gate fine-tune as severe/long model
gate-only high-risk convex-alpha distillation as severe/long model
node-gated highfocus correction as default
```

真正未解决的问题仍是 case-level robustness：少数 incident branch proposal 极差时，模型缺少可靠的局部拒绝机制。Oracle 上限说明这不是“两个分支完全没有互补信息”，而是 gate 的局部选择/插值能力不够。convex-gate 已经证明“连续融合位置监督”有小幅稳定收益；continuous normal-veto 进一步说明“局部往 normal 回拉”可以稳定改善 affected。branch-confidence/uncertainty/proposal-uncertainty v1 则说明弱 adapter 很难学到可靠的极端样本拒绝规则。local selector + hard replay 说明强行喂 hard normal 样本可以救个别极端 case，但会破坏整体分布。最新 oracle sweep 进一步确认高精度 veto 的上限很高，但 binary/continuous high-precision、pairwise ranking、regret regression、node-event normal-veto 都说明仅靠改变 loss 形式或输出粒度还不够。impact-conditioned normal-veto 说明显式事故影响辅助监督能稳定改善 affected，但幅度仍很小。group-aware posthoc 说明 severe/long 事故可以被定向改善，但要牺牲 low/mid/short 和 unaffected。event-conditioned boost 进一步说明，事件级 severity/recovery 适合做分组选择和训练监督，不适合直接作为推理期全局回拉倍率。posthoc normal-better detector 说明，即使 detector AUC 明显提高，也可能把收益转移到 low/short 普通事故而伤害 severe/long。hybrid oracle 说明 source/posthoc 有互补空间，但 predicted event grouping 还不能稳定部署。tail-focused 和 high-risk conservative normal-veto 进一步说明，末端 veto/回拉机制很难把收益稳定投向 severe/high 与 long recovery。主干 high-risk fine-tune 进一步说明，单纯加大高危 final loss 权重也不能解决 severe/long，甚至会破坏高危组。gate-only convex-alpha 微调说明，直接拟合逐点 oracle alpha 会更接近 oracle gate 均值但预测泛化变差。最新 impact correction adapter 说明，把问题改写成冻结 source 后的局部事故影响幅度修正更稳；其中 `anomgate05` 进一步说明，使用 normal/incident 分支分歧作为事故影响 correction 的触发信号，比直接使用 predicted affected-node probability 更可靠。目前应把 `anomgate05` 作为 all-oriented 主候选，同时保留 `balanced_default` 作为更温和备份；下一步如果继续模型突破，重点是缓解 `anomgate05` 在 seed11 high-risk affected 上的小幅回退。
