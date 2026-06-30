# ST-TIS gate-head 微调实验报告

## 目的

ST-TIS-style incident branch 已经提升了事故分支表达能力，但 mixed case study 仍显示 learned gate 在少数样本上会过度相信局部较差的 incident branch。前一轮尝试了两类更强的训练约束：

- branch loss：强制 normal branch 和 incident branch 单独拟合 residual；
- oracle-style gate alignment：用两个分支的 detached 局部误差构造 gate target。

结果显示二者都会明显破坏最终融合。因此本轮改为更保守的校准方式：冻结已训练好的 normal/incident residual branches，只微调 gate head。

## 新增脚本

- `scripts/finetune_sttis_gate_head.py`
- `scripts/sweep_sttis_gate_posthoc.py`
- `scripts/train_dual_branch_sttis_calibrated_gate.py`

其中最终有效的是 `finetune_sttis_gate_head.py`。它从已有 ST-TIS checkpoint 出发：

1. 加载完整 ST-TIS dual-branch checkpoint；
2. 冻结 normal encoder/decoder、incident ST-TIS branch 和 residual decoders；
3. 只训练 `gate_head`，共 28,932 个参数；
4. 训练目标仍是 no-aux residual forecasting loss，不使用 affected label 或 impact label；
5. 训练 3 epoch，学习率 `1e-4`。

## 主结果

| Model | Seed | Beta | All MAE | Affected MAE | Unaffected MAE | All gain | Affected gain |
|:--|--:|--:|--:|--:|--:|--:|--:|
| old dual-branch gate no-aux | 7 | 1.05 | 0.7181 | 1.1234 | 0.5279 | 13.78% | 13.17% |
| old dual-branch gate no-aux | 11 | 1.05 | 0.7198 | 1.1296 | 0.5276 | 13.57% | 12.69% |
| old dual-branch gate no-aux | 23 | 1.00 | 0.7166 | 1.1188 | 0.5279 | 13.96% | 13.53% |
| ST-TIS gate no-aux | 7 | 1.10 | 0.7153 | 1.1240 | 0.5236 | 14.11% | 13.13% |
| ST-TIS gate no-aux | 11 | 1.00 | 0.7161 | 1.1203 | 0.5265 | 14.01% | 13.41% |
| ST-TIS gate no-aux | 23 | 1.00 | 0.7135 | 1.1217 | 0.5220 | 14.32% | 13.30% |
| ST-TIS gate-head fine-tune | 7 | 1.05 | 0.7150 | 1.1239 | 0.5232 | 14.15% | 13.13% |
| ST-TIS gate-head fine-tune | 11 | 1.00 | 0.7138 | 1.1189 | 0.5238 | 14.29% | 13.52% |
| ST-TIS gate-head fine-tune | 23 | 1.00 | **0.7116** | 1.1189 | **0.5205** | **14.56%** | 13.52% |

三随机种子汇总：

| Model | All MAE | Affected MAE | Unaffected MAE |
|:--|--:|--:|--:|
| old dual-branch gate no-aux | 0.7182 ± 0.0016 | 1.1240 ± 0.0054 | 0.5278 ± 0.0002 |
| ST-TIS gate no-aux | 0.7150 ± 0.0013 | 1.1220 ± 0.0019 | 0.5240 ± 0.0023 |
| ST-TIS gate-head fine-tune | **0.7135 ± 0.0017** | **1.1206 ± 0.0029** | **0.5225 ± 0.0017** |

结论：gate-head-only fine-tune 是目前最强版本。它不仅进一步降低 All MAE，也把三 seed 平均 affected MAE 从 1.1220 降到 1.1206。

## 失败路线对照

| Variant | Seed | All MAE | Affected MAE | 结论 |
|:--|--:|--:|--:|:--|
| ST-TIS gate no-aux | 23 | 0.7135 | 1.1217 | 当前强基线 |
| branch loss only | 23 | 0.7443 | 1.1619 | 明显退化 |
| branch loss + gate alignment | 23 | 0.7446 | 1.1616 | 明显退化 |
| post-hoc temp=0.75 | 23 | 0.7134 | 1.1215 | 轻微提升，不足以作为主模型 |
| gate-head fine-tune | 23 | **0.7116** | **1.1189** | 明显提升 |

这个对照说明：不能强迫两个分支都单独变成好 predictor。两个分支在融合框架中应保持互补角色；更有效的做法是保留分支表示，只微调 gate 的融合策略。

## 解释性分析

以 seed 23 为例，gate-head fine-tune 后：

| Inference mode | All MAE | Affected MAE | Unaffected MAE |
|:--|--:|--:|--:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.7777 | 1.2133 | 0.5733 |
| ST-TIS incident residual only | 0.8044 | 1.2443 | 0.5980 |
| Fixed gate = 0.5 | 0.7244 | 1.1439 | 0.5276 |
| Learned gate before fine-tune | 0.7135 | 1.1217 | 0.5220 |
| Learned gate after fine-tune | **0.7116** | **1.1189** | **0.5205** |

单分支指标保持不变，fixed gate 指标也不变，只有 learned gate 变好。这说明收益确实来自 gate 融合策略改善，而不是重新训练分支造成的参数量或表示变化。

Gate 与局部更优分支的一致性也增强：

| Subset | Case | Before | After |
|:--|:--|--:|--:|
| Affected | incident branch better | 0.4502 | 0.4741 |
| Affected | normal branch better | 0.4172 | 0.4376 |
| Unaffected | incident branch better | 0.4424 | 0.4773 |
| Unaffected | normal branch better | 0.4229 | 0.4547 |

整体 gate 均值有所上升，但 incident-better 和 normal-better 之间的差距也扩大，说明 gate-head fine-tune 不是简单加大事故分支权重，而是改善了局部选择。

## Mixed case 观察

sample 88134 仍然是 failure，但已有缓解：

| Model | Learned affected MAE | Fixed affected MAE | Normal-only | Incident-only | Mean affected gate |
|:--|--:|--:|--:|--:|--:|
| ST-TIS gate | 6.8554 | 4.9938 | 2.1877 | 8.0691 | 0.5124 |
| ST-TIS gate-head fine-tune | **6.4817** | 4.9938 | 2.1877 | 8.0691 | **0.4453** |

这说明 gate-head fine-tune 能降低一部分对 incident branch 的过信，但还没有完全解决短恢复、单节点影响样本中的局部失败。

## 当前结论

1. 当前主模型建议更新为 `ST-TIS gate-head fine-tune`。
2. 论文贡献可以表述为两阶段训练：先训练 residual branches 与 gate，再冻结分支微调 gate head。
3. 该方法仍是 no-aux，不使用 affected label、impact label 或未来事故影响标签。
4. branch loss / oracle gate alignment 不应作为主方法，因为它们明显破坏融合。
5. 后续真正值得研究的是更细粒度的 branch reliability，而不是直接让分支单独拟合或用 oracle error 强行监督 gate。

## 可引用文件

- `scripts/finetune_sttis_gate_head.py`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_7/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_11/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_23/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_23_interpretability/summary.md`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_head_finetune_seed_23_case_studies_mixed/case_study_report.md`
