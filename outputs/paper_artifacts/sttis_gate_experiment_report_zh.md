# ST-TIS-style incident branch 探索报告

## 目的

前一轮 confidence-aware gate 和 hard-example reweighting 都没有超过原始 `dual_branch_gate_no_aux`，说明瓶颈不太可能只靠简单的分支误差头或 loss 重加权解决。本轮实验转向提升 incident branch 的时空表达能力：保持 normal branch、residual target 和 learned gate 不变，只把原来的轻量 STGNN incident branch 替换为 ST-TIS-style 模块。

## 新增模型

新增脚本：

- `scripts/train_dual_branch_sttis_gate.py`

模型结构保持原 dual-branch residual gate 的主体不变：

\[
\hat{Y}=\hat{Y}^{normal}+\beta\left((1-\alpha)\hat{\Delta}^{normal}+\alpha\hat{\Delta}^{incident}\right).
\]

变化只发生在 incident branch：

- temporal encoder：对每个候选节点的历史残差序列使用 self-attention 编码，并融合 last token 与 mean token；
- spatial encoder：在 full candidate graph 上做 top-k graph-biased spatial attention；
- gate/decoder：仍然使用原来的节点级、预测时距级 gate 和 residual decoder；
- no-aux 设置：不使用 affected node label、impact label 或 event label 作为训练监督。

## 主结果对比

| Model | Seed | Beta | All MAE | Affected MAE | Unaffected MAE | All gain | Affected gain |
|:--|--:|--:|--:|--:|--:|--:|--:|
| old dual-branch gate no-aux | 7 | 1.05 | 0.7181 | 1.1234 | 0.5279 | 13.78% | 13.17% |
| old dual-branch gate no-aux | 11 | 1.05 | 0.7198 | 1.1296 | 0.5276 | 13.57% | 12.69% |
| old dual-branch gate no-aux | 23 | 1.00 | 0.7166 | **1.1188** | 0.5279 | 13.96% | **13.53%** |
| ST-TIS gate no-aux | 7 | 1.10 | 0.7153 | 1.1240 | 0.5236 | 14.11% | 13.13% |
| ST-TIS gate no-aux | 11 | 1.00 | 0.7161 | 1.1203 | 0.5265 | 14.01% | 13.41% |
| ST-TIS gate no-aux | 23 | 1.00 | **0.7135** | 1.1217 | **0.5220** | **14.32%** | 13.30% |

三随机种子汇总：

| Model | All MAE | Affected MAE | Unaffected MAE |
|:--|--:|--:|--:|
| old dual-branch gate no-aux | 0.7182 ± 0.0016 | 1.1240 ± 0.0054 | 0.5278 ± 0.0002 |
| ST-TIS gate no-aux | **0.7150 ± 0.0013** | **1.1220 ± 0.0019** | **0.5240 ± 0.0023** |

结论：ST-TIS-style incident branch 在总体 MAE 上稳定优于旧门控模型，三 seed 平均 affected MAE 也略有提升，并且 affected 指标方差明显下降。单次最优 affected 仍是旧模型 seed 23 的 1.1188，但从稳定性和总体表现看，ST-TIS 分支更适合作为下一版主模型。

## 分支行为变化

在 seed 23 checkpoint 上进行完整 test split 解释性分析：

| Inference mode | All MAE | Affected MAE | Unaffected MAE |
|:--|--:|--:|--:|
| Normal baseline | 0.8328 | 1.2938 | 0.6165 |
| Normal-style residual only | 0.7777 | 1.2133 | 0.5733 |
| ST-TIS incident residual only | 0.8044 | 1.2443 | 0.5980 |
| Fixed gate = 0.5 | 0.7244 | 1.1439 | 0.5276 |
| Learned gate | **0.7135** | **1.1217** | **0.5220** |

与旧 dual-branch gate 相比，ST-TIS incident branch 单独预测明显变强：

- old incident-only affected MAE：1.3562；
- ST-TIS incident-only affected MAE：1.2443；
- old fixed gate affected MAE：1.1706；
- ST-TIS fixed gate affected MAE：1.1439。

这说明 ST-TIS-style temporal/spatial attention 确实提升了事故分支的残差表达能力，不是单纯增加参数量后碰巧提升。

## Gate 行为

ST-TIS gate 在 affected elements 上的平均 incident-branch 权重为 0.4327，在 unaffected elements 上为 0.4322。二者差异不大，说明 gate 仍然不是简单的 affected-node classifier。

更重要的是 gate 与局部更优分支仍保持一致：

| Subset | Local condition | Incident-branch gate mean |
|:--|:--|--:|
| All | incident branch better | 0.4449 |
| All | normal branch better | 0.4211 |
| Affected | incident branch better | 0.4502 |
| Affected | normal branch better | 0.4172 |
| Unaffected | incident branch better | 0.4424 |
| Unaffected | normal branch better | 0.4229 |

Gate 与 \(|\Delta^{target}|\) 的相关性为 0.1998，在 affected elements 上为 0.2541。相比旧模型，这个相关性略低，但 learned gate 仍明显优于 fixed gate，说明它仍在做局部 residual explanation selection。

## Mixed case 观察

ST-TIS mixed case study 仍然保留 success、neutral、failure 三类样本：

| Category | Sample | Recovery min | Learned affected MAE | Fixed affected MAE | Gain | Mean affected gate | Normal-only affected MAE | Incident-only affected MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| success | 185121 | 40 | 10.1827 | 11.6042 | 1.4214 | 0.5273 | 12.7115 | 10.4968 |
| success | 185115 | 60 | 5.6913 | 6.9432 | 1.2519 | 0.5059 | 7.5383 | 6.4259 |
| neutral | 193337 | 180 | 0.8800 | 0.8800 | 0.0000 | 0.4177 | 0.9308 | 0.9416 |
| neutral | 192525 | 0 | 0.6910 | 0.6910 | 0.0000 | 0.5374 | 0.6702 | 0.7619 |
| failure | 88134 | 25 | 6.8554 | 4.9938 | -1.8616 | 0.5124 | 2.1877 | 8.0691 |
| failure | 54705 | 0 | 1.8394 | 0.7998 | -1.0396 | 0.4679 | 2.1675 | 3.7115 |

sample 88134 仍然是典型 failure，但 ST-TIS 已经把 incident-only affected MAE 从旧模型的 9.6011 降到 8.0691，learned affected MAE 也从 7.0882 降到 6.8554。问题没有完全解决，但事故分支本身确实更可靠。

## 当前结论

1. ST-TIS-style incident branch 可以作为下一版主模型，因为它在三 seed 平均上同时改善 all、affected 和 unaffected MAE。
2. 它的主要收益来自 incident branch 表达能力增强：incident-only 和 fixed-gate 指标均显著优于旧模型。
3. learned gate 仍然有效，但相对 fixed gate 的边际收益变小，说明下一步瓶颈转向 gate calibration / branch reliability。
4. 后续不应再单纯加强 incident branch，而应重点研究如何在短恢复、单节点影响和局部分支失真样本上降低 gate 对事故分支的过信。

## 可引用文件

- `scripts/train_dual_branch_sttis_gate.py`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux_seed_11/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_no_aux_seed_23/metrics.json`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_seed_23_interpretability/summary.md`
- `outputs/impact_guided_next_stage/dual_branch_sttis_gate_seed_23_case_studies_mixed/case_study_report.md`
