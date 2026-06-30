# Confidence-aware gate 探索报告

## 目的

前一轮 mixed case study 发现，原始 dual-branch gate 在少数失败样本上会过度依赖局部较差的 incident branch。例如 sample 88134 中，incident-only affected MAE 明显高于 normal-only affected MAE，但 learned gate 仍给 incident branch 较高权重。

本轮实验尝试加入 branch confidence / branch reliability 机制，验证它是否能降低这类失败样本，同时保持总体预测精度。

## 新增模型

新增脚本：

- `scripts/train_dual_branch_confidence_gate.py`

模型在原 dual-branch gate 基础上增加两个 branch-error heads：

- `normal_error_head`：预测 normal-style residual branch 的局部误差；
- `incident_error_head`：预测 incident residual branch 的局部误差；
- gate logits 由 `raw_gate_logits + confidence_scale * (normal_error_score - incident_error_score)` 调整。

训练时额外尝试三类约束：

- `branch_loss`：让 normal branch 和 incident branch 单独也尽量拟合 residual target；
- `confidence_loss`：让 error heads 预测各自分支的 detached 局部误差；
- `gate_alignment_loss`：用两个分支的 detached 局部误差构造 soft gate target。

这些额外损失只使用 residual target，不使用 affected node label，因此仍属于 no-aux 设置。

## 主结果对比

| Model | Beta | All MAE | Affected MAE | Unaffected MAE | All gain | Affected gain |
|:--|--:|--:|--:|--:|--:|--:|
| dual_branch_gate_no_aux | 1.05 | **0.7181** | **1.1234** | 0.5279 | **13.78%** | **13.17%** |
| confidence_gate_no_aux | 1.05 | 0.7186 | 1.1254 | 0.5278 | 13.71% | 13.02% |
| branchloss_gate_no_aux | 1.05 | 0.7186 | 1.1263 | **0.5273** | 13.71% | 12.95% |

结论：当前 confidence-aware gate 和 branch-loss-only 变体都没有超过原始 dual-branch gate no-aux。因此主模型不应替换。

## 分支行为变化

Confidence-aware 训练确实改变了两个分支本身的质量：

| Model | Normal-only affected MAE | Incident-only affected MAE | Fixed gate affected MAE | Learned gate affected MAE |
|:--|--:|--:|--:|--:|
| original dual-branch gate | 1.2478 | 1.3562 | 1.1706 | **1.1234** |
| confidence-aware gate | 1.2476 | **1.1723** | **1.1587** | 1.1254 |

这说明 branch confidence / branch-error supervision 让 incident branch 单独预测明显变强，fixed fusion 也更好；但最终 learned gate 没有同步变强。换句话说，问题不是 incident branch 完全没有改善，而是新增的 gate 修正没有学到比原始 gate 更好的融合策略。

## Confidence scale sweep

对训练好的 confidence-aware checkpoint 做后验 `confidence_scale` sweep：

| Confidence scale | Best beta | Test all MAE | Test affected MAE | Test unaffected MAE |
|--:|--:|--:|--:|--:|
| 0.00 | 1.10 | 0.7201 | 1.1281 | 0.5286 |
| 0.10 | 1.10 | 0.7198 | 1.1276 | 0.5285 |
| 0.25 | 1.10 | 0.7195 | 1.1268 | 0.5284 |
| 0.50 | 1.10 | 0.7190 | 1.1259 | 0.5282 |
| 0.75 | 1.05 | 0.7189 | 1.1261 | 0.5279 |
| 1.00 | 1.05 | 0.7186 | 1.1254 | 0.5278 |
| 1.25 | 1.05 | 0.7185 | 1.1250 | 0.5277 |
| 1.50 | 1.05 | 0.7184 | 1.1247 | 0.5277 |
| 2.00 | 1.05 | **0.7183** | **1.1244** | 0.5278 |
| 3.00 | 1.00 | 0.7190 | 1.1257 | 0.5282 |
| 4.00 | 1.00 | 0.7195 | 1.1263 | 0.5287 |
| 6.00 | 1.00 | 0.7211 | 1.1284 | 0.5301 |
| 8.00 | 0.95 | 0.7229 | 1.1313 | 0.5313 |

最佳后验 scale 是 2.0，但 test affected MAE 仍为 1.1244，略差于原始 dual-branch gate 的 1.1234。scale 过大后模型明显退化，说明 confidence 修正方向有一定作用，但不能简单越强越好。

对应图：

- `outputs/impact_guided_next_stage/dual_branch_confidence_gate_no_aux/confidence_scale_sweep.png`

## Mixed case 观察

confidence-aware gate 的失败样本仍包括原来的 sample 88134 和 sample 60576：

| Category | Sample | Learned affected MAE | Fixed affected MAE | Gain over fixed | Mean affected gate | Normal-only affected MAE | Incident-only affected MAE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| failure | 88134 | 6.3621 | 4.1181 | -2.2440 | 0.5334 | 1.9345 | 7.6984 |
| failure | 60576 | 2.1180 | 1.0332 | -1.0848 | 0.5743 | 2.9891 | 2.4057 |

相较原始模型，sample 88134 的 incident-only affected MAE 从 9.6011 降到 7.6984，说明事故分支有所改善；但 learned gate 仍然没有足够抑制 incident branch，因此这个失败没有被真正解决。

## 当前结论

1. 当前主模型仍然应保持为 `dual_branch_gate_no_aux`。
2. 简单加入 branch confidence heads 不能直接提升最终预测。
3. 分支监督会让 incident branch 单独变强，但 learned gate 的融合策略可能被破坏或变得过度平滑。
4. 后续若继续做 confidence-aware gate，更合理的方向不是直接用 error heads 改 gate，而是：
   - 对 gate 使用更温和的 regularization；
   - 在 failure-like cases 上加入 hard-example reweighting；
   - 用 uncertainty 或 ensemble disagreement 判断 branch reliability；
   - 将 confidence 作为解释/诊断信号，而不是直接作为 gate logits 的强修正项。

## 论文写法建议

本轮结果暂时不适合作为主方法贡献。可以在内部实验或讨论部分简短提到：我们尝试了 confidence-aware gate，但它没有超过原始 residual gate，说明 gate 的局部选择能力不能通过简单 branch-error prediction 直接替代。正式论文主线仍应聚焦 normal-impact decomposition、dual residual branches 和 learned node-horizon gate。
