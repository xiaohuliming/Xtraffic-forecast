# 实验审计与进度状态

## 当前协议状态，2026-07-21

2026-07-18 前的完整候选模型、dual-branch gate 和 no-aux 数字仍保留在本文件后半部分，用于追溯研究过程和诚实负结果。它们来自旧 XTraffic 数据管线，当前统一标记为 contaminated，不得直接进入论文，也不得被称为已经验证的当前结论。

无泄露 v2 曾完成三地区缓存验收并在 5080 启动 45 项最小判别队列。后续全面复核发现，同一预测时间可能因多个事故重复采样，主事故选择不规范，验证 MAE 没有按全部有效观测点统一聚合。审查还覆盖了活跃事故 split 隔离、完整 affected 并集、IGSTGNN 目标掩码、反向邻接归一化、blind 模式和远程结果身份。

队列停止时已有四个 v2 summary。它们只作为调试记录保留，状态均为 `preliminary contaminated`。

| 任务 | v2 masked MAE | 当前状态 |
|---|---:|---|
| Alameda s0a seed 42 | 11.431783 | preliminary contaminated |
| Alameda s0c seed 42 | 11.193208 | preliminary contaminated |
| ContraCosta s0a seed 42 | 12.036264 | preliminary contaminated |
| ContraCosta s0c seed 42 | 11.822266 | preliminary contaminated |

第 5 项 Orange s0a seed 42 没有形成可用 summary，队列已经停止。日志与不完整 checkpoint 只读保留。

当前协议已升级为 `protocol_v3_noleak_dedup`，缓存 schema 为 3。默认缓存目录是 `outputs/dist_net/region_data_v3_noleak_dedup`，默认结果根目录是 `outputs/protocol_v3_noleak_dedup`。正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。Mac、5080 与 5090 的完整测试结果均为 143 项通过，另有 1 条既有张量转标量警告。

组员 Fu 在 `origin/fu` 的 `625aacf` 提出了空 mask batch 可跳过的处理思路。v3 已通过共享 `MaskedMAEAccumulator` 与远程 runner 的全预测有限性检查吸收这一正确性要求，没有直接 cherry-pick 落后于当前协议的旧分支。

三地区正式 v3 缓存已经落盘到 `outputs/dist_net/region_data_v3_noleak_dedup`。缓存先在 staging 路径完成严格审计，发布到正式路径后再次执行同一审计，两次结果均为 verified。缓存传到 5080 和 5090 后分别完成一次只读严格审计，结果仍为 verified。cache bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。传到 5080 时使用的归档大小为 1,896,581,632 字节，SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。

正式事件样本与 train、val、test 切分如下：

| 区域 | 总样本 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

5080 已完成任务 001 到 004，5090 已完成任务 005 和 006。六项的 prediction artifact 与 summary 均通过 runner 验收。

| 主机 | 任务 | test all masked MAE | test affected masked MAE | test unaffected masked MAE | best val masked MAE |
|---|---|---:|---:|---:|---:|
| 5080 | 001 Alameda s0a seed 42 | 11.339531 | 17.139917 | 11.005998 | 11.682158 |
| 5080 | 002 Alameda s0c seed 42 | 11.025200 | 16.583555 | 10.705584 | 10.939075 |
| 5080 | 003 ContraCosta s0a seed 42 | 12.066377 | 18.344279 | 11.771324 | 11.759196 |
| 5080 | 004 ContraCosta s0c seed 42 | 11.718486 | 17.784800 | 11.433378 | 11.424961 |
| 5090 | 005 Orange s0a seed 42 | 12.025650 | 17.932862 | 11.720226 | 11.872927 |
| 5090 | 006 Orange s0c seed 42 | 11.740712 | 17.583309 | 11.438629 | 11.547319 |

六项均为 seed 42，只能标记为 individually validated v3 result。它们不能直接形成多种子结论、显著性结论或论文最终结论。旧结果和 v2 的四项 `preliminary contaminated` 状态不变。

5080 的任务 005 因迁移到显存更充足的 5090 而被主动终止。runner 记录 code 1，这是人为停止导致的运行状态，不是模型失败。

5090 使用 clean checkout 的精确训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，全套测试为 143 项通过和 1 条既有 warning。同一正式 bundle 的只读严格审计为 verified。正式结果根为 `/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090`。

旧 5090 队列完成任务 005 和 006 后在任务边界安全停止。边界切换的同一秒，旧 runner 曾为 task 007 写入 running state、零字节日志和未完成 staging，但没有 summary 或正式任务目录。这些残留不计为结果，并被 source-host postflight 与上传 allowlist 排除。

2026-07-20 22:23 新加坡时间起，正式执行改为双机分层矩阵。5080 固定执行 tasks 001 到 004、007 到 010、013 到 016、019 到 030，共 24 项。5090 固定执行 tasks 005 到 006、011 到 012、017 到 018、031 到 045，共 21 项。两分片互斥且并集为 canonical 45，所有直接比较组保持同机。5080 runner 重新验收并跳过前四项，当前为 4/24。task 007 的 `attempt_0001` 因两条遗留诊断 PowerShell 扫描耗尽主机内存而在 epoch 9 数据装载阶段失败，没有 summary 或正式任务目录。内存恢复后，runner 已用 `attempt_0002` 重启 task 007。该记录属于主机运维故障，不是模型、数据协议或训练数值失败。截至 2026-07-21 00:02 新加坡时间，`attempt_0002` 已完成 epoch 7，train 为 12.0424，val 为 12.6845。5090 runner 重新验收并跳过前两项，当前为 2/21。截至同一时间，task 011 已完成 epoch 8，train 为 12.3250，val 为 12.3937。

两台机器保留独立结果根、路径相关 `config_hash`、运行时 provenance 和 source-host postflight。两个分片分别通过严格 allowlist 与远端逐文件 SHA-256 复验后，只读 composite index 验证 canonical 45 完整性，不复制任务目录或改写 summary。结果只能称为 `host-stratified distributed 45`，不能称为 homogeneous 5090 matrix。

5090 分片自动收尾进程 PID 为 106466。21 项训练、source-host postflight、严格回传与服务器端哈希全部通过后，5090 才会关机。5080 收尾进程 PID 为 12524。它等待 5090 上传标记后回传 24 项，成功后保持实验室工作站开机。两个进程都已通过独立进程检查和持续 WAIT 日志复验。任一失败都会保持相应机器开机。结果服务器约有 6.3 GB 空闲，双分片粗估约 11.6 GB。服务器有 12.25 GB 未使用 Docker build cache，但尚未获授权清理，当前仍有容量阻塞。

方法实现层级固定为 `RADS ⊂ RADS-O ⊂ NDR-Net`。这一层级只说明组件包含关系，不恢复任何 contaminated 性能主张。

## 2026-07-18 前历史结论

以下内容记录当时的判断。模型不是停留在想法阶段，已经有可运行实现、训练权重、日志、指标和论文草稿。原 residual temporal decay 主模型目录为：

`outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay`

在当时那轮局部审计后，又补了 clean no-aux 复验和导师原始 dual-branch gate 思路的 full baseline。当时单次 seed 7 最好结果来自：

`outputs/impact_guided_next_stage/dual_branch_gate_full_no_aux`

四个关键版本在完整 test split 上的结果如下：

| 模型 | All candidates | Affected candidates |
|---|---:|---:|
| Residual temporal decay | 0.8328 -> 0.7239, +13.07% | 1.2938 -> 1.1308, +12.60% |
| Residual temporal decay no-aux | 0.8328 -> 0.7221, +13.30% | 1.2938 -> 1.1290, +12.74% |
| Dual-branch gate | 0.8328 -> 0.7203, +13.51% | 1.2938 -> 1.1283, +12.80% |
| Dual-branch gate no-aux | **0.8328 -> 0.7181, +13.78%** | **1.2938 -> 1.1234, +13.17%** |

原 residual temporal decay 的 seed 7/11/23 稳定性结果为：

| 指标 | mean ± std |
|---|---:|
| all candidates robust MAE | 0.7242 ± 0.0016 |
| affected candidates robust MAE | 1.1342 ± 0.0041 |

新补的 dual-branch gate no-aux seed 7/11/23 稳定性结果为：

| 指标 | mean ± std |
|---|---:|
| all candidates robust MAE | **0.7182 ± 0.0016** |
| affected candidates robust MAE | **1.1240 ± 0.0054** |

当时最严谨的说法是：dual-branch gated residual no-aux 已经稳定优于原 residual temporal decay，是下一版论文更合适的主模型。2026-07-18 后，这一性能判断降级为 contaminated 历史结论。

## 历史实验当时已完成的可信度检查

1. 表格数字可以从 `metrics.json` 自动追溯，`ablation_summary/learned_normal_ablation_table.csv` 中各行与对应 metrics 文件完全一致。
2. 主模型 cache 的 split 数量为：
   - train: 138528
   - val: 29210
   - test: 27499
3. split 是按 2023 年时间线划分：
   - train: 前 70%
   - val: 中间 15%
   - test: 最后 15%
4. statistical normal baseline 和 blend alphas 使用 `train_valid` 构造，70% 训练边界之后的时间步被 mask，没有直接使用 test 交通流构造预测 baseline。
5. `global_context` 输入没有包含 severity/recovery/spread 这类由未来真实交通计算出来的标签。
6. 原 residual temporal decay 训练日志显示 val loss 从 0.4760 降到 0.4397，best epoch 是第 5 轮。
7. no-aux 复验表明，去掉 event/node 辅助监督后模型不退化，反而略好：all candidates 0.7221，affected candidates 1.1290。
8. full dual-branch gate no-aux 在 seed 7 进一步提升到 all candidates 0.7181，affected candidates 1.1234。
9. dual-branch gate no-aux 的三 seed 均值为 all candidates 0.7182，affected candidates 1.1240，稳定优于原 residual temporal decay。

详细审计报告见：

`outputs/impact_guided_next_stage/experiment_audit/audit_report.md`

## 历史管线当时识别的局部风险

最需要注意的是两个点。

第一，auxiliary supervision 已经不再是主要风险。

当时主模型中：

- `event_aux_weight = 0.05`
- `node_aux_weight = 0.03`
- `heatmap_aux_weight = 0.0`

其中 `event_aux` 和 `node_affected` 是由事故后真实交通状态派生出来的 impact labels。它们没有作为推理输入，因此不是输入端偷看未来；但它们参与了训练辅助损失，因此严格论文实验中需要 clean no-aux 版本。

这一步已经完成：

`outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay_no_aux`

结果显示 no-aux 不仅没有退化，还略好：

| 模型 | All candidates | Affected candidates |
|---|---:|---:|
| residual decay with aux | 0.7239 | 1.1308 |
| residual decay no-aux | 0.7221 | 1.1290 |

所以当时的局部判断是：主效果不是靠未来派生辅助标签带来的。

第二，dual-branch gate no-aux 当时是更强主模型候选，并且已经补了多 seed 稳定性。

它的目录为：

`outputs/impact_guided_next_stage/dual_branch_gate_full_no_aux`

该模型复用了同一份 HDF5 cache、同一 split、同一 residual target 和同一 robust MAE 评估逻辑，区别在于结构上更接近导师原始想法：normal embedding branch 与 incident/STGNN embedding branch 经过 gate 融合后预测事故残差。

这说明我们不应该再把论文故事写成“残差分解优于 gate”。更合适的故事是：

> 保留导师最初的双分支 gate 思想，但将输出目标从完整交通状态改为相对于 learned normal 的事故残差。也就是说，gate 不是简单融合两个完整预测，而是在 residual-impact space 中融合 normal-style residual embedding 和 incident-style residual embedding。

多 seed 结果见：

`outputs/impact_guided_next_stage/dual_branch_gate_no_aux_seed_robustness/seed_robustness_summary.md`

之前建议的 no-aux 命令已经跑完，保留如下作为复现实验命令：

```bash
python3 scripts/train_full_candidate_stgnn_heatmap_model.py \
  --event-root outputs/impact_labels_aggregated/region_area_sensor_window \
  --raw-label-dir outputs/impact_labels \
  --output-dir outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_decay_no_aux \
  --cache-path outputs/impact_guided_next_stage/full_candidate_stgnn_learned_normal_dual/full_candidate_samples.h5 \
  --normal-model-dir outputs/impact_guided_next_stage/normal_stgnn_forecaster \
  --regions Alameda ContraCosta Orange \
  --input-steps 12 \
  --horizon-steps 12 \
  --max-candidate-nodes 36 \
  --candidate-pm-radius 5.0 \
  --anchor-pm-radius 2.0 \
  --epochs 5 \
  --batch-size 192 \
  --hidden-dim 96 \
  --graph-layers 2 \
  --graph-mode undirected \
  --graph-sigma 3.0 \
  --heatmap-aux-weight 0.0 \
  --event-aux-weight 0.0 \
  --node-aux-weight 0.0 \
  --use-normal-delta \
  --use-normal-delta-abs \
  --use-dual-hist-residual \
  --use-temporal-decay-head \
  --seed 7 \
  --max-train-samples 20000 \
  --device auto
```

## 历史阶段已补的导师原始思路 baseline

新增脚本：

`scripts/train_dual_branch_gate_baseline.py`

它复现了“同一输入生成 normal embedding 和 incident/STGNN embedding，再由 gate 融合”的原始思路。为了公平，它复用当时主模型的同一份 HDF5 cache、同一 split、同一 residual target 和同一 robust MAE 评估逻辑。

quick run 设置：

- train samples: 5000
- eval samples per split: 2048
- epochs: 3
- hidden_dim: 96
- graph_layers: 2

quick run 结果：

| 模型 | all candidates | affected candidates |
|---|---:|---:|
| dual-branch gate quick | 0.8422 → 0.7826, +7.08% | 1.3004 → 1.2199, +6.19% |
| current residual decomposition full | 0.8328 -> 0.7239, +13.07% | 1.2938 -> 1.1308, +12.60% |

full run 结果：

| 模型 | All candidates | Affected candidates |
|---|---:|---:|
| dual-branch gate with aux | 0.8328 -> 0.7203, +13.51% | 1.2938 -> 1.1283, +12.80% |
| dual-branch gate no-aux | **0.8328 -> 0.7181, +13.78%** | **1.2938 -> 1.1234, +13.17%** |

dual-branch gate no-aux 多 seed 结果：

| Seed | All candidates | Affected candidates |
|---:|---:|---:|
| 7 | 0.7181 | 1.1234 |
| 11 | 0.7198 | 1.1296 |
| 23 | 0.7166 | 1.1188 |
| mean ± std | **0.7182 ± 0.0016** | **1.1240 ± 0.0054** |

注意：dual-branch gate 这里仍然是在 residual target 上训练，不是完全回到“直接预测完整交通状态”的原始版本。因此它更准确地说是“dual-branch gated residual model”。这反而更适合论文：既接住了导师的双分支 gate 想法，又解决了两个同输入 embedding 语义不清的问题。

## 历史阶段原定下一步优先级

以下事项是数据有效性审计前的原定计划，当前已经被 v3 缓存重建与最小判别矩阵复跑取代。

1. 把论文主模型改成 dual-branch gated residual model。
2. 重新生成 paper artifacts 的主表，把 residual decay no-aux、dual-branch gate no-aux 和多 seed 结果加进去。
3. 修改方法章节：从单 residual branch + temporal decay，调整为 normal-style residual branch + incident graph branch + gate fusion。
4. 再补 1-2 个 case visualization，展示 normal forecast、incident residual、gate 权重和 final prediction 的差异。
