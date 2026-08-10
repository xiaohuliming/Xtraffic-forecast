# protocol v3 seed 42 收敛审计训练前记录

日期：2026-08-11，Asia/Singapore

## 目的与边界

现有无泄漏 42 行表已经完成验证，必须保持只读。它统一使用 seed 42 和最多 30 epoch，适合作为固定训练预算下的描述性比较，但不能证明所有方法都已经收敛。

本轮新增独立矩阵 `protocol_v3_event_convergence_seed42_39_v1`。它只回答一个问题：在相同 v3 无泄漏数据、切分、观测 mask 和评测指标下，把可训练方法的预算提高到 100 epoch 后，现有排序与领先幅度是否稳定。

该矩阵不是最终多种子论文表。当前不覆盖旧结果，也不把本轮结果写入既有 42 行目录。

## 固定范围

- 协议：`protocol_v3_noleak_dedup`
- cache schema：3
- 缓存 bundle：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`
- 地区：Alameda、ContraCosta、Orange
- 方法：13 个可训练方法，不含无参数 HL
- 种子：42
- 任务数：39
- 最大训练轮次：100
- checkpoint：全部有效验证观测点上的全局 masked MAE
- 测试指标：H3、H6、H12 与 12 步 Average 的 MAE、RMSE、MAPE
- 分析标签：训练与验证不可见，只允许在测试预测完成后事后分组

canonical manifest 为 `config/protocol_v3/event_convergence_seed42_v1.json`。

## 训练控制

十个 baseline 使用 patience 30。DCRNN 恢复 100 epoch 配方的 MultiStepLR 里程碑 `[10, 50, 90]`，不再使用 30 epoch 压缩版 `[10, 20, 27]`。D²STGNN 恢复里程碑 `[1, 38, 46, 54, 62, 70, 80]`。ASTGCN 在本轮使用源配方的 100 epoch CosineAnnealingLR。

BiST 与 IGSTGNN 使用 patience 20。NDR-Net 不启用 early stopping，但每轮仍计算相同的全局验证 masked MAE，并在最终测试前恢复最佳验证 checkpoint。

这仍是 v3 标准化适配器下的收敛审计，不应写成所有方法训练引擎的逐行原生复刻。特别是 DGCRN 与 D²STGNN 的上游 curriculum 差异仍需在论文方法说明中披露。

## 新增可审计产物

每个任务除原有 summary、checkpoint、预测与日志外，必须生成 `training_diagnostics.json`。该文件记录：

1. 实际运行轮数与停止原因。
2. 最佳轮次与 stale 计数。
3. 每轮训练目标和全局验证 MAE。
4. 每轮实际使用的学习率和下一轮学习率。
5. 优化器、scheduler、patience 与 checkpoint 指标身份。

runner 与 source-host postflight 均把该文件列为必需 artifact。

## 多卡与回传门禁

`protocol_v3_event_convergence_gpu_probe.py` 会在正式训练前对 39 项逐项执行 loader 加完整优化步探针。探针结果必须完整绑定 task ID、100 epoch 预算和峰值显存，随后生成 task 级 calibration。

`protocol_v3_event_convergence_multi_gpu.py` 使用 task 级工时执行最长任务优先装箱。已知显存兜底为 Orange STTN 与 D²STGNN 使用 microbatch 16，Orange IGSTGNN 与 NDR-Net 使用 microbatch 8。不同微批组不能共享同一 shard。新机器探针可以收紧设置，但必须重新生成并人工复核 plan digest。

训练完成后仍需依次通过 aggregate queue、runner 零退出、source-host postflight、逐文件 SHA-256、隐藏 incoming 精确树校验和原子发布。任一失败都不得把结果写成已验证矩阵。

## 当前状态

截至本文记录时间，尚未启动任何新的 GPU 训练，也没有新的性能数字。代码、39 项 manifest、混合 runner、逐轮诊断、GPU 探针、task calibration、多卡控制器和共享 postflight 已完成。本地全仓回归为 633 passed 和 2 个既有 warning。

下一步是在用户提供的新 RTX 5090 主机上部署 clean checkout，严格复验缓存与 support，运行 39 项探针和 plan-only。人工确认显存、预计墙钟、分片均衡与 plan digest 后，才启动正式 seed 42 收敛矩阵。
