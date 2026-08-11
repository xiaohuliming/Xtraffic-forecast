# protocol v3 seed 42 收敛审计正式启动记录

日期：2026-08-11，Asia/Singapore

## 范围

既有无泄漏 42 行表保持只读。本轮独立矩阵为 `protocol_v3_event_convergence_seed42_39_v1`，包含 13 个可训练方法、3 个地区与 seed 42，共 39 项。训练上限为 100 epoch。本轮用于审计收敛与 scheduler，不覆盖既有 30 epoch 表，也不是最终多种子论文表。

正式主机为 `autodl-container-wh1skgkhxg-578ae9cd`，含 4 张 RTX 5090，每张显存 32607 MiB。正式缓存、图、节点顺序和模型支持均位于 `/root/autodl-tmp` 的独立只读输入目录。

## 输入与门禁

- cache bundle ID：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`
- cache bundle digest：`854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`
- eager 与 lazy 三地区等价性审计 SHA-256：`083f9554fd305beed8ccea91385b59230479dd6edd965974959b23cdd9981f6e`
- 39 项 probe 记录 SHA-256：`4c68571c4ee26b58c56212ba0d59e6d301a7735510ed40bf2ee39e7286876c70`
- task calibration SHA-256：`a6d1c9015b29b70fce51e84e47f0f1fcc33e8a85db9b4e35eadf3841fa2d359a`
- support manifest SHA-256：`727adb9d06365cb38d9147c1df6fb6d5d9b61eb43ad0fdffb6dbacc5c0f87dac`

缓存只读审计通过。三区域 train shuffle、val 和 test 各 20 个 batch 的 eager 与 lazy 数据流逐字节一致。39 项真实 CUDA probe 全部通过。probe 使用控制提交 `9d3527218dff6524e67d5e2a0247151d9dcac299`。随后修复只增加扩展 trainer 的项目根导入引导，不改变数据、模型、损失、优化器、微批或指标，因此 calibration 文件原样绑定到新正式 plan。

## 启动级 incident 与修复

第一次正式启动使用代码 `9d3527218dff6524e67d5e2a0247151d9dcac299`。GPU2 的 Orange IGSTGNN 通过脚本路径启动时立即出现 `ModuleNotFoundError: No module named 'dist_net'`。probe 使用模块入口，所以没有覆盖这一脚本入口差异。

发现后立即停止全部四个 shard。没有任务完成，没有正式任务目录被接受。控制树与 staging 作为 incident 保存在：

`/root/autodl-tmp/v3_convergence/incidents/formal_bootstrap_import_20260811T013204Z`

incident 清单 SHA-256 为 `b6c1e8d31c0bb4e771bae6ea4e8e0457510098c9a726e33d29535af469347341`。

提交 `0338e5ee93d0cf8ef2e162186310c32130fa31df` 为扩展 trainer 增加和 baseline 一致的项目根导入引导。隔离脚本入口测试、远端专项 19 项、本地全仓 636 项和 ruff 全部通过。正式训练 checkout 固定在该提交并保持 clean。

## 正式四卡计划

正式 plan digest 为 `f84709ccf5debcdcd6a766497095d3ca0024bd72614beb6a2785f2ad6f11b9b0`。plan 文件 SHA-256 为 `f3562666927a45d8ee63cb38b45fb2e02892bc57aae9af77af0a875225f29dd6`。NAS plan SHA-256 为 `cd15e5644e95ede921936ab39916e4018ca8cab1e30899aeb9814475ecb33e08`。

| GPU | 任务数 | 探针预计工时 | 首项 |
| --- | ---: | ---: | --- |
| 0 | 17 | 114.132 小时 | Alameda AGCRN |
| 1 | 18 | 114.190 小时 | Alameda LSTM |
| 2 | 2 | 122.362 小时 | Orange IGSTGNN |
| 3 | 2 | 29.697 小时 | Orange STTN |

不同微批组保持隔离。Orange STTN 与 D²STGNN 使用 microbatch 16。Orange IGSTGNN 与 NDR-Net 使用 microbatch 8。有效 batch、全批 loss 分母与一次 optimizer step 契约不变。

2026-08-11 09:45 新加坡时间，正式状态为 `running`、0/39 completed、0 failed。controller PID 为 `41801`。四张卡分别执行 Alameda AGCRN、Alameda LSTM、Orange IGSTGNN 与 Orange STTN。AGCRN、LSTM 与 STTN 分别完成第 4、12 与 1 个 epoch。Orange IGSTGNN 已完成模型构建并进入训练，入口 incident 没有复现。

## 增量保全与最终发布

提交 `40d2cca34bfc2925f20a0afdff56fa09f8363dc6` 让增量灾备侧车支持 `event_convergence` plan。该控制提交不部署进训练 checkout。侧车作为独立文件运行，SHA-256 为 `e2688234a506554e7d1d4e903f07487f5fd4b09bfed25251c75f59018044787c`，PID 为 `43230`。它只复验 runner 已原子提升的 completed task，不读取 staging。每个任务通过后立即上传到 Hostinger 隐藏 `.incremental` 树并逐文件核对 SHA-256。

Hostinger `31.97.147.71` 在启动时约有 24 GB 可用空间，xray 为 active，4 GB swap 正常。GPU 主机使用独立 ED25519 回传身份，指纹为 `SHA256:6z4d2hn0LVkISTgB+C2ei8LherIjHUaJMecDNGOY/9M`。私钥只在 GPU 控制目录，不进入仓库。

最终 finalizer PID 为 `43321`，脚本 SHA-256 为 `d637ecbee4d75c0944322498b80601ea9829bd7bb0925bf3f8a552d42f89f618`。它等待 39/39、runner 退出码 0、无残留 trainer、source-host postflight、远端容量门、逐文件 SHA-256、精确文件集和原子提升。全部成功后自动关机。任一失败都保持主机开机。

运行中的 checkpoint 仍不含 optimizer、scheduler、下一 epoch 游标和随机数状态。主机中断后，已完成并验证的任务可以保留，正在运行的单项必须从 epoch 0 新建 attempt。

## 最新可验证状态

2026-08-11 20:04 新加坡时间，`connect.weste.seetacloud.com:46241` 能建立 TCP，但远端在 SSH 鉴权前关闭连接。当前证据不能证明训练仍在运行，也不能区分欠费、关机、平台回收或 SSH 入口异常。启动时记录的 controller、侧车和 finalizer PID 不再作为存活证据。

Hostinger 增量灾备树已严格保全第一个完成任务 `v3c001_event_lstm_alameda_seed42`。该任务状态为 `completed`，runner return code 为 0，运行 100 epoch，最佳验证 checkpoint 位于 epoch 99。完成状态更新时间为 2026-08-11 03:15:25 UTC，增量复验 marker 时间为 03:16:23 UTC。marker 状态为 `verified_completed_task_backup`，8 个文件的 files digest 为 `6bd8c7313b453745bd7ec0a40ad82df5fb130a17484241364b95e2ad0b17b4fa`。

因此当前矩阵状态更新为 `last_verified_1_of_39_remote_unreachable`。这不是训练完成状态，也不是性能结论。LSTM 的 best epoch 99 说明 30 epoch 对至少该任务会截断仍在改善的训练过程，但不能单凭这一项推断所有方法的排序变化。

恢复访问后必须先只读检查 queue、task state、trainer 进程、正式任务树、staging 和 GPU 状态。已完成并通过 Hostinger 增量复验的任务可以作为恢复候选。进行中的任务因缺少 optimizer、scheduler、下一 epoch 游标和随机数状态，不能从中断 epoch 严格续跑。若需要继续，必须从 epoch 0 新建 attempt，并保留旧 staging 的 incident 清单。
