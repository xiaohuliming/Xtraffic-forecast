# protocol v3 event 扩展预载加速与重启审计

更新时间：2026-08-09 15:03，Asia/Singapore

## 目标

最终目标不变：在 `protocol_v3_noleak_dedup` 下完成 33 项 baseline 与 BiST、IGSTGNN、NDR-Net 三地区 9 项扩展，生成可用于论文的 42 行无数据泄露对比表。

本次只优化 9 项扩展的数据读取和排产。33 项 baseline 保持原训练契约和结果根。

## 根因证据

旧扩展计划只用 CUDA 优化步标定任务耗时，没有计入逐样本 HDF5 读取。runner 任务状态时间戳给出的已完成任务实测耗时如下：

| 任务 | 实测耗时 |
|---|---:|
| IGSTGNN Contra Costa | 5.96 h |
| BiST Alameda | 10.51 h |
| IGSTGNN Alameda | 12.26 h |

按旧分片实测速度外推，四张卡的总时长约为 31.2、43.7、23.4 和 44.7 小时。最慢分片比最快分片长近一倍，旧 LPT 标定不再成立。

Orange IGSTGNN 的 20 个真实训练 batch 数据构建对照如下：

| 模式 | 初始化 | 20 batch 构建 | 每 batch |
|---|---:|---:|---:|
| lazy HDF5 | 3.862 s | 28.5172 s | 1.42586 s |
| eager memory | 11.244 s | 0.270774 s | 0.0135387 s |

两条路径的 batch stream SHA-256 均为 `af7f99015966b930571b3ddd9f111e60f694a537112cdcfe56dd4f76095fddb1`。数据构建提速约 105 倍，主要瓶颈是逐样本 HDF5 I/O。

## 实现与本地门禁

提交 `c78b87a` 已推送到 `main`，包含以下变更：

1. canonical manifest 显式固定 `traffic_loading: eager`。
2. trainer 命令、invocation config、summary 和 config hash 记录同一加载模式。
3. 新增三区域 eager 与 lazy 等价性审计。训练集使用固定 seed shuffle，验证集和测试集顺序读取，逐 batch 做字节级 SHA-256 对照。
4. CUDA 探针分别记录 batch 构建时间和完整优化步时间，正式 task calibration 使用两者之和。
5. LPT 依据新 task calibration 重新装箱。
6. loading audit 文件及其 SHA-256 写入正式 plan、NAS 回传计划和 source-host postflight 契约。

本地最终全仓结果为 606 passed 和 2 个既有 warning。Python 编译、shell 语法与 `git diff --check` 全部通过。

## baseline 恢复与 incident

`36571` 于 2026-08-09 恢复为原 4 卡 RTX 5090 容器。baseline queue 为 6/33 completed、0 failed，但 controller、trainer 与 finalizer 均已停止。六个正式结果保留。Orange DGCRN、Orange AGCRN 与 Alameda D²STGNN 三个中断 attempt 不含 optimizer state 和 epoch cursor，必须从 epoch 0 重跑。

恢复前只读 incident 位于 `/root/autodl-tmp/v3_event_cc_orange/control/incidents/reboot_interruption_20260809T051637Z`。queue、formal tree 与 staging tree 清单 SHA-256 分别为 `10ed910c7e33847fa32931153b651c6c612cd99b7c0e06658dacc99a66663deb`、`acaddda23479ba74af3b38d26d8fc139d081c06c9de4f529207513c24e476070` 和 `b385567c5410a5c0a838bbe45ed9462255d7018ed888474fe57f439dbbfc9dfd`。

baseline 以原 plan digest `23ed8507a1a301daef27bb613bc88d2070ec7052858de7d219e9c845c4d2cb82` 恢复。三个中断任务使用 `attempt_0002`。13:49 的实时状态为 running、6/33、0 failed。

## 等价性、探针与正式计划

旧扩展四卡机已欠费无卡，本轮不再依赖该主机。新扩展全部位于同机隔离目录 `/root/autodl-tmp/v3_event_extensions_eager_v2`，不改写 baseline checkout 与结果根。

远端扩展门禁为 39 passed 和 5 个上游 warning。三地区 train、val 与 test 数据流的 eager 与 lazy 对照全部为 `exact_match`，loading audit SHA-256 为 `083f9554fd305beed8ccea91385b59230479dd6edd965974959b23cdd9981f6e`。九个 CUDA 探针全部为 `PROBE_OK`，probes 与 calibration SHA-256 分别为 `4004d54e282e3b13fb2d12821e1c7c84a8180c9ede9b435b18257441fb59b4a9` 和 `00f89733654d306f5accc370ab8063fafe5f106aa0d4df7e2bf5d4b36436b8fa`。

正式矩阵为 `protocol_v3_event_extensions_eager_5090x4_v2`，host ID 与 baseline 统一为 `event_rtx5090x4_full33`，训练代码 SHA 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。plan digest 为 `2b8bc204115a32bf40d244a94938eb5db27ea229bbed1427542b74c200b4ef3a`，plan 文件 SHA-256 为 `5a43e9224572d3413e8ca6753691a8923bc92c49617871f14eafc23d000302d4`。修正前的不同 host ID 会让合表器误标为跨主机。修正只暂停接力与收尾等待器，四个训练进程均未停止。旧计划 incident 清单 SHA-256 为 `f209d12b952cacb39a95b765b120677b06c44e1e86a9e876bdedfe57e4778cf2`。

| 正式 shard | 任务 | 预计时长 |
|---|---|---:|
| gpu00 | Orange NDR-Net | 41.86 h |
| gpu01 | Orange IGSTGNN | 34.52 h |
| gpu02 | Alameda BiST、Alameda NDR-Net、Contra Costa IGSTGNN | 39.49 h |
| gpu03 | Alameda IGSTGNN、Contra Costa BiST、Contra Costa NDR-Net、Orange BiST | 37.85 h |

2026-08-09 13:40 前，空闲物理 GPU3 已启动正式 gpu00 shard，当前执行 `v3x009_event_ndr_net_orange_seed42`。它使用 eager 加载、microbatch 8 和正式四卡 plan 的 shard manifest，不是临时训练或待迁移结果。14:08 已完成 epoch 2，两轮 validation MAE 分别为 `11.80291315` 与 `11.59224552`。首轮约 22 分钟，当前实测速度快于探针外推。

## 接力、发布与关机门

提交 `e21baf2` 将接力器升级为动态最长任务优先调度。任一物理 GPU 的 baseline 分片完成且无残留训练进程后，它会领取当前预计最长的未启动正式 shard，不再依赖固定的物理 GPU 到扩展 shard 映射。四个手动 shard 全部零退出后，它使用同一 plan digest 运行完整 controller 复验并生成 aggregate queue。relay 脚本 SHA-256 为 `a94db88f47aface7c6d9d85958eaf5b4084c84ece8d8bb9197c33b821627eab9`。动态升级只暂停旧 relay 与 finalize waiter，没有停止任何训练进程。升级 incident 清单 SHA-256 为 `1502f5fe42a3cd1e6d15c30b51df7f84ebb37ffe131a02b50ab876d239242495`。14:08 复验时 relay 与 finalize waiter 均存活，正式控制根没有异常 exitcode，只有 gpu00 映射到物理 GPU3。

baseline finalizer 已从成功后关机改为只发布。提交 `ad70d5a` 的 extension finalize waiter 只在 baseline finalizer 零退出和 extension relay 零退出后启动扩展 finalizer。waiter 脚本 SHA-256 为 `8405db004ff76e70309332141f1cb3d71abe4c2476bf0187969c0d8d13d2be2d`。

Hostinger 结果服务器的 xray 为 active，4 GB swap 空闲，根分区可用 `45030260736` bytes。验证 checkout 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。新 post-publish 主脚本与 eager v2 包装器 SHA-256 分别为 `31295bce299f8ff85db65fb033dcccd406c2a2c3cf8888706b6a3057991acb6b` 和 `341bdcd8866eed4d7032ddac8b506e4d4de889e72b95b2eb3ae51723f7700e56`。合表器将按同 host ID 与不同 code SHA 标记为 `target_disjoint_event_no_leak_code_stratified`。

关机只在 baseline 发布、extension 发布、两份 composite 和 42 行合表全部成功后发生。任一失败都保持开机。

## 增量灾备和断点恢复边界

代码复核确认两个正式 trainer 都只维护 `ckpt_best.pt`。baseline checkpoint 包含 epoch、model state、最佳验证值和配置。扩展 checkpoint 包含 model state、最佳验证值、最佳 epoch、方法、地区和 seed。两者都没有 optimizer、scheduler、下一 epoch 游标、early-stopping 计数和随机数状态。runner 的 staging recovery 只恢复已经零退出并完整产出结果、但在结果验收或原子提升阶段失败的任务，不能恢复训练中的 epoch。因此主机中断后，completed 任务可以跳过，当前 task 必须从 epoch 0 建立新 attempt。

为避免完成任务只留在 AutoDL 数据盘，新增独立脚本 `scripts/remote/linux/protocol_v3_event_incremental_backup.py`。脚本先按正式 plan 重建 invocation 和 config hash，再调用与 runner 相同的 `validate_result`。只有 task state 为 completed、returncode 为 0、final_dir 为 canonical task 目录、结果重新验收通过且目录内没有符号链接时才建立快照。上传目标是 Hostinger 各矩阵正式根下的隐藏 `.incremental` 灾备树。每个文件先按 SHA-256 检查复用，缺失文件写入唯一 `.part`，远端验证大小和哈希后原子改名，全部文件再次验证后才写 task marker。该流程不接触 `.staging`，也不会生成正式 transfer marker 或提升正式目录。

侧车使用 `nice 15` 和 `ionice idle`，不占 GPU。baseline 与扩展 PID 为 `16661` 与 `16662`。脚本 SHA-256 为 `e797f2a218f7f32d0b3a6c2dc6d454f92af96132f9e1d34c541863b906e0f9aa`。首个任务 `v3e010_event_dgcrn_alameda_seed42` 已完成 6 文件和 `168730616` bytes 回传，Hostinger 上的 task marker 已复验。其余 5 个既有 completed baseline 任务正在补传。扩展侧车等待首个 completed task。本地专项测试新增 5 项，全仓回归为 611 passed 和 2 个既有 warning。

## 关键路径优先接力

Orange NDR-Net 的 30 epoch 实际耗时约 6 小时 45 分钟，证明单步 CUDA 探针对 NDR-Net 的训练总时长存在明显高估。原接力顺序 `gpu02,gpu03,gpu01` 会让 Orange IGSTGNN 最后领取资源。提交 `0648b76` 将顺序改为显式参数，默认值为 `gpu01,gpu02,gpu03`，并要求它严格构成 1、2、3 的排列。错误顺序会在读取任何正式环境前 fail closed。

控制层切换发生在 Orange NDR-Net epoch 29 到 30 之间。只暂停 relay 和 finalize waiter，没有停止 baseline 或扩展 trainer。旧控制快照、正式 plan、gpu00 queue 和 PID 文件组成的 incident 清单 SHA-256 为 `c3261aa55b14d92bc0b70f16b44fde2433c1dc2635d2e56669c70c5a4c6b22e5`。新 relay 脚本 SHA-256 为 `97633145e9605678d92678ebbb03ea2b7d892072c0bb02fe2936c435df0252f3`。正式 plan 文件 SHA-256 保持为 `5a43e9224572d3413e8ca6753691a8923bc92c49617871f14eafc23d000302d4`。

Orange NDR-Net 于 2026-08-09 20:25:45 新加坡时间完成并通过 runner 验收。gpu00 为 completed、1/1，manual shard 退出码为 0，summary SHA-256 为 `598d5e2838433d1ad8b3ddb503b554ee0f5b10be81900fcedb1b1c0e621815f0`。20:25:54，relay 日志记录 `LAUNCH physical_gpu=3 formal_shard=gpu01`，随后正式任务 `v3x008_event_igstgnn_orange_seed42` 进入 running。CUDA 复验为 95% 利用率和 16505 MiB 显存。同期 baseline 为 10/33、0 failed，扩展为 1/9、0 failed。专项测试新增 2 项，全仓回归为 613 passed 和 2 个既有 warning。
