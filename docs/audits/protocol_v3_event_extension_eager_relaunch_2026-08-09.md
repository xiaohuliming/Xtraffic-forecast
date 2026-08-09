# protocol v3 event 扩展预载加速与重启审计

更新时间：2026-08-09 13:49，Asia/Singapore

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

2026-08-09 13:40 前，空闲物理 GPU3 已启动正式 gpu00 shard，当前执行 `v3x009_event_ndr_net_orange_seed42`。它使用 eager 加载、microbatch 8 和正式四卡 plan 的 shard manifest，不是临时训练或待迁移结果。

## 接力、发布与关机门

提交 `9048cd2` 新增可审计 GPU 接力器。baseline 物理 GPU2、GPU1 与 GPU0 各自分片完成且无残留训练进程后，它分别启动正式 gpu02、gpu03 与 gpu01 shard。四个手动 shard 全部零退出后，它使用同一 plan digest 运行完整 controller 复验并生成 aggregate queue。relay 脚本 SHA-256 为 `3448ecd78c7f97fc8d578508d87254f988b7929213e04b085f80bd338506972b`。

baseline finalizer 已从成功后关机改为只发布。提交 `ad70d5a` 的 extension finalize waiter 只在 baseline finalizer 零退出和 extension relay 零退出后启动扩展 finalizer。waiter 脚本 SHA-256 为 `8405db004ff76e70309332141f1cb3d71abe4c2476bf0187969c0d8d13d2be2d`。

Hostinger 结果服务器的 xray 为 active，4 GB swap 空闲，根分区可用 `45030260736` bytes。验证 checkout 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。新 post-publish 主脚本与 eager v2 包装器 SHA-256 分别为 `31295bce299f8ff85db65fb033dcccd406c2a2c3cf8888706b6a3057991acb6b` 和 `341bdcd8866eed4d7032ddac8b506e4d4de889e72b95b2eb3ae51723f7700e56`。合表器将按同 host ID 与不同 code SHA 标记为 `target_disjoint_event_no_leak_code_stratified`。

关机只在 baseline 发布、extension 发布、两份 composite 和 42 行合表全部成功后发生。任一失败都保持开机。
