# v3 event baseline 剩余分片的多卡主机布防

日期：2026-08-07

状态：控制层已布防，未部署，未启动训练。没有任何新的正式结果，也没有改动任何已验收结果。

2026-08-08 独立复核后修复了 4 个 P1 缺陷，本文的耗时表和卡数分析已改用中位数重算。修复记录见文末。

2026-08-08 晚，已租到 4 卡 RTX 5090 32GB 并完成实测探针。选型与排产结论已由实测取代投影，范围也从 22 项扩大为 33 项，记录见文末「5090 实测与最终计划」。

## 背景

v3 event baseline 主矩阵为 11 个方法乘 3 个地区，共 33 项，canonical manifest SHA-256 为 `f9cdc1afa5f9f126a6a8094d4d4fc38bc565cf3de277dc4a7742f85a0d2f92ef`。截至本日 runner 层为 14/33。

八卡 4090 D 主机 `autodl-container-5ee0448de2-ca306372` 已在无卡模式恢复 SSH，数据盘完好。只读盘点确认：

| 对象 | 状态 |
|---|---|
| `region_data_v3_noleak_dedup` | 1.8 GB，完好 |
| `v3_event_orange_4090d8` | 1.7 GB，4 项正式结果与 6 项 staging 完好 |
| 五个阶段的 repo 与 bundle | 完好 |
| GPU | 无卡模式下不可见 |

## 已验收与未验收清单

Orange 已通过 runner 验收 4 项，分别位于三个结果根：

| 任务 | 结果根 |
|---|---|
| v3e023 HL | `results/gpu7` |
| v3e024 LSTM | `results/gpu7` |
| v3e028 GraphWaveNet | `gwn_recovery_results/gpu6` |
| v3e030 STTN | `oom_recovery_results/gpu4` |

未验收 staging 的 epoch 进度如下。v3e027 Orange STGCN 从未启动。

| 任务 | 已完成 epoch |
|---|---:|
| v3e025 DCRNN | 19 |
| v3e029 ASTGCN | 12 |
| v3e031 DSTAGNN | 11 |
| v3e026 AGCRN | 4 |
| v3e033 D2STGNN | 4 |
| v3e032 DGCRN | 2 |

## 这些 staging 不可续跑

代码复核确认 `scripts/remote/linux/protocol_v3_event_orange_8gpu.py` 与 `scripts/train_baseline_zoo.py` 都没有 checkpoint 续跑路径。`ckpt_best.pt` 只在训练结束后被读回做测试推理，不保存 optimizer state 或 epoch 游标。v3 runner 契约本身也拒绝把不完整 attempt 当成可恢复结果。

因此上述六项必须从 epoch 0 重跑。这一结论直接决定了硬件选型：既然沉没进度收不回来，就没有理由为了迁就 24 GB 主机而放弃换卡。

## 落盘实测耗时

以下每 epoch 秒数来自八卡主机上 Orange 的 train.log，取该任务全部已落盘 epoch 的中位数，八个分片并发，含调度竞争。这是本次容量规划的唯一实测依据。首版曾错误采用各日志最后一个 epoch，已改正。DGCRN 与 D2STGNN 的中位数分别只基于 2 个和 4 个 epoch，只能用于粗排产。

| 方法 | 每 epoch 中位数秒 | 样本 epoch 数 | 30 epoch 折合小时 | 备注 |
|---|---:|---:|---:|---|
| GraphWaveNet | 365.7 | 30 | 3.0 | 已完成 |
| LSTM | 304.3 | 30 | 2.5 | 已完成 |
| STTN | 1258.0 | 26 | 10.5 | microbatch 16 |
| DCRNN | 2214.6 | 19 | 18.5 | |
| ASTGCN | 2893.0 | 12 | 24.1 | |
| DSTAGNN | 3087.6 | 11 | 25.7 | |
| AGCRN | 5158.8 | 4 | 43.0 | 样本少 |
| D2STGNN | 9252.0 | 4 | 77.1 | microbatch 8，样本少 |
| DGCRN | 19943.2 | 2 | 166.2 | microbatch 8，样本很少 |

Orange STGCN 与 HL 没有实测值，标定文件已显式标记 `measured: false`。

## 卡数分析

地区工作量按节点数乘训练样本数归一到 Orange：Alameda 0.3974，Contra Costa 0.1941，Orange 1.0。Contra Costa 11 项加 Orange 11 项共 22 项。

按 4090 D 中位数标定，总量为 449 卡时，关键路径是 Orange DGCRN 单项 166.2 小时，占全部剩余工作量的 37%。

| 卡数 | 墙钟 | 总卡时 | 空转 |
|---:|---:|---:|---:|
| 2 | 224.5 h | 449 | 0.1% |
| 3 | 166.2 h | 449 | 10.0% |
| 4 | 166.2 h | 449 | 32.5% |
| 6 | 166.2 h | 449 | 55.0% |

按 32 GB 投影标定，即假设 DGCRN 与 D2STGNN 回到原生 batch 32 并叠加 1.5 倍原始加速：

| 卡数 | 墙钟 | 总卡时 | 空转 |
|---:|---:|---:|---:|
| 2 | 74.9 h | 150 | 0.1% |
| 3 | 50.0 h | 150 | 0.3% |
| 4 | 37.8 h | 150 | 1.0% |
| 6 | 28.7 h | 150 | 13.0% |

两份标定给出不同的最优卡数，因此正式排产前必须先跑探针。投影标定状态为 `projection_not_measured`，不得当成实测证据引用。

## 分片边界

Orange 的 4 项已验收结果跑在 4090 D 上，从未执行 source-host postflight，也从未上过结果服务器。若新的 7 项跑在另一台主机上，Orange 这个直接比较组就会跨主机，违反同机纪律。

因此本次计划为 Contra Costa 全部 11 项加 Orange 全部 11 项，共 22 项，全部落在同一台新主机。Alameda 继续留在 5080。最终布局标记为主机分层矩阵，不得写成同质矩阵。旧 4090 D 上的 4 项与 6 项 staging 保持只读，不进入本次 composite。

## 控制层构件

| 文件 | SHA-256 |
|---|---|
| `scripts/remote/linux/protocol_v3_event_multi_gpu.py` | `bdb730647d4fa7a2c65959fc3cf7bdd2d7f1f7eb9cb381437698b06c91901c81` |
| `scripts/remote/linux/protocol_v3_event_gpu_probe.py` | `1c83dce3098392a7cef28c0d56e8963179443b2561515d3dd46d46fe21c87621` |
| `scripts/remote/linux/protocol_v3_event_nas_finalize.sh` | `9effc0ae8bb1c6c4ecabb9482507fc6487c4156e690e1b86272b22532a2730dd` |
| `scripts/remote/linux/protocol_v3_event_deploy_host.sh` | `c6eeffc3f324248d0cda536d1eca5088c974bf2e1c1c2d0c1d13eac92e792938` |
| `scripts/remote/linux/protocol_v3_event_launch_host.sh` | `ff73b46e501008d962278673fab6ad01b9ec4372417b7d3776c7d460f4d36259` |
| `config/protocol_v3/event_baseline_gpu_calibration_v1.json` | `cc7937d3effde7b1774f5045d37a4b7a85fe1c89d626ed651d1c63b2789f8ff1` |
| `config/protocol_v3/event_baseline_gpu_calibration_5090_projected_v1.json` | `afbc44cc08df7f44e7fc83be42506ddcfc007368c4edc9c5fe8d06b2300a3123` |
| `scripts/remote/windows/protocol_v3_event_host_postflight.py` | `1c41a3f20e33f7bffb83690fd3506b4b89f88c3f3a7c0bcca9eeb9fc9f92be25` |

## 设计要点

控制器按标定做最长任务优先装箱，装箱结果对任务输入顺序不敏感，并且不劣于轮转分配。它在结果根写一份严格派生的聚合 `queue_status.json`，字段与单机 runner 相同。聚合状态只有在全部子 runner 退出码为 0、全部分片 state 为 completed 且完成计数等于任务总数时才写 completed，任何一个分片失败即整体 failed。

plan.json 带一个 `plan_digest`，只覆盖决策相关字段，不含时间戳。人工复核 plan 之后把该摘要传给启动脚本，控制器在正式启动时会重算并拒绝任何与复核不一致的计划。

微批覆盖按 task_id 索引而不是模型名。同一个模型在两个地区都存在，按模型名索引会让同卡上的其他任务静默继承覆盖。被覆盖的任务会被强制独占一张卡。

finalizer 的远程命令统一 base64 编码后传输，从结构上排除了 2026-07-27 那类嵌套引号被剥离的故障。成功路径统一收敛到一个出口，因此远程树已验证时也会正确执行可选关机，不会让租用主机空转计费。

## 2026-08-08 独立复核与 P1 修复

一次独立复核在部署前发现 4 个 P1 缺陷，全部已复现、修复并写入回归。

### P1-1 交错分片必然无法通过 postflight

原 postflight 要求所有 source 的 task_id 拼接后严格等于 canonical 顺序。这个约束来自 5080 那种按连续区段切分的场景。多卡 LPT 装箱天然是交错的，第一个 source 的首项是 task032 而 canonical 首项是 task012，因此 22 项即使全部训练成功也会在回传时被拒。

修复方式是给计划增加显式的 `source_layout` 字段。默认值 `canonical_contiguous` 与原行为逐字节一致，5080 的现役计划不受影响。多卡主机使用 `interleaved`，此时要求每个 source 内部仍按 canonical 顺序、source 按 source_id 排序、并集等于 canonical 选择集。约束强度没有降低，只是把顺序要求放到正确的层级。

### P1-2 微批任务必然无法通过 postflight

训练时微批大小进入 shard manifest 并参与 `config_hash`，而 postflight 用不含微批字段的 canonical manifest 重算。同一个 Orange DGCRN 两者分别为 `cab798170f649d63...` 与 `1b579af3e802974e...`，不相等。一旦探针 OOM 并启用微批，训练能完成但回传必被拒。

修复方式是给计划增加 `task_overrides`，按 task_id 声明微批大小，postflight 在重算该任务哈希时应用同一覆盖。声明错误或漏声明都会导致哈希不符而 fail closed，所以这个字段是自证的，不能用来放宽校验。

### P1-3 启动门可能被复核阶段的旧 plan 满足

原启动脚本只检查 plan.json 是否存在，而部署阶段的 `--plan-only` 已经生成过该文件。控制器即使立刻退出，脚本仍会报告启动成功并武装 finalizer。同时正式启动会重新生成 plan，人工复核过的版本没有被锁定。

修复方式是启动前把复核版改名为 `plan.reviewed.json` 并删除旧 pid，启动后要求 plan.json 重新出现、controller.pid 存活、controller_status.json 状态为 running，并要求重算的 `plan_digest` 与复核值一致。控制器自身也接受 `--expect-plan-digest` 并在不符时拒绝启动。

### P1-4 NAS 上的旧结果可能被误认为本轮结果

原 `verify_remote_tree` 只验证远程目录内部自洽，没有和本轮 postflight 绑定。同一 HOST_ID 下若存在旧但自洽的结果树，脚本会直接判定成功、跳过上传，甚至在开启关机选项时关机。

修复方式是把本轮的 `files_digest` 与 `plan_sha256` 传入每一次树验证，并在远程 `_TRANSFER_COMPLETE.json` 中精确匹配。任何未绑定本轮的树都不再被接受。

### 次要修复

探针合并标定时不再把继承自 4090 D 的条目标成本机实测。`measured` 只对本机真正探测过的模型为真，另加 `measured_on` 记录每个条目的来源，整体 `status` 在未全部探测时为 `partially_measured`。`--microbatch` 现在校验取值必须在 1 到正式 batch 32 之间，错误值在 plan-only 阶段即被拒绝，不会拖到训练启动才暴露。耗时表改用中位数。

## 回归

本地全仓为 561 passed 和 2 个既有 warning。新增覆盖包括 26 项控制器单元测试、9 项真实 postflight 端到端测试和 14 项 finalizer 门禁测试。

端到端测试会物化一台完整的四卡交错主机，包含 22 个任务目录、每任务 summary 与 prediction artifact、per-GPU 与聚合 queue status、runner 退出码，然后跑真实的 `build_postflight_report`。它同时覆盖含微批任务的情形，并断言漏声明、错声明、越界声明、source 内乱序、source 间乱序和默认布局下的交错计划都会 fail closed。

`protocol_v3_event_host_postflight.py` 已修改。5080 现役 finalizer 使用的是部署在该机上的独立副本，不受影响；仓库内的 5080 计划在默认布局下继续通过校验，并有专门回归守住这一点。

## 尚未完成

1. 未部署到任何主机，未跑探针，未产出实测标定。
2. 卡数尚未最终确定，等探针结果。
3. 若探针报 `PROBE_OOM`，需要为对应任务指定微批并让它独占一张卡，同时在 provenance 中保留该差异。
4. NAS 目标目录、专用密钥和 host ID 尚未在新主机上建立。
5. BiST、IGSTGNN 与 NDR-Net 的 v3 event adapter 仍未开始，33 项完成后仍不足以填满原 Table 4 的全部行。


## 5090 实测与最终计划

租用主机为 `autodl-container-c0e9kcmfc7-9d9148dc`，4 张 RTX 5090 32GB，驱动 580.105.08，torch 2.12.1+cu130，208 核，754 GB 内存，数据盘 50 GB。

正式缓存由 4090 D 主机云到云直传，16 个文件逐一核对 SHA-256 全部一致，并在 5090 上通过严格只读审计，bundle ID 与 digest 与正式清单相符。

### Orange 实测，batch 32

| 方法 | 微批 | 单卡独占 | 四卡并发 | 峰值显存 |
|---|---:|---:|---:|---:|
| LSTM | 32 | 20 s | 37 s | 3.76 GiB |
| STGCN | 32 | 118 s | 217 s | 1.10 GiB |
| AGCRN | 32 | 153 s | 282 s | 6.92 GiB |
| ASTGCN | 32 | 209 s | 385 s | 6.65 GiB |
| STTN | 16 | 270 s | 497 s | 15.02 GiB |
| DSTAGNN | 32 | 360 s | 887 s | 10.57 GiB |
| DCRNN | 32 | 579 s | 911 s | 8.37 GiB |
| D2STGNN | 16 | 729 s | 1225 s | 29.48 GiB |
| DGCRN | 32 | 811 s | 1350 s | 26.12 GiB |

四卡并发列中，DGCRN、D2STGNN、DCRNN 与 DSTAGNN 为直接实测，其余为单卡实测乘实测平均竞争系数 1.846。竞争系数逐项实测为 1.66、1.68、1.57 与 2.46。

### 显存判定

DGCRN 原生 batch 32 通过，峰值 26.12 GiB，验证了此前由 24 GB 上「失败时仅差 120 MiB」做出的推断。

D2STGNN 在 batch 32 仍缺 2.80 GiB，必须使用 microbatch 16，峰值 29.48 GiB。该峰值在独占与并发两次运行中字节相同，说明是确定性的，不依赖碎片。microbatch 8 的备选为 883 秒和 15.16 GiB，作为 OOM 时的回退。

STTN 在 `expandable_segments` 下能以 batch 32 跑通，但峰值 29.96 GiB 过于贴边。microbatch 16 反而更快，270 秒对 276 秒，且只用 15.02 GiB，因此 microbatch 16 是严格更优选择，不是妥协。

Alameda 与 Contra Costa 的 D2STGNN 与 STTN 均可原生 batch 32，峰值分别为 19.36、8.80、17.86 与 8.03 GiB。因此全矩阵只有两个 Orange 任务需要微批。

### 地区系数重标

原按节点数乘训练样本数推得 Alameda 0.3974、Contra Costa 0.1941。用 D2STGNN 与 STTN 的实测单卡比值重标后为 0.2737 与 0.1329，先前估计高估了两个小地区。新系数记录在标定文件的 `region_scale_measurements` 中。

### 最终计划

范围由 22 项扩大为 33 项，三个地区全部落在这一台主机。多花约 5 小时与 20 卡时，换来整张表同主机、同代码 SHA、同运行时、同卡型，主机分层注脚彻底消除。5080 上已验收的 10 项 Alameda 保持只读，不删除，但不再进入本矩阵。

| 卡数 | 墙钟 | 总卡时 | 空转 |
|---:|---:|---:|---:|
| 3 | 28.0 h | 70 | 16.3% |
| 4 | 18.7 h | 70 | 5.9% |
| 5 | 14.4 h | 70 | 2.1% |

采用 4 卡。GPU3 单独承载两个 microbatch 16 的 Orange 任务，其余三卡承载 31 项。

### 与 4090 D 的对照

按各自真实运行方式，4090 D 八卡并发完成 22 项需 449 卡时、墙钟 166 小时；5090 四卡并发完成 33 项需 70 卡时、墙钟 18.7 小时。差距来自三部分：卡本身更快、32 GB 让 DGCRN 与 STTN 摆脱微批、以及四路并发的竞争损耗小于八路。不应把全部差距归因于显卡本身。

### 控制层补充修复

微批任务原设计强制独占一张卡。33 项下这会把可用卡数从 4 压到 2，是过严约束。已改为按覆盖值分组：同一微批值的任务可以共卡，不同值必须分卡，无覆盖的任务不得与有覆盖的任务同卡。分组后按各组负载比例分配卡数，组内仍用最长任务优先装箱。


## 2026-08-08 正式启动

主机 `autodl-container-c0e9kcmfc7-9d9148dc`，4 张 RTX 5090 32GB。

| 项 | 值 |
|---|---|
| matrix_id | `protocol_v3_event_full33_5090x4_v1` |
| host_id | `event_rtx5090x4_full33` |
| 代码 SHA | `d23a991a570fd1c59e3d2ac8315a5e4063e8097f` |
| plan_digest | `23ed8507a1a301daef27bb613bc88d2070ec7052858de7d219e9c845c4d2cb82` |
| cache bundle | `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4` |
| bundle digest | `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5` |
| 分配器 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 微批覆盖 | `v3e033_event_d2stgnn_orange_seed42=16`，`v3e030_event_sttn_orange_seed42=16` |

启动前门禁：远端 checkout 与 `d23a991` 精确一致且 worktree clean，远端 229 项 tests/remote 通过，正式缓存在该主机通过严格只读审计并返回一致的 bundle digest，人工复核 plan 后由 plan_digest 锁定，控制器重算不符即拒绝启动。

08:56:23 UTC 启动，controller PID 13432，状态 running。09:00 复验四卡均在训练首批任务，GPU0 DGCRN 显存 27,871 MiB，GPU3 D2STGNN 微批 16 显存 31,083 MiB，与探针预测一致。聚合队列为 running、0/33、失败 0。

分配器设置属于运行契约的一部分，已写入 plan 并参与 plan_digest，因此不会出现「复核时是一种内存行为、实跑是另一种」的情况。

### 回传、composite 与自动关机已武装

Hostinger `31.97.147.71` 被选为本轮结果服务器。实测空间约 49.6 GB，可用内存约 3.75 GB。已增加 4 GB swapfile，xray 服务在修改前后均为 active。隔离 Python 环境使用 Python 3.12.3 和 NumPy 2.4.6，与 5090 端 NumPy 版本一致。

5090 端专用 finalizer PID 为 `17206`。目标路径为 `/var/lib/xtraffic/published/protocol_v3_noleak_dedup/event_baseline_phase1/hosts/event_rtx5090x4_full33`。它只在 33/33、runner 零退出、无残留 trainer 和 source-host postflight 全部通过后上传。目标端再执行逐文件 SHA-256、精确 allowlist、原子提升与只读 composite。最后生成 CSV、Markdown 和 LaTeX 后才关闭 5090。任一失败都保持主机开机。

composite 使用 systemd `MemoryMax=3221225472` 和 `MemorySwapMax=4294967296`。控制提交为 `314d45714b61465bce57e8a33c824ce6c47891fd`。finalizer、post-publish 包装、composite 和表格生成器的 SHA-256 分别为 `cfdf74306abb7caa8441ff36b487f94ebfa445991b7b1ceb34f98a3d2c9f70d0`、`28585a7d269218083e2ef83e440d5acf7cc7f0c03498c9d780ce3dd3400a4183`、`2da92291801e516f18ac335fd9a330e75561ce77b16e7897dbc47b99789e0d13` 和 `890006ec3f1132eeb1aa87cd42d4d668a0160b16c68de069d7304a5230a09993`。本地全仓回归为 571 passed 和 2 个既有 warning。

18:24 新加坡时间复验为 0/33 completed、0 failed。四张卡仍在执行首批 Orange DGCRN、DCRNN、DSTAGNN 与 D2STGNN。当时最新完整 epoch 数分别为 2、3、4 和 2。首轮并发 epoch 慢于探针标定，原 18.7 小时只保留为排产估计。

## 2026-08-08 18:55 无泄漏扩展矩阵实现复审

现有 33 项矩阵仍为 0/33 completed、0 failed。Orange DGCRN、DCRNN、DSTAGNN 与 D2STGNN 最新完整 epoch 数分别为 3、4、6 和 3。四张卡的已分配显存仍为 27871、9597、11551 与 31083 MiB，finalizer PID `17206` 持续每 60 秒记录等待。

用户要求的最终无泄漏对比表固定为 42 行。新增矩阵 `protocol_v3_event_extensions_5090x4_v1` 包含 BiST、IGSTGNN 和 NDR-Net 乘三地区共 9 项，canonical manifest 为 `config/protocol_v3/table4_event_extensions_v1.json`。该矩阵尚未部署或训练。

实现复审确认以下数据边界。

1. loader 只从 v3 schema 3 数值缓存按 split index 懒加载，不产生可携带隐藏标签的 object-pickle 样本。
2. train 和 validation batch 不含 `affected_mask` 或任何 analysis-only 字段。该 mask 仅在 test prediction 完成后读取。
3. NDR-Net 的 normalization 与 weekly climatology 仅使用 `0:train_end` 唯一 raw timestamps 和显式 `flow_mask`。
4. IGSTGNN support 必须同时通过逐字节 support manifest 和逐节点 identity 对齐。三地区已验证 `sensors.csv order == region_idx` 与 `station_id == node_order[region_idx]`。
5. validation MAE 使用全部有效观测点统一累计，partial batch 在 train、validation 与 test 全部保留。
6. BiST 与 IGSTGNN 前向不传入 future target。IGSTGNN 训练按上游 engine 的相同顺序更新 curriculum horizon，再只对当前 horizon 中的显式有效观测点反传。
7. Orange NDR-Net 的有效 microbatch 8 同时写入正式 trainer 命令、CUDA 探针与 invocation config hash，不存在隐式容量覆盖。

控制链已覆盖九项真实 CUDA 探针、实测标定、LPT 分片、plan digest 复核、任务级 config hash、source-host postflight、隐藏 incoming、逐文件 SHA-256、原子提升、发布后只读 composite 和成功后自动关机。扩展 composite 通过后，42 行合表器还会核对它与 33 项 index 的 protocol、cache schema、bundle ID 和 bundle digest。任一不一致都拒绝生成表格。

本地全仓回归为 589 passed 和 2 个既有 warning。所有 `tests/remote/*.sh` 失败关闭门禁通过。

实现以提交 `16e90baab74bb767a4f8d7a46f45eef6d87af44d` 推送到 `main`。Hostinger 旁路 checkout 通过 Python 编译门，之后原子切换 `/opt/xtraffic/v3_event_repo`。旧 `314d457` 目录保留为回滚。新 `/opt/xtraffic/bin/event-extension-post-publish` 与仓库脚本的 SHA-256 均为 `80f05af5b0bb87a221e703479fc034f3b717acf07fff5748b583a1a518388edc`。切换后 xray 仍为 active，swap 为 4 GB，根分区可用 `44062988` KiB。未修改 xray 配置，临时 bundle 已清理。
