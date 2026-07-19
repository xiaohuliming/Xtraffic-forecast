# XTraffic protocol v3 全面复核

复核日期：2026-07-19

复核状态：v3 代码审查与回归测试完成，正式缓存已在本地两次及 5080 一次通过严格审计，45 项最小判别矩阵正在运行

正式缓存构建代码 SHA：`29159f626494de52c69b36fe5bba9854f13cb56c`

最终训练代码 SHA：`3a4952837aaf6fbda1403d50b610b1bf40c6098a`

测试结果：Mac 与 5080 均为 143 passed、1 warning
协议版本：`protocol_v3_noleak_dedup`

缓存 schema：3

正式缓存目录：`outputs/dist_net/region_data_v3_noleak_dedup`

正式结果目录：`outputs/protocol_v3_noleak_dedup`

正式 cache bundle ID：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`

正式 cache bundle digest：`854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`

## 结论

v2 修复了 2026-07-18 审计发现的主要因果泄露，但后续全面复核确认，v2 仍存在重复预测时刻、active event 跨 split、验证集指标聚合错误及若干训练实现问题。最先完成的四项 v2 远程结果不能恢复为论文证据，状态统一为 `contaminated preliminary`。其余 v2 队列已经停止。

v3 已完成相应代码修复，并为缓存、训练和远程 runner 增加严格身份契约。正式缓存已经落盘，并在 staging 路径和正式路径各执行一次严格审计，两次均为 `verified`。本报告中的正式样本数、split 数、区域 digest 与 artifact SHA-256 均来自正式路径的 `cache_bundle_manifest.json`。

45 项 v3 最小判别矩阵已于 2026-07-19 20:18 新加坡时间在 5080 严格串行启动。启动时首项 `001_event_staeformer_s0a_alameda_seed42` 为 `running`，GPU 利用率为 93%，显存使用为 10522 MiB。首项 epoch 1 日志已落盘，train loss 为 18.7316，val loss 为 14.7119，用时 351 秒，并保存当前 best checkpoint。随后复查时 GPU 利用率为 96%。当前还没有任务完成并通过 runner 的 prediction artifact 与 summary 复验，因此没有新的 v3 模型结果。缓存通过审计和训练开始都不能证明模型性能，也不能恢复任何旧论文结论。baseline zoo 继续保持 `pending`。

## 复核范围

本次复核覆盖：

- 三地区事件样本的构建、去重、主事故选择和 active event 打包。
- 事件协议与 full-window 协议的切分、训练统计边界和加载时校验。
- masked MAE 的训练、验证、测试及最佳检查点选择。
- Graph WaveNet 与 Fourier Dual Net 的图支持构造。
- 区域分桶采样器的 epoch 行为。
- DIST-Net blind 配置与分析标签隔离。
- IGSTGNN 数据转换、缺失掩码、尾批、elapsed time 和指标累计。
- 缓存内容、代码版本、任务清单、输出目录和远程 runner 的身份绑定。

## v3 正式缓存验收

正式审计清单：

`outputs/dist_net/region_data_v3_noleak_dedup/cache_bundle_manifest.json`

manifest 状态为 `verified`，cache schema 为 3，协议为 `protocol_v3_noleak_dedup`。同一缓存先在 staging 路径完成严格审计，发布到正式路径后再次执行严格审计，两次均通过。

### 样本与 split

| 区域 | 样本数 | 唯一 forecast origin | train | val | test | 去除重复行 | mixed active 清理 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alameda | 45760 | 45760 | 32001 | 6954 | 6805 | 17021 | 8 |
| ContraCosta | 23337 | 23337 | 16414 | 3472 | 3451 | 3884 | 4 |
| Orange | 60015 | 60015 | 42372 | 9098 | 8545 | 45464 | 11 |

表中 `mixed active 清理` 对应 manifest 的 `mixed_active_removed`。三个区域的样本数均与唯一 forecast origin 数完全一致。

### 区域 digest

| 区域 | cache bundle digest |
|---|---|
| Alameda | `736bc6e8db2b93dded9e8c72aca00acfeb28dbb2bbaa3053c3f7d5ca2e83c682` |
| ContraCosta | `0eaaaa34100ef6cf4bea50b514156fd913b58db2197696b2ea21f0cb01ee8d45` |
| Orange | `06ea38c6d0d31890bc9a238a7d4d890f2ec20a5ba17a527b291526e6806deb95` |

### Artifact SHA-256

| 区域 | traffic | samples | relation | graph |
|---|---|---|---|---|
| Alameda | `3fcde0492a1714df3947f49e08e73098f4f1be8d8e1d346a9c892779d3552d45` | `0877a6fc086514321417861d5f0376638e31520e40d0631715467a45168cf798` | `d89540d30b6d7d4f76426433edf437402f01e69467dcfad714a6f718010a045f` | `3fd09d48a747517b0ed659f6b72025eb7829bb4186e348180f96d7d19379f4ba` |
| ContraCosta | `c6b76456c8b2bb759bdad58380ccbd5b108fc21b0e6502a6811255a888dc04b3` | `d456779a9f730cc58afab1616d63ca40815431af0a5ac0c77a016eac20e3576f` | `62afe8b8b70e96f0a427ce8bae8965668ae0fd34a3bc463c6383c15fa2bacf86` | `b00bdbbb7898f674f29ba8752dc53ee8c4abed9e787e1b92dfca2a50798555e9` |
| Orange | `5bfde78529aef84fac8bbc3de13b7bb76416a27725f3be9209a3aa7c041c439b` | `39cd470ab815d7d455ef0b401d68f5746dce402b46b02ae1c42a986e73da7f24` | `ba47f7b282741d0861f97796b4c49fd36fb1898e92257a1346e28b347b8b36d2` | `34e7656620964e6df778b67b86fc8df2a17900f87963b67b95499ef618343074` |

构建器身份均为 clean worktree，正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`。该 SHA 属于缓存 provenance，不等同于最终训练代码 SHA。

缓存与图文件传到 5080 时使用的归档大小为 1,896,581,632 字节，归档 SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。远端合并分块后先核对归档哈希，再解包到最终 clean clone。远端随后以 `--no-write-manifest` 再次运行严格缓存审计，结果为 `verified`，bundle ID 与 bundle digest 和本节完全一致。

## v2 落盘复核

### 重复预测时刻

v2 按事故生成样本。同一 forecast origin 可能因多个事故重复出现，导致相同交通输入与目标被多次计入训练和评估。落盘复核结果如下。

| 区域 | v2 样本行 | 唯一 forecast origin | 重复行 |
|---|---:|---:|---:|
| Alameda | 62789 | 45768 | 17021 |
| ContraCosta | 27225 | 23341 | 3884 |
| Orange | 105490 | 60026 | 45464 |

重复行不是单纯的存储浪费。它们会按照同一时刻附近的事故数量改变损失和指标权重。不同模型若使用事故分支、事故关系特征或 affected 监督，重复行的影响也不会自动抵消。

v2 同一 forecast origin 的首行通常不是距离预测时刻最近的 anchor。v3 对 forecast origin 做唯一化，并按以下稳定规则选择主事故：

1. 优先选择距离 forecast origin 最近的 anchor。
2. 距离相同时使用稳定 event ID 打破并列。
3. 保留该 forecast origin 的完整 origin event 集合与 active event 集合，避免主事故选择丢失分析信息。

### active event 跨 split

只对主事故做 event group 隔离不足以阻止同一个活跃事故跨 split。v2 的完整 active event 身份交集如下。

| 区域 | train 与 val 共享 active event | val 与 test 共享 active event |
|---|---:|---:|
| Alameda | 2 | 1 |
| ContraCosta | 3 | 0 |
| Orange | 6 | 0 |

v3 使用每个样本的完整 active event 集合执行切分清理。加载器再次检查 train、val 和 test 的 active event 集合两两不相交。该约束独立于目标时间集合互斥约束，两项都必须通过。

### 已停止的远程 v2 结果

最先完成的四项 v2 远程任务如下。

| 区域 | 配置 | 种子 | 状态 |
|---|---|---:|---|
| Alameda | s0a | 42 | contaminated preliminary |
| Alameda | s0c | 42 | contaminated preliminary |
| ContraCosta | s0a | 42 | contaminated preliminary |
| ContraCosta | s0c | 42 | contaminated preliminary |

这四项采用重复计权的 v2 样本，并且运行时仍存在本报告列出的训练实现问题。原文件保持只读，不能进入论文主表、显著性检验或跨协议平均。未完成的 v2 队列已经停止。

## v3 修复审查

### 唯一 forecast origin 与最近 anchor

缓存构建器现在为每个 forecast origin 只生成一行。主事故不再由输入顺序决定，而是使用最近 anchor 与稳定 event ID 选择。缓存保存 origin event 数量、active event 数量、主事故槽位和事件身份，加载器检查 forecast origin 唯一性、主事故槽位有效性和事件数量一致性。

### active event split 隔离

构建器在分配 split 时使用完整 active event 集合，而不是只使用主事故。跨边界样本会被清理，加载器对 active event 身份和目标时间集合执行独立的两两互斥校验。

Orange 的 affected 分析标签使用完整 active event 集合计算，然后再把事故特征截断到固定槽位数。分析标签不再受事故槽位截断影响。

### 全局 masked MAE

验证集 MAE 改为累计所有有效观测点的绝对误差和有效计数，最终计算：

```text
masked MAE = sum(abs(pred - target) over valid cells) / valid cell count
```

最佳检查点选择不再对 batch 均值或样本均值做二次平均。该修复对缺失率不一致和最后一个不完整 batch 尤其重要。正式 summary 同时保存误差和与有效计数，便于审计指标分母。

有效点数为零的单个 batch 可以跳过，避免局部空 mask 中断整个 split。全部 batch 累计后，如果整个 split 的有效点总数仍为零，指标计算必须失败。预测张量任意位置出现 NaN 或无穷值时也必须失败，即使该位置对应的 mask 为零。

### 反向图归一化

Graph WaveNet 与 Fourier Dual Net 相关入口统一通过受测图工具构造正向和反向支持。反向支持先转置邻接矩阵，再按对应方向归一化，避免把正向归一化结果直接转置造成错误的度数分母。

### epoch 感知区域采样

区域分桶采样器加入 epoch 状态。同一个种子仍可复现完整训练顺序，但不同 epoch 不再重复完全相同的桶顺序。

### blind 标签隔离

affected 从默认训练样本中移出，仅通过显式分析接口或明确的监督配置读取。DIST-Net 的 blind 配置同时关闭 affected 相关损失项，并使用结构性全假掩码，不再从真实分析标签派生训练信号。

### IGSTGNN 掩码与评估

H5 转换器显式导出 `x_mask` 与 `y_mask`。加载器在掩码缺失、形状错误或样本错位时直接失败。训练、验证和测试统一使用 `y_mask` 计算指标，不再用数值是否为零代替观测有效性。

数据加载和评估覆盖最后一个不完整 batch。elapsed time 在统一路径中传递。验证和测试指标按全部有效观测点累计，避免尾批丢失或 batch 平均改变全局结果。比较脚本要求掩码存在，并在共同有效观测集合上计算配对指标。

### cache 身份契约

正式缓存使用独立 schema 和协议版本。traffic、samples 与 event relation 文件通过区域身份、样本身份、内容 SHA-256 和 cache bundle 绑定。审计清单记录三地区 bundle 内容摘要。正式加载拒绝缺字段、形状不一致、身份错配、内容哈希不符和旧协议缓存。

构建器写入源文件哈希、代码 Git SHA、分支和 dirty 状态。正式构建默认要求 clean worktree。旧 v2 缓存不会被覆盖。

### runner 身份契约

远程 runner 使用固定任务清单启动最小判别矩阵。每个任务绑定以下身份：

- 完整 40 位代码 Git SHA。
- cache bundle ID 与 bundle digest。
- 协议版本与 schema。
- 唯一任务 ID、区域、配置、种子和输出相对路径。
- 独立的新 attempt 目录。

runner 在启动前验证 clean Git checkout、精确代码 SHA、缓存审计清单和任务输出路径。已存在的旧检查点、身份不一致的 summary 或不完整 attempt 不会被当成可恢复结果。

## 回归测试

最终训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a` 已纳入 `origin/fu` 提交 `625aacf` 中经复核合理的部分，并修正了协议路径测试在 Windows 上的分隔符断言。Mac 与 5080 的全套测试结果均为 143 passed、1 warning。新增与加强的测试覆盖：

- forecast origin 唯一化和最近 anchor 选择。
- 完整 origin event 与 active event 保留。
- active event split 隔离与目标时间互斥。
- 因果填补、训练段气候态和 post-train sentinel invariance。
- v3 schema、内容哈希、cache bundle 与 traffic、samples、relation 配对。
- masked MAE 全局累计与有效计数。
- 反向图支持归一化。
- epoch 感知区域采样。
- blind 模式的 affected 标签隔离。
- IGSTGNN 时间对齐、`x_mask`、`y_mask`、尾批和比较掩码。
- runner 清单、代码 SHA、缓存摘要、attempt 和 summary 完整性。

143 passed、1 warning 表示全套测试完成且没有测试失败。正式三地区缓存又通过 staging 与正式路径的两次严格落盘审计。代码测试和缓存审计都不能替代训练后的结果审计。

## 当前有效性判定

| 对象 | 状态 | 可用于论文结论 |
|---|---|---|
| 旧 XTraffic 缓存与结果 | contaminated | 否 |
| v2 最先完成的四项远程结果 | contaminated preliminary | 否 |
| v2 其余最小判别矩阵 | stopped | 否 |
| v3 训练代码 | verified checkout | 是正式训练输入，SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，远端 worktree clean |
| v3 正式缓存 | verified | 可作为 v3 训练输入，不能单独产生模型结论 |
| v3 最小判别矩阵 | running | 否，尚无完成并验收的新结果 |
| v3 baseline zoo | pending | 否，尚未启动 |

## 正式启动门槛

缓存侧前三项启动门槛已经完成：

1. 三地区正式缓存完成构建。
2. 缓存审计脚本通过全部 schema、唯一性、split 隔离、边界、图配对和内容哈希检查。
3. 审计清单落盘并记录正式 cache bundle ID 与 digest。

训练侧启动门槛也已完成：

4. 远程 checkout 与指定代码 SHA 完全一致，worktree clean。
5. runner 环境中的代码 SHA、cache bundle ID 与 digest 和本地审计清单一致。
6. 新结果写入 `outputs/protocol_v3_noleak_dedup` 下的新 attempt，不读取或覆盖 v2 输出。

远端正式 checkout 的 Git SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，worktree clean。runner 的 queue status 记录同一代码 SHA、cache bundle ID、bundle digest、schema 3 和 45 项任务。首项已进入新的 attempt 目录运行，未读取或覆盖 v2 输出。正式缓存保持 `verified`，最小判别矩阵状态更新为 `running`。
