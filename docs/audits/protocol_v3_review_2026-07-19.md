# XTraffic protocol v3 全面复核

复核日期：2026-07-19

状态更新：2026-07-27

复核状态：v3 代码审查与回归测试完成，正式缓存已在本地两次、5080 一次和 5090 一次通过严格审计。runner 层 45/45 已完成或重新验收，5080 的 27 项分片与 5090 的 18 项分片均已回传并通过结果服务器逐文件复验。只读 composite index 已于 2026-07-27 18:28:27 新加坡时间通过复验，最小判别矩阵状态为 `verified_host_stratified_45`。论文分析仍因同机配对、多种子和显著性检验尚未完成而暂停

正式缓存构建代码 SHA：`29159f626494de52c69b36fe5bba9854f13cb56c`

最终训练代码 SHA：`3a4952837aaf6fbda1403d50b610b1bf40c6098a`

测试结果：Mac、5080 与 5090 均为 143 passed、1 warning

最新控制层本地回归：252 passed、1 warning。finalizer 引号修复提交：`c48f232`。composite 内存修复提交：`5f3113b`。full-window 复验优化与最终 composite 控制提交：`93038ec`
协议版本：`protocol_v3_noleak_dedup`

缓存 schema：3

正式缓存目录：`outputs/dist_net/region_data_v3_noleak_dedup`

正式结果目录：`outputs/protocol_v3_noleak_dedup`

正式 cache bundle ID：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`

正式 cache bundle digest：`854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`

## 结论

v2 修复了 2026-07-18 审计发现的主要因果泄露，但后续全面复核确认，v2 仍存在重复预测时刻、active event 跨 split、验证集指标聚合错误及若干训练实现问题。最先完成的四项 v2 远程结果不能恢复为论文证据，状态统一为 `contaminated preliminary`。其余 v2 队列已经停止。

v3 已完成相应代码修复，并为缓存、训练和远程 runner 增加严格身份契约。正式缓存已经落盘，并在 staging 路径和正式路径各执行一次严格审计，两次均为 `verified`。5080 与 5090 又分别完成一次只读严格审计。本报告中的正式样本数、split 数、区域 digest 与 artifact SHA-256 均来自正式路径的 `cache_bundle_manifest.json`。

runner 层 45 个 canonical 任务均已完成或重新验收。最终固定分片为 5080 的 27 项与 5090 的 18 项，两组互不交叉，并集严格覆盖 canonical 45，所有直接 A/B 比较组保持同机。两台机器保留独立结果根、路径相关 `config_hash`、运行时 provenance 与 source-host postflight。最终产物只能称为 `host-stratified distributed 45`，不得称为 homogeneous 5090 matrix。

`rtx5090_18` 与 `rtx5080_27` 均已回传结果服务器，通过逐文件 SHA-256 复验并正式原子提升。task042 于 2026-07-26 23:32:54 新加坡时间完成，使 5080 达到 27/27。5080 finalizer 的 PowerShell 至 native ssh 引号传递故障只影响收尾复验。提交 `c48f232` 修复后复用了全部 152 个已上传文件，没有重训、重传或改写任务结果。`rtx5080_27` transfer marker SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。

首次 composite 复验显示旧 validator 会为 full-window prediction 额外创建两份全尺寸 float32 error 数组，在结果服务器上形成高内存峰值。提交 `5f3113b` 在 float64 audit 完成后原位复用 prediction 缓冲，并在首次 producer 指标校验前释放 actual。提交 `93038ec` 跳过 full-window 中未使用的 affected 与 unaffected 归约，同时完整保留 `all` 指标、有限性、valid_count、float64 audit 与 float32 producer 契约。随机差分、mask、无效 target NaN、staging 与正式目录双验收及 artifact 哈希不变性均通过。这些控制层修复不改变训练 checkout、summary、任务 `config_hash` 或任何正式结果。

控制提交 `93038ec` 的最终 composite 服务于 2026-07-27 18:28:27 新加坡时间以退出码 0 完成。index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`，`execution_layout` 为 `dual_host_stratified`，包含 2 个分片与 45 个任务。5080 分片包含 27 项，5090 分片包含 18 项，canonical tasks001-045 各出现一次。训练 SHA、cache bundle ID、bundle digest、manifest 规范 JSON SHA-256 `92b65401cd945502bd89cf77d42c11881d847b91b4b86aac3b0d40b2988bdb6c` 与两份 source-host postflight 身份全部一致。服务只生成只读 index，没有改写任何 summary 或 artifact。

首次由提交 `c48f232` 控制的 composite 尝试在结果服务器重启前没有形成输出。该尝试作为独立 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。重启后两份正式分片哈希保持不变，最终服务没有复用不完整输出。

composite 完成只恢复 canonical 45 矩阵的完整性有效性，不恢复多种子结论、显著性结论或论文最终结论。5080 上主动终止的旧 task005、5090 边界切换时未形成 summary 的旧 task007，以及 task007 首次内存故障 attempt 均继续作为历史运行记录保留，不计为正式结果。同机配对、多种子和显著性分析仍待完成，baseline zoo 继续保持 `pending`。

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

缓存与图文件传到 5080 时使用的归档大小为 1,896,581,632 字节，归档 SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。远端合并分块后先核对归档哈希，再解包到最终 clean clone。5080 随后以 `--no-write-manifest` 再次运行严格缓存审计，结果为 `verified`。同一正式缓存传到 5090 后也通过一次 `--no-write-manifest` 严格审计。两台远端机器的 bundle ID 与 bundle digest 均和本节完全一致。

## v3 远程执行身份与当前结果

### 结果根与运行身份

5080 结果根为：

`C:\Users\ASUS\traffic_fourier_v3_final\outputs\protocol_v3_noleak_dedup\remote_runs\minimal_45_5080`

5090 结果根为：

`/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090`

两台机器均使用训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a` 和同一个正式 cache bundle。5080 的 manifest 文件采用 CRLF 换行，原始文件 SHA-256 为 `63ea48fcae187e0aa160fa2499dcb2eca46ac9cdead421807abedf6c1b9bf71a`。5090 的 manifest 文件采用 LF 换行，原始文件 SHA-256 为 `7d64c2e7ffcef8088691d63548f7c3addae662a0b52d11291a38cf1aebc8d25c`。两份文件解析后的规范 JSON SHA-256 均为 `92b65401cd945502bd89cf77d42c11881d847b91b4b86aac3b0d40b2988bdb6c`。

两个原始 manifest 摘要的差异只来自换行字节。两份 manifest 解析后具有相同的 45 个任务、相同任务顺序和相同字段值。任务 `config_hash` 不使用原始 manifest 文件摘要，因此换行差异不改变任务身份或结果身份。审计时仍需同时保留原始文件摘要和规范 JSON 摘要，不能把两个原始文件错误写成字节完全相同。

上述两份文件用于确认 canonical production manifest 身份。最终 27 项和 18 项分片不是另外维护的手写任务表，而是由分片器从这份 canonical manifest 只替换任务子集生成。组合验收还会再次要求分片名称与任务集合精确匹配固定契约。

两台远端机器的训练运行时如下。正式矩阵保留两个独立分片，不把两套运行时、绝对路径或路径相关 `config_hash` 混成同机结果。所有直接比较组在分片设计中保持同机。

| 主机 | Python | PyTorch | CUDA | NumPy | h5py | pandas |
|---|---|---|---|---|---|---|
| RTX 5080 | 3.13.3 | 2.10.0.dev20251203+cu128 | 12.8 | 2.2.6 | 3.14.0 | 2.3.1 |
| RTX 5090 | 3.12.3 | 2.8.0+cu128 | 12.8 | 2.3.2 | 3.16.0 | 3.0.3 |

### 首批已验收的六项 v3 结果，2026-07-20 快照

| 主机 | 任务 | test all masked MAE | test affected masked MAE | test unaffected masked MAE | best val masked MAE |
|---|---|---:|---:|---:|---:|
| 5080 | 001 Alameda s0a seed 42 | 11.339531 | 17.139917 | 11.005998 | 11.682158 |
| 5080 | 002 Alameda s0c seed 42 | 11.025200 | 16.583555 | 10.705584 | 10.939075 |
| 5080 | 003 ContraCosta s0a seed 42 | 12.066377 | 18.344279 | 11.771324 | 11.759196 |
| 5080 | 004 ContraCosta s0c seed 42 | 11.718486 | 17.784800 | 11.433378 | 11.424961 |
| 5090 | 005 Orange s0a seed 42 | 12.025650 | 17.932862 | 11.720226 | 11.872927 |
| 5090 | 006 Orange s0c seed 42 | 11.740712 | 17.583309 | 11.438629 | 11.547319 |

六个 summary 与 prediction artifact 均通过 runner 验收。Alameda 的 s0c 减 s0a 差值为 all `-0.314331`、affected `-0.556362`、unaffected `-0.300414`。ContraCosta 的对应差值为 all `-0.347891`、affected `-0.559479`、unaffected `-0.337946`。Orange 的对应差值为 all `-0.284938`、affected `-0.349553`、unaffected `-0.281597`。这些差值只描述 seed 42 的已落盘结果，不构成多种子稳定性结论，也不能恢复旧结果或 v2 结果的有效性。

5080 开始任务 005 后，Orange 配置在该机器上长时间未完成首个 epoch。5090 完成同配置显存预检并正式完成 epoch 1 后，5080 的任务 005 被主动终止以避免重复占用。runner 因子进程退出而记录 code 1。该状态仅表示任务迁移，不表示模型训练失败。任务 001 到 004 的已验收目录保持不变。5080 现已重新投入 24 项正式分片训练。

### 双机固定分片，2026-07-20 历史快照

分片规则由 `scripts/remote/distributed/protocol_v3_distributed.py` 固定，并从 canonical production manifest 只替换任务子集。分片器验证任务集合互斥、并集为 45，并验证每个直接比较组只属于一台主机。

5080 runner 重新验收并跳过任务 001 到 004，状态为 4/24、skipped valid 为 4，正在训练 `007_event_staeformer_s0a_alameda_seed1`。task 007 的 `attempt_0001` 在 epoch 9 数据装载阶段触发 NumPy `ArrayMemoryError`。两条遗留诊断 PowerShell 扫描当时占用约 48 GB 私有内存。该 attempt 没有 summary 或正式任务目录，失败记录原样保留。停止两条诊断进程后，runner 以 `attempt_0002` 重启 task 007，训练与 GPU 均恢复正常。该记录属于主机运维故障，不是模型失败、数据协议失败或训练数值失败。

截至 2026-07-21 00:02 新加坡时间，5080 的 task 007 `attempt_0002` 已完成 epoch 7，train 为 12.0424，val 为 12.6845，用时 359 秒，GPU 利用率为 92%。

5090 runner 重新验收并跳过任务 005 和 006，状态为 2/21、skipped valid 为 2，正在训练 `011_event_staeformer_s0a_orange_seed1`。截至同一时间，task 011 已完成 epoch 8，train 为 12.3250，val 为 12.3937，用时 659 秒，并保存当前 best checkpoint。

两个结果根分别执行 source-host postflight。postflight 复验 queue、任务目录、prediction artifact、summary、任务配置哈希、缓存 bundle、Git SHA 和运行时 provenance，并只导出分片被分配的正式任务、状态和日志。旧任务 005 迁移记录、旧 task 007 残留、未完成 staging 与其他主机任务都不进入 allowlist。两份结果回传后，只读 composite postflight 已验证 canonical 45 完整并集和逐文件 SHA-256，没有复制任务目录，也没有改写既有 summary。

### 自动回传与关机契约，2026-07-21 历史快照

本节记录最初 24/21 分片的自动收尾契约。该契约后来被 27/18 分片契约替代，不代表当前任务分配或结果目录。

5090 已在 `/root/autodl-tmp/protocol_v3_finalize_5090_shard.sh` 部署分片自动收尾守护进程，当前 PID 为 106466。守护进程只有在以下条件全部满足时才关机：

1. 5090 分片 queue status 为 completed，完成数为 21/21，skipped valid 数为 2。
2. source-host postflight 对 21 项重新执行 summary 身份、必需 artifact、prediction artifact 指标复算和运行时 provenance 检查。
3. 严格 allowlist 同步到 `root@8.134.12.5:/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/shards/rtx5090_21`。
4. 结果服务器按逐文件 SHA-256 清单完成全量复验，并将 incoming 目录原子提升为正式目录。

5080 的 PowerShell 收尾进程 PID 为 12524。它等待 5090 上传完成标记，再验收并回传 24 项到同一父目录下的 `rtx5080_24`，成功后保持实验室工作站开机。5090 收尾进程 PID 为 106466。两个进程都已通过独立进程检查和持续 WAIT 日志复验。任一条件失败时，相应收尾进程都会保持机器开机。测试已覆盖队列未完成、postflight 失败、服务器空间不足、传输失败和远端哈希失败。结果服务器约有 6.3 GB 空闲，可回收的未使用 Docker build cache 为 12.25 GB。缓存清理尚未授权，双分片粗估约 11.6 GB，因此完整回传当前仍有容量阻塞。

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

最终训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a` 已纳入 `origin/fu` 提交 `625aacf` 中经复核合理的部分，并修正了协议路径测试在 Windows 上的分隔符断言。Mac、5080 与 5090 的全套测试结果均为 143 passed、1 warning。新增与加强的测试覆盖：

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
| v3 训练代码 | verified checkout | 是正式训练输入，SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，5080 与 5090 的远端 worktree 均 clean |
| v3 正式缓存 | verified | 可作为 v3 训练输入，不能单独产生模型结论 |
| v3 最小判别矩阵 | verified_host_stratified_45 | 仅矩阵完整性有效。只读 composite 已验证 5080 的 27 项与 5090 的 18 项互斥且覆盖 canonical 45，但同机配对、多种子和显著性分析尚未完成，不能恢复论文性能结论 |
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

5080 与 5090 的正式 checkout 均为 Git SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，worktree clean。两台机器的 runner 身份均绑定同一 cache bundle ID、bundle digest 和 schema 3。runner 层已完成或重新验收 45/45，最终分片为 5080 的 27 项与 5090 的 18 项。两份正式分片均已回传结果服务器，通过逐文件 SHA-256 复验并原子提升。两分片保持独立结果身份，只通过只读 composite index 验证完整性。正式缓存保持 `verified`，最小判别矩阵状态为 `verified_host_stratified_45`。composite 只恢复矩阵完整性有效性。同机配对、多种子和显著性分析完成前，不恢复论文性能结论。

## 2026-08-05 v3 event baseline NAS 灾备控制

v3 event baseline 主矩阵为 33 项。2026-08-05 18:38 的 runner 层状态为 14/33，5080 已验收 tasks001-010，Orange 已验收 4 项。5080 正在训练 task011 Alameda D2STGNN。这个运维快照不是完整性结论。

5080 的训练后 NAS 回传使用 `config/protocol_v3/event_baseline_5080_nas_backup_v1.json` 作为固定计划，SHA-256 为 `738bc46122f9af6466c1b605013b57c2cc05874aadcda6682cdf08fb4ed964c8`。计划声明 tasks001-022 且只声明 Alameda 与 Contra Costa，三个结果根的 task ID 互斥，并按 canonical manifest 顺序组成精确 22 项。最后的 tasks010-022 恢复队列是完成门，必须为 13/13 completed 且 runner 零退出。

`scripts/remote/windows/protocol_v3_event_host_postflight.py` 对每项重新派生 `config_hash`，根据声明的代码 SHA 复验 summary，从 prediction artifact 重算 masked metric，检查必需 artifact 和 completed state，最后对白名单文件逐一哈希。它不把前两个按设计遇错停止的 queue 写成整队成功。PowerShell finalizer 实现已验证文件复用、`.part` 上传、NAS 端 SHA-256、精确树比对和原子改名。任何失败都保持主机开机。

NAS 目标为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts/event_rtx5080_alameda_contra`。专用公钥指纹为 `SHA256:Nx5BbjtFe9eRbSJh4hV/YN+reR2pw4hZuFLa9PLKMGc`，私钥未进入仓库。当前等待器正在运行，尚未触发正式上传。该副本完成后只能标记为经验证灾备，不能代替结果服务器正式发布和只读 composite。postflight 核心可由 Linux 5090 复用，但需独立 host ID、分片计划、目标目录、SSH 密钥和 Linux 持久包装。本地全仓回归为 526 passed 和 2 个既有 warning。

## 2026-08-06 v3 event baseline 运行复验

15:28 新加坡时间，全矩阵 runner 层仍为 14/33。5080 的已验收集合为 tasks001-010，Orange 的已验收集合为 HL、LSTM、GraphWaveNet 和 STTN。5080 恢复队列为 running、1/13 completed，当前 task011 Alameda D2STGNN 已进入 epoch 6、batch 500/1001。GPU 利用率为 100%，显存约 15.9 GB，温度约 45 摄氏度。已完成 epoch 的单轮用时约为 15,943 秒，因此 D2STGNN 是当前明显瓶颈。

NAS finalizer 状态仍为 Running，最近日志持续显示 `WAIT queue_state=running completed=1/13`。NAS 的 5080 正式目录与 incoming 均不存在，说明完成门没有被越过。Orange 主机仍为 TCP 可达后由远端立即关闭 SSH，无法核对新结果。Orange 4 项 NAS 灾备仍完整，共 26 个 regular files，NAS `/volume1` 约有 427 GB 可用。15:30 结果服务器可用空间为 `2939228160` bytes，正式结果根为 `21960917234` bytes，仍不足以接受完整 Orange 分片和原子发布余量。本快照只是运维证据，不改变当前有效性标签。
