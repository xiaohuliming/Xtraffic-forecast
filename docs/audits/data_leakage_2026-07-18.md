# XTraffic 数据泄露审计

审计日期：2026-07-18
复核日期：2026-07-27
审计状态：旧问题已确认，v2 复核又发现重复预测时刻、事件跨切分与训练评估实现问题。v3 代码已修复，正式缓存已通过本地两次、5080 一次和 5090 一次严格审计。runner 层 45/45 已完成或重新验收，5080 的 27 项分片与 5090 的 18 项分片均已回传并通过结果服务器逐文件复验。只读 composite index 尚未完成
审计对象：仓库代码与 `outputs/dist_net/region_data/` 中的三个区域缓存

## 当前修复状态

旧缓存、v2 缓存和旧输出保持只读。v2 最小判别矩阵已经停止。其最先完成的四项只能作为 `contaminated preliminary` 保留，不能恢复论文结论。

v3 代码使用 `protocol_v3_noleak_dedup`，默认缓存目录为 `outputs/dist_net/region_data_v3_noleak_dedup`，默认结果目录为 `outputs/protocol_v3_noleak_dedup`。正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。当前代码已纳入 `origin/fu` 提交 `625aacf` 中经复核合理的部分。Mac、5080 与 5090 的训练 checkout 测试结果均为 143 passed、1 warning。正式缓存已在 staging 路径、正式路径、5080 和 5090 各通过一次严格审计，状态为 `verified`。正式 bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，总 digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。runner 层 45/45 已完成或重新验收。最终 5080 分片为 27/27，5090 分片为 18/18，两份正式分片均已回传并通过结果服务器逐文件复验。最新控制层本地回归为 252 passed、1 warning。提交 `5f3113b` 以数值等价的原位计算降低 composite 复验内存，提交 `93038ec` 跳过 full-window 中未使用的事件分组归约。两项修复均不改变正式结果或训练身份。只读 composite index 尚未完成，因此论文分析继续暂停。

完整 v3 复核记录见 [protocol_v3_review_2026-07-19.md](protocol_v3_review_2026-07-19.md)。

已实现：

- `dist_net/protocol.py` 统一定义因果前向填充、完整目标窗切分、event group 隔离、full-window 索引、forecast window 边界、训练拟合掩码和可部署事故特征。
- `scripts/build_full_county_cache.py` 对同一预测时刻只保留一个样本，以最近 anchor 为主事故，并将完整 active event 集合用于切分隔离与 affected 分析标签生成。
- `scripts/build_full_county_cache.py` 不再调用事故匹配生成气候态，不再把最终 duration 和 severity 写入可部署输入，并将 affected 移到显式分析接口。
- v3 H5 记录 schema、协议、精确切分边界、拟合范围、源文件 SHA-256、Git SHA、分支和 dirty 状态。
- `dist_net/data.py` 默认拒绝旧协议缓存，检查唯一 forecast origin、active event 跨 split 隔离、目标时间互斥、缓存配对身份与内容哈希。
- ICSF 使用存储的 `primary_incident_slot` 和预测时刻已知的 elapsed time。
- `scripts/build_igstgnn_data_from_h5.py` 与 RegionData 使用同一 forecast origin，保留 `x_mask` 与 `y_mask`，并校验 anchor event。
- 训练入口按全部有效观测点累计验证集 masked MAE，修正反向图归一化，按 epoch 更新区域采样顺序，并在 blind 模式隔离 affected 标签。
- masked metric 允许跳过有效点数为零的单个 batch，但整个 split 的有效点总数为零时必须失败。预测张量任意位置出现非有限值时也必须失败。
- IGSTGNN 训练与评估显式使用 `y_mask`，覆盖尾批，统一累计全局指标与 elapsed time。
- 缓存和远程 runner 使用协议版本、cache bundle、内容摘要与 Git SHA 的严格身份契约，正式运行拒绝 dirty worktree、身份不匹配和旧检查点复用。
- 回归测试覆盖 prefix invariance、完整目标窗、active event split 隔离、full-window 目标互斥、label-free 拟合范围、事后字段不变性、最近主事故选择、IGSTGNN 窗口与掩码对齐、全局 masked MAE、图归一化、epoch sampler、blind 标签隔离、cache 身份和 runner 身份。

执行进展：

- 2026-07-19 完成 v2 缓存复核，确认三个区域均存在同一 forecast origin 的重复样本。
- 2026-07-19 停止 v2 的 5080 串行队列。最先完成的四项保留原文件，但状态均为 `contaminated preliminary`。
- 2026-07-19 完成 v3 代码复核，Mac 与 5080 的全套测试结果均为 143 passed、1 warning。
- 正式 v3 缓存已完成构建，并在 staging 路径、正式路径和 5080 各通过一次严格审计。
- 5080 的 v3 任务 001 到 004 已完成，prediction artifact 与 summary 均通过 runner 验收。
- 5080 的任务 005 因迁移到 5090 而被主动终止。runner 记录 code 1，这不是模型失败。
- 5090 使用精确代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，全套测试为 143 passed、1 warning，同一 bundle 的只读严格审计为 `verified`。
- 5090 已完成任务 005 和 006，并通过 runner 验收。旧 41 项队列随后在任务边界安全停止。旧 runner 瞬间为 task 007 创建的零字节日志和未完成 staging 没有 summary，已从正式导出中排除。
- 2026-07-20 22:23 新加坡时间起，5080 固定执行 24 项，5090 固定执行 21 项。当时快照分别为 4/24 和 2/21，正在 task 007 与 task 011。两分片互斥、并集为 canonical 45，所有直接比较组保持同机。
- 5080 的 task 007 `attempt_0001` 在 epoch 9 数据装载阶段因主机内存耗尽触发 NumPy `ArrayMemoryError`。两条遗留诊断 PowerShell 扫描占用约 48 GB 私有内存。该 attempt 没有 summary 或正式任务目录。停止两条诊断进程后内存恢复，runner 已用 `attempt_0002` 重启 task 007。这是主机运维故障，不是模型、数据协议或训练数值失败。
- 截至 2026-07-21 00:02 新加坡时间，5080 的 task 007 `attempt_0002` 已完成 epoch 7，train 为 12.0424，val 为 12.6845，用时 359 秒，GPU 利用率为 92%。5090 的 task 011 已完成 epoch 8，train 为 12.3250，val 为 12.3937，用时 659 秒，并保存当前 best checkpoint。
- 2026-07-22 完成 27/18 原子重分片。5080 固定承担 27 项，5090 固定承担 18 项，所有直接比较组保持同机。
- 2026-07-26 23:32:54 新加坡时间，task042 完成并通过 runner 验收。runner 层达到 45/45，5080 达到 27/27。
- 2026-07-27 11:19:49 新加坡时间，`rtx5080_27` 完成结果服务器逐文件复验与正式原子提升。5080 finalizer 引号问题的修复提交为 `c48f232`，该故障没有影响训练结果、source-host postflight 或已上传文件。
- 因此下文旧结果和 v2 初步结果的 `contaminated` 状态不变。

下文“代码证据”描述审计时的旧实现，可在 Git 历史提交 `c89bcfb` 中复核，不代表当前 v2 文件仍含同一代码。

## v3 正式缓存验收

正式审计清单为 `outputs/dist_net/region_data_v3_noleak_dedup/cache_bundle_manifest.json`。清单状态为 `verified`，缓存 schema 为 3，协议为 `protocol_v3_noleak_dedup`。

| 区域 | 正式样本数 | train | val | test | 去除重复行 | mixed active 清理 |
|---|---:|---:|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 | 17021 | 8 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 | 3884 | 4 |
| Orange | 60015 | 42372 | 9098 | 8545 | 45464 | 11 |

| 区域 | 区域 cache bundle digest |
|---|---|
| Alameda | `736bc6e8db2b93dded9e8c72aca00acfeb28dbb2bbaa3053c3f7d5ca2e83c682` |
| ContraCosta | `0eaaaa34100ef6cf4bea50b514156fd913b58db2197696b2ea21f0cb01ee8d45` |
| Orange | `06ea38c6d0d31890bc9a238a7d4d890f2ec20a5ba17a527b291526e6806deb95` |

这些数量和摘要均来自正式落盘 manifest。缓存验收只确认数据协议和 artifact 身份通过审计，不产生模型性能结论。

## v3 首批六项训练有效性，2026-07-20 快照

已完成并通过 runner 验收的六项如下：

| 主机 | 任务 | test all masked MAE | test affected masked MAE | test unaffected masked MAE | best val masked MAE |
|---|---|---:|---:|---:|---:|
| 5080 | 001 Alameda s0a seed 42 | 11.339531 | 17.139917 | 11.005998 | 11.682158 |
| 5080 | 002 Alameda s0c seed 42 | 11.025200 | 16.583555 | 10.705584 | 10.939075 |
| 5080 | 003 ContraCosta s0a seed 42 | 12.066377 | 18.344279 | 11.771324 | 11.759196 |
| 5080 | 004 ContraCosta s0c seed 42 | 11.718486 | 17.784800 | 11.433378 | 11.424961 |
| 5090 | 005 Orange s0a seed 42 | 12.025650 | 17.932862 | 11.720226 | 11.872927 |
| 5090 | 006 Orange s0c seed 42 | 11.740712 | 17.583309 | 11.438629 | 11.547319 |

这些是来自修复后 v3 协议的 individually validated 结果，但六项均只有 seed 42。它们不能支持多种子稳定性、显著性或论文最终结论，也不能使旧缓存、v2 缓存或四项 v2 初步结果恢复有效。

5080 上的任务 005 因迁移到 5090 而被主动终止。runner 记录的 code 1 来自人为停止子进程，不是模型失败。已验收的任务 001 到 004 保持不变。

5080 的正式结果根为 `C:\Users\ASUS\traffic_fourier_v3_final\outputs\protocol_v3_noleak_dedup\remote_runs\minimal_45_5080`。5090 的正式结果根为 `/root/autodl-tmp/protocol_v3_noleak_dedup/remote_runs/minimal_45_5090`。两台机器分别执行 source-host postflight，保留独立 `config_hash`、绝对路径和运行时 provenance。

最终正式分片为 5080 的 tasks001-004、007-010、013-016、019-030、040-042，以及 5090 的 tasks005-006、011-012、017-018、031-039、043-045。5080 为 27/27，5090 为 18/18，runner 层为 45/45。两份分片均已回传并通过结果服务器逐文件 SHA-256 复验与原子提升。只读 composite index 将验证两分片互斥、并集为 canonical 45，且不复制任务目录或改写 summary。完成 composite 和多种子统计前，不恢复论文性能结论。

## v2 后续复核发现

v2 已修复最初审计中的未来插值、标签气候态、事后事故输入、目标窗口跨边界和 IGSTGNN 预测时刻错位，但仍存在会影响实验有效性的独立问题。

### 重复 forecast origin

同一预测时刻可能由多个事故各生成一行，导致完全相同的交通输入与目标被重复用于训练和评估。重复行会按事故密度改变样本权重，使指标不再代表唯一预测时刻的平均误差。

| 区域 | v2 样本行 | 唯一 forecast origin | 重复行 |
|---|---:|---:|---:|
| Alameda | 62789 | 45768 | 17021 |
| ContraCosta | 27225 | 23341 | 3884 |
| Orange | 105490 | 60026 | 45464 |

v2 的同一 forecast origin 采用首行并不能稳定代表最近 anchor。v3 改为每个 forecast origin 只保留一行，主事故按距离预测时刻最近的 anchor 选择，稳定 event ID 用于打破并列。

### active event 跨 split

只隔离主事故不足以防止同一个活跃事故通过另一个样本进入相邻 split。v2 的 active event 身份交集如下。

| 区域 | train 与 val 共享 active event | val 与 test 共享 active event |
|---|---:|---:|
| Alameda | 2 | 1 |
| ContraCosta | 3 | 0 |
| Orange | 6 | 0 |

v3 使用每个样本的完整 active event 集合执行切分隔离。跨 split 的 event group 必须从相邻 split 中清除，加载时再次校验集合两两不相交。

### 训练与评估实现

v2 复核还确认以下问题：

- 验证 MAE 曾先按 batch 或样本求均值，再对均值平均。缺失掩码不同或尾批较小时会改变最佳检查点选择。
- 部分 Graph WaveNet 与 Fourier Dual Net 入口的反向邻接归一化方向错误。
- 区域分桶采样器每个 epoch 重复相同顺序。
- DIST-Net 的 blind 配置仍可能通过损失权重读取 affected 标签。
- IGSTGNN 转换未完整保留目标缺失掩码，训练和评估未统一处理尾批、elapsed time 与全局 masked metric。
- v2 runner 缺少严格的代码、缓存和结果身份绑定，存在旧检查点或错配缓存被误用的风险。

这些问题均纳入 v3 代码修复。正式缓存已经通过审计，模型结论的有效性仍取决于新训练落盘结果。

### 已停止的 v2 初步远程结果

以下四项已经完成，但来自重复计权的 v2 样本，只能保留为 `contaminated preliminary`：

| 区域 | 配置 | 种子 | 状态 |
|---|---|---:|---|
| Alameda | s0a | 42 | contaminated preliminary |
| Alameda | s0c | 42 | contaminated preliminary |
| ContraCosta | s0a | 42 | contaminated preliminary |
| ContraCosta | s0c | 42 | contaminated preliminary |

其余 v2 队列已停止。任何 v2 初步数值都不能与后续 v3 结果合并统计，也不能用于论文主表。

## 结论

旧版 XTraffic 管线存在三个 P0 问题和两个 P1 问题。

最严重的问题是，缓存构建器先对全年每个节点和通道做双向线性插值，再切分训练集、验证集和测试集。历史输入中的缺失值因此可能由预测时刻之后的真实观测填出。几乎所有事件样本至少含一个依赖未来观测的锚点输入，且 23.3187% 到 54.7144% 的样本直接使用了本样本预测窗内的观测作为插值右端点。

第二个 P0 问题是，RADS 系列使用的气候态基线在拟合前读取事故记录并排除事故相关时刻，因此当前的严格 `label-free` 表述不成立。

第三个 P0 问题是，事故严重度、受影响节点标签和最终事故持续时间含有事后信息，却被写入缓存并用于部分模型的输入或损失。

因此，所有使用旧 XTraffic H5 缓存的绝对数值暂停用于论文。共享缓存的配对差可能仍保留方向信号，但不同模型可能以不同方式利用缺失值，不能假定泄露会自动抵消。旧结果应只读保留并标记为 `contaminated`，无泄露结果必须写入新目录。

## 审计口径

未来插值统计以事件样本行计权。对每个样本锚点 `t`、节点 `n` 和 flow 通道，如果 `mask[t,n]` 为零且 `t` 之后存在下一条原始观测，则缓存中的锚点输入依赖未来。如果下一条原始观测位于 `t+1` 到 `t+12`，则记为直接使用本样本预测窗观测。

切分重叠统计使用每个 split 的全部目标时间索引并集。每个样本目标区间为 `t+1` 到 `t+12`。

所有统计均从现有 H5 文件只读计算，没有改写缓存或实验结果。

## P0-1 全年双向插值读取未来观测

### 代码证据

- `scripts/build_full_county_cache.py:145` 到 `scripts/build_full_county_cache.py:164` 对完整年度序列调用 `np.interp`。
- `dist_net/data.py:126` 到 `dist_net/data.py:148` 直接将 `flow_series_imputed` 切成历史输入和预测目标。
- 插值发生在数据切分之前，模型输入没有用原始缺失掩码替换这些未来依赖值。

### 落盘量化

| 区域 | flow 原始缺失率 | 锚点 node-cell 依赖未来 | 锚点 node-cell 直接读预测窗 | 含未来插值的样本 | 直接读预测窗的样本 |
|---|---:|---:|---:|---:|---:|
| Alameda | 18.6543% | 6.1056% | 0.1178% | 99.7962% | 23.3187% |
| ContraCosta | 17.1966% | 5.6659% | 0.1295% | 99.8017% | 34.1204% |
| Orange | 19.4992% | 6.2701% | 0.1350% | 99.9592% | 54.7144% |

一个确定性例子来自 Alameda 测试样本 `sample_idx=53456`。锚点为 `t=89377`，节点 519 的前一条 flow 观测位于 `t=89369`，值为 157。下一条观测位于预测窗内的 `t=89388`，值为 77。缓存锚点值为 123.315788，与两个端点的线性插值精确相等。

### 影响范围

所有读取 `outputs/dist_net/region_data/` 的 XTraffic 实验均受影响：

- GWN、FDN、STID 与 STAEformer
- RGDN、RADS、RADS-O、NDR-Net、RFN 与 RAD-GMixer
- DIST-Net、GWN 加 ICSF、STAEformer 加 ICSF
- XTraffic full-window 九宫格
- 当前 XTraffic baseline zoo
- 从 H5 转换到 IGSTGNN 的实验

`scripts/train_benchmark.py` 使用的 METR-LA、PEMS04 和 PEMS08 标准基准不经过该缓存，不受这一项影响。

### 修复要求

新缓存必须满足 prefix invariance。任意修改时刻 `t` 之后的数据，都不能改变时刻 `t` 及以前的模型输入。

推荐使用仅向前填充，并保留显式缺失掩码。序列开头没有历史观测时，只能使用由训练段拟合的先验值或固定常数。不能使用右侧观测填补左侧缺失。

## P0-2 RADS 气候态基线依赖事故标签

### 代码证据

- `scripts/build_full_county_cache.py:486` 到 `scripts/build_full_county_cache.py:501` 先调用事故匹配与事故窗口掩码，再拟合 `baseline_median`。
- `scripts/build_impact_labels.py:351` 到 `scripts/build_impact_labels.py:363` 使用事故起点、最终持续时间和候选节点确定要排除的数据。

### 落盘量化

| 区域 | 训练 node-time 被事故标签排除 | 至少一个节点被排除的训练时刻 | 无事故掩码基线与缓存基线的 MAE |
|---|---:|---:|---:|
| Alameda | 15.0133% | 97.63% | 1.340 |
| ContraCosta | 12.8380% | 94.02% | 1.061 |
| Orange | 11.6590% | 96.94% | 0.922 |

在计数不少于 8 的基线格子中，两个版本差值大于 1 的比例分别为 18.78%、17.17% 和 14.64%。差值大于 5 的比例分别为 4.66%、3.30% 和 3.22%。

### 影响范围

凡是读取 `x_baseline` 或 `y_baseline` 的方法都受影响，包括 RGDN、RADS、RADS-O、NDR-Net、RFN 与 RAD-GMixer。旧版 s0a 与 s0c 的差值可能混入事故标签清洗气候态带来的贡献，因此当前的“不读任何事故标签”主张不成立。

### 修复要求

建立真正 label-free 的训练段气候态。它只能使用训练段的原始有效流量，或仅由流量自身定义并预注册的鲁棒异常规则。事故掩码版本如需保留，必须明确命名为 `label-informed` 并作为单独消融。

## P0-3 事故输入和监督包含事后信息

### 代码证据

- `scripts/build_impact_labels.py:550` 到 `scripts/build_impact_labels.py:619` 使用事故后的真实流量计算 severity 与 affected。
- `scripts/build_full_county_cache.py:207` 到 `scripts/build_full_county_cache.py:215` 将最终 duration 与 severity 写入 `incident_feat`。
- `dist_net/data.py:202` 到 `dist_net/data.py:205` 在三个 split 中返回事故特征与 affected。
- `dist_net/incident_branch.py` 读取完整 `incident_feat`。
- `dist_net/losses.py:46` 到 `dist_net/losses.py:56` 使用 future-derived affected 加权损失。
- `scripts/train_gwn_icsf.py:48` 到 `scripts/train_gwn_icsf.py:62` 读取最终 duration。
- `scripts/train_staeformer_icsf.py:47` 到 `scripts/train_staeformer_icsf.py:61` 读取最终 duration。

### 落盘量化

| 区域 | active slot 中 severity 非零 | active slot 中最终 duration 非零 |
|---|---:|---:|
| Alameda | 99.714% | 96.043% |
| ContraCosta | 99.847% | 95.531% |
| Orange | 99.819% | 94.979% |

### 影响范围

DIST-Net 读取严重度并用 affected 监督。GWN 加 ICSF 与 STAEformer 加 ICSF 未读取严重度，但明确读取最终事故持续时间。H5 到 IGSTGNN 的转换也输出最终 duration。

affected 可以保留为事后分组评估标签，但必须与可部署输入物理隔离。验证和测试推理只能使用预测时刻已经知道的信息，例如事故类型、位置、起报时间和当前已持续时间。

## P1-1 相邻 split 共享目标时刻和事件

### 代码证据

- `scripts/build_full_county_cache.py:371` 到 `scripts/build_full_county_cache.py:373` 只按样本锚点分配 split。
- `dist_net/data.py:138` 到 `dist_net/data.py:142` 的实际目标是锚点之后 12 步。
- `dist_net/data.py:228` 到 `dist_net/data.py:236` 先生成全部 full-window，再按窗口序号切分。

### 落盘量化

| 区域 | 事件协议 train 与 val 共享目标时刻 | 事件协议 val 与 test 共享目标时刻 | 跨 train 与 val 的 event_idx | 跨 val 与 test 的 event_idx |
|---|---:|---:|---:|---:|
| Alameda | 9 | 10 | 1 | 6 |
| ContraCosta | 11 | 0 | 0 | 0 |
| Orange | 10 | 10 | 5 | 1 |

full-window 协议在 train 与 val 边界共享 11 个目标时刻，在 val 与 test 边界也共享 11 个目标时刻。

### 修复要求

按完整预测目标区间分配 split，丢弃跨边界窗口。事件协议还需按 `event_idx` 做组隔离，保证同一事件不会跨 split。

## P1-2 训练统计越过 split 边界

多个脚本用以下方式计算标准化或残差统计的上界：

```python
hi = int(train_sample_start.max()) + T_h + T_p
```

相关入口包括：

- `scripts/train_staeformer_xtraffic.py`
- `scripts/train_staeformer_icsf.py`
- `scripts/train_rgdn.py`
- `scripts/train_staeformer_deseason.py`
- `scripts/train_rfn.py`
- `scripts/train_rad.py`
- `scripts/build_igstgnn_data_from_h5.py`

事件协议中三个地区的最大训练锚点均为 73583，代码得到 `hi=73607`，而 70% 边界为 73584。因此统计切片读取了 23 个验证时刻。full-window STAEformer 的统计又越过最后一个训练目标 11 步，并与验证目标并集重叠。

统计范围必须由显式 `train_end` 决定，不能从样本锚点反推。修改训练边界之后的数据时，训练标准化量和气候态必须保持字节不变。

## IGSTGNN 转换中的时间语义错位

`dist_net/data.py` 把 `sample_start=t` 定义为预测锚点，历史为 `t-11` 到 `t`，目标为 `t+1` 到 `t+12`。

`scripts/build_igstgnn_data_from_h5.py:150` 到 `scripts/build_igstgnn_data_from_h5.py:152` 却把同一个 `t` 当作历史起点，生成历史 `t` 到 `t+11`，目标 `t+12` 到 `t+23`。相对 RegionData 协议整体右移了 11 步，事故元数据仍来自原始 `t`。

这不是 IGSTGNN 样本内部的 x 与 y 重叠。问题是它与仓库主协议不是同一预测时刻。

落盘标签验证如下。fixed IGSTGNN 标签对正确 RegionData 目标的 MAE 为 30.863、24.623 和 30.468，对实际右移目标的 MAE仅为 `8.3e-7`、`8.0e-7` 和 `1.1e-6`。三个区域完全一致。

同一转换器还有独立的事故锚点错绑。H5 每个样本保存了真正的 `event_idx`，转换器却在 `scripts/build_igstgnn_data_from_h5.py:165` 到 `scripts/build_igstgnn_data_from_h5.py:168` 选择 `active_event_idx` 的第一个有效项。活跃事件按开始时间升序，因此第一项通常是两小时活动窗中最旧的事件，而不是本样本的锚定事故。

| 区域 | 转换器 primary 等于真正 anchor | 错绑率 | 被选 primary 在 t 时的中位年龄 |
|---|---:|---:|---:|
| Alameda | 5.45% | 94.55% | 100 分钟 |
| ContraCosta | 20.99% | 79.01% | 85 分钟 |
| Orange | 2.92% | 97.08% | 110 分钟 |

转换器又把 `Event Time` 和 `event_position` 固定为零。IGSTGNN 因此将一个通常已经发生 85 到 110 分钟、且多半不是锚定事故的事件，当成 shifted 历史末端的当前事故。

`scripts/compute_setting_a_breakdown.py` 用 `_t_idx+11` 匹配 FDN。这个操作只修正了交通输入和目标窗口：

```text
adapter x = [t,t+11]      adapter y = [t+12,t+23]
FDN at u=t+11 x = [t,t+11]      FDN at u=t+11 y = [t+12,t+23]
```

它没有修复事故类型、距离、duration、Event Time 与 event position，也没有让 affected mask 和事故类型回到同一个锚定事件。因此 Setting A 的 all-MAE 数值是同交通窗比较，但最多只能解释为晚期交通窗口上，IGSTGNN 收到陈旧且大面积错绑的事故元数据时与 FDN 的比较。它不能支撑“正确对齐的事故感知 IGSTGNN 不如无标签模型”。affected、事故类型和 affected 优势更大的结论必须作废重跑。

受影响的链路为：

```text
scripts/build_igstgnn_data_from_h5.py
outputs/igstgnn_data_from_ours/
outputs/igstgnn_ours_pipeline/
outputs/igstgnn_ours_pipeline_fixed/
outputs/igstgnn_ours_pipeline_fixed/setting_a_breakdown.json
outputs/setting_a_artifacts/
```

`outputs/diagnostics/significance_tests.txt` 中的 Setting A、`incident_type_breakdown` 产物与主报告中的 matched-window 主张也受影响。

另一个旧入口 `scripts/export_to_igstgnn.py` 正确生成历史 `t-11` 到 `t` 和目标 `t+1` 到 `t+12`。它对应 `baselines/IGSTGNN/data/xtraffic/` 与早期 `baselines/IGSTGNN/experiments/` 链路，不存在 11 步错位。该链路仍受最终 duration 等事故特征因果性问题影响，不能据此跳过 v2 重建。

修复后的转换器必须显式保存 `history_start`、`forecast_origin` 和 `anchor_event_idx`，并逐元素断言转换后的 y 与 RegionData 匹配目标完全一致。

## 不是泄露的项目

NDR-Net 在训练期使用真实 y 构造 oracle 门控蒸馏目标，本身属于正常监督学习。`scripts/train_staeformer_deseason.py` 在推理期只使用历史输入生成门控。它的问题来自外围旧缓存、事故标签清洗气候态和统计范围，而不是 oracle 蒸馏本身。

## 当前结果有效性

详细机器可读清单见 [result_validity.csv](result_validity.csv)。

| 实验族 | 当前状态 | 原因 |
|---|---|---|
| XTraffic 事件协议基线与方法 | contaminated | 双向插值与边界统计 |
| XTraffic full-window | contaminated | 双向插值与目标边界重叠 |
| RADS、RADS-O 与 NDR-Net | contaminated | 双向插值、标签清洗气候态与统计边界 |
| DIST-Net 与 ICSF | contaminated | 双向插值、事后事故输入与边界问题 |
| H5 转 IGSTGNN | contaminated | 双向插值、事后 duration 与时间语义错位 |
| v2 最先完成的四项远程结果 | contaminated preliminary | forecast origin 重复计权，active event 跨 split，验证指标与训练实现仍有问题 |
| v2 其余最小判别矩阵 | stopped | v3 全面复核开始后停止 |
| v3 正式缓存 | verified | staging、正式路径、5080 与 5090 各通过一次严格审计 |
| v3 最小判别矩阵 | shards verified, composite pending | 45 项 runner 验收已完成，5080 的 27 项分片与 5090 的 18 项分片均已回传并通过结果服务器复验，但只读 composite index 尚未完成 |
| XTraffic baseline zoo | pending | 等待 v3 最小判别矩阵 |
| METR-LA、PEMS04 与 PEMS08 | not affected by this audit | 不经过 XTraffic H5 缓存 |

## 2026-07-18 原始修复顺序

1. 冻结旧缓存与旧输出，不再用于论文结论。
2. 建立带 schema version 的 v2 缓存。
3. 改为因果缺失处理并保留输入缺失掩码。
4. 建立不读取事故文件的 train-only 气候态。
5. 按完整目标区间和 event group 重新切分。
6. 将可部署事故输入与事后分析标签拆成不同数据集。
7. 修复统计上界与 IGSTGNN 对齐。
8. 先重跑最小判别矩阵，再决定是否恢复 baseline zoo。

最小判别矩阵如下：

- 三地区 STAEformer s0a 与 s0c
- Alameda 与 ContraCosta 的 s0f 与 s0g
- 三地区 ICSF A/B
- 三地区 full-window GWN、FDN 与 STAEformer

## 必须增加的回归测试

1. Prefix invariance。改变未来数据不得改变当前与过去输入。
2. Split disjointness。三个 split 的目标时间集合和 event group 集合两两不相交。
3. Post-train sentinel invariance。修改训练边界后的数据不得改变 scaler 与 baseline。
4. Label-free baseline independence。移除或修改事故文件不得改变 label-free baseline。
5. Deployable incident invariance。相同历史但不同未来 severity 与最终 duration 时，可部署输入必须完全相同。
6. IGSTGNN alignment。转换目标必须与 RegionData 目标逐元素相等。
7. Cache provenance。缓存必须记录 schema、协议、切分、原始文件哈希、代码 Git SHA 与 dirty 状态。

## 输出保留策略

旧结果和 v2 初步结果不删除、不覆盖。v3 实验使用新缓存目录 `outputs/dist_net/region_data_v3_noleak_dedup` 与新输出根目录 `outputs/protocol_v3_noleak_dedup`。summary 必须补齐协议版本、切分边界、normalizer 与 climatology 的拟合范围、cache bundle ID、内容摘要、Git SHA、dirty 状态和 schema version。正式 runner 必须拒绝缓存身份不匹配、代码身份不匹配、dirty worktree 和已存在的旧检查点。
