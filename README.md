# XTraffic 交通流量预测研究

本仓库研究 XTraffic 2023 上的交通流量预测，重点回答两个问题：

1. 事故标签是否真的能提高预测精度。
2. 只使用历史流量的状态自适应重组机制能否稳定改善强时空骨干。

## 当前状态

> **v2 已冻结。v3 正式缓存已从 cache-build clean commit `29159f626494de52c69b36fe5bba9854f13cb56c` 构建，并通过本地两次、5080 一次和 5090 一次严格复验。最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。runner 层 45/45 已全部完成或重新验收，5090 为 18/18，5080 为 27/27。两个正式分片、只读 composite、同机配对、多种子统计和窗口级显著性分析均已完成。v3 composite index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。这些结果不自动恢复论文结论，最终主张仍由用户根据落盘证据决定。**
>
> 2026-07-18 的审计确认旧版 XTraffic 数据管线存在未来信息泄露和切分边界污染。2026-07-19 的全面复审进一步确认 v2 存在重复 forecast origin、active event 跨 split、验证累计和训练契约问题。此前报告的所有 XTraffic 数值与最先完成的 4 个 v2 远程结果均保留为历史记录，不应作为论文结论或最终基准引用。完整证据见 [v3 协议复审](docs/audits/protocol_v3_review_2026-07-19.md) 和 [数据泄露审计](docs/audits/data_leakage_2026-07-18.md)。

审计前，项目已经完成以下研究路线：

- 用 GWN、FDN、STAEformer 和 IGSTGNN 研究事故标签的预测价值。
- 将自适应去季节化机制整理为 RADS，并迁移到 GWN 与 STAEformer。
- 在 RADS 上继续形成 RADS-O 与 NDR-Net。
- 完成三地区、多种子、等参数消融和一批标准基准实验。
- 启动统一管线下的 baseline zoo，但结果尚未完成权威汇总。

这些工作仍有方法设计和工程价值，但 XTraffic 上的数值证据需要按新协议重建。当前 v3 代码实现了因果填充、label-free train-only 气候态、完整目标窗切分、forecast origin 去重、active event split 隔离、可部署事故特征、显式 mask、全局有效观测点指标累计、缓存身份校验、输出隔离和 IGSTGNN 对齐。协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。训练 checkout 在 Mac、5080 与 5090 均为 143 passed 和 1 个既有 warning，当前本地全仓回归为 480 passed 和 2 个既有 warning。

正式缓存位于 `outputs/dist_net/region_data_v3_noleak_dedup`。它已在本地正式路径和两个远端训练环境完成严格复验，状态均为 `verified`。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。

| 区域 | 正式样本数 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

正式生效的新分片把 5080 固定为 tasks001-004、007-010、013-016、019-030、040-042，共 27 项。5090 固定为 tasks005-006、011-012、017-018、031-039、043-045，共 18 项。两份分片均已完成、回传并验证。两台机器的结果继续按独立结果根、路径相关 `config_hash`、运行时 provenance 与 source-host postflight 管理。只读 composite index 的 `execution_layout` 为 `dual_host_stratified`，共含 45 个任务和 2 个分片，canonical tasks001-045 各出现一次。结果只能称为双机分层矩阵，不能称为 5090 同机矩阵。现有已验收单项不能代替完整多种子统计或最终论文结论。

27/18 控制面、精度契约、staging recovery、Windows 长路径修复与 finalizer 引号修复均已完成。提交 `5f3113b` 将 composite 的 float32 producer 复验改为数值等价的原位数组计算，并在首次指标校验前释放 target 大数组。控制提交 `93038ec` 进一步跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标契约。首次使用 `c48f232` 的无输出尝试已作为 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。最终 composite 服务退出码为 0，内存峰值为 2.1G，swap 峰值为 15.1M，没有改写 summary 或 artifact。同机配对、多种子与窗口级显著性分析产物位于 `outputs/protocol_v3_noleak_dedup/analysis/`。

## IGSTGNN Table 4 复现

IGSTGNN Table 4 使用独立协议 `igstgnn_table4_official_v1`。该协议沿用作者发布数据与官方顺序切分，保留已知预测时间重叠，不能与 v3 无泄露矩阵混合。

原论文表的 13 个方法乘 3 个地区，共 39 项，已经完成训练、三源回传和两次只读 composite 复验。正式 composite index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。NDR-Net 另以独立 9 项矩阵运行三地区与种子 42、1、2，验证 index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。

可直接查看以下已验证产物：

- [39 项复现表](outputs/reproduction/igstgnn_table4_official_v1/table4_verified_v1/table4_reproduced.md)
- [复现值与论文值逐单元对照](outputs/reproduction/igstgnn_table4_official_v1/table4_verified_v1/table4_paper_comparison.csv)
- [NDR-Net 三种子汇总](outputs/reproduction/igstgnn_table4_official_v1/ndr_extension_verified_v1/ndr_extension.md)
- [39 行加 NDR 三行最终表](outputs/reproduction/igstgnn_table4_official_v1/table4_with_ndr_verified_v1/table4_with_ndr_extension.md)

NDR-Net 的 Average MAE 为 Alameda `10.759524 ± 0.040862`、Contra Costa `11.413717 ± 0.024537`、Orange `11.569331 ± 0.031664`。baseline 是单次运行，NDR 是三种子均值和样本标准差，因此目前只支持描述性比较。

为消除单次 BiST、三种子 NDR 和测试尾批处理不一致的问题，仓库新增独立的 18 项 BiST 与 BiST-NDR 严格配对矩阵。两个方法使用相同的三地区、种子 42、1、2、训练控制和完整测试观测集合。5080 队列已启动，当前没有完成任务或性能结论。协议、探针和启动状态见 [配对实验启动记录](docs/audits/igstgnn_table4_bist_ndr_pair_launch_2026-07-31.md)。

## 已确认的数据问题

| 优先级 | 问题 | 主要影响 |
|---|---|---|
| P0 | 对全年序列使用双向线性插值，历史输入会读取未来观测 | 所有使用旧 XTraffic H5 缓存的模型 |
| P0 | RADS 气候态基线在拟合前使用事故记录排除训练时刻 | 严格的 `label-free` 主张不成立 |
| P0 | `severity` 和最终事故持续时间由事故后的真实流量或完整事件生成 | DIST-Net、ICSF 与事故特征模型不满足因果部署 |
| P1 | 事件样本与全滑窗样本在相邻切分间共享目标时刻 | 验证集和测试集不完全独立 |
| P1 | 一个 IGSTGNN 转换入口右移 11 步，并将 79.01% 到 97.08% 的样本错绑到其他活跃事件 | 对应的 Setting A 与事故分组结论需要重建 |
| P0 | v2 同一 forecast origin 可因多个事故重复出现 | 训练与测试指标按重复次数加权 |
| P0 | v2 active event identity 可跨 split | 训练、验证和测试之间仍有事件关联 |
| P0 | 部分验证 MAE 按 batch 或样本均值累计 | 最佳 checkpoint 可能选择错误 |
| P1 | 多个训练入口的 mask、图支持和结果恢复契约不一致 | 对比结果不能可靠配对 |

旧缓存、v2 缓存和旧结果没有被覆盖。v3 默认缓存目录为 `outputs/dist_net/region_data_v3_noleak_dedup`，新训练默认写入 `outputs/protocol_v3_noleak_dedup`。三地区正式缓存已经通过本地、5080 与 5090 的严格复验。v3 runner、两个正式分片、composite 和统计分析已经完成。本地回归测试位于 `tests/data/`、`tests/training/` 和 `tests/remote/`。

## 方法关系

三个名称是逐层包含关系，不是三个并列模型：

```text
RADS
└── RADS-O
    └── NDR-Net
```

- **RADS**：在周期基线与当前持续水平之间按历史状态选择预测锚点。
- **RADS-O**：增加逐预测步门控，并在训练期用未来真值构造 oracle 软目标。
- **NDR-Net**：在 RADS-O 上增加正常态与扰动态双残差头。

审计前的完整方法演进、负结果和数值记录仍保留在 [XTraffic 项目完整汇报](XTraffic项目完整汇报.md) 中。阅读时同时以 [v3 协议复审](docs/audits/protocol_v3_review_2026-07-19.md) 和文首有效性说明为准。

## 仓库导航

| 路径 | 内容 |
|---|---|
| `dist_net/` | XTraffic 数据集接口、采样协议、损失与旧 DIST-Net |
| `fourier_dual_net/` | FDN、去季节化、RADS 与相关探索模型 |
| `baselines/` | GraphWaveNet、STAEformer、IGSTGNN 等基线 |
| `scripts/` | 数据构建、训练、评估和诊断入口，分类见 [scripts/README.md](scripts/README.md) |
| `tests/` | 模型与数据辅助函数测试 |
| `docs/` | 当前文档、审计与历史设计，索引见 [docs/README.md](docs/README.md) |
| `config/` | 实验室共享主机与运行配置 |
| `outputs/` | 已提交摘要、历史产物和本地生成结果，说明见 [outputs/README.md](outputs/README.md) |
| `archive/` | 本地原始 XTraffic 数据，不进入 Git |

## 建议阅读顺序

1. [CLAUDE.md](CLAUDE.md)：会话入口、当前阻塞与协作约定。
2. [v3 协议复审](docs/audits/protocol_v3_review_2026-07-19.md)：v2 冻结原因、v3 修复范围与正式缓存验收状态。
3. [数据泄露审计](docs/audits/data_leakage_2026-07-18.md)：原始泄露问题、影响范围与验收基础。
4. [结果有效性清单](docs/audits/result_validity.csv)：实验族当前是否可用于论文。
5. [XTraffic 项目完整汇报](XTraffic项目完整汇报.md)：完整研究史、当前进度与全部落盘数字出处。
6. [RADS 与 NDR 汇报讲稿](docs/RADS_NDR_汇报讲稿.md)：方法结构和诚实的新颖性定位。

## 关键入口

- `scripts/build_full_county_cache.py`：v3 区域缓存构建入口，按 forecast origin 去重并默认拒绝脏工作区和覆盖既有文件。
- `scripts/build_event_rel_feat.py`：生成与 v3 样本身份绑定的事故与节点关系特征。
- `scripts/audit_protocol_v3_cache.py`：v3 正式缓存最终审计与 bundle manifest 入口。
- `dist_net/protocol.py`：因果填充、切分、窗口和事故特征的共享协议实现。
- `dist_net/data.py`：事件锚定和全滑窗数据集定义，严格校验 schema、协议、身份、mask 与 split。
- `dist_net/training_runtime.py`：正式训练输出、缓存身份与 summary 契约。
- `scripts/remote/windows/protocol_v3_runner.py`：5080 与 5090 共用的 v3 严格串行 runner，固定任务表见同目录 manifest。
- `scripts/train_graphwavenet.py`：GWN 训练。
- `scripts/train_fourier_dual_net.py`：FDN 训练。
- `scripts/train_staeformer_xtraffic.py`：STAEformer 的 XTraffic 训练。
- `scripts/train_rgdn.py`：GWN 骨干上的去季节化与 RADS 变体。
- `scripts/train_staeformer_deseason.py`：STAEformer 骨干上的 RADS、RADS-O 与 NDR-Net。
- `scripts/audit_experiment_integrity.py`：现有实验文件完整性检查。它不覆盖本次数据因果性审计。

## 数据与复现边界

- 原始数据、H5 缓存、模型权重和大体积预测数组默认不提交。
- Git 中的 `outputs/` 只保存选定的文本、表格和论文产物。本地目录可能包含大量被忽略的预测文件。
- 当前没有冻结的根级环境锁文件。v3 缓存记录 schema、协议、切分边界、拟合范围、源文件 SHA-256、Git SHA、分支、dirty 状态和 bundle identity。
- GraphWaveNet 与 STAEformer 的来源版本和许可仍需补充到基线目录。
- 仓库根目录目前没有统一 LICENSE，因此不对本仓库新增代码做超出上游许可的声明。
- 实验室共享远程连接和密码集中明文维护在 `config/lab_hosts.env`，使用方式见 `config/README.md`。

## 结果管理原则

- 旧结果只读保留，并在有效性清单中标记，不原地覆盖。
- 新协议使用独立缓存版本和独立输出标签。当前版本为 `protocol_v3_noleak_dedup`。
- 每个新结果必须记录协议版本、切分边界、预处理拟合范围、缓存哈希、Git SHA 和 dirty 状态。
- 所有论文数字必须来自已落盘文件，不用估算值补表。
