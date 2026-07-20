# XTraffic 交通流量预测研究

本仓库研究 XTraffic 2023 上的交通流量预测，重点回答两个问题：

1. 事故标签是否真的能提高预测精度。
2. 只使用历史流量的状态自适应重组机制能否稳定改善强时空骨干。

## 当前状态

> **v2 已冻结。v3 正式缓存已从 cache-build clean commit `29159f626494de52c69b36fe5bba9854f13cb56c` 构建，并通过本地两次、5080 一次和 5090 一次严格复验。最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。tasks001-006 已通过 runner 验收，但都只有 seed42。正式 45 项现采用双机分层矩阵：5080 为 4/24，task007 已在主机内存故障后用新 attempt 恢复训练；5090 为 2/21，正在 task011。两分片独立验收且所有直接比较组保持同机。完整矩阵和多种子统计尚未完成，论文写作继续暂停。**
>
> 2026-07-18 的审计确认旧版 XTraffic 数据管线存在未来信息泄露和切分边界污染。2026-07-19 的全面复审进一步确认 v2 存在重复 forecast origin、active event 跨 split、验证累计和训练契约问题。此前报告的所有 XTraffic 数值与最先完成的 4 个 v2 远程结果均保留为历史记录，不应作为论文结论或最终基准引用。完整证据见 [v3 协议复审](docs/audits/protocol_v3_review_2026-07-19.md) 和 [数据泄露审计](docs/audits/data_leakage_2026-07-18.md)。

审计前，项目已经完成以下研究路线：

- 用 GWN、FDN、STAEformer 和 IGSTGNN 研究事故标签的预测价值。
- 将自适应去季节化机制整理为 RADS，并迁移到 GWN 与 STAEformer。
- 在 RADS 上继续形成 RADS-O 与 NDR-Net。
- 完成三地区、多种子、等参数消融和一批标准基准实验。
- 启动统一管线下的 baseline zoo，但结果尚未完成权威汇总。

这些工作仍有方法设计和工程价值，但 XTraffic 上的数值证据需要按新协议重建。当前 v3 代码实现了因果填充、label-free train-only 气候态、完整目标窗切分、forecast origin 去重、active event split 隔离、可部署事故特征、显式 mask、全局有效观测点指标累计、缓存身份校验、输出隔离和 IGSTGNN 对齐。协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。纳入 `origin/fu` 的合理修复并补齐 Windows 路径测试后，Mac、5080 与 5090 均为 143 项通过，1 个既有 warning。

正式缓存位于 `outputs/dist_net/region_data_v3_noleak_dedup`。它已在本地正式路径和两个远端训练环境完成严格复验，状态均为 `verified`。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。

| 区域 | 正式样本数 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

5080 分片固定为 tasks001-004、007-010、013-016、019-030，共 24 项。5090 分片固定为 tasks005-006、011-012、017-018、031-045，共 21 项。两台机器保留各自结果根、路径相关 `config_hash`、运行时 provenance 与 source-host postflight，最终通过只读 composite index 验证互斥并集为 canonical 45。结果只能称为双机分层矩阵，不能称为 5090 同机矩阵。现有已验收单项不能代替完整多种子统计或最终论文结论。

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

旧缓存、v2 缓存和旧结果没有被覆盖。v3 默认缓存目录为 `outputs/dist_net/region_data_v3_noleak_dedup`，新训练默认写入 `outputs/protocol_v3_noleak_dedup`。三地区正式缓存已经通过本地、5080 与 5090 的严格复验。v3 队列正在独立的新结果目录运行。本地回归测试位于 `tests/data/`、`tests/training/` 和 `tests/remote/`。

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
