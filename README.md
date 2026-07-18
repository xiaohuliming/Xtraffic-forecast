# XTraffic 交通流量预测研究

本仓库研究 XTraffic 2023 上的交通流量预测，重点回答两个问题：

1. 事故标签是否真的能提高预测精度。
2. 只使用历史流量的状态自适应重组机制能否稳定改善强时空骨干。

## 当前状态

> **无泄露 v2 管线已经实现并通过本地回归测试，正式缓存与实验尚未重跑，论文写作继续暂停。**
>
> 2026-07-18 的代码与落盘数据审计确认，旧版 XTraffic 数据管线存在未来信息泄露和切分边界污染。此前报告的所有 XTraffic 数值均保留为历史实验记录，但在无泄露管线复跑前，不应作为论文结论或最终基准引用。完整证据见 [数据泄露审计](docs/audits/data_leakage_2026-07-18.md)。

审计前，项目已经完成以下研究路线：

- 用 GWN、FDN、STAEformer 和 IGSTGNN 研究事故标签的预测价值。
- 将自适应去季节化机制整理为 RADS，并迁移到 GWN 与 STAEformer。
- 在 RADS 上继续形成 RADS-O 与 NDR-Net。
- 完成三地区、多种子、等参数消融和一批标准基准实验。
- 启动统一管线下的 baseline zoo，但结果尚未完成权威汇总。

这些工作仍有方法设计和工程价值，但 XTraffic 上的数值证据需要按新协议重建。当前代码已经实现因果填充、label-free train-only 气候态、完整目标窗切分、event group 隔离、可部署事故特征、显式训练边界、IGSTGNN 对齐、cache schema 与 provenance。下一步是生成正式 v2 缓存，并在独立结果目录中重跑最小判别矩阵。

## 已确认的数据问题

| 优先级 | 问题 | 主要影响 |
|---|---|---|
| P0 | 对全年序列使用双向线性插值，历史输入会读取未来观测 | 所有使用旧 XTraffic H5 缓存的模型 |
| P0 | RADS 气候态基线在拟合前使用事故记录排除训练时刻 | 严格的 `label-free` 主张不成立 |
| P0 | `severity` 和最终事故持续时间由事故后的真实流量或完整事件生成 | DIST-Net、ICSF 与事故特征模型不满足因果部署 |
| P1 | 事件样本与全滑窗样本在相邻切分间共享目标时刻 | 验证集和测试集不完全独立 |
| P1 | 一个 IGSTGNN 转换入口右移 11 步，并将 79.01% 到 97.08% 的样本错绑到其他活跃事件 | 对应的 Setting A 与事故分组结论需要重建 |

旧缓存和旧结果没有被覆盖。修复后的协议使用 `outputs/dist_net/region_data_v2_noleak`，新训练默认写入 `outputs/protocol_v2_noleak`。本地数据回归测试位于 `tests/data/`。

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

审计前的完整方法演进、负结果和数值记录仍保留在 [XTraffic 项目完整汇报](XTraffic项目完整汇报.md) 中。阅读时以文首的 2026-07-18 有效性说明为准。

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
2. [数据泄露审计](docs/audits/data_leakage_2026-07-18.md)：已确认问题、影响范围与修复验收条件。
3. [结果有效性清单](docs/audits/result_validity.csv)：实验族当前是否可用于论文。
4. [XTraffic 项目完整汇报](XTraffic项目完整汇报.md)：审计前的完整研究史与全部落盘数字。
5. [RADS 与 NDR 汇报讲稿](docs/RADS_NDR_汇报讲稿.md)：方法结构和诚实的新颖性定位。

## 关键入口

- `scripts/build_full_county_cache.py`：v2 区域缓存构建入口，默认拒绝覆盖既有文件。
- `dist_net/protocol.py`：因果填充、切分、窗口和事故特征的共享协议实现。
- `dist_net/data.py`：事件锚定和全滑窗数据集定义，默认拒绝旧协议缓存。
- `scripts/train_graphwavenet.py`：GWN 训练。
- `scripts/train_fourier_dual_net.py`：FDN 训练。
- `scripts/train_staeformer_xtraffic.py`：STAEformer 的 XTraffic 训练。
- `scripts/train_rgdn.py`：GWN 骨干上的去季节化与 RADS 变体。
- `scripts/train_staeformer_deseason.py`：STAEformer 骨干上的 RADS、RADS-O 与 NDR-Net。
- `scripts/audit_experiment_integrity.py`：现有实验文件完整性检查。它不覆盖本次数据因果性审计。

## 数据与复现边界

- 原始数据、H5 缓存、模型权重和大体积预测数组默认不提交。
- Git 中的 `outputs/` 只保存选定的文本、表格和论文产物。本地目录可能包含大量被忽略的预测文件。
- 当前没有冻结的根级环境锁文件。v2 缓存会记录 schema、协议、切分边界、拟合范围、源文件 SHA-256、Git SHA、分支与 dirty 状态。
- GraphWaveNet 与 STAEformer 的来源版本和许可仍需补充到基线目录。
- 仓库根目录目前没有统一 LICENSE，因此不对本仓库新增代码做超出上游许可的声明。
- 实验室共享远程连接和密码集中明文维护在 `config/lab_hosts.env`，使用方式见 `config/README.md`。

## 结果管理原则

- 旧结果只读保留，并在有效性清单中标记，不原地覆盖。
- 新协议使用独立缓存版本和独立输出标签，例如 `protocol_v2_noleak`。
- 每个新结果必须记录协议版本、切分边界、预处理拟合范围、缓存哈希、Git SHA 和 dirty 状态。
- 所有论文数字必须来自已落盘文件，不用估算值补表。
