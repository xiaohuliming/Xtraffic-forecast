# NDR-Net 专用目录

NDR-Net 相关的模型、训练、配置、远程运行脚本、测试、文档和结果，都在这里各存一份。除 `results/` 外均为原文件的逐字节副本，原文件留在原位置不动，历史入口和审计记录引用的仍是原路径。`MANIFEST.csv` 逐个列出每个副本的来源分支、提交、原路径、SHA-256 和大小。

## 目录

| 目录 | 内容 | 来源 |
|---|---|---|
| `results/` | 四组已核验的 NDR-Net 结果，按协议分开，见其中 README | 由原始产物摘取 |
| `core/` | 原 NDR-Net 本体：STAEformer 骨干与 `train_staeformer_deseason.py`，NDR-Net 即其中的 `s0g` 变体 | `codex/fdn-benchmark-v1` |
| `v3_event/` | v3 无泄露事故窗口下 NDR-Net 的训练入口、数据适配、Windows 与 Linux runner、配置和测试，对应 45 行主表 | `codex/fdn-benchmark-v1` |
| `official_table4/` | IGSTGNN Table 4 官方重叠协议下的 NDR 扩展 30 epoch 三种子与 native-100 对照，含 manifest、runner、合并、报告、远程脚本和测试 | `codex/fdn-benchmark-v1` |
| `bist_ndr/` | BiST-NDR 插件配对：模型、runner、报告、配置和测试 | `codex/fdn-benchmark-v1` |
| `research_211461b/` | NDR 研究线完整代码：`ndr_research/` 包、开发矩阵、Graph、full-window、迁移、强对照、多通道候选等训练与评估脚本、Windows 队列和测试 | `bist-ndr-matched` 提交 `211461b`，原名 `codex/bist-ndr-matched`，未合并、未推送 |
| `docs/` | NDR 相关审计、阶段报告、论文互审记录和汇报讲稿 | 工作树，另有 6 份只存在于研究分支 |

## 使用说明

- 副本用于集中查阅和归档，不能直接在本目录内运行。脚本按原仓库根目录解析导入路径，还依赖 `dist_net/`、`fourier_dual_net/rgdn.py`、`scripts/train_baseline_zoo.py` 和 `scripts/reproduction/igstgnn_table4/` 下的通用 factory 等共享模块，这些没有复制。要复现，请在原路径运行。
- `research_211461b/` 的代码只在 `bist-ndr-matched` 分支上，运行要切到该分支的 worktree `/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`。5080 上正在跑的多通道正式运行用的就是这份代码。
- 原文件之后若有修改，这里的副本不会跟着更新。以 `MANIFEST.csv` 的提交和哈希为准，需要时重新生成。
- 研究线的远程运行记录、checkpoint、控制目录和缓存体积太大，没有复制，位于 `outputs/remote_runs/ndr_generalization_20261013/`、`outputs/remote_runs/bist_ndr_pair_retrieved_20261020/` 与 `outputs/protocol_v3_noleak_dedup/ndr_graph_discovery_20261016/`。
- 结果能否用于论文，仍以 `docs/audits/result_validity.csv` 和两份泄露审计为准。
