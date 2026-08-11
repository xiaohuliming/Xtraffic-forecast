# 文档索引

文档按用途分为当前入口、审计、方法说明、历史设计和归档记录。不要仅按文件修改日期判断权威性。

## 当前入口

| 文件 | 用途 |
|---|---|
| `../README.md` | 仓库总览与当前有效性状态 |
| `../CLAUDE.md` | 新会话入口、工作顺序与协作约定 |
| `../XTraffic项目完整汇报.md` | 完整研究历史、当前进度与落盘数字出处 |
| `audits/protocol_v3_review_2026-07-19.md` | v2 冻结原因、v3 修复范围与当前协议状态 |
| `audits/data_leakage_2026-07-18.md` | 原始数据泄露问题与验收基础 |
| `audits/result_validity.csv` | 各实验族的机器可读有效性状态 |
| `audits/igstgnn_table4_4090d_launch_2026-07-29.md` | IGSTGNN Table 4 双机九卡启动、显存恢复、回传和关机状态 |
| `audits/igstgnn_table4_bist_ndr_pair_launch_2026-07-31.md` | BiST 与 BiST-NDR 18 项严格配对协议、探针和 5080 队列状态 |
| `audits/igstgnn_table4_native100_launch_2026-08-01.md` | BiST 与 NDR-Net 原生配方 100 epoch 补充矩阵、探针和 5080 启动状态 |
| `audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md` | 原生配方 100 epoch 矩阵的双机分层、incident、严格回传、composite 和最终表 |
| `audits/protocol_v3_event_4090d8_launch_2026-08-04.md` | v3 event baseline 的八卡 Orange 分片、24 GB 显存 incident 和等效 batch 恢复 |
| `audits/protocol_v3_event_cc_orange_multi_gpu_2026-08-07.md` | 八卡机无卡盘点、staging 不可续跑结论、落盘实测耗时、卡数分析与多卡控制层布防 |
| `audits/protocol_v3_event_extension_eager_relaunch_2026-08-09.md` | 9 项扩展的 HDF5 I/O 根因、eager 等价性门禁、重新标定与远端不可达状态 |
| `audits/protocol_v3_event_convergence_seed42_launch_2026-08-11.md` | 100 epoch 收敛审计的门禁、启动、增量灾备与最新可验证状态 |
| `reports/导师阶段汇报与论文路线_2026-08-11.md` | 面向导师的阶段汇报、当前结论边界、投稿建议和论文补实验清单 |

`XTraffic项目完整汇报.md` 仍是研究历史和数字出处的权威文档，但其中的 XTraffic 数值在 v3 复跑前均属于历史结果。当前是否可用于论文，以 v3 协议复审、数据泄露审计和有效性清单为准。

当前 XTraffic 协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。v3 正式缓存已通过本地两次、5080 一次和 5090 一次严格复验。最小判别矩阵、同机配对、多种子统计和窗口级分析均已完成。无泄漏 42 行论文对比表也已完成并生成 CSV、Markdown、LaTeX、manifest 与 PDF，有效性标签为 `target_disjoint_event_no_leak_code_stratified`。它是 seed 42、最多 30 epoch 的描述性比较，不能直接表述为充分收敛或多种子统计结论。

IGSTGNN Table 4 使用独立协议 `igstgnn_table4_official_v1`，不得与 v3 无泄露结果混合。原 39 项已经完成三源回传和两次完整只读 composite 复验。正式 index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。NDR-Net 独立 9 项扩展也已验证，index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。BiST 与原 NDR-Net 的原生配方 100 epoch 矩阵已完成 18/18，双机 composite index SHA-256 为 `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c`，最终 48 行表位于 `../outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`。BiST 与 BiST-NDR 的统一 30 epoch 配对队列已完成 runner 层 18/18，尚待配对分析。当前本地全仓回归为 637 passed 和 2 个既有 warning。完整身份、哈希和历史 incident 见对应审计报告。

## 当前方法说明

| 文件 | 用途 |
|---|---|
| `RADS_NDR_汇报讲稿.md` | RADS、RADS-O 与 NDR-Net 的层级、机制、无效历史数值和新颖性定位 |

## 历史设计文档

以下文件记录已结束的研究方向。它们用于追溯设计决策，不代表当前实现计划。

| 文件 | 状态 |
|---|---|
| `dist_net_design.md` | 旧 DIST-Net 设计，包含事故特征与双支路线 |
| `impact_guided_start.md` | 旧事故影响引导方向起点 |
| `新架构方向_残差双支笔记.md` | 旧 RGDN 方向简记 |
| `superpowers/specs/` | 已执行或停止的设计规格 |
| `superpowers/plans/` | 已执行或停止的实施计划 |

实验室共享连接配置统一维护在 `config/lab_hosts.env`。历史计划中的旧地址只用于追溯，实际运行先加载这份共享配置。

## 已归档进度文档

`archive/` 只保存已被主报告吸收的旧交接与旧总结。目前包括：

- `项目总结报告_2026-06-10.md`
- `去季节化方向_交接_2026-06-26.md`
- `事故影响感知交通流预测研究方案.md`

新增归档文档必须在文件顶部写明归档日期、替代文档和保留原因。

## 文档维护约定

- 项目状态只在 README、CLAUDE、完整汇报和审计文件中维护。协议状态先看 `audits/protocol_v3_review_2026-07-19.md`。
- 一个数字只认落盘来源，不用估算补齐。
- 历史结果不静默删除。有效性变化通过审计和清单标记。
- 实验室共享地址与密码只在 `config/lab_hosts.env` 维护一份。私钥、个人令牌和个人账号凭据不得写入文档。
- 论文稿引用 XTraffic 数值前，必须检查 `audits/result_validity.csv`。
