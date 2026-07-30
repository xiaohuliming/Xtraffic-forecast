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

`XTraffic项目完整汇报.md` 仍是研究历史和数字出处的权威文档，但其中的 XTraffic 数值在 v3 复跑前均属于历史结果。当前是否可用于论文，以 v3 协议复审、数据泄露审计和有效性清单为准。

当前 XTraffic 协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。v3 正式缓存已通过本地两次、5080 一次和 5090 一次严格复验。runner 层 45/45 已完成或重新验收，5080 分片为 27/27，5090 分片为 18/18。只读 composite index 已于 2026-07-27 18:28:27 新加坡时间验证成功，矩阵有效性状态为 `verified_host_stratified_45`，index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。同机配对、多种子统计与窗口级显著性分析已经完成。是否恢复论文主张仍由用户根据落盘证据决定。

IGSTGNN Table 4 使用独立协议 `igstgnn_table4_official_v1`，不得与 v3 无泄露结果混合。原 39 项已经完成三源回传和两次完整只读 composite 复验。正式 index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。NDR-Net 独立 9 项扩展也已验证，index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。最终 Markdown、CSV 与 LaTeX 表位于 `../outputs/reproduction/igstgnn_table4_official_v1/`。本地全仓回归为 474 passed 和 2 个既有 warning。完整身份、哈希和历史 incident 见 Table 4 启动报告。

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
