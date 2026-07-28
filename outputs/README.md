# Outputs 目录说明

`outputs/` 同时包含已提交的小型结果摘要和本地生成的大型实验产物。Git 默认忽略该目录的大部分内容，只对白名单中的 Markdown、表格与选定论文资产开放跟踪。

## 当前有效性

2026-07-18 审计确认，所有使用旧 XTraffic 区域 H5 缓存的结果都受到数据泄露或协议污染影响。2026-07-19 全面复审又确认 v2 存在重复 forecast origin、active event 跨 split、验证累计和训练契约问题。旧文件与 v2 结果保留用于研究史追溯，不应直接进入论文表格。

实验族状态见：

- [v3 协议复审](../docs/audits/protocol_v3_review_2026-07-19.md)
- [数据泄露审计](../docs/audits/data_leakage_2026-07-18.md)
- [结果有效性清单](../docs/audits/result_validity.csv)
- [中文实验审计与进度状态](impact_guided_next_stage/experiment_audit/status_zh.md)

## 主要目录

| 路径 | 内容 | 状态 |
|---|---|---|
| `diagnostics/` | 汇总表、噪声带、消融与诊断文本 | 历史证据，逐项检查有效性 |
| `baselines/` | GWN、STAEformer、STID 与 baseline zoo | XTraffic 结果待 v3 重跑 |
| `fourier_dual_net/` | FDN 与相关实验 | XTraffic 结果待 v3 重跑 |
| `rgdn/` | RGDN 与 RADS 系列 | 待 label-free baseline 和 v3 重跑 |
| `dist_net/region_data/` | 旧 XTraffic H5 缓存 | `contaminated`，只读冻结 |
| `dist_net/region_data_v2_noleak/` | v2 XTraffic H5 缓存 | `contaminated`，只读冻结，Git 忽略 |
| `protocol_v2_noleak/` | v2 训练结果统一根目录 | 已停止，最先完成的 4 项为 `contaminated preliminary` |
| `dist_net/region_data_v3_noleak_dedup/` | schema 3 的 v3 XTraffic 缓存 | 已通过本地两次、5080 一次和 5090 一次严格复验，状态为 `verified` |
| `protocol_v3_noleak_dedup/` | v3 训练结果统一根目录 | runner 45/45、正式 27/18 分片与只读 composite 均已验证，同机配对、多种子与窗口级显著性分析已完成，见 `protocol_v3_noleak_dedup/analysis/` |
| `reproduction/igstgnn_table4_official_v1/` | IGSTGNN Table 4 官方发布协议复现数据与本地审计 | 正式清单已冻结，5080 的 Alameda HL 与 LSTM 已完成，DCRNN 正在训练 |
| `paper_artifacts/` | 旧事故双支论文资产 | Legacy，不代表当前论文方向 |
| `impact_guided_next_stage/` | 大量旧事故增强实验 | Legacy，不再扩展 |
| `remote_runs/` | 本地远程回传包与队列日志 | Git 忽略 |

v3 正式缓存的 bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。远端传输归档 SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。runner 层已完成或重新验收 45/45。5080 正式分片为 27/27，5090 正式分片为 18/18。只读 composite index 已于 2026-07-27 18:28:27 新加坡时间验证成功。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，`execution_layout` 为 `dual_host_stratified`，index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。控制提交为 `93038ec`。index 共含 45 个任务和 2 个分片，canonical tasks001-045 各出现一次。首次使用 `c48f232` 的无输出尝试已作为 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。最终服务退出码为 0，内存峰值为 2.1G，swap 峰值为 15.1M，没有改写 summary 或 artifact。当前本地全仓回归为 252 passed 和 1 个既有 warning。composite 完成只证明正式双机分层矩阵的身份与完整性，不恢复论文结论。同机配对、多种子统计与窗口级显著性分析已于 2026-07-27 完成，输入为按 composite index 逐份复验的 45 份 summary 与 45 份 test_predictions.npz，产物在 `protocol_v3_noleak_dedup/analysis/`（`matrix_45.csv`、`multiseed_summary.csv`、`paired_seed_level.csv`、`window_pairs.csv`、`analysis_report.md`、`window_pairs_report.md`、`window_pairs.json`）。窗口级结论：s0c 显著优于 s0a（三区域三种子全部 p<0.01），s0g 小幅优于 s0f（多数显著、幅度小），ICSF 注入效应接近零（Alameda 不显著，ContraCosta 与 Orange 窗口级显著但幅度约 0.02-0.03），full_window 排序为 staeformer < fdn < gwn（全部显著）。论文主张是否恢复仍由用户据落盘结果决定。

IGSTGNN Table 4 复现使用作者发布数据与官方顺序切分，独立于 v3 无泄露缓存。正式清单位于 `config/reproduction/igstgnn_table4/matrix.public_material_v2.json`，SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`。该协议保留官方数据中的已知重合，只能声明为按公开材料最大忠实复现。5080 正式结果根是 `D:\t4\results\igstgnn_table4_public_repro_39_v2`。Alameda HL 与 LSTM 已通过五文件验收，Average MAE 分别为 19.649600 与 14.355326，DCRNN 正在训练。

## 保留与命名规则

- 不删除或覆盖旧结果。
- 每次协议修复都使用新缓存和新结果根目录。当前版本为 `protocol_v3_noleak_dedup`，cache schema 为 3。
- 修复前的结果目录应视为只读。
- 大型 `npz`、`pt`、`h5` 和日志不提交。
- 论文表格只能从有效性清单允许的 summary 生成。
- 远程回传包先核对哈希和 summary，再移入对应版本目录。

## 不要执行的操作

- 不要把当前 `dist_net/region_data/` 重新标成 clean。
- 不要把 `region_data_v2_noleak` 或前 4 个 v2 远程结果重新标成 valid。
- 不要用新结果覆盖同名旧 `summary.json`。
- 不要把 `paper_artifacts/` 当成当前论文稿入口。
- 不要在未核对来源时批量删除 `outputs/`。其中包含不可轻易重建的预测数组。
