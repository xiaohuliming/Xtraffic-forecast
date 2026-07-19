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
| `dist_net/region_data_v3_noleak_dedup/` | schema 3 的 v3 XTraffic 缓存 | 已落盘并通过两次严格复验，状态为 `verified` |
| `protocol_v3_noleak_dedup/` | v3 训练结果统一根目录 | 训练尚未启动，不应存在正式 summary |
| `paper_artifacts/` | 旧事故双支论文资产 | Legacy，不代表当前论文方向 |
| `impact_guided_next_stage/` | 大量旧事故增强实验 | Legacy，不再扩展 |
| `remote_runs/` | 本地远程回传包与队列日志 | Git 忽略 |

v3 正式缓存的 bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`，它不是最终训练代码 SHA。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。训练仍未启动。最终训练代码 SHA 等 main 合并后锁定。下一步是传输缓存、执行远端哈希复验，再从最终训练 SHA 的 clean clone 启动队列。

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
