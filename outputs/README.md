# Outputs 目录说明

`outputs/` 同时包含已提交的小型结果摘要和本地生成的大型实验产物。Git 默认忽略该目录的大部分内容，只对白名单中的 Markdown、表格与选定论文资产开放跟踪。

## 当前有效性

2026-07-18 审计确认，所有使用旧 XTraffic 区域 H5 缓存的结果都受到数据泄露或协议污染影响。旧文件保留用于研究史追溯，不应直接进入论文表格。

实验族状态见：

- `docs/audits/data_leakage_2026-07-18.md`
- `docs/audits/result_validity.csv`

## 主要目录

| 路径 | 内容 | 状态 |
|---|---|---|
| `diagnostics/` | 汇总表、噪声带、消融与诊断文本 | 历史证据，逐项检查有效性 |
| `baselines/` | GWN、STAEformer、STID 与 baseline zoo | XTraffic 结果待 v2 重跑 |
| `fourier_dual_net/` | FDN 与相关实验 | XTraffic 结果待 v2 重跑 |
| `rgdn/` | RGDN 与 RADS 系列 | 待 label-free baseline 和 v2 重跑 |
| `dist_net/region_data/` | 旧 XTraffic H5 缓存 | `contaminated`，只读冻结 |
| `dist_net/region_data_v2_noleak/` | v2 XTraffic H5 缓存 | 尚未正式生成 |
| `protocol_v2_noleak/` | v2 训练结果统一根目录 | 尚未重跑 |
| `paper_artifacts/` | 旧事故双支论文资产 | Legacy，不代表当前论文方向 |
| `impact_guided_next_stage/` | 大量旧事故增强实验 | Legacy，不再扩展 |
| `remote_runs/` | 本地远程回传包与队列日志 | Git 忽略 |

## 保留与命名规则

- 不删除或覆盖旧结果。
- 无泄露缓存与结果使用新版本名，例如 `protocol_v2_noleak`。
- 修复前的结果目录应视为只读。
- 大型 `npz`、`pt`、`h5` 和日志不提交。
- 论文表格只能从有效性清单允许的 summary 生成。
- 远程回传包先核对哈希和 summary，再移入对应版本目录。

## 不要执行的操作

- 不要把当前 `dist_net/region_data/` 重新标成 clean。
- 不要用新结果覆盖同名旧 `summary.json`。
- 不要把 `paper_artifacts/` 当成当前论文稿入口。
- 不要在未核对来源时批量删除 `outputs/`。其中包含不可轻易重建的预测数组。
