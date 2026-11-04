# NDR-Net 结果汇总

本目录只收录 STAEformer 版原 NDR-Net 的结果，由已核验的原始产物摘取而成。原文件位置不变。四组结果的协议、split 或预算不同，不能互相比较或合并成一张表。

| 文件 | 协议 | 评估集 | 种子与预算 | 原始来源 |
|---|---|---|---|---|
| `01_v3_noleak_event_test_seed42.csv` | `protocol_v3_noleak_dedup` 事故窗口 | test | seed 42，最多 30 epoch | `outputs/protocol_v3_noleak_dedup/event_full45_fdn_20260812/verified_table/table.csv` |
| `02_v3_noleak_dev_validation_seed42.csv` | `protocol_v3_noleak_dedup` 开发矩阵 | validation，仅 Alameda 与 ContraCosta | seed 42，完整 30 epoch | `outputs/remote_runs/ndr_generalization_20261013/stage1_copied/development_analysis.json` |
| `03_official_overlap_30ep_per_seed.csv` 与 `03_official_overlap_30ep_summary.csv` | `igstgnn_table4_official_v1` | test | seeds 42、1、2，30 epoch | `outputs/reproduction/igstgnn_table4_official_v1/ndr_extension_verified_v1/` |
| `04_official_overlap_native100_summary.csv` | `igstgnn_table4_official_v1` | test | seeds 42、1、2，原生配方 100 epoch | `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/table4_with_native100.csv` |

## Average MAE

| 地区 | 01 v3 test | 02 v3 val | 03 官方 30 epoch | 04 官方 100 epoch |
|---|---:|---:|---:|---:|
| Alameda | 10.875715 | 10.745627 | 10.759524 ± 0.040862 | 10.504639 ± 0.056600 |
| ContraCosta | 11.627960 | 11.275205 | 11.413717 ± 0.024537 | 11.299118 ± 0.012621 |
| Orange | 11.639792 | 未运行 | 11.569331 ± 0.031664 | 11.136362 ± 0.030750 |

03 与 04 的 ± 为三种子样本标准差。02 列为 global validation MAE。

## 使用边界

- 01 与 02 属于无泄露 v3 协议，01 可以进入论文主表，02 是开发用 validation 参照，不能当作 test 结果。
- 03 与 04 的有效性标签是 `official_protocol_with_known_overlap`，不能填入 v3 无泄露主表。
- 单种子数字只作描述性比较，不能写成显著性或 SOTA 结论。
- 原始数值以来源文件为准。来源更新时，按上表路径重新摘取。
