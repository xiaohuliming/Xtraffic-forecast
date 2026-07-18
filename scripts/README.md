# Scripts 索引

`scripts/` 当前包含数据构建、训练、评估、诊断和远程队列脚本。大量历史脚本仍被其他入口和结果记录引用，本轮不做批量移动，以免破坏导入与硬编码路径。新增工作应按本索引选择入口，不要从文件名猜测当前状态。

## 数据与协议

| 入口 | 状态 | 用途 |
|---|---|---|
| `build_full_county_cache.py` | v2 已生成并验收 | 因果填充、label-free train-only baseline、目标窗与 event group 隔离、版本和 provenance |
| `build_impact_labels.py` | 分析专用 | 生成事后 severity 与 affected，不得作为可部署特征 |
| `build_event_rel_feat.py` | v2 已接通 | 校验缓存协议和事件排序后生成事故与节点关系特征 |
| `build_region_sampling_graph.py` | 可复用 | 区域采样图 |
| `build_igstgnn_data_from_h5.py` | v2 已修复 | 与 RegionData 使用同一 forecast origin，并绑定存储的 anchor event |
| `export_to_igstgnn.py` | 历史入口 | 独立的旧 IGSTGNN 导出管线 |
| `test_full_window_loader.py` | v2 检查 | 直接断言三个 split 的目标时间集合两两互斥 |

数据问题与验收条件见 `docs/audits/data_leakage_2026-07-18.md`。

## 当前主训练入口

| 入口 | 模型或任务 |
|---|---|
| `train_graphwavenet.py` | XTraffic GWN |
| `train_fourier_dual_net.py` | XTraffic FDN |
| `train_stid.py` | XTraffic STID |
| `train_staeformer_xtraffic.py` | XTraffic STAEformer |
| `train_rgdn.py` | GWN 骨干上的去季节化与 RADS |
| `train_staeformer_deseason.py` | STAEformer 骨干上的 RADS、RADS-O 与 NDR-Net |
| `train_benchmark.py` | METR-LA、PEMS04 与 PEMS08 标准基准 |
| `train_baseline_zoo.py` | 统一 baseline zoo，旧 XTraffic 缓存结果暂不汇总为正式结论 |

以上 XTraffic 训练入口默认读取 `outputs/dist_net/region_data_v2_noleak`，并将新结果写入 `outputs/protocol_v2_noleak`。v2 缓存已经正式生成并验收，最小判别矩阵正在重跑。旧结果和尚未完成的新结果均不可用于恢复论文结论。

## 事故标签对撞

- `train_gwn_icsf.py`
- `train_staeformer_icsf.py`
- `train_dist_net.py`
- `evaluate_igstgnn_baseline.py`
- `evaluate_igstgnn_affected_breakdown.py`
- `compute_setting_a_breakdown.py`

v2 缓存已将可部署事故输入限制为类型、位置、起报时间导出的当前已持续时长，并把 affected 放入 `analysis_labels`。ICSF 两个入口按 `primary_incident_slot` 读取锚定事故。旧 DIST-Net 的监督逻辑仍需单独审查。

## 诊断与统计

当前仍有参考价值的入口包括：

- `significance_tests.py`
- `incident_type_breakdown.py`
- `diagnose_fdn_failures.py`
- `diagnose_rgdn_injection.py`
- `audit_experiment_integrity.py`
- `validate_forecast_error_against_impact.py`

`audit_experiment_integrity.py` 检查输出文件和实验配对，不检查数据的因果性。两类审计不能互相替代。

## 历史事故增强路线

文件名含以下关键词的脚本主要属于已经停止的事故增强、门控和后处理路线：

```text
impact
dual_branch
gate
veto
selector
sttis
candidate
posthoc
```

它们保留用于复现实验史，不是当前开发入口。对应的大量结果位于 `outputs/impact_guided_next_stage/`。

## 队列与远程运行

`run_*.sh`、`watch_*.sh`、`shutdown_*.sh` 与 `run_zoo_queue_win.py` 是机器相关入口。实验室共享连接参数集中在 `config/lab_hosts.env`。Windows 启动文件位于 `scripts/remote/windows/`。运行完成后，队列日志放到被忽略的 `outputs/remote_runs/`，不要散落在仓库根目录。

`cleanup_dead_outputs.sh` 是历史输出清理辅助脚本。执行前必须先使用其 dry-run 模式核对候选，不能把它当成仓库整理的默认步骤。

## 新脚本约定

- 数据缓存改版使用明确的 schema 和协议版本。
- 训练输出不得覆盖旧目录。
- summary 必须记录协议、split、预处理范围、缓存哈希、Git SHA 和 dirty 状态。
- 数据测试优先写入 `tests/`。一次性诊断脚本才放入 `scripts/`。
- 新入口必须在本文件登记状态与用途。
