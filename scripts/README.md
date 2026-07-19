# Scripts 索引

`scripts/` 当前包含数据构建、训练、评估、诊断和远程队列脚本。大量历史脚本仍被其他入口和结果记录引用，本轮不做批量移动，以免破坏导入与硬编码路径。新增工作应按本索引选择入口，不要从文件名猜测当前状态。

## 数据与协议

| 入口 | 状态 | 用途 |
|---|---|---|
| `build_full_county_cache.py` | v3 正式缓存已验证 | 因果填充、forecast origin 去重、label-free train-only baseline、目标窗与 active event 隔离、版本和 provenance |
| `build_impact_labels.py` | 分析专用 | 生成事后 severity 与 affected，不得作为可部署特征 |
| `build_event_rel_feat.py` | v3 正式缓存已验证 | 校验缓存协议、样本身份和事件排序后生成事故与节点关系特征 |
| `audit_protocol_v3_cache.py` | v3 最终验收入口 | 校验三地区正式缓存、关系特征、split、哈希和 bundle identity |
| `build_region_sampling_graph.py` | 可复用 | 区域采样图 |
| `build_igstgnn_data_from_h5.py` | v3 已修复，待正式缓存 | 与 RegionData 使用同一 forecast origin、anchor event 和显式 x/y mask |
| `export_to_igstgnn.py` | 历史入口 | 独立的旧 IGSTGNN 导出管线 |
| `test_full_window_loader.py` | 辅助检查 | 直接断言三个 split 的目标时间集合两两互斥 |

v2 已因重复 forecast origin、active event 跨 split、验证累计和训练契约问题冻结。当前协议状态与 v3 验收条件见 [v3 协议复审](../docs/audits/protocol_v3_review_2026-07-19.md)，原始泄露问题见 [数据泄露审计](../docs/audits/data_leakage_2026-07-18.md)。

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

以上 XTraffic 训练入口默认读取 `outputs/dist_net/region_data_v3_noleak_dedup`，并将新结果写入 `outputs/protocol_v3_noleak_dedup`。正式入口通过 `dist_net/training_runtime.py` 校验 cache bundle、Git SHA、输出隔离和 summary 契约。v3 正式缓存已经通过本地两次及 5080 一次严格复验，状态为 `verified`。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。45 项 v3 队列运行中，尚无完成并验收的新结果。旧结果和最先完成的 4 个 v2 远程结果均不可用于恢复论文结论。

## 事故标签对撞

- `train_gwn_icsf.py`
- `train_staeformer_icsf.py`
- `train_dist_net.py`
- `evaluate_igstgnn_baseline.py`
- `evaluate_igstgnn_affected_breakdown.py`
- `compute_setting_a_breakdown.py`

v3 缓存把可部署事故输入限制为类型、位置、起报时间导出的当前已持续时长，并把 affected 放入显式 opt-in 的 `analysis_labels`。ICSF 两个入口按 `primary_incident_slot` 读取锚定事故。DIST-Net 的 blind 模式不再隐式使用 affected，标签监督必须显式声明。

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

`run_*.sh`、`watch_*.sh`、`shutdown_*.sh` 与 `run_zoo_queue_win.py` 是历史机器相关入口。实验室共享连接参数集中在 `config/lab_hosts.env`。v3 的 Windows 正式入口为 `scripts/remote/windows/protocol_v3_launch.bat`，队列实现为 `protocol_v3_runner.py`，固定 45 项任务表为 `protocol_v3_manifest.json`。runner 要求 clean Git、精确代码 SHA、cache bundle identity 和全新结果目录。`29159f626494de52c69b36fe5bba9854f13cb56c` 是正式缓存构建代码 SHA，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。缓存已通过本地与远端审计，45 项队列已从该 SHA 的 clean clone 启动。

`cleanup_dead_outputs.sh` 是历史输出清理辅助脚本。执行前必须先使用其 dry-run 模式核对候选，不能把它当成仓库整理的默认步骤。

## 新脚本约定

- 数据缓存改版使用明确的 schema 和协议版本。
- 训练输出不得覆盖旧目录。
- summary 必须记录协议、split、预处理范围、缓存哈希、Git SHA 和 dirty 状态。
- 数据测试优先写入 `tests/`。一次性诊断脚本才放入 `scripts/`。
- 新入口必须在本文件登记状态与用途。
