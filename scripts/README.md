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

以上 XTraffic 训练入口默认读取 `outputs/dist_net/region_data_v3_noleak_dedup`，并将新结果写入 `outputs/protocol_v3_noleak_dedup`。正式入口通过 `dist_net/training_runtime.py` 校验 cache bundle、Git SHA、输出隔离和 summary 契约。v3 正式缓存已经通过本地两次、5080 一次和 5090 一次严格复验，状态为 `verified`。截至 2026-07-22 20:51 新加坡时间，runner 层已完成或重验 38/45 项。27/18 重分片已经批准并完成控制面布防，但仍在等待 task039 边界，尚未实际切换。5080 的旧 24 项已经完成，等待器 PID 2764 正在 `WAIT_BOUNDARY`。5090 的旧队列为 14/21，task039 已完成 epoch 18 并保存新的 best checkpoint，epoch 19 已到 batch 2700/4598。边界 watcher PID 213135 与 transition PID 216008 均在运行。ready marker 尚不存在，task040 在两台机器上均无正式目录。两分片保持独立结果根、`config_hash` 与运行时 provenance，完整矩阵和多种子统计尚未完成。旧结果和最先完成的 4 个 v2 远程结果均不可用于恢复论文结论。

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

`run_*.sh`、`watch_*.sh`、`shutdown_*.sh` 与 `run_zoo_queue_win.py` 是历史机器相关入口。实验室共享连接参数集中在 `config/lab_hosts.env`。v3 的 canonical 任务表为 `protocol_v3_manifest.json`，队列实现为 `protocol_v3_runner.py`。`remote/distributed/protocol_v3_distributed.py` 从 canonical manifest 构建固定 5080 与 5090 分片，保证任务互斥、并集为 45 且直接比较组同机。训练代码 SHA 仍锁定为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。

`remote/distributed/protocol_v3_shard_postflight.py` 在源主机复验各自分片并生成严格导出 allowlist。`remote/distributed/protocol_v3_composite_postflight.py` 在两份分片回传后只读验证 canonical 45 完整性。已批准的 27/18 分片把 5080 目标设为 tasks001-004、007-010、013-016、019-030、040-042，共 27 项，并把 5090 目标设为 tasks005-006、011-012、017-018、031-039、043-045，共 18 项。所有直接比较组继续保持同机。

`remote/linux/protocol_v3_start_5090_shard_after_boundary.sh` 负责等待 task039 的完成边界，阻止 task040 在旧队列启动，并在严格校验后发布 ready marker。旧 5090 finalizer PID 106466 将在边界由 transition 停止。`remote/linux/protocol_v3_finalize_5090_shard.sh` 只在 18 项新分片验收、上传和远端哈希全部通过后关机。`remote/windows/protocol_v3_repartition_5080_27_waiter.cmd` 是 5080 的重分片等待入口。它只在严格验证 ready marker 后启动 27 项 runner 与新 finalizer。旧 5080 finalizer PID 12524 已停止。新远端结果名为 `rtx5080_27` 与 `rtx5090_18`。所有收尾脚本采用失败保持开机策略。

27/18 控制面由 Git 提交 `60405a9`、`e191c27` 与 `30bcb81` 记录。最终 Windows 退出码文件方案已在 5080 冒烟通过，本地全仓测试为 230 passed、1 个既有 warning。结果服务器当前空闲 6,280,982,528 字节。服务器上的 Docker build cache 尚未获得清理授权，因此相关脚本不得清理它。

`cleanup_dead_outputs.sh` 是历史输出清理辅助脚本。执行前必须先使用其 dry-run 模式核对候选，不能把它当成仓库整理的默认步骤。

## 新脚本约定

- 数据缓存改版使用明确的 schema 和协议版本。
- 训练输出不得覆盖旧目录。
- summary 必须记录协议、split、预处理范围、缓存哈希、Git SHA 和 dirty 状态。
- 数据测试优先写入 `tests/`。一次性诊断脚本才放入 `scripts/`。
- 新入口必须在本文件登记状态与用途。
