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

以上 XTraffic 训练入口默认读取 `outputs/dist_net/region_data_v3_noleak_dedup`，并将新结果写入 `outputs/protocol_v3_noleak_dedup`。正式入口通过 `dist_net/training_runtime.py` 校验 cache bundle、Git SHA、输出隔离和 summary 契约。v3 正式缓存已经通过本地两次、5080 一次和 5090 一次严格复验，状态为 `verified`。runner 层 45/45 已完成或重新验收。5090 分片为 18/18，5080 分片为 27/27，`rtx5090_18` 与 `rtx5080_27` 均已验证回传。两分片保持独立结果根、`config_hash` 与运行时 provenance。只读 composite index 已验证成功。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，`execution_layout` 为 `dual_host_stratified`，index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。同机配对、多种子和窗口级显著性分析已经完成。旧结果和最先完成的 4 个 v2 远程结果均不可用于恢复论文结论。

## IGSTGNN Table 4 复现与报告

| 入口 | 状态 | 用途 |
|---|---|---|
| `reproduction/igstgnn_table4/composite_postflight.py` | 已验证 | 在结果服务器只读合并三个正式来源并验证 canonical 39 |
| `reproduction/igstgnn_table4/composite_report.py` | 已验证 | 从精确哈希绑定的 composite index 生成 39 项 CSV、Markdown、LaTeX 和论文值差异表 |
| `reproduction/igstgnn_table4/ndr_extension_report.py` | 已验证 | 核对 NDR 的 49 文件远端回执、9 个任务 provenance，并生成三种子均值和样本标准差 |
| `reproduction/igstgnn_table4/ndr_extension_merge.py` | 已验证 | 将原 39 项与三个 NDR 地区汇总行合并为最终展示表 |
| `reproduction/igstgnn_table4/bist_ndr_pair_manifest.py` | runner 已完成 | 固定 BiST 与 BiST-NDR 的 18 项三地区三种子配对矩阵 |
| `reproduction/igstgnn_table4/bist_ndr_pair_model.py` | runner 已完成 | 在不修改 BiST pristine checkout 的前提下提供 BiST 对照与 BiST-NDR 双头适配 |
| `reproduction/igstgnn_table4/bist_ndr_pair_runner.py` | runner 已完成 | 执行单项配对训练、全局验证 masked MAE、完整测试尾批与五文件结果写入 |
| `reproduction/igstgnn_table4/bist_ndr_pair_matrix_runner.py` | runner 已完成 | 幂等执行 18 项队列并逐项验收正式 artifact |
| `reproduction/igstgnn_table4/native100_compare_runner.py` | 已完成 | 保留 BiST 与原 NDR-Net 各自原生配方，只将最大 epoch 设为 100 |
| `reproduction/igstgnn_table4/native100_compare_matrix_runner.py` | 已完成 18/18 | 幂等执行 18 项 100 epoch 补充矩阵并逐项验收 |
| `reproduction/igstgnn_table4/native100_host_postflight.py` | 已验证 | 按双机计划复验源主机任务，生成精确导出 allowlist、哈希清单与 marker |
| `reproduction/igstgnn_table4/native100_composite_postflight.py` | 已验证 | 只读复验 5080 与 5090 正式树，生成 `verified_native100_dual_host_18` index |
| `reproduction/igstgnn_table4/native100_compare_report.py` | 已验证 | 绑定 composite index，保留既有 42 行，追加 6 行并生成 CSV、Markdown、LaTeX 和 manifest |
| `remote/windows/igstgnn_table4_native100_5080_cc_only.cmd` | 已完成 12/12 | 在 5080 上只选择 Alameda 与 Contra Costa 的 12 个原生配方 100 epoch 任务 |
| `remote/windows/igstgnn_table4_native100_5080_finalize.ps1` | 已验证 | 复验、可恢复回传、远端精确验哈希和原子发布 5080 的 12 项，成功后保持 5080 开机 |
| `remote/linux/igstgnn_table4_native100_5090_orange.sh` | 已完成 6/6 | 在 5090 上只选择 Orange 的 6 个原生配方 100 epoch 任务 |
| `remote/linux/igstgnn_table4_native100_5090_shutdown_waiter.sh` | 验收组件 | 严格复验 Orange 六项、runner 退出码和残留训练进程，供 finalizer 复用 |
| `remote/linux/igstgnn_table4_native100_5090_finalize.sh` | 已完成 | 替代 5090 完成六项后执行本地验收、结果服务器逐文件复验、原子发布和成功后关机 |
| `remote/windows/igstgnn_table4_native100_5080_orange_seeds1_2.cmd` | 已拒绝，禁用 | 5080 Orange 剩余种子隔离测速入口。live speed gate 未通过，脚本入口硬性返回 90，仅保留审计 |

原 39 项 composite index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。NDR 扩展 index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。最终报告位于 `outputs/reproduction/igstgnn_table4_official_v1/`。该协议保留已知重叠，不能与 v3 无泄露结果混用。

原生配方 100 epoch 补充矩阵已完成 18/18，双机正式树及两次全量 composite 复验均通过。正式 index SHA-256 为 `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c`。最终 48 行表位于 `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`，CSV SHA-256 为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`。

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

`remote/linux/protocol_v3_start_5090_shard_after_boundary.sh` 已在 task039 通过验收后发布 ready marker 并启动 18 项新队列。`remote/linux/protocol_v3_finalize_5090_shard.sh` 在 18 项验收、上传和远端哈希全部通过后完成关机。`remote/windows/protocol_v3_repartition_5080_27_waiter.cmd` 已严格验证 marker 并启动 27 项 runner 与新 finalizer。旧 runner、finalizer、watcher、transition 与 5080 等待器均按设计退出。task040 与 task041 已从原 staging 完成可审计恢复，task042 正常完成。5080 达到 27/27，runner 层达到 45/45。`rtx5090_18` 与 `rtx5080_27` 均已验证回传。

后续控制提交补齐精度契约、staging recovery、Windows 长路径与 finalizer 引号修复。提交 `5f3113b` 降低 composite 复验内存，同时保持 float32 producer 与 2 ULP 契约不变。控制提交 `93038ec` 跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标校验。composite 于 2026-07-27 18:28:27 新加坡时间验证成功，共含 45 个任务和 2 个分片，5080 为 27 项，5090 为 18 项，canonical tasks001-045 各出现一次。首次使用 `c48f232` 的无输出尝试已作为 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。最终服务退出码为 0，内存峰值为 2.1G，swap 峰值为 15.1M，没有改写 summary 或 artifact。同机配对、多种子和窗口级显著性分析已经完成，composite 完成仍不自动恢复论文结论。

### v3 event baseline 多卡主机入口

以下入口用于把 v3 event baseline 剩余分片放到一台多卡 Linux 主机上执行。它们不覆盖 5080 的 Alameda 分片，也不改动任何已验收结果。

| 脚本 | 状态 | 用途 |
|---|---|---|
| `remote/linux/protocol_v3_event_multi_gpu.py` | 已布防，未启动 | 通用多卡多地区控制器。按标定做 LPT 装箱，写分片 manifest、聚合 queue status、runner 退出码和 NAS 计划 |
| `remote/linux/protocol_v3_event_gpu_probe.py` | 已布防，未启动 | 用训练脚本自身的构图与前向反向，实测单 batch 耗时和峰值显存，产出可直接使用的标定文件 |
| `remote/linux/protocol_v3_event_deploy_host.sh` | 已布防，未启动 | 幂等部署：clean checkout 校验、bundle 回传、缓存逐文件复用或上传、只读缓存审计、远端门禁测试、探针、出计划。到此为止，不启动训练 |
| `remote/linux/protocol_v3_event_launch_host.sh` | 已布防，未启动 | 复验远端身份与标定后启动控制器，并按需武装 NAS finalizer |
| `remote/linux/protocol_v3_event_nas_finalize.sh` | 已布防，未启动 | Linux 版回传 finalizer。复用 `protocol_v3_event_host_postflight.py` 契约，远程命令 base64 传输，逐文件 SHA-256、`.part` 续传、精确树比对、本轮摘要绑定和原子改名 |

`protocol_v3_event_host_postflight.py` 为支持多卡主机做了两处向后兼容扩展。计划可声明 `source_layout`，默认 `canonical_contiguous` 与原行为完全一致，多卡主机用 `interleaved`，此时改为要求每个 source 内部按 canonical 顺序、source 按 source_id 排序、并集等于 canonical 选择集。计划还可声明 `task_overrides`，按 task_id 记录微批大小，postflight 重算 `config_hash` 时应用同一覆盖。漏声明或声明错误都会因哈希不符而 fail closed。5080 现役 finalizer 用的是部署在该机上的独立副本，不受本次修改影响。

启动前必须人工复核 `plan.json`，并把其中的 `plan_digest` 通过 `EXPECTED_PLAN_DIGEST` 传给启动脚本。该摘要只覆盖决策字段，不含时间戳，控制器在正式启动时会重算并拒绝任何与复核不一致的计划。

标定文件有两份。`config/protocol_v3/event_baseline_gpu_calibration_v1.json` 来自 4090 D 落盘 train.log 的实测每 epoch 秒数，`stgcn` 与 `hl` 标记为未实测。`config/protocol_v3/event_baseline_gpu_calibration_5090_projected_v1.json` 只是规划用投影，状态为 `projection_not_measured`，正式排产前必须由探针产出的实测标定替换。

装箱结论对卡数选择很敏感。按 4090 D 实测标定，Orange DGCRN 单项占全部剩余工作量的三成八，3 卡与 4 卡墙钟相同。按 32 GB 投影标定，微批惩罚消失后 4 卡几乎零空转。因此卡数必须在探针跑完后再定。

`cleanup_dead_outputs.sh` 是历史输出清理辅助脚本。执行前必须先使用其 dry-run 模式核对候选，不能把它当成仓库整理的默认步骤。

原生配方 100 epoch 补充矩阵曾评估把 Orange seeds 1 与 2 迁移到空闲 5080。5080 真实 CUDA 探针通过，但 292 秒 live speed gate 未完成首个 epoch。候选计划 `distributed_native100_seed_rebalance_v2.json` 状态为 `rejected_after_live_speed_gate`，没有正式任务目录。原 `distributed_native100_5080_5090_v1.json` 继续是当前有效计划。

## 新脚本约定

- 数据缓存改版使用明确的 schema 和协议版本。
- 训练输出不得覆盖旧目录。
- summary 必须记录协议、split、预处理范围、缓存哈希、Git SHA 和 dirty 状态。
- 数据测试优先写入 `tests/`。一次性诊断脚本才放入 `scripts/`。
- 新入口必须在本文件登记状态与用途。
