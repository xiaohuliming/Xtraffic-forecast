# XTraffic 项目导航与协作约定

最近更新：2026-08-12，Asia/Singapore

## 新会话先读

1. `README.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`
6. `docs/README.md`

`XTraffic项目完整汇报.md` 是研究历史、实验数字和出处的唯一权威长报告。数据和结果能否用于论文，以两份泄露审计和 `result_validity.csv` 为准。本文只保留当前快照和操作入口，不再追加已被长报告承接的逐日流水。

## 不可混用的协议边界

- 旧 XTraffic 缓存与结果为 `contaminated`，只读保留。
- v2 已冻结。最先完成的 4 项只能标记为 `contaminated preliminary`。
- 当前无泄露协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。
- IGSTGNN Table 4 复现使用独立协议 `igstgnn_table4_official_v1`，有效性标签为 `official_protocol_with_known_overlap`。
- Table 4 官方协议结果不得与 v3 无泄露结果混合。
- NDR-Net 扩展、BiST-NDR 配对和 native-100 都是项目扩展，不是 IGSTGNN 原论文方法行。
- 42 行无泄漏表必须按实际执行布局标注。33 项 baseline 与 9 项扩展使用同一物理主机身份，但训练代码 SHA 不同，因此预期 `validity_label` 为 `target_disjoint_event_no_leak_code_stratified`。
- 所有数字必须来自落盘文件。未通过 source-host postflight、目标端逐文件复验和 composite 的结果，不能写成完整矩阵结论。

## 当前有效性快照

| 对象 | 当前状态 | 主要身份或产物 |
|---|---|---|
| v3 正式缓存 | `verified` | bundle digest `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5` |
| v3 最小判别矩阵 | `verified_host_stratified_45` | composite index `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe` |
| v3 统计分析 | `descriptive_analysis_complete` | `outputs/protocol_v3_noleak_dedup/analysis/` |
| IGSTGNN Table 4 公开材料复现 | `verified_official_protocol_with_known_overlap_39` | index `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2` |
| NDR-Net Table 4 扩展 | `verified_same_protocol_project_method_extension_9` | index `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a` |
| BiST 与 BiST-NDR 30 epoch 配对 | `runner_completed_unanalyzed_18_of_18` | 尚未生成配对性能结论 |
| BiST 与 NDR-Net native-100 | `verified_native100_dual_host_18` | index `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c` |
| v3 full-window baseline | `stopped_supplementary_1_of_33` | 只保留 Alameda HL 补充结果 |
| v3 event baseline | `verified_event_full33_single_host_33` | 33/33、0 failed，composite index `2a6ad2eadfd1ecf5af3604bee632eaa339e1aed7add59e915a71f9d422dd767e` |
| v3 event 全方法扩展 | `verified_event_extensions_single_host_9` | 9/9、0 failed，composite index `3a3970c627af6f709100ec9ebeeab215f834e0f1f65fc359423197ec29bf9ed4` |
| v3 event 无泄漏论文表 | `verified_no_leak_comparison_generated` | 42 行，CSV SHA-256 `9817f5086f836f17265d87071271cc3b724b62420a2db2f2f4f5401e2f37651c` |
| v3 event FDN 补充 | `running_0_of_3` | RTX 5080 正在训练 Alameda，正式代码 `b95a57fa7b789852c983d9b07e74a3847d1db14b` |
| v3 seed 42 收敛审计 | `last_verified_1_of_39_remote_unreachable` | LSTM Alameda 已增量保全，GPU 主机当前无法完成 SSH 握手 |

本地全仓回归为 655 passed 和 2 个既有 warning。

## v3 event FDN 补充运行状态

2026-08-12 20:04:48 新加坡时间，FDN learnable-K3 三地区补充矩阵 `protocol_v3_event_fdn_5080_v1` 在同一局域网的 RTX 5080 正式启动。该矩阵只包含 Alameda、ContraCosta 与 Orange 的 seed 42，共 3 项。它使用 `protocol_v3_noleak_dedup` 正式缓存、target-disjoint event split、显式观测 mask、batch 32、最多 30 epoch 和 patience 6。优化器为 Adam，学习率为 `1e-3`，weight decay 为 `1e-4`，调度器为逐 optimizer step 的 CosineAnnealingLR。该设置与现有 42 行 PDF 的固定 30 epoch 比较口径一致。

正式训练代码 SHA 为 `b95a57fa7b789852c983d9b07e74a3847d1db14b`。canonical manifest SHA-256 为 `2fb0760e1d5b9d1b7217a7d8ea61763ed5492ba45b26044148409156d4fb475b`。启动前，5080 worktree 为 clean，v3 缓存严格复验为 verified，Orange 最大地区的 20 个完整训练 batch CUDA 探针通过，没有 OOM。启动后 runner 与 trainer 持续存活，GPU 利用率为 80% 至 95%，显存约 4.9 GB。Alameda 前 5 个 epoch 均约 133 秒，验证 MAE 从 `17.9032` 降到 `14.0397`。当前正式任务状态为 0/3 completed、0 failed。

增量备份侧车和持久 postflight 门禁均已启动。群晖 NAS 当前 SSH 不可达，侧车会每 60 秒重试，训练不受影响。只有 runner 完成 3/3 且退出码为 0 后，postflight 才会逐项重建 `config_hash`、重新验证 prediction artifact、任务状态与五类必需产物。最终程序会从经过验证的 `test_predictions.npz` 按 baseline 相同定义只读计算 H3、H6、H12 与 Average 的 MAE、RMSE 和 MAPE，再把 FDN 三行插入原 42 行表，生成 45 行 CSV、Markdown、LaTeX、manifest 与 A3 横版 PDF。训练结束并完成严格复验前，不得报告 FDN 新性能数字。

## v3 seed 42 收敛审计运行状态

2026-08-11，现有 42 行固定 30 epoch 表保持只读。新增独立矩阵 `protocol_v3_event_convergence_seed42_39_v1`，范围为 13 个可训练方法乘三地区，共 39 项，不含无参数 HL。该矩阵固定使用 v3 无泄漏缓存、target-disjoint event split、显式观测 mask、seed 42 和最多 100 epoch。它只用于检查收敛与排序稳定性，不是最终多种子论文表。

DCRNN 恢复 100 epoch 的 scheduler 里程碑 `[10, 50, 90]`，baseline patience 为 30。BiST 与 IGSTGNN patience 为 20。NDR-Net 不启用 early stopping，但测试前仍恢复最佳验证 checkpoint。每个任务新增必需产物 `training_diagnostics.json`，逐轮记录实际学习率、验证 MAE、最佳轮次、stale 计数和停止原因。

39 项 CUDA 探针全部通过。task calibration SHA-256 为 `a6d1c9015b29b70fce51e84e47f0f1fcc33e8a85db9b4e35eadf3841fa2d359a`。正式训练代码 SHA 为 `0338e5ee93d0cf8ef2e162186310c32130fa31df`，plan digest 为 `f84709ccf5debcdcd6a766497095d3ca0024bd72614beb6a2785f2ad6f11b9b0`。Orange STTN 与 D²STGNN 使用 microbatch 16，Orange IGSTGNN 与 NDR-Net 使用 microbatch 8。

2026-08-11 20:04 新加坡时间，GPU 入口 `connect.weste.seetacloud.com:46241` 能建立 TCP，但在 SSH 鉴权前由远端关闭。当前不能证明主机仍在训练，也不能据此判断是欠费、关机、平台回收或入口异常。最后可验证进度为 1/39 completed。已完成任务 `v3c001_event_lstm_alameda_seed42` 在 Hostinger 隐藏增量灾备树通过逐文件校验，状态为 `verified_completed_task_backup`，files digest 为 `6bd8c7313b453745bd7ec0a40ad82df5fb130a17484241364b95e2ad0b17b4fa`。该任务运行到 100 epoch，最佳验证 checkpoint 位于 epoch 99，直接证明 30 epoch 对至少这个任务会截断仍在改善的训练过程。

启动时 controller、增量灾备侧车与 finalizer PID 分别为 `41801`、`43230` 与 `43321`。远端不可达后这些 PID 不再视为存活证据。增量灾备只保护已完成并通过 runner 验收的任务。当前 checkpoint 不含 optimizer、scheduler、下一 epoch 游标和随机数状态，因此中断中的单项不能严格续跑，恢复后必须先审计 queue、进程、正式树和 staging，再决定保留或从 epoch 0 新建 attempt。现有 42 行表保持只读。完整记录见 `docs/audits/protocol_v3_event_convergence_seed42_launch_2026-08-11.md`。

## v3 event 无泄漏 42 行表最终状态

2026-08-10 14:51 新加坡时间，11 个 baseline 乘三地区的 33 项与 BiST、IGSTGNN、NDR-Net 乘三地区的 9 项均完成训练和 runner 验收，失败数均为 0。两棵结果树分别通过 source-host postflight、Hostinger 逐文件 SHA-256、隐藏 incoming 精确树校验、原子提升和只读 composite。最终合表包含 14 方法乘 3 地区，共 42 行。

- baseline matrix：`protocol_v3_event_full33_5090x4_v1`
- baseline 训练代码：`d23a991a570fd1c59e3d2ac8315a5e4063e8097f`
- baseline composite index SHA-256：`2a6ad2eadfd1ecf5af3604bee632eaa339e1aed7add59e915a71f9d422dd767e`
- extension matrix：`protocol_v3_event_extensions_eager_5090x4_v2`
- extension 训练代码：`2754b9cdfa5ccf3baca269e83c7379ff506a0e26`
- extension composite index SHA-256：`3a3970c627af6f709100ec9ebeeab215f834e0f1f65fc359423197ec29bf9ed4`
- 最终有效性标签：`target_disjoint_event_no_leak_code_stratified`
- 最终输出目录：`outputs/protocol_v3_noleak_dedup/event_full42_20260810/verified_table/`
- CSV SHA-256：`9817f5086f836f17265d87071271cc3b724b62420a2db2f2f4f5401e2f37651c`
- Markdown SHA-256：`fa5ae4214c93791b7f9f74f3c2a3850959887426c0d769b0dbfb77994c479cf1`
- LaTeX SHA-256：`c28d06ecb3b64483d2f9160304848d04de7ad2905b94445df3b3680689a7f0e7`
- manifest SHA-256：`03796d1c63fcc571e80fc0bda794b6ee146090af3ef1d256e35bdd5bd02bb5ea`
- PDF SHA-256：`f2257394909be53b809bd8d09209f5524c1b1813c1ce09652bc7b0b07d6864e8`

全部 42 行使用 seed 42、最多 30 epoch、相同 target-disjoint event split、显式观测 mask 和相同指标口径。两个来源位于同一物理主机，但训练代码 SHA 不同，因此只能写成 code-stratified 单种子描述性比较，不能写成同一代码 checkout 的同质矩阵。官方重叠协议 Table 4 的旧数字仍不得混入这张表。

本轮发布层修复提交为 `7fff5baf43fa37d850fc07118a91d9b1f930b2f9` 与 `d98907e9a0513fa7abb8170a534ec194893ace4b`。前者从绑定的 `test_predictions.npz` 只读计算 GraphWaveNet 的逐 horizon RMSE 和 MAPE，后者允许 runner 已重新验证的 `skipped_valid` 终态进入 postflight，同时继续强制任务身份、`config_hash` 和全部 artifact 复验。两项修复均未改写训练 summary 或 prediction artifact。

## v3 正式缓存与训练身份

- 缓存构建代码：`29159f626494de52c69b36fe5bba9854f13cb56c`
- 最小判别矩阵训练代码：`3a4952837aaf6fbda1403d50b610b1bf40c6098a`
- cache bundle ID：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`
- cache bundle digest：`854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`
- 缓存目录：`outputs/dist_net/region_data_v3_noleak_dedup`
- 结果根：`outputs/protocol_v3_noleak_dedup`

| 地区 | 总样本 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

v3 缓存已通过本地两次、5080 一次和 5090 一次严格复验。缓存身份与某次训练代码身份必须分开记录。

## v3 event baseline 最终状态

主矩阵为 11 种方法乘 3 地区，共 33 项。当前固定 seed 42、上限 30 epoch、patience 6、batch 32、target-disjoint 70/15/15 切分，使用 v3 去重事件样本。

- canonical manifest：`config/protocol_v3/table4_event_baselines_phase1_v1.json`
- manifest SHA-256：`f9cdc1afa5f9f126a6a8094d4d4fc38bc565cf3de277dc4a7742f85a0d2f92ef`
- 正式单主机矩阵：33/33 completed，0 failed
- source-host postflight：33 tasks，202 files，`7839685515` bytes
- composite 状态：`verified_event_full33_single_host_33`
- composite index SHA-256：`2a6ad2eadfd1ecf5af3604bee632eaa339e1aed7add59e915a71f9d422dd767e`
- 5080 的 10 项和旧 Orange 的 4 项只读保留，不进入当前矩阵
- baseline 与正式 9 项扩展已共同形成经过验证的 42 行无泄漏表

## v3 event 扩展历史运行与 eager 重启准备

2026-08-08 22:16 新加坡时间，matrix `protocol_v3_event_extensions_5090x4_v1` 在主机 `autodl-container-ae8b44827a-75976e27` 启动。训练代码为 `b9e7de62f81bd4dd9a047937cfabc30c7b75d5d3`，计划 digest 为 `b3f029053136a559657dca768836da6673133b2a47d0c57c1304ffc5dd54f75e`。启动复验为 running、0/9 completed、0 failed，四卡分别执行 Orange NDR-Net、Contra Costa IGSTGNN、Alameda BiST 与 Alameda IGSTGNN。

九项真实 CUDA 探针全部通过。实测 LPT 四个分片预计为 23.06、22.82、20.29 与 22.23 小时。Orange IGSTGNN 原生整批在 32 GB 上 OOM，因此使用 microbatch 8。该设置保留有效 batch 24，并写入计划、trainer 命令、invocation config 和 postflight 契约。Orange NDR-Net 同样使用 microbatch 8。其余方法保留原生 batch。

controller PID 为 `13235`，finalizer PID 为 `14224`。finalizer 只在 9/9、runner 零退出、无残留 trainer、source-host postflight、Hostinger 逐文件复验、原子提升与 post-publish 全部通过后关机。任何失败都保持主机开机。Hostinger checkout 已切换到同一提交 `b9e7de6`，旧 `16e90ba` checkout 只读保留。

2026-08-09 复审确认旧探针遗漏逐样本 HDF5 数据构建时间，且旧 LPT 分片实测总时长约为 31.2、43.7、23.4 和 44.7 小时。Orange IGSTGNN 的 20 batch 对照中，lazy 为 28.5172 秒，eager 为 0.270774 秒，batch stream SHA-256 完全相同。提交 `c78b87a` 已把正式扩展固定为 eager traffic，增加三区域 train shuffle、val、test 字节级等价性门禁，并使用 batch 构建加完整优化步重新标定 LPT。loading audit SHA-256 会绑定进 plan、回传计划与 source-host postflight。

旧扩展最后可确认 3/9 completed、0 failed，baseline 最后可确认 6/33、0 failed。2026-08-09 12:10 时两个 AutoDL 入口都表现为 TCP 可达后在 SSH 鉴权前关闭，因此不能确认训练仍在运行。没有执行远端停止、删除、上传或关机操作。恢复细节见 `docs/audits/protocol_v3_event_extension_eager_relaunch_2026-08-09.md`。

### 5080 运行快照

2026-08-06 15:28 新加坡时间，恢复队列为 1/13 completed，正在 `v3e011_event_d2stgnn_alameda_seed42`。task011 已到 epoch 6、batch 500/1001。GPU 持续满载，最近显存约 15.9 GB，温度约 45 摄氏度。该任务每个 epoch 约 15,943 秒，是当前瓶颈。

5080 的 22 项声明结果分布在三个受控结果根。前两个 queue 因已修复缺陷按设计遇错停止，不是当前再次失败。

| 任务 | 结果根 | 代码 SHA |
|---|---|---|
| tasks001-005 | `D:\t4\v3_event_phase1_5080_v2\results` | `c16dc7a8208f834984e76a40a7b6f47f523c210b` |
| tasks006-009 | `D:\t4\v3_event_gwnfix_v1\results` | `e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae` |
| tasks010-022 | `D:\t4\v3_event_dgfix_v1\results` | `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2` |

task010 Alameda DGCRN 的 H3、H6、H12 和 Average MAE 为 `11.284030`、`12.563134`、`14.453520` 和 `12.577300`。summary SHA-256 为 `49dcb9734a0f74e7aed6613045d087c2ab27dd7d1752bf05facff5484f113aec`。

### Orange 主机与灾备

8 卡 RTX 4090 D 主机的端口 `11218` 当前 TCP 可达后由远端立即关闭 SSH，无法区分余额不足、实例停止或平台回收。不得写成已确认的具体原因。未验收 staging 保留在原数据盘，恢复 GPU 后需逐项检查 checkpoint 恢复契约。

Orange 4 项灾备已在群晖 NAS 通过 25/25 文件复验和原子改名。正式目录为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/orange_4090d8_partial_4_v1`，共 26 个 regular files 和 `1652693542` bytes。`files.sha256` 的 SHA-256 为 `6297b0403746faea2b9c8da0c174952db670cfbd2bdb8f20bea28289118c118a`。该树只是 `verified_partial_disaster_backup_4`，不是正式发布。

## 5080 训练后自动 NAS 回传

持久计划任务 `XTraffic_ProtocolV3_Event_NAS_Finalize_v1` 当前为 Running，每 60 秒检查一次。它只在以下条件全部满足后上传：

1. tasks010-022 恢复队列 13/13 completed。
2. runner 退出码为 0。
3. 没有 runner、baseline trainer 或 GraphWaveNet trainer 残留。
4. tasks001-022 的 `config_hash`、summary、prediction artifact、必需产物和 completed state 逐项复验通过。
5. NAS 容量、逐文件 SHA-256、精确 allowlist 和目录树全部通过。

任何失败都 fail closed，保留源结果并保持 5080 开机。最终 NAS 目标为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts/event_rtx5080_alameda_contra`。当前正式目录和 incoming 都不存在，这与等待训练完成的状态一致。

- 固定计划：`config/protocol_v3/event_baseline_5080_nas_backup_v1.json`
- 计划 SHA-256：`738bc46122f9af6466c1b605013b57c2cc05874aadcda6682cdf08fb4ed964c8`
- postflight SHA-256：`b8605c1725fe9de5ffc74d19815f460b19be812dd495f9814fe41a92eb5b1bf0`
- finalizer SHA-256：`7295346daf7eb972ead64db27706f4234157ae783e68c69421097e6787349c70`
- 5080 专用 NAS 公钥指纹：`SHA256:Nx5BbjtFe9eRbSJh4hV/YN+reR2pw4hZuFLa9PLKMGc`

私钥只存在 5080 的 `D:\t4\v3_event_nas_control\id_ed25519`，不进入 Git。postflight 核心是跨平台 Python，未来 5090 可复用同一契约，但必须使用独立 host ID、分片计划、NAS 目录、SSH 密钥和 Linux 持久包装。

## Table 4 官方协议复现与扩展

- 39 项公开材料最大忠实复现已完成。清单 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`，index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。
- 复现与论文报告值的 Average MAE 绝对差均值为 `2.450150`。该差异必须保留，不得写成作者数值的完全复现。
- NDR-Net 扩展 9/9 已复验。三地区 Average MAE 的三种子均值与样本标准差为 Alameda `10.759524 ± 0.040862`、Contra Costa `11.413717 ± 0.024537`、Orange `11.569331 ± 0.031664`。
- native-100 双机分层矩阵 18/18 已完成两次只读 composite。最终 48 行表位于 `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`，CSV SHA-256 为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`。
- BiST 与 BiST-NDR 的统一 30 epoch 配对队列已 18/18 runner completed，但尚未完成配对统计和性能结论。

完整数字、每份 artifact 哈希、主机分片和 incident 请查看 `XTraffic项目完整汇报.md` 与 `docs/audits/igstgnn_table4_*.md`。

## v3 event 全 33 项在 4 卡 5090 上运行中

2026-08-08 08:56 UTC 启动。完整记录见 `docs/audits/protocol_v3_event_cc_orange_multi_gpu_2026-08-07.md`。

代码 SHA `d23a991a570fd1c59e3d2ac8315a5e4063e8097f`，plan_digest `23ed8507a1a301daef27bb613bc88d2070ec7052858de7d219e9c845c4d2cb82`。三个地区全部同机，最终表不再是主机分层矩阵。5080 上已验收的 10 项 Alameda 与 4090 D 上的 4 项 Orange 保持只读，不进入本矩阵。原 18.7 小时是基于探针标定的排产估计，当前正式四路并发的首轮 epoch 慢于标定，完成时间须按后续实测更新。

实测显示只有 Orange 的 D2STGNN 与 STTN 在 32 GB 上放不下 batch 32，两者使用 microbatch 16 并独占 GPU3。全程 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，该设置已写入 plan 并参与 plan_digest。

回传路径已解决。Hostinger `31.97.147.71` 当前约有 49.6 GB 可用空间，已启用 4 GB swap，xray 保持 active。5090 上的 finalizer PID 为 `17206`，正在等待 33/33。成功后会执行 source-host postflight、隐藏 incoming 上传、逐文件 SHA-256、原子提升、远端只读 composite 和论文列结构表格生成，然后自动关闭 5090。任一环节失败都保持主机开机。

远端 composite 在 systemd 中运行，`MemoryMax` 为 3 GB，`MemorySwapMax` 为 4 GB。控制提交为 `314d45714b61465bce57e8a33c824ce6c47891fd`。composite、表格生成器和 post-publish 包装的 SHA-256 分别为 `2da92291801e516f18ac335fd9a330e75561ce77b16e7897dbc47b99789e0d13`、`890006ec3f1132eeb1aa87cd42d4d668a0160b16c68de069d7304a5230a09993` 和 `28585a7d269218083e2ef83e440d5acf7cc7f0c03498c9d780ce3dd3400a4183`。

18:55 新加坡时间复验仍为 0/33 completed、0 failed。Orange DGCRN、DCRNN、DSTAGNN 与 D2STGNN 分别完成 epoch 3、4、6 与 3。finalizer PID `17206` 持续正常等待。

## v3 无泄漏全方法 42 行表已完成代码准备

最终表的固定范围为 11 个已在运行的 baseline，加 BiST、IGSTGNN 和 NDR-Net，乘三地区，共 42 行。全部使用 `protocol_v3_noleak_dedup`、cache schema 3、seed 42、30 epoch 上限、target-disjoint event split、显式观测 mask 和全部有效观测点累计的 validation masked MAE。各模型保留公开配方的 batch、optimizer 和 scheduler，表中会明确记录这些差异。

新增 canonical manifest 为 `config/protocol_v3/table4_event_extensions_v1.json`，精确包含 9 项。数据适配器不物化 object-pickle split，不把 `affected_mask` 放入任何训练或验证 batch。NDR-Net 的 normalization 和周内时段中位数只在唯一 raw train timestamps 上拟合，并使用 v3 显式 `flow_mask`。IGSTGNN 静态支持文件仅在 `sensors.csv order == region_idx` 且 `station_id == node_order[region_idx]` 逐项完全一致后加载。

新增部署、真实 CUDA 探针、LPT 多卡控制、source-host postflight、严格回传、发布后 composite 与 42 行合表器。两个发布树必须分别验证，且协议、cache bundle ID 和 digest 全部相同，才允许输出 CSV、Markdown、LaTeX 和可追溯 manifest。正式 9 项已于 2026-08-09 启动第一个 shard，任务未完成前不得填入预测数字。

合表器还会从两个 composite index 分别读取 `matrix_id`、`host_id`、`execution_layout` 与 `code_sha`，写入 manifest 的 `source_indexes`，任一字段缺失或 `code_sha` 非 40 位十六进制都拒绝出表。顶层 `execution_layout` 与 `validity_label` 按实测布局取值，同主机同 SHA 为 `single_host`，主机不同为 `host_stratified`，同主机不同 SHA 为 `code_stratified`。`table.md` 抬头同时写明两个来源的主机与代码 SHA，因此论文表本身携带该注脚。runtime provenance 尚未采集，postflight 与 composite 都没有该字段。

实现提交 `16e90baab74bb767a4f8d7a46f45eef6d87af44d` 已推送到 `main`。Hostinger 的 `/opt/xtraffic/v3_event_repo` 已原子切换到该提交，旧 `314d457` checkout 保留为回滚。`/opt/xtraffic/bin/event-extension-post-publish` 已安装，SHA-256 为 `80f05af5b0bb87a221e703479fc034f3b717acf07fff5748b583a1a518388edc`。切换后 xray 仍为 active，swap 仍为 4 GB，根分区约有 44.1 GB 可用。临时 bundle 已清理。

## 当前实时运行，2026-08-09 14:08 新加坡时间

`36571` 对应的单台 4 卡 RTX 5090 已恢复 SSH 和 GPU。baseline 断电前有 6/33 项完成、0 项失败。Orange DGCRN、Orange AGCRN 与 Alameda D²STGNN 三个未完成 attempt 没有 optimizer resume 证据，已从 epoch 0 建立 `attempt_0002` 重跑。恢复前 incident 的 queue、formal tree 与 staging tree 清单 SHA-256 分别为 `10ed910c7e33847fa32931153b651c6c612cd99b7c0e06658dacc99a66663deb`、`acaddda23479ba74af3b38d26d8fc139d081c06c9de4f529207513c24e476070` 与 `b385567c5410a5c0a838bbe45ed9462255d7018ed888474fe57f439dbbfc9dfd`。14:08 的 baseline aggregate 仍为 running、6/33、0 failed。三个重跑任务已分别完成 epoch 1、4 与 4。

旧扩展四卡机已欠费无卡，不再进入当前计划。扩展代码在同机隔离目录 `/root/autodl-tmp/v3_event_extensions_eager_v2` 部署，训练代码 SHA 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。三地区 train shuffle、val 和 test 数据流的 eager 与 lazy 字节级对照全部为 `exact_match`，audit SHA-256 为 `083f9554fd305beed8ccea91385b59230479dd6edd965974959b23cdd9981f6e`。9 项真实 CUDA 探针全部为 `PROBE_OK`，probes 与 calibration 的 SHA-256 为 `4004d54e282e3b13fb2d12821e1c7c84a8180c9ede9b435b18257441fb59b4a9` 与 `00f89733654d306f5accc370ab8063fafe5f106aa0d4df7e2bf5d4b36436b8fa`。

正式扩展矩阵为 `protocol_v3_event_extensions_eager_5090x4_v2`，host ID 与 baseline 统一为 `event_rtx5090x4_full33`，plan digest 为 `2b8bc204115a32bf40d244a94938eb5db27ea229bbed1427542b74c200b4ef3a`。四个 shard 的探针保守预计为 41.86、34.52、39.49 和 37.85 小时。空闲物理 GPU3 已正式启动最长的 `v3x009_event_ndr_net_orange_seed42`，使用 eager 加载与 microbatch 8。14:08 已完成 epoch 2，validation MAE 从 `11.80291315` 降到 `11.59224552`，实际速度快于探针外推。提交 `e21baf2` 的动态接力器会在任一物理 GPU 可用时立即领取当前最长的未启动正式 shard。relay 与 finalize waiter 进程存活，控制根没有异常 exitcode。relay 脚本 SHA-256 为 `a94db88f47aface7c6d9d85958eaf5b4084c84ece8d8bb9197c33b821627eab9`。旧 host ID 修正 incident 和动态升级 incident 清单 SHA-256 为 `f209d12b952cacb39a95b765b120677b06c44e1e86a9e876bdedfe57e4778cf2` 与 `1502f5fe42a3cd1e6d15c30b51df7f84ebb37ffe131a02b50ab876d239242495`，训练进程全程未停止。

baseline finalizer 已重新武装为只发布不关机。扩展 finalizer 只在 baseline 发布成功和 9 项正式 aggregate 成功后启动。Hostinger 为 xray active，4 GB swap 空闲，根分区可用 `45030260736` bytes。只有 33 项发布、9 项发布、两份只读 composite 与 42 行合表全部通过后才自动关机。任一门禁失败都保持开机。动态接力和收尾控制提交为 `e21baf2` 与 `ad70d5a`，已推送到 `main`。

## 增量灾备与断点边界，2026-08-09 15:03 新加坡时间

当前四张 RTX 5090 均在同一物理主机工作。baseline 占用物理 GPU0 至 GPU2，扩展 Orange NDR-Net 占用物理 GPU3。后续任一 baseline 分片释放 GPU 后，动态接力器会在同机启动剩余扩展 shard。

现有 baseline 与扩展 trainer 的 `ckpt_best.pt` 只包含最佳模型权重及少量最佳指标信息。它们不包含 optimizer、scheduler、下一 epoch 游标、early-stopping 状态与随机数状态，因此不能作为严格训练断点。主机中断后，runner 会保留并跳过已完成且复验通过的正式任务，但当前正在训练的任务必须从 epoch 0 建立新 attempt。若 AutoDL 数据盘也被回收，本机已完成任务同样会丢失。

新增 `scripts/remote/linux/protocol_v3_event_incremental_backup.py` 作为独立灾备侧车。它只扫描 runner 已标记 completed、已原子提升且重新通过 `validate_result` 的任务。每个任务完成后立即用低 CPU 与 idle I/O 优先级逐文件上传到 Hostinger 的隐藏 `.incremental` 灾备树，远端逐文件校验 SHA-256 后再写任务完成 marker。它不读取或上传 staging checkpoint，不修改两个训练 checkout，不改变训练代码 SHA、plan digest、config hash 或最终发布门。

baseline 与扩展侧车 PID 为 `16661` 与 `16662`。脚本 SHA-256 为 `e797f2a218f7f32d0b3a6c2dc6d454f92af96132f9e1d34c541863b906e0f9aa`。首个任务 `v3e010_event_dgcrn_alameda_seed42` 已在 Hostinger 完成 6 个文件和 `168730616` bytes 的逐文件回传及任务 marker 验证。其余 5 个既有 completed baseline 任务正在补传。扩展当前尚无 completed 任务，侧车每 60 秒等待。隐藏灾备树不是正式发布，最终仍需原 finalizer 的完整 source-host postflight、精确 allowlist、原子提升和 composite。

## Orange IGSTGNN 关键路径优先接力，2026-08-09 20:26 新加坡时间

Orange NDR-Net 的实际运行显著快于单个完整优化步探针外推。旧接力顺序会先启动探针预计更长的 gpu02 和 gpu03，让实际关键路径 Orange IGSTGNN 最后启动。提交 `0648b76` 把剩余正式 shard 顺序改为显式且严格校验的 `gpu01,gpu02,gpu03`。顺序缺项、重复或包含非法 shard 时接力器会 fail closed。该改动只决定空闲物理 GPU 领取哪个既有正式 shard，不修改 plan、manifest、训练 checkout、config hash 或任务身份。

远端切换前只暂停 relay 和 finalize waiter，四个训练进程均未停止。切换 incident 清单 SHA-256 为 `c3261aa55b14d92bc0b70f16b44fde2433c1dc2635d2e56669c70c5a4c6b22e5`。新 relay 脚本 SHA-256 为 `97633145e9605678d92678ebbb03ea2b7d892072c0bb02fe2936c435df0252f3`，plan 文件 SHA-256 仍为 `5a43e9224572d3413e8ca6753691a8923bc92c49617871f14eafc23d000302d4`。relay 与 finalize waiter 的新 PID 为 `50215` 和 `50364`。

Orange NDR-Net 于 20:25:45 完成五文件验收，正式 gpu00 queue 为 completed、1/1、退出码 0。summary SHA-256 为 `598d5e2838433d1ad8b3ddb503b554ee0f5b10be81900fcedb1b1c0e621815f0`。20:25:54，接力器把物理 GPU3 交给正式 gpu01，启动 `v3x008_event_igstgnn_orange_seed42`。启动复验为 running、0/1，CUDA 利用率 95%，显存使用 16505 MiB。同期 baseline 为 10/33、0 failed，扩展为 1/9、0 failed。本地全仓回归为 613 passed 和 2 个既有 warning。

## 历史：v3 event 剩余分片的多卡主机计划

2026-08-07 布防记录。

八卡 4090 D 实例已在无卡模式恢复，数据盘完好，Orange 4 项正式结果与 6 项 staging 都在。但代码复核确认 `train_baseline_zoo.py` 与 event runner 都没有 checkpoint 续跑路径，那 6 项 staging 必须从 epoch 0 重跑。

计划把 Contra Costa 全部 11 项与 Orange 全部 11 项共 22 项放到同一台新多卡主机。Orange 重跑 11 项而不是 7 项，因为原 4 项从未做过 source-host postflight，混用会让 Orange 比较组跨主机。Alameda 留在 5080。

关键实测：Orange DGCRN 在 24 GB 卡上每 epoch 20,587 秒，单项占剩余工作量的 38%，是唯一的关键路径。按实测标定 3 卡与 4 卡墙钟相同；按 32 GB 投影标定 4 卡几乎零空转。卡数必须等探针实测后再定，投影标定标记为 `projection_not_measured`，不得当证据引用。

入口顺序为 `protocol_v3_event_deploy_host.sh` 到探针到 `protocol_v3_event_multi_gpu.py --plan-only` 到人工复核 plan 到 `protocol_v3_event_launch_host.sh`。回传复用已审计的 `protocol_v3_event_host_postflight.py` 契约，Linux finalizer 为 `protocol_v3_event_nas_finalize.sh`。

## 下一步顺序

1. 持续监控 baseline 的 6/33 恢复队列和 Orange NDR-Net 正式 shard。
2. 确认每个 baseline shard 完成时，接力器只在对应物理 GPU 无残留训练进程后启动扩展 shard。
3. 等待 9/9 后运行正式 aggregate，再等 baseline finalizer 发布成功。
4. 扩展 finalizer 执行 source-host postflight、Hostinger 严格发布、只读 composite 和 42 行合表。
5. 最终表中明确标注 v3 无泄漏协议、seed 42 与单次运行口径，不与官方重叠协议的 Table 4 数字混用。

## 实验纪律

- 保留诚实负结果，不为追求正结果反复改停止线。
- 种子约定为 42、1、2。单种子 baseline 与三种子均值只能做描述性比较。
- 指标使用 masked MAE，验证指标必须按全部有效观测点累计。
- `affected` 只是事后分组指标，除非实验明确声明标签监督。
- 新结果必须记录协议、split、normalizer 与 climatology 拟合范围、缓存哈希、Git SHA、dirty 状态和 schema version。
- 不得把 Windows 5080 与 Linux 主机的绝对路径、`config_hash` 或运行时 provenance 混成同机结果。
- 报告类正文不用破折号，不用括号内插说。代码注释保持最少，commit 信息使用英文。

## 项目方法和仓库约定

- 项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。
- `scripts/README.md` 是脚本分类入口。
- `outputs/README.md` 说明历史结果和本地产物的关系。
- `docs/archive/` 保存被主报告吸收的旧文档。
- 不批量移动仍被历史入口引用的脚本或结果。
- 本地原始数据位于根目录 `archive/`，不进入 Git。
- 远程运行包和日志放入被忽略的 `outputs/remote_runs/`。

## 凭据与远程运行

实验室共享主机、地址和密码集中明文维护在 `config/lab_hosts.env`。脚本可以直接加载该文件，也允许环境变量覆盖。私钥、个人令牌和个人账号凭据仍不进入仓库。

## 历史追溯

以下内容不再在本文重复，均已有完整落盘记录：

- v2 冻结、去重与 active event 隔离问题：`docs/audits/protocol_v3_review_2026-07-19.md`
- 原始数据泄露根因：`docs/audits/data_leakage_2026-07-18.md`
- 45 项 v3 分片、task040、task041 与 composite incident：`XTraffic项目完整汇报.md`
- IGSTGNN Table 4 九卡恢复、task034 和三源 composite：`docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md` 与主报告
- native-100 双机重分片与替代 5090：`docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md`
- v3 event Orange 八卡运行与恢复：`docs/audits/protocol_v3_event_4090d8_launch_2026-08-04.md`
