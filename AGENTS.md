# XTraffic Agent 入口

开始工作前依次阅读：

1. `CLAUDE.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`

当前 XTraffic 的旧缓存、v2 缓存和既有结果均不得写成已验证论文结论，也不得覆盖旧 summary。v2 已冻结。除原有泄露问题外，复审还确认了重复 forecast origin、active event 跨 split、验证指标累计错误和训练契约不完整。5080 已停止 v2 旧队列，最先完成的 4 个 v2 结果只可视为 `contaminated preliminary`。

当前修复协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`。最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。训练 checkout 在 Mac、5080 与 5090 均为 143 passed 和 1 个既有 warning，当前本地全仓回归为 502 passed 和 2 个既有 warning。三地区正式缓存已在 `outputs/dist_net/region_data_v3_noleak_dedup` 落盘，并通过本地两次、5080 一次及 5090 一次严格复验。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。

2026-07-20 22:23 新加坡时间起，正式运行口径由“5090 同机 45 项”改为双机分层矩阵。两台机器保持独立结果根、独立 `config_hash`、独立 postflight 和运行时 provenance。最终只能称为 `host-stratified distributed 45`，不得称为 homogeneous 5090 matrix。固定分片不交叉且并集严格为 45 项，所有直接 A/B 比较组保持在同一台机器。

2026-07-22，用户批准的 27/18 重分片已于 22:49 新加坡时间完成原子切换。5080 分片共 27 项：tasks001-004、007-010、013-016、019-030、040-042。5090 分片共 18 项：tasks005-006、011-012、017-018、031-039、043-045。两分片互斥且并集严格为 canonical 45，所有直接比较组保持同机。task039 完整落盘后通过 `BOUNDARY_RESULT_OK`，relay ready marker 已发布且保持完好。

5080 的旧 24 项队列已于 2026-07-22 18:13:06 新加坡时间完成 24/24，skipped valid 为 4，runner 退出码为 0。旧 24 项 source-host postflight 与 134 文件 allowlist 已通过。task007 的首个 attempt 因两条遗留诊断 PowerShell 扫描耗尽主机内存而失败，没有 summary 或正式任务目录，第二个 attempt 已完整重跑并验收。这是主机运维故障，不是模型失败。旧 task005 迁移记录不进入正式导出 allowlist。

2026-07-27 11:19:49 新加坡时间的发布层历史快照为：runner 层已完成或重新验收 45/45，5090 为 18/18，5080 为 27/27。`rtx5090_18` 与 `rtx5080_27` 均已在结果服务器完成逐文件复验和正式原子提升。该快照时只读 composite index 尚未完成，论文分析仍暂停。

task040 旧失败的根因是 producer float32 `.mean()` 与 runner float64 audit 的精度语义不同。新控制层保留 float64 audit，只对 graphwavenet、fourier_dual_net、staeformer 与 staeformer_icsf 精确复现 producer，summary 必须处于 2 ULP 内。恢复路径只接受 prior queue、failed task state、最新 zero-exit attempt、canonical path、非空日志和完整 artifact 全部通过的候选。修复提交为 `dfb6b7f` 与 `33fae6c`，本地全仓为 247 passed 和 1 个既有 warning，独立复审未发现 P0 或 P1。

部署探针曾发现 CMD 环境变量展开问题。Python preflight 拒绝了合成 task ID，没有 trainer 启动，但 queue 被短暂改写。异常快照已保存，SHA-256 为 `bdcb4718859b8dfbaf52d588e2d5af2789dcec42659f8927879807fd120869ab`。queue 随后根据多份落盘证据原子恢复，task state 与 artifact 未改变。最终实现由 Python 直接读取环境变量，CMD 不再展开其值。

task040 已于 11:23 新加坡时间从 `attempt_0001` 完成可审计 recovery，正式 task state 为 completed，没有 `attempt_0002`。summary、ckpt、prediction artifact 与 train log 哈希均与恢复前一致。

task041 在完成 30 个 epoch 与测试推理后，因旧 summary 临时路径达到 282 字符而触发 Windows 传统路径限制。checkpoint、prediction artifact 与 train log 完整，失败不是模型或数据问题。提交 `e7bf755` 缩短临时名并增加 incident-specific repair 与 completed state 幂等保留，本地全仓为 249 passed 和 1 个既有 warning。5080 dry-run 完成正式 cache 全量对照后，task041 于 17:08 从原 `attempt_0001` 原子恢复，staging 与正式目录均通过 runner 验收，没有 `attempt_0002`。summary 与 decomp mask SHA-256 为 `115c0cfba5d6a256a27011a6169797ac48606cb90a914a942919854e399e27ac` 与 `55ae377b62287d5098a895c5608cb8050d79520df2a5936f0754c0a18bef5d5a`，原三份 artifact 哈希未变。

2026-07-26 17:10，runner 以 `KEEP_COMPLETED` 保留 tasks040-041 的恢复状态并启动 task042 `attempt_0001`。17:15 的落盘日志为 epoch 1、batch 1500/4598，GPU 使用 10,799 MiB，利用率 97%，温度 63 摄氏度。runner、trainer 与 finalizer 当时各有 1 个进程。

task042 于 2026-07-26 23:32:54 新加坡时间完成并通过 runner 验收。all 与 unaffected masked MAE 均为 `10.780716896057129`，affected 为 `null`，best val masked MAE 为 `10.51922192820698`，有效点数为 `103270922`，参数量为 `1536372`。5080 队列由此达到 27/27，runner 层达到 45/45。

5080 finalizer 首次执行结果服务器目录树复验时遇到 PowerShell 至 Windows native ssh 的引号传递问题。该故障不影响训练结果、source-host postflight 或已上传文件。提交 `c48f232` 完成修复，本地全仓为 250 passed 和 1 个既有 warning，PowerShell 5.1 AST、Windows native argv 与真实 incoming 文件树复验均通过。finalizer 复用 152 个已上传文件，没有重训、重传或改写任务结果。`rtx5080_27` 于 2026-07-27 11:19:49 新加坡时间正式发布，transfer marker SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。

首次 composite 复验暴露结果服务器内存峰值风险。提交 `5f3113b` 在 float64 audit 后原位复用 prediction 缓冲计算 float32 producer error，并在首次指标校验前释放 actual。提交 `93038ec` 跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标契约。两项修复不改变 artifact、summary、训练 checkout 或任务 `config_hash`。本地全仓为 252 passed 和 1 个既有 warning，独立复审未发现 P0 或 P1。

只读 composite index 已于 2026-07-27 18:28:27 新加坡时间验证成功。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，`execution_layout` 为 `dual_host_stratified`，index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。index 由控制提交 `93038ec` 生成，共含 45 个任务和 2 个分片。5080 分片为 27 项，5090 分片为 18 项，canonical tasks001-045 各出现一次。首次使用 `c48f232` 的无输出尝试已作为 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。最终服务退出码为 0，内存峰值为 2.1G，swap 峰值为 15.1M。整个 composite 过程没有改写 summary 或 artifact。

composite 完成只证明正式双机分层矩阵的身份与完整性，不恢复论文结论。同机配对、多种子统计与窗口级显著性检验已于 2026-07-27 完成，输入为按 composite index 逐份复验的 45 份 summary 与 45 份 test_predictions.npz，分析脚本为 `scripts/protocol_v3_matrix_analysis.py`、`scripts/remote/distributed/protocol_v3_window_pairs.py` 与 `scripts/protocol_v3_window_pairs_report.py`，产物在 `outputs/protocol_v3_noleak_dedup/analysis/`。窗口级结论：s0c 显著优于 s0a（三区域三种子全部 p<0.01），s0g 小幅优于 s0f（多数显著、幅度小），ICSF 注入效应接近零（Alameda 不显著，ContraCosta 与 Orange 窗口级显著但幅度约 0.02-0.03），full_window 排序为 staeformer < fdn < gwn（全部显著）。论文主张是否恢复仍由用户据落盘结果决定。

IGSTGNN Table 4 复现使用独立协议 `igstgnn_table4_official_v1` 与矩阵 `igstgnn_table4_public_repro_39_v2`，不得与 v3 无泄露矩阵混合。正式 39 项清单 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`，claim scope 为 `public_material_maximum_fidelity_reproduction`，validity label 为 `official_protocol_with_known_overlap`。5080 上的本轮 Table 4 训练部署基于提交 `5aadfff`，Mac 复现测试现为 181 项通过与 1 个既有 warning，5080 Windows 训练相关门为 127 项通过与 2 项平台跳过。Alameda HL 与 LSTM 已正式完成并通过五文件验收。HL 的 H3、H6、H12 与 Average MAE 为 15.916486、19.056049、25.661367、19.649600，LSTM 为 12.825795、14.208229、17.033611、14.355326。LSTM prediction SHA-256 为 `8f70ed5a8264c007e2baf5453bf18d0d621ad0502ad29e39ea34f83315b0e769`。

task003 首次通过 SSH 附着运行，在完成 3 个 epoch 后于 2026-07-28 22:32:25 新加坡时间按计划中断，以迁移到持久控制层。runner 状态为 `interrupted`，退出码为 130，没有正式任务目录或正式结果。incident 位于 `D:\t4\incidents\table4_003_attached_interrupted_20260728T143225Z`，incident SHA-256 为 `24da66f71bac2ecfe8e3654ad1630cf97723f7b1f9c2c4fd64b12db7e666be6b`。提交 `fa5b458` 与 `8ad61bf` 增加 fail-closed 接力器和固定 5080 启动命令。接力器 SHA-256 为 `3410e9eab5a53c421afc3c93eaeb00f49dbf371dc86165a8d1e15b7bd5054023`，Mac 专项测试 2 项通过。Windows 任务计划 `XTraffic_IGSTGNN_Table4_v2` 于 22:36:39 启动，执行时限为 `PT0S`，开机后可恢复。它先运行 tasks003、013 pilot 门，全部通过后再启动完整 39 项。新 task003 于 22:36:40 从 epoch 0 开始，GPU 使用 11,899 MiB，利用率 82%。

2026-07-29，Table 4 正式执行切换为两台 RTX 4090 D 主机的 9 卡固定分片。计划 SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。五卡机承担 Orange 13 项，四卡机承担 Alameda 与 Contra Costa 26 项。5080 旧 pilot 在 epoch 38 后停止，没有 task003 正式目录，计划任务已禁用，incident 清单 SHA-256 为 `29c58bad6635ae3c7d476ec875cd577153ab4a95981d32f2a16baf89248bde4a`。

两端 27 个正式数据文件均通过逐文件 SHA-256 复验。首次启动因 Mac UID 501 所有权被安全门拒绝，没有 trainer 或正式结果。修正所有权并重新验哈希后，五卡机与四卡机分别于 00:47:14 和 00:47:32 新加坡时间正式重启。

截至 2026-07-29 10:57 新加坡时间，原九卡 runner 已完成并逐任务验收 21/39，四卡机为 13/26，五卡机为 8/13。剩余 18 项中有 7 项实际失败，11 项因 shard 遇错即停而未启动。STGCN 失败来自零对角线邻接矩阵被重复减单位矩阵，D2STGNN、DGCRN 与 IGSTGNN 失败来自 24 GB 显存容量。旧双卡 Orange IGSTGNN 恢复在反向传播时仍然 OOM，没有正式 task039。

提交 `b266c89`、`05e2951`、`1a5582f` 与 `af6456a` 增加带独立 provenance 的 24 GB 恢复层。STGCN 使用幂等自环移除。DGCRN 使用无状态 QR 预热。多卡执行把较小余数分片分给主卡。Orange D2STGNN 使用两次 32 样本微前向和一次优化器更新，按全批有效观测点加权梯度，有效 batch 仍为 64。真实探针均通过。Alameda D2STGNN 与 IGSTGNN 峰值约为每卡 9.9 GiB 和 7.6 GiB。Orange DGCRN、D2STGNN 与 IGSTGNN 峰值约为每卡 5.9 GiB、最高 13.5 GiB 和最高 10.0 GiB。STGCN 修正 GSO 全为有限值。本地全仓为 446 passed 和 2 个既有 warning，两台 Linux 训练门各为 158 passed、12 个平台跳过和 1 个既有 warning。

正式恢复控制器于 10:57 启动。四卡机 PID 为 `39176`，从 task005 开始。五卡机 PID 为 `40326`，从 task031 开始。恢复前 21 项的五文件清单已经锁定，四卡机与五卡机清单 SHA-256 为 `3e89aac6c8b5ebbf2bc8d2df4cc3f518587418464d9a49fd8b349e93967f4513` 与 `ee52c8e11ba150c3d49b65cf8d0efe480e46df0f5134ff310b3de14a440c1c61`。恢复 runner 与持久控制脚本 SHA-256 为 `52829dd65362ef03127f391b7054aff1c0feeacf011bc4af974e2be408f07a76` 与 `23980df3dfe42afe952bc52febb97e3fd85e1c1629b86e70b2b2679b3c8a5b4b`。只有双机 source-host postflight、结果服务器逐文件复验和原子提升全部通过后才自动关机，任何失败都会保持开机。

双机 finalizer 已设置结果服务器严格回传、逐文件复验、原子提升和成功后自动关机。关机命令显式使用 `/usr/bin/shutdown`。任何训练、postflight、容量、传输或远端复验失败都会保持机器开机。完整启动记录见 `docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md`。

截至 2026-07-29 17:05 新加坡时间，原 Table 4 矩阵已正式落盘并由任务 runner 验收 34/39。四卡机为 25/26，仅余 task021 Contra Costa STTN。五卡机为 9/13，正在执行 task036 Orange DGCRN，之后仍需 task037、task039 与 task034。结果服务器仍没有正式主机目录。

NDR-Net 补充对比保持原 39 项清单不变，使用独立矩阵 `igstgnn_table4_ndr_extension_9_v1`。扩展为 NDR-Net 乘三地区乘种子 42、1、2，共 9 项。扩展清单 SHA-256 为 `2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4`，配置 SHA-256 为 `6ca46be0fb76543a61a8bd32996c7598ac674bbe239b701d1a92481f63be9422`，九卡分片计划 SHA-256 为 `f20be0db02fbd04732640cd5b024f7012ce6a16ea155d6a90c0962839f857522`。claim scope 为 `same_protocol_project_method_extension`，validity label 仍为 `official_protocol_with_known_overlap`。扩展不得写成 IGSTGNN 原论文的方法行，也不得与 v3 无泄露矩阵混合。

扩展管线先用绑定的 `incident_stats.npz` 把官方标准化 flow 还原为原始流量单位，再只用 train split 的 x 与 y 位置拟合周内时段中位数基线。验证 MAE 按所有有效观测点累计。评测沿用 H3、H6、H12 与 12 步平均。实现提交为 `cd14b90`、`d67ba1c` 与 `085e68d`。Mac 全仓为 455 passed 和 2 个既有 warning，两台 Linux 扩展门各为 33 passed。

扩展部署在两台主机的独立 `/root/autodl-tmp/table4/ndr_repo`，没有改写原矩阵 checkout。原 finalizer 的关机由安全哨兵延期。第一次扩展控制器因正常的 Python 符号链接被过严安全门拒绝，没有训练、结果或关机动作。修复后的四卡机与五卡机延期控制器 PID 为 `447869` 与 `410737`，均在等待原矩阵回传成功。每台主机启动扩展正式分片前会先做最大地区的真实 CUDA 前向反向探针。扩展只有在独立 postflight、结果服务器逐文件复验和原子提升全部通过后才移除延期哨兵并调用 `/usr/bin/shutdown -h now`。

截至 2026-07-30 00:18 新加坡时间，原矩阵 runner 已验收 37/39。四卡机为 26/26，134 文件 postflight digest 为 `a43a4023cb7460fcd2ad2d2cece0adc01c7394588df206461744e67ec94550dd`，结果服务器逐文件复验和原子提升已于 17:19 通过。五卡机为 11/13，task039 Orange IGSTGNN 已完成 44/100 epoch，之后只余 task034 Orange STTN。原矩阵完整回传前不得生成最终 39 项表。

四卡机原有的六项 NDR 已完成并保持只读。2026-07-30，用户开机后批准把 Orange 三种子从五卡机转移到四卡机并行训练。重分配计划为 `igstgnn_table4_ndr_extension_4gpu_rebalance_v2`，SHA-256 为 `56a71fe2a3b8101ae282dc6f9b9efab07874108a40922276285206e99cbf2612`。原 batch 16 的单卡 Orange backward 探针 OOM，没有正式结果。提交 `87526d1` 用两次 max microbatch 8 前向、全批 loss 分母和一次优化器更新保持有效 batch 16。真实探针 loss 为 `42.45340157`，CUDA 峰值为 12,632,052,224 bytes。

四卡机正式 NDR 控制器 PID 为 `3677`，GPU 0、1、2 分别执行 Orange seeds 42、1、2。00:18 三项均完成 epoch 1，validation MAE 为 13.19330061、13.06060743、12.91147725。完成控制器 PID `5614` 会复验旧六项的 30 个 artifact 哈希，再执行 9 项 postflight、结果服务器逐文件复验、原子提升和关机。五卡机旧 NDR 等待器 PID `422803` 已停止，incident 清单摘要为 `556c4289e2b4ec41b35b40d1f9a72190e1ad1c3c5a55edf1986843a70aa6e8a1`。五卡机新关机等待器 PID `201649` 只负责在原 39 项回传复验后关机。控制脚本提交为 `dd2d087` 与 `b5d9751`。任何失败都会保留对应主机开机。

NDR 扩展现已完成并正式发布 9/9。结果服务器目录含 49 个文件和 940,488,092 字节，files digest 为 `ab21c8e300ba930e5611cb9c07ff5b047eb03d281ca669c756242707231c87e7`。task039 Orange IGSTGNN 已完整落盘，Average MAE、RMSE、MAPE 为 13.285545、23.504318、14.457104，summary SHA-256 为 `ab0288a4b8fe5a17c33b43b758bbc5851b0e63cc0cce7456a07eb5fe451ce998`。原矩阵 runner 层达到 38/39。

五卡机上的 task034 Orange STTN 单卡训练在尝试额外分配 11.22 GiB 时 OOM，没有正式结果。用户要求关机后，五卡机既有 12 项先完成关机前 artifact 清单并正常关闭，尚未正式回传。用户随后开启四卡机并批准接管 task034。独立恢复计划 `igstgnn_table4_task034_4gpu_recovery_v1` 的 SHA-256 为 `022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5`，execution layout 为 `cross_host_single_task_recovery`。

提交 `4935b9b` 增加 STTN 四卡整批恢复、真实完整优化步探针、独立 postflight 和严格回传。本地全仓为 465 passed 和 2 个既有 warning，四卡机专项门为 13 passed。2026-07-30 09:21:02 新加坡时间，控制器 PID `2928` 启动。batch 64 探针 loss 为 `170.42190551757812`，四卡峰值为 16,082,649,088、16,060,899,840、16,060,899,840 与 16,060,899,840 bytes。09:21:28 探针通过并启动正式 task034。首个 epoch 的训练 loss 为 29.8492，验证 loss 为 21.1960。task034 只有在五文件验收、独立 postflight、结果服务器逐文件复验和原子提升全部通过后才触发四卡机关机。最终 39 项必须标记为包含跨主机单任务恢复，不得称为原九卡同质矩阵。

五卡机已重新开启并完成既有 12 项只读回传。提交 `918af63` 的固定计划 `igstgnn_table4_host5_partial_export_12_v1` SHA-256 为 `cd81652493690d1d329f5eecd45fe541c9de7db26c93650fc490a2507616845d`，明确排除 task034。source-host postflight 验证 12 项和 60 个 artifact，总字节数为 1,975,159,969，files digest 为 `5c11ad4596228f08613b783d8a42ed9334e9976ce8cff4d23d09f6b79f055490`。结果服务器容量门首次安全暂停，随后只清理系统归档日志和包缓存，释放 1,753,280,512 bytes。重试完成逐文件复验和原子提升，transfer marker SHA-256 为 `53f73b3e7ae2f92535eed44fb0fc934d7be45387de579b8dfc3fcb853afd54cc`。五卡机已自动关机并确认 SSH 不可达。

task034 于 2026-07-30 10:38:24 新加坡时间完成训练。H3、H6、H12 与 Average MAE 为 12.652452、13.311158、14.048541 与 13.212108，Average RMSE 与 MAPE 为 23.502980 与 14.091901%。prediction SHA-256 为 `c6064688ab3bedd36b4f058ce899a97b8882645c4169979d0a6170fabf1f773a`。

首次 postflight 暴露 provenance 层级读取错误，提交 `a9511d5` 修复 `factory.execution_adapter` 读取。随后 finalizer 暴露单任务报告相对 shard 根但源根仍指向外层结果目录的问题，提交 `51148f1` 修正并补回归。两次均 fail closed，没有重训或 artifact 改写。本地全仓为 469 passed 和 2 个既有 warning，四卡机专项测试为 4 passed。

task034 已正式发布到 `task034_4gpu_recovery_v1/hosts/table4_4090d4_task034`。source bytes 为 170,575,062，files digest 为 `6ce435df4293dad11602378b1e5b9acf18ffde1717f5e0fdfb5922f3282e8258`，transfer marker SHA-256 为 `e393e8238b25677b55e7f1ae00813c3f0a9eb46f7a532c0eb7bba21ee4758d68`，postflight SHA-256 为 `fa8a3d0f0e0fba9d67fa569392edc83c9a74d9131b81439ea7644c7fbc700e15`。结果服务器独立复验五份结果均为 OK，incoming 不存在。四卡机随后自动关机，连续三次 SSH 探测均返回连接关闭。原 39 项任务结果当时达到 39/39，该快照时仍需三源只读 composite index 验证后生成最终表。

2026-07-30，原 39 项三源 composite 已通过两次完整只读复验。控制提交为 `dc1f8a54919d611ac074e128d6f957c03dd7b0e6`，source spec SHA-256 为 `38645a51935618e4f12a96374d5d9695711bdf6581d984ec2baa15be63345e56`，validator SHA-256 为 `8d988a7164fb48c526ab0e7df7b1832ecc374bac108371a354a12582298b0e10`。正式 index 的状态为 `verified_official_protocol_with_known_overlap_39`，SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。它包含 3 个来源、39 个唯一 canonical task、199 个文件和 3,399,778,682 字节。第二次完整复验除时间字段外与正式 index 一致，没有改写任何结果。

论文同结构 39 项复现表已生成。CSV SHA-256 为 `60160b443f63425c3b10cd30e29253db85a17d68e2a3cf12ac5bcf195449cacc`，逐单元论文对照表 SHA-256 为 `8f1f80ea3ec29f27efc0ddefd3d4711f06376f0655d68ec51c078016f37a0e98`。复现与论文报告值的 Average MAE 绝对差均值为 2.450150，必须保留差异，不得写成作者数值的完全复现。

NDR 9 项正式树的结果服务器复验回执 SHA-256 为 `dbd9dcb355a3324707e728a1942465612cd443695b51ea6359e80dc03d1be100`，NDR 扩展 index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。三地区 Average MAE 的三种子均值与样本标准差为 Alameda `10.759524 ± 0.040862`、Contra Costa `11.413717 ± 0.024537`、Orange `11.569331 ± 0.031664`。最终 39 行加 3 个 NDR 汇总行的 CSV SHA-256 为 `057e0fa94bbea24d834ed16ed00b94b0599a9e9081321cc205643793c47f34c1`。NDR 只能标为 `same_protocol_project_method_extension`。baseline 为单次运行，NDR 为三种子汇总，因此当前只支持描述性比较。全仓回归为 474 passed 和 2 个既有 warning。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。

BiST 与 BiST-NDR 使用独立配对矩阵 `igstgnn_table4_bist_ndr_pair_18_v1`。矩阵 SHA-256 为 `90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab`，配置 SHA-256 为 `b763fb35b392e265a410325efa0ef34afa3b0d16c7c2e56c71bc3243844e8764`。矩阵包含两种方法、三地区和种子 42、1、2，共 18 项。两个方法必须使用相同的 `partial_batch: retain`、测试观测集合、训练控制和全局验证 masked MAE。

Mac 全仓回归为 480 passed 和 2 个既有 warning，5080 Windows 专项门为 5 passed。Orange 最大地区的两个真实 CUDA 完整优化步探针均通过。5080 正式队列于 2026-07-31 10:06:03 新加坡时间在独立 `D:\t4\bist_ndr_repo` 启动，计划任务名为 `XTraffic_BiST_NDR_Pair_v1`。首项 `pair001_bist_aligned_alameda_seed42` 已于 10:14 完成并通过五文件验收，H3、H6、H12 与 Average MAE 为 `11.205525`、`11.784402`、`12.675788` 与 `11.740657`。队列为 1/18，正在执行 `pair002_bist_aligned_alameda_seed1`。单项结果不能形成 BiST-NDR 性能结论。该矩阵不得写成 IGSTGNN 原论文方法，也不得与 v3 无泄露矩阵混合。

BiST 与 BiST-NDR 配对队列已于 2026-07-31 13:41:13 新加坡时间完成。runner 为 completed，18/18 任务全部验收，失败和 skipped 均为 0，正式任务树含 18 个目录与 90 个文件。它是统一 30 epoch 训练控制下的结构配对实验，不是填回原 Table 4 的原生配方补跑。当前只确认 runner 完整性，尚未生成配对性能结论。

BiST 与原 STAEformer 版 NDR-Net 的 100 epoch 补充对比使用独立矩阵 `igstgnn_table4_bist_ndr_native100_18_v1`。矩阵 SHA-256 为 `e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246`，配方 SHA-256 为 `37ebcb574f516bc1b662ade0d5264deda96a20d017d20edc7a57253daab5e66b`，实现提交为 `53f4b3a`。矩阵包含两种方法、三地区和种子 42、1、2，共 18 项。BiST 保留原生 batch 64、drop partial batch、Adam 0.002、MultiStepLR 和 patience 20，仅把上限从 500 epoch 改为 100。NDR-Net 保留原生 batch 16、retain partial batch、Adam 0.001、CosineAnnealingLR、全局验证 masked MAE 和无 early stopping 契约，仅把上限从 30 epoch 改为 100。

Mac 全仓回归为 489 passed 和 2 个既有 warning，5080 Windows 专项门为 25 passed。Orange 最大地区真实 CUDA 完整优化步探针均通过，BiST 与 NDR-Net 峰值分配显存为 `3064998912` 与 `12633100800` bytes。5080 计划任务 `XTraffic_Table4_Native100_v1` 于 2026-08-01 00:02:23 新加坡时间在独立 `D:\t4\native100_repo` 启动，执行时限为 `PT0S`。启动复验时为 0/18 完成、0 失败，正在执行 `n100001_bist_alameda_seed42`。该矩阵沿用 Table 4 的 H3、H6、H12 与 Average 指标列，当前不得形成性能结论，不得覆盖原有结果，也不得与 v3 无泄露矩阵混合。

最终入表器为 `scripts/reproduction/igstgnn_table4/native100_compare_report.py`。它在 18 项全部通过五文件复验后，保留既有 42 行并追加 6 个地区方法行，输出同结构 CSV、Markdown、LaTeX 和可追溯 manifest。

2026-08-01 12:18 新加坡时间起，原生配方 100 epoch 矩阵使用计划 `igstgnn_table4_native100_5080_5090_v1` 按地区双机分层执行。计划 SHA-256 为 `d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e`，execution layout 为 `dual_host_region_stratified`。5080 选择 Alameda 与 Contra Costa 的 12 项，5090 选择 Orange 的 6 项。两分片互斥且并集严格为 canonical 18，同一地区的 BiST 与 NDR-Net 保持同机。

5080 的旧 18 项计划任务已停止并禁用。task013 中断时没有正式任务目录，incident `files.sha256` 摘要为 `94802ea874935de3db70eb3d5b730ec14e978662ebeada4014bc7eb09f34be23`。新计划任务 `XTraffic_Table4_Native100_5080_CC_v1` 从 9 个已验证结果继续，12:40 前 task013 已完成 epoch 17。5080 结果根内已有 Orange BiST tasks007-009 不得进入最终 composite。

5090 使用训练归档提交 `ace318b`。Orange 九个数据文件已逐文件复验，split manifest SHA-256 为 `c37b78f19766fec9ac57d28d6a7becd8a3b334261cef4e7863f29b0bbcc7b019`。5090 专项门为 25 passed。Orange BiST 与 NDR-Net 真实 CUDA 完整优化步探针峰值分配显存为 `3142229504` 与 `12682121728` bytes。5090 持久队列于 12:38:57 启动，选择 6 项，初始状态为 0/6、0 失败，执行 task007。12:40 时 task007 已完成 epoch 6。Mac 全仓为 493 passed 和 2 个既有 warning。最终必须在双机 postflight、结果服务器逐文件复验和只读 composite 完成后才能生成 18 项表。

用户已明确要求 5090 训练完成后关机。失败关闭守护脚本 `igstgnn_table4_native100_5090_shutdown_waiter.sh` SHA-256 为 `d0168a7f1b3b2b1b5386de1cebb52681fc1319babe8df1eddb5344228706bbb1`，远端持久 PID 为 `20225`。它每 60 秒检查一次，只在 Orange 6/6 逐任务五文件复验、queue completed、0 失败、runner 退出码 0 且没有残留训练进程时调用 `/usr/bin/shutdown -h now`。异常时保持开机。守护启动时 5090 为 4/6 完成，task017 已到 epoch 82。

5080 地区分片已于 2026-08-01 15:52:38 新加坡时间完成，声明内 12/12 全部通过 runner 验收，失败数为 0。用户报告主机随后断电。23:37 局域网 SSH 恢复后复验确认 completed 状态、12 个已验证任务和空闲 GPU，断电没有造成训练中断或重跑。结合 5090 的 4/6，完整矩阵当时为 16/18 已验收。

2026-08-02 11:11，新替代 5090 从头重跑后已完成 Orange BiST 三种子 3/6，正在 `n100016_ndr_net_orange_seed42`。空闲 5080 先通过 `n100017` 真实 CUDA 完整优化步探针，loss 为 `42.24521446`，峰值分配显存为 `12633100800` bytes。随后隔离 live speed gate 运行 292 秒仍未完成首个 epoch。同期 5090 的 `n100016` 真实速度约为每 epoch 3.5 分钟。预计收益不足以抵消修改正式分片、来源规范和 finalizer 的风险，候选计划 `igstgnn_table4_native100_seed_rebalance_v2` 已标记为 `rejected_after_live_speed_gate`。

5080 辅助计划任务已停止并禁用，没有正式任务目录。incident 位于 `D:\t4\native100_orange_5080_recovery\incidents\seed_rebalance_no_speedup_20260802T032250Z`，清单摘要为 `4ce4bf5402c69d3f11adb95a711fe8f2df3e18161a3e81dcec1410ef592e45ed`。原按地区双机计划继续生效。11:25:57 时 5090 为 3/6 完成、0 失败，task016 完成 epoch 14，完整矩阵为 15/18，finalizer PID `2815` 正常等待。5080 保持空闲，测速 incident 不得进入最终 composite。本地全仓回归为 498 passed 和 2 个既有 warning。

2026-08-02，旧 5090 端口仍不可达。新端口 13647 对应全新容器 `autodl-container-dbf949bc7b-086867e4`，数据盘为空，旧机的 4/6 正式结果与 task017 epoch 82 临时状态均未恢复。旧守护当时不满足 6/6 关机条件，因此离线不能认定为成功收尾。当前只记录为疑似余额或供应商停机，旧机未回传结果不得进入 composite。

替代 5090 已重新部署训练归档提交 `ace318b`、BiST pristine commit `dd94adf7721fcbb9e3feb5d1b44040305199a4cc` 与 Orange 九个正式文件。数据逐文件 SHA-256 复验通过。BiST 和 NDR-Net 完整优化步探针 loss 为 `148.53358459` 与 `42.48369026`，峰值显存为 `3142229504` 与 `12682121728` bytes，与旧机一致。Orange 六项队列于 2026-08-02 09:48:50 新加坡时间从 0/6 重启，launcher PID 为 `2314`，matrix runner PID 为 `2318`。09:56:35 时 task007 到 epoch 48，队列为 0/6、0 失败。

替代主机使用专用受限结果服务器 SSH 身份，公钥指纹为 `SHA256:LpNIDomLiv0Vm6YHvJvN5miilCwUSSXSPXnRNhAt5hg`。fail-closed finalizer SHA-256 为 `a1c85fc95b5af92b51dcd44e209cc132a93a1b7bf7459d4552fcfe42fa892597`，持久 PID 为 `2815`。它只在六项本地五文件复验、runner 零退出、无训练进程、结果服务器容量门、逐文件 SHA-256、精确 allowlist 和原子提升全部通过后执行关机。任何失败都会保留机器开机。本地全仓回归现为 496 passed 和 2 个既有 warning。

2026-08-03，原生配方 100 epoch 矩阵已完成全部收尾。替代 5090 的 Orange 6/6 与 5080 的 Alameda、Contra Costa 12/12 均为 completed，失败数为 0。5090 正式树含 31 个结果文件和 `1109955208` bytes，files digest 为 `1ae14803b44dcd88190dbf81ef47431ce161238749fec0c5e37c67acc37b9a52`。结果服务器复验和原子提升后，5090 已自动关机。

5080 正式树含 61 个结果文件和 `686213987` bytes，files digest 为 `5d2de42964e581ea68f6612e7c05b4672268b22f267f887c191e62bc1470abf9`。旧 Orange BiST tasks007-009 不在导出 allowlist 内。Windows SSH 子进程不退出事故发生在目标文件写入与哈希校验成功之后，没有损坏正式结果。提交 `4920a4d` 增加无标准输入 SSH 与已验证文件容量复用，重启后完成严格回传。5080 保持开机。

两份正式树已完成两次全量只读 composite 复验。正式 index 状态为 `verified_native100_dual_host_18`，SHA-256 为 `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c`，含 2 个来源、18 个 canonical task、92 个结果文件和 `1796169195` bytes。第二次复验除时间字段外与正式 index 完全一致。composite 控制提交为 `a34dc958f336e06779c4419bbdb5303c93098c91`。

最终 48 行表已保存在 `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`。CSV、Markdown、LaTeX 和 manifest SHA-256 分别为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`、`e29348144e19d728dc614016b7d3653dd2587e0eb4fbb32b7569d4d99ce3ae5a`、`66bd422b30e63b500d7e33c103982735c5ea53af60b760463ae2b7167eef262c` 和 `e9a4859fac07abd532f48f0075f6cbfa3e491f80af32d18082291680b74416bf`。100 epoch Average MAE 三种子汇总为 Alameda BiST `10.938871 ± 0.011328`、NDR-Net `10.504639 ± 0.056600`，Contra Costa BiST `11.582189 ± 0.010355`、NDR-Net `11.299118 ± 0.012621`，Orange BiST `11.765688 ± 0.005298`、NDR-Net `11.136362 ± 0.030750`。该表继承 `official_protocol_with_known_overlap`，不得与 v3 无泄露矩阵混合。本地全仓回归为 502 passed 和 2 个既有 warning。
