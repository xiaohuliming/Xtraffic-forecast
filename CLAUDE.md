# XTraffic 项目导航与协作约定

最近更新：2026-07-30

## 新会话先读

1. `README.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`
6. `docs/README.md`

`XTraffic项目完整汇报.md` 是研究历史、实验数字和出处的唯一权威长报告。数据是否仍可用于论文，以 v3 复审、2026-07-18 泄露审计和有效性清单为准。

## 当前状态

审计前的研究线已经完成：

- RADS 在 GWN 与 STAEformer 两个骨干、三个区域上完成验证。
- RADS-O 增加逐 horizon 门控与训练期 oracle 蒸馏。
- NDR-Net 在 RADS-O 上增加正常态与扰动态双残差头。
- 三者是包含关系：`RADS ⊂ RADS-O ⊂ NDR-Net`。
- 审计前记录的最低历史 MAE 为 Alameda 的 NDR-Net 11.052。
- Orange 的 STAEformer RADS 验证已经完成。README 中曾写迁移仍在进行，属于过期状态。
- P2 六锚点库实验失败，方法加注线已经关闭。
- baseline zoo 已启动并有本地回传结果，但尚未形成权威诊断汇总。

以上是历史实验进度，不是当前可发表状态。

## 当前协议状态

2026-07-18 审计确认旧 XTraffic 管线存在 P0 数据问题：

1. 全年 `np.interp` 会把未来观测写入历史输入。
2. RADS 气候态在拟合前使用事故标签排除数据，严格 label-free 主张不成立。
3. severity、affected 与最终 duration 含事后信息，部分模型却将其用于输入或损失。
4. 事件协议和 full-window 协议存在切分边界目标重叠。
5. 多个脚本的 scaler 或残差统计越过训练边界。

因此，所有旧 XTraffic 绝对数值和配对结论均暂停进入论文。旧结果只读保留，不能删除或原地覆盖。

v2 随后修复了原始泄露问题，但全面复审又确认了四类影响实验有效性的缺陷：

1. 同一 forecast origin 可能因多个事故重复进入训练或测试，指标会被重复样本扭曲。
2. active event identity 仍可能跨越 split，按锚定事故分组不足以保证严格隔离。
3. 部分验证 MAE 按 batch 或样本均值累计，没有按全部有效观测点统一计算。
4. 多个训练入口缺少一致的 mask、图支持、缓存身份、输出隔离和恢复契约。

因此 v2 已冻结，5080 队列已停止。队列最先完成的 4 个结果只能标记为 `contaminated preliminary`，不能用于论文、模型选择或恢复旧结论。v2 的三地区样本数、远端测试记录和启动时 GPU 状态保留为历史运维记录，不再视为正式协议验收证据。

当前修复版本为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`，它只代表缓存 provenance。最终训练代码 SHA 已锁定为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。训练 checkout 在 Mac、5080 与 5090 均为 143 passed 和 1 个既有 warning，当前本地全仓回归为 252 passed 和 1 个既有 warning。默认缓存目录为 `outputs/dist_net/region_data_v3_noleak_dedup`，默认结果根目录为 `outputs/protocol_v3_noleak_dedup`。

三地区 v3 正式缓存已在 `outputs/dist_net/region_data_v3_noleak_dedup` 落盘，并通过本地两次、5080 一次及 5090 一次严格复验。正式 bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。

| 区域 | 正式样本数 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

缓存状态现为 `verified`。远端传输归档 SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。5080 解包后与 5090 传输后均再次运行严格缓存审计，结果仍为同一 bundle ID、bundle digest 和 `verified`。

5080 上的 v3 tasks001-004 已完成，并通过 runner 对 prediction artifact、有效点数及 all、affected、unaffected 指标的验收。task005 在迁移到 5090 后被主动终止，runner 如实记录退出码 1。这是计划内迁移，不是模型失败。

最先落盘的 5080 tasks001-004 与 5090 tasks005-006 合计为 6 个 seed42 正式 v3 单项结果。它们与冻结的 4 个 v2 `contaminated preliminary` 结果性质不同，不能混为一谈。2026-07-27 11:19:49 新加坡时间的发布层历史快照为：runner 层 45/45 全部完成或重新验收，5090 分片为 18/18，5080 分片为 27/27，两个分片均已回传并在结果服务器完成逐文件 SHA-256 复验与原子提升。该快照时只读 composite index 尚未完成，因此配对比较、多种子统计、显著性检验与论文结论仍保持暂停。

2026-07-20 22:23 新加坡时间起，正式运行口径改为双机分层矩阵。两台机器保留独立结果根、各自主机路径相关的 `config_hash`、独立 source-host postflight 与运行时 provenance。最终产物称为 `host-stratified distributed 45`，不再称为“5090 同机完整矩阵”。固定分片不交叉且并集严格覆盖 canonical 45，所有直接比较组保持同机。

5080 的旧固定分片为 tasks001-004、007-010、013-016、019-030，共 24 项。2026-07-22 18:13:06 新加坡时间，队列达到 24/24，skipped valid 为 4，runner 输出 `ALL_DONE` 并以退出码 0 结束。task007 的 `attempt_0001` 在 epoch 9 数据装载阶段因两条遗留诊断 PowerShell 扫描占用约 48 GB 私有内存而触发 NumPy `ArrayMemoryError`。该 attempt 没有 summary 或正式任务目录，失败记录原样保留。停止两条诊断进程后，runner 以 `attempt_0002` 完整重跑并验收 task007。这是主机运维故障，不是模型、数据协议或训练数值失败。旧 task005 迁移记录不进入正式导出 allowlist。18:14:38，5080 source-host postflight 对 24 项复验通过。18:14:45，导出 allowlist 确认 134 个文件，总计 3,372,164,933 字节。

5090 的旧固定分片为 tasks005-006、011-012、017-018、031-045，共 21 项。task005 与 task006 在分片启动时由 runner 重新验收并跳过。task039 于 2026-07-22 22:49 新加坡时间完整落盘，随后通过 `BOUNDARY_RESULT_OK`。task005 的 all、affected、unaffected MAE 为 12.025650、17.932862、11.720226。task006 为 11.740712、17.583309、11.438629。这两项都只有 seed42，只能作为方向性证据。旧 5090 队列在更早的边界切换时为 task007 创建了零字节日志、running state 和未完成 staging，未产生 summary 或正式目录，postflight、上传和 composite index 均明确排除。

2026-07-22 20:40 新加坡时间，用户批准将 ContraCosta full-window 的 tasks040-042 整组迁到 5080。目标固定分片现为 5080 共 27 项，任务是 tasks001-004、007-010、013-016、019-030、040-042。5090 共 18 项，任务是 tasks005-006、011-012、017-018、031-039、043-045。两组互不交叉，并集严格为 canonical 45。tasks037-039、tasks040-042 与 tasks043-045 三个直接比较组各自保持同机。

27/18 已成功越过原子提交点。5090 旧 runner、旧 finalizer、边界 watcher 与 transition 均已按设计退出。relay ready marker 已发布且保持完好。5090 新队列已完成 18/18，source-host postflight 在 2026-07-23T04:40:34Z 验证通过。`rtx5090_18` 已回传并通过 marker 复验，包含 97 个 source files、10,071,444,446 source bytes 和 18 个 task dirs。5090 已按计划关机，SSH 不可达。

5080 task040 的旧失败根因已经修复。控制层保留原 float64 审计，仅对明确使用 float32 `.mean()` 的 producer 从同一 prediction artifact 精确复现指标。summary 必须与 producer 值相差不超过 2 ULP，producer 与 float64 audit 的差异还必须处于归约误差界内。`staeformer_deseason` 不进入兼容分支。staging recovery 只接受 prior queue 与 task state 身份一致、最新 attempt、严格整数 returncode 0、canonical 路径、非空日志和全部 artifact 重新验收通过的候选，任何失败都不得静默重训。

修复提交为 `dfb6b7f` 与 `33fae6c`，本地全仓回归为 247 passed 和 1 个既有 warning。独立复审未发现 P0 或 P1。2026-07-26 的远端 shadow validation 通过后，新 runner、distributed control、launcher 与 finalizer 已按 SHA-256 部署到独立 ops 根，训练 checkout 仍为 clean `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。

控制部署期间的恶意环境变量探针暴露了 CMD 展开缺陷。Python preflight 安全拒绝该合成 task ID，没有 trainer 启动，但 `queue_status.json` 被短暂改写为 failed 0/27。异常快照已保存到 `state/control_incidents`，SHA-256 为 `bdcb4718859b8dfbaf52d588e2d5af2789dcec42659f8927879807fd120869ab`。随后根据落盘 shutdown state、task040 failed state、cache manifest、canonical manifest、clean Git SHA 和 24 个 `skipped_valid` state 原子恢复 queue。task state 与训练 artifact 从未被改写。

task040 已于 2026-07-26 11:23 新加坡时间通过新 runner 的完整重验并原子提升，未重训且不存在 `attempt_0002`。summary、checkpoint、prediction artifact 与 train log 的 SHA-256 分别保持为 `be9c84f7bd1608d40ad9e0c623854962cf223715695d38351228d20d9557a97d`、`7537f941f7e8972c2768d8fae7f2481d7f0d981b0d4f5b19a8e588925cd70df8`、`34b74015d3e11b504f28ca89f868b497aa2de4956b75ba078d8010cafde50f45` 和 `51aba30d3dd452ba58f28fb53ea8c254ef908654a4d365b1ff5259c88ba5cd63`。

task041 于 2026-07-26 13:40 新加坡时间完成 30 个 epoch 和测试推理后，在写入 summary 临时文件时触发 Windows 传统路径长度限制。正式目标路径长度为 239，旧临时路径长度为 282。checkpoint、prediction artifact 与 train log 均已完整落盘，失败发生在结果写入阶段，不是模型训练、数据协议或数值失败。提交 `e7bf755` 将 summary 临时名缩短，并增加 completed state 幂等保留和 incident-specific repair。Mac 全仓回归为 249 passed 和 1 个既有 warning，真实 5080 dry-run 对 103,270,922 个有效点完成 cache 全量对照，producer MAE 为 11.21213436126709，float64 audit 为 11.212104746452527。

2026-07-26 17:08，新脚本从哈希锁定的 task041 `attempt_0001` 重建 `summary.json` 与 `decomp_mask.json`，两份文件 SHA-256 分别为 `115c0cfba5d6a256a27011a6169797ac48606cb90a914a942919854e399e27ac` 和 `55ae377b62287d5098a895c5608cb8050d79520df2a5936f0754c0a18bef5d5a`。runner 在 staging 与正式目录各完成一次完整验收后原子提升。原 checkpoint、prediction artifact 与 train log 哈希保持不变，没有 `attempt_0002`。incident record SHA-256 为 `630e2d31ca2b8599a0ca9901f5d50e85049ca438a4ca894da744c49eecf4ced4`，completed task state SHA-256 为 `4bb3cb200c77391d24e3b331b64b5787b70ed42163a1db617635b758aecbfe22`。task040 state SHA-256 仍为 `c54ae78dfdd6ec4e8e4b0005f373e2a712f8686a3fd211fd15cb1703fa5f2312`。17:10，runner 以 `KEEP_COMPLETED` 保留 tasks040-041 的审计状态并启动 task042 `attempt_0001`。17:15 的落盘日志显示 task042 为 epoch 1、batch 1500/4598，GPU 使用 10,799 MiB，利用率 97%，温度 63 摄氏度。finalizer 正在等待 27/27 后执行 postflight 与回传。`rtx5080_27` 仍不存在。

task042 于 2026-07-26 23:32:54 新加坡时间完成并通过 runner 验收。all 与 unaffected masked MAE 均为 `10.780716896057129`，affected 为 `null`，best val masked MAE 为 `10.51922192820698`，有效点数为 `103270922`，参数量为 `1536372`。5080 队列由此达到 27/27，runner 层达到 45/45。

5080 finalizer 首次执行结果服务器目录树复验时，PowerShell 传给 Windows native ssh 的嵌套双引号被剥离，导致远端 AWK 表达式损坏。该故障只影响收尾复验，不影响训练结果、source-host postflight 或已上传文件。提交 `c48f232` 将 AWK 改为无嵌套引号的等价实现。本地全仓回归为 250 passed 和 1 个既有 warning，PowerShell 5.1 AST、Windows native argv 与真实 incoming 文件树复验均通过。finalizer 复用已上传的 152 个文件，没有重训、重传或改写任务结果。`rtx5080_27` 于 2026-07-27 11:19:49 新加坡时间完成正式原子提升，包含 27 个任务目录和 4,822,525,731 字节源数据。transfer marker SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。

首次 composite 复验暴露了结果服务器内存峰值风险。旧 validator 会在完整 prediction 与 actual 之外再创建两份全尺寸 float32 error 数组。提交 `5f3113b` 改为在 float64 audit 完成后原位复用 prediction 缓冲，并在首次 producer 指标校验前释放 actual。提交 `93038ec` 进一步跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标、有限性、valid_count、float64 audit 与 float32 producer 契约。500 组随机 float32、mask 与无效 target NaN 差分均保持逐元素及 masked mean 完全一致，artifact SHA-256、summary、训练 checkout 与任务 `config_hash` 均不变。本地全仓为 252 passed 和 1 个既有 warning，独立复审未发现 P0 或 P1。

只读 composite index 已于 2026-07-27 18:28:27 新加坡时间验证成功。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，`execution_layout` 为 `dual_host_stratified`，index SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。控制提交为 `93038ec`。index 共含 45 个任务和 2 个分片，5080 分片为 27 项，5090 分片为 18 项，canonical tasks001-045 各出现一次。首次使用 `c48f232` 的无输出尝试已作为 incident 保留，incident SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。最终服务退出码为 0，内存峰值为 2.1G，swap 峰值为 15.1M。复验只读取正式分片，没有改写任何 summary 或 artifact。

## IGSTGNN Table 4 复现状态

2026-07-28 启动独立的 `igstgnn_table4_official_v1` 复现线，目标是按公开材料最大忠实复现论文 Table 4 的 13 个方法乘 3 个地区，共 39 项。该复现使用作者发布的 Kaggle 数据与官方 70%、15%、15% 顺序切分，不使用 `protocol_v3_noleak_dedup` 缓存。官方协议保留已知重复与跨 split 重合，因此有效性标签固定为 `official_protocol_with_known_overlap`，不得写成无泄露结果。

正式 v2 矩阵清单位于 `config/reproduction/igstgnn_table4/matrix.public_material_v2.json`，SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`。LargeST、BiST、IGSTGNN 上游提交分别锁定为 `b7c6ff5f7f0c10051ed10a319b2cfe58021312f9`、`dd94adf7721fcbb9e3feb5d1b44040305199a4cc`、`64c6e243466f791f1b75e50421d63e39577ec9fa`。BiST 三地区实际随机 seed 已分别冻结为 676615、683452、150462。5080 上的本轮 Table 4 训练部署基于提交 `5aadfff`。Mac 复现测试现为 181 项通过与 1 个既有 warning，5080 的 Windows 训练相关测试为 127 项通过、2 项平台跳过、1 项仅用于模拟不可在原生 Windows 执行的竞态测试被排除。

正式数据与三个 pristine checkout 已部署到 5080 的 `D:\t4`。Alameda HL 与 LSTM 已完成并通过五文件复验，测试均按 LargeST 源码实际消费 2,176 条，丢弃不足一个 batch 的 28 条尾部样本。HL 的 H3、H6、H12 与 Average MAE 分别为 15.916486、19.056049、25.661367、19.649600，prediction SHA-256 为 `63482403b21e74577e2c74cd3de198bdfafa8c90e8692843a506adcf2614e6b0`。LSTM 的对应 MAE 为 12.825795、14.208229、17.033611、14.355326，prediction SHA-256 为 `8f70ed5a8264c007e2baf5453bf18d0d621ad0502ad29e39ea34f83315b0e769`。

task003 首次 SSH 附着运行在完成 3 个 epoch 后于 22:32:25 按计划中断，目的是迁移到不依赖 SSH 的持久执行。该尝试退出码为 130，没有正式任务目录，不影响 HL 或 LSTM。incident SHA-256 为 `24da66f71bac2ecfe8e3654ad1630cf97723f7b1f9c2c4fd64b12db7e666be6b`。提交 `fa5b458` 与 `8ad61bf` 增加经过测试的 fail-closed 接力器和固定启动命令。Windows 任务计划 `XTraffic_IGSTGNN_Table4_v2` 于 22:36:39 启动，先执行 tasks003 与 013，二者均通过后才启动完整 39 项。新 task003 已于 22:36:40 从 epoch 0 开始。接力状态位于 `D:\t4\control\scheduled_handoff_status.json`，正式结果根为 `D:\t4\results\igstgnn_table4_public_repro_39_v2`。

2026-07-29，正式执行已切换到两台 RTX 4090 D 主机，共 9 张 24 GB GPU。固定计划 `igstgnn_table4_4090d_9gpu_v1` 的 SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。五卡机承担 Orange 13 项，四卡机承担 Alameda 与 Contra Costa 26 项。三地区方法内比较均保持同机。两端 27 个正式数据文件均通过逐文件 SHA-256 复验。5080 旧 pilot 在 epoch 38 后停止，没有 task003 正式目录，计划任务已禁用，incident 清单 SHA-256 为 `29c58bad6635ae3c7d476ec875cd577153ab4a95981d32f2a16baf89248bde4a`。

首次九卡启动因 rsync 保留 Mac UID 501 被所有权安全门拒绝。9 个 shard 均在 trainer 启动前退出，没有正式结果。修正所有权并重新完成数据哈希校验后，五卡机与四卡机分别于 00:47:14 和 00:47:32 新加坡时间正式重启。

截至 2026-07-29 10:57 新加坡时间，原九卡 runner 已完成并逐任务验收 21/39，四卡机为 13/26，五卡机为 8/13。其余 18 项中有 7 项实际失败，11 项因 shard 遇错即停而未启动。STGCN 的根因是发布邻接矩阵已经为零对角线，而 LargeST 入口仍重复减单位矩阵，产生含 NaN 的缩放拉普拉斯。D2STGNN、DGCRN 与 IGSTGNN 的失败属于 24 GB 显存容量。旧双卡 Orange IGSTGNN 恢复在反向传播时仍然 OOM，没有形成正式 task039。

提交 `b266c89`、`05e2951`、`1a5582f` 与 `af6456a` 增加带独立 SHA-256 provenance 的恢复层。STGCN 只执行幂等自环移除。DGCRN 在 DataParallel 前按 GPU 串行预热 QR。多卡分片把较小余数分片分给主卡。Orange D2STGNN 使用两次 32 样本微前向，按全批有效观测点加权梯度，仍只执行一次 curriculum 更新、一次梯度裁剪和一次优化器更新，有效 batch 保持 64。pristine 上游目录、正式数据、优化器、epoch、scheduler、early stopping 与指标口径均未修改。

真实远端探针全部通过。Alameda D2STGNN 四卡峰值约为每卡 9.9 GiB，Alameda IGSTGNN 四卡约为 7.6 GiB。Orange DGCRN 五卡约为 5.9 GiB，Orange D2STGNN 两微批五卡最高约为 13.5 GiB，Orange IGSTGNN 五卡最高约为 10.0 GiB。STGCN 修正后的 GSO 全为有限值，单批训练峰值约为 1.1 GiB。本地全仓回归为 446 passed 和 2 个既有 warning。两台 Linux 训练相关门各为 158 passed、12 个平台跳过和 1 个既有 warning。完整 Linux 全复现测试中的 macOS sandbox 用例及 NumPy 2 专用构造用例不属于 Linux 训练门。

正式恢复控制器于 10:57 启动。四卡机 PID 为 `39176`，当前从 task005 Alameda STGCN 开始。五卡机 PID 为 `40326`，当前从 task031 Orange STGCN 开始。恢复前 21 项的五文件快照已经落盘，四卡机 65 个文件清单 SHA-256 为 `3e89aac6c8b5ebbf2bc8d2df4cc3f518587418464d9a49fd8b349e93967f4513`，五卡机 40 个文件清单 SHA-256 为 `ee52c8e11ba150c3d49b65cf8d0efe480e46df0f5134ff310b3de14a440c1c61`。恢复 runner SHA-256 为 `52829dd65362ef03127f391b7054aff1c0feeacf011bc4af974e2be408f07a76`，持久控制脚本 SHA-256 为 `23980df3dfe42afe952bc52febb97e3fd85e1c1629b86e70b2b2679b3c8a5b4b`。完整进度见 `docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md`。

两台主机继续执行自动收尾。只有 source-host postflight、严格 allowlist 上传、结果服务器逐文件 SHA-256、精确目录树和原子提升全部通过后，源主机才调用 `/usr/bin/shutdown -h now`。任一步失败时主机保持开机。结果服务器目前仍没有正式 Table 4 主机目录，不能写成已上传或已完成。

截至 2026-07-29 17:05 新加坡时间，原 39 项矩阵已正式落盘并由任务 runner 验收 34/39。四卡机为 25/26，仅余 task021 Contra Costa STTN。五卡机为 9/13，正在执行 task036 Orange DGCRN，之后仍需 task037、task039 与 task034。结果服务器仍没有正式主机目录。两台主机的原矩阵关机都由安全哨兵延期，原结果完成 source-host postflight、上传和结果服务器复验后会保持开机，随后进入 NDR-Net 扩展。

NDR-Net Table 4 扩展使用独立矩阵 `igstgnn_table4_ndr_extension_9_v1`。它不修改原 39 项清单。扩展共 9 项，为 NDR-Net 在 Alameda、Contra Costa、Orange 三地区和种子 42、1、2 的组合。扩展清单 SHA-256 为 `2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4`，配置 SHA-256 为 `6ca46be0fb76543a61a8bd32996c7598ac674bbe239b701d1a92481f63be9422`，九卡分片计划 SHA-256 为 `f20be0db02fbd04732640cd5b024f7012ce6a16ea155d6a90c0962839f857522`。claim scope 固定为 `same_protocol_project_method_extension`，validity label 仍为 `official_protocol_with_known_overlap`。

扩展训练在模型、损失和评测前使用绑定的 `incident_stats.npz` 把官方标准化 flow 还原为原始流量单位。周内时段中位数基线只从官方 train split 的 x 与 y 位置拟合。验证 MAE 按所有有效观测点统一累计。结果继续使用原 Table 4 的 H3、H6、H12 和 12 步平均指标，并写入独立五文件 artifact。Mac 真实 Contra Costa 数据适配与 MPS 前向反向探针通过，全仓为 455 passed 和 2 个既有 warning。两台 4090 D 主机的扩展专项门各为 33 passed。

扩展实现提交为 `cd14b90`、`d67ba1c` 与 `085e68d`。扩展源部署在独立目录 `/root/autodl-tmp/table4/ndr_repo`，不会改写仍在运行的原矩阵 checkout。两台延期控制器第一次因虚拟环境 Python 是正常符号链接而被过严安全门拒绝，没有训练、结果或关机动作。修复后四卡机与五卡机控制器 PID 分别为 `447869` 与 `410737`，均已进入等待原矩阵正式回传完成的状态。每台主机在启动扩展正式分片前还会执行该主机最大地区的一次真实 CUDA 前向反向探针。扩展只有在独立 postflight、结果服务器逐文件复验和原子提升全部成功后才移除延期哨兵并关机。

截至 2026-07-29 17:25 新加坡时间，原矩阵 runner 已验收 36/39。四卡机为 26/26，source-host postflight 验证 134 个文件，files digest 为 `a43a4023cb7460fcd2ad2d2cece0adc01c7394588df206461744e67ec94550dd`。首次 finalizer 因缺少 BatchMode 回传密钥在传输前停止，没有改写结果。专用 Ed25519 密钥部署并验证后，四卡机于 17:19 完成结果服务器逐文件复验和原子提升，关机由 NDR 哨兵延期。提交 `ce7a8c2` 把同一回传身份设为原 finalizer 的默认值，双机脚本 SHA-256 已对齐。五卡机为 10/13，正在执行 task037 Orange D2STGNN，首个落盘 epoch 的验证 MAE 为 21.0824，之后仍需 task039 与 task034。

四卡机的 NDR 首次真实 CUDA 探针暴露 `scripts/train_staeformer_deseason.py` 顶层导入了 NDR 不需要的 `h5py`。该次探针在 trainer 和正式目录建立前失败。提交 `77d8815` 把只服务旧入口的数据依赖移到 `main()` 内，并新增无 `h5py` 导入回归。本地全仓为 456 passed 和 2 个既有 warning，双机真实导入门均通过。四卡机重启后的 Contra Costa 探针输出为 `16×496×12`，loss 为 `37.95976257`，CUDA 峰值为 8,128,185,856 bytes。17:24 探针通过，PID `456092` 启动四个正式 shard runner。17:25 四张卡利用率为 93% 到 97%，显存为 8,644 MiB 到 9,416 MiB，ndr004 已落盘 epoch 1，验证 MAE 为 12.62008389。五卡机 NDR 控制器 PID `422803` 继续等待原 Orange 分片完成和正式回传。

2026-07-30 00:08 新加坡时间起，NDR 扩展改为四卡机单主机重分配。六项 Alameda 与 Contra Costa 结果已完成并保持只读，Orange 三种子转移到同一四卡机的 GPU 0、1、2 并行训练。固定计划为 `igstgnn_table4_ndr_extension_4gpu_rebalance_v2`，SHA-256 为 `56a71fe2a3b8101ae282dc6f9b9efab07874108a40922276285206e99cbf2612`。该计划仍对应同一个 9 项扩展矩阵，不修改原 39 项矩阵，也不改变 claim scope 与 validity label。

Orange 原始 batch 16 的首次单卡真实探针在 backward 因 24 GB 显存不足而 fail closed，没有 trainer 或正式结果。提交 `87526d1` 实现两次 max microbatch 8 前向、全批 loss 分母、一次梯度裁剪、一次 optimizer step 和一次 scheduler step，保持有效 batch 16。梯度等价回归已通过。修复后的 Orange 探针 loss 为 `42.45340157`，峰值显存为 12,632,052,224 bytes。正式控制器 PID 为 `3677`。00:18 的落盘进度为三个任务均完成 epoch 1，validation MAE 分别为 13.19330061、13.06060743、12.91147725，三张卡利用率为 98% 到 99%。

四卡机完成控制器 PID `5614` 会等待 9 项完整后先复验旧六项 30 个 artifact 的锁定哈希，再运行 9 项 postflight、独立回传、结果服务器逐文件复验和原子提升，最后关机。五卡机旧 NDR 等待器 PID `422803` 已安全停止，incident 清单摘要为 `556c4289e2b4ec41b35b40d1f9a72190e1ad1c3c5a55edf1986843a70aa6e8a1`。五卡机只继续原矩阵。00:18 时原矩阵为 37/39，task039 Orange IGSTGNN 已完成 44/100 epoch，之后只余 task034 Orange STTN。新关机等待器 PID `201649` 只在五卡机原结果完成回传并在结果服务器重新复验后移除延期哨兵和关机。控制脚本提交为 `dd2d087` 与 `b5d9751`。任一训练、哈希、postflight、传输或结果服务器复验失败时，对应主机保持开机。

此后 NDR 扩展已完成 9/9，并在结果服务器完成逐文件复验与正式提升。正式目录含 49 个文件和 940,488,092 字节，files digest 为 `ab21c8e300ba930e5611cb9c07ff5b047eb03d281ca669c756242707231c87e7`。四卡机随后按原控制链关机。五卡机的 task039 Orange IGSTGNN 也已完整落盘，Average MAE、RMSE、MAPE 为 13.285545、23.504318、14.457104，summary SHA-256 为 `ab0288a4b8fe5a17c33b43b758bbc5851b0e63cc0cce7456a07eb5fe451ce998`。原矩阵由此达到 38/39。

五卡机上的 task034 Orange STTN 在首个单卡训练步尝试额外分配 11.22 GiB 时 OOM，没有正式 task034 目录。用户要求关机后，控制层先确认 trainer 为 0、既有 12 项完整保留并生成 60 个 artifact 的关机前清单，再正常关闭五卡机。该主机尚未执行原分片 postflight 和正式回传，其 12 项结果仍保存在持久盘。

用户重新开启四卡机并批准由四卡机接管 task034。独立恢复计划为 `igstgnn_table4_task034_4gpu_recovery_v1`，SHA-256 为 `022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5`。执行布局固定标记为 `cross_host_single_task_recovery`，不得写成原五卡机同机结果。提交 `4935b9b` 增加 STTN 四卡整批适配、完整优化步探针、独立 postflight 和严格回传控制链。本地全仓为 465 passed 和 2 个既有 warning，四卡机专项门为 13 passed。

2026-07-30 09:21:02 新加坡时间，四卡机控制器 PID `2928` 启动。重启后的 GPU UUID 集合已锁入恢复计划。真实 batch 64 探针完成前向、反向、梯度裁剪和一次优化器更新，loss 为 `170.42190551757812`，四张卡峰值分配显存为 16,082,649,088、16,060,899,840、16,060,899,840 与 16,060,899,840 bytes。探针于 09:21:28 通过后启动正式 task034。首个落盘 epoch 的训练 loss 为 29.8492，验证 loss 为 21.1960，训练和验证时间为 42.5066 秒与 4.5379 秒。只有 task034 五文件验收、独立 postflight、结果服务器逐文件复验和原子提升全部通过后，四卡机才自动关机。

用户随后重新开启五卡机，提交 `918af63` 增加只读 12 项导出计划、严格 postflight 与回传控制器。本地全仓为 469 passed 和 2 个既有 warning，五卡机专项门为 4 passed。固定导出计划 `igstgnn_table4_host5_partial_export_12_v1` 的 SHA-256 为 `cd81652493690d1d329f5eecd45fe541c9de7db26c93650fc490a2507616845d`，只允许 12 个已完成任务和 60 个正式 artifact，明确排除 task034。

五卡机 source-host postflight 两次得到同一结果，validated tasks 为 12，文件数为 60，总字节数为 1,975,159,969，files digest 为 `5c11ad4596228f08613b783d8a42ed9334e9976ce8cff4d23d09f6b79f055490`。结果服务器初始容量不足，控制器在传输前安全暂停。只清理系统归档日志和包缓存后释放 1,753,280,512 bytes，没有删除任何正式结果。重试使用 512 MiB 安全余量，正式树完成逐文件 SHA-256、精确目录和原子提升。transfer marker SHA-256 为 `53f73b3e7ae2f92535eed44fb0fc934d7be45387de579b8dfc3fcb853afd54cc`，postflight SHA-256 为 `427a21568b8739c355a5ca5851d01e622947556f8dcf9ae6572cd45dfc19c2d4`。五卡机随后自动关机，SSH 已不可达。

task034 于 2026-07-30 10:38:24 新加坡时间完成正式训练与五文件落盘。H3、H6、H12 与 Average MAE 分别为 12.652452、13.311158、14.048541 与 13.212108，Average RMSE 与 MAPE 为 23.502980 与 14.091901%。prediction SHA-256 为 `c6064688ab3bedd36b4f058ce899a97b8882645c4169979d0a6170fabf1f773a`。

首次 task034 postflight 因读取了错误的 provenance 层级而失败。提交 `a9511d5` 改为从 `factory.execution_adapter` 读取。复验通过后，finalizer 又发现单任务报告路径相对于 shard 根，而旧控制器传入外层结果根。提交 `51148f1` 修正 finalizer 源根并增加回归。两次均为控制层 fail closed，不改写任何训练 artifact，也没有重训。结果服务器容量门随后在传输前暂停，改用 512 MiB 安全余量后完成 5 个文件和 170,575,062 bytes 的逐文件复验、原子提升和正式树二次复验。

task034 正式目标为 `task034_4gpu_recovery_v1/hosts/table4_4090d4_task034`。files digest 为 `6ce435df4293dad11602378b1e5b9acf18ffde1717f5e0fdfb5922f3282e8258`，transfer marker SHA-256 为 `e393e8238b25677b55e7f1ae00813c3f0a9eb46f7a532c0eb7bba21ee4758d68`，postflight SHA-256 为 `fa8a3d0f0e0fba9d67fa569392edc83c9a74d9131b81439ea7644c7fbc700e15`。incoming 目录已不存在，五份结果逐文件复验均为 OK。四卡机随后自动关机并连续三次确认 SSH 不可达。原 39 项的任务结果层当时达到 39/39，最终有效性在该快照时仍等待三源只读 composite index。

2026-07-30，原 39 项三源只读 composite 已完成。源规范 SHA-256 为 `38645a51935618e4f12a96374d5d9695711bdf6581d984ec2baa15be63345e56`，validator SHA-256 为 `8d988a7164fb48c526ab0e7df7b1832ecc374bac108371a354a12582298b0e10`，控制提交为 `dc1f8a54919d611ac074e128d6f957c03dd7b0e6`。结果服务器于 06:52:11 UTC 启动正式复验，06:52:41 UTC 完成。状态为 `verified_official_protocol_with_known_overlap_39`，共验证 3 个来源、39 个任务、199 个文件和 3,399,778,682 字节。canonical tasks001 至 039 各出现一次，Alameda、Contra Costa、Orange 各 13 项，13 个方法各 3 项。index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。第二次独立完整只读复验得到相同任务集合、指标和 index 内容，差异仅为复验时间字段。两次过程都没有改写 summary 或 artifact。

39 项论文同结构复现表已经从 composite index 生成。`table4_reproduced.csv` SHA-256 为 `60160b443f63425c3b10cd30e29253db85a17d68e2a3cf12ac5bcf195449cacc`，逐单元论文对照表 SHA-256 为 `8f1f80ea3ec29f27efc0ddefd3d4711f06376f0655d68ec51c078016f37a0e98`，Markdown 与 LaTeX 的 SHA-256 为 `d1e39f0a9ea5e93dfff68f8f9c0e407cbe245b210a1f215cc99802b62bf3a843` 与 `fe85057ad0492674464196290aa7754d35250aeedbeffd1326fbc4eea8fafe1d`。论文报告值与复现值的 Average MAE 绝对差均值为 2.450150，复现值减论文值的范围为负 6.216278 至正 0.719624。该差异必须保留，不能把复现表写成作者报告值。

NDR 扩展的正式树也已完成独立复验。结果服务器逐文件回执 SHA-256 为 `dbd9dcb355a3324707e728a1942465612cd443695b51ea6359e80dc03d1be100`，49 文件 checksum log SHA-256 为 `d0f7092ef8b02be8b833a695009971e4f616033f2a5beb7847574c24ceab2f88`。NDR 扩展索引 SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。Alameda、Contra Costa、Orange 的三种子 Average MAE 均值与样本标准差依次为 `10.759524 ± 0.040862`、`11.413717 ± 0.024537`、`11.569331 ± 0.031664`。Orange 三项使用显存适配后的训练代码摘要 `396de4834480ca2fdf4bfb750b84958b2a1492b444e886898cfb199ca60f88d3`，其余六项为 `0c7035ca38b81fbcbfa505092c5c19aa9c99fed9040d80eaa7008de6fe3e0c1d`。两个摘要均逐任务写入 provenance，不代表协议或指标身份混合。

最终 39 行加 3 个 NDR 汇总行的 CSV、Markdown 与 LaTeX 已生成，SHA-256 分别为 `057e0fa94bbea24d834ed16ed00b94b0599a9e9081321cc205643793c47f34c1`、`1097b82ed71ec26d9657ff28cebcefd5c3455a179b3ef8ad6ba68529b7a118d8` 与 `33047e4d0e11929ae1c8c4e3121dd6aad9bb8684769713d58c3621fb961e22ce`。NDR 的 Average MAE 在 Orange 低于所有单次 baseline，在 Alameda 与 Contra Costa 略高于 BiST。该比较仅为描述性展示，因为 baseline 是单次运行，NDR 是三种子均值。当前本地全仓为 474 passed 和 2 个既有 warning。

## BiST 与 BiST-NDR 严格配对实验

2026-07-31 建立独立矩阵 `igstgnn_table4_bist_ndr_pair_18_v1`，用于判断 BiST 骨干是否能改善 NDR。矩阵为 2 个方法乘 3 个地区乘种子 42、1、2，共 18 项。矩阵 SHA-256 为 `90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab`，配置 SHA-256 为 `b763fb35b392e265a410325efa0ef34afa3b0d16c7c2e56c71bc3243844e8764`。两个方法统一使用 batch 16、30 个 epoch、相同优化器、全局验证 masked MAE 和 `partial_batch: retain`，测试观测集合严格一致。

BiST-NDR 通过外部适配器复用 BiST 编码、backcast residual 和自适应残差图传播，并增加正常残差头、扰动残差头与 NDR gate。BiST pristine checkout 没有被修改或复制。Mac 全仓回归现为 480 passed 和 2 个既有 warning，5080 Windows 专项门为 5 passed。Orange 最大地区的两个真实 CUDA 完整优化步探针均通过，`bist_aligned` 与 `bist_ndr` 的峰值分配显存分别为 `439906816` 与 `457959424` bytes。

5080 使用独立 checkout `D:\t4\bist_ndr_repo` 和结果根 `D:\t4\bist_ndr_pair\results`。第一次 SYSTEM 计划任务因子进程没有继承 ASUS 用户 site-packages 而在 NumPy 导入阶段退出，没有任务目录或正式结果。修复启动脚本与子进程 `PYTHONPATH` 后，计划任务 `XTraffic_BiST_NDR_Pair_v1` 于 2026-07-31 10:06:03 新加坡时间正式重启，执行时限为 `PT0S`。首项 `pair001_bist_aligned_alameda_seed42` 已于 10:14 完成 30 个 epoch，并通过五文件验收，best validation MAE 为 `11.69943708`。H3、H6、H12 与 Average MAE 为 `11.205525`、`11.784402`、`12.675788` 与 `11.740657`。队列为 1/18，正在执行 `pair002_bist_aligned_alameda_seed1`。单项结果不得用于报告 BiST-NDR 性能结论。完整记录见 `docs/audits/igstgnn_table4_bist_ndr_pair_launch_2026-07-31.md`。

该配对队列已于 2026-07-31 13:41:13 新加坡时间完成。runner 状态为 completed，18/18 任务全部验收，失败和 skipped 均为 0，正式任务树含 18 个目录与 90 个文件。这条实验线是统一 30 epoch 训练控制下的 BiST 与 BiST-NDR 结构配对，与下方用于填回原 Table 4 的 BiST 与原 NDR-Net 100 epoch 补充矩阵彼此独立。当前只确认 runner 完整性，尚未生成配对性能结论。

## BiST 与 NDR-Net 原生配方 100 epoch 补充对比

2026-08-01 建立独立矩阵 `igstgnn_table4_bist_ndr_native100_18_v1`，用于生成可直接填入既有 Table 4 的 BiST 与 NDR-Net 100 epoch 补充行。矩阵为两种方法乘三地区乘种子 42、1、2，共 18 项。矩阵 SHA-256 为 `e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246`，配方 SHA-256 为 `37ebcb574f516bc1b662ade0d5264deda96a20d017d20edc7a57253daab5e66b`，实现提交为 `53f4b3a`。

BiST 保留原生 batch 64、drop partial batch、Adam 0.002、MultiStepLR 和 patience 20，只把上限从 500 epoch 改为 100。NDR-Net 保留原生 batch 16、retain partial batch、Adam 0.001、CosineAnnealingLR、全局验证 masked MAE 和无 early stopping 契约，只把上限从 30 epoch 改为 100。评测列仍为 H3、H6、H12 与 Average，每组报告 MAE、RMSE 与 MAPE。

Mac 全仓回归为 489 passed 和 2 个既有 warning，5080 Windows 专项门为 25 passed。Orange 最大地区真实 CUDA 完整优化步探针均通过，BiST 与 NDR-Net 峰值分配显存为 `3064998912` 与 `12633100800` bytes。5080 于 2026-08-01 00:02:23 新加坡时间在 `D:\t4\native100_repo` 启动计划任务 `XTraffic_Table4_Native100_v1`，执行时限为 `PT0S`。启动复验时队列为 0/18，失败数为 0，正在执行 `n100001_bist_alameda_seed42`，GPU 利用率为 70%。完整记录见 `docs/audits/igstgnn_table4_native100_launch_2026-08-01.md`。

`scripts/reproduction/igstgnn_table4/native100_compare_report.py` 已锁定最终入表路径。它只在 18 项全部通过五文件复验后生成报告，保留既有 42 行，追加 6 个 BiST 100ep 与 NDR-Net 100ep 的地区行，输出同结构 CSV、Markdown、LaTeX 和 manifest。

该矩阵仍使用 `igstgnn_table4_official_v1` 与 `official_protocol_with_known_overlap` 标签。在 18 项全部落盘和验收前，不报告 100 epoch 性能结论，不覆盖既有 BiST 500 epoch 单次结果或 NDR-Net 30 epoch 三种子结果，也不得与 v3 无泄露结果混合。

2026-08-01 12:18 新加坡时间起，原生配方 100 epoch 补充矩阵改为按地区分层的双机执行。计划 `igstgnn_table4_native100_5080_5090_v1` SHA-256 为 `d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e`。5080 只选择 Alameda 与 Contra Costa 的 12 项，5090 只选择 Orange 的 6 项。两分片互斥且并集为 canonical 18，同一地区的 BiST 与 NDR-Net 保持同机。最终布局只能称为 `dual_host_region_stratified`。

5080 的旧 18 项计划任务已停止并禁用。被中断的 task013 没有正式任务目录，incident 的 `files.sha256` 摘要为 `94802ea874935de3db70eb3d5b730ec14e978662ebeada4014bc7eb09f34be23`。新计划任务 `XTraffic_Table4_Native100_5080_CC_v1` 于 12:18 启动，跳过 9 个已验证的正式任务，重新执行 task013。12:40 前 task013 已完成 epoch 17。5080 结果根内已有的 Orange BiST tasks007-009 保持只读，但必须从最终 composite 选择中排除。

5090 使用训练归档提交 `ace318b`，Orange 九个数据文件已与 Mac 逐文件校验一致，split manifest SHA-256 为 `c37b78f19766fec9ac57d28d6a7becd8a3b334261cef4e7863f29b0bbcc7b019`。BiST 与 NDR-Net 的真实 CUDA 完整优化步探针均通过，峰值分配显存为 `3142229504` 与 `12682121728` bytes。5090 持久队列于 12:38:57 启动，选择 Orange 六项，初始状态为 0/6、0 失败，当前为 task007。12:40 时 task007 已完成 epoch 6。当前 Mac 全仓为 493 passed 和 2 个既有 warning。完整重分片记录见 `docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md`。

2026-08-01 23:45，新加坡时间，用户要求 5090 在训练完成后自动关机。失败关闭守护脚本 SHA-256 为 `d0168a7f1b3b2b1b5386de1cebb52681fc1319babe8df1eddb5344228706bbb1`，远端持久 PID 为 `20225`。它只在 Orange 六项逐任务五文件复验、queue completed、0 失败、runner 退出码 0 且没有残留训练进程时调用 `/usr/bin/shutdown -h now`。任何异常都会保持机器开机。守护启动时 5090 为 4/6 完成，task017 已到 epoch 82。

5080 分片已于 2026-08-01 15:52:38 新加坡时间完成，声明内 12/12 全部通过 runner 验收，失败数为 0。用户报告随后发生断电。23:37 局域网 SSH 恢复后，completed queue、12 个已验证任务和空闲 GPU 均已确认，断电没有造成训练中断或重跑。结合 5090 的 4/6，完整矩阵当时为 16/18 已验收。

## v2 冻结记录

2026-07-19 曾从提交 `bf3045ac` 构建 v2 三地区缓存。事件样本 train、val、test 数分别为 Alameda 43973、9450、9366，ContraCosta 19159、4026、4040，Orange 75395、15727、14368。full-window 三地区曾记录为 73561、10501、21013，远端曾有 28 项测试通过。45 项队列于 00:39 启动，启动判活时 GPU 利用率为 91%，显存使用约 6.0 GiB。

这些数字只用于追溯 v2 运维历史。它们不能证明 v2 数据和指标有效，也不能写入论文结果。

## 下一步顺序

1. 将已验证的 Table 4 复现表作为复现结果使用，同时保留公开材料不完整和官方协议已知重叠的限制。
2. 如果需要对 NDR 与 baseline 做统计推断，应为选定 baseline 补跑相同的种子 42、1、2。当前 39 项单次运行只能支持描述性比较。
3. 只根据完整的 v3 落盘结果决定原项目论文主张。Table 4 复现结果与 v3 无泄露结论必须分开解释。

## 最小判别矩阵

- 三地区 STAEformer s0a 与 s0c
- Alameda 与 ContraCosta 的 s0f 与 s0g
- 三地区 ICSF A/B
- 三地区 full-window GWN、FDN 与 STAEformer

## 实验纪律

- 所有汇报数字必须来自已落盘文件，严禁估算填充。
- XTraffic 旧缓存与旧 summary 只读冻结。
- v2 缓存和最先完成的 4 个 v2 远程结果同样只读冻结，并标记为 `contaminated preliminary`。
- 当前双机分层矩阵的 runner、正式分片、composite 与统计分析均已完成。composite 完成不自动恢复论文结论，是否恢复由用户根据落盘分析决定。
- IGSTGNN Table 4 复现只能称为按公开材料最大忠实复现。缺少作者未公开的 baseline XTraffic adapter、逐地区配置、checkpoint 与运行日志时，不得声称逐命令完全复现。
- 不得把 Windows 5080 与 Linux 5090 的 `config_hash`、绝对路径和运行时 provenance 混成同机结果。
- 新结果必须记录协议、split、normalizer 与 climatology 拟合范围、缓存哈希、Git SHA、dirty 状态和 schema version。
- 等参数比较优先，多种子约定为 42、1、2。
- 指标使用 masked MAE，并按需要报告 all、affected 与 unaffected。
- affected 只能作为事后分组指标，除非实验明确声明使用标签监督。
- 负结果如实保留，不为追求正结果反复改停止线。
- 报告类正文不用破折号，不用括号内插说明。
- 代码注释保持最少，commit 信息使用英文。

## 仓库维护

- `scripts/README.md` 是脚本分类入口。
- `outputs/README.md` 说明历史结果和本地产物的关系。
- `docs/archive/` 保存被主报告吸收的旧文档。
- 不批量移动仍被历史入口引用的脚本或结果，避免破坏导入与路径。根目录只保留项目级文档和配置。
- 本地原始数据位于根目录 `archive/`，不进入 Git。
- 远程运行包和日志放到被忽略的 `outputs/remote_runs/`。

## 凭据与远程运行

实验室共享主机、地址和密码集中明文维护在 `config/lab_hosts.env`。运行脚本可以直接 `source` 该文件，也允许环境变量覆盖。私钥、个人令牌和个人账号凭据仍不进入仓库。
