# XTraffic 项目导航与协作约定

最近更新：2026-07-28

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

正式 v2 矩阵清单位于 `config/reproduction/igstgnn_table4/matrix.public_material_v2.json`，SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`。LargeST、BiST、IGSTGNN 上游提交分别锁定为 `b7c6ff5f7f0c10051ed10a319b2cfe58021312f9`、`dd94adf7721fcbb9e3feb5d1b44040305199a4cc`、`64c6e243466f791f1b75e50421d63e39577ec9fa`。BiST 三地区实际随机 seed 已分别冻结为 676615、683452、150462。5080 上的本轮 Table 4 训练部署基于提交 `5aadfff`。Mac 复现测试为 179 项通过与 1 个既有 warning，5080 的 Windows 训练相关测试为 127 项通过、2 项平台跳过、1 项仅用于模拟不可在原生 Windows 执行的竞态测试被排除。

正式数据与三个 pristine checkout 已部署到 5080 的 `D:\t4`。Alameda HL 与 LSTM 已完成并通过五文件复验，测试均按 LargeST 源码实际消费 2,176 条，丢弃不足一个 batch 的 28 条尾部样本。HL 的 H3、H6、H12 与 Average MAE 分别为 15.916486、19.056049、25.661367、19.649600，prediction SHA-256 为 `63482403b21e74577e2c74cd3de198bdfafa8c90e8692843a506adcf2614e6b0`。LSTM 的对应 MAE 为 12.825795、14.208229、17.033611、14.355326，prediction SHA-256 为 `8f70ed5a8264c007e2baf5453bf18d0d621ad0502ad29e39ea34f83315b0e769`。task003 Alameda DCRNN 已于 22:19:57 新加坡时间启动，正式结果根为 `D:\t4\results\igstgnn_table4_public_repro_39_v2`。

## v2 冻结记录

2026-07-19 曾从提交 `bf3045ac` 构建 v2 三地区缓存。事件样本 train、val、test 数分别为 Alameda 43973、9450、9366，ContraCosta 19159、4026、4040，Orange 75395、15727、14368。full-window 三地区曾记录为 73561、10501、21013，远端曾有 28 项测试通过。45 项队列于 00:39 启动，启动判活时 GPU 利用率为 91%，显存使用约 6.0 GiB。

这些数字只用于追溯 v2 运维历史。它们不能证明 v2 数据和指标有效，也不能写入论文结果。

## 下一步顺序

1. 完成 Alameda 的 DCRNN 与 IGSTGNN 正式 pilot，每项都必须通过五文件复验。
2. pilot 通过后从同一不可变清单启动完整 39 项单卡顺序队列，已有正式 pilot 仅在重新验收通过后跳过。
3. 生成与论文同结构的 Table 4、逐单元差值表和数字追溯索引。
4. 只根据完整的 v3 落盘结果决定原项目论文主张。Table 4 复现结果与 v3 无泄露结论必须分开解释。

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
