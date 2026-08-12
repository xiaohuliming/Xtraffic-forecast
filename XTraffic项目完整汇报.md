# XTraffic 交通流量预测研究：完整汇报

初版日期：2026-06-17，持续更新，最近更新 2026-08-12
作者：科研实习项目组
说明：本文是本项目研究历史、实验数字与出处的唯一权威长报告。原《项目总结报告_2026-06-10》与《去季节化方向_交接_2026-06-26》已并入本文并归档到 docs/archive，不再单独维护。本文中所有数字均来自已落盘的实验文件，包括 test_predictions.npz、summary.json、训练日志与诊断文件，无任何估算或记忆填充。数字来自落盘文件不等于实验协议无误。2026-07-18 数据有效性审计发现旧 XTraffic 管线存在未来信息泄露，因此审计前数字均保留为历史记录，当前是否可用于论文以 `docs/audits/result_validity.csv` 为准。按全局写作规范，正文不使用破折号与括号内插说明。

## 2026-07-18 数据有效性审计

论文写作暂停。旧版 XTraffic 缓存构建器对全年序列做双向线性插值，导致历史输入读取未来观测。RADS 气候态在拟合前使用事故记录排除训练数据，严格 label-free 表述不成立。部分事故模型还读取由事故后真实流量生成的 severity、affected 或最终 duration。事件协议与 full-window 协议存在边界目标重叠，多个训练脚本的统计范围又越过训练边界。

因此，本文下方所有审计前 XTraffic 绝对数值与配对结论均标记为历史结果。它们不能直接进入论文，也不能被解释为泄露已经在同窗比较中自动抵消。旧输出只读保留，当前正式无泄露复跑使用独立的 `protocol_v3_noleak_dedup` 缓存与结果目录。

完整代码证据、三个区域的落盘量化、受影响实验清单和修复验收条件见 `docs/audits/data_leakage_2026-07-18.md`。机器可读状态见 `docs/audits/result_validity.csv`。

## 2026-07-18 无泄露 v2 代码修复完成

代码层修复已经完成。共享协议 `dist_net/protocol.py` 统一实现因果前向填充、完整目标窗切分、event group 隔离、full-window 目标互斥、train-only 拟合范围和可部署事故特征。正式构建入口 `scripts/build_full_county_cache.py` 不再使用事故记录拟合气候态，不再把最终 duration 与 severity 写入推理输入，并将 affected 物理移入分析标签区域。`dist_net/data.py` 默认拒绝旧协议缓存，主要训练入口默认读取 `region_data_v2_noleak` 并写入独立的 `protocol_v2_noleak` 结果目录。ICSF、统计上界和 IGSTGNN 的预测时刻与锚定事故对齐也已修复。

v2 缓存会记录 schema version、协议版本、精确切分边界、预处理拟合范围、原始文件 SHA-256、构建代码 Git SHA、分支与 dirty 状态。数据回归测试覆盖未来数据不改变历史输入、split 目标集合互斥、同一事件不跨 split、事后事故字段不改变可部署输入、主事故在截断后仍可定位、窗口语义对齐和旧缓存拒绝。当前完整测试结果为 28 项通过，Python 编译、Shell 语法和补丁格式检查均通过。

## 2026-07-19 无泄露 v2 正式缓存验收与后续复核

已在 Mac 上用完整原始数据生成三个区域的正式 v2 H5 与事故关系缓存，并传到 5080。缓存 provenance 指向 main 的 `bf3045ac`，构建工作区为 clean。Alameda 的事件样本 train、val、test 数为 43973、9450、9366。ContraCosta 为 19159、4026、4040。Orange 为 75395、15727、14368。三个区域的 event group 与目标时刻集合均两两无交集，主事故槽位逐样本精确对应。full-window 三个区域均为 73561、10501、21013，三个 split 的目标时刻集合两两无交集。

远端复核包括传输包与缓存逐级 SHA-256、三地区 full-window loader、缓存 schema 与 provenance、可部署事故字段，以及完整 28 项测试。全部通过。旧缓存和旧结果未覆盖。

5080 的 Tailscale 地址在线。旧 contaminated baseline zoo 队列已经停止并保留日志。45 项 v2 最小判别队列于 2026-07-19 00:39 启动，严格串行。队列覆盖三地区 STAEformer s0a 与 s0c，Alameda 与 ContraCosta 的 s0f 与 s0g，三地区 STAEformer ICSF 对照，full-window 三模型九宫格，以及核心方法的 seeds 42、1、2。

后续全面复核发现，v2 仍会让同一预测时间因多个事故重复进入样本集，且简单保留首个来源并不等价于选择最近锚点。验证 MAE 还存在按 batch 或样本均值再次平均的问题。进一步检查又发现活跃事故身份可能跨 split、截断后的事故槽位没有覆盖完整 affected 并集、IGSTGNN 转换会丢失目标缺失掩码，以及部分图模型反向邻接归一化、blind 模式和远程结果完整性仍需收紧。因此 v2 不能作为最终无泄露协议。

停止队列时已有四个远程 summary 落盘。它们只作为 v2 调试痕迹保留，统一降级为 `preliminary contaminated`，不得进入论文或用于恢复主张。

| 任务 | v2 masked MAE | 当前状态 |
|---|---:|---|
| Alameda s0a seed 42 | 11.431783 | preliminary contaminated |
| Alameda s0c seed 42 | 11.193208 | preliminary contaminated |
| ContraCosta s0a seed 42 | 12.036264 | preliminary contaminated |
| ContraCosta s0c seed 42 | 11.822266 | preliminary contaminated |

第 5 项 Orange s0a seed 42 没有形成可用 summary，队列在全面协议复核时停止。已有日志与不完整 checkpoint 只读保留，不视为结果。

## 2026-07-19 v3 全面修复阶段

全面审查已经覆盖数据构建、事件去重、切分隔离、掩码传播、验证指标聚合、图支持矩阵、训练输入边界、输出身份校验和远程队列恢复策略。v3 将同一预测时刻规范化为唯一样本，按距预测时刻最近的事故锚点选择主事故，并用稳定事件身份处理并列情况。所有活跃事故参与 split 污染检查与 affected 并集计算，模型可见的事故槽位截断不再改变分析标签。验证 MAE 统一按全部有效观测点聚合，IGSTGNN 的 `x_mask` 与 `y_mask` 从转换、加载、训练到比较全链路保留。正式训练默认拒绝缺少身份、哈希、掩码或结果清单的产物。

新协议为 `protocol_v3_noleak_dedup`，缓存 schema 为 3。默认缓存目录是 `outputs/dist_net/region_data_v3_noleak_dedup`，默认结果根目录是 `outputs/protocol_v3_noleak_dedup`。正式缓存构建代码 SHA 为 `29159f626494de52c69b36fe5bba9854f13cb56c`。最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。Mac、RTX 5080 与 RTX 5090 的完整回归测试结果均为 143 项通过，另有 1 条既有张量转标量警告，没有测试失败。

组员 Fu 在 `origin/fu` 的 `625aacf` 提出了空 mask batch 可跳过的处理思路。v3 已通过共享 `MaskedMAEAccumulator` 与远程 runner 的全预测有限性检查吸收这一正确性要求，没有直接 cherry-pick 落后于当前协议的旧分支。

截至本次更新，三地区正式 v3 缓存已经落盘到 `outputs/dist_net/region_data_v3_noleak_dedup`。缓存先在 staging 路径完成严格审计，发布到正式路径后再次执行同一审计，两次结果均为 verified。缓存传到 5080 后完成一次只读严格审计，传到 5090 后又完成一次只读严格审计，两台远端机器的结果也均为 verified。cache bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。5080 传输归档大小为 1,896,581,632 字节，SHA-256 为 `6c5d3ccb53af30d6d90c1032453c83cb463711456839dad151ba676e2bb80ea5`。

正式 v3 事件样本与 train、val、test 切分如下。Alameda 共 45760 个样本，切分为 32001、6954、6805。ContraCosta 共 23337 个样本，切分为 16414、3472、3451。Orange 共 60015 个样本，切分为 42372、9098、8545。

45 项 v3 最小判别矩阵于 2026-07-19 20:18 新加坡时间在 RTX 5080 上严格串行启动。远端 worktree clean，代码 SHA、bundle ID、bundle digest 与 45 项 manifest 均由 runner 锁定。2026-07-20 完成首批结果验收后，部分训练负载先迁移到 5090，随后改为固定双机分片。详细状态见下一节。

## 2026-07-20 v3 首批验收结果与双机分片训练

截至 2026-07-20 22:23 新加坡时间，5080 已完成任务 001 到 004，5090 已完成任务 005 和 006。六项均由 runner 从 prediction artifact 重新计算全部指标，并核对 summary、代码 SHA、缓存 bundle 与任务身份。六项均使用 seed 42。

| 主机 | v3 任务 | all masked MAE | affected masked MAE | unaffected masked MAE | 状态 |
|---|---|---:|---:|---:|---|
| 5080 | `001_event_staeformer_s0a_alameda_seed42` | 11.3395311389686 | 17.139916776471978 | 11.005997748058482 | runner validated |
| 5080 | `002_event_staeformer_s0c_alameda_seed42` | 11.02520006450289 | 16.58355499420212 | 10.705583907329194 | runner validated |
| 5080 | `003_event_staeformer_s0a_contracosta_seed42` | 12.066376961556534 | 18.344278999946876 | 11.771324191231155 | runner validated |
| 5080 | `004_event_staeformer_s0c_contracosta_seed42` | 11.718486203953718 | 17.78479960645264 | 11.433377808804519 | runner validated |
| 5090 | `005_event_staeformer_s0a_orange_seed42` | 12.025649762966154 | 17.932862345879170 | 11.720225703973032 | runner validated |
| 5090 | `006_event_staeformer_s0c_orange_seed42` | 11.740712128157245 | 17.583309336354100 | 11.438628915585708 | runner validated |

这六项是当前协议下有效的单项结果，与上方四项 `preliminary contaminated` 的 v2 调试结果完全分开。Alameda、ContraCosta 和 Orange 的 s0c 比 s0a 的 all 指标分别低 0.314331、0.347891 和 0.284938，affected 与 unaffected 指标方向一致。由于这六项均为 seed 42，这些差值只能作为方向性观察，不能恢复多种子结论、显著性结论或论文主张。

5080 开始运行第 5 项 `005_event_staeformer_s0a_orange_seed42` 后，Orange 任务在 16 GB 显存配置上极慢。为将负载迁移到更合适的机器，项目主动终止该子训练进程。runner 如实记录退出码 1。该退出码是操作性迁移记录，不是模型失败，也没有产生可验收 summary。任务 001 到 004 的已验收结果及日志继续保留。

新接入节点实际为 32 GB NVIDIA GeForce RTX 5090。该节点已建立最终训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a` 的 clean checkout，完整回归测试为 143 项通过与 1 条既有警告，正式缓存只读严格审计为 verified。旧的 41 项队列完成任务 005 和 006 后，在任务边界被安全终止。边界切换的同一秒，旧 runner 曾为任务 007 写入 running state、零字节日志和未完成 staging，但没有产生 summary 或正式任务目录。这些残留不属于任何正式分片，source-host postflight 和上传 allowlist 均明确排除。

2026-07-20 22:23 新加坡时间起，正式执行改为双机分层矩阵。5080 固定执行 tasks 001 至 004、007 至 010、013 至 016、019 至 030，共 24 项。5090 固定执行 tasks 005 至 006、011 至 012、017 至 018、031 至 045，共 21 项。两组任务互不交叉，并集严格为 canonical 45。每个直接 A/B 比较组完整保留在同一台机器。在分片启动时，5080 runner 重新验收并跳过 tasks 001 至 004，队列快照为 4/24，正在训练 task 007。5090 runner 重新验收并跳过 tasks 005 和 006，队列快照为 2/21，正在训练 task 011。

5080 的 task 007 `attempt_0001` 在完成 epoch 8 后，于 epoch 9 数据装载阶段触发 NumPy `ArrayMemoryError`。两条遗留诊断 PowerShell 扫描当时占用约 48 GB 私有内存，属于主机运维故障。该 attempt 没有产生 summary 或正式任务目录，失败记录原样保留。只停止这两条诊断进程后内存恢复，runner 以 `attempt_0002` 重启 task 007，队列仍为 4/24、skipped valid 为 4，训练和 GPU 均恢复正常。该记录不是模型失败、数据协议失败或训练数值失败。

在 2026-07-21 00:02 新加坡时间的历史快照中，5080 的 task 007 `attempt_0002` 已完成 epoch 7，train 为 12.0424，val 为 12.6845，用时 359 秒，GPU 利用率为 92%。5090 的 task 011 已完成 epoch 8，train 为 12.3250，val 为 12.3937，用时 659 秒，并保存当前 best checkpoint。两台队列当时分别为 4/24 与 2/21。

截至 2026-07-23 13:50 新加坡时间，runner 层共完成或重新验收 42/45。5090 的新 18 项分片已完成 18/18，skipped valid 为 15，并通过 source-host postflight、严格导出、结果服务器逐文件哈希与原子提升。结果服务器上的 `rtx5090_18` 状态为 `verified`，包含 97 个源文件、10,071,444,446 字节源数据和 18 个正式任务目录。postflight 验证时间为 2026-07-23 12:40:34 新加坡时间。5090 随后离线，与成功回传后自动关机契约一致。

5080 的 task040 已完成 30 个 epoch 与测试推理，trainer 退出码为 0，但 runner 在原子提升前拒绝了结果。summary 的 all MAE 为 11.305081367492676，runner 从同一 prediction artifact 以 float64 重算为 11.305101714317631，差值为 0.000020346824955552734。queue 因此为 failed 24/27，runner 退出码为 4，task040 没有正式任务目录，tasks041 至 042 未启动。有效点数 103,270,922、artifact 身份、掩码和有限性均一致。summary 数值可由 producer 的 float32 平均精确复现，因此当前诊断是指标精度契约不一致，不是数据泄露、模型训练失败或 artifact 损坏。

Windows 5080 与 Linux 5090 的绝对路径和运行时环境不同，因此两台机器保留独立结果根、独立 `config_hash`、独立运行时 provenance 和独立 source-host postflight。最终结果只能称为 `host-stratified distributed 45`。只读 composite index 会验证两个分片互斥并完整覆盖 45 项，但不会复制任务目录、改写 summary 或把跨主机结果伪装成同机矩阵。论文写作继续暂停，只有完整矩阵与多种子比较完成后才重新评估原主张。

### 自动回传与关机安排

5090 的 `rtx5090_18` 已完整回传。transfer marker 记录 `validated_tasks=18`、`actual_skipped_valid_tasks=15` 和 `status=verified`。marker 的 file list、manifest、postflight 与文件树 digest 均已落盘。5090 SSH 当前不可达，这是成功收尾后关机的预期状态。

5080 的 runner 与 finalizer 已在 task040 校验失败后安全退出，GPU 没有训练计算进程。task040 的 staging 仍完整保留，其中 summary 为 7,895 字节，checkpoint 为 1,306,955 字节，prediction artifact 为 483,524,632 字节，train log 为 3,420 字节。关机前写入 `C:\Users\ASUS\protocol_v3_5080_shutdown_state.json`。用户要求关机后，5080 已执行 Windows 正常关机，并于 2026-07-23 13:50 新加坡时间通过 SSH 超时与 ping 不可达确认离线。

重分片控制代码已经过提交 `60405a9`、`e191c27` 和 `30bcb81` 固化。本地全仓回归为 230 项通过和 1 条既有警告，5080 真实 PowerShell AST 解析与退出码冒烟验证均通过。接收完整 `rtx5090_18` 后，结果服务器当前空闲 9,322,172,416 字节。`rtx5080_27` 尚不存在。当时下一步必须先审查控制层 float32 producer 与 float64 runner 的兼容校验和 staging recovery，不能把 task040 写成正式结果，也不能在未审查恢复路径前盲目重跑。

---

## 第一部分：一页速览

本部分记录 2026-07-18 审计前的研究叙事，用于追溯项目演进。当前结论状态以上方审计说明为准。

### 我们在做什么

研究 XTraffic 这个大型交通数据集上，事故信息能否增强交通流量预测。XTraffic 是 2024 年发布的、目前最大的带事故标注的公开交通数据集，由北京航空航天大学团队构建，发布动机就是支持事故感知的交通预测。

### 审计前我们得到了什么结论

一句话：在 XTraffic 上，事故标签对流量预测没有增益。任何不看标签的模型，无论是我们自己的轻量模型 FDN，还是社区公认的标准 SOTA 模型 STAEformer，都不输甚至明显优于使用全套事故标签的事故感知模型 IGSTGNN。

更进一步，我们能主动纠正文献：对标的 KDD'26 论文 IGSTGNN 之所以显得是 SOTA，是因为它的对比基线过弱，且其官方代码存在一个会无条件触发的训练 bug。我们用一个公认的标准 label-free 模型直接超过了它声称的 SOTA 成绩。

### 审计前的论文定位

从最初设想的方法论文，最终确定为应用与复现性研究论文。核心命题是事故标签无增益，证据链包括标准 SOTA 模型 label-free 超越事故感知模型、官方代码 bug 取证、事故模块即插即用主张复现失败、多种架构尝试系统性无增益、显著性检验与多种子背书。对标 IGSTGNN 本身能中 KDD，我们处在同一问题赛道，证据更完整，结论更反直觉。

历史方法层级固定为 `RADS ⊂ RADS-O ⊂ NDR-Net`。RADS 是状态自适应锚点重组机制，RADS-O 在其上加入逐 horizon 门控与 oracle 蒸馏，NDR-Net 再加入双残差头。该层级是代码与消融关系，不代表审计前性能结论已经在 v3 上复现。

### 最新进展，2026-07-18，发现数据泄露并完成 v2 代码修复

代码与落盘 H5 审计确认，旧数据管线存在未来插值、label-informed 气候态、事后事故特征、切分目标重叠与统计范围越界。所有使用旧 XTraffic 缓存的模型都受影响。标准基准 METR-LA、PEMS04 与 PEMS08 不经过该缓存，不受本次审计影响。项目从论文写作切回数据修复与最小判别矩阵复跑，旧结果保留但不再视为已关账证据。

同日完成无泄露 v2 的代码实现与本地回归验证。正式数据入口、数据集加载器、主要训练脚本、ICSF 和 IGSTGNN 转换器已经切换到版本化协议与独立输出目录。仓库根目录的重复训练脚本已经清理，脚本、文档、远程启动文件和本地产物均有明确目录与索引。当时的阻塞从代码修复转为正式缓存生成和实验复跑。

### 最新进展，2026-07-19，v2 队列停止并进入 v3 全面修复

三地区正式 v2 缓存和事故关系缓存已经生成。事件协议与 full-window 协议的目标时刻集合均完成互斥验收，event group 不跨 split，可部署事故输入不含最终 duration、severity 与 affected，provenance 指向干净的 main 提交 `bf3045ac`。远端 28 项测试全部通过。

45 项最小判别队列曾在 5080 上串行启动。全面复核随后确认 v2 仍存在同一预测时刻重复采样、主事故选择不规范和验证指标聚合不精确等问题，队列已经停止。已完成的四个 summary 统一标记为 `preliminary contaminated`，第 5 项没有形成可用 summary。

v3 已完成代码修复，schema 为 3，协议名为 `protocol_v3_noleak_dedup`。正式缓存构建代码 SHA 是 `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 是 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。截至 7 月 19 日，Mac 与 5080 均为 143 项通过与 1 条既有警告。三地区正式缓存已经落盘，本地两次及 5080 一次严格审计均为 verified。45 项队列于 20:18 启动。截至 21:55，首项已完成 epoch 16，队列完成数仍为 0/45，当时尚无完成并验收的新结果。

### 最新进展，2026-07-20，六项 v3 结果验收并启动双机分片

5080 已完成 tasks 001 至 004，5090 已完成 tasks 005 至 006。六项 prediction artifact 与 summary 均通过 runner 复算和身份验收。它们是 v3 下有效的 seed 42 单项结果，但仍不足以支持多种子结论。5080 上的第 5 项曾因迁移到更合适的硬件而主动终止，退出码 1 不是模型失败。

新接入的远端节点是 32 GB RTX 5090。clean checkout 锁定最终训练代码 SHA `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，完整测试为 143 项通过与 1 条既有警告，缓存通过第 4 次严格审计。审计次数包括本地两次、5080 一次与 5090 一次。完成 tasks 005 至 006 后，旧 41 项队列在任务边界安全停止。旧 runner 瞬间为 task 007 创建的零字节日志和未完成 staging 没有 summary，不计为结果并被导出 allowlist 排除。

正式 45 项从该时点起采用固定双机分片。5080 执行 24 项，当时为 4/24。task 007 的首个 attempt 因遗留诊断进程耗尽主机内存而失败，未产生 summary，随后用第二个 attempt 恢复训练并完成 epoch 7。5090 执行 21 项，当时为 2/21。截至 2026-07-21 00:02 新加坡时间，task 011 已完成 epoch 8。两个分片互斥且并集为 canonical 45，所有直接比较组保持同机。两台机器独立验收与上传，最终只生成只读 composite index，不将跨主机路径和运行时身份混为同机结果。

### 最新进展，2026-07-22，27/18 重分片已完成切换

截至 22:56 新加坡时间，runner 层共完成或重新验收 39/45。5080 的旧 24 项分片已全部完成，runner 退出码为 0。source-host postflight 对 24 项复验通过，严格导出 allowlist 包含 134 个文件，共 3,372,164,933 字节。task039 也已完成并通过 5090 runner 与边界 transition 的双重验收。

22:49，新加坡时间，task039 边界通过，ready marker 原子发布。旧 5090 runner、旧 finalizer、watcher 与 transition 随后按设计退出。新 5090 队列为 15/18，正在 task043。runner PID 为 241437，finalizer PID 为 241585。task043 已到 epoch 2 batch 1100/2299，GPU 利用率为 83%。

新 5080 队列为 24/27，正在 task040。runner PID 为 17260，finalizer PID 为 6152，边界 waiter 已按设计退出。task040 已到 epoch 3 batch 1200/2299，GPU 利用率为 70%。用户授权处理容量后，结果服务器仅清理未使用的 Docker build cache，可用空间增至 19,434,708,992 字节，4 个业务容器保持运行。控制提交仍为 `60405a9`、`e191c27` 和 `30bcb81`。

### 最新进展，2026-07-23，5090 分片完成，5080 安全关机

5090 已完成新分片 18/18。`rtx5090_18` 通过 source-host postflight、97 文件严格导出、10,071,444,446 字节源数据逐文件哈希与结果服务器原子提升。18 个任务目录均已落盘，transfer marker 状态为 `verified`。5090 随后按收尾契约关机。

5080 的 task040 训练与推理完成，但 summary 的 float32 MAE 与 runner 的 float64 重算相差 0.000020346824955552734，超过当前容差。runner 因此 fail closed，queue 为 24/27，task040 未形成正式结果。只读诊断确认有效点数、掩码、有限性和 artifact 身份一致。现有 staging、checkpoint、prediction artifact 与日志均完整保留。

用户要求暂时关闭 5080。确认没有训练、runner、finalizer 或上传进程后，系统写入关机状态记录并执行正常关机。13:50 新加坡时间，SSH 与 ping 均不可达。当前正式进度为 42/45，剩余 tasks040 至 042。结果服务器空闲 9,322,172,416 字节，`rtx5080_27` 尚未上传。

### 进展快照，2026-07-26，tasks040-041 已恢复，task042 已启动

task040 的精度契约修复已完成。新 validator 首先保留原 float64 audit。只有 graphwavenet、fourier_dual_net、staeformer 与 staeformer_icsf 的 prediction 和 actual 均为 float32，且原 audit 校验失败时，才从同一 artifact 精确复现 producer 的 float32 `.mean()`。summary 必须与 producer 值相差不超过 2 ULP，producer 与 float64 audit 的差异还必须处于归约误差界内。staeformer_deseason 不进入兼容分支。

staging recovery 采用显式、失败即停止的契约。prior queue、failed task state、task ID、最新 attempt、严格整数 returncode 0、canonical staging 与日志路径、config hash、缓存身份、artifact schema、mask、finite、valid_count 和 metric 必须全部重新验收通过。恢复前后逐文件 SHA-256 必须一致，任何失败都不会静默创建新 attempt 或重训。相关控制提交为 `dfb6b7f` 与 `33fae6c`，本地全仓回归为 247 项通过与 1 条既有 warning。独立复审未发现 P0 或 P1。远端 shadow validation 也直接使用 task040 的 483,524,632 字节 prediction artifact 通过。

控制部署探针曾发现 CMD 展开环境变量的缺陷。Python preflight 安全拒绝了合成 task ID，没有 trainer 启动，但 `queue_status.json` 被短暂改写为 failed 0/27。该异常快照已保存到 `state/control_incidents`，SHA-256 为 `bdcb4718859b8dfbaf52d588e2d5af2789dcec42659f8927879807fd120869ab`。queue 随后依据落盘 shutdown state、task040 failed state、cache manifest、canonical manifest、clean Git SHA 与 24 个 `skipped_valid` state 原子恢复。task state 和训练 artifact 从未被改写。最终实现由 Python 直接读取恢复任务环境变量，CMD 不再展开其值。

2026-07-26 11:23 新加坡时间，runner 从 task040 `attempt_0001` 完成可审计 recovery 并原子提升正式目录。没有重训，也没有 `attempt_0002`。summary、checkpoint、prediction artifact 与 train log 的 SHA-256 分别保持为 `be9c84f7bd1608d40ad9e0c623854962cf223715695d38351228d20d9557a97d`、`7537f941f7e8972c2768d8fae7f2481d7f0d981b0d4f5b19a8e588925cd70df8`、`34b74015d3e11b504f28ca89f868b497aa2de4956b75ba078d8010cafde50f45` 和 `51aba30d3dd452ba58f28fb53ea8c254ef908654a4d365b1ff5259c88ba5cd63`。恢复 task state 记录 validator SHA-256 `95a5fc7ed3cfae1fd029e3f7dcae5995ad60c4c3427de1ba327c7c97cf6872a6`。

task041 于 13:40 新加坡时间完成 30 个 epoch 和测试推理。写入 `summary.json` 时，旧原子写入逻辑生成了 282 字符的临时路径，超过 5080 上未启用长路径支持的传统 Windows 限制。正式 summary 目标路径为 239 字符。checkpoint、prediction artifact 与 train log 均已完整落盘，prediction NPZ 通过 ZIP CRC、schema、dtype、shape、有限性和正式 cache 全量对照。有效点数为 103,270,922，producer float32 MAE 为 11.21213436126709，独立 float64 audit 为 11.212104746452527。该故障发生在结果写入阶段，不是模型训练、数据协议或数值失败。

控制提交 `e7bf755` 将同目录原子临时名缩短为 `.s.<uuid16>.tmp`，并增加 completed state 幂等保留与 task041 专用修复入口。Mac 全仓回归为 249 项通过与 1 条既有 warning，独立复审未发现 P0 或 P1。5080 的最终 dry-run 对全部正式证据完成验收且未写入文件。17:08，修复脚本从哈希锁定的原 `attempt_0001` 重建 `summary.json` 与 `decomp_mask.json`，两份文件 SHA-256 分别为 `115c0cfba5d6a256a27011a6169797ac48606cb90a914a942919854e399e27ac` 和 `55ae377b62287d5098a895c5608cb8050d79520df2a5936f0754c0a18bef5d5a`。runner 在 staging 与正式目录分别完成完整验收后原子提升。原 checkpoint、prediction artifact 与 train log 哈希保持不变，没有 `attempt_0002`。incident record SHA-256 为 `630e2d31ca2b8599a0ca9901f5d50e85049ca438a4ca894da744c49eecf4ced4`，completed task state SHA-256 为 `4bb3cb200c77391d24e3b331b64b5787b70ed42163a1db617635b758aecbfe22`。task040 state SHA-256 也保持为 `c54ae78dfdd6ec4e8e4b0005f373e2a712f8686a3fd211fd15cb1703fa5f2312`。

截至 17:15，新加坡时间，runner 层正式进度为 44/45。5080 队列为 26/27。runner 以 `KEEP_COMPLETED` 保留 tasks040-041 的恢复状态，并于 17:10 启动 task042 `attempt_0001`。落盘日志显示 task042 为 epoch 1、batch 1500/4598，GPU 使用 10,799 MiB，利用率 97%，温度 63 摄氏度。runner、trainer 与 finalizer 各有 1 个进程。Minecraft 进程继续保留，没有被终止。finalizer 已启动并等待 27/27，之后将执行 source-host postflight、严格 allowlist 与 `rtx5080_27` 回传。`rtx5080_27` 与 composite index 尚不存在。

### 最新进展，2026-07-27，runner 45/45、双分片发布与 composite 验证

task042 于 2026-07-26 23:32:54 新加坡时间完成并通过 runner 验收。all 与 unaffected masked MAE 均为 `10.780716896057129`，affected 为 `null`，best val masked MAE 为 `10.51922192820698`，有效点数为 `103270922`，参数量为 `1536372`。5080 队列由此达到 27/27，runner 层达到 45/45。

5080 finalizer 首次执行结果服务器目录树复验时，PowerShell 传给 Windows native ssh 的嵌套双引号被剥离，导致远端 AWK 表达式损坏。该故障只影响收尾复验，不影响训练结果、source-host postflight 或已上传文件。提交 `c48f232` 将 AWK 改为无嵌套引号的等价实现。本地全仓回归为 250 项通过与 1 条既有警告，PowerShell 5.1 AST、Windows native argv 与真实 incoming 文件树复验均通过。finalizer 复用已上传的 152 个文件，没有重训、重传或改写任务结果。`rtx5080_27` 于 2026-07-27 11:19:49 新加坡时间完成正式原子提升。

结果服务器上的 `rtx5080_27` 包含 27 个任务目录和 152 个文件，共 4,822,525,731 字节。transfer marker 记录 `validated_tasks=27`、`actual_skipped_valid_tasks=24` 与 `status=verified`，marker SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。5090 的 `rtx5090_18` 保持 `verified`，marker SHA-256 为 `fa00a8e202455eb70505bd266ad05c54a5d17068e315e742913481555cd15084`。训练与分片发布层已经完成，最终产物仍只能称为 `host-stratified distributed 45`。

首次 composite 复验暴露了结果服务器内存峰值风险。旧 validator 在完整 prediction 与 actual 之外，还会创建减法临时数组和完整 producer error 数组。提交 `5f3113b` 改为在 float64 audit 完成后原位复用 prediction 缓冲，并在首次 producer 指标校验前释放 actual。提交 `93038ec` 进一步跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标、有限性、valid_count、float64 audit 与 float32 producer 契约。500 组随机 float32、mask 与无效 target NaN 差分均保持逐元素及 masked mean 完全一致。staging 与正式目录双验收、artifact SHA-256 不变性、summary、训练 checkout 与任务 `config_hash` 也保持不变。本地全仓为 252 项通过与 1 条既有警告，独立复审未发现 P0 或 P1。

旧控制包发起的首次 composite 尝试在结果服务器重启前没有生成日志内容或索引。重启后无法从内核日志确认终止原因，因此记录为 unknown，没有归因为模型、数据或确定的 OOM。空日志已独立归档，incident record SHA-256 为 `166923ef9a63b80ecbc065f6e455eb393341e526ef6411c8c5ccd16ebfa2c70a`。重启后的两份 transfer marker、source-host postflight 和任务目录哈希均保持不变。

2026-07-27 18:28:27 新加坡时间，控制提交 `93038ec` 的 composite postflight 以受内存限制的独立 systemd 服务完成，退出码为 0。服务报告 2.1 GiB 内存峰值与 15.1 MiB 交换峰值，结束后主机交换策略恢复原值，4 个业务容器保持运行。最终只读索引的 `status` 为 `verified`，`execution_layout` 为 `dual_host_stratified`，包含 45 个任务和 2 个分片。5080 为 27 项，5090 为 18 项，canonical tasks 001 至 045 各出现一次。训练 SHA、cache bundle、manifest canonical SHA 与两份 source-host postflight 身份全部一致。索引 SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。composite 没有复制任务目录，也没有改写 summary 或 artifact。最小判别矩阵的完整性状态由此更新为 `verified_host_stratified_45`。该状态只证明双机分层矩阵完整，不等于恢复论文结论，同机配对、多种子与显著性分析仍然暂停。

### 审计前旧管线历史，2026-06-25，旧云 5090 实验

本段全部来自 2026-07-18 审计前的旧数据管线与当时的云 5090，不能与 2026-07-20 启动正式 v3 的当前 5090 混为一谈。两个当时被称为决定性实验的项目已完成或接近完成，历史解释均指向事故标签无增益。决定性实验一，全滑窗协议复跑已完成，三模型乘三区域共九格，标准滑窗十二进十二出，整条时序七比一比二时序切分，masked MAE all 口径，三模型共用同一 loader 窗口逐元素对齐。当时九格均低于 IGSTGNN 主表自报的 Alameda 12.69、ContraCosta 13.43 与 Orange 13.13，详见第 7.4 节。决定性实验二，把事故标签经 ICSF 注入 STAEformer 骨干，事件协议同窗对照，三区域当时均已完成。Alameda 加 ICSF 略差 0.026，ContraCosta 完全相同，Orange 略好 0.061，三个差值都在种子噪声带内且符号混杂，单种子无法与零区分，详见第 7.4 节。同时 STAEformer 的 Orange 事件锚定历史基线已补齐。以上数字只保留为审计前历史，不是当前论文证据。

导师会后，方向从事故导向收敛到分解导向，尝试了一个新架构 RGDN，去季节化加残差引导双支加邻居残差注入。结果为负，机制已查清，详见第 4.7 节。唯一可保留的是去季节化这个标准小技巧，它本身不是新方法。当时继续维持事故标签无增益的历史判断，RGDN 被视为另一个 label-free 数据点。继此前五种调制增强之后，这次换范式的 RGDN 同样无增益。这些判断仍需 v3 正式矩阵重新验证。

### 最新进展，2026-06-27，自适应去季节化 v0c 首次站住

沿去季节化这条线继续深挖，得到一个真增益，修正了上一条里架构探索就此结束的判断。先诊断出 v0b 去季节化的瓶颈，它害事故节点，affected 加 0.200，因为把周期基线加回去对正在偏离的事故节点是错的。对症设计自适应去季节化 v0c，按每个节点近期偏离基线的幅度，动态在周期基线与当前观测值之间选锚点，正常节点信周期，事故节点锚当前水平，残差预测照常叠加。三区域 Alameda 与 ContraCosta 与 Orange，seeds 42/1/2，等参数，全程不读标签，v0c 比 v0b 分别减 0.229 与 0.157 与 0.249，每区域逐种子零重叠，是项目连续六次架构增强失败后第一个干净增益，三战三胜。消融显示增益一半来自全局持续混合、一半来自异常自适应加权，两块都真实。效率上 v0c 用约五分之一到六分之一参数，在 Alameda 与 Orange 两区域追平或反超 STAEformer，ContraCosta 略落后。把方法套到 STAEformer 骨干的代码已写好待跑。详见第 4.8 节。

### 最新进展，2026-07-02，v0c 重定位为 RADS 并在 STAEformer 上迁移成功，全项目新低 11.258

先用两次干净负结果把岔路封死。为回应 v0c 可能被评为 GWN 封装的担心，试了两条独立轻量骨干路线，RegimeFlowNet 状态转移 TCN 与 RAD-GMixer 分解混合器，均明显落后 GWN 系列，等参数控制证明差距是架构性的而非容量性的，独立轻量骨干这条路被判死。期间产出一个规律，qov 多变量输入的价值是骨干相关的，弱骨干上是真增益，加到强 GWN 上反而因过拟合变差。详见第 4.9 节。

随后与导师对齐重定位，v0c 不是新骨干而是机制，正式命名 RADS，状态自适应去季节化重组，可插入任意强时空骨干。决定性验证是把 RADS 套到最强 label-free 骨干 STAEformer 上，Alameda 三种子，s0c 均值 11.258 对原始 11.392，配对差三种子全负均值减 0.134，**11.258 为全项目历史最低**。RADS 至此在 GWN 与 STAEformer 两个骨干上都有正增益，model-agnostic 重组层主张成立，附诚实边界，STAEformer 侧约 1.5 倍种子带需更多区域收紧。详见第 4.10 节。

### 最新进展，2026-07-03，NDR 阶梯全线通过，NDR-Net 成立，全项目新低 11.052

两条推进。其一，ContraCosta 上的 s0a 对 s0c 完成，配对差全负且逐种子零重叠，STAEformer 侧凑齐两区域，1.5 倍带疑虑被对冲，Orange 第三区域在另一张卡上串行跑，预计 7 月 5 日出。其二，外部 NDR-Net 双态模型提案经评审拆成有界阶梯，只赌可验证的假设，停止线为三种子配对差全负且均值小于等于负 0.06。阶梯 A1 即 RADS-O 已跑完并决定性胜出，做法是把 RADS 门控升级为逐 horizon 形式，并在训练期用真实未来构造软 oracle 锚点选择目标做蒸馏，置信加权并按节点尺度做单位校准，测试期门控仍只由历史流量算出，无泄漏。Alameda 三种子，s0f 11.115 对 s0c 11.258，配对差负 0.151 与负 0.192 与负 0.085 全负，均值负 0.143，零重叠，且三种子极差仅 0.009，oracle 监督把门控钉到跨种子完全一致的最优点。当晚 B 档 s0g 与消融 s0e 亦完成，s0g 压线通过且零重叠，NDR-Net 命名成立，完整阶梯 11.392 到 11.258 到 11.179 到 11.115 到 11.052，每级配对全负零重叠，归因干净。详见第 4.12 节。

### 最新进展，2026-07-04，NDR-Net 第二区域验证通过，CC 新低 11.733

ContraCosta 阶梯完成。复合体干净复现，s0g 对 s0c 配对差全负零重叠，均值负 0.148，端到端 12.038 到 11.733 改善 0.305，与 Alameda 的 0.340 一致。诚实差异，中间级归因区域相关，CC 上门控形式与 oracle 单独约为零，增益几乎全由双残差头扛，组件有协同，复合体总量跨区域稳定。方法层级关系一句话，RADS 是机制地基，锚点在周期基线与持续水平间自适应，NDR-Net 是长在其上的完整双态预测头，前者证据最广，两骨干乘多区域，后者绝对性能最强，两区域零重叠。详见第 4.12 节。

### 历史进展，2026-07-06，当时判断全部实验证据关账

Orange 的 s0a 对 s0c 三种子完成，配对差负 0.114 与负 0.132 与负 0.031 全负，均值负 0.092，方向与前两区域一致，强度最弱且非零重叠，如实记录。当时据此判断 RADS 在 GWN 与 STAEformer 两骨干乘三区域共六格全部配对为正增益，NDR-Net 在两区域复合零重叠，主命题三段论与方法证据全部关账，并决定转入论文写作。2026-07-18 审计已经推翻“关账”状态，数字保留为历史记录。写作前的新颖性查证已完成，机制组件均有先例，含 2025 年 trend-event 双通道解耦等并发工作，故 RADS 与 NDR 定位为主命题下的建设性配菜而非全新方法，related work 主动引用近似工作，详见第 9.1 节。方法升级路线的 P1 检测性诊断已完成，双任务腿被杀死，但副产品是主命题最直接的机制证据，被标注事故在流量偏离中几乎不可见，详见第 4.13 节。

### 接下来要做什么

代码阶段已经完成旧结果冻结、v3 schema 与 provenance、因果缺失处理、train-only climatology、完整目标窗和 event group 互斥切分、预测时刻去重、事故输入与分析标签拆分、训练统计边界修正、IGSTGNN 掩码对齐及本地回归测试。

当前待执行事项如下。

1. v2 缓存与四个 preliminary summary 已冻结为 contaminated，不再续跑。
2. 三地区 `region_data_v3_noleak_dedup` 已从 clean 提交重建完成，本地两次、5080 一次与 5090 一次严格审计均为 verified。
3. 最终训练代码 SHA 已锁定。训练 checkout 仍为 clean `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。最新控制层全仓回归为 252 项通过与 1 条既有警告。
4. runner 层 45/45 已完成或重新验收。5090 分片 18/18 与 5080 分片 27/27 均已完成、回传并验证。
5. task042 已完成并正式验收，tasks040-042 的直接比较组保持在同一台 5080。
6. tasks040-041 的恢复没有创建新 attempt，task042 为正常完成的 `attempt_0001`。
7. 结果服务器已保存状态为 `verified` 的 `rtx5090_18` 与 `rtx5080_27`。
8. finalizer 引号处理修复提交为 `c48f232`，composite 内存修复提交为 `5f3113b` 与 `93038ec`，本地全仓为 252 项通过与 1 条既有警告。
9. 只读 composite index 已生成并验证。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，索引 SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。
10. 同机配对、多种子与显著性分析已于 2026-07-27 完成，产物见 `outputs/protocol_v3_noleak_dedup/analysis/`。下一步由用户据落盘结果决定 baseline zoo 与论文写作是否恢复。

### 最新进展，2026-07-28，IGSTGNN Table 4 全基线复现启动

项目新增一条与 v3 无泄露研究矩阵严格隔离的论文复现线，协议为 `igstgnn_table4_official_v1`。目标是按公开材料最大忠实复现 IGSTGNN Table 4 的 Alameda、Contra Costa、Orange 三地区与 13 个方法，共 39 项。该线使用作者 Kaggle 发布数据和官方顺序 70%、15%、15% 切分，不使用项目 v3 缓存。官方发布协议中存在已知重复预测时刻和跨 split 重合，因此有效性标签为 `official_protocol_with_known_overlap`，结果不得被表述为无泄露。

LargeST、BiST 与 IGSTGNN 的上游提交分别锁定为 `b7c6ff5f7f0c10051ed10a319b2cfe58021312f9`、`dd94adf7721fcbb9e3feb5d1b44040305199a4cc`、`64c6e243466f791f1b75e50421d63e39577ec9fa`。正式 v2 矩阵清单 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`。BiST 的三地区随机 seed 已分别冻结为 676615、683452、150462。正式数据、三个 pristine checkout 和提交 `5aadfff` 的精确归档已部署到 5080 的短路径 `D:\t4`。Mac 复现测试现为 181 项通过与 1 条既有 warning，5080 Windows 训练相关测试为 127 项通过、2 项平台跳过。

Alameda HL 已完成并通过 checkpoint、prediction、provenance、summary 与 train log 五文件验收。测试集按 LargeST 源码消费 2,176 条，丢弃不足一个 batch 的尾部 28 条。H3 的 MAE、RMSE、MAPE 为 15.916486、26.710745、18.755560。H6 为 19.056049、31.813951、21.526349。H12 为 25.661367、42.373371、28.095365。Average 为 19.649600、32.688443、22.271724。prediction artifact SHA-256 为 `63482403b21e74577e2c74cd3de198bdfafa8c90e8692843a506adcf2614e6b0`。

Alameda LSTM 已于 2026-07-28 22:02:36 新加坡时间完成，随后通过 runner 复验与独立二次复验。测试集按 LargeST 源码消费 2,176 条。H3 的 MAE、RMSE、MAPE 为 12.825795、22.250841、17.630659。H6 为 14.208229、24.909077、17.839295。H12 为 17.033611、29.907513、20.389889。Average 为 14.355326、25.072953、18.407032。prediction artifact SHA-256 为 `8f70ed5a8264c007e2baf5453bf18d0d621ad0502ad29e39ea34f83315b0e769`。

task003 Alameda DCRNN 首次于 22:19:57 通过 SSH 附着启动。实测单 epoch 训练约 169 秒，验证约 18 秒。为了让数小时至数天的训练不依赖 SSH 会话，该次尝试在完成 3 个 epoch 后于 22:32:25 按计划中断。runner 正确落盘 `interrupted` 和退出码 130，没有发布正式任务目录，不影响 HL 或 LSTM。incident 目录为 `D:\t4\incidents\table4_003_attached_interrupted_20260728T143225Z`，incident SHA-256 为 `24da66f71bac2ecfe8e3654ad1630cf97723f7b1f9c2c4fd64b12db7e666be6b`。

提交 `fa5b458` 新增 fail-closed 持久接力器及 2 项测试，提交 `8ad61bf` 固定 5080 启动命令。整个 Table 4 复现测试集更新为 181 项通过与 1 条既有 warning。接力器远端 SHA-256 为 `3410e9eab5a53c421afc3c93eaeb00f49dbf371dc86165a8d1e15b7bd5054023`。Windows 任务计划 `XTraffic_IGSTGNN_Table4_v2` 于 22:36:39 启动，执行时限为 `PT0S`，开机后可恢复，且忽略重复并发实例。它先运行 task003 与 task013 组成的 pilot 门，两项全部通过后才启动完整 39 项。新 task003 于 22:36:40 从 epoch 0 开始，启动复验时显存使用 11,899 MiB，GPU 利用率 82%，温度 53 摄氏度。最终交付包括同结构 Table 4、论文值差异表、逐数字追溯索引和公开材料缺口清单。

2026-07-29，正式执行从 5080 单卡 pilot 切换为两台 RTX 4090 D 主机，共 9 张 24 GB GPU。固定计划 `igstgnn_table4_4090d_9gpu_v1` 的 SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。五卡机承担 Orange 13 项，四卡机承担 Alameda 与 Contra Costa 26 项，因此三地区各自的全部方法都在同一台主机内完成。正式数据共 27 个文件与 9,051,956,107 字节，两端逐文件 SHA-256 均通过，数据哈希清单 SHA-256 为 `bcf88f2902d900ce9baed5c33b8d064637dce1a3d9498994c7db5eb855d2d831`。

5080 旧 pilot 在完成 epoch 38 后停止，没有 task003 正式目录。任务计划已禁用，相关 Python 进程为 0。incident 清单 SHA-256 为 `29c58bad6635ae3c7d476ec875cd577153ab4a95981d32f2a16baf89248bde4a`。首次九卡启动因 rsync 保留 Mac UID 501 被训练入口的所有权安全门拒绝。9 个 shard 均在 trainer 启动前退出，没有正式结果。两端修正所有权并重新完成全部数据哈希校验后，五卡机与四卡机分别于 00:47:14 和 00:47:32 新加坡时间正式重启。

Orange IGSTGNN 的默认分配器与 `cudaMallocAsync` 两次单卡诊断均在首个 forward 因 24 GB 物理容量不足而 OOM，没有形成正式任务目录。提交 `f43fb58` 增加保留完整 batch 的双卡执行适配，提交 `25b5307` 增加自动恢复控制器。控制器 PID 为 `20105`，等待 Orange BiST 释放 GPU 3 后，使用物理 GPU 4 与 GPU 3 执行 task039。pristine 上游源码、数据、batch、loss、优化器、epoch 与 early stopping 均不修改。该执行差异会保留在 provenance 中，最终不得写成单卡逐命令复现。

两台主机均已布防严格自动收尾。各自固定 shard 完成后先执行 source-host postflight，再按精确 allowlist 上传到结果服务器。结果服务器逐文件计算 SHA-256、核对精确目录树并原子提升。只有正式目标再次完整复验通过后，源主机才调用 `/usr/bin/shutdown -h now`。任一步失败都会保持机器开机。完整记录见 `docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md`。

### 最新进展，2026-07-30，NDR 9/9 发布与 task034 四卡恢复

NDR-Net Table 4 扩展已完成 9/9。结果服务器正式目录通过逐文件 SHA-256、精确目录树和原子提升，含 49 个文件和 940,488,092 字节，files digest 为 `ab21c8e300ba930e5611cb9c07ff5b047eb03d281ca669c756242707231c87e7`。该扩展继续独立标记为 `same_protocol_project_method_extension`，不属于 IGSTGNN 原论文 39 项。

原矩阵 task039 Orange IGSTGNN 已完整落盘并通过五文件验收。H3、H6、H12 与 Average MAE 为 12.503220、13.296460、14.551459 与 13.285545，summary SHA-256 为 `ab0288a4b8fe5a17c33b43b758bbc5851b0e63cc0cce7456a07eb5fe451ce998`。原矩阵由此达到 38/39。

最后一项 task034 Orange STTN 在五卡机单卡训练时尝试额外分配 11.22 GiB 并 OOM，没有正式结果。用户要求关机后，五卡机既有 12 项先完成 60 个 artifact 的关机前清单，随后正常关闭。五卡机尚未正式回传。用户重新开启四卡机并批准接管 task034。

独立恢复计划 `igstgnn_table4_task034_4gpu_recovery_v1` 的 SHA-256 为 `022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5`。execution layout 固定为 `cross_host_single_task_recovery`。提交 `4935b9b` 增加 STTN 四卡整批恢复、真实完整优化步探针、独立 postflight 和严格回传。本地全仓为 465 passed 和 2 个既有 warning，四卡机专项门为 13 passed。

2026-07-30 09:21:02 新加坡时间，四卡机控制器 PID `2928` 启动。batch 64 smoke 完成前向、反向、梯度裁剪和一次优化器更新，loss 为 `170.42190551757812`。四卡峰值分配显存约为每卡 16.1 GB。09:21:28 smoke 通过后，正式 task034 开始训练。首个 epoch 的训练 loss 为 29.8492，验证 loss 为 21.1960。只有五文件验收、独立 postflight、结果服务器逐文件复验和原子提升全部成功后，四卡机才自动关机。

用户随后重新开启五卡机。提交 `918af63` 建立明确排除 task034 的 12 项只读导出计划。计划 SHA-256 为 `cd81652493690d1d329f5eecd45fe541c9de7db26c93650fc490a2507616845d`。source-host postflight 验证 12 项、60 个 artifact 和 1,975,159,969 bytes，files digest 为 `5c11ad4596228f08613b783d8a42ed9334e9976ce8cff4d23d09f6b79f055490`。

结果服务器容量门首次在传输前安全暂停。只清理系统归档日志和包缓存后释放 1,753,280,512 bytes，没有删除正式结果。重试完成逐文件 SHA-256、精确目录树和原子提升。transfer marker SHA-256 为 `53f73b3e7ae2f92535eed44fb0fc934d7be45387de579b8dfc3fcb853afd54cc`。五卡机随后自动关机并确认 SSH 不可达。

task034 于 2026-07-30 10:38:24 新加坡时间完成正式训练。H3、H6、H12 与 Average MAE 分别为 12.652452、13.311158、14.048541 与 13.212108。Average RMSE 为 23.502980，Average MAPE 为 14.091901%。prediction SHA-256 为 `c6064688ab3bedd36b4f058ce899a97b8882645c4169979d0a6170fabf1f773a`，有效观测点数为 25,963,481。

首次收尾因 provenance 层级读取错误而 fail closed。提交 `a9511d5` 修正 postflight 从 `factory.execution_adapter` 读取。第二次收尾确认 postflight 已通过，但 finalizer 把报告中的相对路径解析到外层结果目录。提交 `51148f1` 改为传入 task034 shard 根。两次故障都发生在训练完成之后，没有重训、没有改写五份 artifact，也没有建立第二个正式任务目录。全仓回归为 469 passed 和 2 个既有 warning。

修正后的收尾验证 5 个结果文件和 170,575,062 bytes。结果服务器容量门先按默认 2 GiB 余量暂停，在证明没有 incoming 或正式目标后改用 512 MiB 余量。正式树完成逐文件 SHA-256、精确目录、原子提升和二次复验。目标为 `task034_4gpu_recovery_v1/hosts/table4_4090d4_task034`，files digest 为 `6ce435df4293dad11602378b1e5b9acf18ffde1717f5e0fdfb5922f3282e8258`，transfer marker SHA-256 为 `e393e8238b25677b55e7f1ae00813c3f0a9eb46f7a532c0eb7bba21ee4758d68`，postflight SHA-256 为 `fa8a3d0f0e0fba9d67fa569392edc83c9a74d9131b81439ea7644c7fbc700e15`。incoming 已不存在，五份结果均为 OK。四卡机随后自动关机，连续三次 SSH 探测均确认不可达。

原 Table 4 复现的任务结果层由此达到 39/39。该时间点只证明三个正式来源树已经发布，不等同于最终矩阵已验证。当时仍需建立三源只读 composite index，证明 task001 至 task039 各出现一次且身份一致，之后才能生成最终对比表。

### 最新进展，2026-07-30，Table 4 三源复验与最终对比表完成

原 39 项三源只读 composite 已在结果服务器完成。控制提交为 `dc1f8a54919d611ac074e128d6f957c03dd7b0e6`，源规范 SHA-256 为 `38645a51935618e4f12a96374d5d9695711bdf6581d984ec2baa15be63345e56`，validator SHA-256 为 `8d988a7164fb48c526ab0e7df7b1832ecc374bac108371a354a12582298b0e10`。正式复验验证 3 个来源、39 个任务、199 个文件和 3,399,778,682 字节，状态为 `verified_official_protocol_with_known_overlap_39`。canonical tasks001 至 039 各出现一次，三个地区各 13 项，13 个方法各 3 项。正式 index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。第二次独立完整复验得到相同任务集合、来源映射、summary 指标和 index 内容，仅复验时间字段不同。

论文同结构 39 项复现表已经生成。CSV SHA-256 为 `60160b443f63425c3b10cd30e29253db85a17d68e2a3cf12ac5bcf195449cacc`，逐单元论文值对照 CSV SHA-256 为 `8f1f80ea3ec29f27efc0ddefd3d4711f06376f0655d68ec51c078016f37a0e98`。复现值与论文报告值的 Average MAE 绝对差均值为 2.450150，范围按复现值减论文值计算为负 6.216278 至正 0.719624。这说明公开材料最大忠实复现与作者表格仍有显著差异，不能宣称逐命令或逐数值完全复现。

39 项复现的 Average MAE 如下。此表来自逐任务复验后的正式 composite index。

| 方法 | Alameda | Contra Costa | Orange |
|---|---:|---:|---:|
| HL | 19.649600 | 19.729146 | 20.541655 |
| LSTM | 14.378215 | 15.018553 | 15.217621 |
| DCRNN | 13.817049 | 14.710834 | 13.928103 |
| AGCRN | 11.935185 | 13.110660 | 12.742069 |
| STGCN | 12.619235 | 13.493722 | 13.156748 |
| GWNET | 12.376839 | 13.155501 | 13.314068 |
| ASTGCN | 13.213007 | 13.820336 | 13.615977 |
| STTN | 12.403421 | 13.282988 | 13.212108 |
| DSTAGNN | 12.431880 | 12.945845 | 12.918909 |
| DGCRN | 13.012386 | 16.070231 | 13.247819 |
| D²STGNN | 13.421594 | 13.911030 | 14.639624 |
| BiST | 10.726239 | 11.354355 | 11.692508 |
| IGSTGNN | 12.763026 | 13.762482 | 13.285545 |

NDR 扩展的正式 49 文件树通过独立结果服务器复验。回执 SHA-256 为 `dbd9dcb355a3324707e728a1942465612cd443695b51ea6359e80dc03d1be100`，扩展 index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。三地区的三种子汇总如下。

| 方法 | 地区 | Average MAE | Average RMSE | Average MAPE |
|---|---|---:|---:|---:|
| NDR-Net | Alameda | 10.759524 ± 0.040862 | 20.063200 ± 0.062496 | 13.734791 ± 0.042310 |
| NDR-Net | Contra Costa | 11.413717 ± 0.024537 | 20.619580 ± 0.055982 | 14.688178 ± 0.023821 |
| NDR-Net | Orange | 11.569331 ± 0.031664 | 21.928622 ± 0.036889 | 13.055433 ± 0.020880 |

NDR 在 Orange 的 Average MAE 比单次 BiST 低 0.123177。在 Alameda 和 Contra Costa，NDR 分别比单次 BiST 高 0.033285 和 0.059362。该结果按真实落盘数字保留。前 39 项 baseline 是各方法单次运行，NDR 是种子 42、1、2 的均值和样本标准差，因此当前比较只能作为描述性结果。若需要显著性结论，应为选定 baseline 补齐相同三种子。

最终 39 行加 3 个 NDR 汇总行的 CSV、Markdown 与 LaTeX SHA-256 分别为 `057e0fa94bbea24d834ed16ed00b94b0599a9e9081321cc205643793c47f34c1`、`1097b82ed71ec26d9657ff28cebcefd5c3455a179b3ef8ad6ba68529b7a118d8` 与 `33047e4d0e11929ae1c8c4e3121dd6aad9bb8684769713d58c3621fb961e22ce`。本地全仓回归为 474 passed 和 2 个既有 warning。IGSTGNN Table 4 复现线现已完成训练、回传、三源 composite、论文值差异表、NDR 三种子扩展和数字追溯。它仍不得与 `protocol_v3_noleak_dedup` 的无泄露研究结论混合。

### 最新进展，2026-08-01，BiST 与 NDR-Net 原生配方 100 epoch 补跑启动

为了让补跑结果能够直接放入既有 Table 4，新建独立矩阵 `igstgnn_table4_bist_ndr_native100_18_v1`。它包含 BiST 与原 STAEformer 版 NDR-Net、Alameda、Contra Costa、Orange 三地区、种子 42、1、2，共 18 项。矩阵 SHA-256 为 `e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246`，配方 SHA-256 为 `37ebcb574f516bc1b662ade0d5264deda96a20d017d20edc7a57253daab5e66b`，实现提交为 `53f4b3a`。

本次不统一两个方法的优化器和 early stopping，而是保留每个方法的原生配方。BiST 保留 batch 64、drop partial batch、Adam 0.002、MultiStepLR 和 patience 20，只把最大 epoch 从 500 改为 100。NDR-Net 保留 batch 16、retain partial batch、Adam 0.001、CosineAnnealingLR、全局验证 masked MAE 和无 early stopping 契约，只把最大 epoch 从 30 改为 100。所有结果继续报告 H3、H6、H12 与 12 步 Average 下的 MAE、RMSE 与 MAPE。

Mac 全仓回归为 489 passed 和 2 个既有 warning，5080 Windows 专项门为 25 passed。Orange 最大地区的两个真实 CUDA 完整优化步探针均通过。BiST 峰值分配显存为 3,064,998,912 bytes，NDR-Net 为 12,633,100,800 bytes。5080 计划任务 `XTraffic_Table4_Native100_v1` 于 2026-08-01 00:02:23 新加坡时间启动，执行时限为 `PT0S`。启动复验时队列为 0/18，失败数为 0，正在执行 `n100001_bist_alameda_seed42`。

专用入表器 `native100_compare_report.py` 已通过完整与缺项矩阵回归。它会逐项复算 18 份结果，缺少任意一项就拒绝出表。完整时保留既有 42 行并追加 6 个地区方法行，自动生成同结构 CSV、Markdown、LaTeX 与 manifest。

这组补跑继承 Table 4 官方切分的已知重合，有效性标签仍为 `official_protocol_with_known_overlap`。18 项全部落盘和验收前，不得报告 100 epoch 下的性能结论，不覆盖已有 BiST 500 epoch 单次结果或 NDR-Net 30 epoch 三种子结果，也不得与 v3 无泄露矩阵混合。完整启动记录见 `docs/audits/igstgnn_table4_native100_launch_2026-08-01.md`。

### 最新进展，2026-08-01，RTX 5080 与 RTX 5090 双机重分片

为了缩短剩余训练时间，100 epoch 补充矩阵已切换为按地区分层的双机执行。固定计划 `igstgnn_table4_native100_5080_5090_v1` 的 SHA-256 为 `d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e`。RTX 5080 只选择 Alameda 与 Contra Costa 的 12 项，RTX 5090 只选择 Orange 的 6 项。两分片交集为空，并集严格为 18 项。同一地区内 BiST 与 NDR-Net 的三个种子保持在同一台机器，因此地区内直接比较没有跨主机。

5080 切换时 task013 只有临时训练目录，没有正式结果。现场清单摘要为 `94802ea874935de3db70eb3d5b730ec14e978662ebeada4014bc7eb09f34be23`。旧 18 项计划任务已禁用，新任务 `XTraffic_Table4_Native100_5080_CC_v1` 从 9 个已验证的声明内结果继续。12:40 前 task013 已完成 epoch 17。结果根中切换前完成的 Orange BiST 三项保持只读，但最终 composite 明确排除它们。

5090 的 Orange 九个数据文件均已与 Mac 原件逐文件复验。split manifest SHA-256 为 `c37b78f19766fec9ac57d28d6a7becd8a3b334261cef4e7863f29b0bbcc7b019`。BiST 与 NDR-Net 的真实 CUDA 完整优化步探针均通过，峰值分配显存为 3,142,229,504 与 12,682,121,728 bytes。5090 持久队列于 12:38:57 启动，选择 6 项，初始失败数为 0。12:40 时首项 Orange BiST seed 42 已完成 epoch 6。Mac 全仓回归更新为 493 passed 和 2 个既有 warning。

最终结果必须标记为 `dual_host_region_stratified`。双机训练结束后仍需分别完成 source-host postflight、结果服务器逐文件复验与只读 composite。完成这些门禁前不生成 18 项最终表。详细记录见 `docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md`。

23:45，新加坡时间，用户要求 5090 在训练完成后自动关机。新增失败关闭守护脚本，SHA-256 为 `d0168a7f1b3b2b1b5386de1cebb52681fc1319babe8df1eddb5344228706bbb1`。守护会重新验证 Orange 六项的矩阵身份、双机计划、queue status、runner exit code 和每项五文件 artifact，并确认训练进程完全退出。只有全部通过才执行 `/usr/bin/shutdown -h now`，任何异常都会保持开机。远端持久 PID 为 `20225`。启动守护时 5090 为 4/6 完成、0 失败，task017 已完成 epoch 82。

5080 分片已在当天 15:52:38 完成，声明内 12/12 全部通过 runner 验收，失败数为 0。用户报告主机随后断电。23:37 局域网连接恢复后，completed queue、12 个已验证任务和空闲 GPU 均已复验。断电没有造成训练中断或重跑。结合 5090 的 4/6，完整矩阵当时为 16/18 已验收。

2026-08-02，旧 5090 端口持续不可达。用户提供的新端口对应全新空数据盘容器 `autodl-container-dbf949bc7b-086867e4`，旧机的 4/6 正式结果和 task017 epoch 82 临时状态均未恢复。旧守护在最后状态不满足 6/6 关机门，因此不能把旧机离线写成成功自动关机。当前只记录为疑似余额或供应商停机，未回传的旧机结果不进入 composite。

替代 5090 重新部署训练归档提交 `ace318b`、锁定的 BiST pristine checkout 与 Orange 九个正式数据文件。全部数据逐文件 SHA-256 复验通过。BiST 和 NDR-Net 的真实 CUDA 完整优化步探针再次通过，loss 为 `148.53358459` 与 `42.48369026`，峰值分配显存为 `3142229504` 与 `12682121728` bytes，与旧机完全一致。Orange 六项于 2026-08-02 09:48:50 新加坡时间从头重跑。09:59:56 时 task007 到 epoch 70，队列为 0/6、0 失败。本地全仓回归为 496 passed 和 2 个既有 warning。

新 finalizer SHA-256 为 `a1c85fc95b5af92b51dcd44e209cc132a93a1b7bf7459d4552fcfe42fa892597`，持久 PID 为 `2815`。它使用专用受限 SSH 身份，在六项本地五文件验收、runner 零退出、结果服务器容量门、逐文件 SHA-256、精确 allowlist、原子提升和正式树二次复验全部成功后才关机。任何失败都会保持主机开机。完整细节见 `docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md`。

11:11 时替代 5090 已完成 Orange BiST 三种子，队列为 3/6 并开始 `n100016_ndr_net_orange_seed42`。空闲 5080 先对 `n100017_ndr_net_orange_seed1` 完成真实 CUDA 完整优化步探针，loss 为 `42.24521446`，峰值分配显存为 `12633100800` bytes。随后在隔离结果根启动 live speed gate。5080 运行 292 秒仍未完成首个 epoch，5090 的 `n100016` 则从 10:37:43 启动并在 11:23 前完成 13 个 epoch，真实速度约为每 epoch 3.5 分钟。

5080 辅助分片的预计收益很小，不足以抵消变更已锁定来源和 finalizer 的审计成本。候选计划因此被标记为 `rejected_after_live_speed_gate`，对应计划任务已停止并禁用。runner 受控终止退出码为 2，没有正式任务目录。现场清单摘要为 `4ce4bf5402c69d3f11adb95a711fe8f2df3e18161a3e81dcec1410ef592e45ed`。原按地区双机计划继续生效。11:25:57 时 5090 为 3/6 完成、0 失败，task016 完成 epoch 14，完整矩阵为 15/18，finalizer PID `2815` 正常等待。5080 保持空闲且测速 incident 不进入最终 composite。本地全仓回归为 498 passed 和 2 个既有 warning。

### 最新进展，2026-08-03，100 epoch 双机矩阵、composite 与最终表完成

替代 5090 已完成 Orange 6/6，失败数为 0。source-host postflight、结果服务器逐文件 SHA-256、精确 allowlist、原子提升与正式树复验全部通过。5090 正式分片含 31 个结果文件和 `1109955208` bytes，files digest 为 `1ae14803b44dcd88190dbf81ef47431ce161238749fec0c5e37c67acc37b9a52`。发布后 5090 按用户要求自动关机。

5080 已完成 Alameda 与 Contra Costa 12/12 的严格回传，失败数为 0。正式分片含 61 个结果文件和 `686213987` bytes，files digest 为 `5d2de42964e581ea68f6612e7c05b4672268b22f267f887c191e62bc1470abf9`。旧 Orange BiST tasks007-009 不在导出 allowlist 内。Windows SSH 子进程曾在文件写入与远端哈希校验成功后未退出。提交 `4920a4d` 增加无标准输入 SSH 和已验证文件容量复用，重启后跳过已匹配文件并完成原子发布。该 incident 没有损坏或重训任务。5080 保持开机。

两份正式树已完成两次全量只读 composite 复验。正式 index 状态为 `verified_native100_dual_host_18`，SHA-256 为 `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c`。它包含 2 个来源、18 个 canonical task、92 个结果文件和 `1796169195` bytes。第二次复验除时间字段外与正式 index 完全一致。composite 控制提交为 `a34dc958f336e06779c4419bbdb5303c93098c91`。

最终入表提交为 `416586aa625e1bd9897283fedfbdf46190b4a29a`。报告保留既有 42 行并追加 6 个地区方法行，产物位于 `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`。CSV、Markdown、LaTeX 与 manifest SHA-256 分别为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`、`e29348144e19d728dc614016b7d3653dd2587e0eb4fbb32b7569d4d99ce3ae5a`、`66bd422b30e63b500d7e33c103982735c5ea53af60b760463ae2b7167eef262c` 与 `e9a4859fac07abd532f48f0075f6cbfa3e491f80af32d18082291680b74416bf`。

100 epoch 三种子 Average MAE 均值与样本标准差为：

| 地区 | BiST 100ep | NDR-Net 100ep | NDR-Net 的 MAE 差值 |
|---|---:|---:|---:|
| Alameda | `10.938871 ± 0.011328` | `10.504639 ± 0.056600` | `-0.434232` |
| Contra Costa | `11.582189 ± 0.010355` | `11.299118 ± 0.012621` | `-0.283071` |
| Orange | `11.765688 ± 0.005298` | `11.136362 ± 0.030750` | `-0.629326` |

这是同一 100 epoch 上限下保留各自原生配方的描述性比较。它继承 `official_protocol_with_known_overlap`，不是 `protocol_v3_noleak_dedup` 结论，NDR-Net 行也不是 IGSTGNN 原论文方法。本地全仓回归为 502 passed 和 2 个既有 warning。

---

## 第二部分：研究目标与演进路径

项目经历了几次重要转向，每次转向都由实验结果驱动，而非主观调整。

1. 起点：基于 XTraffic，研究利用事故信息增强流量预测。
2. 第一阶段：对流量做傅里叶变换分解与可视化，发现受事故影响节点与正常节点在高频段差异很小，事故的频域特征并不显著。
3. 第二阶段：放弃 affected 与 normal 二分思路，转为纯信号分解，低频学主变化，高频残差学扰动，两条 GraphWaveNet 各管一支，构成 FourierDualNet，简称 FDN。
4. 第三阶段：与 IGSTGNN 在同一数据管线下严格对比，期间发现其官方代码 bug 并修复重训，结论是 label-free 的 FDN 不输甚至更好。
5. 第四阶段：为增加架构创新性，尝试五种架构增强，全部无增益。
6. 第五阶段：把 FDN 移植到标准基准 METR-LA 与 PEMS04 与 PEMS08，等参数消融揭示频谱路由真实增益很小。
7. 第六阶段：把当前标准 SOTA 模型 STAEformer 放进 XTraffic，发现它 label-free 即超过用标签的 IGSTGNN，也超过我们的 FDN，论文主张最终从模型最强转向标签无增益。
8. 第六阶段后期：核对 IGSTGNN 原论文，明确其工作目的与漏洞，规划决定性实验。
9. 第七阶段，2026-06-22：导师会后转向分解导向，尝试 RGDN，去季节化加残差引导双支加注入，等参数消融结果为负，注入被诊断为优化陷阱，去季节化是唯一小增益，详见第 4.7 节。
10. 第八阶段，2026-06-27 到 06-30：沿去季节化深挖，诊断出 v0b 害事故节点，对症设计自适应去季节化 v0c，按节点近期异常幅度在周期基线与当前观测之间动态选锚点。三区域等参数多种子，v0c 比 v0b 干净减 0.229 与 0.157 与 0.249，每区域逐种子零重叠，三战三胜，是连败六次后第一个架构侧真增益，且不读标签，详见第 4.8 节。
11. 第九阶段，2026-06-30 到 07-01：为回应 v0c 是否 GWN 封装的担心，探索两条独立轻量骨干，RFN 与 RAD-GMixer，均为干净负结果，等参数控制证明差距是架构性的。附带发现 qov 输入的价值是骨干相关的，加到强 GWN 上反而变差。独立轻量骨干岔路封死，详见第 4.9 节。
12. 第十阶段，2026-07-01 到 07-02：v0c 重定位为 RADS 机制，骨干降级为残差函数的实现。在 STAEformer 上迁移验证成功，Alameda s0c 11.258 为全项目历史最低。两区域 Alameda 减 0.134 与 ContraCosta 减 0.156 配对全负，ContraCosta 零重叠。RADS 在 GWN 与 STAEformer 两骨干上都有正增益，跨骨干跨区域，model-agnostic 主张已支撑，详见第 4.10 节。
13. 第十一阶段，2026-07-03：外部 NDR-Net 提案拆成有界阶梯并设停止线。A1 即 RADS-O，逐 horizon 门控加训练期 oracle 锚点选择蒸馏，Alameda 配对差三种子全负均值负 0.143，零重叠，全项目新低 11.115，三种子极差仅 0.009。当晚 B 档双残差头压线通过且零重叠，A0 消融归因干净，NDR-Net 成立，全项目新低 11.052，状态条件动态图默认不做，详见第 4.12 节。
14. 第十二阶段，2026-07-04：ContraCosta 补 NDR 阶梯第二区域验证，复合体零重叠复现，负 0.148，CC 新低 11.733，中间级归因区域相关如实记录。
15. 第十三阶段，2026-07-06：Orange 收官，配对全负均值负 0.092。当时判断 RADS 六个骨干乘区域格全部为正，全部实验证据关账，并转入论文写作。该状态于 2026-07-18 被数据审计推翻。
16. 第十四阶段，2026-07-18：代码与 H5 审计确认全年未来插值、label-informed 气候态、事后事故特征、split 重叠、统计范围越界与 IGSTGNN adapter 错位错绑。论文写作暂停，旧 XTraffic 结果统一冻结为 contaminated。
17. 第十五阶段，2026-07-18：完成无泄露 v2 代码修复。因果填充、train-only climatology、目标窗和 event group 隔离、可部署事故输入、统计边界、IGSTGNN 对齐、schema 与 provenance 已接入正式入口。28 项本地测试通过。仓库根目录与文档索引完成整理。下一阶段只生成正式缓存并重跑最小判别矩阵。
18. 第十六阶段，2026-07-19：三地区正式 v2 缓存与事故关系缓存生成完成，事件协议、full-window 协议、schema、provenance 和远端 28 项测试通过。45 项最小判别矩阵在 5080 上开始串行重跑。
19. 第十七阶段，2026-07-19：v2 后续复核发现同一预测时刻重复采样、主事故选择不规范、验证 MAE 聚合不精确，以及活跃事故、分析标签、IGSTGNN 掩码、图支持和远程结果身份仍有完整性缺口。队列停止，四个已完成 summary 降级为 preliminary contaminated，第 5 项未形成可用 summary。
20. 第十八阶段，2026-07-19：v3 全面修复完成。schema 升级为 3，协议切换为 `protocol_v3_noleak_dedup`，143 项测试通过并保留 1 条既有警告。`origin/fu` 的 `625aacf` 所提空 mask batch 正确性要求已通过共享指标累加器与 runner 有限性检查吸收，没有直接合并旧分支。
21. 第十九阶段，2026-07-19：三地区正式 v3 缓存从构建代码 SHA `29159f626494de52c69b36fe5bba9854f13cb56c` 重建完成。staging 与正式路径两次严格审计均为 verified。
22. 第二十阶段，2026-07-19：v3 修复经 PR #3 合并到 main，补齐 Windows 路径测试后将最终训练代码 SHA 锁定为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。正式缓存归档传到 5080 并通过 SHA-256 与只读严格审计复验。45 项最小判别矩阵于 20:18 启动。截至 21:55，首项已完成 epoch 16，队列完成数为 0/45，尚无完成并验收的新结果。
23. 第二十一阶段，2026-07-20：5080 上 tasks 001 至 004 完成并通过 runner 验收，得到四项 v3 seed 42 单项结果。task 005 因训练负载迁移主动终止，退出码 1 不代表模型失败。32 GB RTX 5090 完成相同代码 SHA 的 clean checkout、143 项测试与缓存严格审计，并完成 tasks 005 至 006。至此共有六项正式 seed 42 结果通过 runner 验收。
24. 第二十二阶段，2026-07-20：为了让 5080 分担训练，同时避免 Windows 与 Linux 的路径相关 `config_hash` 和运行时 provenance 被错误混合，正式口径改为 `host-stratified distributed 45`。5080 固定承担 24 项，5090 固定承担 21 项。两分片互斥、并集为 canonical 45，所有直接比较组保持同机。旧 5090 队列在任务边界停止时留下 task 007 的零字节日志和未完成 staging，没有 summary，已从正式导出中排除。
25. 第二十三阶段，2026-07-20：5080 的 task 007 `attempt_0001` 因两条遗留诊断进程耗尽主机内存而在 epoch 9 数据装载阶段失败。该 attempt 没有 summary 或正式任务目录。清理两条诊断进程后，runner 用 `attempt_0002` 恢复训练。5090 分片继续运行，两台机器的自动收尾进程均已重新部署并复验。
26. 第二十四阶段，2026-07-22：5080 固定分片达到 24/24，runner 退出码为 0，source-host postflight 和严格导出 allowlist 均通过。5090 达到 14/21 并进入 task039，runner 层合计 38/45。用户批准 27/18 重分片后，控制契约、两端 finalizer、边界 transition、ready marker 和严格 postflight 已更新并布防。控制提交为 `60405a9`、`e191c27` 和 `30bcb81`，本地回归为 230 项通过和 1 条既有警告，5080 真实 PowerShell AST 与退出码冒烟验证通过。截至 20:51 新加坡时间，新 runner 尚未启动，ready marker 与 task040 正式目录均不存在。
27. 第二十五阶段，2026-07-22：22:49，task039 完成并通过边界验收，ready marker 原子发布。旧 5090 runner、旧 finalizer、watcher 与 transition 按设计退出。5090 新 18 项队列继承 15 项并开始 task043，5080 新 27 项队列继承 24 项并开始 task040。runner 层合计达到 39/45。
28. 第二十六阶段，2026-07-22：用户授权处理结果服务器容量。仅清理 12.25 GB 未使用 Docker build cache，清理后 build cache 为 0 B，可用空间为 19,434,708,992 字节。4 个业务容器、ready marker 与结果根均保持正常，容量阻塞解除。
29. 第二十七阶段，2026-07-23：5090 完成新分片 18/18，通过 source-host postflight、严格导出、结果服务器逐文件哈希与原子提升。`rtx5090_18` 状态为 `verified`，包含 97 个源文件、10,071,444,446 字节源数据和 18 个任务目录。5090 随后按契约关机。
30. 第二十八阶段，2026-07-23：5080 的 task040 完成训练与推理，但因 producer float32 summary 与 runner float64 重算之间的 0.000020346824955552734 差异被 fail closed。queue 为 24/27，task040 未提升。确认 staging 完整且无协议进程后，按用户要求写入关机记录并正常关机，13:50 新加坡时间确认离线。
31. 第二十九阶段，2026-07-26：完成 task040 的 producer 精度契约修复、float64 独立审计和 fail-closed staging recovery。task040 从原 zero-exit `attempt_0001` 完成无重训原子提升，没有 `attempt_0002`。
32. 第三十阶段，2026-07-26：task041 完成 30 个 epoch 与测试推理后，因 282 字符 summary 临时路径触发 Windows 传统路径限制。哈希锁定的 incident-specific repair 补齐两份未落盘文件，经 staging 与正式目录双重验收后原子提升，没有 `attempt_0002`。
33. 第三十一阶段，2026-07-26：runner 以 `KEEP_COMPLETED` 保留 tasks040-041 的审计状态并启动 task042。正式进度达到 44/45，5080 为 26/27，finalizer 等待 27/27 后执行 postflight 与回传。
34. 第三十二阶段，2026-07-26：task042 完成并通过 runner 验收。5080 达到 27/27，runner 层达到 45/45。
35. 第三十三阶段，2026-07-27：修复 finalizer 的 PowerShell 至 native ssh 引号传递问题。提交 `c48f232` 经本地 250 项回归、PowerShell 5.1 AST、Windows native argv 与真实 incoming 文件树复验。`rtx5080_27` 随后完成正式原子提升，未重训、重传或改写任务结果。
36. 第三十四阶段，2026-07-27：定位 composite 复验的单任务内存峰值，提交 `5f3113b` 以数值等价的原位计算消除两份全尺寸临时数组，并在 producer 指标校验前释放 actual。提交 `93038ec` 跳过 full-window 中未使用的事件分组归约。全仓 252 项通过，独立复审未发现 P0 或 P1。
37. 第三十五阶段，2026-07-28 到 07-29：启动独立 IGSTGNN Table 4 公开材料最大忠实复现。5080 完成 HL 与 LSTM 后，正式执行切换为两台 RTX 4090 D 主机的 9 卡固定分片。正式计划、数据、上游提交、训练代码家族、自动 postflight、结果服务器复验与成功后关机均完成锁定。Orange IGSTGNN 的 24 GB 单卡 OOM 被归类为硬件容量事件，保留完整 batch 的双卡恢复控制器已启动。

---

## 第三部分：数据与统一实验管线

| 项目 | 设置 |
|---|---|
| 数据集 | XTraffic 2023，加州主线传感器 |
| 区域 | Alameda 521 节点，ContraCosta 496 节点，Orange 990 节点 |
| 输入输出 | 历史 12 步预测未来 12 步，5 分钟粒度，即 1 小时预测 1 小时 |
| 通道 | flow、occupancy、speed 三通道输入，目标为 flow |
| 切分 | 按时间 70/15/15 |
| 采样 | 事件锚定采样，每个事故事件附近取 offset 0/6/12 三个样本 |
| 指标 | masked MAE，按 all、affected、unaffected 三组报告 |
| 训练 | Adam，lr 1e-3，cosine 调度，batch 48，30 epoch，seed 42/1/2 |

2026-07-18 审计修正。表中切分比例描述不等于严格互斥。旧实现按锚点分 split，目标窗口会跨边界，少量 event_idx 也跨 split。full-window 的两个相邻边界各共享 11 个目标时刻。多个标准化脚本又读取了训练边界之后的 11 或 23 个时刻。v2 协议必须按完整目标区间和 event group 重新切分。

一个需要明确强调的协议差异：我们采用事件锚定采样，测试集全部是事故场景附近的窗口。IGSTGNN 原论文采用全时段标准滑窗，测试集绝大多数是正常时段。这个差异极其重要，详见第七部分。审计前我们曾把事件锚定协议解释为事故感知模型的主场，并据此加强标签无增益的论证。当前该解释需要在修复未来事故特征、样本错绑与缓存泄露后重新验证。

---

## 第四部分：我们做了什么，按时间顺序

### 4.1 FourierDualNet 模型

架构如下。

```
x_hist ─┬─ FFT 低频掩码，可学习，初始化在 K=3 ─→ Main 分支 GraphWaveNet ─→ y_main ─┐
        └─ 1 减掩码，高频残差             ─→ Pert 分支 GraphWaveNet ─→ y_pert ─┴─ y = y_main + y_pert
```

掩码为 7 个频率 bin 上的 sigmoid 权重，随训练自学习。两分支均为标准 GraphWaveNet，带自适应邻接。模型不读取任何事故标签、事故位置或传感器元数据。

### 4.2 复现 IGSTGNN 并发现官方代码 bug

复现过程中发现 9360 条测试预测中只有 390 个唯一标签，而 9360 除以 24 恰好等于 390。追查到官方 src/utils/dataloader.py 的多线程填充代码写的是 batch_samples[i 减 start_idx]，导致每个线程都只读 batch 前若干元素。线程数是硬编码算术 num_threads 等于 max(batch_size 整除 2, 1)，只要 batch 大于等于 4，公开发布的代码必然把有效 batch 退化为 2 个唯一样本。我们所有对比都使用修复后的 IGSTGNN，这是对它最有利、对我们最严格的设定。

### 4.3 五种架构增强尝试

为增加方法创新性，在 FDN 骨架上系统尝试了五种事故思路或调制模块。

| 配置 | 内容 | Alameda all MAE | 相对 FDN baseline 11.976 |
|---|---|---:|---|
| +A | 时间嵌入加差异化空洞卷积 | 11.913 | 噪声量级 |
| +B | 跨分支注意力 | 12.255 | 变差 |
| +E | 输入条件化频谱掩码 | 12.322 | 变差 |
| D3 | 逐节点逐步长门控融合 | 12.066 | 种子噪声内 |
| ICSF 移植 | 把 IGSTGNN 事故模块搬到 GWN | 12.384 | 零增益 |

五次全部无增益。结论是 FFT 加双骨干加调制模块这一范式已到性能上限。

### 4.4 多种子噪声带

FDN 与 GWN 在 seed 42、1、2 下各训三遍。

| 区域 | FDN learnable | GWN | 差距 | 显著性 |
|---|---|---|---:|---|
| Alameda | 12.020 ± 0.039 | 12.402 ± 0.042 | 减 0.382 | 9.4 倍噪声 |
| ContraCosta | 13.057 ± 0.066 | 13.480 ± 0.077 | 减 0.423 | 5.9 倍噪声 |

FDN 对 GWN 的优势远超种子噪声。同时 D3 的 12.066 落在 baseline 分布 12.020 ± 0.039 的 1.2 倍标准差内，确认无增益。

### 4.5 标准基准移植与等参数改正

把 FDN 与 GWN 移植到标准基准。

| 数据集 | 类型 | FDN | GWN | 差距 |
|---|---|---|---|---:|
| METR-LA | speed | 3.077 ± 0.008 | 3.083 ± 0.005 | 减 0.006，平 |
| PEMS04 | flow | 19.655 ± 0.129 | 20.340 ± 0.093 | 减 0.686，6.1 倍噪声 |
| PEMS08 | flow | 15.554 ± 0.102 | 16.062 ± 0.046 | 减 0.509，6.4 倍噪声 |

一个重要的自我改正。FDN 是双骨干，参数 622743，恰为单骨干 GWN 311368 的两倍。上面这个对比是 2 倍参数对 1 倍参数，不是等参数。补做 GWN 侧等参数三变体消融后发现，把总优势拆开，容量占七到八成，频谱路由仅占两到三成。路由的合并净效应，6 个种子点配对检验，均值减 0.137，6 比 6 种子一致为负，t 等于负 5.8，真实但只有约 0.7%。结合 STID 骨干上路由零增益与 METR-LA 速度数据上零增益，机制结论是频谱路由是一个轻量的周期注入模块，仅在周期主导信号加不建模周期的骨干这一交集内有效。它不足以单独支撑一篇方法论文。

### 4.6 引入标准 SOTA 模型 STAEformer

把 STAEformer 这个 ICCV 2023 公认 SOTA、不使用任何标签的纯 transformer 放进 XTraffic。结果见第六部分。这是改变论文定位的关键实验。

### 4.7 老师新方向，去季节化残差引导双支网络 RGDN，负结果

2026-06-22。导师会后，研究方向从事故导向收敛到分解导向。复用缓存里 train-only 且事故掩码的气候态基线 baseline_median，按日型 day_kind 与时刻 tod 索引，做去季节化，模型只预测对基线的偏离，未来基线查表已知直接加回。设计上残差支线做图传播管空间扰动，主支线节点局部管周期节律，并经一步自适应邻接图卷接收一份邻居残差摘要注入，最后相加融合。全程不读任何标签。吸取此前 2 倍参数对 1 倍参数的教训，所有变体参数严格对齐到约 31.6 万。

Alameda 种子 42，等参数消融如下。

| 变体 | 参数 | all | affected | unaffected |
|---|---:|---:|---:|---:|
| v0a 原始单 GWN | 315648 | 11.811 | 17.511 | 11.456 |
| v0b 去季节单 GWN | 315648 | 11.711 | 17.711 | 11.337 |
| v2 双支无注入 | 298136 | 11.779 | 17.810 | 11.404 |
| v1 RGDN 完整 | 308660 | 11.884 | 18.028 | 11.502 |
| v1g RGDN 门控注入 | 308661 | 11.922 | 18.097 | 11.537 |

三条读法。

第一，去季节化是唯一真增益。v0b 比 v0a 在 all 上减 0.100，等参数，且单个 GWN 就够，不需要双支。它对正常节点有用，unaffected 减 0.119，对事故节点反而轻微变差，affected 加 0.200，因为事故节点本就偏离正常态，加回基线帮不上它们。

第二，双支残差引导机制不成立。v1 比 v0b 差 0.174，注入是负担，v1 比 v2 差 0.105，双支本身 v2 对 v0b 加 0.068，落在种子噪声带内。

第三，注入是优化陷阱，已用诊断坐实。注入那张可学习邻接退化成近乎均匀，行熵比 0.987，等于给每个节点喂一份全局平均残差这条劣质捷径，设计想要的邻居重要性没有长出来。模型重度依赖它，推理时把注入强行置零，预测直接崩坏，MAE 跳到 22.6。再给一个初始化为 1 的可学习标量门控，让模型可以自由把注入关到 0，门控却停在 0.898，v1g 11.922 反而更差，逃不出陷阱。

结论。这个 RGDN 架构方向在 Alameda 上不成立，且有清楚的机制解释。唯一捞到的去季节化是个标准操作而非新方法，经典出处是 STL 季节趋势分解与交通领域的 Historical Average 基线、ASTGCN 的日周与周周期通道、Periodic Residual Learning，现代对标是 DLinear，后者的发现正是简单分解打赢复杂结构，与我们这里一致。继此前五种调制增强全部无增益之后，这次换范式的 RGDN 同样无增益，再次印证加结构打不过简单 label-free 基线。结果为单种子，方向清楚，headline 约 2 到 4 倍噪声，多种子可进一步收紧。数据见 outputs/diagnostics/rgdn_round1_results.txt。

---

### 4.8 自适应去季节化 v0c，第一个站住的架构侧增益，2026-06-27

承接 4.7。RGDN 的双支与注入都是负结果，唯一捞到的是去季节化本身。沿这条线深挖，先定位 v0b 的明确瓶颈再对症改造，得到自适应去季节化 v0c。这是项目里连续六次架构增强失败之后第一个干净、可复现、等参数、多种子的真增益，且全程不读任何标签。

2026-07-18 审计修正。模型前向没有读取事故标签，但它使用的 `baseline_median` 在拟合前通过事故记录排除了训练数据，因此严格 label-free 表述不成立。旧输入还含未来插值。本节所有增益与机制结论均等待无事故掩码 climatology 和 v2 缓存复跑。

诊断先行。v0b 在 all 上比原始单 GWN 减 0.10，但这是被拖累后的净值。它对正常节点 unaffected 减 0.119 是赚的，对事故节点 affected 加 0.200 是赔的。原因清楚，未来气候态基线假设节点回到周期正常态，但事故节点在预测窗内仍偏离基线，把这条错误的基线加回去就是在害事故节点。

对症设计。给去季节化加一个逐节点、逐时刻的自适应权重 alpha，按该节点近期相对基线的偏离幅度决定锚点用多少基线、用多少当前观测值。偏离幅度 r 为历史窗内该节点 flow 减基线的绝对值均值再除以残差标准差，是一个有物理意义的异常量，完全来自流量本身，不读标签。权重 alpha 等于 exp 负 relu r 减 r0 再除 tau，r0 与 tau 可学。锚点等于 alpha 乘未来基线 加 1 减 alpha 乘 persistence，persistence 取历史最后一帧观测值。最终预测等于锚点加模型预测的偏离量。

三条性质。第一，正常节点 r 小，alpha 趋近 1，锚点几乎全是周期基线，且 alpha 等于 1 时精确退化回 v0b，所以理论下界就是 v0b，最坏不会更差。第二，事故节点 r 大，alpha 趋近 0，锚点切换到当前真实水平，正对 v0b 害事故节点那个失血点。第三，残差头预测照常叠加，alpha 只换锚点不抢残差的活，参数只比 v0b 多 r0 与 tau 两个，是干净的等参数对照。

为什么这次没重蹈门控覆辙。此前 D3 门控与 RGDN 注入门控都退化成全局平均捷径。这里 alpha 的输入是有物理意义的异常量 r 而非自由学习的标量，参数只有两个，且 relu 把正常节点钉死在 alpha 等于 1，结构上几乎无法退化。诊断坐实，alpha 在事故节点上约 0.81，低于正常节点约 0.89，确实按异常分化，不是常数。

Alameda 结果，seeds 42/1/2 均值，等参数单 GWN 骨干，v0a 为单种子。

| 变体 | 参数 | all | affected | unaffected |
|---|---:|---:|---:|---:|
| v0a 原始单 GWN | 315648 | 11.811 | 17.511 | 11.456 |
| v0b 去季节 | 315648 | 11.681 | 17.633 | 11.311 |
| v0d 常数 alpha 混合 | 315649 | 11.559 | 17.428 | 11.194 |
| v0c 自适应 alpha | 315650 | 11.452 | 17.226 | 11.092 |

逐种子 v0c 全低于 v0d 全低于 v0b，每一层零重叠。v0c 比 v0b 在 all 上减 0.229，对噪声带 0.080 约 2.9 倍，配对 t 约负 9.7。新跑的 v0b 种子 42 为 11.721，与 4.7 节 round1 的 11.711 一致，对照复现，整条对比可信。affected 与 unaffected 同时改善，且 affected 减 0.407 比 unaffected 减 0.219 还多，v0b 那笔失血被打回来了。

消融把增益分成两块。v0d 用一个可学常数 alpha 均匀混合，比 v0b 减 0.122，约一半增益来自全局掺一点当前观测值这个标准的残差持续混合思路。v0c 在 v0d 之上按异常加权再减 0.108，另一半来自对事故节点多掺持续值。两块都真实且可加，自适应那一半不是冗余。

诚实的机制细节。优化器把 r0 学到约 0.03，tau 学到约 4，于是 alpha 等于 exp 负 r 除 4 作用在所有节点，连正常节点也掺约 11% 当前观测值。它学到的是全局持续混合且对事故节点多掺，而非设计预期里只动事故节点。这泛化出了比预设更好的解，且没退化成捷径。

三区域泛化，v0c 对 v0b，seeds 42/1/2 均值，all MAE。

| 区域 | v0b | v0c | 差值 | affected 差 | unaffected 差 |
|---|---:|---:|---:|---:|---:|
| Alameda | 11.681 | 11.452 | 减 0.229 | 减 0.407 | 减 0.219 |
| ContraCosta | 12.453 | 12.297 | 减 0.157 | 减 0.310 | 减 0.149 |
| Orange | 12.744 | 12.495 | 减 0.249 | 减 0.503 | 减 0.237 |

v0c 在三区域三战三胜，每个区域都是多种子且逐种子零重叠，Orange 的差距 0.249 最大且 affected 减 0.503 最深。三区域 affected 改善都明显大于 unaffected，与主命题一致，周期先验搞定大头，难点是事故偏离。alpha 的特化有区域差异，Alameda 与 Orange 的 r0 都塌到约 0.02，是全局持续混合加事故多掺，ContraCosta 的 r0 学到约 0.45，relu 钉死部分保住、混合更定向，但净胜利方向一致。诊断三区域一致，alpha 在事故节点约 0.83 低于正常节点约 0.90。

与 STAEformer 的效率对比，二者同为事件锚定 batch 16 协议，可直接比。三区域 all MAE，Alameda v0c 11.452 对 STAEformer 11.391，ContraCosta 12.297 对 12.116，Orange 12.495 对 12.500。读法要诚实，v0c 在 Alameda 落在噪声带内基本打平，在 Orange 略微反超，在 ContraCosta 落后约 0.18。STAEformer 综合仍是最强或持平，但 v0c 用 31.6 万到 32.5 万参数，只有 STAEformer 154 万到 201 万的约五分之一到六分之一，就在两个区域追平或反超五倍大的 SOTA，在三个区域稳定赢过自己的去季节化底座。v0c 的价值是极轻加干净增益，一个普通 GWN 加两个参数的自适应去季节化技巧。

一个比较口径的边界要讲清。v0a 到 v0d 同属 RGDN 单 GWN 管线 batch 16 配置，内部严格等参数可比，v0c 的纯 GWN 对照应取本节的 v0a 即 11.811。第五部分跨模型总表里的 GWN 与 FDN 与 STID 是更早 FDN 阶段的不同训练配置，不可与本节 v0a 到 v0d 直接相减。

边界与待办，诚实列出。第一，三区域 v0b 对 v0c 已全部完成于 2026-06-30，三战三胜。第二，正式配对显著性检验需要每跑约 700MB 的预测落盘文件，尚未拉取，留在远端盘待取，当前结论基于逐种子零重叠这一更强的直接证据。第三，骨干无关性验证，把这套去季节化加自适应 alpha 套到 STAEformer 骨干的代码已写好，见 scripts/train_staeformer_deseason.py，待 GPU 空出后跑。预判去季节那半在已有时间嵌入的 STAEformer 上大概率蒸发，参考项目内频谱路由在自带 time embedding 的 STID 上增益为零，持续锚那半可能留住，两种结果都对论文有利。

数据出处 outputs/diagnostics/adaptive_deseason_results.txt 与 outputs/rgdn 各区域 summary.json。代码为 fourier_dual_net/rgdn.py 的 AdaptiveAlpha 与单支 adaptive 与 const_alpha 分支，以及 scripts/train_rgdn.py 的变体 v0c 与 v0d。

---

### 4.9 独立轻量骨干探索，RFN 与 RAD 两次负结果，2026-06-30 到 07-01

v0c 站住后出现一个方向问题，v0c 套在 GWN 上，容易被评为 GWN 封装而非新方法。为此探索了两条独立轻量骨干路线，均为干净负结果，但它们把路线判清楚了，并产出一个重要的骨干相关性发现。

第一条，RegimeFlowNet 简称 RFN，状态转移导向，输入为 flow 与 occupancy 与 speed 三通道去季节残差，骨干为节点局部多尺度因果 TCN，不用图。Alameda seeds 42/1/2。RFN-v0 在 4.5 万参数下 flow-only 12.366，qov 三通道 12.288，qov 干净带来减 0.078。等参数控制把参数放大到 32.1 万与 GWN 对齐后，flow-only 12.350，qov 12.204，qov 增益扩大到减 0.146，但整体仍落后 GWN 原始 0.39、落后 v0b 0.52。结论一，qov 输入在弱骨干上是真增益。结论二，差距是架构性的而非容量性的，缺图传播堆参数补不回来。

顺手做了反向验证，把 qov 三通道输入加到强 GWN 骨干上，变体 v0e 等于 v0c 加 qov，v0f 等于 v0b 加 qov。结果反转，v0f 11.753 比 v0b 差 0.072，v0e 11.525 比 v0c 差 0.073。诊断是过拟合，v0e 验证集比 v0c 低但测试集反而高。这给出一个干净规律，**qov 的价值是骨干相关的**，弱骨干缺信息时 qov 是真增益，强骨干从 flow 已抽够信息时 qov 是冗余并被用于过拟合。这与项目铁律一致，强 label-free 骨干不吃外挂，频谱路由在 STID 上为零，ICSF 在 STAEformer 上为零，qov 在 GWN 上为负。

第二条，RAD-GMixer，状态自适应分解图混合网络，把 v0c 机制做成模型内部层，配多尺度直接线性时间混合器与受物理约束的有向局部道路图。Alameda seeds 42/1/2，hidden 128。rad_v0 无图 12.570，rad_v1 加局部道路图 12.486。这是全项目试过的最弱模型，比 RFN 的 TCN 还差 0.28，离 GWN-v0c 差 1.03。两个机制解释，其一多尺度线性混合器丢失 12 步窗口内的局部顺序细节，弱于因果 TCN，其二受约束的局部道路图只带来减 0.083，远小于 GWN 自适应图约 0.5 的空间建模价值，说明物理道路邻接不是这个任务的主要预测图，真正有用的是数据驱动的动态相关图。

两次负结果合起来的结构性结论。**XTraffic 上的强性能需要 GWN 或 STAEformer 级别的时空表示能力，轻量局部模型与线性混合器顶不上，独立轻量骨干这条岔路已被封死。** 数据 outputs/diagnostics/rfn_results.txt 与 rad_results.txt。代码 fourier_dual_net/regimeflow.py 与 radgmixer.py，scripts/train_rfn.py 与 train_rad.py。

### 4.10 v0c 重定位为 RADS，并在 STAEformer 上迁移成功，2026-07-01 到 07-02

两次独立骨干负结果之后，与导师对齐了重定位。v0c 不是新骨干，而是一个可插入强时空骨干的机制，正式命名 **RADS，Regime-Adaptive De-seasonalization，状态自适应去季节化重组**。核心公式为预测等于 alpha 乘周期基线 加 1 减 alpha 乘持续锚点 加 骨干输出的残差，其中 alpha 由节点近期偏离幅度决定，骨干只是残差函数的一个实现，GWN 从方法主体降级为其中一个实现。论文主线相应修正，不再说简单模型打赢复杂模型，而是说事故标签不是有效增益源，有效的是强时空骨干学出的内生状态表示，配合 RADS 修正正常态与异常态的重组方式。

决定性验证是把 RADS 套到最强 label-free 骨干 STAEformer 上。代码 scripts/train_staeformer_deseason.py，变体 s0a 原始对照与 s0c 加 RADS，通道零换成去季节残差且保留 tod 与 dow 时间嵌入，输出端 alpha 锚点同 v0c，等参数只多 2 参。三区域 seeds 42/1/2，结果如下。

| 区域 | s0a 原始 | s0c 加 RADS | 配对差均值 | 逐种子重叠 |
|---|---:|---:|---:|---|
| Alameda | 11.392 | 11.258 | 减 0.134 | 约 1.5 倍带，非零重叠 |
| ContraCosta | 12.038 | 11.881 | 减 0.156 | 零重叠，干净 |
| Orange | 12.340 | 12.248 | 减 0.092 | 配对全负，非零重叠 |

**s0c 在 Alameda 的 11.258 是全项目历史最低数字**，低于 STAEformer 基线 11.391 也低于 GWN-v0c 的 11.452。对照复现完美，Alameda s0a 均值 11.392 与旧单种子基线 11.391 一致。三区域九个种子对全部为负。Alameda 配对差负 0.041 与负 0.178 与负 0.183，ContraCosta 负 0.111 与负 0.145 与负 0.213，Orange 负 0.114 与负 0.132 与负 0.031。强度排序 ContraCosta 零重叠最硬，Alameda 约 1.5 倍带，Orange 配对一致但非零重叠最弱。ContraCosta 的 s0c 11.881 同时低于 STAEformer 该区域基线 12.116 与 GWN-v0c 该区域的 12.297。

机制解释与预判一致，STAEformer 自带时间嵌入已内化部分周期信息，去季节那半增益被吸收，持续锚那半与时间嵌入正交所以存活，故 STAEformer 侧增益约 0.14 到 0.16 小于 GWN 侧的 0.16 到 0.25，但方向稳定。alpha 诊断仍分化，两区域事故节点约 0.73 到 0.75 低于正常节点约 0.82 到 0.84。另有一个本地锚点层诊断支撑机制故事，按节点偏离强度分五桶，误差高度集中在最高偏离桶，该桶内周期锚点误差 56.2 而理想锚点选择可到 28.3，说明锚点选择的头空间正集中在 RADS 对准的异常节点上。

结论。**RADS 在两个结构完全不同的骨干上、全部六个骨干乘区域格子里都是配对正增益，GWN 三区域减 0.16 到 0.25 全零重叠，STAEformer 三区域配对全负减 0.09 到 0.16，model-agnostic 重组层的主张关账**。诚实边界两条，其一增益在带时间嵌入的骨干上会缩水，与频谱路由在 STID 上蒸发同源，其二 STAEformer 侧强度随区域递减，Orange 最弱且非零重叠。可选补充，STAEformer 上的 s0b 与 s0d 消融，正式配对显著性。数据 outputs/diagnostics/rads_staeformer_results.txt。

### 4.11 RADS 方法与性能速览，可直接对外

本节把 RADS 用一段自洽的话讲清楚，供对外汇报直接引用。

RADS 全称 Regime-Adaptive De-seasonalization，状态自适应去季节化重组。它不是一个新模型或新骨干，而是一个可插入任意强时空预测骨干的轻量机制层，解决去季节化预测在事故与异常节点上的一个系统性错误。

要解决的问题。交通预测的标准做法之一是去季节化，每个节点有一条历史统计出的周期基线，模型不直接预测流量而是预测对基线的偏离，最后把基线加回。死穴在事故节点，出事时该路段已不在周期正常态，而未来基线仍假设它回到正常水平，把这条错误基线加回就是在害事故节点。实测普通去季节化对正常节点有用而对事故节点变差。

RADS 的机制。给每个节点每个未来时刻算一个自适应权重 alpha，动态在两个锚点之间选择，预测等于 alpha 乘周期基线 加 1 减 alpha 乘当前持续水平 再加骨干输出的残差。alpha 由该节点近期偏离幅度决定，偏离越大 alpha 越小，越不信周期越锚当前状态。正常节点 alpha 趋近 1 退化为标准去季节化，事故节点 alpha 趋近 0 锚在当前实际水平。alpha 前向只由流量计算，但旧缓存中的周期基线使用事故标签清洗，故完整管线不能称为严格 label-free。关键在于骨干只是残差函数的一个实现，可以是 GWN 或 STAEformer 或任何强时空模型，RADS 不改骨干内部，只改预测目标与锚点选择。这是它骨干无关的设计含义，是否形成有效增益等待 v2 复跑。

性能汇总，全部同协议 masked MAE，seeds 42/1/2，越低越好。

骨干一 GWN 图卷积，三区域。

| 区域 | 普通去季节 v0b | GWN 加 RADS 即 v0c | 增益 |
|---|---:|---:|---|
| Alameda | 11.681 | 11.452 | 减 0.229，2.9 倍带零重叠 |
| ContraCosta | 12.453 | 12.297 | 减 0.157，零重叠 |
| Orange | 12.744 | 12.495 | 减 0.249，零重叠 |

骨干二 STAEformer 纯 Transformer 当前 SOTA，两区域。

| 区域 | 原始 s0a | STAEformer 加 RADS 即 s0c | 增益 |
|---|---:|---:|---|
| Alameda | 11.392 | 11.258 | 减 0.134，约 1.5 倍带 |
| ContraCosta | 12.038 | 11.881 | 减 0.156，零重叠 |

一句话性能总结。RADS 在两个结构完全不同的骨干与多个区域上都能稳定把误差往下压 0.13 到 0.25，且只多 2 个参数不读标签。卖点是极轻、通用、可解释、不依赖事故标签，而非碾压式超越。层级关系，RADS 是机制地基，其上依次叠加逐 horizon 门控、oracle 蒸馏与双残差头构成完整的 NDR-Net 双态预测头，后者在 Alameda 与 ContraCosta 分别做到 11.052 与 11.733 的全项目与该区域历史最低，见第 4.12 节。

### 4.12 RADS-O，oracle 门控蒸馏，有界阶梯 A1 决定性胜出，2026-07-03

背景。外部提出的 NDR-Net 双态重组模型方案经评审拆成三个可分离假设，按风险收益排成有界阶梯。A0 为逐 horizon 门控本身，A1 为 A0 加训练期 oracle 门控蒸馏，命名 RADS-O，B 为再加正常态与扰动态双残差头，即 NDR-v0，C 为状态条件动态图，因踩在当年注入陷阱的地盘上默认不做。停止线，三种子配对差对上一级全负且均值小于等于负 0.06 才继续，负 0.03 到负 0.06 为灰区默认停。此线为有界加注，不阻塞论文主线。

设计。门控从 s0c 的两参数升级为逐 horizon 三十六参数，g 等于 sigmoid 括号内 a_h 减 softplus b_h 乘偏离强度 d 减 softplus c_h 乘短期斜率 u。训练期用真实未来构造软 oracle 目标，g 星等于真实值到持续锚点的距离除以到两个锚点距离之和，含义是未来更接近周期基线则 g 星趋近 1，更接近持续水平则趋近 0。监督用 Huber 损失乘置信权重，置信为两锚点误差分离度乘以锚点间距对节点残差尺度的截断比，只有两个锚点真有差异且优劣明确的位置才被强监督。实现上有一个必要的单位校准，门控损失乘以逐节点残差尺度换算到流量单位，否则 oracle 梯度比主损失经由锚点混合作用在门控上的梯度小约两个数量级，监督形同虚设。测试期门控仍只由历史流量计算，无任何泄漏。

结果，Alameda seeds 42/1/2，配对对照 s0c。

| seed | s0c | s0f 即 RADS-O | 配对差 |
|---|---:|---:|---:|
| 42 | 11.261 | 11.110 | 负 0.151 |
| 1 | 11.310 | 11.118 | 负 0.192 |
| 2 | 11.203 | 11.119 | 负 0.085 |
| 均值 | 11.258 | 11.115 | 负 0.143 |

三种子配对全负，均值负 0.143 远超负 0.06 门槛，逐种子零重叠，停止线决定性通过。**11.115 为全项目历史最低**，对原始 STAEformer 11.392 累计改善 0.277。affected 同步改善，17.038 到 16.849。最值得注意的是稳定性，s0f 三种子极差仅 0.009 而 s0c 为 0.107，oracle 监督把门控钉到跨种子完全一致的最优点，三个种子学到的门控在事故与正常节点上的均值到小数点后三位都相同，0.532 与 0.632。机制解读，手调的 RADS 门控约 0.73 与 0.82 离最优很远，oracle 蒸馏把工作点推向重得多的持续混合，且门控由偏离强度 d 驱动，斜率项 u 学到近零。这与第 4.10 节的锚点层 oracle 诊断相互印证，高偏离节点的锚点选择头空间确实是高价值瓶颈。新脚本同时把按偏离强度分五桶的 MAE 落进 summary，分桶按输入侧信号定义，与模型无关，后续任何基线可配对比较。

当晚 B 档与 A0 消融全部完成，阶梯全线通过。s0g 双残差头对 s0f 配对差负 0.047 与负 0.072 与负 0.070，全负零重叠，均值负 0.063 压线通过，s0e 消融把归因拆干净。完整阶梯如下，每一级都三种子配对全负且逐种子零重叠。

| 阶梯 | 均值 | 该级贡献 | 机制 |
|---|---:|---:|---|
| s0a 原始 STAEformer | 11.392 | 无 | 对照 |
| s0c 加 RADS | 11.258 | 负 0.134 | 锚点自适应 |
| s0e 加逐 horizon 门控，A0 | 11.179 | 负 0.079 | 门控形式 |
| s0f 加 oracle 蒸馏，A1 | 11.115 | 负 0.063 | 监督信号，另把种子极差从 0.025 压到 0.009 |
| s0g 加双残差头，B | 11.052 | 负 0.063 | 双态残差分工 |

总计对原始 STAEformer 改善 0.340，**11.052 为全项目历史最低**。门控演化路径进一步支撑机制解读，主损失自学只能从 0.73 与 0.82 挪到 0.71 与 0.76，oracle 监督才推到 0.53 与 0.63 的最优点并跨种子钉死。按命名规则 A 加 B 都过线，**NDR-Net 成立**，克制表述为把强时空编码器改造成正常态与扰动态分离的双态预测头，不是新骨干。诚实边界一条，s0g 过线是压线，均值负 0.063 对门槛负 0.06，靠零重叠与种子一致性背书。

第二区域验证，2026-07-04，ContraCosta 阶梯完成。复合体干净复现，s0g 对 s0c 配对差负 0.176 与负 0.162 与负 0.107，全负零重叠，均值负 0.148，端到端 12.038 到 11.733 改善 0.305，与 Alameda 的 0.340 一致，11.733 同为 ContraCosta 历史最低。但中间级归因呈区域相关，Alameda 上三个组件各贡献一块，ContraCosta 上门控形式与 oracle 单独约为零，增益几乎全由双残差头扛，负 0.120 且零重叠。机制解读，oracle 在 CC 单独不赚钱但把门控钉在 0.52 与 0.62，双残差头站在这个钉死的门控上发力，组件间有协同，复合体总量跨区域稳定而单组件归因随区域移动。论文表述应以复合体两区域零重叠为主张，以 Alameda 阶梯为归因研究，并如实说明 CC 的归因差异。数据出处 outputs/diagnostics/rads_staeformer_results.txt。

---

## 第五部分：核心结果总表

以下为审计前历史总表。下表模型曾在统一的事件锚定管线上评估，但共同使用了含未来插值的旧缓存，不能再写成当前可直接横向比较的有效结果。STAEformer 为 15 epoch，其余为 30 epoch。

| 模型 | 用标签 | Alameda all | ContraCosta all | Orange all |
|---|---|---:|---:|---:|
| STAEformer | 否 | 11.391 | 12.116 | 12.500 |
| FDN learnable | 否 | 11.976 | 13.133 | 12.996 |
| GWN | 否 | 12.402 | 13.480 | 13.013 |
| STID | 否 | 12.727 | 13.466 | 13.792 |

IGSTGNN 用全套事故标签，在旧 matched windows 子集上的历史成绩如下。2026-07-18 专项审计确认，该链路的 H5 转换器将交通窗右移 11 步，并将 79.01% 到 97.08% 的样本错绑到其他活跃事件。Setting A 的加 11 匹配只对齐了交通窗，没有修复事故语义。因此下表不能支撑事故标签无增益，affected 与事故类型结论作废待重跑。

| 模型 | 用标签 | Alameda all | ContraCosta all | Orange all |
|---|---|---:|---:|---:|
| IGSTGNN 修复版 | 是 | 13.074 | 13.720 | 13.765 |
| FDN 同窗口 | 否 | 12.298 | 13.748 | 13.296 |

历史读法是 label-free 阵营内部 STAEformer 第一，FDN 第二，GWN 第三，STID 第四。历史 matched windows 表曾被解释为 FDN 在 Alameda 与 Orange 优于 IGSTGNN，ContraCosta 持平。当前状态是两张表均受旧缓存问题影响，matched windows 表又叠加事故锚点错绑。所有排序和标签价值主张都等待 v2 复跑。

---

## 第六部分：STAEformer 结果及其意义

以下是 STAEformer 在旧 XTraffic 缓存上训练 15 epoch 的历史结果。绝对数值与跨模型结论等待 v2 复跑。

| 区域 | all MAE | affected | unaffected | 参数量 |
|---|---:|---:|---:|---:|
| Alameda | 11.391 | 17.161 | 11.031 | 1560372 |
| ContraCosta | 12.116 | 18.337 | 11.809 | 1536372 |
| Orange | 12.500 | 18.107 | 12.226 | 2010612 |

Orange 区域曾于 2026-06-23 在审计前旧管线的云 5090 上补齐，all 12.500，affected 18.107，unaffected 12.226，参数量 2010612。该记录不能与 2026-07-20 当前正式 v3 的 5090 混用。旧三区域表当时完整，但相关绝对数值与模型比较均已冻结，等待 v3 重跑。

意义分两面，都要诚实陈述。

不利的一面。STAEformer 赢了我们的 FDN，Alameda 领先 0.585，ContraCosta 领先约 1.0，而且只训了 15 epoch。FDN 是 XTraffic 最强 label-free 模型这个说法被证伪。这也是我们几轮前就提醒、并建议论文主张不要赌在 FDN 最强上的原因，现在那个担心成真。同时需要诚实指出，STAEformer 参数量 156 万，约为 FDN 的 2.5 倍，它赢 FDN 有一部分来自容量。

有利的一面，而且这一面把论文救活了。论文的核心论点从来不该是 FDN 最强，而是事故标签无增益。STAEformer 这个结果对该论点是最强的一记证据。当前标准基准 SOTA，完全不看任何事故标签，在 XTraffic 上不输甚至优于用全套事故标签的 IGSTGNN。证据从我们自己的 FDN 赢 IGSTGNN，升级成社区公认的最强 label-free 模型超过 label-using 模型。审稿人无法再说也许是你的模型恰好对路，因为现在赢的是公认 SOTA。

---

## 第七部分：IGSTGNN 原论文核对

我们逐页读完了 IGSTGNN 的 KDD'26 正式论文，标题 Incident-Guided Spatiotemporal Traffic Forecasting，作者为北航 Lixiang Fan 等，代码即我们复现所用、含 bug 的那个仓库。

### 7.1 他们写了什么

主张是现有模型只学历史规律，忽略事故这种突发扰动，他们显式建模事故影响。两个核心模块，ICSF 用注意力加一个空间关系张量把事故初始空间影响注入最后一个历史时刻，TIID 用高斯衰减建模事故影响随时间消散，衰减参数固定为 1.0。在 XTraffic 三个区域达到 SOTA，并声称 ICSF 与 TIID 可即插即用提升别的模型。同时发布了 XTraffic 数据集。

他们论文 Table 4 报告的平均 MAE 为 Alameda 12.69，ContraCosta 13.43，Orange 13.13，号称比第二名 DSTAGNN 的 13.45 好 5.65%。

### 7.2 他们发这篇论文的目的

把话说透。他们的目的是默认事故标签一定有用，然后把用好标签这件事做到极致，设计专门模块，跑大量实验，顺便发布一个数据集。这是一个常见的、能发表的工程范式，故事自洽，有数据贡献，所以能中 KDD。

但他们跳过了最该问的那个问题，就是标签到底有没有用。他们做了去掉模块的消融，但从未做过同一个模型喂标签对比不喂标签的干净对照。我们的项目就是回过头补做了这个被跳过的实验，发现答案是否定的。

### 7.3 三个可论证的漏洞，每个都有我们的实验证据

审计后边界。官方 dataloader 的线程索引 bug 属于静态代码事实，不受本次数据审计影响。由本仓库旧 XTraffic 缓存和旧 IGSTGNN adapter 得出的性能证据则全部待重跑。尤其是 matched-window、affected 与事故类型证据当前无效。

漏洞一，基线过弱导致 SOTA 主张注水。他们对比的最强模型是 2022 年的 DSTAGNN 与 D2STGNN 与 BiST，没有放入 STAEformer 这个 2023 年公认 SOTA。我们一跑，label-free STAEformer 在 Alameda 是 11.391，已低于他们论文自报的 IGSTGNN 12.69。他们的 SOTA 是和过时模型比出来的。同时注意，他们报告的 GWNET 基线 Alameda 平均 17.76，比我们正常训练的 GWN 12.40 高出约 5 MAE，系统性偏高，暗示其基线可能未调到位。

漏洞二，缺少去标签对照。我们把他们的 ICSF 与 TIID 模块忠实移植到干净的 GWN 上，12 个 run，三种子，全部零增益。Alameda 上 GWN 12.402，加 ICSF 12.384，加 ICSF 与 TIID 12.391，差异都在种子噪声内。即便在他们整模型唯一赢过 FDN 的格子，即 ContraCosta 碰撞类窗口的受影响节点，ICSF 挂在 GWN 上也只移动 0.02 到 0.12，在噪声内。说明他们的优势并非来自事故模块本身。

漏洞三，代码 bug 让主表数字本身存疑。他们用 V100 加 batch 48 训练，正好触发 num_threads 等于 batch 整除 2 的 dataloader bug，有效 batch 退化为 2。若 Table 4 是用公开代码跑的，主表数字就是带 bug 训练的产物。论文表述需克制，我们能证明的是公开代码如此，不能断言作者跑论文时用的就是这份代码。

### 7.4 协议差异的精确核对

本节记录审计前协议与结果。2026-07-18 后有三项修正。第一，full-window 的三个模型共享同一旧缓存，因此都受未来插值影响，且相邻 split 共享 11 个目标时刻。第二，STAEformer 加减 ICSF 的事故输入读取最终 duration，不满足起报时因果可用性。第三，H5 转 IGSTGNN 的 Setting A 交通窗虽经加 11 对齐，事故锚点与事故时刻语义仍严重错误。以下数字只作历史记录。

我们的旧事件锚定采样只在事故窗口附近取样，sample_start 等于事故起点加 offset，offset 取 0/6/12，整个测试集都是事故场景，Alameda 约 9000 个样本。他们的全滑窗在全时段每个时刻取样，绝大多数样本是正常场景，样本量几万。这就是当时复现的 IGSTGNN 为 13.07 而非他们的 12.69 的协议原因。2026-06-25 曾在旧云 5090 上完成标准全滑窗实验。该实验属于审计前旧管线历史，与 2026-07-20 当前正式 v3 的 5090 训练不是同一批实验。

协议，标准滑窗十二进十二出，整条时序七比一比二做时序切分，训练段统计做 z-score，masked MAE all 口径。三模型共用同一个 loader，sample_start 与 split 逐元素对齐，train 与 val 与 test 窗口数为 73567 与 10510 与 21020。结果如下，单位 MAE，seed 42，含早停。

| 区域 | GWN 无标签 | FDN 无标签 | STAEformer 无标签 | IGSTGNN 主表 用标签 |
|---|---:|---:|---:|---:|
| Alameda | 11.052 | 10.735 | 10.382 | 12.69 |
| ContraCosta | 11.314 | 11.132 | 10.750 | 13.43 |
| Orange | 11.641 | 11.293 | 10.856 | 13.13 |

三点。第一，九格全部远低于 IGSTGNN 主表自报数，连最弱的无标签 GWN，三十一万参数，都在三区域全胜，STAEformer 领先约 2.3 到 2.7 MAE。第二，FDN 在三区域全部赢 GWN，差值 0.317 与 0.182 与 0.348，STAEformer 小于 FDN 小于 GWN 单调一致，与事件锚定协议下的排序一致。第三，诚实边界，这是标准全滑窗协议对照，不是逐字复现 IGSTGNN 的切分，因为其单文件 incidents_data.npy 的切分预处理脚本不在其放出的仓库里，无法逐字重现。全滑窗数比事件锚定低，因为全滑窗含大量正常车流，事件锚定测试窗更难，两协议不可直接互比，但各自内部排序一致。数据落盘在 outputs 下的三个 fullwindow 目录，每区一个 summary.json，注意该协议无标签，affected 与 unaffected 按设计为空，只取 all。

决定性实验二，STAEformer 加减 ICSF，2026-06-26 完成。把 IGSTGNN 的 ICSF 模块忠实移植注入最强 label-free 骨干 STAEformer 的归一化 flow 通道最后一帧，事件锚定协议同窗 A/B，seed 42，方法与 GWN 加减 ICSF 一致。

| 区域 | base all | 加 ICSF all | Δ 等于 ICSF 减 base |
|---|---:|---:|---:|
| Alameda | 11.283 | 11.309 | +0.026 |
| ContraCosta | 11.968 | 11.968 | +0.000 |
| Orange | 12.436 | 12.375 | −0.061 |

三个 Δ 都极小且符号混杂，Alameda 略差，ContraCosta 完全相同，Orange 略好，最大幅度 0.061，全在种子噪声带 0.1 到 0.2 以内，单种子下无法与零区分。结论，事故标签对社区公认最强 label-free 模型 STAEformer 同样没有任何稳定增益，与 GWN 加减 ICSF 零增益完全一致。这把论文最硬的一锤补全，证据从 FDN 与 GWN 加标签无用，升级到最强 label-free 模型加标签也无用。诚实边界，单种子，不夸大成加标签必然更差。数据 outputs/baselines/staeformer_icsf 下每区的 base 与 icsf 两个 summary.json。

---

## 第八部分：完整证据清单

所有结论都有落盘文件支撑，复现命令见项目根目录的项目总结报告。

1. FDN 对 GWN，三区域全部显著，配对 t 检验、Wilcoxon、置换检验三者一致，多种子噪声带 6 到 9 倍背书。
2. label-free 模型在 matched windows 上不输用标签的 IGSTGNN，Alameda 与 Orange 明显优于，ContraCosta 持平。
3. STAEformer 这个标准 SOTA，label-free 即超过用标签的 IGSTGNN。
4. IGSTGNN 官方 dataloader bug 取证，硬编码线程数，无条件触发。
5. ICSF 与 TIID 即插即用主张复现失败，12 个 run 零增益。
6. 事故类型拆解，危险物类窗口占 42% 到 58%，稀释一切聚合指标，IGSTGNN 唯一赢的格子与其事故模块无关。
7. 五种架构增强系统性无增益。
8. 频谱路由的诚实结论，等参数下容量占七到八成，路由仅占两到三成，约 0.7%，仅在周期主导信号加周期盲骨干交集有效。

---

## 第九部分：诚实的可发表性评估

不吹。我们手里是一篇反驳加复现失败加负结果的论文。真正的卖点不是被动的标签没用，而是主动纠正文献，即事故感知 SOTA 的优势是 bug 加不可复现的模块造成的假象，一个标准 label-free 模型就能打赢它。主动纠错比被动负结果好发得多。

一个审稿人必问、目前尚未排除的缺口。我们只测了聚合与 affected 节点的流量 MAE。审稿人会问，XTraffic 是为事故感知预测专门造的，你确定在它作者关心的具体场景下标签也没用，也许标签对预测拥堵起始或恢复时间有用，只是对流量 MAE 没用。这个我们没有排除，应在后续实验中正面回应。

校准过的 venue 预期。这条路大概率是一篇扎实的应用或基准或复现性论文，对标 IGSTGNN 那个级别的应用赛道、数据集赛道，或好的交通领域期刊是够的。冲顶会主赛道的纯方法赛道，以现在的内容不敢打包票。这是诚实话。

### 4.13 P1 检测性诊断，门控不是事故检测器，但挖出主命题最直接的机制证据，2026-07-06

为评估把方法升级为预测加检测双任务的可行性，做了一个零训练的分析。用 Alameda 阶梯九个跑落盘的 gate 诊断，把 1 减 gate 当无标签异常分数，用 affected 标签当考卷，按事件后 0 与 30 与 60 分钟三档评估。表面读数 AUC 约 0.68，前 5% 精度约为基率两倍。但三刀下去检测器主张死亡。第一，gate 的 AUC 与裸输入偏离量 d 完全相同，路由器对排序无附加值。第二，AUC 在事件后一小时内完全不增长，真事件信号必随事件展开而增强。第三，决定性分解，纯节点身份即哪些走廊常年偏离就有 0.647，剥掉节点画像后的超额偏离只剩 0.55，逐样本更是 0.523 接近抛硬币，即被标注的事故在流量偏离里几乎不可见，事发一小时仍无签名。gate 方差分解显示其约一半是节点静态特质、一半是对任意来源流量偏离的时变响应，但时变那半与事故标签不对齐。

判决与收获。作为方法的双任务腿被预注册停止线杀死。但副产品是全项目对主命题最直接的机制证据，它解释了整条因果链，事故标签无法帮助预测是因为标签本身不携带流量动态信息，affected 节点误差高是因为它们是常年难预测的走廊而非被事件扰动的节点，RADS 真正做的是适应节点的持久可靠性画像加任意来源的实时偏离，而非识别事故。论文中 gate 应表述为节点可靠性与偏离自适应路由器，不再称 regime 或事故检测器。数据出处 outputs/diagnostics/p1_gate_detection.txt。

### 4.14 可预测性地板与可回收误差地图，2026-07-06

为回答还能否设计性能更优的模型，做了零训练的地板分析。两个噪声地板启发式，Poisson 计数地板 8.85，局部 bin 噪声地板约 10.61，真实不可约地板大概率在 9 到 10.6 之间，而当前最好的 NDR 为 11.05。含义，任何未来模型在 all MAE 上总共只剩约 0.5 到 2.2 的可赢空间，这定量解释了为什么本项目与整个领域在此类基准上的单篇增益都只有 1 到 3 个百分点，指标已经贴近物理地板。按输入偏离五分位的可回收缺口地图显示，几乎全部可回收误差集中在最高两个偏离桶,桶五实际 21.16 对地板约 12.8 缺口 8.4,桶四缺口 4.7,低偏离桶已在启发式地板之下即该处车流强欠散、比 Poisson 更规律。结论,任何更优模型必须在高偏离桶取胜,别处无仗可打,这与第 4.10 节锚点层 oracle 诊断相互印证。数据出处 outputs/diagnostics/predictability_floor.txt。

杠杆实测,2026-07-06 当晚。L1 三种子集成,把 s0g 的三个 checkpoint 预测平均,Alameda all 从均值 11.052 降到 10.888,减 0.164,affected 从 16.728 降到 16.497,**全项目首个低于 11 的数字**,零训练零新颖性纯方差削减,同时把与地板的剩余距离压缩到约 0.3 到 1.9。L2 验证集拟合的逐节点逐 horizon 偏差校正为净负,三种子 all 全部变差 0.048 到 0.143,集成上加 0.085,尽管验证集上拟合出的偏差幅度约 1.1,减掉它反而伤测试,说明 best-val 选点已消掉系统性偏差,剩余是期段特异噪声。这给 P1 补上机制推论,常年难预测的走廊是慢性方差而非慢性偏差,校正类方法无仗可打,L2 判死。数据出处 outputs/diagnostics/l1l2_results.txt。

### 4.15 P2 锚点库有界加注，一晚被停止线干净毙掉，2026-07-07

方法创新的最后一注按约定只给一次机会。s0h 在 s0f 之上把两锚点扩成六锚点库,周期基线、persistence、上游邻居水平、下游邻居水平、每节点弛豫衰减、昨日同刻,配泛化的逆误差 oracle 软目标路由。Alameda 三种子结果 11.756,对 s0f 配对差正 0.581 与正 0.642 与正 0.698,三对全正大幅劣化,高偏离桶反而从 21.2 恶化到约 23,预注册停止线直接毙掉,无灰区无续命。

事后诊断三条,写进论文的诚实消融。其一,六个相关锚点上的软 oracle 目标把路由抹平,各锚点权重近乎均匀且 affected 与 unaffected 几乎无差,六锚点的模糊平均是比两锚点果断混合更差的锚,K 等于 2 的 oracle 有效正因目标尖锐。其二,邻居水平锚点错定,别的路的绝对流量水平与本节点量纲不符,正确形式应是偏离空间即本节点基线加邻居标准化残差乘本节点尺度。其三,昨日同刻原始值携带日间噪声,中位数基线本来就是为了平滑它。按一次性有界加注纪律,方法创新线就此关闭,RADS-O 与 NDR-Net 证据不受影响,论文定位为发现型。数据出处 outputs/diagnostics/p2_anchor_bank_results.txt。

### 9.1 RADS 与 NDR 的新颖性查证，2026-07-06

写作前做了文献查证，结论必须诚实。RADS 与 NDR 的机制组件全部是已发表的积木，没有一篇与我们完全一致，但这属于弱新颖性。去季节化即基线加偏离，出自 Williams 等的自适应季节模型，已有数十年。持续锚点与周期锚点的切换，出自电网预测的 adaptive seasonal persistence，甚至有专利。regime 门控即正常态与扰动态区分，出自交通领域基于变点检测的 regime-switching 模型。最需要警惕的是，2025 年已有 trend-event 双通道解耦的交通预测工作，以及 STWave 与 AAAI 2025 的时空趋势季节分解网络，把趋势当正常态、事件当扰动态做双通道预测，与 NDR-Net 的正常扰动双残差头概念高度重合，区别仅在它们未做骨干无关验证也无 oracle 蒸馏。跨领域还有 switch-gated 双头即稳定分支加转移残差分支的同型设计。

真正还能主张的新颖点只有两处，其一是 oracle 门控蒸馏，用未来真值构造软目标监督门控，未检索到完全一致的，但本质是预测组合与门控的一个变体，其二是系统的骨干无关跨骨干跨区域即插即用验证，两个结构完全不同的骨干乘三区域，这个验证的系统性比机制本身更有分量。

战略含义与既定定位一致，不是坏消息。作为纯方法论文卖，新颖性薄且有并发工作，审稿人会拿上述几篇质疑，风险高。作为复现与应用论文里的建设性配菜卖则完全成立，主命题是事故标签无增益这个反证，RADS 与 NDR 只是标签没用但从流量识别状态有用的正面例证，机制是否首创不影响该论点。写作动作，related work 主动且诚实地引用上述近似工作，把贡献表述为在事故标签无增益命题下的一个轻量可解释即插即用状态重组层，而非全新机制。精确引用，写 related work 时以全文为准。其一 STWave，When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks，ICDE 2023，Fang 等，用小波把交通解耦为稳定趋势与波动事件两通道各自建模再融合，是与 NDR 正常扰动双头概念最接近的先行工作。其二 STDN，Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting，AAAI 2025 第 39 卷 11463 至 11471 页，Cao 等，在网络内部对每节点做趋势与季节分解。其三 HyperD，Hybrid Periodicity Decoupling Framework for Traffic Forecasting，arXiv 2511.09275，2025 预印本，周期与残差双支解耦。另有经典线，Williams 自适应季节交通模型，交通 threshold autoregression 即 regime-switching 预测，电网 adaptive seasonal persistence 专利。与 NDR 的区别要在论文里写清，上述工作都是在输入侧做信号分解后双通道处理，NDR 是共享编码器在输出侧做锚定双头,加训练期 oracle 门控蒸馏与跨骨干即插即用验证,后两者未检索到先例。

---

## 第十部分：接下来要做什么

2026-07-06 时项目曾判断全部实验证据关账。2026-07-18 数据有效性审计推翻了这一状态判断。2026-07-19 对 v2 的后续全面复核又确认，仅修复最初泄露点仍不足以保证样本唯一性、验证选择和全链路结果身份正确。v2 队列已经停止，四个 v2 preliminary summary 继续标记为 contaminated。v3 正式缓存已完成本地两次、5080 一次与 5090 一次严格审计，结果均为 verified。

runner 层现已完成或重新验收 45/45。5090 新分片为 18/18，5080 新分片为 27/27。两者分别以 `rtx5090_18` 与 `rtx5080_27` 回传到结果服务器，marker 状态均为 `verified`。5080 finalizer 的引号传递故障只影响收尾复验，提交 `c48f232` 修复后复用了全部 152 个已上传文件，没有重训、重传或改写任务结果。两个结果根保持独立，只读 composite index 已于 2026-07-27 18:28:27 新加坡时间完成验证。索引确认两分片互斥、并集为 canonical 45。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，索引 SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。论文统计分析已于 2026-07-27 完成：同机配对、多种子统计与窗口级显著性检验的输入全部按 composite index 逐份复验，产物见 `outputs/protocol_v3_noleak_dedup/analysis/`。窗口级结论为：s0c 显著优于 s0a（三区域三种子全部 p<0.01），s0g 小幅优于 s0f，ICSF 注入效应接近零（Alameda 不显著，ContraCosta 与 Orange 窗口级显著但幅度约 0.02-0.03），full_window 排序 staeformer < fdn < gwn（全部显著）。这些仍是描述性统计，是否恢复论文主张由用户决定。最先落盘的六项 seed42 指标仍只能用于方向性观察。审计前最低历史数字 Alameda NDR-Net 11.052 仍不得作为当前有效结果。

已经完成的代码与管理事项如下。

1. 冻结 `outputs/dist_net/region_data/` 与全部旧 XTraffic 输出。有效性状态写入 `docs/audits/result_validity.csv`，不删除也不覆盖旧 summary。
2. 为 v3 缓存定义 schema 3，记录原始文件哈希、切分边界、缺失处理、climatology 范围、Git SHA、dirty 状态、缓存 bundle 与配对身份。
3. 将全年双向插值改为 prefix-invariant 的因果缺失处理。
4. 建立完全不读取事故文件的 train-only climatology。
5. 按完整目标区间切分，并按全部活跃事件身份隔离事件样本。
6. 将 deployable incident inputs 与 analysis-only labels 物理拆开。
7. 让 scaler 和残差统计严格使用显式 train_end，并修复 IGSTGNN 的 forecast origin、目标对齐、事件绑定与目标缺失掩码传播。
8. 对同一预测时刻去重，按最近锚点与稳定事件身份选择主事故，并用全部活跃事故计算 affected 并集。
9. 将验证 MAE 统一为所有有效观测点的总误差除以总有效点数，同时修复反向邻接归一化与 blind 模式的分析标签隔离。
10. 增加数据、训练与远程运行完整性回归测试。Mac、5080 与 5090 的完整测试结果均为 143 项通过，另有 1 条既有警告。
11. 将最终训练代码 SHA 锁定为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`，并在 5080 与 5090 建立相同 SHA 的 clean checkout。
12. 完成缓存归档 SHA-256、bundle ID、bundle digest 与远端严格审计复验。本地两次、5080 一次与 5090 一次严格审计均为 verified。
13. 在独立 v3 结果目录启动严格串行队列，runner 对每项 prediction artifact 与 summary 执行复算和身份校验。
14. 将正式训练入口默认切换到 `region_data_v3_noleak_dedup` 与 `protocol_v3_noleak_dedup`，旧缓存默认拒绝加载。
15. 增加正式运行的代码 SHA、缓存 bundle、任务清单、新鲜输出目录与 summary 身份校验。
16. 整理根目录、脚本索引、文档索引、远程启动文件和实验室共享配置。
17. 完成三地区正式 v3 traffic、samples 与 `event_rel_feat` 构建。正式路径为 `outputs/dist_net/region_data_v3_noleak_dedup`。
18. 在 staging 与正式路径各执行一次严格缓存审计，两次均为 verified。cache bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。
19. 验收正式事件样本。Alameda 为 45760，split 为 32001、6954、6805。ContraCosta 为 23337，split 为 16414、3472、3451。Orange 为 60015，split 为 42372、9098、8545。
20. 在 5080 完成并验收 tasks 001 至 004，保留四项 v3 seed 42 的 prediction artifact、summary 与日志。
21. 主动结束 5080 上尚未形成 summary 的 task 005，将退出码 1 明确归类为训练迁移记录。
22. 验收 32 GB RTX 5090 的代码、依赖、测试、缓存与 Orange batch 16 显存条件，并完成 tasks 005 至 006。
23. 在旧 5090 队列的任务边界安全停止，识别并排除未形成 summary 的 task 007 零字节日志和未完成 staging。
24. 建立固定双机分片、source-host postflight、严格导出 allowlist、只读 composite index 和双机自动收尾契约。
25. 启动 5080 的 24 项分片与 5090 的 21 项分片。两个分片互不交叉，完整覆盖 canonical 45，所有直接比较组保持同机。
26. 完成 5080 的 24 项固定分片。runner 退出码为 0，source-host postflight 对 24 项复验通过，严格导出 allowlist 包含 134 个文件，共 3,372,164,933 字节。
27. 批准并布防 27/18 重分片。固定任务表、source-host postflight、两端 finalizer、ready marker、原子 transition 与 composite 契约均已更新。控制提交为 `60405a9`、`e191c27` 和 `30bcb81`。本地回归为 230 项通过和 1 条既有警告，5080 真实 PowerShell AST 与退出码冒烟验证通过。
28. 在 task039 完整验收后完成 27/18 原子切换。5090 继承 15 项并开始 task043，5080 继承 24 项并开始 task040。旧控制进程均按设计退出。
29. 在用户授权后仅清理结果服务器未使用的 Docker build cache。build cache 从 12.25 GB 降为 0 B，4 个业务容器保持运行，可用空间增至 19,434,708,992 字节。
30. 完成并回传 5090 的 18 项分片。结果服务器验证 97 个源文件、10,071,444,446 字节源数据和 18 个任务目录，transfer marker 为 `verified`。5090 按计划关机。
31. 对 5080 task040 的校验失败完成只读诊断，确认是 producer float32 与 runner float64 的指标精度契约不一致。保留完整失败状态与 staging，不把 task040 计入正式结果。
32. 确认 5080 没有训练、校验或上传进程，写入关机状态记录，并按用户要求正常关机。
33. 完成 producer 精确复现、float64 独立审计、strict JSON 与显式 fail-closed staging recovery。控制提交为 `dfb6b7f` 与 `33fae6c`。
34. 完成本地 247 项全仓回归、独立代码复审、5080 shadow validation、控制包哈希部署与独立 ops import 复验。
35. 在 CMD 探针发现环境变量展开缺陷后，将恢复任务改为由 Python 直接读取。保存异常 queue 快照，并从多份落盘证据原子恢复正式 queue。没有 trainer 启动，task state 与 artifact 未改变。
36. 从 task040 `attempt_0001` 完成无重训 recovery。恢复前后 summary、checkpoint、prediction artifact 与 train log 哈希一致，没有创建 `attempt_0002`。随后启动 task041。
37. 定位 task041 在训练与推理完成后的 Windows 临时路径长度故障。以 TDD 缩短 summary 临时名，并增加 completed state 幂等保留与哈希锁定的 incident-specific repair。提交为 `e7bf755`，本地全仓回归为 249 项通过与 1 条既有警告。
38. 在 5080 完成 task041 最终 dry-run。正式 cache 全量对照、prediction artifact、checkpoint、CUDA decomp mask、summary bytes 与 config hash 均通过，dry-run 前后证据哈希不变。
39. 从 task041 原 `attempt_0001` 原子恢复正式结果。staging 与正式目录均通过 runner 验收，没有创建 `attempt_0002`。runner 随后以 `KEEP_COMPLETED` 保留 tasks040-041 状态并启动 task042。
40. task042 以 `attempt_0001` 正常完成并通过 runner 验收。5080 达到 27/27，runner 层达到 45/45。
41. 以 TDD 修复 finalizer 从 PowerShell 调用 Windows native ssh 时的引号传递问题。提交 `c48f232` 通过本地 250 项回归与三层真实 Windows 复验。
42. `rtx5080_27` 在结果服务器完成 152 个文件的逐文件复验与正式原子提升。transfer marker 状态为 `verified`，SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。
43. 以 TDD 降低 composite 复验内存和无效归约。提交 `5f3113b` 与 `93038ec` 保持 float32 producer 精确语义、2 ULP 契约、full-window 的 `all` 指标校验和全部正式结果身份不变，全仓回归为 252 项通过与 1 条既有警告。
44. 在结果服务器重启后复验两份正式 transfer marker 与 source-host postflight。旧 composite 尝试未产生输出，终止原因按 unknown 保存到 incident record，没有覆盖或删除旧证据。
45. 以控制提交 `93038ec` 完成只读 composite postflight。服务退出码为 0，索引验证 canonical tasks 001 至 045 各一次、5080 与 5090 分片互斥、任务并集为 45，以及训练 SHA、cache bundle、manifest 与 source-host postflight 身份一致。index 的 `status` 为 `verified`，矩阵有效性状态为 `verified_host_stratified_45`，索引 SHA-256 为 `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe`。
46. 建立独立 `igstgnn_table4_official_v1` 复现线，冻结 39 项清单与三个上游提交，完成官方数据 70%、15%、15% 顺序切分和五文件 artifact 契约。
47. 在 5080 完成 Alameda HL 与 LSTM。旧 DCRNN pilot 在切换时完成 epoch 38，但没有正式目录。计划任务已禁用，状态与 staging 作为 superseded incident 保留。
48. 完成两台 RTX 4090 D 主机的 9 卡固定分片、Linux host runner、source-host postflight、结果服务器严格回传和成功后关机闭环。分片计划 SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。
49. 两台主机完成 27 个正式数据文件的逐文件 SHA-256 复验。首次所有权安全门拒绝保留为 incident，修正 UID 后重新验收并正式启动。
50. 识别 Orange IGSTGNN 的 24 GB 单卡容量不足。提交 `f43fb58` 保留完整 batch 并使用双卡 DataParallel，提交 `25b5307` 建立资源等待、task039 恢复、queue 重验、全主机 postflight、回传和关机控制器。本地全仓为 440 passed 和 2 个既有 warning。
51. 复核双机九卡落盘状态，确认原 runner 已完成并逐任务验收 21/39。四卡机为 13/26，五卡机为 8/13。其余 18 项中 7 项实际失败，11 项因 shard fail closed 未启动。结果服务器尚无正式 Table 4 主机结果。
52. 定位 STGCN 的确定根因。正式发布邻接矩阵已经为零对角线，LargeST STGCN 入口仍重复减单位矩阵，导致缩放拉普拉斯含 NaN。上游 masked 指标把 NaN loss 清零，正式有限值检查最终拒绝发布无效预测。
53. 完成 24 GB 恢复层。提交 `b266c89` 建立带独立 provenance 的恢复 runner，提交 `05e2951` 增加 DGCRN QR 预热，提交 `1a5582f` 让主卡接收较小余数分片，提交 `af6456a` 以全批有效点加权的两次微前向保持 Orange D2STGNN 的有效 batch 64。
54. 完成六组真实远端训练探针。STGCN GSO 全为有限值。Alameda D2STGNN 与 IGSTGNN 四卡通过。Orange DGCRN、D2STGNN 与 IGSTGNN 五卡通过。最高探针显存为 Orange D2STGNN 的约 13.5 GiB。所有探针使用临时目录，没有发布正式结果。
55. 完成本地 446 项全仓回归和 2 个既有 warning。两台 Linux 主机的训练相关门各为 158 passed、12 个平台跳过和 1 个既有 warning。完整 Linux 复现测试中的 macOS sandbox 用例和 NumPy 2 专用构造用例不属于训练门。
56. 锁定恢复前 21 项 artifact。四卡机 65 文件清单 SHA-256 为 `3e89aac6c8b5ebbf2bc8d2df4cc3f518587418464d9a49fd8b349e93967f4513`，五卡机 40 文件清单 SHA-256 为 `ee52c8e11ba150c3d49b65cf8d0efe480e46df0f5134ff310b3de14a440c1c61`。
57. 于 2026-07-29 10:57 新加坡时间启动正式恢复。四卡机控制器 PID 为 `39176`，从 task005 开始。五卡机控制器 PID 为 `40326`，从 task031 开始。恢复完成后会自动续跑完整分片，只在 source-host postflight、结果服务器逐文件复验和原子提升全部通过后关机。
58. 截至 2026-07-29 17:05，原矩阵已正式落盘并由任务 runner 验收 34/39。四卡机为 25/26，仅余 task021 Contra Costa STTN。五卡机为 9/13，正在执行 task036 Orange DGCRN，之后仍需 task037、task039 与 task034。结果服务器仍没有正式主机目录。
59. 在不修改原 39 项清单的前提下建立 NDR-Net Table 4 独立扩展。扩展为 NDR-Net 乘三地区乘种子 42、1、2，共 9 项。扩展清单 SHA-256 为 `2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4`，配置 SHA-256 为 `6ca46be0fb76543a61a8bd32996c7598ac674bbe239b701d1a92481f63be9422`，九卡分片计划 SHA-256 为 `f20be0db02fbd04732640cd5b024f7012ce6a16ea155d6a90c0962839f857522`。claim scope 独立标记为 `same_protocol_project_method_extension`。
60. 完成官方标准化 flow 到原始流量单位的逆变换适配，只用 train split 拟合周内时段中位数基线，并按全部有效点累计验证 MAE。Mac 真实数据适配与 MPS 前向反向探针通过，全仓为 455 passed 和 2 个既有 warning。两台 Linux 扩展专项门各为 33 passed。扩展部署在独立 checkout，未改写原矩阵代码。
61. 为原 finalizer 部署关机延期哨兵。第一次扩展控制器因正常 Python 符号链接被过严安全门拒绝，没有训练、结果或关机动作。修复后四卡机与五卡机延期控制器 PID 为 `447869` 与 `410737`。它们正在等待原矩阵正式回传完成，之后会先执行最大地区 CUDA 前向反向探针，再启动六项与三项扩展分片。只有扩展结果在服务器逐文件复验和原子提升成功后才会关机。
62. 四卡机完成原矩阵 26/26 和 134 文件 source-host postflight。首次 finalizer 因没有 BatchMode 回传身份在传输前停止，任务和结果未改写。部署主机专用 Ed25519 身份后，四卡机于 17:19 完成结果服务器逐文件复验与原子提升。关机由 NDR 哨兵延期。提交 `ce7a8c2` 把同一身份设为双机原 finalizer 的默认值。
63. 四卡机 NDR 首次真实 CUDA 探针发现旧训练入口顶层导入了扩展不需要的 `h5py`。该次探针没有建立 trainer 或正式结果。提交 `77d8815` 收窄导入边界并新增无 `h5py` 回归，本地全仓为 456 passed 和 2 个既有 warning，双机真实导入门通过。
64. 四卡机 NDR 重启探针通过，Contra Costa shape 为 `16×496×12`，loss 为 `37.95976257`，CUDA 峰值为 8,128,185,856 bytes。17:24，PID `456092` 启动 4 个正式 shard runner，承担 Alameda 与 Contra Costa 6 项。17:25 四张卡利用率为 93% 到 97%，显存为 8,644 MiB 到 9,416 MiB。原矩阵总进度为 36/39，五卡机为 10/13，正在执行 task037 Orange D2STGNN，NDR 控制器 PID `422803` 继续等待。
65. 四卡机完成 Alameda 与 Contra Costa 的六项 NDR 后关机。用户重新开机并批准由四卡机接管 Orange 三种子。重分配计划 `igstgnn_table4_ndr_extension_4gpu_rebalance_v2` 的 SHA-256 为 `56a71fe2a3b8101ae282dc6f9b9efab07874108a40922276285206e99cbf2612`。旧六项保持原 shard 路径，只读复验后跳过。Orange seeds 42、1、2 分别固定到 GPU 0、1、2。
66. Orange 原 batch 16 单卡 backward 探针因 24 GB 显存不足而 fail closed，没有正式结果。提交 `87526d1` 使用两次 max microbatch 8 前向，以全批分母计算 loss，只做一次梯度裁剪、一次 optimizer step 与一次 scheduler step，有效 batch 仍为 16。梯度等价回归通过。真实 CUDA 探针 loss 为 `42.45340157`，峰值分配显存为 12,632,052,224 bytes。
67. 四卡机正式 NDR 控制器 PID `3677` 于 2026-07-30 00:08:27 新加坡时间启动。00:18 三项均完成 epoch 1，validation MAE 为 13.19330061、13.06060743、12.91147725。完成控制器 PID `5614` 会先证明旧六项 30 个 artifact 哈希未改变，再执行完整 9 项 postflight、独立回传、结果服务器复验、原子提升与关机。
68. 五卡机旧 NDR 等待器 PID `422803` 已定向停止，原矩阵未受影响。incident 清单摘要为 `556c4289e2b4ec41b35b40d1f9a72190e1ad1c3c5a55edf1986843a70aa6e8a1`。新关机等待器 PID `201649` 只在原矩阵回传和结果服务器正式树复验通过后关机。00:18 原矩阵为 37/39，task039 Orange IGSTGNN 已完成 44/100 epoch，之后只余 task034 Orange STTN。控制提交为 `dd2d087` 与 `b5d9751`。
69. NDR 扩展完成 9/9 并正式发布到结果服务器。正式树含 49 个文件和 940,488,092 字节，files digest 为 `ab21c8e300ba930e5611cb9c07ff5b047eb03d281ca669c756242707231c87e7`。task039 Orange IGSTGNN 完整落盘，原矩阵达到 38/39。
70. 五卡机 task034 Orange STTN 单卡训练因额外 11.22 GiB 分配请求 OOM，没有正式结果。五卡机完成既有 12 项关机前清单后按用户要求关闭。提交 `4935b9b` 建立四卡机跨主机单任务恢复，计划 SHA-256 为 `022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5`。真实 batch 64 完整优化步探针通过，正式 task034 于 09:21:28 启动。
71. 用户重新开启五卡机。提交 `918af63` 建立严格排除 task034 的 12 项只读导出。source-host postflight 验证 60 个 artifact 和 1,975,159,969 bytes，files digest 为 `5c11ad4596228f08613b783d8a42ed9334e9976ce8cff4d23d09f6b79f055490`。结果服务器清理非实验缓存后完成逐文件复验和原子提升，五卡机自动关机。
72. task034 于 10:38:24 完成。提交 `a9511d5` 与 `51148f1` 修复两个 fail-closed 收尾问题，没有重训或 artifact 改写。task034 的 5 个文件和 170,575,062 bytes 在结果服务器完成逐文件复验、原子提升和正式树二次复验，transfer marker SHA-256 为 `e393e8238b25677b55e7f1ae00813c3f0a9eb46f7a532c0eb7bba21ee4758d68`。四卡机自动关机，原 39 项任务结果层达到 39/39。
73. 完成原 39 项三源 composite 与 NDR 9 项独立索引。原矩阵状态为 `verified_official_protocol_with_known_overlap_39`，正式 index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。NDR 扩展 index SHA-256 为 `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a`。论文同结构 39 项表与 NDR 三种子汇总表均已生成并保留协议限制。
74. 建立 BiST 与 BiST-NDR 严格配对矩阵 `igstgnn_table4_bist_ndr_pair_18_v1`。矩阵为两种方法乘三地区乘种子 42、1、2，共 18 项。矩阵 SHA-256 为 `90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab`，配置 SHA-256 为 `b763fb35b392e265a410325efa0ef34afa3b0d16c7c2e56c71bc3243844e8764`。两个方法使用相同训练控制、`partial_batch: retain`、全局验证 masked MAE 和测试观测集合。
75. 通过 Mac 全仓 480 项回归、5080 Windows 5 项专项门、Alameda 两种方法 CPU 探针和 Orange 两种方法 CUDA 完整优化步探针。5080 在独立 checkout `D:\t4\bist_ndr_repo` 启动正式队列。第一次 SYSTEM 启动因 site-packages 没有传给子进程而在训练前退出，没有正式结果。修复后计划任务 `XTraffic_BiST_NDR_Pair_v1` 于 2026-07-31 10:06:03 新加坡时间重启。
76. 首项 `pair001_bist_aligned_alameda_seed42` 于 10:14 完成 30 个 epoch，并通过五文件验收，best validation MAE 为 `11.69943708`。H3、H6、H12 与 Average MAE 为 `11.205525`、`11.784402`、`12.675788` 与 `11.740657`。队列完成数为 1/18，失败数为 0，正在执行 `pair002_bist_aligned_alameda_seed1`。该单项结果不能形成 BiST-NDR 性能结论。
77. BiST 与 BiST-NDR 配对队列于 2026-07-31 13:41:13 新加坡时间完成。runner 为 completed，18/18 任务全部验收，失败和 skipped 均为 0，正式任务树含 18 个目录与 90 个文件。这是统一 30 epoch 训练控制下的结构配对实验，与用于填回原 Table 4 的原生配方 100 epoch 补充矩阵分开管理。当前尚未生成配对性能结论。
78. 建立 BiST 与原 STAEformer 版 NDR-Net 的 100 epoch 补充对比矩阵 `igstgnn_table4_bist_ndr_native100_18_v1`。它保留各自原生 batch、optimizer、scheduler、partial batch 和 early stopping 契约，只把两者最大 epoch 设为 100。矩阵为两方法乘三地区乘三种子，共 18 项，并使用原 Table 4 的指标列。
79. 通过 Mac 全仓 489 项回归、5080 Windows 25 项专项门与 Orange 两种方法真实 CUDA 完整优化步探针。计划任务 `XTraffic_Table4_Native100_v1` 于 2026-08-01 00:02:23 新加坡时间启动。启动复验时队列为 0/18，失败数为 0，正在执行 `n100001_bist_alameda_seed42`。
80. 增加 100 epoch 专用入表器。它仅在 18 项均通过五文件复验时产生 48 行最终表，保留原 42 行并追加 6 行 BiST 100ep 与 NDR-Net 100ep 三种子汇总。输出包含 CSV、Markdown、LaTeX 和可追溯 manifest。
81. 固化双机地区分层计划 `igstgnn_table4_native100_5080_5090_v1`。5080 选择 Alameda 与 Contra Costa 的 12 项，5090 选择 Orange 的 6 项。两分片互斥且并集为 canonical 18，同地区的 BiST 与 NDR-Net 保持同机。计划 SHA-256 为 `d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e`。
82. 安全切换 5080 队列。task013 中断时没有正式任务目录，incident 清单摘要为 `94802ea874935de3db70eb3d5b730ec14e978662ebeada4014bc7eb09f34be23`。新计划任务从 9 个已验证结果继续。5080 结果根内既有 Orange BiST 三项保持只读，但被最终选择规则排除。
83. 5090 完成 Orange 九个数据文件的逐文件复验、25 项训练门禁与两种方法真实 CUDA 完整优化步探针。持久 Orange 六项队列于 12:38:57 启动，初始状态为 0/6 完成和 0 失败。Mac 全仓回归为 493 passed 和 2 个既有 warning。
84. 根据用户要求部署 5090 训练完成自动关机守护。守护脚本 SHA-256 为 `d0168a7f1b3b2b1b5386de1cebb52681fc1319babe8df1eddb5344228706bbb1`，远端持久 PID 为 `20225`。它只在 Orange 6/6 逐任务五文件复验、queue completed、runner 零退出且训练进程完全结束后关机，任何异常都会保持开机。
85. 5080 地区分片于 15:52:38 完成并验收 12/12，失败数为 0。主机随后断电，23:37 恢复局域网连接后确认结果与 completed 状态完整，GPU 空闲。完整矩阵当时为 16/18 已验收。
86. 旧 5090 端口在 4/6、task017 epoch 82 后不可达。替代端口对应全新空数据盘容器，故旧机未回传结果全部排除。替代 5090 重新通过九文件数据哈希和两项真实 CUDA 完整优化步探针，并于 2026-08-02 09:48:50 从 0/6 重跑 Orange 六项。
87. 部署替代 5090 的 fail-closed 回传 finalizer。它先执行六项本地五文件复验，再完成结果服务器容量检查、逐文件 SHA-256、精确 allowlist 与原子提升，成功后自动关机，任何异常保持开机。
88. 对空闲 5080 执行 Orange NDR 真实 CUDA 探针和 292 秒隔离 live speed gate。探针通过但没有完成首个 epoch，候选辅助分片被拒绝并禁用，没有正式任务目录。原 5090 Orange 六项计划继续执行。
89. 替代 5090 完成 Orange 6/6，失败数为 0。31 个结果文件和 1,109,955,208 bytes 通过 source-host postflight、结果服务器逐文件 SHA-256、精确 allowlist、原子提升与正式树复验。files digest 为 `1ae14803b44dcd88190dbf81ef47431ce161238749fec0c5e37c67acc37b9a52`。随后 5090 自动关机。
90. 5080 完成 Alameda 和 Contra Costa 12/12 严格回传。61 个结果文件和 686,213,987 bytes 通过复验，files digest 为 `5d2de42964e581ea68f6612e7c05b4672268b22f267f887c191e62bc1470abf9`。旧 Orange BiST 三项被导出 allowlist 排除。提交 `4920a4d` 修正 Windows SSH 子进程不退出并使恢复路径复用已验证文件。5080 保持开机。
91. 完成两次全量只读 composite 复验。正式 index 状态为 `verified_native100_dual_host_18`，SHA-256 为 `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c`，含 2 个来源、18 个 canonical task、92 个结果文件和 1,796,169,195 bytes。第二次复验除时间字段外完全一致。
92. 生成 48 行最终表，保留原 42 行并追加 6 个 100 epoch 三种子汇总行。CSV SHA-256 为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`。三地区 Average MAE 上 NDR-Net 均低于 BiST，但结论局限于 `official_protocol_with_known_overlap` 和各自原生配方的描述性比较。

当前 100 epoch 补充矩阵已完成的事项如下。

1. BiST 与 NDR-Net 的 18 项原生配方 100 epoch 补充矩阵已逐任务验收并发布。
2. H3、H6、H12 和 Average 同结构对比表已生成，并保留三种子均值和样本标准差。
3. 原 BiST 500 epoch 单次行、NDR-Net 30 epoch 三种子行和新 100 epoch 补充行均保留，没有互相覆盖。

后续如需对统一 30 epoch 的 BiST 与 BiST-NDR 结构配对矩阵形成结论，应单独执行三种子汇总与窗口级配对分析。它与本次原生配方 100 epoch 补充表保持分离。

---

## 第十一部分：关键文件与资产

| 文件 | 作用 |
|---|---|
| README.md | 仓库总览、当前有效性警示与阅读顺序 |
| CLAUDE.md | 新会话入口、当前阻塞与协作约定 |
| docs/README.md | 当前文档、历史设计与归档索引 |
| scripts/README.md | 数据、训练、诊断和历史脚本索引 |
| outputs/README.md | 结果目录状态与保留策略 |
| docs/audits/data_leakage_2026-07-18.md | 当前数据泄露审计，含代码证据、落盘量化与修复验收条件 |
| docs/audits/result_validity.csv | 各实验族的机器可读有效性状态 |
| docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md | Table 4 双机九卡部署、incident、双卡恢复、自动回传与关机状态 |
| docs/audits/igstgnn_table4_bist_ndr_pair_launch_2026-07-31.md | BiST 与 BiST-NDR 18 项严格配对协议、探针、启动 incident 与 5080 队列状态 |
| docs/audits/igstgnn_table4_native100_launch_2026-08-01.md | BiST 与 NDR-Net 原生配方 100 epoch 矩阵、探针、身份与 5080 启动状态 |
| docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md | 原生配方 100 epoch 矩阵的双机分层、incident、严格回传、composite 和最终表记录 |
| outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/ | 最终 48 行 CSV、Markdown、LaTeX、manifest 与正式 composite index |
| scripts/reproduction/igstgnn_table4/native100_host_postflight.py | 5080 与 5090 源主机精确导出复验 |
| scripts/reproduction/igstgnn_table4/native100_composite_postflight.py | 双机正式树逐文件、逐任务只读 composite 复验 |
| scripts/reproduction/igstgnn_table4/native100_compare_report.py | 从哈希绑定 composite index 生成 100 epoch 最终表 |
| docs/archive/项目总结报告_2026-06-10.md | 已归档，止于 06-10 的全部结果、13 节审计实验、14 节决策、复现命令，详情已并入本文 |
| docs/archive/去季节化方向_交接_2026-06-26.md | 已归档，去季节化方向的会话交接，核心已并入本文第 4.8 与第十二部分 |
| dist_net/protocol.py | 无泄露 v3 的因果填充、窗口、切分、事件身份与共享协议 |
| scripts/build_full_county_cache.py | v3 XTraffic 缓存构建器，执行预测时刻去重、主事故规范化与完整活跃事故处理 |
| scripts/build_event_rel_feat.py | v3 事故与节点关系缓存，校验协议、稳定事件身份和配对缓存身份 |
| scripts/build_igstgnn_data_from_h5.py | v3 IGSTGNN 转换器，统一预测时刻、锚定事故与输入目标掩码 |
| scripts/audit_protocol_v3_cache.py | v3 正式缓存的 schema、哈希、计数、split、掩码与关系特征验收入口 |
| dist_net/data.py | 默认拒绝旧缓存并校验内容身份的事件锚定与 full-window 数据接口 |
| dist_net/metrics.py 与 graph.py | 全局 masked MAE 聚合与统一图支持矩阵实现 |
| dist_net/run_integrity.py 与 training_runtime.py | 正式训练的代码、缓存、任务、输出与 summary 身份校验 |
| scripts/remote/windows/protocol_v3_runner.py | 5080 与 5090 使用的严格远程 runner，按 manifest 串行执行任务，复算 prediction artifact 指标并验收 summary 与运行身份 |
| scripts/remote/windows/protocol_v3_repair_task041_longpath.py | 哈希锁定的 task041 Windows 长路径 incident 修复入口，仅接受已审计 attempt 并在双重验收后原子提升 |
| scripts/remote/distributed/protocol_v3_distributed.py | 从 canonical manifest 构建固定 5080 与 5090 分片，并验证互斥、完整覆盖和直接比较组同机 |
| scripts/remote/distributed/protocol_v3_shard_postflight.py | 在每台源主机复验分片结果身份、指标、运行时 provenance 与严格导出 allowlist |
| scripts/remote/distributed/protocol_v3_composite_postflight.py | 在结果服务器只读验证两个分片的 canonical 45 完整并集与逐文件 SHA-256 |
| scripts/remote/windows/protocol_v3_launch_5080_shard.cmd 与 hidden.vbs | 在 5080 启动 27 项分片 runner，继承 24 项并新跑 tasks040 至 042 |
| scripts/remote/windows/protocol_v3_repartition_5080_27_waiter.cmd 与 ps1 | 在 5080 等待并严格验证 27/18 ready marker，只在 task039 边界后启动新 runner 与 finalizer |
| scripts/remote/windows/protocol_v3_validate_repartition_marker.py | 拒绝重复 JSON 键、非有限数字、错误哈希、错误 PID 与非法边界任务 |
| scripts/remote/windows/protocol_v3_finalize_5080_shard.cmd 与 ps1 | 复验并回传 27 项分片为 `rtx5080_27`，依赖 `rtx5090_18` 完整上传，成功后保持工作站开机 |
| scripts/remote/linux/protocol_v3_stop_at_boundary.py 与 protocol_v3_start_5090_shard_after_boundary.sh | 在 task039 完成边界原子发布 ready marker，再启动 18 项分片 runner 与 finalizer |
| scripts/remote/linux/protocol_v3_finalize_5090_shard.sh | 复验并回传 18 项分片为 `rtx5090_18`，远端全量哈希通过后关机 |
| tests/data/、tests/training/ 与 tests/remote/ | 数据协议、训练正确性与远程运行完整性回归测试 |
| config/lab_hosts.env | 实验室共享主机、地址与密码的集中明文配置 |
| scripts/train_fourier_dual_net.py | FDN 训练 |
| scripts/train_graphwavenet.py | GWN 训练 |
| scripts/train_stid.py | STID 训练 |
| scripts/train_gwn_icsf.py | ICSF 移植对撞 |
| scripts/train_benchmark.py | 标准基准 FDN 与 GWN |
| scripts/train_gwn_spectral.py | 等参数路由对容量消融 |
| scripts/train_staeformer_xtraffic.py | STAEformer 移植 XTraffic |
| fourier_dual_net/rgdn.py 与 deseason.py | RGDN 模型与去季节化分解，含 AdaptiveAlpha 与单支 adaptive 与 const_alpha 分支 |
| scripts/train_rgdn.py | RGDN 等参数变体训练，v0a 到 v1g，含 v0c 自适应与 v0d 常数消融 |
| scripts/train_staeformer_deseason.py | STAEformer 上的 RADS、RADS-O 与 NDR-Net，审计前实验已完成 |
| fourier_dual_net/regimeflow.py 与 scripts/train_rfn.py | RFN 独立骨干探索，负结果，见 4.9 |
| fourier_dual_net/radgmixer.py 与 scripts/train_rad.py | RAD-GMixer 独立骨干探索，负结果，见 4.9 |
| outputs/diagnostics/adaptive_deseason_results.txt | 审计前 v0c、v0b 与 v0d 历史结果 |
| outputs/diagnostics/rfn_results.txt 与 rad_results.txt | 审计前独立骨干负结果与 qov 诊断 |
| outputs/diagnostics/rads_staeformer_results.txt | 审计前 RADS、RADS-O 与 NDR-Net 汇总 |
| docs/RADS_NDR_汇报讲稿.md | 方法线完整讲稿，机制逐层解释加数值例子加诚实定位加精确引用，可直接对外汇报 |
| scripts/diagnose_rgdn_injection.py | 注入优化陷阱诊断 |
| docs/superpowers/ 下 2026-06-17-rgdn 的 spec 与 plan | RGDN 设计与实现计划 |
| scripts/significance_tests.py | 配对显著性检验 |
| scripts/incident_type_breakdown.py | 事故类型拆解 |
| outputs/diagnostics/ | seed_noise_band、icsf_collision_results、gwn_routing_vs_capacity、benchmark_transfer_results、rgdn_round1_results 等 |
| baselines/IGSTGNN/ | IGSTGNN 官方代码，含已修复的 dataloader |
| baselines/STAEformer/ | STAEformer 官方模型 |
| 论文/2602.02528v1.pdf | IGSTGNN KDD'26 原论文 |

算力由本地 GPU 工作站与临时云 GPU 提供。实验室共享连接信息集中明文维护在 `config/lab_hosts.env`，运行前可直接加载。当前本地 Mac 不承担正式训练，只用于文档、静态检查和具备依赖时的纯数据测试。

---

## 最新进展，2026-08-04 v3 baseline 协议边界修正

为了得到与 Table 4 列结构一致且不含 target 重叠的对比结果，项目新建了独立的 v3 全窗口 baseline 矩阵。它与 `igstgnn_table4_official_v1` 官方最大忠实复现线互不覆盖。后者保留已知样本重叠，只用于和论文表格对照。新矩阵使用 `protocol_v3_noleak_dedup`、target-disjoint 70/15/15 切分和 cache schema 3。

第一阶段包含 HL、LSTM、DCRNN、AGCRN、STGCN、GraphWaveNet、ASTGCN、STTN、DSTAGNN、DGCRN 和 D2STGNN，共 11 个方法乘 3 个地区，合计 33 项。训练控制为 seed 42、30 epoch 上限和 patience 6。BiST 与 IGSTGNN 暂未包含，原因是它们还没有与 v3 缓存契约对齐的 adapter。两者将在完成适配与审计后作为第二阶段加入。NDR-Net 同样必须使用 v3 数据单独重跑，现有官方重叠协议下的 NDR-Net 数字不能填入 v3 表。

第一阶段 canonical manifest SHA-256 为 `8734ea3632622a852b63d8faa6a36ebfebec196c7703654d93e5739ba3e777b0`，训练代码 SHA 为 `2db8bff83ea791bd83dc62d9a1feb0386a2eb9e9`。三地区正式 cache 再次通过只读复验，bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。Mac 全仓回归为 507 passed 和 2 个既有 warning，5080 最终 checkout 专项门为 10 passed。

5080 正式计划任务 `XTraffic_ProtocolV3_Baseline_Phase1_v3` 于 2026-08-04 18:28:58 新加坡时间启动，正式结果根为 `D:\t4\v3_baselines_phase1_5080_v3\results`。复审确认 IGSTGNN 的官方前向必须使用事故特征、事故距离和传感器元数据，因此不能在所有滑窗都纳入的 full-window 数据上仍声称与 IGSTGNN 源管线一致。full-window 队列在 1/33 正式验收后停止并禁用，只作常规时序预测补充实验。Alameda LSTM 的未验收 attempt 保留为中断记录。Alameda HL 是唯一通过 runner 验收的任务。它的 H3、H6、H12 和 Average MAE 分别为 `13.664771357540491`、`17.141543678361064`、`24.58813473938541` 和 `17.835280131426384`，Average RMSE 为 `31.697551962208454`，Average MAPE 为 `28.380292960200414`，有效观测点数为 `79942548`。summary SHA-256 为 `2c7d1fb7fe0580dfade9e8fb1c4b8582364a5044dd2dbad293a1f31ee325cf8e`。

正式启动前出现两次失败关闭。v1 由于 Windows checkout 把 manifest 换行符转为 CRLF，哈希门拒绝启动。v2 的 HL 训练返回码为 0，但 baseline-zoo 的 staging 产物目录与控制层期待不一致，runner 因此未提升结果。两次都是 0/33 正式结果，旧任务已禁用，incident 保留供审计且不进入最终 composite。对应修复提交为 `a5830a8` 和 `2db8bff`。

用户先前确认的 event 口径现为 v3 主对比。新清单 `config/protocol_v3/table4_event_baselines_phase1_v1.json` 先覆盖相同的 11 个已支持方法乘 3 地区，合计 33 项。BiST 与 IGSTGNN 将在同一 v3 event 样本集和切分上接入，不再尝试把 IGSTGNN 塞入不具备完整事故上下文的 full-window 队列。全仓回归为 511 passed 和 2 个既有 warning。

event v1 首次启动在 HL 产物阶段因测试集未显式加载 `affected_mask` 而失败，没有产生正式任务目录。修复提交为 `6e4d115`，新队列固定在 `c16dc7a8208f834984e76a40a7b6f47f523c210b`。计划任务 `XTraffic_ProtocolV3_Event_Baseline_Phase1_v2` 使用无执行时限的 `PT0S` 设置。正式结果根为 `D:\t4\v3_event_phase1_5080_v2\results`。当前为 1/33 完成、0 失败，正在训练 Alameda LSTM。event Alameda HL 的 H3、H6、H12 和 Average MAE 为 `14.5656612415774`、`17.672297556357695`、`24.177859877122867` 和 `18.262501186926563`，Average RMSE 和 MAPE 为 `31.760263449673126` 和 `25.456400502008623`，summary SHA-256 为 `9644760e084e70bfa8fe372ad2e11aecc826771f4ecb244d0c0685a611e999a3`。

## 最新进展，2026-08-04 v3 event baseline 八卡加速

新 SeetaCloud 实例经硬件复验为同一容器内的 8 张 RTX 4090 D，每张约 24 GB。Orange 的 11 个 baseline 全部保留在该主机内，只按方法固定分片到 GPU 0 至 7。Orange 的直接方法比较不跨主机。RTX 5080 继续执行 Alameda 和 Contra Costa。最终布局必须标记为主机分层矩阵，不得写成同质矩阵。

正式缓存与图文件已通过 bundle 逐文件复验。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。八卡控制器 PID 为 `2429`，matrix ID 为 `protocol_v3_event_orange_4090d8_v1`，首轮提交为 `2303557dc7fc3f5e87760bae0b31d407b544e5f5`。

DGCRN、D2STGNN 和 STTN 首轮分别在额外申请 120 MiB、2.80 GiB 和 5.61 GiB 时 OOM。三项都没有正式任务目录，traceback 作为 incident 保留。提交 `845804de1586eb6620496298d264c789c6a140c9` 增加等效 batch 32 的显存恢复。梯度按全 batch 有效观测点归一化，microbatch 只用于限制激活显存，每个有效 batch 仍只做一次优化器更新。DGCRN 使用 microbatch 8，STTN 使用 16。D2STGNN 在 16 时仍 OOM，提交 `5bced13246ee3eb1e903726f0e9c60c839f46209` 将其降为 8。三者均已通过首个完整有效 batch。D2STGNN 在 microbatch 8 时的首 batch loss 为 `141.838`，显存约 19.5 GiB。Mac 全仓回归为 517 passed 和 2 个既有 warning。

5080 event 队列已验收 2/33，当前执行 Alameda DCRNN。新验收的 Alameda LSTM Average MAE 为 `12.950932743865955`，H3、H6 和 H12 MAE 为 `11.472908787131686`、`12.867387335945871` 和 `15.258082126124597`，summary SHA-256 为 `cfe37cfb481623214afa4516125100f8e6d05d518fe71cf84a895203b50f9452`。当前 Orange 尚无本轮新的正式性能结论。详细运维与 incident 记录见 `docs/audits/protocol_v3_event_4090d8_launch_2026-08-04.md`。

## 最新进展，2026-08-04 v3 event baseline 初始化修复与续跑

5080 event 队列达到 3/33 正式验收，当前执行 Alameda AGCRN。Alameda DCRNN 的 H3、H6、H12 和 Average MAE 为 `12.167831568642377`、`14.011284383846979`、`17.427471336632582` 和 `14.215668450826508`。Average RMSE 为 `24.984026878436037`，Average MAPE 为 `20.944177481318572`，best validation masked MAE 为 `14.91492617373589`，有效观测点数为 `34427151`。summary SHA-256 为 `806a6ab4f91ffca55609b908c82d6e5b028d5b26421a06e7f7336dfac09b2801`。

4090 D 的 DSTAGNN 首轮每个训练 batch 都产生非有限 loss，评测门安全拒绝生成正式结果。代码对照确认 v3 baseline factory 没有复现上游 LargeST engine 对 ASTGCN 与 DSTAGNN 的显式参数初始化。ASTGCN 首轮虽然数值有限，但实现契约同样不完整，因此在没有正式结果时停止。提交 `d8760d1a48c30d7d87130aef1712c85b8aa76359` 恢复上游初始化语义，并让 AGCRN 复用同一初始化入口。Mac 全仓回归为 519 passed 和 2 个既有 warning，远端专项测试为 12 passed。

修正后的独立恢复矩阵 `protocol_v3_event_orange_upstream_init_recovery_v1` 已启动。DSTAGNN 在 GPU 3 的首个完整 batch loss 为 `198.078`，ASTGCN 在 GPU 5 的首个完整 batch loss 为 `200.413`，两者均为有限值。GPU 5 完成 ASTGCN 后接续 STGCN。DGCRN、D2STGNN 与 STTN 的等效 batch 恢复仍保持运行。Orange 当前没有新任务通过五文件正式验收，因此仍为 0/11，不得提前填表或形成性能排名。

## 最新进展，2026-08-04 23:43 v3 event baseline GraphWaveNet 修复

全矩阵当前有 7/33 个任务通过 runner 正式验收。5080 已完成 Alameda 的 HL、LSTM、DCRNN、AGCRN 与 STGCN，共 5 项。新增两项的结果如下：

| 方法 | H3 MAE | H6 MAE | H12 MAE | Average MAE |
|---|---:|---:|---:|---:|
| AGCRN | `11.656392` | `12.615384` | `13.980622` | `12.566474` |
| STGCN | `13.367683` | `14.344828` | `16.129805` | `14.419148` |

Alameda GraphWaveNet 已完成 30 epoch，best validation masked MAE 为 `12.934727422724507`。测试推理随即因 event 测试集没有加载 `affected_mask` 而失败关闭。该问题位于 GraphWaveNet 独立入口，不影响其训练 checkpoint，也没有产生正式 task006。提交 `edf4c1e` 让 event 测试集加载只用于分析的标签，训练集和验证集继续不加载该标签，因此没有训练泄露。提交 `e1bea79` 增加同 GPU 恢复分片。本地全仓回归为 521 passed 和 2 个既有 warning。

5080 使用 clean checkout `e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae` 启动 tasks006-022 的 17 项 continuation。新结果根为 `D:\t4\v3_event_gwnfix_v1\results`，当前从 Alameda GraphWaveNet 干净重跑。旧根内 5 个正式结果保持只读。首次启动因默认结果根误写为绝对路径而被配置门拒绝，没有 trainer 或结果。incident 保留后，manifest 已改回相对路径并正常启动。Windows 专项门为 12 passed。

Orange 已验收 HL 与 LSTM，共 2/11。结果如下：

| 方法 | H3 MAE | H6 MAE | H12 MAE | Average MAE |
|---|---:|---:|---:|---:|
| HL | `15.407021` | `18.471072` | `24.892803` | `19.025464` |
| LSTM | `12.191571` | `13.618700` | `16.221246` | `13.724441` |

旧 Orange GraphWaveNet 在 epoch 25 主动停止，没有正式结果。修正后的矩阵 `protocol_v3_event_orange_4090d8_gwn_recovery_v1` 已在 GPU 6 从 epoch 0 干净重跑 GraphWaveNet，完成后接续 AGCRN。4090 D checkout 同样为 `e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae`，专项门为 16 passed。部署 bundle SHA-256 为 `e810c8c56b9715e2ff4e8b286c48d4ae0c0f92ba049698e66850bcbe36761359`。其余 DCRNN、DGCRN、D2STGNN、STTN、DSTAGNN 与 ASTGCN 继续训练或恢复。所有结果仍需经过单任务验收、双机 postflight、结果服务器复验和只读 composite 后才能生成最终表。

## 最新进展，2026-08-05 12:10 v3 event 主机离线与 5080 续跑

八卡 4090 D 的 `11218` 端口连续三次显示 TCP 建连成功后由远端立即关闭 SSH。结果服务器没有本轮 v3 event 结果目录。该证据说明实例当前不可用，但无法区分余额不足、手动停止或平台回收。最后在线时 Orange HL 与 LSTM 已通过 runner 验收，尚未完成远端回传。因此当前 runner 层进度为 11/33，其中 5080 的 9 项可直接访问，4090 D 的 2 项处于未回传状态。若同一实例重新开机并挂回原数据盘，应先锁定和回传现有 task artifact，再检查其他中间 attempt。

5080 新增验收的 Alameda 四项如下：

| 方法 | H3 MAE | H6 MAE | H12 MAE | Average MAE |
|---|---:|---:|---:|---:|
| GraphWaveNet | 待从 prediction artifact 汇总 | 待汇总 | 待汇总 | `12.589115` |
| ASTGCN | `12.250701` | `13.715483` | `16.391159` | `13.838033` |
| STTN | `11.632755` | `12.569361` | `14.021482` | `12.552551` |
| DSTAGNN | `11.671517` | `12.846888` | `14.307856` | `12.627796` |

Alameda DGCRN 在 epoch 1 的最后一个单样本 partial batch 上失败。上游 `torch.squeeze` 未指定维度，batch size 为 1 时把 batch 维一起删除，随后 `transpose` 触发维度越界。该 attempt 没有正式 task010。提交 `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2` 去除无条件 squeeze，对常规 batch 的张量和值不做改变，并增加 batch 1 前向回归测试。Mac 全仓为 522 passed 和 2 个既有 warning，5080 专项门为 13 passed。

5080 已使用 clean checkout `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2` 启动 tasks010-022 的独立恢复分片。结果根为 `D:\t4\v3_event_dgfix_v1\results`，计划任务为 `XTraffic_ProtocolV3_Event_Baseline_DGCRNFix_v1`。Alameda DGCRN 首个 batch loss 为 `137.815`，显存占用约 10.6 GiB，训练正常。部署 bundle SHA-256 为 `a2c6e990eb9883884e064ed2d5b5c1c681393e96ca7af192749d76dccbb3d902`。

---

## 第十二部分：去季节化方向的数据现实与设计决策

本节并入自原去季节化交接文档，记录 v0c 这条线的方向决策与数据现实，供后续工作参考。

设计路线的选择。去季节化在 v0b 之上有两条可走，一是固定基线只换粒度，二是引入自适应机制按节点状态动态决定锚点。考虑到项目里加结构或门控已连败六次，自适应这条路有退化成捷径的历史风险，但 alpha 的输入是有物理意义的异常量而非自由学习的标量，且 relu 把正常节点钉死在 alpha 等于 1，结构上几乎无法退化，故选自适应这条路并加足护栏，等参数加多种子加逐种子零重叠，允许干净失败。结果是站住了，见第 4.8 节。

数据现实，三类协变量的可用性。第一，节假日已在数据内。time_enc 的第五列是 2023 年美国联邦假日标志，构建见 scripts/build_full_county_cache.py 的 US_FEDERAL_HOLIDAYS_2023。但气候态基线 baseline_median 形状为 2 乘 288 乘 N 乘 3，只分工作日与周末两类日型，没有单独建假日类，假日当前错用工作日或周末基线。改进点是把基线粒度扩成三类，或把假日当协变量喂给学习式基线，预期对 all 影响很小，因为假日只占全年约 3%，是正确性修补而非主增益。第二，天气数据集里没有，要用需外接 Meteostat 或 NOAA ISD，按传感器经纬度找最近站取 2023 年逐小时再广播到 5 分钟，属独立数据工程，有站点稀疏与对齐与仅 2023 一年的风险，先不外接，作后续可插拔项。第三，活动不可行，数据里除事故外无其他活动事件，投入产出比太低，放下。

当时已定的下一步顺序是先用 in-data 的 holiday 与自适应把骨架搭通，再做骨干无关性验证。两项后来均已完成，审计前结果见第 4.8 至 4.12 节。天气仍作为后续可插拔项，holiday 感知基线仍是正确性小修补。无泄露 v3 缓存重建、本地和远端严格审计、clean checkout、runner 45/45、两个正式分片回传与只读 composite 验证现已完成。5090 为 18/18，5080 为 27/27，矩阵有效性状态为 `verified_host_stratified_45`。当前优先级是执行同机配对、多种子与显著性分析。所有判读继续坚持等参数、多种子和显著性，不在单种子结果上收官。方法实现层级始终为 `RADS ⊂ RADS-O ⊂ NDR-Net`。

## 最新进展，2026-08-05 12:37 八卡机无卡恢复

八卡 4090 D 实例以无卡模式恢复 SSH。主机名仍为 `autodl-container-5ee0448de2-ca306372`，原 `/root/autodl-tmp` 数据盘、正式结果、staging 与日志均完整。当前没有 CUDA 设备和训练进程，所以没有尝试在无卡模式继续训练。

停机前实际完成的 Orange 正式结果为 4/11，比此前最后在线快照多 GraphWaveNet 与 STTN 两项。当前结果如下：

| 方法 | H3 MAE | H6 MAE | H12 MAE | Average MAE |
|---|---:|---:|---:|---:|
| HL | `15.407021` | `18.471072` | `24.892803` | `19.025464` |
| LSTM | `12.191571` | `13.618700` | `16.221246` | `13.724441` |
| GraphWaveNet | 待从 prediction artifact 汇总 | 待汇总 | 待汇总 | `12.950535` |
| STTN | `11.697110` | `12.646399` | `13.804152` | `12.585341` |

四项结果及关键控制证据已回传到 Mac 的 `outputs/protocol_v3_noleak_dedup/remote_recovery/partial_export_4_v1`。25 个文件全部通过 SHA-256，源文件总字节数为 `1652698342`，清单 SHA-256 为 `6297b0403746faea2b9c8da0c174952db670cfbd2bdb8f20bea28289118c118a`。这是只读灾备，不是结果服务器正式发布。

结果服务器所在 40 GB 分区已满，可用空间为 0。本轮没有删除历史结果、Docker 数据或系统文件。正式回传将在扩容或完成安全空间审核后继续。八卡机上还保留 D2STGNN epoch 4、AGCRN epoch 4、DSTAGNN epoch 11、ASTGCN epoch 12、DGCRN epoch 2 与 DCRNN epoch 19 的未验收 staging，恢复有卡模式后需按 checkpoint 恢复契约逐项审计。

5080 仍在训练 Alameda DGCRN。12:37 已完成 epoch 2，epoch 3 到 batch 600/1001。全矩阵 runner 层进度修正为 13/33，当前 13 项均已有可访问的内容哈希保护，其中 5080 为 9 项，Mac partial export 为 Orange 4 项。最终性能表仍需完成其余训练、结果服务器原子发布、各源 postflight 和只读 composite。

13:09，结果服务器安全清理完成。服务器端约 1.7 GB 的 `/root/xtraffic_stage` 已确认是与 Mac 旧污染缓存完全同哈希的冗余副本，且没有进程占用，随后删除。systemd 历史日志从 800.3 MB 清至 75.7 MB，APT 可再生成缓存也已清除。正式结果树仍为 21 GB，Docker 活跃服务和所有实验结果未改动。ext4 根分区保留比例从约 4.5% 调至 1%，仍保留约 407.6 MiB 紧急空间。最终分区从 100% 和 0 可用恢复到 93% 和 `3267715072` bytes 可用，约 3.1 GB。

这 3.1 GB 可以暂存当前 Orange 四项，但不足以稳妥容纳完整 Orange 11 项、上传临时树和原子发布余量。因此当前不把 partial export 重新塞满服务器，Mac 的 25 文件验证副本继续承担灾备。正式发布前建议把结果盘扩到至少 60 GB，更稳妥的目标是 80 GB。

18:20，Orange 四项 partial export 已进一步同步到群晖 NAS。目标为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/orange_4090d8_partial_4_v1`。数据先进入隐藏临时目录，NAS 本机验证清单 SHA-256 与 Mac 一致，并确认清单内 25/25 文件全部为 OK，随后在同一 Btrfs volume 内原子改名。最终包含 26 个 regular files，精确文件字节和为 `1652693542`，NAS 仍有约 427 GB 可用。该副本是 `verified_partial_disaster_backup_4`，不能代替正式结果服务器发布和最终 composite。

5080 同时完成 Alameda DGCRN。其 H3、H6、H12 与 Average MAE 为 `11.284030`、`12.563134`、`14.453520` 与 `12.577300`。全矩阵正式进度达到 14/33，其中 5080 为 10 项，Orange 为 4 项。当前正在训练 Alameda D2STGNN，epoch 1 已到 batch 700/1001。

## 最新进展，2026-08-05 18:38 5080 训练后自动回传 NAS

5080 已安装并启动持久计划任务 `XTraffic_ProtocolV3_Event_NAS_Finalize_v1`。当前任务状态为 running，日志表明它正在等待 tasks010-022 恢复队列。该等待器与 trainer 独立，不会中断或改写当前训练。只有在恢复队列达到 13/13 completed、runner 退出码为 0、无残留 trainer 的情况下，它才进入主机结果验收。

5080 的 22 项声明结果分布在三个受控结果根。旧 phase1 根包含 tasks001-005，GraphWaveNet 恢复根包含 tasks006-009，DGCRN 恢复根包含 tasks010-022。它们分别绑定代码 `c16dc7a8208f834984e76a40a7b6f47f523c210b`、`e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae` 和 `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2`。前两个 queue 因训练缺陷按设计遇错停止，所以验收不要求它们整队 completed，而是对每个已接受 task 单独重算 `config_hash`，从 prediction artifact 重算 masked metric，并核对 summary 与 completed state。最后的 13 项恢复根负责整队完成门。

## 最新进展，2026-08-08 18:24 v3 event 单主机全 33 项与自动发布

旧的 5080 和 4090 D 部分结果保持只读，不再进入本轮正式对比。正式矩阵 `protocol_v3_event_full33_5090x4_v1` 在单台 4 卡 RTX 5090 上从头执行 11 方法乘 3 地区，共 33 项。三地区使用同一训练代码、运行时和卡型，因此本轮结果不需要主机分层注脚。

训练代码 SHA 为 `d23a991a570fd1c59e3d2ac8315a5e4063e8097f`，plan digest 为 `23ed8507a1a301daef27bb613bc88d2070ec7052858de7d219e9c845c4d2cb82`，cache bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。Orange D2STGNN 与 STTN 使用 microbatch 16，其余任务使用 batch 32。截至本快照为 0/33 completed、0 failed。GPU0 至 GPU3 分别在执行 Orange DGCRN、DCRNN、DSTAGNN 与 D2STGNN。首轮正式 epoch 慢于探针标定，原 18.7 小时只能作为排产估计，不再当成确定完成时间。

结果发布已切换到 Hostinger `31.97.147.71`。该机约有 49.6 GB 可用空间，已启用 4 GB swap，xray 在配置前后均为 active。5090 finalizer PID `17206` 正在等待矩阵完成。它只在 queue 33/33、runner 零退出、无残留 trainer、source-host postflight、容量门、逐文件 SHA-256、精确 allowlist、隐藏 incoming 原子提升、远端 composite 和表格生成全部通过后关闭 5090。任一失败都保留主机开机。

远端 composite 使用独立 systemd 单元，内存上限为 3 GB，swap 上限为 4 GB。它会在结果服务器上再次核对全部上传文件，重算每项结果的验收契约和预测指标，然后生成 33 行 CSV、Markdown 和 LaTeX。控制提交为 `314d45714b61465bce57e8a33c824ce6c47891fd`，本地全仓回归为 571 passed 和 2 个既有 warning。

这张 33 行表只包含当前已接入 v3 event 协议的 11 个 baseline。用户要求的最终全方法论文表还需要 BiST、IGSTGNN 与 NDR-Net 使用同一无泄漏 event 样本、target-disjoint 切分、验证累计和指标口径重跑。官方重叠协议的旧数字不能填入 v3 表。

## 最新进展，2026-08-08 18:55 无泄漏 42 行对比表已完成训练前准备

最终论文对比表固定为 14 种方法乘 3 地区，共 42 行。正在 4 卡 RTX 5090 上运行的 33 项包含 HL、LSTM、DCRNN、AGCRN、STGCN、GraphWaveNet、ASTGCN、STTN、DSTAGNN、DGCRN 和 D²STGNN。新增 9 项包含 BiST、IGSTGNN 和 NDR-Net 的三地区 seed 42 运行。新增矩阵使用 `config/protocol_v3/table4_event_extensions_v1.json`，与 33 项保持独立结果根、独立 postflight 和独立 composite。

新数据适配器直接懒加载 v3 schema 3 数值缓存，不再生成 IGSTGNN 旧 object-pickle split。训练和验证 batch 只含输入、目标、显式 observation mask、时间特征与模型必需的事故特征。`affected_mask` 不进入训练或验证，只在测试预测完成后从 analysis labels 读取，用于既有 result validator 要求的事后分组 artifact。

NDR-Net 的 flow normalization、weekly climatology、残差标准差与逐节点残差标准差只使用 `0:train_end` 中的唯一 raw timestamps 和显式 `flow_mask` 拟合。这避免了按重叠窗口重复计权，也不再把真实零值默认当成缺失。BiST 与 IGSTGNN 使用同一 train-only flow normalization。IGSTGNN 的公开静态 support 保留为模型图定义，但仅在三地区 `sensors.csv order` 与 v3 `region_idx` 完全相等，且 `station_id` 与 `node_order[region_idx]` 完全相等后加载。

训练契约为 `protocol_v3_noleak_dedup`、cache schema 3、seed 42、30 epoch 上限、target-disjoint event split、显式观测 mask 和全部有效观测点累计的 validation masked MAE。三个扩展方法保留各自公开或项目原生 batch、optimizer、scheduler 与结构。因此这是同数据协议和同评测口径的方法对比，不应误写成所有超参完全相同。

BiST 与 IGSTGNN 的前向调用不传入 future target。IGSTGNN 的 curriculum horizon 按上游 engine 相同的步骤顺序更新，当前有效 horizon 的 loss 分母只包含该范围中的显式有效观测点。Orange NDR-Net 使用 microbatch 8，这一有效值显式写入正式命令、探针与 config hash。

控制层已包含正式 support 哈希绑定、九项真实 CUDA 探针、按实测耗时的 LPT 装箱、plan digest 复核、九项 source-host postflight、结果服务器严格回传、发布后只读 composite 与 42 行合表器。合表器仅在 33 项与 9 项的 verified index 协议、cache bundle ID 与 digest 一致时工作。输出为 CSV、Markdown、LaTeX 和 manifest。

## 最新进展，2026-08-08 22:18 无泄漏扩展 9 项已在四卡 5090 启动

新增主机 `autodl-container-ae8b44827a-75976e27` 已完成正式缓存、图结构、节点顺序、模型 support 与锁定上游源码部署。缓存严格审计返回预期 bundle ID `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4` 和 digest `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。远端 23 项专项测试与三地区 support identity 门全部通过。

九项真实 CUDA 探针全部通过。Orange IGSTGNN 原生 batch 24 在 32GB 上 OOM，改为 microbatch 8 后峰值分配降至约 16.58GB。该实现用三次微前向累计一次优化更新，有效 batch、全批 loss 分母与 curriculum 逻辑不变。Orange NDR-Net 同样使用 microbatch 8。两项任务级覆盖均进入正式 plan、trainer 命令、config hash 与 postflight 契约。训练代码为 `b9e7de62f81bd4dd9a047937cfabc30c7b75d5d3`，本地全仓回归为 590 passed 和 2 个既有 warning。

最终计划 digest 为 `b3f029053136a559657dca768836da6673133b2a47d0c57c1304ffc5dd54f75e`。四卡 LPT 分片的探针估计为 23.06、22.82、20.29 与 22.23 小时。2026-08-08 22:16 新加坡时间启动后，队列为 running、0/9 completed、0 failed。controller PID 为 `13235`，finalizer PID 为 `14224`。

Hostinger 结果服务器 checkout 已切换到同一正式提交，旧 `16e90ba` 保留为回滚。GPU 主机到 Hostinger 的专用受限 SSH 身份和约 45.1GB 可用容量已验证。9 项训练完成后会先做 source-host postflight，再逐文件 SHA-256 上传到隐藏 incoming，目标端精确复验后原子提升并执行只读 composite。全部成功才自动关机。42 行表仍必须等待独立 33 项 baseline matrix 完成并通过 composite 后生成。

本地全仓回归为 589 passed 和 2 个既有 warning，所有远程 shell fail-closed 门禁通过。正式 9 项尚未运行，因此还没有 BiST、IGSTGNN 或 NDR-Net 的 v3 论文数字。下一步是租用多卡 RTX 5090，运行部署、九项探针和 plan-only，在人工复核 plan digest 后启动训练。

同一时点只读复验现有 33 项矩阵仍为 0/33 completed、0 failed。Orange DGCRN、DCRNN、DSTAGNN 与 D²STGNN 分别完成 epoch 3、4、6 与 3。finalizer PID `17206` 正常等待，现有训练 checkout 没有被新实现改写。

实现已以提交 `16e90baab74bb767a4f8d7a46f45eef6d87af44d` 推送到 `main`。Hostinger 结果服务器已完成旁路 checkout 验证与原子切换，旧 `314d457` checkout 保留供回滚。新 extension post-publish 入口已安装，SHA-256 为 `80f05af5b0bb87a221e703479fc034f3b717acf07fff5748b583a1a518388edc`。切换前后 xray 均为 active，4 GB swap 保持正常，根分区约有 44.1 GB 可用。没有修改代理配置，临时传输 bundle 已删除。

通过 postflight 后，finalizer 根据精确 allowlist 上传到 NAS 的隐藏 incoming 目录。已存且同哈希的文件可直接复用，新文件先写 `.part`，每个文件都在 NAS 端核对 SHA-256。全树还需通过 `transfer.sha256`、精确文件名集和无符号链接检查，最后才同 volume 原子改名为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts/event_rtx5080_alameda_contra`。任何训练、验收、容量、传输或 NAS 复验失败都会 fail closed，保留 5080 开机。

专用 NAS SSH 密钥指纹为 `SHA256:Nx5BbjtFe9eRbSJh4hV/YN+reR2pw4hZuFLa9PLKMGc`。私钥只存放在 5080 的 `D:\t4\v3_event_nas_control\id_ed25519`，不进入仓库。直连 NAS 探针已成功。计划 SHA-256 为 `738bc46122f9af6466c1b605013b57c2cc05874aadcda6682cdf08fb4ed964c8`，postflight 与 finalizer SHA-256 为 `b8605c1725fe9de5ffc74d19815f460b19be812dd495f9814fe41a92eb5b1bf0` 和 `7295346daf7eb972ead64db27706f4234157ae783e68c69421097e6787349c70`。postflight 是跨平台 Python 实现，未来 5090 可复用相同验收合同和 NAS 原子发布流程，但必须使用独立 host ID、分片计划、目标目录和 SSH 密钥。这些 NAS 产物定位为可验证灾备，仍需结果服务器正式发布和最终 composite 才能恢复完整矩阵有效性。本地全仓回归为 526 passed 和 2 个既有 warning。

## 最新进展，2026-08-06 15:28 5080 D2STGNN 持续训练

全矩阵 runner 层仍为 14/33，没有新的正式验收任务。5080 已验收 tasks001-010，Orange 已验收 HL、LSTM、GraphWaveNet 和 STTN。5080 当前的 tasks010-022 恢复队列为 1/13 completed，正在 `v3e011_event_d2stgnn_alameda_seed42`。15:28 时已进入 epoch 6、batch 500/1001。GPU 利用率持续 100%，显存约 15.9 GB，温度约 45 摄氏度。

task011 从 2026-08-05 15:09 开始。已完成 epoch 1 至 epoch 4 的单 epoch 用时分别约为 15,923、15,943、15,942 和 15,943 秒，约为 4 小时 25 分。epoch 1 至 epoch 4 的 validation loss 为 `15.3966`、`13.8653`、`13.7192` 和 `15.6938`，epoch 3 是该时点的最佳验证。这是当前队列的明显瓶颈。它仍受 30 epoch 上限和 patience 6 控制，不因运行缓慢改动训练契约。

NAS 自动回传计划任务仍为 Running，持续记录 `completed=1/13`。NAS 上的 5080 正式目标和 incoming 均不存在，这说明 finalizer 没有越过训练完成门。Orange 主机仍在 TCP 建连后立即关闭 SSH，无法得到新的落盘证据。Orange 的 4 项 NAS 灾备仍为 26 个 regular files，`/volume1` 约有 427 GB 可用。15:30 对结果服务器的实时复验显示可用空间为 `2939228160` bytes，使用率为 93%，正式结果根为 `21960917234` bytes。它仍不足以支撑完整 Orange 分片的长期存储和原子发布余量。

`CLAUDE.md` 已在本次同步中从 63 KB 的历史流水改为当前导航和操作快照。已删除的内容只是与本报告和专项审计重复的启动、故障、恢复和完成流水。关键哈希、协议边界、当前队列、发布门、实验纪律和历史入口均已保留。

## 最新进展，2026-08-09 12:10 扩展数据预载加速已完成代码门禁

对 9 项无泄漏扩展的实测复审确认，旧计划把 CUDA 优化步当作完整 batch 成本，遗漏了逐样本 HDF5 数据读取。runner 状态文件显示 IGSTGNN Contra Costa、BiST Alameda 和 IGSTGNN Alameda 的单项耗时分别为 5.96、10.51 和 12.26 小时。旧四卡分片按实测速度外推的总时长约为 31.2、43.7、23.4 和 44.7 小时，原 LPT 排产已经严重失衡。

Orange IGSTGNN 的 20 个真实训练 batch 对照进一步锁定根因。lazy 模式初始化为 3.862 秒，20 batch 构建为 28.5172 秒。eager 模式初始化为 11.244 秒，20 batch 构建仅为 0.270774 秒。两条路径的 batch stream SHA-256 都是 `af7f99015966b930571b3ddd9f111e60f694a537112cdcfe56dd4f76095fddb1`。数据构建速度提升约 105 倍，底层 traffic 数组总内存规模远低于主机可用内存，不需要物化逐样本张量。

提交 `c78b87a` 已推送到 `main`。canonical manifest 现在显式固定 `traffic_loading: eager`。trainer 命令、invocation config、summary 与 config hash 记录同一加载模式。新增审计覆盖 Alameda、Contra Costa 与 Orange 三地区，训练集按固定 seed shuffle，验证集和测试集顺序读取，逐 batch 要求字节级一致。CUDA 探针分别测量 batch 构建和完整优化步，正式 task calibration 使用两者之和，随后重新运行 LPT。loading audit 文件及其 SHA-256 同时写入正式 plan、NAS 回传计划和 source-host postflight 契约。

本地实现先建立失败测试，再完成代码。最终全仓为 606 passed 和 2 个既有 warning。Python 编译、Linux shell 语法与 diff 检查全部通过。用户自己的 `table4_副本.csv` 保持未跟踪且未修改。

旧扩展最后有落盘证据的状态为 3/9 completed、0 failed，baseline 最后有证据的状态为 6/33、0 failed。2026-08-09 12:10 时，`50989` 和 `36571` 两个 AutoDL 入口均能建立 TCP，但 SSH 在鉴权前由远端关闭。通过 Hostinger 解析真实入口后再测，跨境 TCP 超时。本机直连真实 IP 仍是 TCP 成功后 SSH 关闭。现有证据不能证明训练仍在运行，也不能断言是余额、手动停机、无卡启动或平台回收。本轮没有执行任何远端停止、删除、覆盖、上传或关机动作。

恢复后先读取 queue、task state、进程和 GPU 状态。扩展旧 3 项与其他 attempt 先生成只读 SHA-256 incident 清单，再在独立 `/root/autodl-tmp/v3_event_extensions_eager_v2` 部署 `c78b87a`。只有三区域等价性门、9 项 loader-aware CUDA 探针、重新排产和 plan digest 人工复核全部通过，才启动完整 9 项重跑。33 项 baseline 全程保持原训练契约。详细证据见 `docs/audits/protocol_v3_event_extension_eager_relaunch_2026-08-09.md`。

## 最新进展，2026-08-09 14:08 四卡 5090 恢复与动态接力运行

`36571` 对应的 4 卡 RTX 5090 已恢复。另一台扩展主机欠费无卡，不再进入当前训练计划。baseline 在主机中断前已正式验收 6/33，失败数为 0。中断时 Orange DGCRN、Orange AGCRN 与 Alameda D²STGNN 没有可用的 optimizer resume 状态，因此从 epoch 0 建立 `attempt_0002` 重跑。恢复前 queue、formal tree 与 staging tree incident 清单 SHA-256 分别为 `10ed910c7e33847fa32931153b651c6c612cd99b7c0e06658dacc99a66663deb`、`acaddda23479ba74af3b38d26d8fc139d081c06c9de4f529207513c24e476070` 和 `b385567c5410a5c0a838bbe45ed9462255d7018ed888474fe57f439dbbfc9dfd`。14:08 实时状态为 baseline running、6/33、0 failed。Orange DGCRN、Orange AGCRN 与 Alameda D²STGNN 的新 attempt 分别完成 epoch 1、4 与 4。

旧扩展四卡机已欠费无卡。新扩展使用当前主机的空闲 GPU3，并与 baseline 保持独立 checkout、control 和 results root。训练代码 SHA 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。三地区 train shuffle、val 与 test 数据流的 eager 与 lazy 字节级对照全部为 `exact_match`，audit SHA-256 为 `083f9554fd305beed8ccea91385b59230479dd6edd965974959b23cdd9981f6e`。9 项真实 CUDA 探针全部通过，probes 与 calibration SHA-256 分别为 `4004d54e282e3b13fb2d12821e1c7c84a8180c9ede9b435b18257441fb59b4a9` 和 `00f89733654d306f5accc370ab8063fafe5f106aa0d4df7e2bf5d4b36436b8fa`。

正式矩阵 `protocol_v3_event_extensions_eager_5090x4_v2` 与 baseline 共用 host ID `event_rtx5090x4_full33`，plan digest 为 `2b8bc204115a32bf40d244a94938eb5db27ea229bbed1427542b74c200b4ef3a`。四个 shard 的探针保守预计为 41.86、34.52、39.49 与 37.85 小时。物理 GPU3 已启动正式 gpu00 shard 的 Orange NDR-Net，使用 eager 加载和 microbatch 8。14:08 已完成 epoch 2，validation MAE 从 `11.80291315` 降到 `11.59224552`。提交 `e21baf2` 的动态接力器会让任一空闲物理 GPU 领取当前预计最长的未启动 shard，避免固定映射在单机剩余工时变化后造成空转。relay 脚本 SHA-256 为 `a94db88f47aface7c6d9d85958eaf5b4084c84ece8d8bb9197c33b821627eab9`。旧 host ID 修正 incident 与动态升级 incident 清单 SHA-256 分别为 `f209d12b952cacb39a95b765b120677b06c44e1e86a9e876bdedfe57e4778cf2` 与 `1502f5fe42a3cd1e6d15c30b51df7f84ebb37ffe131a02b50ab876d239242495`。升级过程没有停止训练。

14:08 复验时，baseline controller、三个 baseline trainer、Orange NDR-Net trainer、baseline finalizer、动态 relay 与 extension finalize waiter 全部存活。正式控制根没有异常 exitcode，尚未启动第二个扩展 shard。

baseline finalizer 已改为只发布不关机。提交 `ad70d5a` 的收尾等待器只在 baseline 发布成功且 9 项 aggregate 成功后启动扩展 finalizer。Hostinger 的 xray 为 active，4 GB swap 空闲，根分区可用 `45030260736` bytes。post-publish 主脚本与 eager v2 包装器 SHA-256 为 `31295bce299f8ff85db65fb033dcccd406c2a2c3cf8888706b6a3057991acb6b` 和 `341bdcd8866eed4d7032ddac8b506e4d4de889e72b95b2eb3ae51723f7700e56`。最终表将按同主机不同代码 SHA 标记为 `target_disjoint_event_no_leak_code_stratified`。只有 baseline 严格发布、extension 严格发布、两份只读 composite 与最终 42 行表全部通过后才自动关机。任一门禁失败都保持开机。

## 最新进展，2026-08-09 15:03 已完成任务开始增量灾备

当前全部正式训练都在 `36571` 对应的单台 4 卡 RTX 5090 主机上。baseline 使用物理 GPU0 至 GPU2，扩展 Orange NDR-Net 使用物理 GPU3。动态接力器将在 baseline 分片释放 GPU 后继续启动剩余扩展 shard。

本次复核明确了现有断点边界。baseline 的 `ckpt_best.pt` 包含 epoch、model state、最佳验证值和配置，扩展 checkpoint 包含 model state、最佳验证值与最佳 epoch。两者都不保存 optimizer、scheduler、下一 epoch 游标、early-stopping 状态或随机数状态。runner 只能跳过已经 completed 并重新验收通过的正式任务，不能从训练中断的 epoch 续跑。因此此次主机中断后 6 个 completed 任务得到保留，Orange DGCRN、Orange AGCRN 与 Alameda D²STGNN 三个进行中任务必须从 epoch 0 重跑。

新增 `scripts/remote/linux/protocol_v3_event_incremental_backup.py` 作为独立灾备侧车。它不修改 baseline 或扩展训练 checkout，也不改变训练代码 SHA、plan digest、config hash 或正式发布契约。侧车按正式 plan 重建 invocation 和 config hash，复用 runner 的 `validate_result` 严格复验 completed task，再以低 CPU 与 idle I/O 优先级逐文件传到 Hostinger 隐藏 `.incremental` 树。远端先校验文件 SHA-256，再写 task marker。staging checkpoint 和运行中半成品不会上传，隐藏树也不会被当作正式发布。

baseline 与扩展侧车 PID 为 `16661` 与 `16662`，脚本 SHA-256 为 `e797f2a218f7f32d0b3a6c2dc6d454f92af96132f9e1d34c541863b906e0f9aa`。首个任务 `v3e010_event_dgcrn_alameda_seed42` 已在 Hostinger 完成 6 文件、`168730616` bytes 和 task marker 的复验。其余 5 个既有 completed baseline 任务正在补传。之后每新增一个 completed task 都会自动回传。最终论文表仍必须等待两个矩阵完整 postflight、正式原子发布、两份 composite 和 42 行合表通过。本地全仓回归为 611 passed 和 2 个既有 warning。

## 最新进展，2026-08-09 20:26 Orange IGSTGNN 关键路径已启动

Orange NDR-Net 的实际训练从 13:40 运行到 20:25，明显快于 41.86 小时的单步探针外推。旧动态接力器仍按探针的 shard 总量排序，下一项会选择 gpu02，而不是实际剩余关键路径 gpu01 的 Orange IGSTGNN。继续该顺序会让 Orange IGSTGNN 延后约 5 到 7 小时。

提交 `0648b76` 把接力优先级改为显式且严格校验的 `gpu01,gpu02,gpu03`。非法、缺失或重复顺序会 fail closed。该优先级只改变哪个空闲物理 GPU 领取哪个既有正式 shard。九项正式 plan digest 仍为 `2b8bc204115a32bf40d244a94938eb5db27ea229bbed1427542b74c200b4ef3a`，plan 文件 SHA-256 仍为 `5a43e9224572d3413e8ca6753691a8923bc92c49617871f14eafc23d000302d4`。训练 checkout、manifest、config hash、有效 batch 和结果身份均未改变。

远端切换时只暂停原 relay 与 finalize waiter，baseline 三个 trainer 和 Orange NDR-Net trainer 全程存活。旧控制层快照保存在 `relay_priority_reorder_20260809T1218Z`，清单 SHA-256 为 `c3261aa55b14d92bc0b70f16b44fde2433c1dc2635d2e56669c70c5a4c6b22e5`。新 relay 脚本 SHA-256 为 `97633145e9605678d92678ebbb03ea2b7d892072c0bb02fe2936c435df0252f3`。新 relay 与 finalize waiter PID 为 `50215` 和 `50364`，均为独立持久会话。

Orange NDR-Net 于 20:25:45 通过 runner 五文件验收，gpu00 queue 为 completed、1/1、退出码 0，summary SHA-256 为 `598d5e2838433d1ad8b3ddb503b554ee0f5b10be81900fcedb1b1c0e621815f0`。接力器在 9 秒后的 20:25:54 把同一物理 GPU3 分配给正式 gpu01，启动 `v3x008_event_igstgnn_orange_seed42`。复验时 queue 为 running、0/1，CUDA 利用率为 95%，显存使用 16505 MiB。baseline 同期为 10/33、0 failed，扩展为 1/9、0 failed。全仓回归为 613 passed 和 2 个既有 warning。

## 最新进展，2026-08-10 14:52 无泄漏 42 行论文对比表正式完成

`protocol_v3_event_full33_5090x4_v1` 的 33 项 baseline 与 `protocol_v3_event_extensions_eager_5090x4_v2` 的 9 项扩展均完成训练，runner 侧分别为 33/33 与 9/9，失败数均为 0。baseline source-host postflight 验证 33 项、202 个文件和 `7839685515` bytes。extension source-host postflight 验证 9 项、58 个文件和 `2365389476` bytes。两棵结果树均在 Hostinger 完成逐文件 SHA-256、隐藏 incoming 精确文件集校验、原子提升和只读 composite。

baseline composite 状态为 `verified_event_full33_single_host_33`，index SHA-256 为 `2a6ad2eadfd1ecf5af3604bee632eaa339e1aed7add59e915a71f9d422dd767e`。extension composite 状态为 `verified_event_extensions_single_host_9`，index SHA-256 为 `3a3970c627af6f709100ec9ebeeab215f834e0f1f65fc359423197ec29bf9ed4`。最终合表状态为 `verified_no_leak_comparison_generated`，包含 Alameda、Contra Costa 与 Orange 三地区的 14 种方法，共 42 行。

最终表全部使用 `protocol_v3_noleak_dedup`、cache schema 3、seed 42、最多 30 epoch、相同 target-disjoint event split、显式观测 mask 和统一的评测指标。11 个 baseline 的训练代码 SHA 为 `d23a991a570fd1c59e3d2ac8315a5e4063e8097f`。BiST、IGSTGNN 与 NDR-Net 的训练代码 SHA 为 `2754b9cdfa5ccf3baca269e83c7379ff506a0e26`。两个矩阵在同一物理主机运行，但代码 SHA 不同，所以最终有效性标签为 `target_disjoint_event_no_leak_code_stratified`。论文中只能把它描述为单种子、code-stratified 的无泄漏比较，不能描述为同一 checkout 的同质矩阵，也不能混入 `official_protocol_with_known_overlap` 的 Table 4 数字。

最终产物已从 Hostinger 拉回 Mac，目录为 `outputs/protocol_v3_noleak_dedup/event_full42_20260810/verified_table/`。CSV、Markdown、LaTeX、manifest 与 PDF SHA-256 分别为 `9817f5086f836f17265d87071271cc3b724b62420a2db2f2f4f5401e2f37651c`、`fa5ae4214c93791b7f9f74f3c2a3850959887426c0d769b0dbfb77994c479cf1`、`c28d06ecb3b64483d2f9160304848d04de7ad2905b94445df3b3680689a7f0e7`、`03796d1c63fcc571e80fc0bda794b6ee146090af3ef1d256e35bdd5bd02bb5ea` 与 `f2257394909be53b809bd8d09209f5524c1b1813c1ce09652bc7b0b07d6864e8`。

收尾时发现两项控制层契约缺陷。GraphWaveNet 的正式 producer 只承诺 scalar MAE，旧 composite 错误要求 summary 中存在 12 步 `mae_h`、`rmse_h` 和 `mape_h`。提交 `7fff5baf43fa37d850fc07118a91d9b1f930b2f9` 改为从已绑定的 `test_predictions.npz` 只读重算表格指标，并对其他模型继续核对 summary 与 prediction artifact。42 份正式结果的逐项数值复验显示，非 GraphWaveNet 指标最大差为约 `3.55e-14`，GraphWaveNet scalar MAE 的最大 float32 舍入差为约 `5.83e-7`。另一项缺陷是扩展聚合器会把已存在且重新验证通过的正式任务标记为 `skipped_valid`，旧 postflight 只接受字面值 `completed`。提交 `d98907e9a0513fa7abb8170a534ec194893ace4b` 在继续验证任务身份、`config_hash` 与全部 artifact 的前提下接受这两种 runner 终态。两项修复均未改写任何训练 summary、checkpoint 或 prediction artifact。

第一次 baseline post-publish 因 GraphWaveNet summary 契约不匹配而 fail closed。一次无卡模式 baseline postflight 因 2 GB cgroup 内存上限被杀，失败退出码均已归档。extension 第一次 postflight 因 `skipped_valid` 状态被旧门禁拒绝，失败回执同样保留。恢复后 baseline 与 extension 均重新完整运行 source-host postflight。extension 最终闭环日志明确包含 `POST_PUBLISH_START`、`POST_PUBLISH_OK` 和 `FINALIZE_SUCCESS`，fresh composite 与已有 42 行表逐字节一致。最终控制提交 `d98907e` 的全仓回归为 614 passed 和 2 个既有 warning。

## 最新进展，2026-08-11 seed 42 收敛审计已完成训练前代码门禁

现有无泄漏 42 行表保持只读。它统一使用 seed 42 与最多 30 epoch，适合作为固定训练预算下的描述性比较，但部分方法在预算上限附近仍在改善，因此不能把现有领先幅度直接写成充分收敛后的差距。

新增独立矩阵 `protocol_v3_event_convergence_seed42_39_v1`。矩阵包含 13 个可训练方法乘 Alameda、ContraCosta 与 Orange，共 39 项，不含无参数 HL。数据仍固定为 `protocol_v3_noleak_dedup`、cache schema 3、target-disjoint event split、显式观测 mask 与同一组 H3、H6、H12、Average 指标。当前阶段只运行 seed 42，用来判断训练预算与 scheduler 修复是否稳定改变排序。它不是最终多种子论文表，也不覆盖既有 42 行结果。

baseline 最大训练轮次提高到 100，patience 为 30。DCRNN 恢复原 100 epoch 里程碑 `[10,50,90]`，避免 patience 在关键降学习率前误杀。D²STGNN 恢复 `[1,38,46,54,62,70,80]`。BiST 与 IGSTGNN patience 为 20。NDR-Net 不启用 early stopping，但测试前恢复最佳验证 checkpoint。每项新增必需的 `training_diagnostics.json`，逐轮记录训练目标、全局验证 MAE、实际学习率、下一轮学习率、stale 计数、最佳轮次与停止原因。

多卡控制层统一调度十个 baseline、BiST、IGSTGNN 与 NDR-Net。正式启动前会在目标主机对 39 项逐项运行 loader 加完整优化步探针，并生成 task 级 calibration。控制器按实测工时做最长任务优先装箱，并隔离不同微批组。历史显存兜底为 Orange STTN 与 D²STGNN 使用 microbatch 16，Orange IGSTGNN 与 NDR-Net 使用 microbatch 8。新主机实测探针与 plan digest 人工复核完成前，不启动正式训练。

混合 runner、100 epoch 契约、统一诊断产物、GraphWaveNet 探针、39 项 calibration、多卡控制器和 `event_convergence` source-host postflight 已通过本地门禁。全仓回归为 633 passed 和 2 个既有 warning。该段记录的是训练前快照。完整训练前记录见 `docs/audits/protocol_v3_event_convergence_seed42_preflight_2026-08-11.md`。

## 最新进展，2026-08-11 09:40 seed 42 收敛审计四卡正式启动

独立矩阵 `protocol_v3_event_convergence_seed42_39_v1` 已在主机 `autodl-container-wh1skgkhxg-578ae9cd` 的 4 张 RTX 5090 上正式启动。既有无泄漏 42 行表保持只读，本轮 39 项只用于检查 100 epoch 训练预算、DCRNN 与 D²STGNN scheduler 修复以及方法排序是否稳定。

缓存只读审计再次通过，bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。三区域 train shuffle、val 与 test 各 20 个 batch 的 eager 与 lazy 数据流逐字节一致，loading audit SHA-256 为 `083f9554fd305beed8ccea91385b59230479dd6edd965974959b23cdd9981f6e`。39 项真实 CUDA probe 全部通过，probes 与 task calibration SHA-256 分别为 `4c68571c4ee26b58c56212ba0d59e6d301a7735510ed40bf2ee39e7286876c70` 与 `a6d1c9015b29b70fce51e84e47f0f1fcc33e8a85db9b4e35eadf3841fa2d359a`。

第一次正式启动暴露扩展 trainer 的脚本入口缺少项目根导入。Orange IGSTGNN 在任何 epoch 前以 `ModuleNotFoundError` 退出。发现后立即停止四个 shard，没有任务完成或正式目录被接受。incident 清单 SHA-256 为 `b6c1e8d31c0bb4e771bae6ea4e8e0457510098c9a726e33d29535af469347341`。提交 `0338e5ee93d0cf8ef2e162186310c32130fa31df` 增加与 baseline 一致的入口引导，不改变数据、模型、损失、优化器、微批或指标。隔离入口测试、本地全仓 636 项、远端专项 19 项和 ruff 全部通过。

正式 plan digest 为 `f84709ccf5debcdcd6a766497095d3ca0024bd72614beb6a2785f2ad6f11b9b0`。GPU0 至 GPU3 的任务数为 17、18、2 与 2，探针预计工时为 114.132、114.190、122.362 与 29.697 小时。最长分片约为 5.1 天。Orange STTN 与 D²STGNN 使用 microbatch 16。Orange IGSTGNN 与 NDR-Net 使用 microbatch 8。各方法的有效 batch 与一次 optimizer step 契约保持不变。

09:40 的正式状态为 running、0/39 completed、0 failed。controller PID 为 `41801`。四张卡分别执行 Alameda AGCRN、Alameda LSTM、Orange IGSTGNN 与 Orange STTN。Orange IGSTGNN 已完成模型构建并进入训练，首次入口故障没有复现。

提交 `40d2cca34bfc2925f20a0afdff56fa09f8363dc6` 让增量灾备侧车支持 convergence plan。侧车不进入训练 checkout，不改变正式代码 SHA 或 plan digest。PID 为 `43230`，每个 runner completed task 会先重新验收，再逐文件上传 Hostinger 隐藏 `.incremental` 树。Hostinger 当前约有 24 GB 可用空间，xray 为 active，4 GB swap 正常。最终 finalizer PID 为 `43321`。它只在 39/39、runner 退出码 0、source-host postflight、容量门、逐文件 SHA-256、精确文件集与原子提升全部通过后自动关机。任一失败都保持主机开机。完整证据见 `docs/audits/protocol_v3_event_convergence_seed42_launch_2026-08-11.md`。

## 最新进展，2026-08-11 20:04 收敛审计首项已保全，GPU 主机状态不可验证

100 epoch 收敛审计最后可验证进度为 1/39。`v3c001_event_lstm_alameda_seed42` 已完成 100 epoch，runner return code 为 0，最佳验证 checkpoint 位于 epoch 99。Hostinger 隐藏增量灾备树已对该任务完成逐文件复验，marker 状态为 `verified_completed_task_backup`，8 个文件的 files digest 为 `6bd8c7313b453745bd7ec0a40ad82df5fb130a17484241364b95e2ad0b17b4fa`。该单项直接说明 30 epoch 对 Alameda LSTM 会截断仍在改善的训练过程，但不能据此替代完整 39 项结果或推断所有方法排序。

20:04 新加坡时间，`connect.weste.seetacloud.com:46241` 可以建立 TCP，但远端在 SSH 鉴权前关闭连接。当前不能证明四卡主机仍在训练，也不能判断原因是欠费、关机、平台回收或入口异常。启动时记录的 controller、增量侧车和 finalizer PID 不再作为存活证据。矩阵状态更新为 `last_verified_1_of_39_remote_unreachable`。

当前 checkpoint 不包含 optimizer、scheduler、下一 epoch 游标和随机数状态。主机恢复后应先只读审计 queue、task state、进程、正式树和 staging。已完成且通过 Hostinger 增量复验的任务可以保留。中断中的单项不能从原 epoch 严格续跑，继续时必须从 epoch 0 新建 attempt，并为旧 staging 生成 incident 清单。现有 42 行无泄漏表继续保持只读，其状态不受本次中断影响。

导师汇报与论文路线已经整理到 `docs/reports/导师阶段汇报与论文路线_2026-08-11.md`。当前建议把论文主线写成无泄漏评测协议与 NDR-Net 方法的联合贡献，而不是只写成对 IGSTGNN 数值的复现或超越。投稿前仍需完成收敛审计、明确最终训练控制、多种子统计、考虑时序相关性的配对检验、效率报告，并优先补充外部数据集或跨地区泛化证据。

## 最新进展，2026-08-12 20:16 FDN 无泄漏三地区补充已启动

旧 FDN event 结果使用了审计前缓存，状态仍为 contaminated，不能填入无泄漏论文表。现有 v3 FDN 结果只覆盖 full-window 协议，也不能与 event 表直接混用。因此新增独立矩阵 `protocol_v3_event_fdn_5080_v1`，用正式 v3 无泄漏 event 缓存重新训练 Fourier Dual Net 的 learnable-K3 版本。

矩阵包含 Alameda、ContraCosta 与 Orange 三个 seed 42 任务。训练设置为 batch 32、最多 30 epoch、patience 6、Adam 学习率 `1e-3`、weight decay `1e-4`、gradient clip 5.0 与逐 optimizer step 的 CosineAnnealingLR。数据固定为 cache schema 3、唯一 forecast origin、target-disjoint event split、active event split isolation 和显式观测 mask。FDN 不读取事故标签。canonical manifest SHA-256 为 `2fb0760e1d5b9d1b7217a7d8ea61763ed5492ba45b26044148409156d4fb475b`，正式训练代码 SHA 为 `b95a57fa7b789852c983d9b07e74a3847d1db14b`。

5080 启动前通过 clean checkout、专项测试、缓存严格复验与 Orange 真实 CUDA 训练探针。2026-08-12 20:04:48 新加坡时间正式启动。当前 runner 为 running、0/3 completed、0 failed，正在 Alameda。前 5 个 epoch 各约 133 秒，验证 MAE 从 `17.9032` 持续降到 `14.0397`。GPU 利用率为 80% 至 95%，显存约 4.9 GB，没有 OOM 或非有限值。

训练之外新增两道只读控制。增量备份侧车只处理 runner 已完成且重新验收通过的正式任务。群晖 NAS 当前 SSH 不可达，侧车保持每 60 秒重试，不会影响训练。持久 postflight 只在 3/3 和 runner 退出码 0 后执行，逐任务重建 config hash，复验 summary、checkpoint、prediction、train log 与 decomp mask，并为全部允许发布文件生成 SHA-256 清单。最终合表器从已绑定的 prediction artifact 按 baseline 相同公式计算逐 horizon 与平均 MAE、RMSE、MAPE，把 FDN 插入原 42 行只读表，生成 45 行论文表与 A3 横版 PDF。合表器和 PDF 已用合成验收树完成测试与视觉检查。当前本地全仓回归为 655 passed 和 2 个既有 warning。正式训练完成和 postflight 通过前，本节不报告新性能数字。

## 最新进展，2026-08-12 21:30 FDN v1 失败已保全，v2 已独立重启

FDN v1 的 Alameda 已完成全部 30 epoch，最佳验证 MAE 为 `12.381733565128126`，但在测试推理时触发 `KeyError: affected_mask`。因此 v1 runner 以失败状态退出，进度为 0/3，没有接受任何正式结果。失败事件保留 8 个文件，共 2,654,734 bytes，incident manifest SHA-256 为 `8b3438baf85c9fa3a77b33d2834ec5c76e2f931f4c7031d4e897ea439d3b7105`。

根因是 FDN event trainer 没有在 test split 显式启用事后分组标签。修复后，test split 可读取 `affected_mask`，train 与 val split 仍然不读取该标签，所以没有把事故分析标签注入训练。修复通过本地 656 项全仓测试、傀 10 项专项测试和真实 Alameda 数据边界探针。

为避免同一任务混合两个代码 SHA，没有使用 v1 checkpoint 直接补推理。2026-08-12 21:29:26 新加坡时间，新矩阵 `protocol_v3_event_fdn_5080_v2` 在 `D:\t4\v3_event_fdn_repo_v2` 和 `D:\t4\v3_event_fdn_5080_v2` 从 epoch 0 正式启动。训练代码 SHA 为 `0981f8bce84a942e8efd958867802be2a59d3cdb`，训练 manifest 与之前完全相同。当前 0/3 completed、0 failed，正在 Alameda，GPU 利用率为 95%，显存约 5.0 GiB。新的持久训练、增量备份和 postflight 计划任务已分别启动。全部 3 项完成、源主机严格复验和 Mac 回传完成前，不报告 FDN 正式性能数字。

v2 首个 CMD postflight 等待器在启动时序窗口内退出，未运行复验且未影响训练。控制层已替换为 PowerShell 持久等待器，SHA-256 为 `76b53179e40ef2726de6b930679406c22ce6c1e6f9963747adff0db43a57b5e8`。新等待器已在 SYSTEM 计划任务下确认 running，它只等待 runner exitcode，不触碰 trainer 或正式结果。这是控制层修复，不改变正式训练代码 SHA、manifest 或 config hash。
