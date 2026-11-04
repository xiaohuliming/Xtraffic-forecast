# NDR 性能与泛化比较的执行覆盖

日期：2026-10-14，Asia/Singapore

## 当前研究结果

原八项 event 开发矩阵仍在 5080 按冻结顺序执行。当前只有 Alameda 和 ContraCosta 原 NDR 的两项完整结果，validation global MAE 分别为 10.745627126021652 与 11.27520497996187。它们已独立复算及完整回传，不能由此判断候选或 SOTA 胜出。来源为 `control/baseline_pair_partial_verified.json`，最新实测句柄见 `stage1_poll_27.json`。Alameda mask_signed 已完成 21/30 轮，supervisor 与 worker 均存活。后继校准仍为 `waiting_verified_predecessor`，没有 CUDA worker，见 `transfer_calibration_poll_07.json`。

本轮补齐的是公平比较的实际代码与工程验证。原队列、部署字节、48 小时截止和旧结果均未改变，没有新增并行 GPU 工作。

| 比较要求 | 当前可执行入口 | 尚缺的实验证据 |
|---|---|---|
| 原 NDR 与两种门控假设 | 原八项 `train_ndr_research.py` | 八项完整验收、三种子确认 |
| 节点无关模型在相同地区训练 | `train_ndr_graph_indomain.py` | H64/H240 的 CUDA 校准与完整训练 |
| 学习型跨地区强对照 | `train_shared_staeformer_transfer.py` | 真实 SharedSTAEformer 三种子训练 |
| 三种子两源模型共同目标测试 | `evaluate_ndr_source_cohort.py` | 六个完整 checkpoint、冻结测试和独立复算 |
| 原 NDR、STA、BiST、地区内 Graph 共同目标测试 | `evaluate_ndr_indomain.py` | 各真实方法完整训练及冻结测试 |
| 相同日期的三种子配对分析 | `analyze_ndr_cohort.py` | 真实已冻结评测报告 |
| 全窗口独立采样、训练与冻结评测 | `fullwindow_data.py`、`train_ndr_fullwindow.py`、`train_strong_fullwindow.py`、`evaluate_ndr_fullwindow.py` | CUDA 校准、同协议完整重训与真实冻结测试 |

## 学习型迁移强对照

SharedSTAEformer 保留原 STAEformer 的 152 维主干，将 adaptive embedding 固定为共享时间参数 `[12,1,80]`，没有节点身份参数。参数量为 1,061,172，空间 attention 仍有稠密 N² 成本。输入缺失处理与共享图模型采用相同 canonical history 和 source pooled statistics。标量中心来自 source template 的均值，尺度来自 source residual scale，这是明确的项目标准化配方。

模型输出与移植相同共享时间权重的原主干一致，动态 N、节点置换、被遮蔽 NaN、输入白名单和状态形状检查通过。真实 Alameda 与 ContraCosta 的有限 CPU smoke 已在新进程复算，差为零，属于 `verified_smoke_source_validation_subset`。它不是原论文 zero-shot 复现，也不能把 backbone、图、输入与 loss 的整体差异称为门控的单因素贡献。

来源为 `control/shared_staeformer_actual_module_probe.json` 与独立 [工程说明](shared_staeformer_transfer_engineering_2026-10-14.md)。

## 地区内公平比较与全窗口边界

SharedGraphNDR 的地区内入口使用原 event train/val、相同 train-only 周模板及节点残差尺度，在模型外传入静态道路与尺度。checkpoint 使用 `graph_indomain_checkpoint_v1`，完整验证状态为 `verified_graph_indomain_checkpoint_complete_validation`。重建与测试前检查真实 schema、train 统计和道路身份，不把两源权重伪装成单地区训练。

两地区 CPU smoke 每轮仅一个 train 和 val batch，虽完成三十个小规模循环，仍是 `smoke_only`。证据位于 `graph_indomain_cpu_smoke_final/local_verification.json`。完整训练尚未执行。

已验证的旧全窗口 27 项表在每地区使用 73561、15757、15757 个 train、val、test 样本，比例为 .7/.15、stride 1、历史和未来各十二步。新增 reader 直接绑定正式 traffic 和 graph 字节，另生成独立 full-window sampling identity，不打开 event samples/relation 数据，不把 event 数量当成全窗口数量。

全窗口 synthetic batch 已通过原 NDR、共享图、STA、真实 BiST 前向检查。真实 Alameda 只验证六十四步训练前缀和一个训练窗口，没有读取 test 数值或 Orange 数据。新 float64、unique raw train fitter 与旧 BiST 的 float32 scaler 差异需保留，新方法与强对照须重新同协议训练，不能直接把旧 30 轮背景表与新 100 轮结果称为充分收敛 SOTA。来源为 `control/fullwindow_protocol_gap_review.json` 和 `control/fullwindow_adapter_implementation_receipt.json`。

随后已补齐独立全窗口训练入口。NDR 四变体及 SharedGraphNDR H64/H240 支持完整优化器、scheduler、随机数、loader 游标恢复和独立进程 validation 复算，提交为 `dd0ebbab56248e1ec15a0077199162e21f01efe0`。真实 Alameda baseline 与 ContraCosta Graph 的有限 CPU smoke 只使用极少 train、val batch，新进程复算差为零。来源为 `fullwindow_cpu_smoke_final/local_verification.json`，属于工程验收。

真实 STAEformer 与锁定原 BiST 的全窗口 runner 已实现 native 和 matched 配方，提交为 `3f644b69dc9e391ad28690f2c988ec341b02304b`。BiST native 保留整有效 batch 的 BatchNorm 语义，不能把微批拆分称为相同原配方。两地区各一个 CPU train、val batch 的校准已通过，尚未完整训练。来源为 `control/strong_fullwindow_runner_implementation_review.json`。

全窗口冻结评测提交为 `f390ea5bac97c1652424f4b85be5ed039bef8da8`。三种子、真实 class、完整 validation 和独立进程证据通过后，专用 grant-bound reader 才开放精确 test 窗口。一次窗口供所有模型和五种场景配对使用。原开发 reader 的 test 数值 API 继续关闭。来源为 `control/fullwindow_evaluation_engineering_receipt_20261014.json`，当前只有合成评测验证。

三种子分析已支持 `frozen_fullwindow_evaluation_complete` 矩阵报告。按声明的模型 ID 选择真实方法及训练种子，绑定 source、reader、sampling、statistics 和 data identity，保留各侧训练地区与 float64、旧 float32 边界。分析范围为 `matched_fullwindow_regional_supervision`，不与 event 或跨地区监督范围合并。目前没有真实全窗口 checkpoint cohort 或正式 test 分数。

## 目标测试与验证

Source cohort 门禁绑定两个真实模型类、各三种子、完整训练与完整 source validation、source pool 数值身份和独立进程证据。新证书或已记录的 fresh-runner receipt 可证明独立复算。一次目标窗口读取供所有方法和五种缺失场景使用，历史校准的 prefix 与 mode 不重置同一家族预算，失败照计。三种子、两种模式及两次独立推理需预先登记十二个种子推理槽。固定 controls 使用 source-only 统计量，STA 的 center/scale 仍固定于 source checkpoint。

地区内与跨地区模型的训练范围在每模型报告和三种子合表中保留。Orange 是结构开发期间未读监督评分的主确认地区，另外两个 LOCO fold 只表示排除于各自权重和统计拟合，不能写成研究期间完全未见。

2026-10-14 新增全窗口 cohort 解析的 35 项合成测试通过，包含真实合成 evaluator 输出、声明模型选择、真 class 和 seed、身份漂移、event 混入及 float32 误用。分析提交为 `1566581b0da8d954675d65c1a46732e0b650fca8`。相关研究目录最新汇总为 337 passed，0 failures、errors 或 skipped，Ruff、compile 和 diff 检查通过。来源为 `control/ndr_related_validation_v3_20261014.xml` 和 `control/ndr_related_validation_summary_v3_20261014.json`，测试时该代码提交为 clean。先前 271 项记录仍保留。原全仓缺资产与安全锁失败没有被绕过，这不是全仓回归或精度报告。

下一步仍是完成现有固定八项，保全原产物并按整矩阵判断；取回随后真实 CUDA 校准成本，再冻结完整训练和确认预算。代码可运行、CPU smoke 和 metadata inspection 不构成 SOTA 结论。


## 固定开发矩阵的本地自动保全

原训练继续运行，最新独立检查为 Alameda mask_signed 完成 22/30 轮，源端 supervisor 与 worker 均存活。本地观察进程 PID 为 45878，每 300 秒查询同一固定八项的真实句柄和 queue.completed。既有两项基线再次通过本地精确文件树与 checkpoint 检查。

新完成项调用已验证的 `retrieve_ndr_development.py`，再次核对独立完整 validation、源端传输前后哈希和本地精确文件树后才发布。完整矩阵分析只在源端 completed、八项全部保全及 postflight 准入通过后生成。SSH 观察超时只记录不可用并继续等待，不重启训练。计划摘要、完成顺序、绝对截止和本地 Python 源码均被锁定。没有修改训练队列、增加 GPU 工作或读取真实 test。

一次性检查使用独立记录文件，实际回归确认主观察进程的状态与单实例锁均保持不变。控制器通过 Ruff 和 compile，观察进程与源端句柄已实测存活。证据为 `control/stage1_preservation_watch_launch_verified.json`，实时状态为 `control/stage1_preservation_watch_status.json`。这项控制器不计入此前 337 项模型工程测试，也不构成新的模型精度结果。控制器只在本地运行，工作站继续保持开机。


## 04:00 后连接状态变更

Tailscale peer 自 04:00 离线，既有两条 SSH 路径均超时，当前源训练句柄无法验证。前述 running 和进度数属于对应回执时的已验证历史状态。最后成功回执 `stage1_poll_30.json` 为 2/8，Alameda mask_signed 完成 23/30 轮。本地保全进程仍存活，等待连接恢复，不因观察失败重启训练。当前矩阵和后继校准的完成情况未知。见 `ndr_5080_connectivity_incident_2026-10-14.md`。


## 连接与严格恢复后的更新

源端故障已经确认与系统关机相符，旧结果和严格断点双端保全后，原八项沿同一配置与绝对截止继续。最新矩阵 3/8，ContraCosta mask_signed 完成 27/30 轮，真实句柄存活。Alameda mask_signed 的完整 validation MAE 为 10.678324453161249，十一文件已经回传复验。后继六项新守候进程 39636 正等待完整前置验收，没有 CUDA child。相关研究目录最新 383 项通过。以上更新替代此前的暂时不可验证状态，仍不是候选晋级或 SOTA 结论。见 `ndr_strict_recovery_2026-10-14.md`。


## 2026-10-15 近期公开强方法的补齐范围

DualCast-S 已有真实 v3 train、verify 和 calibrate 入口，此前无 trainer 的历史描述已过时。当前九种成本配置仍将其延期；同协议 CUDA 成本、完整训练与共同冻结评测尚未完成。仅比较现有 NDR、Graph、STA 和 BiST 不足以自动支持广义 SOTA。

已在 5080 的独立新根准备锁定作者上游，commit、tree、官方 origin、四文件哈希与 clean 独立回读通过。完整成本入口及专用 dualcast kind 的共同冻结评测适配正在独立分支实现，既有四十三文件 producer 字节已再次确认不变。原八项、六项及九配置成本草案没有调整，没有新增 CUDA 工作。具体边界和回执见 ndr_sota_scope_gap_2026-10-15.md。
