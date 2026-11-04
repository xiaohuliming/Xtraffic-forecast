# NDR 跨区域训练与强对照准备

日期：2026-10-13，Asia/Singapore

## 原始目标与当前运行

用户目标保持为更好的 XTraffic 预测性能、更强的泛化能力，并以 RTX 5080 实验争取超过现有强方法。当前八项 NDR 门控开发队列仍在独立 5080 源码快照中运行，本工作区不改动该快照，不新增并行 GPU 工作。

本分支 `codex/ndr-transfer-training` 从研究提交 `006bc5c` 开始，用于补齐节点无关模型的实际训练和留一地区评估。前一轮的模型形状验收不作为泛化精度结果。

## 严格 source-only 协议

每折仅以两个源地区的 train 拟合、训练，以这两个地区的完整 validation 选 checkpoint。第三个地区不参与训练或开发选优。预处理采用源地区按地区等权的 `7 × 288` pooled weekly template 和源残差尺度，源 train、source validation 和目标推理始终使用同一预处理定义。

目标读取器不消费目标的 cached climatology、normalizer、incident 或 affected 数组。原始 flow 只取当前 forecast origin 所需历史，最终标签读取需要显式冻结评估门。cache 整文件哈希验证不等于消费其中的统计量或监督标签，访问审计区分二者。

使用目标训练时间前缀拟合逐节点气候态和尺度的实验单独标为 historical calibrated transfer。目标前缀截止和回退规则提前固定，模型权重保持冻结，目标 validation 不用于重新选参数。

一个早期 CPU 工程探针曾在 Orange 读取一个 validation batch，其中含未来 flow 和 mask。该探针只检查随机模型的前向形状与有限性，没有计算预测误差或开发评分。该事实保留，不能把这次探针称为 target-label-unseen。后续工程探针改用仅返回当前 origin 及以前历史的 API，目标 validation 与 test 的监督评估均要求冻结门。

## 模型与优化规则

SharedGraphNDR 的构造配方不包含节点数量或地区身份。动态节点尺度从 batch 传入，不添加地区专属参数。两个残差头与门控输出均还原为原始 flow 单位。

完整 effective batch 的有效点数、teacher confidence 总和和 specialization 权重均值在 microbatch 前冻结。原 soft anchor teacher 与 projected teacher 是显式独立开关。每 epoch 覆盖两个源地区的全部训练样本，通过预注册的区域贡献权重平衡不同样本量。source validation 先在各地区按全部有效点累计，再对地区 MAE 取简单均值选 checkpoint。

checkpoint 保存优化器、调度器、随机数、各地区 loader 的排列与游标、交错区域计划游标、预处理数值与身份、源码、缓存和运行时。中断只可在身份一致的 last checkpoint 恢复。

## 最终评估规则

冻结门绑定 checkpoint 哈希、源码摘要、构造配方、完整 source validation 的独立复算记录、两个源地区与 heldout 地区、预处理模式、目标 split、固定缺失场景和评估 ledger。先验证这些绑定，再读取目标监督 collection。

缺失场景固定为干净历史、随机缺失 10%、随机缺失 30%、末尾连续缺失 3 步和 6 步。改变的仅是历史 observation mask 与对应历史值，目标和目标 mask 保持一致。报告全部有效点的 MAE、RMSE、MAPE、十二步指标、实际日块误差和与计数。日块配对 bootstrap 重算分母，不能简单平均不同有效点数量的日 MAE。

持久化 ledger 对 test 试验预先计数，失败的试验也保留。它防止本工具无记录地重复读取测试集，不代表加密权限系统。不得把看过测试结果后的调参称为冻结评价。

## DualCast-S 强对照适配

作者公开仓库为 https://github.com/suzy0223/DualCast，锁定 commit 为 `50425c12f2fc4e6ba6069d028540d2373e5e945a`。模型源码 SHA-256 为 `47462de01c827296a8aed4b6e628bd7dd041fee64a209f026c7452cc3c65d336`。本地引用源码留在被忽略的研究 references 目录，没有把第三方源码并入本分支。

适配器执行锁定的作者模型体，只用原生 Torch 实现其实际使用的 degree 与 sum scatter，避免给共享主机安装旧版二进制依赖。历史 calendar 由真实 v3 weekday 映射到作者的 Friday-origin 分类，周末 flag 从同一 weekday 推导，保持原型分组含义。预测输入不包含 target。

默认模型维度 64 已通过 CPU 前后向形状和有限性 probe，另有五项适配测试通过。这不是精度、GPU 速度或显存结论。

原训练脚本的无 mask L1 和重叠窗口切分不能复用于 v3。原 reciprocal KL 辅助项在分布一致时无定义，保留单样本尾 batch 时需要明确的兼容规则。原 DBI 代码有 detached reduction，不能在比较中悄悄改成另一种方法。模型含节点数量相关的原型和位置参数，适合作同协议地区内强对照，不能直接作为节点无关迁移模型。

DualCast-S 已完成独立 train、verify 和 calibrate 入口以及 CPU 工程重载。两种具名兼容配方分别对未定义辅助项失败退出，或明确省略该批无定义项并逐批记录，原作者 reciprocal KL 和 detached DBI 的有定义表达式保留。详见 [兼容配方与工程证据](dualcast_v3_compatibility_2026-10-13.md)。CUDA 校准、完整开发训练和同协议性能比较尚未完成。BiST、原 NDR、原 STAEformer 仍是主对照，不用 CPU 工程证据替代完整性能比较。

## 已落地的工程验收

两源 pooled template 的正式拟合已完成，只使用 Alameda 与 ContraCosta 的原始训练前缀。源身份为 `733dd7ee2eadf151745cfc4ccf1962b9c3dd18cbba8fe0af568823c77b0ba63c`。未构造 Orange reader 的真实 CPU 训练 smoke 完成 30 epochs 和 120 次更新，之后在新进程独立复算 best checkpoint，两源 MAE 差均为零。仅覆盖各四个 source validation 样本，状态为 `verified_smoke_source_validation_subset`，不允许据此生成正式目标测试门。

默认预载两源 channel 0 与 observation mask 至 val_end，共 454,354,920 bytes。lazy 与 eager 的批次和梯度逐字节一致，目标角色在读取任何值前拒绝预载。访问审计区分物理 H5 读取与缓存切片。训练统计先拟合 source train，再进行预载。

新 source-only trainer 的源码摘要仅绑定实际 import 依赖和四个强制核心文件，共 24 个实际依赖。早期 smoke 的独立复验曾因未运行的 strong-reference 文件被旧全目录快照包含而拒绝，此日志保留。新入口修正过度绑定，没有放松任何已绑定文件的内容复验，没有修改正在运行的旧源码包。

真实 v3 Alameda 上，raw STAEformer 与锁定的 BiST 也完成有界 CPU smoke 与两个新进程重载验证，均为 `verified_smoke_only`。BiST 含 BatchNorm2d，拆 microbatch 会改变统计，native 默认为 batch 64、microbatch 64；matched-ndr 为 16、16。显式拆分会记录 BatchNorm override，不当成原生批次行为。

## 后续固定容量比较

低成本共享图候选固定 hidden 64、道路两层、八个全局 token、四个 head，参数为 123,832。另一项预声明容量候选固定 hidden 240，其余架构保持相同，参数为 1,602,760，相对 Alameda 原 NDR 的 1,582,308 个参数高 1.29%。该候选旨在检查性能差距是否来自容量，不称三个地区都严格等参数。

两个宽度先做同机 CUDA 校准，后续 source validation 的开发选择不使用目标区域评分。各模型都必须真实训练、独立完整 source validation 复算和回传复验，工程参数量不证明速度、显存或泛化优势。

## OpenCity 公开预训练对照边界

公开仓库已锁定为 `eca64c9d9c65ff803d0f9d02c65e2544a0d2e96f`。默认输入和输出都是 288 步，patch length 为 12，预测 head 固定 24 个 patch。直接提供本项目十二步历史时，原 patch 自适应的整数除法会产生零长度 patch，不是可直接加载的同十二步对照。

其 forward 的未来 lbls 参数只取 calendar 通道，不能据函数名推断使用了未来 flow。代码中的实例标准化仅由历史输入拟合。数据生成脚本设置 2020，注释称论文实验用 2019；尚未核验每个公开 checkpoint 的精确训练清单，因此不宣称它与 XTraffic 2023 完全独立。

尚未下载预训练权重或数据。若采用 288 步历史并只评前十二步，应作为有额外上下文和外部预训练的独立对照，不能拼成同十二步输入的同质表。

## 回归范围与队列衔接

NDR 相关目录验收为 167 passed，Ruff、编译与差异检查通过。额外全仓检查为 785 passed、83 failed 和一个既有 warning。其中 67 项缺少该隔离 worktree 未携带的历史数据、上游 checkout 或旧表 manifest，16 项被已有 sandbox-exec 哈希锁拒绝。没有修改历史资产或安全锁，不能把全仓写成全通过。XML 与分组记录保存在原工作区 `outputs/remote_runs/ndr_generalization_20261013/control/`。

测试预算按候选源码家族和 heldout 地区计数，跨 checkpoint 种子、预处理模式和历史前缀不重置。首条访问固定家族上限，后续 gate 修改上限会被拒绝。三种子和两个预处理模式应在读取 test 前预声明六次预算，失败也占用预算。独立复算若需要再次读取 test，也必须提前纳入该预算。

两个图模型宽度只在当前八项队列完整结束、源 postflight、逐文件与完整同协议矩阵准入通过后进行串行 CUDA 工程校准。衔接控制器不自动启动下一阶段正式训练，不评估 target 或 test，不重启失败的前置队列。计时和显存准入成功后再冻结下一阶段训练计划。

## 2026-10-13 19:11 后续工程校准已排队

研究代码已提交为 `473197463917c586205fb9b106dc610133dc498b`。不含凭据、数据与第三方原始源码的部署包共 457 个文件，SHA-256 为 `08d684fe73ec2cc1d749e4069cfcca5e9f3e2ca7b663498c07c19a89ed8db524`。5080 上独立 clean snapshot commit 为 `734fd219389b13179068314cfc9524d74ed7b480`，两端 457 文件逐一哈希一致。

固定工程计划 digest 为 `a1e2553da1b19dc04cb39bdb74dd701f12749444a5c31c6f4a7093b47cda708b`。两个宽度分别执行三次预热和二十个计时 batch 的两源训练与 validation 代价校准，再以每源每轮两批、三十轮的有界 CUDA optimizer smoke 完整遍历两源 validation，最后各由新进程复算 best checkpoint。只有工程 scope，不自动进入完整新模型训练，不实例化 heldout 目标读取器，不评估 test。

独立计划任务为 `XTraffic_NDR_Transfer_Calibration_20261013`，无触发器，pythonw 后台运行。19:11 实测 supervisor PID 41776 存活，状态 `waiting_verified_predecessor`，没有 CUDA worker；它同时实测原 supervisor PID 6232 和当前原 worker 存活。没有抢占当前 GPU 工作。

前置条件是原八项队列 completed、原 plan digest 匹配、source postflight 通过、精确文件树与全部文件哈希一致，并通过独立矩阵准入的同代码、同 runtime、同目标、同 mask、同配方和非 smoke 检查。前置失败保持失败，不自动重启。观察失败仅等待，不据此推断旧任务停止。

工程准入总截止为 2026-10-16 00:10:22，Asia/Singapore，每工程 job 一小时上限，GPU 忙碌等待最多三十分钟。全部完成后仍保持开机，不自动传输。工程控制器与原训练使用不同源码根和结果根。

当前仍没有节点无关模型的正式 CUDA 精度、三种子、跨区域获胜或 SOTA 结论。后续正式训练计划须依据本轮实测工时、显存和完整 checkpoint 验收重新冻结。

## 前置准入的真实 schema 复核

已完成一项只读同行复核，直接使用已保全的真实 Alameda baseline config、summary 和 verification。其 21 文件源码摘要逐键匹配原冻结计划，配置摘要与两个报告一致，batch 16、microbatch 8、hidden 16、threads 3 和 reliable_anchor false 等比较字段齐全。

基于真实字段构造的完整八项合成 fixture 可以通过矩阵及衔接准入，Windows 反斜杠文件路径正确处理。真实状态仍为 running、1/8 时，准入返回未就绪，没有误启动后续 GPU 工作。相关八项测试通过，未修改已部署源码。这个检查只确认 schema 与路径兼容，不能替代将来真实八项完整验收。

## 两项基线保全与三种子统计准备

八项开发矩阵最新已完成两项原 NDR，Alameda 与 ContraCosta 的完整 validation MAE 为 10.745627126021652 和 11.27520497996187。22 文件、77,124,280 bytes 已回传，逐文件哈希、checkpoint 结构和源端前后不变性通过。这个进展不满足完整八项前置条件，节点无关候选的 CUDA 工程队列仍须等待，不启动正式训练或目标评分。

新增 `scripts/retrieve_ndr_development.py` 可逐项保全已完成结果，只有源端完整 postflight 和精确八项文件树通过准入后才生成整矩阵分析。工具已在两项真实基线上验收，回传与矩阵准入相关 17 项测试通过。文件布局见 `stage1_copied/`，两项参照摘要为 `control/baseline_pair_partial_verified.json`。

新增 `ndr_research/cohort.py` 与 `scripts/analyze_ndr_cohort.py` 只消费冻结评测报告，不打开 traffic 标签。三种子必须共享同一区域、split、目标、mask、日期、扰动、模型构造与完整训练配方。teacher、优化器与 epoch 预算不能混用；相同数值历史前缀的不同授权回执不被误判成不同方法。

统计按真实日期重采样，同一次日期抽样同时用于两方法和全部种子，并重新累计有效点分母。另给出配对种子加共同日期的层次 bootstrap，避免将 seed 与 day 的乘积当独立样本。seed 42 作为发现保留，seeds 1、2 的确认结果单列。三种子初始化证据有限，相邻日期的依赖未由独立日块抽样保留，区间只做描述性报告。相关 20 项测试和独立边界探针通过，比较范围和各模型训练地区在合表中显式保留，混合监督范围不会被汇总成同地区同监督条件。目前没有真实三种子精度结果。

Orange 是结构开发期未读取监督评分的主确认地区。另两个留一地区 fold 在权重和统计拟合中排除各自目标地区，但 Alameda 与 ContraCosta 曾参与结构开发，不能把补充三 fold 均值表述成三个完全未见地区的独立确认。早期 Orange 一个无评分工程 batch 的读取记录继续保留。

剩余执行缺口记录在 `control/execution_gap_review.json`。目前已冻结的是两容量选择与六项 CUDA 工程任务，没有完整的三种子乘三 fold 正式训练矩阵。下一步先取回实际校准工时和完整 source validation 工程重载，留出统计累计、checkpoint 和独立复验的额外工时，再冻结完整开发与确认预算。仅 forward 的校准 validation 时延不能直接当成完整评测工时。

## 地区内强模型冻结评测已具体化

新增 `ndr_research/indomain_evaluation.py` 与 `scripts/evaluate_ndr_indomain.py`，支持原 NDR、门控候选、原 STAEformer、真实 BiST 和 SharedGraph checkpoint。它们统一使用完整 event test 与五种缺失场景，共享真实目标、mask 和日块计数。clean 保留原基线正式缓存的因果填充输入，corrupted 只从扰动后保留的原始观测重建历史，不保留被遮蔽的真实值。

原 NDR 与共享图模型的整数 schema，以及强对照的字符串 schema 按具体模型种类验收。真实已完成 Alameda baseline 的 metadata-only inspect 成功，严格重建 1,582,308 个参数，没有建立数据 reader 或计算准确率。强对照与共享图的实际 checkpoint writer 也由合成工程测试覆盖。

最终 gate 要求结构开发关闭、完整开发训练与独立 validation 验收、具体 checkpoint 和配置哈希、同正式 bundle、原始输入处理策略及明确监督范围。共享 ledger 的家族键包含地区与模型家族，不因 seed、checkpoint、授权 ID、model ID 或 gate 路径更换而重置。每次访问按不同训练种子的推理槽计数，三种子的首次评估和独立再次推理须事前登记六槽，失败仍占槽，首次访问后不能改变上限。

模型报告保留训练地区、是否在被评地区训练、监督范围及配对比较范围。SharedGraph 和本地区强基线的混合范围可以描述，不能写成同一地区同监督条件。三种子合表也保留这一边界，不自动生成 SOTA 或 winner 结论。

2026-10-14 相关研究目录的汇总 CPU 回归为 227 passed，Ruff、编译和差异检查通过。JUnit 和摘要分别为 `control/ndr_related_validation_20261014.xml` 与 `control/ndr_related_validation_summary_20261014.json`。研究代码位于隔离分支 `codex/ndr-transfer-training`，截至该次检查为 clean commit `1470ed8becc5af6fe6d115cf20fd6726f607885e`。这个验收不替代新模型完整 CUDA 训练、真实三种子 test、跨地区胜出或全仓历史资产验收。
