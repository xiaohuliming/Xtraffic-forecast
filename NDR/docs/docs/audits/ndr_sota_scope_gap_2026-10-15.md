# NDR 当前 SOTA 比较范围与近期强对照缺口

日期：2026-10-15，Asia/Singapore。

当前目标仍是改进 NDR 的性能与泛化。原八项 event 矩阵、节点无关模型、多种子、跨地区和独立 full-window 比较的范围保留。原队列没有修改。

已有真实强对照为原 NDR、原 STAEformer 和锁定作者 BiST。SharedSTAEformer 是节点无关的项目改造，用于学习型迁移比较。旧 event leader 为 NDR，旧 full-window leader 为 BiST，两份旧表不能混用，不能把新百轮方法与旧三十轮方法直接称为公平 SOTA 比较。

现有 execution_gap_review.json 要求至少一个近期公开强方法。其 DualCast-S 没有完整 trainer 的旧描述已被后续实现替代，但同协议 CUDA 校准与完整训练仍未完成。next_stage_budget_readiness_20261014.json 将两项成本明确延期，当前九种成本配置也不含 DualCast。完成现有九种配置不能自动关闭该比较缺口。

DualCast-S 的完整 v3 训练入口已存在。它使用完整 batch 32 与作者锁定模型和原有定义辅助式，兼容配方明确记录未定义辅助项省略。因 masked 主损失、split、统计和适配边界，不称 exact native，也不充当节点无关迁移基线。完整成本测量入口与专用正式评测接入已经在独立分支完成，训练数学和停止线保持。当前没有执行新 CUDA 任务。

作者公开 bundle 为 249,523 bytes，SHA-256 为 `f5d46e7a3de1011936de468bd8beaa0b1ed68fbd7ce352a7dcbdce58703bc9c3`。已在 5080 独立根 `C:/XTrafficResearch/ndr_dualcast_20261015/prepared_50425c1_2850d42e63` 准备 pristine 上游，保持 commit `50425c12f2fc4e6ba6069d028540d2373e5e945a`、tree `af642ea5ac1c95874e7a735ecb733699b64b6af8`、官方 origin、四文件哈希与 clean。独立回读通过。仅完成标准库、Git 与文件操作，没有读数据、构造模型、写执行计划或注册任务。

OpenCity 仅可作为另列的长上下文预训练对照，目前其预训练年份与站点重叠尚未完整核验。FlashST、STGformer、GraphSparseNet 和 HyperD 目前只有已存论文与官方代码链接，不能写成已比较方法。当前 capped 30/100 预算也不能代替 BiST native 500 轮充分收敛证据。若最终需要广义 SOTA 主张，仍须在论文比较范围冻结后核验发表截止日前适用公开方法。

源端准备回执见 `control/dualcast_upstream_source_preparation_2850d42e6367408190daf63e9fdbbc80.json`。回读与范围记录为 `control/dualcast_upstream_independent_readback_20261015.json` 和 `control/ndr_sota_scope_gap_review_20261015.json`。


## 05:47 工程收口与成本源码准备

成本入口提交为 `6e9e0697a70408eaeb0cfa8f88cfb055ae0b86fd`，最终 trainer SHA-256 为 `aaf51d54b1cfc4373295cb47d5c6465a72c5283648b11c1d0795c201f2f14e0b`。原三十六项成本及适配检查通过，随后针对新增 CPU verifier budget 字段的受影响检查再次通过。两份回执分别保留，没有把它们计为三十七个不同检查。

正式 DualCast 的地区内冻结评测接入已提交为 59db4f1ba148d00594ca972dac58816efc3953ed。它使用独立 kind、正式 schema 和兼容 scope，保留作者、道路、normalizer、独立 PID 与完整 validation 身份，拒绝成本或 smoke artifact。优化器、StepLR、RNG 与完整 loader 状态及真实 train、val indices 均有专用检查。真实作者模型的可训练 D_S 会与构造 adjacency 共享内存，重建时只克隆构造输入，避免改写冻结 checkpoint，训练公式没有改变。最终九十五项局部合成回归、Ruff、compile、diff 与独立静态复核通过。旧六十三项与中间回执保留。

已从成本提交的原始 Git blobs 生成三十五文件包，排除同期 evaluator 未提交字节，ZIP 为 122,713 bytes，SHA-256 为 `c0715b4da08006e0207cd7dedac4a4460084d498a29dadd5a1d15fbbe4a07f64`。独立解压中的真实模块导入和四份 parser、validator 通过。成本源码已在前述独立 5080 根的 repo 准备，实际最小 Git 为 `993c434dd27ebcabe14d03b6f08bcf901efec3d0`，clean。源端三十五文件与四文件作者身份又独立回读通过。来源提交、实际最小仓库提交和未部署的 evaluator 提交不能混淆。

首次标准库准备脚本存在内层换行转义导致的 Python SyntaxError，在脚本体执行前停止。只读确认没有 repo 或完成回执，原 ZIP 保留。修正版先在本地 compile，再上传并按哈希执行。原失败脚本与 incident 没有覆盖或伪装为完成。

独立四步成本草案为 `control/dualcast_separate_complete_cost_jobs_draft_20261015.json`，每地区 calibrate 后串行独立 verify。实际预算、deadline 和 timeout 仍为 null，专用执行控制尚未冻结。原九种成本配置、原八项及六项未修改。正式 auxiliary audit 的长期 I/O 与实际 CUDA 成本仍未测，不能使用合成 CPU 回执代替训练预算。工程总回执为 `control/dualcast_engineering_closeout_20261015.json`，没有真实 test grant、正式 DualCast 训练或 SOTA 性能结果。


## 两源缓存身份的只读复核

独立代理使用当前 A/C 缓存 metadata 核验物理 station、origin、全部十二步 target 时间和原始 incident ID。站点交集、跨 train/val 以及合并两源后的 target 时间与原始事件交集均为零。本地事件行号有数字碰撞，映射真实身份后不构成交叉，不能仅比较 region-local index 或带命名空间的 agg_id。

回执 `control/source_pool_cross_region_metadata_audit_20261015.json` 为 20,719 bytes，SHA-256 为 `b5bb8aceed0ce0620569184e538616e3d154f0a3b78f6dd1b6a0982a96ff9055`。metadata、config、node_order 和事件 CSV 的实读身份与既有完整 HDF5 哈希声明明确区分，未重复全 HDF5 hash。根代理仅复核回执字节与结果字段，没有宣称再次独立重算。没有读取 flow、traffic mask、target、severity、duration 或 Orange 数值，没有推理或添加新门禁。该检查支持当前 A+C 源数据池的身份边界，不是泛化精度证据。


## 独立四步成本控制准备完成

控制提交为 `0a9f7b562955cfdefc68c3f2f7361617d252c540`，44项合成检查及最终字节独立静态复核通过。runner SHA-256 为 `59662b971e87784d7c39c660485501e2f049ee8d7da875350fcae86afd7e0ebf`。真实 Mac completed-wrapper 与 nested admission、全工程预算在 idle 和最终 inventory/postflight 的检查、Windows 文件占用时先保存 failed 状态均已覆盖。旧26、43项和中间回执保留。

实际 DualCast verifier 不会从 checkpoint 自动恢复数据目录。最初四步草案的两条 verify 缺少 data-dir、graph-dir，原草案已保留，另存显式同 cost 路径的 paths-ready 草案，不能让正式计划在运行时静默 fallback。旧十八步的九条 verify 已逐条确认原本含明确路径，无需修改。

两文件独立控制包为21,974bytes，SHA-256为 `55b6f9087dfe7748d635eb5e2d4c0fe8dd2beb207669beec38408bd040d57ea7`，已准备到5080独立根 `C:/XTrafficResearch/ndr_dualcast_20261015/control_0a9f7b5_a17b7fbb4e`。隔离help、源端35/4字节准备前后不变与控制独立回读通过。raw source receipt哈希和Mac JSON文件哈希分别绑定，不混同换行或排版差异。

当前只有准备状态。正式engineering budget、deadline、timeout仍未冻结，没有执行计划或计划任务，没有新CUDA工作。原八项现已完整No-Go，原六项工程校准正在执行，仍须完整source和Mac准入以及真实前驱句柄退出后才可派发成本测量。总回执为 control/dualcast_control_closeout_20261015.json。


## 2026-10-15 13:55 真实显存失败与等价执行准备

DualCast原整批32在N521发生边展开CUDA OOM，0/4，原5文件28247bytes已完整保全。新逐fragment checkpoint接入58fc8a6在真实作者小CPU链与101项去重检查通过，36文件源码包及源端ba5daa2准备回读通过，GPU尚未验证。详见control/dualcast_memory_source36_closeout_20261015.json与dualcast_memory_v2/production_integration_receipt.json。CPU内存分配和数值等价不能代替5080峰值、deterministic或正式训练性能。

Graph完整7配置仍运行，13:55:55实测H64完成10/30轮、30270updates，Source42808/41808与Mac保全32320均存活。后继STA/BiST四项仅准备好命令与成本，未派发，原native30/100仍不足以声称作者500cap与patience20收敛复现。Orange、最终test、正式多种子与SOTA比较继续未完成。


## 14:36 后继工具准备边界

当前formal7实测为pooled Graph H64完整20/30轮，进程与Mac保全存活，不能读取部分排名。新controller3264e4e、分析器4a69cd9与BiST native入口bcd4982合计115项相关检查通过。DualCast新控制2文件源端准备完成，生产数值probe v3仅CPU四例2612张量字节相同，源端只复制文件；BiST33文件包只完成本地隔离导入。所有新GPU可行性、强对照精度与充分收敛比较仍缺。回执为outputs/remote_runs/ndr_generalization_20261013/control/development_preparation_closeout_20261015/closeout.json。

七项完整保全后先准入独立DualCast数值和成本，再串行补原STA/BiST固定30四项。native500/patience20是独立补充，500轮上限不证明充分收敛。strong4仍需新runner与Mac取回适配器，已有成本和source43 trainer可复用，不重复工程测量。Orange和最终test继续保持未开放。原NaN替代接口反例已更正，真实共同前向原已拒绝全量NaN，不用于夸大生产缺陷或结果改善。


## 15:29 近期公开对照更新与首项保全

H64首项完整训练与独立验证已保全14文件，仍仅1/7。H240正在原冻结队列训练。新增strong4控制06dc932、numeric90072583、成本审计cf1e与终态d118已完成190项相关检查和源码准备，均没有新真实执行计划或CUDA结果。先完整七项、再numeric与DualCast成本、最后strong4，保持串行。

有限检索新增[MAGE官方论文](https://papers.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html)与[代码](https://github.com/PoorOtterBob/MAGE)，作为未来同协议地区内近期强对照候选。其固定节点参数不能充当whole-model新节点迁移基线，全网络6:2:2数字也不能直接与事件v3比较。官方main配方与论文学习率、balance更新存在须声明的材料差异，不能静默切换engine或改代码后声称原论文复现。[HyperD](https://github.com/ll121202/HyperD)需要严格train-only周期统计。[GenCast](https://github.com/suzy0223/Generalising-Traffic-Forecasting-to-Unobserved-Regions-GenCast)另用天气与无传感器任务，应单列输入条件。三者仅锁定源码、静态核验，没有适配或性能结果，不能把近期方法范围视为已经关闭。原始检索与15文件证据位于control/recent_reference_refresh_20261015。


## 17:23 MAGE 同协议入口与源端准备

MAGE四文件已窄提交fdb6483c45b5a99808f387d8dc86cf1c8aa35dd4，锁定作者clean f1fdd27/tree cb06f17。独立训练配方public_main_v3_compat包含fixed30、fixed100和public300_p30，明确exact_native_claim为false。保留真实FFN dropout0.3、冻结零routing bias与作者模型公式，输入仅历史流量和历史TOD、DOW/7，train-only标准化及逆变换各一次。TOD96是embedding bank，不能证明15分钟数据；v3仍为5分钟、历史12步和预测12步。全网络6:2:2论文分数继续不进入事件v3比较。

最终32项CPU JUnit全绿；9项adapter作者对照仅沿用原执行者报告，不合并计算独立新测试数。独立审查与真实CPU部分validation中断恢复通过，恢复状态逐值完全一致，没有原始张量字节对比。35文件ZIP138137bytes与pristine bundle1601127bytes完成隔离help、import和源端独立准备回读。源端实际Git0d3530e25d34085c01b7c779344ff0724a9d9847为clean，原source43和source36不变。总证据为control/mage_implementation_source_closeout_20261015.json，SHA e47903de488c80bc59ebade24b761eb5b67369820feafdde4f9f3301dfc70de0。

原formal7于17:23:52 Singapore实测H240为29/30轮，源supervisor42808与worker38804存活，完整7仍未保全或排名。Mac observer65178已按原独占锁恢复，只恢复观察器，没有重启训练。后继仍先完整七项、再DualCast数值与成本、最后strong4；MAGE尚无实际CUDA成本、执行计划、训练、精度或SOTA结果，需要独立预算和串行准入。公开近期方法比较范围仍未关闭，Orange和最终test未开放。


19:23:44 Singapore更新：原formal7为3/7完整保全，44文件49,503,504bytes独立复核通过，Alameda H240为10/30轮，source42808/46708与Mac observer65178实活。MAGE无新增GPU或配方变更。文档步骤误输出已有selection_mae元数据，未比较或用于模型选择，不能再称全团队完全盲态；见control/formal7_incidental_metadata_exposure_20261015.json。完整七项、强对照、多种子及泛化证据仍待完成。


## 2026-10-15 19:28 MAGE 论文与公开实现差异

有界原始来源审计核对官方论文第5页Eq.14、第6页Eq.17与公开selector、engine。论文将beta加入被选专家系数，并规定active sign-SGD与初始学习率0.02；公开main将beta只用于top-k成员选择，权重仍来自未加beta的sigmoid归一化，bias更新为注释，学习率0.01。现有public_main_v3_compat保留其公开代码身份，不能单独支持超越完整论文MAGE或SOTA的结论。

论文Eq.9的非负归一化约束与Eq.14的beta后系数如何衔接，以及批计数、尾批和更新时点仍需显式解释。后续paper_equation_v3_reconstruction作为尚缺的独立强对照，不能仅取消注释就宣称论文复现，也不静默改当前兼容入口。原formal7与既定串行后继不变。证据为control/mage_public_vs_paper_recipe_gap_audit_20261015.json，SHA635d00c3bfaeaed5dd397c585d78467bcc20817b8b56c253d09cbb99f5b7a790。


## 19:35 BiST 收敛对照源码准备

既有native500/patience20源33文件与独立作者18文件已在5080完成源码准备和Root独立回读，实际clean Git1000a030。仅文件、Git、help和parser，没有新CUDA或训练；原source43、source36和作者未改。回执control/bist_native_source_host_independent_readback_20261015.json。500轮上限及源码就绪不能作为已收敛或SOTA结果，后继仍须单独预算和串行准入。


## 20:02 主机重启与最终评估适配

原formal7因19:54:32.500主机非正常重启停止，当前3/7保全与第21轮完整断点通过CPU结构审计，尚无恢复GPU。旧status running无效，须新建boot-aware恢复与composite，不能把中断当原14Popen全成功。恢复前置与证据见ndr_formal7_strict_recovery_2026-10-15.md。

9064395最终地区内评估适配的122项CPU检查、8项独立门禁反例和source51源端回读已通过，没有真实heldout或CUDA正向准入。它补齐新checkpoint格式的工程能力，不补齐真实收敛、多种子、泛化或MAGE论文公式差异。原数值、成本和strong4准备版本未来须显式准入恢复composite，不能直接使用中断前驱。
