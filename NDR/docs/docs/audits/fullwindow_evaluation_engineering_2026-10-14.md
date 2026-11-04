# 全窗口独立冻结评测工程验收

日期：2026-10-14。

本次新增全窗口同地区时间外测试入口，未读取真实 test 或 Orange 监督数据，也未使用 GPU。所有正向评测测试均使用临时目录内的合成 H5 和合成 checkpoint 验收产物，不构成训练精度结果。

新增模块为 `ndr_research/fullwindow_evaluation.py`，CLI 为 `scripts/evaluate_ndr_fullwindow.py`。没有修改原全窗口开发 reader、训练器、旧自动 test 路径或远程队列。

## 协议与角色

采样协议显式为 `full_window`。原始材料权限来自已验证 v3 的 raw traffic 和静态 graph，样本使用独立全窗口 origin 和 split 身份，不打开 event samples、relation、incident 或 affected 数据。

正式原始材料对应 train、val、test 样本数为 73561、15757、15757，history 和 horizon 均为 12，stride 为 1，按原始时间轴 .7/.15 分段。相邻 origin 在同一 split 内共享目标，三个 split 间 target-disjoint。这项性质在报告中保留，不写成 event 去重采样。

评测角色为 `same_region_temporal_heldout`。它使用同一已训练地区的未来 test 时间窗，不是跨地区 source-only 推理。reader 和 sampling 身份必须与 checkpoint 保存的训练 reader 精确一致，测试统计量只来自 checkpoint，不在 test 阶段重新拟合。

## 真实 checkpoint 准入

NDR 与 Graph 使用 `fullwindow_ndr_checkpoint_v1`，正式验证状态为 `verified_fullwindow_ndr_checkpoint_complete_validation`。StrongReference 使用 `fullwindow_strong_checkpoint_v1`，正式验证状态为 `verified_fullwindow_strong_complete_validation_capped`。实际 wrapper 与 native class 均通过真实 factory 构造和 strict state load 检查。

gate 绑定 checkpoint、summary、独立 verification 的字节哈希，以及真实 class、factory、运行环境、源码、reader、sampling、statistics、data identity 和完整 validation 选集。summary 必须完成固定 30 或 100 epoch 的预算，验证 PID 必须与训练 PID 不同。验证需复现 origin、目标、mask、有效点和 collection，以及全局 MAE。smoke、subset validation、未完成训练和 event checkpoint 不可准入。

保存的 FullWindowStatistics 数组与数值 identity 逐项绑定，拟合范围严格为 raw train 前缀，scalar reduction 为 float64。NDR 注册的 scalar 和 node scale、StrongReference 注册的 flow mean/std 都与该保存统计量核对。三个种子必须共享结构、源码、训练配方、运行环境与统计语义。

旧 verified27 BiST scalar fitter 使用 float32，本轮独立 fitter 使用 float64。这一差异在每个模型与顶层报告中显式保留。完整训练预算为 capped development，不能宣称原 native 收敛已复现，也不能直接把旧 30 epoch 表与新 100 epoch 分数合并为同质 SOTA。

## 独立冻结 reader

FullWindowFrozenGrant 绑定 authorization、gate、checkpoint cohort、原 source reader、sampling 和 saved statistics 的精确 SHA-256。

FullWindowFrozenReader 使用原开发类的 target metadata 模式核对原材料和独立采样。为了比较原训练 metadata，计算一个 role 为 source 的 metadata 副本 digest，原对象仍保持 target，专用评测对象保持 `same_region_temporal_heldout`。这一转换不开放原开发类的数值 API。

专用 reader 仅提供 `get_test_batch`，每次要求相同冻结 grant、唯一合法 test index 和精确 24 步窗口。H5 数值读取仅包含 `flow_series_imputed` 与 `flow_mask` 的 channel 0。没有 source-fit、train/val preload 或重新拟合接口。grant 与 statistics 没有公开 setter。

创建 gate 不构造测试 reader。evaluate 先完成完整 gate 和源码检查，构造真实模型，创建共享 ledger reservation，再创建专用 test reader。失败也消耗预算，未完成检查的调用不会留下成功报告。

## 配对与缺失处理

每个 test 窗口只读取一次，供所有冻结真实模型、五个场景复用相同目标、mask 和 origin。clean 场景直接保留原缓存的因果填补值，不经会清零未观测位置的 perturb_history。新增缺失只保留原始 observation mask 中仍可见的值。

baseline NDR、projected-oracle NDR、原 STA 和 BiST 的腐蚀历史由共同的 canonical_history 重建，禁止复用被移除观测或旧缓存填补值。Graph、MaskSignedNDR 和 SanitizedNDR 使用各自原有观测 sanitization。所有 forward 均通过同一七字段 target-free adapter，不传 target、target mask、incident 或 affected。

指标使用 float64 累加全体有效观测点，报告 MAE、RMSE、MAPE、12 个 horizon 和真实 times_ns 对应的日块总量。观测零值参与 MAE 和 RMSE，仅从接近零分母的 MAPE 中排除。每模型和每场景保留同一完整 collection 哈希，以及全窗口 legacy origin 哈希。

推理前后检查真实模型 raw state、constructor、eval mode、checkpoint、gate、verification、summary、训练与评测源码、保存的统计量、原材料 fingerprint 和静态道路。运行期间的模型 aux 和 scenario 只是推理缓存，不作为重新训练或改动结构的入口。

## 家族预算与报告

至少两个真实方法家族，每家族需 seed 42、1、2。共享账本每次 cohort 访问消耗 3 个 distinct training seed inference slots，三种子与一次独立重复共预声明 6 个 slots。family 排除 checkpoint hash、seed、model ID、gate 路径和 authorization，首次访问锁定预算，失败照计。

gate schema 为 `fullwindow_frozen_eval_v1`，成功报告状态为 `frozen_fullwindow_evaluation_complete`。顶层 `mode` 和 `evaluation_role` 均为 `same_region_temporal_heldout`，`sampling_protocol` 为 `full_window`。报告保留 family identity、reader、sampling、saved statistics、float64 与 legacy float32 差异，以及每个模型的真实 class、native class、训练范围、配方、运行环境和来源。

## 工程结果

新增 7 项纯合成测试通过。联合全窗口 data、NDR training、strong training、source cohort 与 transfer evaluation 的相关回归共 92 项通过。新增源码通过 Ruff、compile 和 CLI help 检查。

两个训练器作者分别只读核对 NDR/Graph 与 StrongReference 的 schema、factory、normalizer、完整 validation、PID、summary 与来源字段，接口均相容，没有修改评测实现或读取真实测试材料。

本次结果只验证工程边界。尚无真实完整 test 分数、训练收益、跨地区收益或超过 SOTA 的证据。
