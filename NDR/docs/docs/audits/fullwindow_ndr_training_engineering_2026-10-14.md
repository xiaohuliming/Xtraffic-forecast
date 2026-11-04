# 全窗口 NDR 与 SharedGraphNDR 训练入口工程审计

## 独立协议和方法

新增 `ndr_research/fullwindow_training.py`、`scripts/train_ndr_fullwindow.py` 与 `tests/ndr_research/test_fullwindow_training.py`。入口使用已经独立核验的 FullWindowRegionReader、FullWindowStatistics 与 FullWindowBatchLoader，不构造 ResearchData，也不复用 event 的样本数或评测身份。原 event 代码、数据入口和正在运行的队列均未修改。

采样固定为 0.7 train、0.15 val、stride 1、历史 12 步与预测 12 步。正式各地区训练样本为 73561，验证样本为 15757。训练与验证的预测目标互不交叉，相邻 origin 在同一 split 内仍共享部分预测目标。统计量只使用训练段唯一时间戳的原始观测值。原始 flow 和 mask 仅读取 channel 0，道路图绑定真实节点顺序。

方法包含真实原 NDR 的 baseline、mask_signed、sanitized_control、projected_oracle，以及真实 SharedGraphNDR 的 H64 与 H240。原 NDR 保留原始 STAEformer 主干及节点参数，mask_signed 的 router width 固定为 16，没有新增容量搜索。Graph 的道路层数、共享 token、head 与 dropout 保持既定配方。

原始 clean 历史沿 `canonical_causal_imputation` 保留正式缓存的因果填补。需要 sanitization 的实际方法仍由自己的历史 mask 重建。原 NDR 使用其原始完整 effective batch loss normalizers 和模型注册的训练节点尺度。Graph 使用原 GraphLoss 与运行时训练节点残差尺度。teacher 和实际 variant 绑定，gate 与 specialization 权重固定为 1.0 和 0.3。

正式默认使用 FP32、种子 42、1、2、30 或 100 epoch、effective batch 16、microbatch 8、Adam 和逐有效更新的 CosineAnnealingLR，不提前停止。选择指标为整个 validation 的全部有效观测点累计 global MAE，保留尾批与完整观测 mask。

## 身份与恢复

checkpoint schema 为 `fullwindow_ndr_checkpoint_v1`。保存真实 model_class 与 model_factory、statistics_state、statistics_identity、reader_identity、sampling_identity、data_identity 和 source_snapshot。正式 claim_scope 为 `fullwindow_validation_development_only`。numeric test 没有开发入口。

完整 checkpoint 保留 optimizer、scheduler、Python、NumPy、Torch RNG、fullwindow loader 的 permutation、cursor、epoch，以及训练循环阶段、完成批次数和指标累计。恢复同时核对物料、采样、训练统计量、模型工厂、runtime、源码与已提交的游标和 epoch。

verify 在新进程重建真实模型，重新拟合相同训练统计量，核对完全相同的 validation origins、目标、mask 与 global MAE。正式状态为 `verified_fullwindow_ndr_checkpoint_complete_validation`，子集状态另标为 `verified_fullwindow_ndr_smoke_validation_subset`。

calibrate 使用真实完整 MAE、oracle gate 和 specialization objective，不以纯 MAE 或前向代替训练耗时。val 计时包括全局与分步长归约、目标和 mask 的 CPU hash 工作。校准记录仅用于工程预算，不生成精度结论。

## 已执行检查

13 项新增测试、Ruff 和 Python 编译通过。测试使用合成 raw-material bundle，覆盖五种方法的真实 objective 调用、runtime per-node sigma、独立 fullwindow 采样身份、完整 val 的全有效点计数和尾批，以及 NDR 训练阶段与 Graph validation 前阶段的逐位一致恢复。恢复检查使用原 fullwindow loader schema，不伪装为 event loader。

真实 v3 CPU 检查位于 `/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/fullwindow_cpu_smoke_final/`。

| 地区 | 方法 | effective / micro | 每轮 train / val 批次 | 已完成子集 epoch | 更新次数 | val 样本 | 新进程 MAE 差 |
|---|---|---|---|---|---|---|---|
| Alameda | 原 NDR baseline | 2 / 1 | 1 / 1 | 13 | 14 | 2 | 0 |
| ContraCosta | SharedGraphNDR H64 | 2 / 1 | 1 / 1 | 30 | 30 | 2 | 0 |

Alameda 在 15 秒循环上限到达后保存在 validation 前的可恢复边界。ContraCosta 在同一时间上限内完成子集的 30 轮。每轮只有一个训练和验证批次，不能表示正式训练已完成。两个地区还各执行一次真实 CPU calibration，使用 effective 16、micro 8、warmup 0、timed 1，完整三项损失均被执行。

28 个实际导入源码文件当前哈希与训练快照一致，两个地区的 best 和 last checkpoint 均严格加载且数值有限。所有检查都没有创建真实 Orange reader，没有读取 test 数值，没有使用 GPU。真实完整 validation 尚未执行。

本地复验记录 SHA-256 为 `e79d6661f2521040f5be9251f857275ea22634e7b3b35d4af66813c1b6c6ff3d`。此记录是训练入口与损失路径的工程证据，不能作为精度、泛化、收敛或 SOTA 结论。
