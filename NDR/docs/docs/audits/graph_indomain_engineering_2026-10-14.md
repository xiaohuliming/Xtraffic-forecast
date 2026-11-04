# SharedGraphNDR 单地区训练入口工程审计

## 比较范围

新增入口让真实 SharedGraphNDR 在一个地区的原 event train 与 val 上训练和选优，可与相同地区、协议和监督预算的 NDR、BiST 及 STAEformer 比较。此入口属于单地区监督训练，不属于留一地区或严格零样本迁移。没有修改原训练入口、模型、损失、数据桥接或已部署队列。

新增文件为 `ndr_research/graph_indomain.py`、`scripts/train_ndr_graph_indomain.py` 和 `tests/ndr_research/test_graph_indomain_training.py`。

## 固定方法与数据

模型为原 `SharedGraphNDR`。隐藏宽度仅接受 64 和 240，历史与预测步数均为 12，道路层数为 2，全局共享 token 为 8，attention head 为 4，dropout 为 0。没有节点或地区身份参数，没有注册按节点排列的尺度状态。

数据沿用 `ResearchData` 和 `prepare_research_batch`。批次仅新增 `region.edge_index` 与训练段拟合的 `residual_std_by_node`，然后交给原 `graph_batch_tensors`。周模板和节点残差尺度的拟合范围为原始训练时间前缀，保留原始观测 mask。`data_identity` 原样保存，同时记录 `normalizer_identity` 与 `graph_identity`。

训练直接调用原 `optimize_graph_batch`。teacher 可选 original 或 projected，gate 和 specialization 权重固定为 1.0 与 0.3。使用 FP32、Adam 和按完成有效批次更新的 CosineAnnealingLR。CLI 默认 effective batch 为 16，microbatch 为 8，正式 epoch 预算仅接受 30 或 100，种子仅接受 42、1、2。

选择指标是该地区 validation 全部有效观测点累计的 global masked MAE，不使用各预测步长均值替代选择指标。完整验证保留尾批，不读取 test。CPU 训练必须显式声明 train 与 validation 批次上限，产物标记为 smoke_only。

## 保存与恢复

checkpoint schema 为 `graph_indomain_checkpoint_v1`，model_class 为 `SharedGraphNDR`。完整 checkpoint 保存 optimizer、scheduler、Python、NumPy、Torch RNG、两个 loader 的排列和游标，以及 epoch、阶段和已完成批次计数。恢复核对配置、源码、runtime、缓存、训练统计量、道路图和游标。

独立验证要求不同进程重新读取 checkpoint，复现相同 validation 样本、目标 hash、mask hash 和 global MAE。正式状态为 `verified_graph_indomain_checkpoint_complete_validation`，工程子集状态为 `verified_graph_indomain_smoke_validation_subset`。

calibrate 独立计时 train 和 val 的完整 effective batch，记录读取耗时与 CUDA 内存。校准使用真实模型和损失，不生成精度结论。

## 工程验证

9 项新增测试通过。测试覆盖原 event 输入桥接、训练统计量和道路图绑定、CPU 子集要求、完整 validation 的全有效点累计与尾批，以及训练阶段和 validation 前阶段的逐位一致恢复。schema、配置、道路图与游标偏移会被拒绝。独立进程要求与子集状态标签也经过检查。Ruff、Python 编译和 diff 格式检查通过。

两地区真实 v3 缓存 CPU 冒烟测试位于 `/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013/graph_indomain_cpu_smoke_final/`。

| 地区 | teacher | H | effective / micro | 每轮 train / val 批次 | 更新次数 | val 样本 | 新进程 MAE 差 |
|---|---|---|---|---|---|---|---|
| Alameda | original | 64 | 16 / 8 | 1 / 1 | 30 | 16 | 0 |
| ContraCosta | projected | 64 | 16 / 8 | 1 / 1 | 30 | 16 | 0 |

两个 run 均在 15 秒训练循环上限内完成子集的 30 轮。该 epoch 数仅表示每轮一个训练批次和一个验证批次，不表示覆盖完整训练集。子集 scheduler 周期随冻结批次数缩短，不能作为正式完整矩阵的性能证据或继续扩展为正式训练。未打开 Orange，没有读取 test，没有使用 GPU。

两份 best 和 last checkpoint 通过严格加载和有限值检查，25 个实际导入源码文件前后哈希一致。完整真实 validation 尚未执行，此处仅验证正式代码路径与有限工程子集。

local verification SHA-256 为 `232ab4289fc7ea00493cc46a47b8a64a7f1d0b25bcadb4d43ce9a94a5e0108af`。

任何精度、收敛或 SOTA 判断都必须等待正式完整训练、独立完整 validation、source-host postflight、逐文件复验以及冻结 test 比较。
