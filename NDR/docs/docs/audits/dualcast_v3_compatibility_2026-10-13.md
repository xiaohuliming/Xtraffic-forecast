# DualCast-S 独立 v3 训练入口与兼容边界

日期：2026-10-13，Asia/Singapore。

当前状态为 CPU 工程验证完成，没有运行 CUDA，没有读取 test，也没有形成性能比较结论。新入口为 `scripts/train_dualcast_v3.py`，训练模块为 `ndr_research/dualcast_training.py`。第三方源码继续位于被忽略的 references 目录，没有并入 Git。

## 作者源码与原项

锁定作者 commit 为 `50425c12f2fc4e6ba6069d028540d2373e5e945a`。相关文件均通过完整 SHA-256 绑定。

| 文件 | SHA-256 |
|---|---|
| `DualCastS/model/model_.py` | `47462de01c827296a8aed4b6e628bd7dd041fee64a209f026c7452cc3c65d336` |
| `DualCastS/model/train.py` | `f288756a394eeb9774ffb1f1fa3f05b8106b82e7eb70b2a741eef9a0c25798a4` |
| `DualCastS/main.py` | `89fedc9c081457e2a277d0f9702ed38759d15490cfa944044ce5b01ee3e21d9e` |
| `DualCastS/utils/utils_.py` | `8ebe772f78ea3c06204b74768eab06fb9df62fdc19ffaa2a71b20bd56529d2f1` |

作者 `train.py:51` 使用 intrinsic 与 environment 的 batchmean KL 倒数，权重为 1。第 55 至 56 行使用环境特征的随机样本排列与原特征的 KL 倒数，权重为 0.01。负 logits 本身可以正常参与 softmax，问题是零 KL、数值舍入产生的非正 KL，以及倒数产生的非有限值。

作者 `train.py:59` 至第 99 行的 DBI 使用群内 Frobenius norm 除以成员数。第 93 行的 `torch.tensor(r)` 脱离计算图，第 96 行再按有效群数归约。新入口直接从锁定作者文件提取三个辅助损失 If AST，在有定义的批次上执行原表达式，没有把 DBI 改成可微实现。原来的梯度脱离 warning 可以出现在工程日志中。

作者模型使用 BatchNorm2d。批内 KL 排列、prototype 群组成和 BatchNorm 都依赖完整批次，因此本入口不提供微批拆分选项。作者默认批次为 32，缩小真实批次会显式标注覆盖。

## 两种明确配方

| 配方 | 未定义辅助项处理 |
|---|---|
| `v3_masked_native_equations` | 失败退出，不把无效表达式当成训练结果 |
| `v3_masked_defined_aux` | 只省略该批未定义的项，逐批记录项名与原因 |

相同环境分布的 KL 数学值为零。保留的一条样本尾批会导致环境自比较。只有一个有效 prototype 群时，作者 DBI 的有效群数为零。有效 anchors 重合也会使群间距离为零。这些情况都明确记录，不加 KL epsilon，不改成另一种距离或对比损失，不恢复作者 DBI 的梯度。

两种配方均不能称 exact native。共同差异包括 v3 的显式观测 masked 主损失、唯一 raw train timestamps 统计、无泄露 event split、真实星期映射与仅周末标记、Torch degree/scatter 适配，以及保留全局特征 batch 轴。作者尾批 `squeeze()` 原来会删除 batch 轴，修复后的尾批辅助式因此也不能声称与原错误行为完全相同。

默认模型仍为 embed dimension 64、1 层、8 heads、17 prototypes。保留作者 Adam 学习率 0.001、weight decay 0、每 10 epoch 乘 0.9 的 StepLR。30 或 100 epoch 固定预算另行记录，不用验证集分数改停止线。验证选择使用全部有效观测点累计的 raw-flow MAE，验证尾批不删除。

## 工程证据

`outputs/remote_runs/dualcast_v3_cpu_smoke_20261013/` 保存正式 v3 Alameda 的有界 CPU 训练与新进程复验。该次使用默认模型维度，真实有效批次为 2，只执行一次更新，验证集合为 2 条样本、全部 12 步和 10,301 个有效观测点。

训练进程为 13684，复验进程为 16338。复验状态为 `verified_smoke_source_collection`，checkpoint SHA-256 为 `c86ef2103021a0451369f4b518c28e7378f605ba184062a5e80afe2cd6dc2bcf`。源码集合、作者文件、完整模型状态、optimizer、scheduler、Python 与 NumPy 与 Torch RNG、loader permutation 与 cursor、source data 与配方身份均落盘。

该次环境随机排列为恒等排列，环境倒数项按明确兼容配方省略一次，记录在 `auxiliary_audit.jsonl`。有效的 filter KL 倒数和原 detached DBI 保留。预测、更新梯度和新进程复算均通过工程检查。

这不是 Alameda 全量 validation 性能报告，也不是作者 PeMS 分数复现。正式开发模式禁止 CPU 与样本子集混入，独立 verification 会要求完整 source validation。CUDA 准入仍需在空闲 5080 上运行完整目标函数的暖批与计时批校准，当前工程结果不能用作 GPU 预算或 SOTA 证据。

`outputs/remote_runs/dualcast_v3_cpu_calibration_20261013/` 另外通过了完整目标函数的 CPU 校准入口检查，每阶段暖批 1、计时批 1，使用完整真实批次 2。其状态为 `calibrated_cpu_engineering_only`，`cuda_budget_eligible` 为 false，不按预测误差选择配置。训练阶段同样记录了环境自比较项省略，DBI 的作者实现和有效 filter 倒数项仍参与代价测量。

本地相关训练与适配器测试为 21 passed，Ruff、Python 编译与差异检查通过。单独数值测试验证了有定义项与作者表达式一致、DBI 的 Frobenius 归约和 detached 梯度、尾批未定义项的记录、一次随机排列的 RNG 消耗，以及完整 optimizer、scheduler、loader cursor 与 RNG 恢复后的逐 Tensor 参数一致性。
