# SharedSTAEformer 迁移对照工程验收

日期：2026-10-14。

本次新增可训练的节点无关 STAEformer 项目适配，作为后续两源训练和留一区域评测的真实对照。尚未在 GPU 上完成正式训练，也未读取 Orange 目标数据或任何 test 标签。

## 固定结构

源码为 `ndr_research/shared_staeformer.py`，训练入口为 `scripts/train_shared_staeformer_transfer.py`。

原始实现来自仓库 `baselines/STAEformer/STAEformer.py`。输入和输出均固定 12 步，flow-only 输入维度为 1。保留原始 24 维 flow embedding、24 维 TOD embedding、24 维 DOW embedding、256 维 FFN、4 个 attention head、3 层时序 attention、3 层空间 attention、0.1 dropout 和 mixed projection。

固定保留 80 维 adaptive embedding，将参数形状改为 `12 × 1 × 80`，沿节点维共享。主干维度保持 152，参数总数为 1,061,172。输出 reshape 使用输入的实际节点数，不修改共享对象的 `num_nodes`。模型没有节点或区域身份参数，只有 `flow_center` 和 `flow_scale` 两个标量 buffer。

空间 attention 仍为原始 dense attention，计算和 attention score 空间成本随节点数平方增长。该模型是项目的共享时间适配，不能写成原论文的跨区域零样本结果。

## 输入和训练边界

公共 forward 与 SharedGraphNDR 使用同一七字段接口。预测只读取 `flow_hist`、`x_baseline`、`history_mask` 和 `time_feat`。`y_baseline`、`edge_index` 和 `node_scale` 不参与预测，也不读取未来 flow、目标标签或 incident 标签。

缺失历史通过共同的 `canonical_history` 处理，先用窗口内最后一个已观测值做因果填充。窗口起始缺失使用 source-only weekly climatology fallback。观测到的零流量保留为有效值，缺失位置的 NaN 等 sentinel 不参与算术。

输入 scalar center 为两个源区域冻结 pooled weekly template 的周平均，scalar scale 为同一 source-only preprocessing 的 pooled residual scale。该约定属于 source-template scalar normalization，不能称为原模型 raw-flow z-score，也没有新增或拟合目标区域统计量。

训练只使用 raw-flow masked MAE。每个 microbatch 的误差和除以完整 effective batch 的有效点数。无有效目标的 batch 不更新 optimizer 或 scheduler。没有 NDR 双头、teacher、specialization loss 或辅助缓存。

有效 batch 顺序和区域 objective multiplier 复用已验证的两源训练规则，各区域的 epoch objective 总贡献相同。正式 checkpoint 由两源完整 validation 的区域 global masked MAE 等权平均选择。只有显式 engineering smoke 可缩小 validation 集合，checkpoint 和独立验证保留同一集合身份。

固定训练配方为 Adam，learning rate 0.001，weight decay 0.0001，gradient clip 5，CosineAnnealingLR，eta min 0.00001。完整 effective update 后步进 scheduler。FP32、确定性运行，默认 effective batch 为 16，microbatch 为 2，无 early stopping。正式预算为 30 epoch 或明确标记的 100 epoch 补充，CLI 不暴露容量搜索选项。

checkpoint 使用真实 `SharedSTAEformer` class，记录源码、运行环境、缓存、source statistics、模型配置和选集身份，保存 optimizer、scheduler、Python RNG、NumPy RNG、Torch RNG、区域顺序游标、train 和 validation loader 状态。独立 source verification 重新加载选中 checkpoint 并复算相同观察点。

## 工程证据

合成测试覆盖原 native 模块绑定共享 adaptive 权重后的精确输出等价，同一 state 跨不同节点数 strict load，节点排列等变，calendar 恢复，缺失 sentinel 不变性，零流量保留，ignored future-baseline 和图字段不变性，effective masked denominator，区域等权完整 source validation，以及中途恢复模型、optimizer、scheduler、RNG 和 loader 的逐值一致。

新增 8 项测试通过。连同 SharedGraph、transfer data 和 transfer training 相关回归，共 59 项测试通过。新增 Python 文件通过 Ruff 和 compile 检查。

真实 v3 缓存工程探针仅构造 Alameda 和 ContraCosta 两个 source reader，heldout 记录为 Orange。CPU smoke 使用 effective batch 2、microbatch 1、每区域每 epoch 1 个 train batch 和 1 个 validation batch，wallclock 预算 35 秒。运行停在 18 个完整 smoke epoch 后，下一个 epoch 已完成 1 个 effective update，状态为 `stopped_resumable`。这些不是完整 source train epoch。

产物目录为 `outputs/remote_runs/shared_staeformer_cpu_smoke_20261014/`。独立新进程验证状态为 `verified_smoke_source_validation_subset`，重算差值为 0。每源 validation 只含 2 个样本，不是全 validation，也不是性能或泛化证据。

选中 checkpoint SHA-256 为 `5f3e55e85af3773a554c8306cc4bac916364cccd371e61cc14fb63bb7edd92ab`。

源码 snapshot digest 为 `84aa052046ad6769cbf9b5c4572a998f1ca71bf2b618371e000bddff987a7ea9`。

full-effective-batch CPU calibration 位于 `outputs/remote_runs/shared_staeformer_cpu_calibration_20261014/`。它使用 effective batch 16、microbatch 2，每源各计时 1 个 train batch 和 1 个 validation batch，无 warmup，包含 reader 成本。Alameda train 和 validation 分别为 5.427 秒和 2.300 秒，ContraCosta 分别为 4.984 秒和 2.095 秒。单批冷运行数据仅供工程参考，不能用于 RTX 5080 ETA 或精确速度结论。

当前工程结果不改变已有正式结果，不支持超过 NDR、现有 SOTA 或跨区域泛化更优的声明。与 SharedGraphNDR 的后续比较属于整体方法比较，其主干、空间传播和训练 loss 不同，不能解释为 gate 的独立贡献。
