# IGSTGNN Table 4 原生配方 100 epoch 补充对比

## 目标与边界

2026-08-01 启动独立矩阵 `igstgnn_table4_bist_ndr_native100_18_v1`，补跑 BiST 与 NDR-Net 在三个地区、种子 42、1、2 下的 18 项结果。它的目标是生成能够直接填入既有 Table 4 结构的 100 epoch 补充行，不是建立统一训练器或改写两个方法的原生训练契约。

本矩阵继续使用 `igstgnn_table4_official_v1` 的公开数据、顺序切分和评测口径。它继承已知 forecast origin 重合，有效性标签仍为 `official_protocol_with_known_overlap`，不得与 `protocol_v3_noleak_dedup` 结果混合。NDR-Net 仍是项目方法扩展，不得写成 IGSTGNN 原论文方法行。

## 与原表一致的训练口径

- BiST 复用锁定的上游 BiST factory。保留 batch 64、`partial_batch: drop`、Adam 学习率 0.002、weight decay 0.00001、MultiStepLR 与 patience 20，只将最大 epoch 从 500 改为 100。
- NDR-Net 复用已验证的 STAEformer NDR 扩展管线。保留 batch 16、`partial_batch: retain`、Adam 学习率 0.001、weight decay 0.0001、CosineAnnealingLR、全局验证 masked MAE 与最优 checkpoint 选择，只将最大 epoch 从 30 改为 100。NDR-Net 原生管线没有 early stopping，因此不人为加入。
- 评测列保持 H3、H6、H12 与 12 步 Average，每组都报告 MAE、RMSE 与 MAPE。
- 两个方法使用同一官方 split 和同一测试观测集，但保留各自原生 batch、optimizer、scheduler 与 early stopping 设计。因此本试验回答的是两个完整方法在同一数据与 100 epoch 上限下的表格对比，不是等优化器消融。

## 锁定身份

- 实现提交：`53f4b3a`
- 矩阵清单：`config/reproduction/igstgnn_table4/matrix.native100_compare_v1.json`
- 矩阵 SHA-256：`e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246`
- 配方说明：`config/reproduction/igstgnn_table4/native100_compare_profile_v1.json`
- 配方 SHA-256：`37ebcb574f516bc1b662ade0d5264deda96a20d017d20edc7a57253daab5e66b`
- 原 39 项清单 SHA-256：`d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`
- 原 NDR 配方 SHA-256：`6ca46be0fb76543a61a8bd32996c7598ac674bbe239b701d1a92481f63be9422`

## 验证

- Mac 全仓回归：487 passed，2 个既有 warning。
- 5080 Windows 专项门：25 passed。
- Alameda CPU 单批次探针：BiST 与 NDR-Net 均通过。
- Orange 5080 真实 CUDA 完整优化步探针：BiST loss 为 `148.53871155`，峰值分配显存为 `3064998912` bytes。NDR-Net loss 为 `42.43656731`，峰值分配显存为 `12633100800` bytes，使用两次 microbatch 8 保持有效 batch 16。
- 5080 生产文件与 Mac 逐文件 SHA-256 一致。

## 5080 启动状态

- 独立 checkout：`D:\t4\native100_repo`
- 独立结果根：`D:\t4\native100_compare\results`
- Windows 计划任务：`XTraffic_Table4_Native100_v1`
- 执行时限：`PT0S`
- 启动时间：2026-08-01 00:02:23 新加坡时间
- 启动后状态：0/18 完成，0 失败，正在执行 `n100001_bist_alameda_seed42`
- 启动后 GPU 状态：显存使用 3965 MiB，利用率 70%，温度 53 摄氏度

完整 18 项落盘前，不报告 BiST 与 NDR-Net 的 100 epoch 性能结论，也不覆盖已有 BiST 500 epoch 单次结果或 NDR-Net 30 epoch 三种子结果。
