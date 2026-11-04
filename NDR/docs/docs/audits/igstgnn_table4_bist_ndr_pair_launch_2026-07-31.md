# BiST 与 BiST-NDR 严格配对实验启动记录

## 实验目的

本实验回答一个限定问题：在完全一致的官方 Table 4 split、训练样本、验证累计方式、测试观测集合和三种子约定下，把 NDR-Net 的 STAEformer 骨干换成 BiST 是否带来稳定提升。

本实验不是 IGSTGNN 原论文方法行，也不是 `protocol_v3_noleak_dedup` 结果。它继续使用 `igstgnn_table4_official_v1`，有效性标签固定为 `official_protocol_with_known_overlap`。

## 固定矩阵

- 矩阵 ID 为 `igstgnn_table4_bist_ndr_pair_18_v1`
- 矩阵 SHA-256 为 `90f5ceb4a0e1306d3f1ff11268b76220a289a37a922c6926665825a56c9cdeab`
- 配置 SHA-256 为 `b763fb35b392e265a410325efa0ef34afa3b0d16c7c2e56c71bc3243844e8764`
- 基础 39 项矩阵 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`
- 方法为 `bist_aligned` 与 `bist_ndr`
- 地区为 Alameda、Contra Costa、Orange
- 种子为 42、1、2
- 任务总数为 18

两个方法使用相同的 batch 16、30 个 epoch、Adam、CosineAnnealingLR、全局验证 masked MAE 和 `partial_batch: retain`。测试集最后不足一个 batch 的样本不会丢弃。

## 模型适配

`bist_aligned` 使用固定 BiST 上游提交 `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`，输入为官方标准化流量，损失和输出在原始流量单位计算。

`bist_ndr` 保留 BiST 编码、backcast residual 和自适应残差图传播。外部适配器从共享 BiST 表示生成正常残差头与扰动残差头，再使用 NDR 的 horizon gate、oracle gate 蒸馏和专门化损失。正常头锚定仅由 train split 拟合的周内时段中位数，扰动头锚定最近观测值。

BiST 上游目录保持 pristine。由于该上游 checkout 没有许可证文件，适配器只在仓库外层调用其公开类，没有复制或修改上游源文件。

## 验证

- Mac 全仓回归为 480 passed 和 2 个既有 warning
- 5080 Windows 专项测试为 5 passed
- Alameda 两个方法的真实 CPU 完整优化步探针通过
- Orange `bist_aligned` CUDA 探针 loss 为 `153.23732758`
- Orange `bist_aligned` 峰值分配显存为 `439906816` bytes
- Orange `bist_ndr` CUDA 探针 loss 为 `32.52203941`
- Orange `bist_ndr` 峰值分配显存为 `457959424` bytes
- 两个 Orange 探针都保持有效 batch 16，并使用两个 microbatch 8
- 正式训练代码摘要为 `1bc7f513a4f7a51c967ad9e4f65527c0e9d3345c844c799232025b37a75ca692`

## 5080 正式启动

5080 使用独立 checkout `D:\t4\bist_ndr_repo`，结果根为 `D:\t4\bist_ndr_pair\results`。Windows 计划任务名为 `XTraffic_BiST_NDR_Pair_v1`，执行时限为 `PT0S`，开机后可恢复。

第一次计划任务以 SYSTEM 账户启动时看不到 ASUS 用户 site-packages，因此 NumPy 导入失败。失败发生在任务训练入口之前，没有正式任务目录或正式结果。启动脚本随后显式保留用户 site-packages，并修复子进程覆盖 `PYTHONPATH` 的问题。两项修复都有回归测试。

正式队列于 2026-07-31 10:06:03 新加坡时间重新启动。首项 `pair001_bist_aligned_alameda_seed42` 已于 10:14 完成 30 个 epoch，并通过五文件验收。best validation MAE 为 `11.69943708`。H3、H6、H12 与 Average MAE 为 `11.205525`、`11.784402`、`12.675788` 与 `11.740657`。prediction shape 为 `2204 × 12 × 521`，有效观测点数为 `10989142`。队列完成数为 1/18，失败数为 0，正在执行 `pair002_bist_aligned_alameda_seed1`。

首项只是 Alameda BiST seed 42 的单项结果，不得用于报告 BiST-NDR 性能结论。当前控制链只负责本机持久训练与五文件任务验收，没有配置自动关机或结果服务器发布。

## 队列完成更新

队列已于 2026-07-31 13:41:13 新加坡时间完成。`queue_status.json` 记录的 runner 状态为 completed，18/18 任务全部通过任务级五文件验收，失败数为 0，skipped 数为 0。正式任务树包含 18 个任务目录和 90 个文件。

这一状态只证明 runner 层的任务完整性。完成多种子汇总和窗口级配对分析前，不报告 BiST-NDR 优于或劣于 BiST 的性能结论。该统一 30 epoch 结构配对实验与 `igstgnn_table4_bist_ndr_native100_18_v1` 原生配方 100 epoch 补充矩阵彼此独立。
