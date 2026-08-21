# FDN benchmark v1 启动审计

_日期：2026-08-22，Asia/Singapore_

当前状态：本地训练前代码门禁完成，RTX 5080 当前关机，正式训练尚未启动

---

## 🎯 研究问题

本矩阵用于判断 Fourier Dual Net 是否具有独立方法论文价值。主要比较是 FDN learnable-K3 与等参数 DualGWN。单 GWN 只用于测量容量贡献，不能作为频谱路由的唯一机制对照。

独立重复单位是训练 run 的随机种子。测试滑窗相互重叠，不作为独立重复。模型差值按相同数据集与相同种子配对，统计报告使用种子层方向一致性和按 288 个 forecast origin 分块的层级 bootstrap。

## 📋 正式矩阵

- 协议：`fdn_benchmark_v1`
- 矩阵：`fdn_benchmark_v1_go_no_go_pems04_pems08`
- 数据集：PEMS04、PEMS08
- 模型：GWN、DualGWN、FDN fixed-K3、FDN learnable-K3
- 种子：42、1、2
- 任务数：24
- 输入与输出：12 步输入，12 步预测
- 切分：按原始时间戳 60%、20%、20%，目标时间集合两两不相交
- 归一化：只使用训练段唯一原始时间戳中的非零有效观测
- 缺失值：零值作为缺失，归一化后输入填零，目标使用显式 mask
- 训练：100 epoch，不提前停止
- checkpoint：全局 validation MAE 最优
- optimizer：Adam，学习率 0.001，weight decay 0.0001
- scheduler：每个 optimizer step 更新的 CosineAnnealingLR
- batch size：48

canonical manifest 位于 `config/fdn_benchmark_v1/go_no_go_pems04_pems08_v1.json`，SHA-256 为 `2a1a999f67492d39e00309cda52737dc6e79bb14d3df3e672a28f183d4216ac6`。

## 🔐 数据身份

| 文件 | SHA-256 |
|---|---|
| `PEMS04.npz` | `95a3c9b720fffdb85f0330d09bfab41b0b3cad0ca86c0d7d5f3accacb4ac999a` |
| `PEMS04_distance.csv` | `68ffaa80447eb34428a9999dc28363c61406ec0fd714686d0582d2386ac6f5f7` |
| `PEMS08.npz` | `e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075` |
| `PEMS08_distance.csv` | `859d68e7583ed4e6016588eaaca494d3ece286dae361614c5c1833017f87eaa7` |

## ✅ 完成门

每个正式任务必须保存并重新验收以下产物：

1. `summary.json`
2. `config.json`
3. `ckpt_best.pt`
4. `test_predictions.npz`
5. `training_diagnostics.jsonl`
6. `train.log`
7. FDN 任务额外保存 `decomp_mask.json`

runner 从 prediction artifact 重新计算 MAE、RMSE、MAPE、H3、H6 与 H12 指标，并核对完整 test forecast origin。训练代码 SHA、config hash、数据 SHA 与邻接 SHA 必须逐项一致。

## 📍 Go 或 No-Go 门

只有以下条件全部满足，才继续扩展 PEMS03、PEMS07 与完整论文基线：

1. PEMS04 与 PEMS08 的三种子平均 FDN learnable-K3 减 DualGWN 差值均为负。
2. 六个数据集与种子配对点至少五个为负。
3. 至少一个数据集的层级 bootstrap 95% 置信区间上界低于零。

否则停止把 FDN 作为独立方法论文，不为追求正结果修改停止线。

## 🔍 本地门禁

- 新增专项测试：10 passed
- 真实 PEMS04 FDN CPU 优化步探针：通过
- 真实 PEMS08 DualGWN CPU 优化步探针：通过
- Python 编译：通过
- Ruff：通过
- 全仓回归：668 passed，2 个既有 warning
- diff whitespace 检查：通过

## ⚙️ RTX 5080 状态

2026-08-22 的局域网和 Tailscale SSH 探测均超时。Raspberry Pi 可访问，但没有保存 5080 的邻居 MAC，也没有已配置的 Wake-on-LAN 命令。当前不能远程唤醒 5080。

主机上线后必须先执行只读状态检查，确认 GPU 空闲、旧 XTraffic 计划任务仍禁用、磁盘容量充足。随后部署 clean checkout、逐文件复验数据、运行八项 CUDA 优化步探针，再审核 plan 与代码 SHA。只有这些门禁通过后才启动 24 项正式队列。
