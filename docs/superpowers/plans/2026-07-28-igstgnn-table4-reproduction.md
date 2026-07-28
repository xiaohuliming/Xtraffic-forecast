# IGSTGNN Table 4 公开材料最大忠实复现计划

状态：2026-07-28 数据、源码和本地单元门已通过，5080 已完成数据与源码部署，正式训练尚未启动

## 目标

使用 IGSTGNN 作者公开的数据、代码和论文说明，重跑 Table 4 中 Alameda、Contra Costa、Orange 三个地区的全部 13 个方法，生成同结构的 Horizon 3、Horizon 6、Horizon 12 和 Average 指标表。

当前主目标是复现论文表格，不是复用 XTraffic 项目的 v3 无泄露缓存。两套协议必须使用独立的数据根、结果根、矩阵名和有效性标签。

## 主复现协议

协议名：`igstgnn_table4_official_v1`

1. 数据只使用作者发布的 Kaggle `Data for IGSTGNN` version 1。
2. 每个地区使用发布的 `incident_all.npy`、`incident_stats.npz`、`adj_matrix.npy`、`sensors.csv` 和两个映射文件。
3. 使用官方 `prepare_splits.py` 按发布顺序直接切分 70%、15%、15%。
4. 不重新随机打乱，不重新归一化，不从项目 v3 HDF5 转换。
5. 历史输入为 12 个五分钟步，预测目标为未来 12 个五分钟步。
6. 主复现保留官方样本中的重复预测时刻、相邻窗口重叠和官方指标语义。任何去重、split 隔离、因果补值或显式 mask 修复都会改变论文协议，只能进入独立的无泄露稳健性实验。
7. Table 4 的 Average 使用 12 个逐 horizon 指标的算术平均，不使用全体元素池化指标。
8. 主表使用官方 LargeST 和 IGSTGNN 的 masked MAE、RMSE、MAPE 语义。额外的 float64 审计指标单独保存，不替换论文口径数字。

## 官方数据核查结果

Kaggle 发布文件中的 `incident_all.npy` 是一维 object array。样本数如下：

| 地区 | 发布样本数 | 官方脚本 train | val | test | 论文 Table 1 事故数 |
|---|---:|---:|---:|---:|---:|
| Alameda | 14,687 | 10,280 | 2,203 | 2,204 | 14,687 |
| Contra Costa | 5,585 | 3,909 | 837 | 839 | 5,587 |
| Orange | 18,700 | 13,090 | 2,805 | 2,805 | 18,700 |

发布样本要求每条记录包含单个 incident context。该发布协议实质上是 one-sample-per-incident 的事件样本协议，不是全年每个五分钟预测时刻组成的完整滑动窗口协议。

Contra Costa 发布文件比论文 Table 1 少 2 条。正式复现必须使用实际发布的 5,585 条，并在结果报告中保留这一差异，不能补造缺失样本。

## 源码锁定

| 范围 | 来源 | 锁定提交 | 使用原则 |
|---|---|---|---|
| IGSTGNN | `fanlixiang/IGSTGNN` | `64c6e243466f791f1b75e50421d63e39577ec9fa` | 使用官方模型、dataloader、engine、指标和训练入口 |
| HL 至 D2STGNN 共 11 个方法 | `liuxu77/LargeST` | `b7c6ff5f7f0c10051ed10a319b2cfe58021312f9` | 使用对应 model、engine 和实验入口，只增加 XTraffic 官方 split 适配层 |
| BiST | `PoorOtterBob/BiST` | `dd94adf7721fcbb9e3feb5d1b44040305199a4cc` | 在仓库外锁定运行，不把无许可证源码复制进本仓库 |

IGSTGNN 官方仓库没有公开 Table 4 baseline 的 XTraffic adapter、逐地区配置、checkpoint 或运行日志。LargeST 也没有发布这三个地区的 70%、15%、15% 配置。BiST 的 Table 4 适配代码和准确配置同样没有公开。因此，本实验能声明为“按公开材料最大忠实复现”，不能在缺少作者补充材料时声明为逐命令完全复现。

所有必要适配必须限制在数据读取、输出归档和新硬件兼容层。每个补丁都要记录文件、原因、语义影响和 SHA-256。改变模型结构、损失、调度器、早停、curriculum、初始化或指标的补丁不得进入主复现矩阵。

## 执行矩阵

主复现表只有一个论文口径运行层：

- 13 个方法乘 3 个地区，共 39 个表格单元。
- HL 为确定性方法，共 3 次独立执行。
- 其余 12 个学习方法共 36 次训练。
- 现有 v3 full-window GWN、旧 baseline zoo 和旧 IGSTGNN 转换结果全部不复用。

如果主复现完成后再做三种子稳健性扩展，独立执行数为 12 个学习方法乘 3 个地区乘 3 个种子，再加 3 个 HL，共 111 次。该扩展必须使用另一个矩阵名和另一张表，不能称为 Table 4 严格复现。

## 训练和指标口径

1. IGSTGNN 使用官方 seed 2025、Adam、初始学习率 0.002、最大 100 epoch、patience 20。Alameda 和 Contra Costa batch size 为 48，Orange 为 24。
2. 11 个 LargeST 方法使用锁定提交中的模型专属 engine、优化器、调度器、curriculum、初始化和默认训练参数。seed 为 2023，batch size 为 64，训练集使用 NumPy 全局随机排列，三个 split 均按源码丢弃不足一个 batch 的尾部样本。任何论文未披露的 XTraffic 参数必须在配置清单中标记为公开材料缺口。
3. D2STGNN 使用 LargeST 的公开默认 batch size 64、最大 100 epoch 和 patience 30。DGCRN 与 D2STGNN 的一天时间槽数保留 288，作为五分钟数据的必要公开材料推断。
4. BiST 使用官方提交的模型和训练逻辑，batch size 为 64，最大 500 epoch，patience 20，一天时间槽数按锁定 XTraffic 注册配置保留为 96。官方代码每次运行调用 `torch.randint(999999, (1,))` 产生 seed，因此三个地区必须分别冻结实际随机值并写入不可变矩阵清单。BiST 仓库没有许可证，只在仓库外执行锁定 checkout，不复制源码。
5. LargeST 与 BiST 的正式 loader 会丢弃尾部样本。Alameda 的 train、val、test 实际消费数为 10,240、2,176、2,176，分别丢弃 40、27、28 条。Contra Costa 实际消费数为 3,904、832、832，分别丢弃 5、5、7 条。Orange 实际消费数为 13,056、2,752、2,752，分别丢弃 34、53、53 条。IGSTGNN 使用源码的保留尾部语义，消费全部样本。
6. 主表保存 Horizon 3、6、12 的 MAE、RMSE、MAPE，以及 12 个 horizon 指标的算术平均。
7. 每次运行保存 checkpoint、完整预测、标签、实际使用的有效掩码、逐 horizon 指标、训练日志、环境清单、GPU 信息、源码提交和配置哈希。

## 已完成的启动验收

1. 官方三地区 split 已落盘并完成两次本地严格复验。5080 上的 27 个文件共 9,051,956,107 字节，三个 split manifest 的 SHA-256 与本地完全一致。
2. LargeST、BiST 和 IGSTGNN 的锁定 checkout 已部署到 5080。远端 HEAD、tree 和 dirty 状态均通过检查。
3. 深度审计 v2 已区分原始记录与模型实际消费窗口。Alameda 和 Contra Costa 的模型实际消费集合没有重复预测时刻或跨 split 重合。Orange 保留官方协议中的 9 个重复预测时刻，以及 train 对 val 1 个、train 对 test 3 个跨 split 重合，因此有效性标签保持 `official_protocol_with_known_overlap`。
4. 13 个方法均已通过真实上游前向、checkpoint round trip 和测试输出单元门。12 个学习方法均通过真实上游单 batch 参数更新测试。
5. macOS 本地当前为 164 项通过，另有 1 个 DCRNN 上游弃用 warning。Windows 原生安全文件读取、reparse point 和 hardlink 防护、独占结果发布、支持文件读取时哈希绑定均已纳入测试。
6. 5080 当前空闲可用，正式数据和源码已部署到短路径 `D:\t4`。训练必须等最终矩阵清单、代码提交和 Windows CUDA 复验全部通过后启动。
7. 正式 v2 矩阵清单已生成，共 39 项。清单 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`。BiST 在 Alameda、Contra Costa、Orange 的冻结 seed 分别为 676615、683452、150462。

## 正式启动顺序

1. 生成并冻结 39 项 v2 矩阵清单，三个地区分别记录 BiST 实际随机 seed。
2. 提交并推送清单、配置、训练适配层、断点续跑器和测试。
3. 将该提交的精确文件快照部署到 5080，复验 Git 内容哈希、上游 checkout、正式 split 和 Windows CUDA。
4. 依次运行 Alameda 的 HL、LSTM、一个图模型和 IGSTGNN 四个正式 pilot。pilot 与全矩阵使用同一不可变清单和结果根，因此通过验收的结果可以直接被完整队列复用。
5. pilot 通过后启动完整 39 项单卡顺序队列。runner 只跳过完整通过五文件验收的已有任务，任一异常立即停止，不能覆盖损坏或不完整目录。

## 启动门

正式全量训练前必须依次通过：

1. 数据门：下载完成、逐文件 SHA-256、NPY 头、样本数、键、shape、排序和官方 split 复验通过。
2. 源码门：三个上游提交、许可证状态、补丁清单和环境镜像摘要锁定。
3. 指标门：用固定合成张量证明 Table Average、masked MAE、RMSE、MAPE 与官方实现逐值一致。
4. 单元门：13 个方法全部通过前向 shape、单 batch 训练、checkpoint 恢复和测试输出验收。
5. Pilot 门：先在 Alameda 跑 HL、LSTM、一个图模型和 IGSTGNN。验证输出结构、显存、速度、早停和表格生成。
6. 资源门：Pilot 通过后再生成正式 39 项分片并启动多机训练。

## 最终交付

1. 与论文 Table 4 同结构的复现表。
2. 复现值与论文值的逐单元绝对差和相对差表。
3. 每个数字到预测文件、配置、日志和源码 SHA 的追溯索引。
4. 公开材料缺口、必要适配和已知协议风险清单。
5. 独立有效性标签。主复现结果不得并入 `protocol_v3_noleak_dedup` 的无泄露结论。
