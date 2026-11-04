# NDR 插件抗过拟合变体的工程准备与 5080 执行记录

日期：2026-10-21，Asia/Singapore。
范围：protocol_v3_noleak_dedup event 协议，Alameda 与 ContraCosta 开发区，seed 42，原生 BiST 配方 100 轮。
前因：2026-10-21 的同预算对照显示 BiST-NDR 在 100 轮预算下第 33 到 41 轮达到最佳后验证回升，训练 MAE 持续下降，而原生 BiST 到第 87 到 96 轮仍在改善。本轮实现两种直接针对该失败模式的训练变体，并在 5080 上做小型对照。

## 代码改动

分支 `codex/bist-ndr-matched`，工作树 `/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`，窄提交 `548ac88dd2419c5e7be94eb864c61f566a02691f`，父提交 `b4d469ef3b4a94f3057c8326d1d2fd33fca75496`，未推送。改动只涉及三个文件：

| 文件 | SHA-256 |
|---|---|
| `ndr_research/strong_references.py` | `d50651cd3e80b48e88ab94a772288833d7eb36ee4c24b0154f8656c46b976101` |
| `scripts/train_ndr_strong_reference.py` | `77280628ee5719b628d00908626ae5b08a0b559a32a136d238420d1dd5181a2e` |
| `tests/ndr_research/test_bist_ndr_antioverfit_variants.py` | `53ab44a9679b53a62c9d098c9795b0ec238ab91eee4f7dc351d43440022a42ed` |

`scripts/train_ndr_research.py`、`ndr_research/data.py`、`scripts/reproduction/igstgnn_table4/bist_ndr_pair_model.py`、`scripts/train_staeformer_deseason.py` 与控制器 `scripts/remote/windows/ndr_matched_budget_serial_queue.py` 字节未变，哈希见 `receipts/changed_and_dependency_sha256_20261021.txt`。

两种变体以新的 `method` 值加入，`bist` 与 `bist_ndr` 的 recipe 字典与计划 A 落盘的 `training_recipe` 逐键相等，已在准备时核对。

`bist_ndr_anneal`：模型与 `bist_ndr` 完全相同，仍是锁定 BiST 上的 `BiSTNDR` 适配器。门控蒸馏与专门化损失的权重从 1.0 与 0.3 按已完成有效更新数线性退火到 0，退火终点为计划总更新数的一半，计划总更新数等于轮数乘每轮训练批数，含保留尾批。ContraCosta 为 100 乘 257 等于 25700 次更新，终点 12850；Alameda 为 100 乘 501 等于 50100 次更新，终点 25050。权重在每次更新前由 `plugin_loss_weights` 按 checkpoint 中的 `progress["updates"]` 解析，严格恢复后按已完成更新数继续。配方 `plugin_loss.schedule` 记录名称、单位、终点比例、初值与终值；checkpoint 的 `progress["plugin_loss_schedule"]`、每轮历史行与 summary 记录最后一次更新实际使用的权重、已完成更新数、计划总更新数与终点。

`bist_ndr_regularize`：模型为 `BiSTNDRHeadDropout`，是 `BiSTNDR` 的子类，只在正常头与扰动头的输入处各加一个 `nn.Dropout(0.2)`，训练模式生效，eval 模式返回原张量，前向与 `bist_ndr` 逐位相同。正常头由共享编码重新经原生回归层计算，以便其输入也能带 dropout 掩码；该转置视图与原生 MLP 卷积的张量步长一致，无 dropout 时结果逐位相同。优化器仍为 Adam 学习率 0.002、eps 1e-8、裁剪 5、MultiStepLR 里程碑 30、60、80、100，但分成两个参数组：原生 BiST 参数 weight decay 1e-5，插件新增参数即两个复制头与门控 weight decay 1e-3。配方 `plugin_regularization` 记录 dropout、参数组与各自的 weight decay。

## CPU 验证

新增 12 项测试与既有 27 加 11 项强对照测试一起运行，50 项通过，Ruff 通过。JUnit 回执 `receipts/junit_antioverfit_variants_20261021.xml`，SHA-256 `1eb9661e1d7a31044eefcd0592b1a6d3e461683ab030d336772a98f399162836`。新测试覆盖：

1. 两个变体都注册为插件方法，参数数与 state_dict 键与 `bist_ndr` 相同，upstream 记录变体身份，跨变体 checkpoint 被身份检查拒绝。
2. 退火数值：计划 200 次更新时终点 100，第 0、25、50、75、100、150 次更新的权重分别为 1.0 与 0.3、0.75 与 0.225、0.5 与 0.15、0.25 与 0.075、0 与 0、0 与 0；缺少计数时拒绝；常量配方忽略计数。
3. 配方字段：两变体与原生 BiST 共用优化器字典；`bist_ndr` 配方不含新键。
4. 第 0 次更新时退火臂的目标值、三个损失分项与更新后参数与 `bist_ndr` 逐位相同；中途与终点后目标值等于 MAE 加缩放后的辅助项。
5. 两次更新后保存 checkpoint，恢复进程得到相同的已完成更新数与权重，下一步目标值与参数逐位相同；完整训练循环的历史行与 summary 记录每轮权重。
6. 参数组划分：原生组与插件组的 id 集合分别等于原生参数与非原生参数，插件组参数量等于 `added_parameters`，名称只来自三个插件模块，weight decay 为 1e-5 与 1e-3；原生 weight decay 被改动时拒绝。
7. dropout 只在训练生效：eval 前向与直接构造的 `BiSTNDR` 逐位相同，训练模式下两次不同种子前向互不相同且与 eval 不同，dropout 置零后恢复逐位相同；退火臂 eval 前向同样逐位相同。
8. 正则臂 checkpoint 恢复两组优化器并继续一步逐位相同；独立进程验证能重建正则臂并复现 MAE。
9. CLI 接受两个方法。

真实 ContraCosta v3 缓存上的 CPU smoke 使用 batch 4、1 轮、各 2 个训练与验证批，产物见 `cpu_smoke/`，不含 checkpoint 字节：

| 检查 | 结果 |
|---|---|
| `bist_ndr_anneal` 训练 smoke | 完成，2 次更新，计划 2 次更新，最后一次更新权重已为 0，验证 MAE 15.012586 |
| 其独立进程 verify | `verified_smoke_only`，训练 PID 3496，验证 PID 3540，MAE 逐值复现 |
| `bist_ndr_regularize` 训练 smoke | 完成，2 次更新，验证 MAE 14.962952 |
| 其独立进程 verify | `verified_smoke_only`，训练 PID 3584，验证 PID 3623，MAE 逐值复现 |
| 两变体 CPU 校准路径 | `calibrated_cpu_engineering_only`，4104 个训练批与 868 个验证批，不输出任何分数 |

smoke 数字只证明代码路径可运行，不是精度证据。

## 源码包与主机准备

46 文件源码包成员列表与计划 A 的包完全相同，只有两个成员字节变化。ZIP 从提交 `548ac88` 的已提交字节生成，SHA-256 `949b86f3c7276700c7c84e986be34f2139bc8aed3bf9c98dbd8fb22fa7f9d9fa`，191418 字节。在 5080 新根 `C:/XTrafficResearch/bist_ndr_antioverfit_20261021/prepared_548ac88_949b86f3/repo` 解包，46 个文件逐字节回读与清单一致，源端最小 Git 为 `5a9607cbb7bbba3a11ce7c5e543d49da4395c585`，46 个跟踪文件，工作树 clean。pristine BiST 仍为 `dd94adf` 且 clean。控制器复用 `C:/XTrafficResearch/bist_ndr_matched_20261020/control_b4d469e/ndr_matched_budget_serial_queue.py`，SHA-256 `bddc1e2295493ccc3134ba5a7dafcbff77784ae49a86323b1a7ecc63b2e124b0`，与工作树提交 `b4d469e` 的文件一致。训练入口在源端只做 help 导入检查，方法列表含两个新变体，没有读取数据。回执见 `source_package/source_host_preparation_receipt.json`。

## 阶段 0 校准

计划 `stage0/stage0_calibration_plan.json`，digest `4710a1ce8371e148e53a44970ffceac46e3177b3226e8af2ffcde3d7682073b8`，四个校准作业：ContraCosta 与 Alameda 各一次 `bist_ndr_anneal` 与 `bist_ndr_regularize`，100 轮预算，batch 64，3 个预热批与 20 个计时批，每项超时 900 秒，总预算 7200 秒，绝对截止 2026-10-21T06:49:51.697268Z，最低空闲显存 12000 MiB。源端 admit-only 返回 0 后以无触发器计划任务 `XTraffic_BiSTNDR_AntiOverfit_Stage0_20261021` 启动，控制器 PID 42692，首个 worker PID 43860 于 04:53:36Z 启动，启动时空闲显存 14548 MiB。前两次启动尝试因 PowerShell 脚本解析错误在源端没有执行任何命令，没有注册任务，也没有创建状态文件，已由后续探测确认。

阶段 0 于 04:55:01Z 以 `completed_all_jobs` 结束，控制器预算用时 84.9 秒，四项均 `calibrated_cuda_full_effective_batches` 且 `cuda_budget_eligible` 为真，校准文件不含任何分数键。26 个文件已取回，传输前后源端清单逐字节相同，本地逐文件哈希一致，回执 `stage0/closeout.json`。100 轮 p90 外推与冻结超时：

| 臂 | 地区 | 均值外推秒 | p90 外推秒 | 冻结超时秒 | 训练峰值显存 |
|---|---|---:|---:|---:|---:|
| bist_ndr_anneal | ContraCosta | 2941.7 | 3016.1 | 6625 | 1.70 GB |
| bist_ndr_regularize | ContraCosta | 3013.6 | 3075.9 | 6714 | 1.77 GB |
| bist_ndr_anneal | Alameda | 6012.6 | 6096.7 | 11246 | 1.78 GB |
| bist_ndr_regularize | Alameda | 6102.2 | 6223.5 | 11436 | 1.85 GB |

冻结规则与计划 A 相同：1.5 倍 p90 加每轮 15 秒加 600 秒。作为参照，计划 A 的 bist_ndr 在 ContraCosta 与 Alameda 的 p90 外推为 3776.4 与 7429.0 秒，实际训练用时 3299.8 与 5741.9 秒。

## 计划 C 冻结与启动

计划 `plan_c/plan_c_antioverfit.json`，digest `ef811faff7c20389634af88f67abd2f39239efb52c2587cf1749335130a37a01`，文件 SHA-256 `28dcea040e279724d68b37024027126690fa265be8a49377ec0aee3a61bc16c2`。八个作业按 ContraCosta anneal、其验证、ContraCosta regularize、其验证、Alameda anneal、其验证、Alameda regularize、其验证的顺序串行执行，每个训练后由新进程重载 best.pt 复验。四个训练超时合计 36021 秒，验证各 600 秒，总预算 42300 秒，绝对截止 2026-10-21T16:44:50.090795Z，busy wait 600 秒，最低空闲显存 12000 MiB。源端 admit-only 返回 0 后以无触发器计划任务 `XTraffic_BiSTNDR_AntiOverfit_PlanC_20261021` 启动，控制器 PID 47668，首项 ContraCosta anneal 的 worker PID 36200 于 04:59:54Z 启动，启动前空闲显存 14546 MiB，首个 worker 起步后空闲 12460 MiB。预期训练总时长约 5.0 小时，不含验证与等待。回执 `plan_c/launch_receipt.json`。

预注册判据：变体必须在 Alameda 与 ContraCosta 的完整验证 MAE 同时严格低于计划 A 的 BiST-NDR 100 轮值 10.746248808511963 与 11.414684481229234，并同时严格低于原生 BiST 100 轮值 10.800637365300256 与 11.27915198139963，才算通过；一个地区不满足即不通过，不以平均抵消。通过者再在 STAEformer 版 NDR-Net 上以同机制做 30 轮两地区对照，参照 10.745627 与 11.275205。不读 test，不读 Orange，不停止用户程序。

## 计划 C 执行过程与主机不可达事件

本地每 10 分钟读取一次控制器状态并保存快照于 `plan_c/state_snapshots/`。截至最后一次成功读取 08:53:25Z，八个作业中六个成功：ContraCosta anneal 训练 2941.4 秒并通过独立验证，ContraCosta regularize 训练 3046.3 秒并通过独立验证，Alameda anneal 训练 5883.1 秒并通过独立验证。Alameda regularize 的 worker PID 48972 自 08:18:35Z 起训练中，控制器 PID 47668 心跳 08:52:36Z，当时 GPU 已用显存 5590 MiB，比此前快照的约 3700 MiB 多出约 1.9 GB，额外占用者未识别。08:12:56Z 与 08:43:22Z 两次读取返回空输出，之后读取恢复，属于链路抖动。

父会话于 12:14Z 报告主机自约 09:10Z 起离线，Tailscale 显示 win-09sjo85ojod offline、last seen 3h ago。本会话 12:16Z 的探测记录：LAN 与 Tailscale 地址 ping 各 3 包全部丢失，22 端口均超时，两条链路 ssh 均 255，Tailscale 状态为 offline。记录见 `incident_host_unreachable_20261021/incident_record.json` 与同目录探测原件。

按均值外推，Alameda regularize 训练应于约 10:00Z 结束，冻结超时截止 11:29:11Z，主机在均值外推时间之前已不可达，该 worker 是否完成、超时或随主机一起停止均未知。本会话不据此推断该臂或计划失败，不重启，不重新派发，不注册新任务。控制器拒绝已有状态文件的计划，无触发器任务不会自行重跑，因此不存在重复启动的可能。主机恢复可达后先只读读取 controller_state；若控制器已终态则按原流程取回分析；若主机曾重启或原进程缺失，则按 2026-10-17 strong4 的先例做只读 forensic 保全并记录关闭。计划绝对截止 16:44:50Z 不延长。

## 2026-11-03 结果与关闭

plan C 的 8 个作业于 2026-10-21T10:03:09Z 全部成功，62 个文件由子会话取回并通过前后清单与逐文件哈希核对，分析因子会话触及额度上限延后到 2026-11-03 由父会话完成。anneal 为 Alameda 10.813460、ContraCosta 11.419115，regularize 为 10.810845 与 11.403502，均未在两地区同时低于 plan A 的原 BiST-NDR 100 轮与原生 BiST 100 轮，预注册判据不通过，不进入 STAEformer 版同机制对照、seeds 1、2 或 Orange。四条曲线仍在第 18 到 33 轮后回升。报告见 `docs/reports/NDR插件抗过拟合候选结果_2026-11-03.md`，收口回执 `plan_c/closeout.json`。plan C 之后没有新的 GPU 工作。
