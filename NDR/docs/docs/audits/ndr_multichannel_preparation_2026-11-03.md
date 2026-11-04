# NDR-Net 多通道输入候选的工程准备与 5080 执行记录

日期：2026-11-03，Asia/Singapore。
范围：protocol_v3_noleak_dedup event 协议，Alameda 与 ContraCosta 开发区，seed 42，原 s0g 配方 30 轮。
前因：路由器侧改动、节点无关图模型、插件迁到 BiST 骨干的长预算对照与两种抗过拟合变体均为 No-Go。插件侧空间已基本用尽，剩余空间在骨干能看到的输入。v3 缓存含 flow、occupancy 与 speed 三通道，原 NDR-Net 只用 flow。低流量在自由流与拥堵下都会出现，速度与占有率是扰动态的直接签名。本轮把 occupancy 与 speed 的去季节化残差作为骨干附加输入，检验 STAEformer 版 NDR-Net 能否因此在两个地区同时改善。

## 方法

候选记为 multichannel。原 s0g 的流量残差输入、时间嵌入、门控、两个锚点、双残差头、oracle 蒸馏与专门化损失全部不变。新增部分只有三处：

1. occupancy 与 speed 各自用 train 时间戳上已观测点拟合周内 7 乘 288 槽位中位数气候态与全局残差标准差，拟合代码与 flow 的 `fit_weekly_climatology` 逐步相同，对 flow 通道复现原函数的全部数组与标量。
2. 每个历史点的附加残差等于观测值减气候态再除以该通道残差尺度。未观测点以零残差进入，不读取缓存中的插补值。speed 约 38% 缺失且缺失处多被零填充，若直接使用插补值会产生巨大的假残差。
3. 两个附加残差经一个无偏置线性投影进入 STAEformer 的输入嵌入，与原 flow 输入投影的输出相加。投影在构造时为零，且在全部原参数之后以不消耗随机数的方式创建，因此同一种子下候选的初始权重与原 s0g 逐位相同，初始预测也逐位相同。附加参数为 48 个。

两地区拟合得到的附加残差尺度：

| 地区 | occupancy 残差尺度 | speed 残差尺度 |
|---|---:|---:|
| Alameda | 0.025130 | 4.707329 |
| ContraCosta | 0.024215 | 4.618688 |

## 代码布局与一个设计决定

分支 `codex/bist-ndr-matched`，工作树 `/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`，窄提交 `211461b6629d1879cc3f1b8f7df6ef6228f5987c`，父提交 `548ac88dd2419c5e7be94eb864c61f566a02691f`，未推送。只新增三个文件：

| 文件 | SHA-256 |
|---|---|
| `ndr_research/multichannel.py` | `597757eee20d0c24df42dc18afc7bf623c1c71099725c29e6df100a98b50dbd9` |
| `scripts/train_ndr_multichannel.py` | `47971d9e6568b0be5cc1b3523112321749e889c1c02a2c72e6f01eb9a7c2b044` |
| `tests/ndr_research/test_multichannel.py` | `b9b290672da02ab0b6a9b5c211be9a62a75e6702e60b335cd821d43946c6b452` |

第一版实现直接在 `train_ndr_research.py`、`ndr_research/models.py` 与 `ndr_research/data.py` 中加入 multichannel 变体。全量回归随即暴露一个问题：stage1 参照运行的源码快照记录了这三个文件的哈希，最终地区评估的准入会核对快照，因此原地修改会让本分支再也无法收录 stage1 的 NDR-Net checkpoint，既有测试 `test_actual_complete_baseline_checkpoint_structure_requires_no_target_data` 也因此失败。最终实现改为只新增文件，这三个文件与 stage1 快照中的其余 18 个文件逐字节不变，并有测试直接核对两个地区的快照。被否决的原地修改补丁与其失败回归日志保存在控制目录 `receipts/`。

新入口 `train_ndr_multichannel.py` 以独立模块实例加载未修改的 `train_ndr_research.py`，只把批数据准备、前向、checkpoint 工厂参数与模型工厂四个接口绑定到多通道版本。训练循环、microbatch 有效批损失分母、严格断点恢复、checkpoint 写入与独立进程验证均是原训练器的同一份代码。只有校准函数另写，在原输出上增加 p90 批耗时、预算轮数与 `cuda_budget_eligible`，供冻结规则使用。

## CPU 验证

专项测试 28 项全部通过。研究线 `tests/ndr_research` 与串行控制器测试合计 570 项通过，33 项为既有跳过。JUnit 回执在控制目录 `receipts/`。专项测试覆盖：

1. stage1 两个地区参照快照中的 21 个文件与当前工作树逐字节相同。
2. 新入口的四个绑定接口确实替换，原训练器模块对象未受影响，复用函数的字节码与原训练器相同。
3. 同一种子下候选的原有参数与原 s0g 逐位相同，附加投影为零。eval 与训练模式下前向逐位等于原模型，损失、损失分项与共享参数梯度逐位相同，附加投影的梯度非零。
4. 附加投影非零后改变附加输入会改变预测。未观测点填入 NaN、无穷、负十亿或零时预测不变且有限，反向梯度有限；已观测点出现非有限值被拒绝。
5. 加载器给出的附加历史与 flow 历史来自同一批样本，flow 字段与原加载器逐字节相同，错位时报错；断点状态与原加载器同构。
6. 附加通道气候态拟合对 flow 通道复现原函数，只使用本通道的 train 段。
7. microbatch 1 与 2 的目标值与梯度等于完整有效批。
8. 附加通道全部缺失时，候选在关闭梯度裁剪下与原训练器逐位相同；在正式裁剪 5.0 下只在舍入范围内相符。差异来自裁剪总范数多归约一个全零梯度参数，已通过关闭裁剪的对照确认。
9. 中断后严格恢复与连续训练逐位相同，独立验证能从 checkpoint 工厂参数重建候选并复现 MAE。
10. 命令行记录 multichannel 配方，基线配方字段与 stage1 相同，另一进程验证通过；只接受 multichannel 变体。
11. CPU 校准给出 p90 字段且不含任何分数键。
12. 真实 ContraCosta 缓存上附加批全部有限，零初始化候选的前向与原模型逐位相同。

真实 ContraCosta 缓存上的命令行 smoke 使用 batch 4、1 轮、各 2 个训练与验证批，训练、独立进程验证与 CPU 校准均退出 0，验证 MAE 逐值复现。smoke 数字只证明代码路径可运行，不是精度证据，产物在控制目录 `cpu_smoke/`，不含 checkpoint 字节。

全量回归的第二次运行中，控制器测试 `test_insufficient_gpu_memory_waits_then_stops_at_deadline_or_budget` 失败一次。该测试以真实墙钟为假时钟起点，截止时间转成 ISO 字符串时截断亚微秒，剩余时间偶尔为 299.9999997615814 而非 300。单独重跑三次均通过，控制器与该测试本轮均未改动，原件保留在 `receipts/`。

## 源码包与主机准备

48 文件源码包等于上一轮抗过拟合包的 46 个成员加两个新文件，46 个旧成员字节未变，覆盖 CPU smoke 实际记录的 37 文件运行时快照。ZIP 由提交 `211461b` 的已提交字节生成，SHA-256 `ab61bbef88d1adfb13bac6d55b21b62aa11fee4cca2c432a6f6dc8987bbc24b2`，201461 字节。在 5080 新根 `C:/XTrafficResearch/ndr_multichannel_20261103/prepared_211461b_ab61bbef/repo` 解包，48 个文件逐字节回读一致，源端最小 Git `2436aa8753582f87947cfba11996a55dc2624ddd`，48 个跟踪文件，工作树 clean。源端在 `.git/info/exclude` 写入 `__pycache__/`，使运行时生成的字节码不再把运行记成 dirty，包内容不变。help 导入检查退出 0 且未读数据。控制器仍为 `C:/XTrafficResearch/bist_ndr_matched_20261020/control_b4d469e/ndr_matched_budget_serial_queue.py`，SHA-256 `bddc1e2295493ccc3134ba5a7dafcbff77784ae49a86323b1a7ecc63b2e124b0`。源端运行时为 Python 3.13.3、torch 2.10.0.dev20251203+cu128、numpy 2.2.6，与 stage1 参照相同。回执 `source_package/source_host_preparation_receipt.json`。

## GPU 共享情况

准备与校准时 GPU 利用率约 99%，占用者为 `C:\Mining\SRBMiner-Multi-3-8-3\SRBMiner-MULTI.exe`。按约定不停止任何用户程序，本工作没有启动或停止它。它只占少量显存，空闲显存约 14.8 GB，高于 12000 MiB 门槛，但计算被分时共享。校准测得每个训练批 0.71 到 0.76 秒，约为 2026-10-20 无竞争时同配方 0.32 秒的 2.3 倍。确定性算法已开启，竞争只影响用时，不影响数值结果。

## 阶段 0 校准

计划 `stage0/stage0_calibration_plan.json`，digest `96826b143cba3adb5a09cc6bce3367ad1b045a1ce85e711770b040156f8b5a58`，两个校准作业各 900 秒超时，总预算 3600 秒。admit-only 返回 0 后以无触发器计划任务 `XTraffic_NDR_Multichannel_Stage0_20261103` 启动，控制器 PID 11064，99.3 秒内以 `completed_all_jobs` 结束。14 个文件取回，传输前后源端清单逐字节相同。校准产物 `cuda_budget_eligible` 为真，不含任何分数键。

| 地区 | 训练批均值秒 | 训练批 p90 秒 | 30 轮 p90 外推秒 | 冻结超时秒 | 训练峰值显存 |
|---|---:|---:|---:|---:|---:|
| ContraCosta | 0.7134 | 0.7256 | 23419.2 | 36179 | 4.15 GB |
| Alameda | 0.7586 | 0.7667 | 48374.2 | 73612 | 4.49 GB |

冻结规则与前两轮相同：1.5 倍 p90 外推加每轮 15 秒加 600 秒。收口回执 `stage0/closeout.json`。

## 正式计划冻结与启动

计划 `plan/formal30_multichannel_plan.json`，digest `766584961039cbed798e4c403db2abb567ff4781f17167ac87479421ad137011`，文件 SHA-256 `a9150be5d22d368e51974f73b9ddc655e9425ab79e2e95a59fa7dba5a68475c2`。四个作业按 ContraCosta 训练、其验证、Alameda 训练、其验证串行执行，每个训练后由新进程重载 best.pt 复验。两个训练超时合计 109791 秒，验证各 600 秒，总预算 118800 秒，余量留给 GPU 忙等，绝对截止 2026-11-04T20:02:13.008086Z，busy wait 600 秒，最低空闲显存 12000 MiB。admit-only 返回 0 后以无触发器计划任务 `XTraffic_NDR_Multichannel_Formal30_20261103` 启动，控制器 PID 12484，首项 ContraCosta 训练 worker PID 22924 于 11:02:52Z 启动，启动前空闲显存 14757 MiB。回执 `plan/launch_receipt.json`。

## 预注册判据

参照为 stage1 固定矩阵中已验证的原 NDR-Net 30 轮值，Alameda 10.745627126021652，ContraCosta 11.27520497996187，同一验证集合、同一配方与同一运行时，不重跑。候选必须在两个地区的独立验证 MAE 都严格低于参照才算通过，通过后做 seeds 1、2 匹配确认；任一地区不满足即 No-Go，不以平均抵消。分析脚本会先核对候选与参照的验证有效点数、样本选择、目标与 mask 哈希逐项相同，配方字段相同，再读数值。候选多读了 occupancy 与 speed 历史，报告不能称其与只用 flow 的方法为同资料比较。不读 test，不读 Orange，不停止用户程序。

## 不能据此声称的事

CPU 测试与 smoke 只是工程证据，没有精度、收敛、泛化或 SOTA 结论。即使通过，单种子开发区结果也只构成进入多种子确认的资格。

## 5080 运行中断与迁往 AutoDL

ContraCosta 训练 worker 于 11:02:52Z 启动，第 1 轮训练 641.98 秒，完成验证并写出 best.pt 与 last.pt。第 2 轮跑到第 100 批后，该进程于 11:15:29.845Z 以退出码 1 结束，墙钟 757.15 秒。stderr 日志为 0 字节，没有 Python traceback，控制器的终止方式记录为空，说明进程是被外部终止而不是程序出错。时间与用户随后发出的停止在 5080 上训练的指示吻合，具体由谁结束进程没有记录。控制器按规则记为 `failed_nonzero_exit`，以 `failed_stopped` 结束，其余三个作业为 `not_started`。11:18:23Z 主机上启动了用户程序 `gpu_solve.py`，本工作只读了它的命令行，没有触碰。

13 个文件已取回到 `plan/retrieved_failed/`，传输前后源端清单逐字节相同，本地哈希一致。关闭记录 `plan/closure_5080_20261103.json`。第 1 轮验证值只是中断曲线点，不构成结果。last.pt 绑定 5080 的运行时，不在其他主机恢复。没有在 5080 上重启、恢复或派发任何作业，两个无触发器计划任务保留在已注册状态，控制器拒绝已有状态文件，不会再次运行。本地后台轮询已停止。

用户要求改用 AutoDL GPU。迁移包位于控制目录 `autodl_bundle/`：48 文件源码包与上文相同，控制器为同一 `bddc1e22` 文件，数据只含 Alameda 与 ContraCosta 的 samples、traffic 与 event relation 文件、正式缓存清单和两个区域图，共 9 个文件 916,796,227 字节。relation 文件训练不读，但其哈希进入区域缓存摘要，缺少时通不过正式清单绑定。从只含这 9 个文件的目录布局出发，两个地区都通过了清单哈希绑定并完成限批 CPU smoke，数据身份与 5080 校准时的区域缓存摘要、产物哈希、气候态数组与计数逐字节相同，只有 float64 归一化统计量因 numpy 版本不同相差 1 到 12 个 ulp。上传脚本在远端逐文件校验 SHA-256，远端准备脚本在本地模拟目录中完成了包校验、48 文件回读、最小 Git 与两个入口的 help 检查。新的计划作者脚本生成 Linux 路径的计划，可选同机原 NDR-Net 参照；含参照的测试计划已由同一控制器准入通过，且准入没有创建任何目录。

5080 的校准与冻结超时只适用于 5080 与当时的 GPU 负载，AutoDL 上需要重新做阶段 0 校准。stage1 参照值来自 5080，若 AutoDL 上只跑候选，结论属于跨主机比较。建议在同一 AutoDL 主机上用未修改的 `train_ndr_research.py` 同时训练原 NDR-Net，并把判据改为候选在两个地区同时严格低于 stage1 参照与同机参照；该修改须在任何 AutoDL 结果之前写入冻结计划。

## 用户改回 5080：停止挖矿与第二次正式运行

用户随后决定继续使用 5080，并要求先停止挖矿任务。只读核查显示，挖矿由系统级计划任务 `\PRL-Miner` 以 SYSTEM 身份运行 `C:\Mining\start.bat`，脚本会把 GPU 功率限制为 250 W 并在矿工退出后 30 秒内重启它；该任务只有开机触发器。11:46Z 从高完整性级别的 SSH 会话运行了用户自己的 `C:\Mining\stop.bat`，它结束该任务并终止 `start.bat` 循环与 `SRBMiner-MULTI.exe`。20 秒后矿工进程为 0，任务回到 Ready，GPU 利用率由 99% 降到 4%，功耗由 250 W 降到 51 W，空闲显存 15434 MiB。任务定义、开机触发器与 250 W 功率上限均未改动，运行 `C:\Mining\resume.bat` 即可恢复挖矿。回执 `miner_stop/stop_receipt.json`。

启动前在源端复核源码根，48 个文件逐字节一致，最小 Git 仍为 `2436aa87…` 且 clean。第二次正式计划 `plan_r2/formal30_multichannel_r2_plan.json`，digest `fc6a156aed7dbb311bb926b51ebf57b58f038e11761b7a851684b40bc80a6cce`，作业、命令、配方与预注册判据与第一次相同，只换了新的运行根。训练超时沿用挖矿负载下的阶段 0 校准，ContraCosta 36179 秒、Alameda 73612 秒，作为上界，使挖矿若被中途恢复也不会误触超时。总预算 118800 秒，绝对截止 2026-11-04T20:47:49.850199Z。admit-only 返回 0 后以无触发器计划任务 `XTraffic_NDR_Multichannel_Formal30_R2_20261103` 启动，控制器 PID 13080，ContraCosta 训练 worker PID 4336 于 11:48:23Z 启动，启动前 GPU 利用率 4%、空闲显存 15435 MiB。回执 `plan_r2/launch_receipt.json`，后台只读轮询记录在 `plan_r2/monitor.log`，并记录矿工进程是否重新出现。

用户随后说明训练期间不需要 250 W 上限，该上限只用于挖矿，并要求训练结束后恢复挖矿。12:06Z 在高完整性级别的 SSH 会话中执行 `nvidia-smi -pl 360`，把功率上限恢复为该卡的默认值 360 W，生效值确认为 360 W。该设置在重启或驱动重置后失效，`start.bat` 每次启动矿工都会重新设为 250 W。功率上限只影响速度，在确定性算法下不影响数值结果。后台轮询已换为 `monitor_formal30_r2_resume.sh`：控制器一到终态，且本运行的进程全部退出后，它运行用户的 `C:\Mining\resume.bat`，并核对矿工已启动、功率回到 250 W；本运行仍有进程存活时不恢复。回执 `miner_stop/power_and_resume_receipt.json`。去掉上限后的前 4 轮 ContraCosta 每轮约 4.75 分钟，与 stage1 参照的无竞争速度一致。
