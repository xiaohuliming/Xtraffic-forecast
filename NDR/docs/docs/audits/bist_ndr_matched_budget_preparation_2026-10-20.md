# BiST、BiST-NDR 与 NDR-Net 同预算公平对照的 CPU 工程准备

日期：2026-10-20，Asia/Singapore。
范围：protocol_v3_noleak_dedup event 协议，Alameda 与 ContraCosta 开发区，seed 42。
状态：代码与 CPU 验证完成，成本校准计划已写出，没有冻结预算，没有创建、启动或恢复任何 GPU 任务。

## 目的与前因

2026-10-20 完成分析的官方重叠协议 18 项配对矩阵显示，NDR 插件装到 BiST 骨干上，在 30 轮匹配契约下三地区九种子全部严格优于 BiST 匹配对照。该对照是被截断的 BiST，同契约下 BiST-NDR 仍略高于 STAEformer 版 NDR-Net，因此需要在 v3 event 协议下让 BiST、BiST-NDR 与 NDR-Net 共用同一个更长预算做公平对照。本轮只做 CPU 工程准备与成本校准计划，按用户要求不自动启动 GPU。

## 代码基线的选择

5080 上实际生产的强对照入口是研究线的 `scripts/train_ndr_strong_reference.py`，它带完整断点、独立进程验证与校准模式，strong4 计划就是用它以 `--method bist --recipe native` 运行。因此本轮在该入口上增加 `bist_ndr` 方法，而不是改旧的 `train_protocol_v3_event_extension.py`。NDR-Net 一臂继续使用研究线的 `scripts/train_ndr_research.py --variant baseline`，它已支持任意正整数 epochs 与完整断点。

新代码位于独立分支 `codex/bist-ndr-matched`，工作树 `/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`，基于 `codex/ndr-transfer-training` 的 `5f531400d6a50ad60ad5db39add8f6f6681921f7`。主仓库位于 iCloud 同步的 Desktop 目录，松散 Git 对象在新 checkout 时反复出现 mmap 超时，因此该工作树以 `--no-checkout` 创建，再从同一提交的已落盘兄弟工作树复制文件并重建索引，`git status` 在改动前为 clean。三个文件已于同日窄提交为 `1a718c6f0a4d60801ceae8f9207acc00d9d4cd95`，未推送。

## 改动内容

`ndr_research/strong_references.py`：

- `METHODS` 增加 `bist_ndr`。新增 `StrongReferenceNDR`，内部持有 `scripts/reproduction/igstgnn_table4/bist_ndr_pair_model.py` 中的 `BiSTNDR` 适配器，即 18 项配对矩阵实际训练过的那个类：共享 BiST 编码，从原生回归层与 decoder 复制出正常头与扰动头，原 36 参数逐步门控，正常锚点为仅 train 拟合的周内中位数，扰动锚点为最近观测。锁定 BiST 身份、原生 engine 与 test loader 均未创建的检查沿用原 `_native_bist`。
- `forward(flow_hist, time_feat, x_baseline, y_baseline)` 校验形状、有限性与归一化日历后调用插件，输出原始流量单位。`factory_kwargs` 记录 `residual_std` 与逐节点 `residual_std_by_node`，checkpoint 可据此重建。
- `reference_recipe` 接受 `bist_ndr`。`native` 配方与 `bist` 共用锁定 BiST 公开 profile 的优化器：Adam 学习率 0.002、weight decay 1e-5、eps 1e-8、裁剪 5、MultiStepLR 里程碑 30、60、80、100、gamma 0.5、按 epoch 步进、batch 64。`matched-ndr` 配方沿用 NDR 的 batch 16 与 cosine。配方新增 `plugin_loss`，固定 lambda_gate 1.0 与 lambda_specialization 0.3，并记录损失源码 SHA；新增 `comparison_role` 标注 backbone_control、plugin_candidate 与 raw_backbone_control。`early_stopping` 保持 false。

`scripts/train_ndr_strong_reference.py`：

- `strong_forward` 对 `bist_ndr` 传入两个锚点。`backward_reference_batch` 对插件臂调用原 `compute_loss`，并在激活 microbatch 之前用 `build_ndr_loss_normalizers` 在完整有效批上固定门控蒸馏与专门化损失的分母，与已验证的配对 runner 和原 s0g microbatch 契约一致。原生 BiST 与 STAEformer 的损失路径未变。
- `optimize_reference` 从配方 `plugin_loss` 读取权重。`main` 在新建 `bist_ndr` 运行时把 train-only 气候态统计写入模型 factory。
- `source_snapshot` 显式加入 `baselines/GraphWaveNet/model.py`，因为插件损失模块经 `fourier_dual_net/rgdn.py` 的裸 `import model` 到达该文件，AST 闭包看不到它。该改动会改变所有新运行的 snapshot digest，旧 checkpoint 不能用新代码恢复，这是预期行为。

## CPU 验证

新增 `tests/ndr_research/test_bist_ndr_strong_reference.py` 共 11 项，与既有 27 项强对照测试一起运行，38 项通过，Ruff 通过。JUnit 回执 SHA-256 为 `a1b5132a562e5e4ec8d6e4bb8fd39a91a584a56a390fd76bd197e8d628a2881a`。新测试覆盖：

1. 包装器确实是锁定 BiST 上的 `BiSTNDR`，门控、双头与 oracle 监督标志为真，附加参数数等于总参数减原生参数。
2. 包装器前向与直接构造的配对适配器在相同权重下逐位相等，扰动锚点等于最近观测，正常锚点等于未来气候态。
3. 在确定性训练模式下，新实现的目标值与梯度和已验证配对 runner 的 `backward_bist_ndr_pair_batch` 一致，相对误差 1e-6。
4. microbatch 1 与 2 的目标值、三个损失分项与梯度一致，证明分母来自完整有效批。
5. 空有效批不更新任何参数或统计量。
6. `bist_ndr` native 配方与 `bist` native 配方的优化器字典相等，30 与 100 轮均不早停。
7. checkpoint 保存、加载、恢复后再训练一步与原进程逐位相同，factory 含逐节点尺度。
8. 验证累计全部有效点并保留尾批，CPU 校准不输出任何分数且 `cuda_budget_eligible` 为 false。
9. 原生控制类拒绝 `bist_ndr`，插件类拒绝错误形状或非正的尺度。
10. CLI 接受 `--method bist_ndr --epochs 100`，snapshot 声明插件适配器、损失源码与 GraphWaveNet 模型文件。

真实 ContraCosta v3 缓存上的 CPU smoke 使用 batch 4、1 轮、各 2 个训练与验证批：

| 检查 | 结果 |
|---|---|
| `bist_ndr` native 训练 smoke | 完成，best epoch 1，2 次更新，验证 MAE 14.983754 |
| 独立进程 `--mode verify` | `verified_smoke_only`，训练 PID 71465，验证 PID 71494，MAE 逐值复现 |
| `bist` native 训练 smoke | 完成，2 次更新，验证 MAE 66.906506 |
| `bist_ndr` CPU 校准路径 | `calibrated_cpu_engineering_only`，4104 个训练批与 868 个验证批，不输出分数 |

smoke 的 `claim_scope` 为 `smoke_only`，数值只证明代码路径可运行，不是精度证据。两个 smoke 的 MAE 差异来自 2 次更新后锚点初始化的差别，不能解读为方法比较。插件附加参数在 ContraCosta 为 37,692。

## 公平对照契约

- 三臂共用 100 轮上限、不早停、每轮完整验证、按全有效点 validation masked MAE 选 checkpoint、FP32、3 线程、保留尾批、训练后独立进程复验。
- `bist` 与 `bist_ndr` 共用锁定 BiST 原生优化器契约，两者差异只在插件。
- `ndr_net` 使用原 s0g 配方：Adam 学习率 0.001、weight decay 1e-4、裁剪 5、按更新步 cosine 退火到 1e-5、batch 16、microbatch 8。100 轮 cosine 是新的一臂，不能用已验证的 30 轮数字 10.745627 与 11.275205 代替。
- `bist_ndr` 对 `ndr_net` 是同预算的两种完整方法比较，不是单因素比较，报告时必须如此表述。

预注册判据：R1，`bist_ndr` 在 Alameda 与 ContraCosta 的完整验证 MAE 都严格低于 `bist`，一个地区退化即 No-Go，不以平均抵消。R2，`bist_ndr` 在两地区都严格低于同预算 `ndr_net` 才进入确认资格，否则记为中性或负结果。R3，R1 与 R2 通过后才做 seeds 1、2 匹配确认与 Orange 保留地区确认，test 保持关闭。R4，披露 best epoch、完整验证曲线、参数量、运行时间与峰值显存，best epoch 为 100 视为截断。R5，任何数字不构成 SOTA、泛化或多种子结论，官方重叠协议配对数字不与 v3 表混用。

## 成本校准计划

阶段 0 在 5080 上运行六个校准任务，`bist` 与 `bist_ndr` 使用 `train_ndr_strong_reference.py --mode calibrate --epochs 100 --batch-size 64 --microbatch-size 64 --warmup-batches 3 --timed-batches 20`，`ndr_net` 使用 `train_ndr_research.py --mode calibrate --epochs 100 --batch-size 16 --microbatch-size 8 --warmup-batches 5 --timed-batches 20`，两地区各一次。校准输出完整批的均值与 p90 秒数、峰值显存与 100 轮外推秒数，不读取验证分数。冻结规则为训练超时等于 1.5 倍 p90 外推加每轮断点写盘开销加 600 秒，验证超时 600 秒，busy wait 600 秒，最低空闲显存 12000 MiB。

既有实测只用于粗估：

| 臂 | 来源 | Alameda | ContraCosta |
|---|---|---:|---:|
| BiST native 100 轮 | 2026-10-15 cost18 CUDA 校准外推 | 5144.5 秒 | 2571.4 秒 |
| NDR-Net 30 轮 | 2026-10-13 CUDA 校准外推 | 4.75 到 5.02 小时 | 2.26 到 2.43 小时 |
| NDR-Net 100 轮 | 上行按 100/30 线性放大 | 15.8 到 16.7 小时 | 7.5 到 8.1 小时 |
| BiST-NDR 100 轮 | 未在 CUDA 测过 | 暂按不超过 BiST 的 1.5 倍 | 同左 |

串行总训练约 29 到 30 小时，再加校准、六次验证与断点开销，乘 1.5 余量约 45 小时。建议拆成两个独立冻结计划：A 为两地区 BiST 与 BiST-NDR，含余量约 6 小时，回答插件在原生 100 轮配方下是否改善 BiST；B 为两地区 NDR-Net 100 轮，含余量约 36 小时，提供同预算的领先者参照。

## 派发前的准入门禁

1. strong4 首项异常关机的 forensic 保全完成并关闭该计划，其绝对截止 2026-10-17T07:55:57.682187Z 已过。
2. 主机空闲：至少 12000 MiB 空闲显存，无训练进程，不停止任何用户程序。
3. pristine BiST 身份核验，commit `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`，tree `1dc3c4de312ec8cb2eaf2aafc7b680aa247d19be`。
4. 从 `codex/bist-ndr-matched` 的 clean 提交打包源码并在源端逐字节独立回读。
5. 阶段 0 CUDA 校准完成且 `cuda_budget_eligible` 为真，再冻结超时、绝对 UTC 截止与计划 digest。
6. 既有 `ndr_strong4_development_queue.py` 只接受八任务 strong4 schema，本矩阵需要新的或参数化的串行控制器及其 CPU 测试。
7. Mac 取回 collector 与逐任务独立验证绑定。
8. 用户明确授权启动 GPU 工作。本草案不创建、调度或恢复任何 GPU 任务。

## 产物

- 代码：`/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`，分支 `codex/bist-ndr-matched`，窄提交 `1a718c6f0a4d60801ceae8f9207acc00d9d4cd95`，未推送。`scripts/train_ndr_strong_reference.py` SHA-256 `7e443e3bd39b57ba661c2999d72b5357be59045e96a737764ba38ab83bc9537d`，`ndr_research/strong_references.py` `c55f5caba175239c7b39ef808cc624201dbd3bc080b96aca9a7848d5d74dc733`，`tests/ndr_research/test_bist_ndr_strong_reference.py` `243ce2821adb4bd8642c8d9c39eaaa1ecdb4944d800a58b8d8c09e75c666ec47`。
- 控制目录：`outputs/remote_runs/ndr_generalization_20261013/control/bist_ndr_matched100_preparation_20261020/`，含草案 `matched100_development_plan.draft.json`，SHA-256 `f64712d22b5d399ada62eb6c64be3fbfa02161a8e4d569864831b65418086e89`，JUnit 回执，源码哈希清单，三个 CPU smoke 的 config、summary、verification 与 calibration 原件，不含 checkpoint 字节。
- 该草案 `production_launchable` 为 false，`budget_frozen` 为 false。

## 不能据此声称的事

没有 CUDA 可行性、精度、收敛、泛化或 SOTA 结论。CPU smoke 与 CPU 校准只是工程证据。18 项配对矩阵的 6% 到 9% 改善属于官方重叠协议与 30 轮截断对照，不是本计划的预期值。

## 2026-10-20 15:11 执行阶段：校准完成与计划 A 启动

用户随后要求不再逐步确认，由本会话自行规划执行。三个文件已窄提交为 `1a718c6f0a4d60801ceae8f9207acc00d9d4cd95`。参数化串行控制器 `scripts/remote/windows/ndr_matched_budget_serial_queue.py` 与 82 项 CPU 测试由子代理完成，Ruff 通过，窄提交为 `b4d469ef3b4a94f3057c8326d1d2fd33fca75496`，说明见 `docs/audits/ndr_matched_budget_controller_2026-10-20.md`。

strong4 首项崩溃现场 41 个文件已只读复制到 `control/strong4_crash_closure_20261020/`，逐文件哈希一致，第 19 轮验证值只是中断曲线点。该计划已按截止过期关闭，不恢复，见同目录 `Root_closure_20261020.json`。

46 个文件的源码包以 `1a718c6` 为来源，ZIP SHA-256 `2dd7daf8b64798c078c4db2f20edb2cf1da4d7112d53220cca1b5e1ade990f81`，在 5080 新根 `C:/XTrafficResearch/bist_ndr_matched_20261020/prepared_1a718c6_2dd7daf8/repo` 解包，46 个文件逐字节回读一致，源端最小 Git 为 `e2b855860ed0162a8cc06c169667dc3d80cb8686`。pristine BiST 仍为 `dd94adf` 且 clean。两个训练入口在源端只做 help 导入检查，没有读取数据。

阶段 0 六项 CUDA 校准由控制器以无触发器计划任务 `XTraffic_BiSTNDR_Stage0_Calibration_20261020` 串行执行，计划 digest `385bf23eff70ecfbb3d53186340317b08371db49c1a693111a01b6b6b975c2f0`，124.6 秒内全部成功，38 个文件已取回并逐文件复验。100 轮 p90 外推：

| 臂 | Alameda 秒 | ContraCosta 秒 | 训练峰值显存 |
|---|---:|---:|---:|
| BiST native | 6778.9 | 3321.8 | 1.71 GB |
| BiST-NDR native | 7429.0 | 3776.5 | 1.78 GB |
| NDR-Net baseline，均值外推 | 68533.8 | 34090.3 | 4.49 GB |

计划 A 已冻结并启动：16 个作业，BiST 与 BiST-NDR 在原生 BiST 配方下各跑 30 轮与 100 轮，两地区，每个训练后独立进程验证。顺序为 ContraCosta 30 轮对、Alameda 30 轮对、ContraCosta 100 轮对、Alameda 100 轮对。训练超时按 1.5 倍 p90 加每轮 15 秒加 600 秒冻结，八个训练超时合计 54151 秒，验证各 600 秒，总预算 72000 秒，绝对截止 2026-10-21T03:11:19.215139Z，busy wait 600 秒，最低空闲显存 12000 MiB。计划 digest `c4bbad06168187893ffa274d299f70ca1875a492c4501ab75dcb737bc87a99f2`，源端 admit-only 返回 0 后以计划任务 `XTraffic_BiSTNDR_PlanA_NativeBiST_20261020` 启动，控制器 PID 29876，首项 ContraCosta BiST native30 的 worker PID 33788 已在 07:11:26Z 启动。预期训练总时长约 7.6 小时，不含验证与等待。

30 轮 BiST-NDR 可与已验证的 NDR-Net 30 轮开发值 10.745627 与 11.275205 做描述性对照，两者数据与验证集合相同但配方不同。NDR-Net 100 轮参照约需 28.5 小时，作为计划 B 待计划 A 结果后决定。全部数字仍限 seed 42 开发区，不读 test 或 Orange。

## 2026-10-21 结果与关闭

计划 A 的 16 个作业于 2026-10-20T13:49:10Z 全部成功，控制器预算用时 6.63 小时，没有超时、失败或 GPU 等待。122 个文件已取回，传输前后源端清单逐字节相同，本地哈希全部一致。预注册分析见 `docs/reports/BiST_NDR同预算对照结果_2026-10-21.md`。

30 轮下 BiST-NDR 在两地区严格优于原生 BiST，100 轮下 Alameda 只剩 0.50% 改善，ContraCosta 退化 1.20%，R1 在 100 轮失败。没有任何一臂在两地区同时低于 NDR-Net 30 轮开发值。按预注册规则不进入 R2、seeds 1、2 或 Orange，计划 B 不启动。计划 A 结束后没有新的 GPU 工作。
