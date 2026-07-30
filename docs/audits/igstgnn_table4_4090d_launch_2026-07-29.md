# IGSTGNN Table 4 双机九卡启动进度报告

更新时间：2026-07-29 10:57 新加坡时间

## 当前结论

IGSTGNN Table 4 复现已从 5080 单卡 pilot 切换为两台 RTX 4090 D 主机。两台主机共有 9 张 24 GB GPU。正式协议仍为 `igstgnn_table4_official_v1`，正式矩阵仍为 `igstgnn_table4_public_repro_39_v2`。本轮结果只能称为 `public_material_maximum_fidelity_reproduction`，有效性标签仍为 `official_protocol_with_known_overlap`。

39 项固定分片清单位于 `config/reproduction/igstgnn_table4/distributed_4090d_9gpu_v1.json`，SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。五卡机承担 Orange 的 13 项，四卡机承担 Alameda 与 Contra Costa 的 26 项。每个地区的全部方法保留在同一台主机上，地区内直接比较不会跨主机。

原九卡 runner 已停止在可审计的失败保护状态。已完成并逐任务验收 21/39，四卡机为 13/26，五卡机为 8/13。两台机器均在线，GPU 在恢复启动前为空闲。结果服务器尚未收到正式主机结果。

正式恢复控制器已于 10:57 启动。四卡机 PID 为 `39176`，当前执行 task005 Alameda STGCN。五卡机 PID 为 `40326`，当前执行 task031 Orange STGCN。恢复层不修改 pristine 上游源码或正式数据，并对每个输入语义或硬件适配记录独立 SHA-256 provenance。

## 固定身份

| 项目 | 值 |
|---|---|
| 正式矩阵 SHA-256 | `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78` |
| 九卡分片计划 SHA-256 | `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc` |
| 正式数据文件数 | 27 |
| 正式数据总字节数 | `9051956107` |
| 数据哈希清单 SHA-256 | `bcf88f2902d900ce9baed5c33b8d064637dce1a3d9498994c7db5eb855d2d831` |
| 初始双机控制提交 | `0ce8b19828128ea7d0272f2e594322b288f2a30a` |
| 多卡 IGSTGNN 适配提交 | `f43fb58` |
| 自动恢复控制提交 | `25b5307` |
| 多卡 IGSTGNN 训练代码 SHA-256 | `641b0ac88078f33b3368cbc7fa352ad60c9238493fb40537e82bbf28075f28ad` |
| 最终恢复控制提交 | `af6456afc33106db0082c3635c524fb30c70c157` |
| 恢复 runner SHA-256 | `52829dd65362ef03127f391b7054aff1c0feeacf011bc4af974e2be408f07a76` |
| 持久恢复脚本 SHA-256 | `23980df3dfe42afe952bc52febb97e3fd85e1c1629b86e70b2b2679b3c8a5b4b` |
| 本地全仓回归 | 446 passed，2 个既有 warning |
| 两台 Linux 训练相关门 | 各 158 passed、12 skipped、1 个既有 warning |

LargeST、BiST 与 IGSTGNN 的 pristine 上游提交继续锁定为 `b7c6ff5f7f0c10051ed10a319b2cfe58021312f9`、`dd94adf7721fcbb9e3feb5d1b44040305199a4cc` 与 `64c6e243466f791f1b75e50421d63e39577ec9fa`。两台主机的三个 checkout 均通过 clean worktree 与 `git fsck` 检查。

## 数据部署与启动门

正式数据先传到五卡机，再通过两台云主机之间的链路续传到四卡机。两端均重新计算 27 个文件的 SHA-256，文件数和全部摘要一致。首次同步保留了 Mac 的 UID 501。训练入口的所有权安全门在加载数据前拒绝了全部 9 个 shard，退出码均为 3，没有 trainer、正式任务目录或模型结果。

两台主机的首次控制目录、queue 和日志均作为 incident 保留。五卡机 incident 清单 SHA-256 为 `5fa184ce5c9f37acf3faf63d33c88117d403ab1c038a312423efa4bc5ef58346`，四卡机为 `d6d29a5f2e5b0a14c2fd1a989ee25af7aa86f6e6a11f39a8c250cd2b4272270b`。数据所有权修正为 root 后，两端再次完成逐文件哈希校验，数据内容没有变化。

五卡机正式重启时间为 2026-07-29 00:47:14 新加坡时间，supervisor PID 为 `17208`。四卡机正式重启时间为 00:47:32，supervisor PID 为 `16728`。

## 5080 切换记录

5080 的旧 pilot 在新双机矩阵通过数据门后停止。停止时 task003 已完成 epoch 38，但没有 task003 正式目录。Windows 任务计划 `XTraffic_IGSTGNN_Table4_v2` 已禁用，相关 Python 进程为 0。旧 Alameda HL 与 LSTM 正式结果保持只读，不进入本轮双机九卡正式导出。

5080 incident 位于 `D:\t4\incidents\table4_003_5080_superseded_20260728T163820Z`。incident 清单 SHA-256 为 `29c58bad6635ae3c7d476ec875cd577153ab4a95981d32f2a16baf89248bde4a`。live 接力状态标记为 `superseded`，旧 pilot queue 标记为 `interrupted`。

## Orange IGSTGNN 显存事件

Orange IGSTGNN 在默认 CUDA 分配器下于首个 forward 尝试分配 2.10 GiB 时发生 OOM。改用 `cudaMallocAsync` 后仍在同一计算阶段发生 OOM，证明问题不是内存碎片。两次尝试都没有形成正式任务目录。

默认分配器 incident 清单 SHA-256 为 `799911d2315cd1a886a20ce181d20ec3fea97fddf9293cfa3c5870bb25828e27`。第二次分配器诊断保留在五卡机控制日志与 gpu4 queue 中。它们属于硬件容量诊断，不是模型性能失败。

提交 `f43fb58` 增加显式 `--igstgnn-data-parallel` 入口。它保留完整 batch，由单一 loss 与优化器处理汇总输出。上游模型的未注册图张量在副本建立时复制到对应设备。pristine 上游目录保持未修改。此适配属于执行层差异，最终报告不得写成单卡逐命令复现。

双卡恢复在首批反向传播时再次 OOM，只差 270 MiB，没有形成正式 task039。该记录证明 Orange IGSTGNN 需要超过两张 24 GB 卡。

## 21/39 失败保护快照

四卡机完成 13 项，五卡机完成 8 项。21 项均有五份正式 artifact，并由原 runner 逐任务验收。四卡机 65 个文件的恢复前 SHA-256 清单摘要为 `3e89aac6c8b5ebbf2bc8d2df4cc3f518587418464d9a49fd8b349e93967f4513`。五卡机 40 个文件的对应摘要为 `ee52c8e11ba150c3d49b65cf8d0efe480e46df0f5134ff310b3de14a440c1c61`。

7 个实际失败任务是 Alameda STGCN、D2STGNN、IGSTGNN，Orange STGCN、DGCRN、D2STGNN、IGSTGNN。其余 11 项没有启动，因为 matrix runner 对每个 shard 执行 fail closed。

STGCN 的发布邻接矩阵已经是零对角线。LargeST STGCN 入口又减一次单位矩阵，使节点度数出现负值，缩放拉普拉斯包含 NaN。上游 masked 指标会把 NaN loss 清零，所以训练没有及时失败，直到正式预测有限值检查才被拦截。恢复层在输入已经无自环时临时加入单位矩阵，让原入口的减法只执行一次。最终送入图归一化的无自环图与发布图完全相同。

D2STGNN、DGCRN 与 IGSTGNN 的其余错误为 24 GB 物理容量。DGCRN 还暴露 PyTorch 2.3 在多个 DataParallel 线程首次并发初始化 `torch.linalg.qr` 时的懒加载竞态。恢复层在建副本前按 GPU 串行执行无状态 QR 预热。

## 正式恢复适配

提交 `b266c89` 建立独立恢复 runner，提交 `05e2951` 增加 DGCRN QR 预热，提交 `1a5582f` 把较小余数分片分给主卡，提交 `af6456a` 增加 Orange D2STGNN 的等效全批微步。

Orange D2STGNN 无法把 batch 64 一次分给五张 24 GB 卡，因为至少四张卡需要接收 13 个样本。正式恢复把全批拆成两个 32 样本微前向。每个微批在五卡间分为 6、7、7、6、6。两个微批按全批有效观测点数加权并累积梯度，之后只执行一次梯度裁剪和一次优化器更新。每个原始 batch 仍只推进一次 curriculum。该流程保持有效 batch 64，不把 microbatch 当成两个训练 step。

真实远端探针结果如下。

| 任务 | 执行方式 | 峰值显存 |
|---|---|---|
| Alameda STGCN | 单卡，幂等自环移除 | 约 1.1 GiB |
| Alameda D2STGNN | 四卡完整 batch | 每卡约 9.9 GiB |
| Alameda IGSTGNN | 四卡完整 batch | 每卡约 7.6 GiB |
| Orange DGCRN | 五卡完整 batch，QR 预热 | 每卡约 5.9 GiB |
| Orange D2STGNN | 五卡，两次 32 样本微前向 | 最高约 13.5 GiB |
| Orange IGSTGNN | 五卡完整 batch | 最高约 10.0 GiB |

所有探针都执行了真实前向、反向和优化器更新。探针使用临时目录，没有发布正式 task artifact。

## 自动回传与关机

两台主机完成各自固定 shard 后，会执行 source-host postflight。postflight 要求 queue 完成、任务和文件 allowlist 精确、五份 artifact 齐全、运行时主机身份一致、GPU 型号为 RTX 4090 D，并按训练代码家族验证 digest。

通过 postflight 后，finalizer 只上传 allowlist 文件到 `root@8.134.12.5:/root/xtraffic_results/igstgnn_table4_official_v1/distributed_4090d_9gpu_v1/hosts`。结果服务器会逐文件重新计算 SHA-256、核对精确目录树并原子提升。只有正式目标再次完整复验通过后，源主机才调用 `/usr/bin/shutdown -h now`。postflight、容量、传输、远端哈希或原子提升任一步失败时，主机保持开机。

## 17:05 进度与 NDR-Net 扩展

截至 2026-07-29 17:05 新加坡时间，原 39 项矩阵已正式落盘并由任务 runner 验收 34 项。四卡机为 25/26，仅余 task021 Contra Costa STTN。五卡机为 9/13，当前为 task036 Orange DGCRN，后续仍需 task037 Orange D2STGNN、task039 Orange IGSTGNN 与 task034 Orange STTN。结果服务器仍没有正式主机目录。

原矩阵本身不增加方法或任务。NDR-Net 使用独立扩展矩阵 `igstgnn_table4_ndr_extension_9_v1`。它包含三地区与种子 42、1、2 的全部组合，共 9 项。扩展清单 SHA-256 为 `2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4`，配置 SHA-256 为 `6ca46be0fb76543a61a8bd32996c7598ac674bbe239b701d1a92481f63be9422`，分片计划 SHA-256 为 `f20be0db02fbd04732640cd5b024f7012ce6a16ea155d6a90c0962839f857522`。

四卡机承担 Alameda 与 Contra Costa 六项。五卡机承担 Orange 三项。扩展 claim scope 为 `same_protocol_project_method_extension`，validity label 为 `official_protocol_with_known_overlap`。它可以与原 39 项在同一官方协议表中并列展示，但必须明确标为项目方法补充行，不能写成 IGSTGNN 原论文的第 14 个公开方法。

官方 split 中的 flow 已经标准化。扩展入口使用同一 split manifest 绑定的 `incident_stats.npz` 先逆变换到原始流量单位，再拟合训练集周内时段基线、计算损失和评测。基线只使用 train split 的 x 与 y 位置。验证 MAE 使用全部有效观测点的误差总和除以有效点总数。测试输出沿用官方 H3、H6、H12 与 12 步算术平均。

实现提交为 `cd14b90`、`d67ba1c` 与 `085e68d`。Mac 真实 Contra Costa 数据适配耗时约 4.4 秒，峰值内存约 1.42 GiB。MPS 前向反向探针输出为 `16×496×12`，loss 为 `37.98567963`。本地全仓为 455 passed 和 2 个既有 warning。两台 Linux 主机的扩展专项测试各为 33 passed。

扩展代码部署在独立 `/root/autodl-tmp/table4/ndr_repo`，所有新增与修改文件均与 Mac 逐文件 SHA-256 一致。该目录不改写原 39 项仍在使用的 `/root/autodl-tmp/table4/repo`。原 finalizer 已部署关机延期哨兵。第一次扩展控制器因虚拟环境 `python` 是正常符号链接而被过严安全门拒绝，没有 trainer、正式结果、传输或关机动作。链接解析检查修复后，四卡机 PID `447869` 与五卡机 PID `410737` 已进入等待原矩阵回传完成的状态。

每台主机开始扩展正式分片前会先在空闲卡上运行最大地区的真实 CUDA 前向反向探针。任何探针、训练、postflight、传输或结果服务器复验失败都会保留关机延期哨兵并保持主机开机。只有扩展独立结果树在结果服务器逐文件复验和原子提升成功后，控制器才移除延期哨兵并调用 `/usr/bin/shutdown -h now`。

## 17:25 四卡机原结果发布与 NDR 正式启动

四卡机于 17:11 完成原矩阵 26/26。source-host postflight 验证 134 个 allowlist 文件，files digest 为 `a43a4023cb7460fcd2ad2d2cece0adc01c7394588df206461744e67ec94550dd`。首次 finalizer 已通过 postflight，但默认 BatchMode SSH 没有可用回传身份，因此在传输前以鉴权失败停止。任务目录、summary、prediction、checkpoint、train log 与 postflight 均未被改写，延期哨兵保持完好，主机保持开机。

双机各自生成了仅用于结果服务器的 Ed25519 身份。公钥按幂等方式加入结果服务器，私钥只保存在对应主机控制目录。双机到结果服务器的 key-only SSH 均已真实验证。四卡机使用显式身份重新运行 finalizer，于 17:18 完成 1,254,043,651 bytes 与 134 个文件的上传，于 17:19 完成结果服务器逐文件 SHA-256、精确目录树和原子提升，随后记录 `FINALIZE_SUCCESS shutdown_deferred`。提交 `ce7a8c2` 将该身份路径设为原 finalizer 的默认值，专项 shell 测试覆盖成功关机、延期关机、远端复验失败与 postflight 失败。修改后的脚本已在两台主机逐文件 SHA-256 对齐。

四卡机 NDR 首次 CUDA 探针发现官方复现环境未安装 `h5py`。NDR 模型本身不读取 HDF5，依赖来自 `scripts/train_staeformer_deseason.py` 的旧入口顶层导入。探针在正式任务目录和 trainer 建立前失败，记录为 fail-closed incident。提交 `77d8815` 将只服务旧入口的数据集、loader、运行上下文和训练统计导入移入 `main()`，并新增阻断 `h5py` 时模型模块仍可导入的子进程回归。Mac 全仓结果为 456 passed 和 2 个既有 warning，两台 Linux 主机的真实模型导入均输出 `NDR_IMPORT_OK`。

四卡机 NDR 控制器以 PID `456092` 重启。Contra Costa 的真实 CUDA 前向反向探针输出为 `NDR_EXTENSION_SMOKE_OK`，shape 为 `16×496×12`，loss 为 `37.95976257`，峰值分配显存为 8,128,185,856 bytes。探针于 17:24 通过后，控制器启动四个正式 shard runner。GPU 0 到 GPU 2 分别开始 Alameda seeds 42、1、2，GPU 3 开始 Contra Costa seed 42。17:25 四张卡利用率为 93%、97%、97%、93%，显存为 9,416 MiB、9,416 MiB、9,416 MiB、8,644 MiB。ndr004 的首个落盘 epoch 训练 loss 为 25.55628414，验证 MAE 为 12.62008389。

同一时刻，原矩阵总进度为 36/39。五卡机为 10/13，task036 Orange DGCRN 已完整落盘，task037 Orange D2STGNN 正在执行。task037 首个落盘 epoch 的训练 loss 为 27.5222，验证 MAE 为 21.0824。后续仍需 task039 Orange IGSTGNN 与 task034 Orange STTN。五卡机 NDR 控制器 PID `422803` 继续等待原分片完成、source-host postflight 和结果服务器正式复验。

## 00:18 四卡机接管 Orange NDR

2026-07-30 00:08 新加坡时间起，NDR 扩展从双机 6/3 分片切换为四卡机单主机 9 项重分配。新计划 ID 为 `igstgnn_table4_ndr_extension_4gpu_rebalance_v2`，SHA-256 为 `56a71fe2a3b8101ae282dc6f9b9efab07874108a40922276285206e99cbf2612`。它仍引用不可变 9 项清单 `2442f1f29b6e3dcef1d3f6999c0ce9dbf474e43644f98cd6f2aec765908db5b4`。六项 Alameda 与 Contra Costa 正式 artifact 保持在原四个 shard 中，runner 先逐项验收并跳过。GPU 0、1、2 随后分别执行 ndr007、ndr008、ndr009，GPU 3 在完成旧结果复验后退出。

首次 Orange 单卡探针使用原 batch 16，在 backward 尝试额外分配 2.80 GiB 时 OOM。探针没有创建正式任务目录、summary 或 prediction。提交 `87526d1` 增加精确有效 batch 适配。每个 batch 仍为 16，训练拆成两次 max microbatch 8 前向，loss 的有效点数、gate confidence、gate target 均按全批分母归一化，只执行一次梯度裁剪、一次 optimizer step 和一次 scheduler step。小模型差分测试确认 full batch 与两微批的总 loss 及全部参数梯度一致。

修复后的 Orange 真实 CUDA 探针输出 `NDR_EXTENSION_SMOKE_OK`，shape 为 `16×990×12`，loss 为 `42.45340157`，microbatches 为 2，峰值分配显存为 12,632,052,224 bytes。计划与适配提交为 `b67f30f` 与 `87526d1`。Mac 全仓回归为 458 passed 和 2 个既有 warning。正式 host controller PID 为 `3677`，启动时间为 00:08:27。00:18 的三张训练卡利用率为 98% 到 99%，显存为 18,686 MiB。三项均完成 epoch 1，validation MAE 依次为 13.19330061、13.06060743、12.91147725。

四卡机完成控制器 PID `5614` 正在等待 host status。它要求旧六项锁定清单的 30 个 artifact 逐文件哈希保持不变，然后重新运行完整 9 项 postflight。正式回传使用独立结果根 `ndr_extension_4gpu_rebalance_v2/hosts`，结果服务器必须通过逐文件 SHA-256、精确目录树和原子提升。全部通过后才移除四卡机的新关机 guard 并调用 `/usr/bin/shutdown -h now`。

五卡机旧 NDR handoff PID `422803` 在没有 Orange trainer 的等待态被定向停止。原矩阵进程未受影响，关机延期哨兵保持。incident 位于 `incidents/ndr_extension_waiter_rebalanced_20260729T161049Z`，清单摘要为 `556c4289e2b4ec41b35b40d1f9a72190e1ad1c3c5a55edf1986843a70aa6e8a1`。新关机等待器 PID `201649` 只等待五卡机原矩阵的 transfer marker 与 `FINALIZE_SUCCESS shutdown_deferred`，随后使用绑定密钥重新复验结果服务器正式树，成功后才移除哨兵并关机。控制提交为 `dd2d087` 与 `b5d9751`。

00:18 时原矩阵总进度为 37/39。五卡机已完成 11/13，task039 Orange IGSTGNN 已完成 44/100 epoch，之后只余 task034 Orange STTN。五卡机原矩阵未完成回传，最终 Table 4 仍不得生成。

## 09:21 task034 跨主机四卡恢复

四卡机 NDR 扩展随后完成 9/9，并在结果服务器完成逐文件复验和原子提升。正式目录含 49 个文件和 940,488,092 字节，files digest 为 `ab21c8e300ba930e5611cb9c07ff5b047eb03d281ca669c756242707231c87e7`。task039 Orange IGSTGNN 也已完整落盘。其 H3、H6、H12 与 Average MAE 为 12.503220、13.296460、14.551459 与 13.285545，summary SHA-256 为 `ab0288a4b8fe5a17c33b43b758bbc5851b0e63cc0cce7456a07eb5fe451ce998`。原矩阵 runner 层达到 38/39。

五卡机继续 task034 Orange STTN 时，单卡空间注意力在首个训练步尝试额外分配 11.22 GiB 并 OOM。该失败没有 summary 或正式 task034 目录。用户要求关闭五卡机后，控制层确认 trainer 为 0、既有 12 项保持完整，并生成 60 个 artifact 的关机前清单。五卡机随后正常关闭。该主机尚未完成原分片 postflight 和正式回传。

用户重新启动四卡机并批准由该机接管 task034。重启后四张卡均为 RTX 4090 D，但 GPU0 已更换。锁定 UUID 依次为 `GPU-c12dfe6a-f3b4-2056-8a0c-1b0a629feb36`、`GPU-70077372-d84a-eeb9-7ffc-ff3f33f25653`、`GPU-af61ffd0-be5f-a346-ec99-1a6dd755b670` 与 `GPU-f89fbd8b-93b0-65dd-ca38-386cbf894bb8`。

独立计划 `igstgnn_table4_task034_4gpu_recovery_v1` 的 SHA-256 为 `022d3d9a0c635f94ce5f7b7760c022ba38162bbccfe68675b33cd6bbe78834c5`。计划只允许 task034，原始来源仍记录为五卡机的 `table4_4090d5_gpu0`，实际恢复主机记录为四卡机。execution layout 固定为 `cross_host_single_task_recovery`。正式结果写入独立根，不覆盖四卡机原 26 项或五卡机原 12 项。

提交 `4935b9b` 增加 STTN 四卡整批 DataParallel、真实完整优化步 smoke、独立 postflight 和通用严格 finalizer 身份绑定。本地全仓回归为 465 passed 和 2 个既有 warning。四卡机部署文件逐份 SHA-256 与本地一致，专项测试为 13 passed。

四卡机控制器 PID `2928` 于 2026-07-30 09:21:02 新加坡时间启动。真实 smoke 使用 batch 64，完成前向、反向、梯度裁剪和一次优化器更新。loss 为 `170.42190551757812`，prediction shape 为 `64×12×990×1`。GPU0 至 GPU3 的峰值分配显存为 16,082,649,088、16,060,899,840、16,060,899,840 与 16,060,899,840 bytes。09:21:28 smoke 验证通过后，控制器自动启动正式 task034。

正式训练首个落盘 epoch 的训练 loss 为 29.8492，训练 RMSE 为 43.5428，训练 MAPE 为 0.7190，验证 loss 为 21.1960，验证 RMSE 为 34.2392，验证 MAPE 为 0.3215。训练与验证耗时为 42.5066 秒和 4.5379 秒。控制器会在训练完成后重新验证精确五文件 allowlist，再执行独立 source-host postflight、结果服务器逐文件 SHA-256、精确目录树复验和原子提升。全部成功后调用 `/usr/bin/shutdown -h now`，任何失败都保持四卡机开机。

## 09:48 五卡机 12 项正式回传

用户重新开启五卡机并要求立即回传既有结果后，主机核验显示 5 张 GPU 全空闲，没有 Table 4 trainer。正式 summary 恰好 12 份，任务为 Orange tasks027 至 039 中除 task034 之外的全部项目。task034 正式目录仍不存在。

提交 `918af63` 增加只读固定计划 `igstgnn_table4_host5_partial_export_12_v1`、独立 postflight 和 fail-closed 回传控制器。计划 SHA-256 为 `cd81652493690d1d329f5eecd45fe541c9de7db26c93650fc490a2507616845d`。本地全仓回归为 469 passed 和 2 个既有 warning，五卡机专项测试为 4 passed。该计划只允许 12 项和 60 个正式 artifact，不读取失败 queue 作为完成证据，不包含 task034。

两次 source-host postflight 均验证 12 项、60 个文件和 1,975,159,969 bytes，files digest 均为 `5c11ad4596228f08613b783d8a42ed9334e9976ce8cff4d23d09f6b79f055490`。结果服务器初始可用空间为 1,967,017,984 bytes，低于源文件加安全余量，首次控制器在传输前保持等待。没有 incoming 正式数据，也没有结果改写。

结果服务器只清理系统归档日志、APT 缓存、pip 缓存和 npm 缓存，共释放 1,753,280,512 bytes，没有删除任何正式实验树。等待控制器被定向停止并保留日志。重试控制器使用 512 MiB 余量，随后完成 60 文件传输、incoming 树逐文件 SHA-256、原子提升和正式树二次复验。

正式目标为 `original_host5_partial_12_v1/hosts/table4_4090d5_partial12`。transfer marker SHA-256 为 `53f73b3e7ae2f92535eed44fb0fc934d7be45387de579b8dfc3fcb853afd54cc`，postflight SHA-256 为 `427a21568b8739c355a5ca5851d01e622947556f8dcf9ae6572cd45dfc19c2d4`，transfer manifest SHA-256 为 `b3e773916c28907bcaa93adaf236cededd08e798f233f538633c57758e609bb4`。四卡机使用独立密钥再次验证正式树精确包含 60 个结果文件、无 incoming 目录且 marker 与 postflight 身份一致。五卡机随后自动关机，SSH 返回连接关闭。

同一时刻，四卡机 task034 已落盘到 epoch 35。训练 loss 为 14.3433，验证 loss 为 14.0013，训练仍在继续。

## 下一步

1. 等待四卡机 task034 完成独立 postflight、正式回传和自动关机。
2. 建立原 39 项三源只读 composite index，并验证 canonical tasks001 至 039 各出现一次。
3. 为已发布的 NDR 9 项建立独立索引。
4. 从逐份复验的 summary 与 predictions 生成论文同结构原表、NDR-Net 三种子补充行、逐单元差值表和数字追溯索引。
