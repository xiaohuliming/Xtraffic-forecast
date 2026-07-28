# IGSTGNN Table 4 双机九卡启动进度报告

更新时间：2026-07-29 01:16 新加坡时间

## 当前结论

IGSTGNN Table 4 复现已从 5080 单卡 pilot 切换为两台 RTX 4090 D 主机。两台主机共有 9 张 24 GB GPU。正式协议仍为 `igstgnn_table4_official_v1`，正式矩阵仍为 `igstgnn_table4_public_repro_39_v2`。本轮结果只能称为 `public_material_maximum_fidelity_reproduction`，有效性标签仍为 `official_protocol_with_known_overlap`。

39 项固定分片清单位于 `config/reproduction/igstgnn_table4/distributed_4090d_9gpu_v1.json`，SHA-256 为 `d6eec02c9935a5699cdd814d72be85317575e361fe8f244ff7cb47e4e67fffcc`。五卡机承担 Orange 的 13 项，四卡机承担 Alameda 与 Contra Costa 的 26 项。每个地区的全部方法保留在同一台主机上，地区内直接比较不会跨主机。

四卡机的 4 个 shard 正常运行。五卡机的 Orange HL 已完成，另外 3 个单卡 shard 正常运行。Orange IGSTGNN 因 24 GB 单卡物理显存不足进入经过记录的双卡恢复流程。恢复控制器 PID 为 `20105`，它会等待 Orange BiST 释放 GPU 3，再使用物理 GPU 4 与 GPU 3 执行完整 batch 的 IGSTGNN。该流程不缩小 batch，不修改 pristine 上游源码，不修改数据、损失、优化器、epoch 或 early stopping 契约。

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
| 本地全仓回归 | 440 passed，2 个既有 warning |
| 两台 Linux 原始训练门 | 各 112 passed，1 个既有 warning |
| 多卡仓库远端专项门 | 15 passed |

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

## 自动回传与关机

两台主机完成各自固定 shard 后，会执行 source-host postflight。postflight 要求 queue 完成、任务和文件 allowlist 精确、五份 artifact 齐全、运行时主机身份一致、GPU 型号为 RTX 4090 D，并按训练代码家族验证 digest。

通过 postflight 后，finalizer 只上传 allowlist 文件到 `root@8.134.12.5:/root/xtraffic_results/igstgnn_table4_official_v1/distributed_4090d_9gpu_v1/hosts`。结果服务器会逐文件重新计算 SHA-256、核对精确目录树并原子提升。只有正式目标再次完整复验通过后，源主机才调用 `/usr/bin/shutdown -h now`。postflight、容量、传输、远端哈希或原子提升任一步失败时，主机保持开机。

## 下一步

1. 验证 Orange IGSTGNN 双卡运行通过首个真实 batch，并确认两张卡同时持有模型副本。
2. 保持 8 个常规 shard 与 Orange IGSTGNN 自动运行。
3. 等待两台主机各自完成 postflight、回传、结果服务器复验和关机。
4. 两份正式主机结果都提升后，建立只读 39 项 composite index。
5. 从逐份复验的 summary 与 predictions 生成与论文同结构的 Table 4、逐单元差值表和数字追溯索引。
