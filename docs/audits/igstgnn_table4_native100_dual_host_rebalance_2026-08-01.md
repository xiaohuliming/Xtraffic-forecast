# IGSTGNN Table 4 原生配方 100 epoch 双机重分片

## 目的

2026-08-01，用户提供一台 RTX 5090，用于缩短 `igstgnn_table4_bist_ndr_native100_18_v1` 的剩余训练时间。正式执行从单台 RTX 5080 的 18 项队列切换为按地区分层的双机矩阵。Alameda 与 Contra Costa 保留在 RTX 5080，Orange 的 BiST 与 NDR-Net 三个种子全部放在 RTX 5090。这样同一地区内 BiST 与 NDR-Net 的直接比较仍在同一台机器完成。

该切换不改变矩阵清单、数据协议、随机种子、训练配方、指标或五文件结果契约。最终执行布局必须标记为 `dual_host_region_stratified`，不得称为单机同质矩阵。

## 锁定身份

- 矩阵 ID：`igstgnn_table4_bist_ndr_native100_18_v1`
- 矩阵 SHA-256：`e4fe8a6987e06dd573212f88665b181b03b83e9fbad43b9f08781ecbedcf8246`
- 基础 39 项清单 SHA-256：`d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`
- 配方 SHA-256：`37ebcb574f516bc1b662ade0d5264deda96a20d017d20edc7a57253daab5e66b`
- 训练代码归档提交：`ace318b`
- 双机计划：`config/reproduction/igstgnn_table4/distributed_native100_5080_5090_v1.json`
- 双机计划 SHA-256：`d66b47c1e4c31e42a4704002e6c98d673b1c2d19f5762684c67d049ce5a7ad6e`

## 固定分片

RTX 5080 分片共 12 项：

- `n100001` 至 `n100006`，Alameda 与 Contra Costa 的 BiST 三种子
- `n100010` 至 `n100015`，Alameda 与 Contra Costa 的 NDR-Net 三种子

RTX 5090 分片共 6 项：

- `n100007` 至 `n100009`，Orange 的 BiST 三种子
- `n100016` 至 `n100018`，Orange 的 NDR-Net 三种子

两分片交集为空，并集严格等于 canonical 18 项。同一地区的两个方法与三个种子都在同一主机上。

## RTX 5080 切换

旧 18 项计划任务 `XTraffic_Table4_Native100_v1` 已停止并禁用。切换时 `n100013_ndr_net_contra_costa_seed42` 只有临时训练目录，没有正式任务目录。中断现场保存在 `D:\t4\native100_compare\incidents\task013_interrupted_for_5090_rebalance_20260801T041822Z`，其 `files.sha256` 摘要为 `94802ea874935de3db70eb3d5b730ec14e978662ebeada4014bc7eb09f34be23`。

新的 5080 计划任务为 `XTraffic_Table4_Native100_5080_CC_v1`，启动脚本 SHA-256 为 `eecb23803e25455eb6cd7cd66df29e44fd2111669a43f5a81f93795ddaf9dbac`。新队列只声明 12 个非 Orange 任务。2026-08-01 12:18 新加坡时间的启动复验为 skipped valid 9 项，失败 0 项，正在从头执行 `n100013_ndr_net_contra_costa_seed42`。12:40 前的最新日志已完成 epoch 17，GPU 显存使用约 10,562 MiB，利用率 95%。

5080 结果根内保留了切换前已完成的 Orange BiST `n100007` 至 `n100009`。这些目录没有被改写，但最终 composite 必须按计划的 `declared_task_ids_only` 规则排除它们，只选择 5090 重跑的 Orange 六项。

5080 分片已于 2026-08-01 15:52:38 新加坡时间完成。queue state 为 completed，声明内 12/12 全部通过 runner 验收，失败数为 0。用户报告主机随后断电。23:37 局域网 SSH 恢复后复验确认 completed 状态与 12 个已验证任务仍在，GPU 空闲，因此断电没有中断训练或导致重跑。

## RTX 5090 部署与验证

5090 为单卡 RTX 5090，显存 32,607 MiB。独立代码根为 `/root/autodl-tmp/native100_repo`，独立数据根为 `/root/autodl-tmp/native100_data/official_splits`，独立结果根为 `/root/autodl-tmp/native100_orange/results`。

BiST 上游源码包与 Orange 数据首次解包后保留了 Mac UID 501。安全门分别拒绝了上游源码与数据目录，没有启动正式训练，也没有正式结果。修正远端目录所有权后，上游 BiST 提交仍为 `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`，tree 仍为 `1dc3c4de312ec8cb2eaf2aafc7b680aa247d19be`，工作树为空。5090 训练专项门随后为 25 passed。

Orange 数据目录严格包含 9 个允许文件。三个 split 的 SHA-256 为：

- train：`e37dd6b32a2ffc1fc721f3047044d68b29a42af9edc008715e79e7feba2471b2`
- val：`2396d02d3fa4412d4c986239c2c865fec8b214e19946409cf6d862efb68f039b`
- test：`d95b874a204154f392f7a29379f788bd657b1cf56b40eb220385460d28843a2e`
- split manifest：`c37b78f19766fec9ac57d28d6a7becd8a3b334261cef4e7863f29b0bbcc7b019`

所有辅助文件的字节数与 SHA-256 也与 Mac 原件逐一一致。数据传输后可用数据盘空间约 45 GB。

5090 上两个 Orange 真实 CUDA 完整优化步探针均通过：

- BiST seed 42，loss `148.53358459`，峰值分配显存 `3142229504` bytes
- NDR-Net seed 42，loss `42.48369026`，两次 microbatch，峰值分配显存 `12682121728` bytes

## 正式启动状态

5090 启动脚本 SHA-256 为 `862d6e05221bfad46e3ee02692bf375db113cabb785559031c4e4835a2758d94`。持久控制进程于 2026-08-01 12:38:57 新加坡时间启动，launcher PID 为 `3072`，matrix runner PID 为 `3075`。启动后的 queue state 为 running，选择任务 6 项，完成 0 项，失败 0 项，当前任务为 `n100007_bist_orange_seed42`。12:40 时该任务已完成 epoch 6，GPU 显存使用 3,870 MiB，利用率 56%，温度 56 摄氏度。

当前 Mac 全仓回归为 493 passed 和 2 个既有 warning。双机训练完成后仍需分别执行 source-host postflight，回传结果服务器后逐文件复验，再生成只读 composite index。完成 composite 前不得生成或发布最终 18 项表。

## RTX 5090 自动关机守护

2026-08-01 23:45 新加坡时间，用户明确要求 5090 在训练完成后关机。守护脚本 `scripts/remote/linux/igstgnn_table4_native100_5090_shutdown_waiter.sh` 的 SHA-256 为 `d0168a7f1b3b2b1b5386de1cebb52681fc1319babe8df1eddb5344228706bbb1`。本地与 5090 真实环境测试均通过三种情形：队列未完成时保持开机，六项完整复验且 runner 零退出时执行关机，runner 非零退出时保持开机。

远端持久守护 PID 为 `20225`，父进程为 PID 1，每 60 秒检查一次。关机前会重新验证矩阵、基础清单、双机计划、queue status、runner exit code、六个精确任务目录以及每项五文件 artifact，并确认没有残留训练进程。只有全部通过才调用 `/usr/bin/shutdown -h now`。任何缺项、哈希错误、非零退出或残留训练进程都会 fail closed 并保持机器开机。启动守护时 5090 为 4/6 完成、0 失败，task017 已完成 epoch 82。完整矩阵当时为 16/18 已验收。

## RTX 5090 供应商停机与替代主机重跑

2026-08-02，旧 5090 端口 52464 持续不可达。用户认为可能是余额不足。新端口 13647 可以连接，但对应容器为 `autodl-container-dbf949bc7b-086867e4`，其 `/root/autodl-tmp` 初始为空，旧机的结果根、控制日志与 task017 临时 checkpoint 均不存在。旧守护在最后一次可观测状态仅为 4/6 且 task017 epoch 82，不可能通过 6/6 本地关机门，因此旧机离线不能认定为自动关机成功。供应商停机原因尚未由账单或平台事件记录证实，审计只保留“疑似余额或供应商停机”表述。

旧机结果没有回传结果服务器，也没有出现在新容器数据盘。它们不进入正式 composite。替代主机从头重跑 Orange 六项，仍把 Orange 的 BiST 与 NDR-Net 三种子保持在同一台 32 GB RTX 5090 上。执行布局仍为按地区双机分层，但最终 provenance 必须记录替代容器，不得伪装成旧 5090 续跑。

替代主机重新部署训练归档提交 `ace318b`。BiST pristine checkout 的 commit 为 `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`，tree 为 `1dc3c4de312ec8cb2eaf2aafc7b680aa247d19be`，工作树为空。Orange 九个数据文件从 Mac 重新传输并逐文件 SHA-256 复验，split manifest SHA-256 仍为 `c37b78f19766fec9ac57d28d6a7becd8a3b334261cef4e7863f29b0bbcc7b019`。替代环境为 Python 3.12.3、PyTorch 2.12.1+cu130 与 NVIDIA GeForce RTX 5090。

两个真实 CUDA 完整优化步探针再次通过：

- BiST seed 42，loss `148.53358459`，峰值分配显存 `3142229504` bytes
- NDR-Net seed 42，loss `42.48369026`，两次 microbatch，峰值分配显存 `12682121728` bytes

探针数值与旧 5090 完全一致。替代主机的持久 Orange 六项队列于 2026-08-02 09:48:50 新加坡时间从 0/6 启动。launcher PID 为 `2314`，matrix runner PID 为 `2318`。09:59:56 时队列为 0/6、0 失败，正在执行 `n100007_bist_orange_seed42` epoch 70。本地全仓回归为 496 passed 和 2 个既有 warning。

旧守护只完成本机复验后关机，不满足当前结果服务器回传要求。新增 `scripts/remote/linux/igstgnn_table4_native100_5090_finalize.sh`，SHA-256 为 `a1c85fc95b5af92b51dcd44e209cc132a93a1b7bf7459d4552fcfe42fa892597`。替代主机使用专用受限 SSH 身份连接结果服务器，公钥指纹为 `SHA256:LpNIDomLiv0Vm6YHvJvN5miilCwUSSXSPXnRNhAt5hg`。finalizer 持久 PID 为 `2815`。它按以下顺序 fail closed：

1. 等待 runner 写出退出码。
2. 复用已测试的六任务本地五文件验收。
3. 确认没有残留训练进程。
4. 生成精确 31 文件 allowlist、逐文件 SHA-256 和 source-host postflight。
5. 执行结果服务器容量门并上传到 incoming 目录。
6. 在结果服务器逐文件验哈希并比对精确文件集合。
7. 原子提升正式目录并再次复验。
8. 仅在以上全部成功后调用 `/usr/bin/shutdown -h now`。

任何本地验收、容量、传输、哈希、allowlist、原子提升或最终复验失败都会保持主机开机。完成结果服务器正式发布前，矩阵状态仍为运行中，不得生成 18 项最终表。
