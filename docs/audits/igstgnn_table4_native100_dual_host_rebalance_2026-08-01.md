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

当前 Mac 全仓回归为 493 passed 和 2 个既有 warning。双机训练完成后仍需分别执行 source-host postflight，回传结果服务器后逐文件复验，再生成只读 composite index。完成 composite 前不得生成或发布最终 18 项表。本轮 5090 没有配置自动关机。
