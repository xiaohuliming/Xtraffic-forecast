# NDR 第四项独立验收与原矩阵续队

日期：2026-10-15，Asia/Singapore。

## 当前运行与保全

显卡 owner 再核查确认游戏 GPU 进程退出，没有外部 Python 或未知 owner。已部署的严格续队启动后只验证 ContraCosta mask_signed 现有完整三十轮 best checkpoint，没有重训该项。独立完整 validation 通过，global MAE 为 11.322497249797081，best epoch 为 14。十一文件、38,134,437 bytes 已完整回传 Mac，源端前后不变、逐文件大小和 SHA-256 一致，本地 best 和 last 严格结构与有限性通过。本地未重复完整 validation 推理。

00:07 的源端实测为 running、4/8，supervisor 39640、Alameda sanitized_control worker 35384 均存活，GPU 利用率 69%、已用 6896 MiB。continuation ID 为 `23b45b4db6b744888a3c650f4d44cc78`。四项独立结果不等于完整矩阵准入，不进行候选晋级或 SOTA 判断。

原后继六项工程校准 wrapper 已恢复。守候进程 28004 实测存活，动态跟随新 main PID 39640，等待完整八项 postflight、精确文件树、matrix admission 和原句柄退出，没有 CUDA child。原六项参数、控制器和 provider 字节，以及 2026-10-16 00:10:22 截止均保持不变。

本地自动保全进程 61786 实测存活，每三百秒观察固定队列。前三项收据继续保留，第四项收据已加入。旧停止进程 47645 的状态与锁另行保全，没有覆盖存活 observer。轮询入口已支持 continuation 日志身份。

## 启动故障与修正

第一次启动请求得到 Windows result 2147942667，对应目录名称无效。工作目录经过多层 Python 字符串转换后包含换行和回车，原 repo 目录本身存在。没有启动控制器或 GPU 工作，没有增加训练产物。诊断回执与原启动请求保留。

修正通过结构化 JSON 参数更新两项任务 action 的工作目录，执行路径规范化。控制器 argv 保持原字节，原 Python、训练参数、任务顺序、缓存、模型代码和截止未改变。修正后的启动请求被实际 supervisor、worker 句柄和训练日志证明已经执行。注册任务或启动请求本身没有被当成训练成功证据。

续队控制器 SHA-256 为 `d46726dcd7952dbb266d2f6a3cd05fc1a7c871998d9aad32fd60b3192b8bd9c0`，provider 为 `23c999a6e4321da1d3298d6eb9e9682798e112648b5e2f6751a3850a9e052015`。此前 88 项联合检查与源端 CPU 准入均通过。没有修改模型训练数学或放宽 GPU owner 门禁。

## 固定预算与判读边界

主队列截止保持 2026-10-15 16:10:22，单项上限十二小时。SAN 和 Projected 的四项均按原参数和原顺序从零开始。依据两个地区此前较慢实测外推，剩余训练和验收接近总截止。该外推是经验估计，不是严格完成保证。新的 busy、OOM、非有限值或截止会按既有规则保留失败，不为获得正结果更改停止线。

完成八项并通过完整门禁后，才分析 mask 相对 baseline 与 sanitized，以及 Projected 相对 baseline。两开发地区均须满足原严格改善条件。后续三种子确认的发现与确认结果分开，Orange 和真实 test 继续封存。

## 证据入口

回执根为 `outputs/remote_runs/ndr_generalization_20261013/`。

- `control/continuation_resumed_audit_20261014T155555Z.json`
- `control/stage1_continuation_scheduler_poststart_01.json`
- `control/stage1_continuation_scheduler_path_diagnosis.json`
- `control/continuation_scheduler_path_repair_20261015.json`
- `control/stage1_continuation_poll_02.json`
- `control/continuation_running_live_verification_20261015.json`
- `control/contracosta_mask_signed_retrieval_result_20261015.json`
- `stage1_copied/receipts/contracosta_mask_signed_seed42_ff27c6a47be24fab96e91f0030b5092a.json`
- `control/stage1_preservation_watch_status.json`


## 本地保全恢复补记，01:29

旧本地 observer PID 61786 已确认不存在，对应 exec session 也不存在，旧状态与锁已独立保全。退出原因没有足够证据确认。原同一观察程序以独立 session 启动，PID 5281、parent 1、process group 5281 实测存活。控制代码、三百秒频率、原停止时间和验收器均未改动，没有重新启动远端 queue。

最新源端 supervisor 39640 和 worker 35384 均实测存活，Alameda sanitized_control 完成 10/30 轮，completed 仍为 4/8。恢复回执为 `control/stage1_preservation_detached_launch_20261015.json` 和 `control/stage1_preservation_detached_live_verification_20261015.json`。此前 00:07 的 observer 快照保留为历史。
