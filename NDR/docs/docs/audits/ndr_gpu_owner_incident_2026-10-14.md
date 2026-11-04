# NDR 队列的 GPU 占用 incident

日期：2026-10-14，Asia/Singapore。

GPU 已于后续复查确认空闲，原矩阵已在 2026-10-15 恢复为 4/8 running。本文故障和 Ready 快照作为历史保留，最新实测见 `ndr_verify_only_continuation_2026-10-15.md`。

## 实测状态

22:07:27，主监督进程 30580 在 ContraCosta mask_signed 训练子进程正常退出后停止。子进程退出码为零，summary 和 last checkpoint 均记录完整三十轮。故障发生在独立验证启动前的 GPU owner 检查，队列 completed 仍为前三项，不能标记为四项已验收。

随后实测 GPU unknown owner 为 VALORANT-Win64-Shipping.exe 和 EpicWebHelper.exe，PID 分别为 27724 和 36436。查询没有 Python GPU owner，空闲显存 10524 MiB，利用率 27%。这份实际回执支持真实外部占用，不足以将故障归因为 WDDM 退出残留竞态。没有放宽 owner 门禁，没有停止游戏或其他竞争进程。

## 双端完整保全

原结果树五十四文件、154,652,550 bytes 已完整保存到源端 `C:/XTrafficResearch/ndr_20261013/continuation_20261014_2207/incident_backup` 和 Mac `outputs/remote_runs/ndr_generalization_20261013/stage1_owner_failure_preserved_20261014/source_host`。源树、备份树和 Mac 副本的精确文件集、大小与 SHA-256 一致，传输前后源文件保持不变。

当前第四项 best 和 last 的 epoch 为 14 和 30，配置摘要一致，模型参数有限。本地未重复完整 validation 推理。这不替代源端独立验证，也不构成完整矩阵或候选晋级证据。

后继六项工程校准因前置队列进入 terminal incomplete 而停止。它仍为零任务，四文件、4762 bytes 已另行双端保全。后继恢复必须复用原六项 argv、原控制器、严格 GPU provider 和 2026-10-16 00:10:22 截止。

## 恢复边界

用户已授权本研究的设计、训练、恢复与最终评测。当前仅需要 GPU 空闲，已请求用户结束游戏并保持工作站运行。恢复程序须先直接独立验证现有第四项完整权重，绝不重新训练它。前面三项保持精确字节，后面四项维持原顺序、原训练参数和新任务目录。

原八项 plan digest 为 `ad530e29fdcf2bd0cdabdd739349814e805f12e11cfab6341e9180651807672b`。主队列原截止仍为 2026-10-15 16:10:22，单项十二小时预算不重置。严格 owner 门禁可在原三十分钟窗口内等待，达到原截止或竞争未解除时保留故障，不终止其他进程。

两候选的后续确认政策已在完整结果出来前冻结。若两者都通过，均进入 seeds 1、2 确认。mask_signed 每个独立种子、每个开发地区须分别优于 baseline 与 sanitized control；projected 每个独立种子、每个地区须优于 baseline。seed 42 保留为发现结果，三种子只能作描述性稳定性比较。执行预算和任务清单将在 CUDA 成本完成后另行冻结。Orange 与真实 test 仍按冻结评测声明封存。

## 证据入口

全部回执位于 `outputs/remote_runs/ndr_generalization_20261013/`。

- `stage1_recovery_poll_10.json`
- `stage1_continuation_gpu_owners_01.json`
- `control/stage1_continuation_dual_preservation_result.json`
- `control/stage1_continuation_process_and_calibration_probe.json`
- `control/stage1_continuation_local_checkpoint_audit.json`
- `control/transfer_calibration_continuation_dual_preservation.json`
- `control/stage1_confirmation_policy_20261014.json`
- `control/next_stage_budget_readiness_20261014.json`


## 续队工程已就绪

独立续队代码 `scripts/remote/windows/ndr_research_continuation.py` 已通过 88 项联合检查、Ruff、编译和独立只读审查。验证时 clean Git SHA 为 `2fc6b06d08b26bdb56f058125e2f60fd884203c3`，控制器 SHA-256 为 `d46726dcd7952dbb266d2f6a3cd05fc1a7c871998d9aad32fd60b3192b8bd9c0`。原 provider、模型训练代码和冻结计划没有改动。

七个部署文件与 Mac 字节一致，源端 CPU 准入通过完整保全、原句柄死亡、前三项严格结构和当前第四项训练完成检查，没有执行 GPU 工作或完整 validation 推理。主续队计划任务 `XTraffic_NDR_EventGate_Continuation_20261014_2207` 已注册为 Ready，无触发器且未启动。后继任务 `XTraffic_NDR_Transfer_Calibration_Continuation_20261014_2207` 复用原 wrapper，也为 Ready，未启动。不是正在运行的训练。

原本地保全进程 47645 已死，状态与锁已保全，轮询入口已兼容 continuation 日志身份。新主队列实际启动后才重新启用本地观察，不改写旧收据。

补充证据为 `control/stage1_continuation_deployment_receipt.json`、`control/stage1_continuation_source_admission.json`、`control/stage1_continuation_parked_task_receipt.json`、`control/transfer_calibration_continuation_parked_task_receipt.json`、`control/continuation_controls_validation_summary_20261014.json` 和 `control/stage1_continuation_local_watcher_readiness.json`。
