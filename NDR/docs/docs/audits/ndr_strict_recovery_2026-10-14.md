# NDR 固定开发矩阵的严格恢复与当前进展

日期：2026-10-14，Asia/Singapore。

以下恢复成功快照已由 22:07 的 GPU 占用 incident 取代。当前状态见 `ndr_gpu_owner_incident_2026-10-14.md`。

## 原因与保全

系统 1074 记录于 03:57 由 RuntimeBroker 发起关闭电源，系统 13 随后记录关机。用户补充关闭可能由其关机操作引起。原 worker 以 0x40010004 退出，原 supervisor 与后继校准守候进程均停止。没有将这次外部中断当作模型精度失败。

原失败结果树共 38 文件、116,391,668 bytes，已在 Mac 与源主机原结果根之外双份完整保全。精确文件集、逐文件大小和 SHA-256，以及传输前后源树不变性通过。两项已验证基线继续保持原字节。原第 23 轮 last.pt 具有完整 optimizer、scheduler、随机数、loader 和训练进度，新进程恢复前核对原配置、运行环境、缓存与源码。

## 恢复与预算

恢复控制器独立部署，原训练 repo、原八项计划及其摘要未改变。中断任务从第 23 轮的完成边界继续执行第 24 轮。故障前已消耗 16,536.291687 秒，包含未保存的第 24 轮 900 批次工作；剩余单项额度为 26,663.708313 秒。没有重置单项十二小时预算或原 48 小时绝对截止，截止仍为 2026-10-15 16:10:22。

新监督进程为 30580。已记录的第 24 轮 1、100 至 900 共十个重放批次的 optimizer update 计数与 objective 均和旧故障日志完全一致。未逐一比较未记录批次，不能把这项工程证据当作精度或泛化提升。

Windows WDDM 的桌面程序不等同于 CUDA 训练进程。实机查询核对两项 Tencent 签名的 WeGame WebHelper 与 Recorder，GPU 引擎当时均为零，并将通用 executable 名称限定到其已核实产品路径。Python GPU 进程和未知 CUDA owner 仍被拒绝。恢复计划任务无触发器，以同账号 SID 的 pythonw 与无窗口子进程执行。运行会阻止自动空闲睡眠，不能防止用户关机或注销。

## 当前实际进度

最新 `stage1_recovery_poll_06.json` 为 3/8 completed。当前 ContraCosta mask_signed 完成 27/30 轮，worker 30768 与 supervisor 30580 均实测存活。Alameda mask_signed 已完成三十轮并经源端独立完整 validation 验收，其 global MAE 为 10.678324453161249，best epoch 为 18。十一文件、39,021,870 bytes 已完整回传，源端前后不变、逐文件哈希和本地 best、last 严格结构校验通过。本地没有重复完整 validation 推理。

这是单地区、单种子、固定预算开发结果。原八项还未通过完整矩阵 postflight，不作候选晋级或 SOTA 判断。仍须保留 sanitized 控制和 projected teacher 对照，以及 ContraCosta 完整结果。

## 后继队列与本地观察

后继六项工程校准的旧目录只有一个状态文件，439 bytes，已经独立保全，确认从未执行任何 job。新 wrapper 保持原六项 argv、原控制器字节与 plan 摘要，固定截止为 2026-10-16 00:10:22。新守候进程 39636 实测存活，正在等待源端完整八项 postflight、精确文件树、matrix admission 和原当前句柄退出，没有 CUDA 子进程。GPU owner provider SHA 被独立锁定，不自动启动正式候选训练。

本地自动保全进程再次以绝对数据路径启动。先前启动时的收据校验失败保留为 incident，当前全部四份旧收据及原 local_verified 函数重新检查均通过，不能将历史失败归因于某份坏收据。回传使用 task-local SSH connection reuse，原验收器与旧收据不改。观察脚本已经跟随恢复日志路径。本机 Tailscale 曾停止，已恢复其已有连接；观察失败没有触发远端重启。

## 工程验证与范围

研究目录最新 383 passed，0 failures、errors 或 skipped，Ruff、compile 与 diff 检查通过。代码提交为 cf2b15db6f96d20d98b28d70bd79b75b5e247602，测试时 clean。证据为 `control/ndr_related_validation_v4_20261014.xml` 及 `control/ndr_related_validation_summary_v4_20261014.json`。恢复 controller、checkpoint 与运行记录位于 `outputs/remote_runs/ndr_generalization_20261013/`。

本轮没有读取真实 test，没有启动 Orange 监督训练或评分，没有新增已验证模型性能结论。研究目标仍未完成。
