# NDR 5080 连接中断记录

日期：2026-10-14，Asia/Singapore。

Tailscale 只读状态显示本机在线，5080 peer 离线，最后在线时间为 04:00。Tailscale 地址和既有 LAN 地址的 SSH 均连接超时，三份失败回执保留。不能据此判断工作站关机、训练停止、失败或已经完成。

最后可验证的正式状态来自 `stage1_poll_30.json`，为 2/8 completed，Alameda mask_signed 完成 23/30 轮，正在 epoch 24。supervisor 6232、worker 48172 和 GPU 工作均在该次观察中存活。当前远端句柄无法重新验证，不再把最后可验证状态写成当前 running。

两项原 NDR 基线已完整取回并复验。本地保全观察进程 45878 仍实测存活，状态为 `observation_unavailable_queue_not_restarted`，继续等待同一固定计划的连接恢复。其缓存的源端 live 字段只是最后成功观测，不能当作当前源端活性证明。观察进程没有重启远端训练、修改队列或改变绝对截止。

已请求用户确认工作站和网络状态。连接恢复后先重新核对原 supervisor、当前 worker、固定 plan digest、queue 和 checkpoint，不因 SSH 超时重排训练。旧文件、strict optimizer、scheduler、RNG 与 loader 状态继续保留，当前没有执行恢复或新 GPU 工作。

完整证据为 `outputs/remote_runs/ndr_generalization_20261013/control/source_connectivity_incident_20261014_0400.json`。固定八项和后继校准的完整完成情况暂时未知，本轮没有新增精度或 SOTA 结论。
