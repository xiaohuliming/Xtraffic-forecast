# NDR 同预算串行队列控制器

日期：2026-10-20，Asia/Singapore。
范围：BiST、BiST-NDR 与 NDR-Net 同预算对照在 RTX 5080 Windows 工作站上的串行执行外壳。
状态：两个新文件位于工作树 `/Users/xhlm/.codex/worktrees/bist-ndr-matched/科研实习`，分支 `codex/bist-ndr-matched`，基于 `1a718c6f0a4d60801ceae8f9207acc00d9d4cd95`，未提交。没有创建任何真实计划、预算、GPU 任务或远端连接。

## 定位

本控制器只是执行外壳。它不知道哪个配方正确，不估算成本，不冻结预算，也不判读任何精度。真实计划仍需先在 5080 上完成 CUDA 校准，再按校准结果冻结每项超时、总预算与绝对截止，并取得用户对该次 GPU 执行的明确授权，之后才能用本控制器运行。合成 CPU 测试通过不代表 CUDA 可行性、吞吐、显存或任何实验结论。

## 文件

| 文件 | SHA-256 |
|---|---|
| `scripts/remote/windows/ndr_matched_budget_serial_queue.py` | `bddc1e2295493ccc3134ba5a7dafcbff77784ae49a86323b1a7ecc63b2e124b0` |
| `tests/remote/test_ndr_matched_budget_serial_queue.py` | `eca21c943fe78a87c80ef7b8c52839b941572b0fc6a6e32574dacc31cb04dbae` |

控制器只依赖 Python 标准库，不导入研究代码，运行方式为 `python ndr_matched_budget_serial_queue.py --plan plan.json`，加 `--admit-only` 时只做准入。退出码 0 表示准入通过或全部完成，1 表示 `failed_stopped`，2 表示准入被拒，3 表示 `stopped_deadline_or_budget`。

## 计划 schema

`schema_version` 固定为 `ndr_matched_budget_serial_plan_v1`。必需键为 `schema_version`、`purpose`、`plan_digest`、`budget_frozen`、`absolute_deadline_utc`、`total_budget_seconds`、`busy_wait_seconds`、`minimum_free_gpu_mib`、`python`、`run_root` 与 `jobs`。可选键只有 `state_path`、`journal_path` 与 `metadata`，前两者默认分别为 `run_root/controller_state.json` 与 `run_root/owned_processes.jsonl`，`metadata` 用于记录来源说明并计入摘要。其他任何键都会被拒绝。

每个 job 必须恰好包含 `job_id`、`role`、`command`、`timeout_seconds`、`depends_on`、`success_artifact` 与 `output_dir`。`role` 取 `calibrate`、`train` 或 `verify`。`success_artifact` 是相对 `output_dir` 的路径。

`plan_digest` 等于去掉 `plan_digest` 键后，按 `sort_keys=True`、`separators=(",", ":")`、`allow_nan=False` 序列化的 JSON 的 SHA-256，与项目既有 `canonical_digest` 一致。模块提供 `seal_plan` 供作者生成摘要。

## 准入规则

准入只读，不创建目录、不写文件、不启动子进程，在执行主机本地运行。

1. 计划文件必须是 UTF-8 JSON 对象，重复键、NaN 与 Infinity 一律拒绝。
2. 键集合必须与 schema 完全一致，`budget_frozen` 必须为 JSON `true`，`purpose` 非空。
3. `plan_digest` 必须是 64 位小写十六进制，并与重新计算的摘要相等。
4. `absolute_deadline_utc` 必须是带显式偏移的 ISO 8601 时间，且晚于准入时刻。
5. `total_budget_seconds` 必须有限且为正，布尔值不算数字。`busy_wait_seconds` 必须满足大于 0 且不超过 600。`minimum_free_gpu_mib` 必须有限且不低于 12000。
6. 全部 job 的 `timeout_seconds` 之和不得超过 `total_budget_seconds`。
7. `python` 必须是本机已存在的绝对路径，每个 job 的 `command[0]` 必须与它逐字相等。
8. `run_root` 必须是绝对路径，已存在时必须是真实目录，不存在时最近的已存在祖先必须是可写目录。状态文件与 journal 必须位于 `run_root` 内、互不相同且尚不存在，因此同一计划不能执行两次。
9. `job_id` 只允许字母、数字、下划线、点和连字符且唯一。`depends_on` 只能引用更早的 job。每个 verify 恰好依赖一个更早的 train。命令是非空字符串列表，超时有限且为正。
10. `output_dir` 必须严格位于 `run_root` 内，不得与 `run_root/controller_logs` 重叠，不得包含状态文件或 journal。calibrate 与 train 的输出目录互不嵌套，且必须是新目录，只有带 `--resume` 的 train 才要求目录已存在。verify 的输出目录只允许落在其依赖 train 的目录内或另取新路径。所有成功产物路径互不相同，准入时均不得已经存在。
11. 命令里出现 `--mode`、`--output-dir`、`--checkpoint`、`--resume` 或 `--device` 时，必须各只出现一次，并与计划字段一致。`--mode` 等于 role，`--output-dir` 等于 `output_dir`，verify 的 `--checkpoint` 必须位于依赖 train 的输出目录，`--resume` 只能是该 train 输出目录下的 `last.pt`。命令包含 `train_ndr_strong_reference.py` 或 `train_ndr_research.py` 时更严格，必须显式给出 `--mode`，train 与 calibrate 必须给出 `--output-dir`，verify 必须给出 `--checkpoint`，并拒绝这五个旗标的缩写写法，因为 argparse 默认接受前缀缩写。

通过时打印一行 `{"plan_digest": ..., "status": "admitted_no_worker"}` 并以 0 退出，拒绝时打印 `admission_rejected` 与原因并以 2 退出。

## 执行语义

执行前再次核对计划文件字节与准入时一致、截止未过、状态文件与 journal 不存在，然后以独占方式创建 journal，第二个控制器无法接管同一计划。

每个 job 依次执行以下步骤。先确认全部依赖已成功。再进入 GPU 空闲等待，可注入的 provider 返回空闲 MiB，读数低于门槛、抛出异常或不是有限非负数都视为不空闲，每次等待不超过 `busy_wait_seconds`，截止或总预算在等待中耗尽时以 `stopped_deadline_or_budget` 停止。默认 provider 调用 `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`，多卡时取最小值。随后复查成功产物不存在、输出目录状态与准入一致，用 `subprocess.Popen` 启动子进程，标准输入为空，标准输出与标准错误分别写入 `run_root/controller_logs/<job_id>.stdout.log` 与 `.stderr.log`，Windows 上使用 `CREATE_NO_WINDOW`。子进程结束、单项超时或全局截止与预算耗尽三者先到者决定终态。

单项超时时只终止本控制器创建的这个子进程，先 terminate，宽限期后再 kill，记录 `failed_timeout` 并以 `failed_stopped` 停止队列。全局截止或预算在子进程运行中耗尽时同样只终止自己的子进程，job 记为 `terminated_deadline_or_budget`，队列为 `stopped_deadline_or_budget`。非零退出记为 `failed_nonzero_exit`。退出码为 0 但成功产物缺失或内容不符记为 `failed_success_artifact`。任何失败都保留全部文件，不重试，不延长预算，其余 job 记为 `not_started`。控制器自身出错时也只停止自己的子进程。

成功判据与真实产物对齐。train 要求 `summary.json` 的 `status` 等于 `training_completed_validation_pending_independent_verification`。verify 要求 `verification.json` 的 `status` 以 `verified_` 开头，两个入口共五种验证状态都满足。calibrate 要求 `status` 以 `calibrated_` 开头，请求设备为 CUDA 时，若产物带 `cuda_budget_eligible` 键则必须为 true。

journal 为 JSONL，只追加并在每行后 flush 与 fsync。每次启动写一行 `spawned`，含 `job_id`、`pid`、`command`、`started_at`，结束时另写一行 `exited`，含 `returncode`、`ended_at`、终态与终止方式，旧行从不改写。状态文件每次状态转换后先写同目录临时文件，fsync 后用 `os.replace` 原子替换，替换遇到 Windows 共享冲突时短暂重试，运行中每 60 秒写一次心跳。train 记录 `training_pid`，verify 同时记录依赖 train 的 `training_pid` 与自身的 `verification_pid`，并标注两者是否不同。总预算从执行开始按单调时钟计算。

## 与训练入口核对后的两处调整

两个训练入口在新建 train 与 calibrate 运行时都以 `exist_ok=False` 创建输出目录。如果控制器把日志写进 `output_dir`，训练会立即因目录已存在而失败，所以日志统一放在 `run_root/controller_logs`，控制器从不预先创建 train 或 calibrate 的输出目录。

`train_ndr_research.py --mode calibrate` 写出的 `calibration.json` 状态为 `calibrated_engineering_only`，没有 `cuda_budget_eligible` 键，也不记录设备类型。若严格要求该键，NDR-Net 一臂的 CUDA 校准永远无法通过。控制器对这种产物改为读取同目录 `config.json` 的 `runtime.device_type`，只有其值为 `cuda` 才接受，并在状态中把 `cuda_budget_eligible` 记为 null，证据来源标为训练入口未报告资格。这只证明该次运行在 CUDA 上完成，不等于入口给出的预算资格。

## 测试覆盖

`python3 -m pytest tests/remote/test_ndr_matched_budget_serial_queue.py -q -p no:cacheprovider` 结果为 82 passed，`python3 -m ruff check` 对两个新文件结果为 All checks passed。测试只用标准库与 pytest，在 macOS 上运行，子进程是 `sys.executable -c` 的小脚本，不读取数据、不用 GPU、不连接远端。

1. 合法计划准入通过，命令行 `--admit-only` 输出 `admitted_no_worker` 后退出 0，且没有创建 `run_root`，也没有任何子进程留下痕迹。真实训练入口形状的五项计划同样准入通过。
2. 修改字段后不重算摘要、伪造摘要、截止已过、`budget_frozen` 为 false 或其他非 true 值、verify 没有恰好一个 train 依赖、verify 排在其 train 之前，均被拒绝。
3. 另有 41 种 schema 违规与 7 种训练入口旗标不一致，以及重复键、NaN、Infinity、非对象和非 JSON 计划，均被拒绝。
4. calibrate、train、verify 三项按计划顺序串行执行，train 输出目录由子进程自己以 `exist_ok=False` 创建成功，verify 只在 train 的 `summary.json` 存在后运行，训练 PID 与验证 PID 分别记录且不同，并与子进程自报的 PID 一致。
5. train 退出码为 0 但状态为 `stopped_resumable` 时判为失败，verify 不启动。中间 job 非零退出时队列停止，已产生文件全部保留。成功产物缺失判为失败。
6. 永不退出的假子进程在单项超时后只被 terminate 一次，没有 kill，终态写入状态与 journal，后续 job 不启动。真实休眠子进程在 1 秒超时后被终止。
7. GPU 空闲显存不足或 provider 抛错时按 `busy_wait_seconds` 分步等待，假时钟越过截止或总预算后以 `stopped_deadline_or_budget` 停止，退出码 3，没有任何子进程。provider 先失败、读数为 NaN 或低于门槛，再达到 12000 MiB 后，job 正常运行。截止在子进程运行中到达时只终止自己的子进程。
8. journal 每次启动恰好一行 `spawned` 与一行 `exited`，每次状态写入时的 journal 内容都是下一次的前缀。状态文件全部经同目录临时文件加 `os.replace` 写入，结束后不残留临时文件。
9. CUDA 校准资格为 false 时判为失败。NDR-Net 校准产物在 `config.json` 记录 CUDA 运行时被接受且资格记为 null，记录 CPU 运行时被拒绝。
10. 准入后计划文件被改动时拒绝执行，不创建任何目录。默认 provider 能解析多行输出并取最小值，空输出与命令缺失都视为不空闲。控制器源码不含 `os.kill`、`taskkill`、`psutil` 等终止他人进程的调用，在全新隔离进程中导入只引入标准库模块。

## 已知边界

1. 终止只作用于 Popen 直接子进程，没有使用 Windows Job 对象。若 `python` 指向 venv 的转发启动器，真正的解释器是其子进程，超时终止可能留下孙进程，记录的 PID 也不是训练进程自身的 PID。真实计划应让 `python` 指向实际解释器，例如此前 strong4 使用的 `C:/Python313/python.exe`。
2. GPU 门禁只看空闲显存，不识别其他占用者，也从不停止任何其他程序。用户保留的游戏等程序只会让控制器继续等待，直到截止或预算耗尽。
3. 总预算从执行开始计时，GPU 等待计入预算。超时后的 terminate 与 kill 宽限期默认各 30 秒，最坏情况下实际结束时间会略晚于截止。
4. 控制器不复核 checkpoint 内容、数据身份、源码快照或验证集一致性，这些仍由训练入口的独立验证与后续完整保全和 postflight 负责。控制器的成功状态不能替代源端 postflight、Mac 逐文件复验或任何科学判读。
