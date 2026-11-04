# NDR 原六项工程校准的完整取回验收

日期：2026-10-15，Asia/Singapore。

本轮增加纯本地验收入口 `scripts/admit_ndr_transfer_engineering.py`，没有改动已经冻结的 43 文件成本测量包、原训练代码或原八项和六项计划。真实六项仍等待前置训练完成，本轮没有执行真实取回或 CUDA 工作。

该入口只接受原六项顺序和 plan digest。完整文件树包含四个任务目录、六份日志、状态、历史 recovery 控制文件及 engineering postflight 自身。两个独立 verifier 的结果位于对应 smoke 目录，不能误算成额外模型或独立任务目录。

源端前后清单都通过文件大小和 SHA-256 检查，两次 UTC 观察严格递增。结果明确证据范围为两个端点的文件哈希一致。部署 Git 与原回执保持 `734fd219`，不改写成当前本地源码身份。单份 inventory 只能支持本地结构结论。

本地默认检查四个 best、last checkpoint 的 schema、配置、模型参数、真实 Adam 状态、scheduler、loader 游标及 RNG 结构，并核对三十轮限定训练、完整 source validation、选中 best 与独立验证的配置、哈希、有效点数和指标关联。它不读取真实 H5 或 Orange，不重复完整 validation 推理。旧验证回执没有 PID 字段，独立性依据为冻结的串行 train、verify 计划及完整控制结果，不报告已重新复核独立 PID。

回传包装保存在原工作区 control/retrieve_transfer_calibration_complete.py。它只在源端六项完成、postflight 通过、代码和部署身份一致、真实进程退出后复制。进程查询错误不会被当成退出。复制使用新 staging，源端只读，逐文件复验后才提升目录并写完成回执。提升失败保留 staging 和明确的待提升回执，不留下虚假的成功记录。

两项旧 cost 的 validation 仍只有前向，有限训练的内部 elapsed 也不包含完整启动、预处理、独立 verifier 和取回。完整训练、单独 validation、checkpoint I/O 和 verifier 的预算继续为 null，不因工程验收通过而填成实测成本。

新增二十八项合成验收检查通过，传输和完整合成检查点提升集成的十六项检查通过。Ruff、编译和 diff 检查通过。JUnit 与范围回执位于原工作区 outputs/remote_runs/ndr_generalization_20261013/control/。这些是工程准备证据，不能写成真实六项完成或新的精度、泛化结论。
