# pooled Graph H240 确认执行外层

本文件只记录已获选路线的本地工程契约。没有新的训练计划，没有冻结运行预算，没有远端准备或 GPU 操作。`production_launchable` 保持 `false`。用户保留游戏负载，后继 GPU 不得启动。

## 固定范围

完整七项发现分析 `62cd73dd` 已按原 `2ccd8c9a` policy 选择 pooled Graph H240。regional64 与 regional240 失败，不进入原配方确认。seed42 只构成候选资格，不构成 Orange 泛化、充分收敛或 SOTA。

确认只新增四次训练和四次独立验证。顺序固定如下：

1. seed1 Graph H240 训练，然后独立验证。
2. seed1 SharedSTA 训练，然后独立验证。
3. seed2 Graph H240 训练，然后独立验证。
4. seed2 SharedSTA 训练，然后独立验证。

每项 argv 从原 `b4f11e80` 计划的相应两种任务复制，只改 seed 和新输出路径。Graph 保留 effective16 与 micro16。SharedSTA 保留 effective16 与 micro2。两者保留 FP32、30 epoch、全部训练样本、全部验证观测点及原统计量拟合范围。完整四条模板的规范 JSON 摘要由 `TEMPLATES_SHA` 固定。

原 seed42 两项使用 `a6ad860c` 完整回执和完整七项逐文件树。源端与 Mac 都不重复反序列化原 seed42 权重。

## 执行与保全

模型生产仓库仍为实际 clean `3b09dbf1`，位于原 source43 路径。新增 controller、collector 与 OS 库必须在独立 control code root。不能把新增文件塞进模型仓库，也不能把当前工作树 `cc38299b` 写成 source43。

`graph_h240_confirmation.py` 提供固定八项调度、严格计划结构、四前驱纯联接，以及仅 seeds1 和 2 的显式原生 schema 校验。原 `ndr_formal_development_queue.py` 的 `native_checkpoint` 保持原字节。新增 `validate_native_pair` 明确适配 seed，不改 trainer、模型、数据加载器、推理器或原 validator 的 seed42 逻辑。

外部监督复用 `shared_sta_norm_serial_queue.py` 的 `DeadlineState`、原 `watchdog_function`、`DeadlineClient`、基础设施捕获与原 Handle 等待。原 `engine`、`dispatch`、`fixed_launcher`、`run_queue` 和原始终态观测函数码不变。它们使用 Graph 自己的 stage 与八项身份，不能把 n8 四项账本当作确认完成。

隔离 namespace 的覆盖仅包括 Graph 的 plan、state、postflight、native 与 inventory schema，`SCOPE`、`STAGES`、环境标识、`__file__`，以及 `validate_plan`、`expected_jobs`、`admit`、`_preliminary`、`source_native_validate`、`validate_terminal` 和 `_collector`。`engine` 的两个扩展回调通过显式参数传入。原 OS 与 Handle 函数的码和常量由测试逐项核对。

控制入口使用 `-B` 并向原命令继承 `PYTHONDONTWRITEBYTECODE=1`。原训练 argv 不因此改变。原始 `before_popen`、`owned_handle_created_before_identity_query`、身份、`owned_wait_result_observed` 与 `owned_wait_return` 必须顺序完整。原 Handle、创建代次、严格 prewait image、原 wait 返回值与 GetExitCodeProcess 都要一致。未知值保留 `null`。新鲜 exit 观测不补写旧 wait。最终 native artifact 必须与训练结束和验证结束时原始捕获的 SHA 相同。超时只终止本次已证身份的独占 Job。

`retrieve_graph_h240_confirmation.py` 复用已固定 opaque collector 的路径、精确树、原 Handle 与双 frame 检查。源端八项结束后对每份新 best 与 last 各反序列化一次，构造原模型并严格载入完整 state。该过程不重新推理。Mac 仅校验 JSON 和全部 opaque 字节。

完整八项、源 postflight、全部原 Handle 等待、实际基础设施终态和完整逐文件 Mac 保全均通过后，`family6` 才读取原 seed42 与新 seeds1、2 的六个验证文件。每个 seed 的两个地区必须各自严格改善。平局或任一地区失败都不能被种子均值、地区均值抵消。输出保留诚实负结果。

## 已完成 strong4 前驱联接

实际复用文件是 `scripts/remote/windows/mage_train_only_cost_admission.py`，SHA `a81b0db2`。只使用 `predecessor_admission` 和其中的 `strong4_completed_admission`，不依赖 MAGE 模型或 MAGE 成本结果。

Root 已明确批准的唯一标量覆盖是私有 `FunctionType` namespace 的 `SUCCESSOR_SHA`，由旧 `eec22a84` 改为新 `5f951b3d`。原 helper 模块字典保持完全不变。两个原函数共享私有 globals，函数码、defaults 和 closure 不变。私有函数引用仅连到这两个相同原函数码的克隆。其他全局身份不变。cost 的 `admission_helper` 必须与显式 `successor_adapter` 为同一完整 path 与 SHA 记录。

`strong4_completed_admission` 原样调用 `collector.load_frozen`、两份 `validate_inventory`、原源目录两次 `validate_copied`、实际 source 与 author Git、完整 source 和 control 文件树，以及真实 exit 观测。它不调用 fresh `q.admit`、`admit_stage` 或 `admit_strong4`，所以能验证已存在结果根和过去 deadline 的原完整终态。

前驱顺序保持 complete7、numeric、DualCast 完整 cost 的成功或失败保全、strong4 fixed30 完整成功保全。numeric 已有真实完成回执。cost 与 strong4 尚未运行，因此当前不能通过实际生产 admission。

## 预算与依赖

每项 timeout、总预算、当前 boot 与未来绝对 UTC deadline 必须来自将来明确冻结的独立计划。代码没有默认训练预算，没有自动恢复、自动重试、自动后继、自动传输或关机。旧 n8 的预算不会被移用。

完整实际代码依赖与每项 SHA 见相邻 `graph_h240_confirmation_runtime_closure_2026-10-16.json`。其中现有 source63、source64 路径只来自已落盘的历史 source 准备回执，本次未做远端刷新。source43 的 43 文件清单保持完整独立。

当前 `cc38299b` 工作树原本缺少 `ndr_research/cost_profile.py`，另有三个 trainer 与 source43 清单不同。这是既有分支差异，本次没有改它们。因此不能从当前工作树直接运行生产模型，也不能静默更换生产 checkout。

CPU 依赖临时副本位于被忽略的 `outputs/confirmation_cpu_dependencies`。Windows 不使用这个 fallback。完整独立 control 包尚未准备或上传，真实 Windows 八项控制运行与真实数据保全仍需前驱完成后另行验收。
