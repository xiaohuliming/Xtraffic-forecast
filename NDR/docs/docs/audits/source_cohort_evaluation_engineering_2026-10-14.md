# 两源真实模型冻结评测入口

日期：2026-10-14。

本次补齐 SharedGraphNDR 与 SharedSTAEformer 的真实 checkpoint 比较入口。尚未进行真实目标区域评测，不存在新的 test 性能或泛化结论。

新增实现为 `ndr_research/source_cohort_evaluation.py`，CLI 为 `scripts/evaluate_ndr_source_cohort.py`，测试为 `tests/ndr_research/test_source_cohort_evaluation.py`。没有修改旧 transfer evaluator、transfer data 或正在排队的部署文件。

## 真实模型与准入

入口区分 `shared_graph` 和 `shared_staeformer`，对应实际类 `SharedGraphNDR` 和 `SharedSTAEformer`。准入时调用各自已有的 schema-one checkpoint loader，核对内容 marker、配置、完整预算 summary、原 source validation verification 和独立进程证据。实际构造声明的类并执行 strict state load，防止只修改 class 名称。

两个方法各需 seed 42、1、2，共六份 checkpoint。同一方法的三个种子必须具有相同结构、训练配方、源码、运行环境和 source pool 数值语义。两个方法必须具有同一 source_regions、heldout fold、缓存 bundle、source template、source residual scale 和拟合语义。统计比较排除访问收据与 authorization ID，保留真实数值哈希和拟合边界。

正式准入要求全部 source train epoch，30 或 100 个完整 epoch 的 final summary、完整两源 validation、source-only selection、未使用 target/test。原 selected collection 与 source verification 必须具有相同 origin、样本、有效点、target 和 mask 哈希，并复现 MAE。CPU limited-batch smoke、未完成训练、subset validation 或错误 class 均不能准入。

报告保留每个真实模型的 class、kind、family、训练种子、训练范围、参数量、完整训练源码快照、运行环境、训练配方、source statistics 和数据身份，以及 checkpoint、summary、verification、独立证据和 validation selection 的哈希。

## 独立 source 验证证据

`certify-source` 必须在新 CLI 进程执行。checkpoint 必须早于该进程启动时刻存在。它仅重建两个 source reader，复算完整 source validation，核对原运行环境、源码、数据和选集，不构造目标 reader。

该模式可以直接承担训练后的最后一次独立 source verification。省略 `--verification` 时，在新 output 目录同时生成旧 schema-one 兼容的 `verification.json` 和 `source_certificate.json`，共用一次完整 source validation，不要求先运行旧 verifier。提供旧 verification 时，绑定其原始字节哈希，并保持原文件不变。

证书 schema 为 `source_cohort_independent_validation_v1`，状态为 `certified_independent_complete_source_validation`。它绑定真实 class、checkpoint、summary、原 verification、源码、source pool、runtime、data identity、全部 selected source observations、验证 PID、进程启动时间和 checkpoint 修改时间。若旧 checkpoint 已记录训练 PID，还要求验证 PID 与训练 PID 不同。

若已具有真实独立 runner 新进程收据，可用 `source_cohort_runner_process_receipt_v1` 代替新证书。收据状态为 `recorded_independent_complete_source_validation`，需相同哈希和独立进程绑定，`verification_execution` 为 `fresh_runner_verify_process`，`runner_mode` 为 `verify`，`returncode` 为 0。它复用已经绑定的完整 source verification，无需额外复算。CLI 使用 `--independent-source-certificate` 指向这份独立证据，manifest 中字段名为 `certificate`。

## 冻结 gate 与预算

gate schema 为 `source_cohort_frozen_v1`。创建 gate 需要模型 manifest、目标 region、authorization、结构开发已关闭的显式声明和共享 ledger。创建阶段不会构造目标 reader。

gate 固定六个 checkpoint 的全部准入身份、三种子、event/test 采样协议、五个缺失场景、两个 source-only control、perturbation seed、迁移模式和 test 预算。evaluate 阶段不能覆盖这些设置。

账本沿用 RegionalEvaluationLedger 的 hash chain 和独占写锁。每次 cohort 访问消耗 3 个 distinct training seed inference slots。默认计划为三种子、source-only 与 historical-calibrated 两个模式、各一次独立复算，共 12 个 slots。若明确仅做 source-only，两次访问共 6 个 slots。第一次 reservation 冻结预算，失败也消耗 slots。

family 由目标 region 和两个真实模型家族组成。它不包含 checkpoint seed、gate 路径、authorization、迁移 mode 或历史 prefix。这些变更不会重置 family 预算。完整研究必须使用同一共享 ledger。

## 配对评测

完成 gate、源文件和真实模型加载检查后，先 reservation，再构造 role 为 target 的 reader。所有 test 访问都具有 checkpoint-cohort-bound TestEvaluationGate。

每个目标窗口仅读取一次，然后在六个真实模型与两个固定 control 上复用相同的原始目标、mask、origin 和历史缺失扰动。forward 只接七个历史与静态字段，不传 target、target mask 或 incident。缺失场景为 clean、10% random missing、30% random missing、末尾 3 步缺失和末尾 6 步缺失。

指标通过 MaskedForecastSums 以 float64 累加有效观测误差。零 target 参与 MAE 和 RMSE，仅从接近零分母的 MAPE 中排除。各模型具有相同完整 collection 哈希，并使用缓存真实 times_ns 对应的日块保存误差和有效点总量。

历史校准仅使用明确授权的固定目标前缀，没有梯度更新。两个真实模型共用该模式下的历史 fallback、climatology 和 runtime scale。SharedSTAEformer 的内部 scalar center 和 scalar scale 仍来自 checkpoint 中冻结的 source pool。persistence 与 pooled climatology control 始终重建 source-only preprocessing，不能悄悄获得目标统计量。

推理前后检查所有模型参数和 buffer、constructor、eval mode、gate、checkpoint/summary/verification/独立证据、training source snapshot、evaluator source snapshot 和 source pool。失败不生成成功报告，并在 ledger 记录 failed。

## 报告接口与工程证据

成功报告状态为 `frozen_source_cohort_evaluation_complete`。顶层保存 `family_sha256`、`family_identity`、`family_scope`、`region`、`heldout_region`、`source_regions`、`training_seeds`、`mode`、`evaluation_mode`、collection 和 models。family scope 为 `two_source_model_families_and_heldout_across_seeds_modes_prefixes`。`models[id]` 保存 collection、五种 scenarios、真实模型身份和训练来源。

新增 9 项纯合成测试通过。联合 STAEformer transfer、transfer evaluation、indomain evaluation 和 cohort 相关回归共 85 项通过。Ruff 与 compile 检查通过。合成测试使用临时目录中的合成准入产物和 synthetic target reader，没有真实 Orange 或 test 数据。

独立只读审查未发现实质问题。另一审查进程用临时合成 ledger 验证 model ID、checkpoint hash、authorization、gate 路径、mode 和 prefix 不会重置 family，失败消耗 3 个 slots，首个 reservation 锁定 12 个 slots，四次访问后拒绝新 reservation。审查收据位于主项目 `outputs/remote_runs/ndr_generalization_20261013/control/source_cohort_readonly_review.json`，SHA-256 为 `c9845a3685c02b91f6cefc6cbc7dbe0f1f96e747e2e9426ff77068cc44976d30`。这项审查没有打开真实缓存或使用 GPU。

另一个真实新 CLI 进程加载已有有限 CPU smoke checkpoint 时，被完整正式 source-only training 门槛拒绝。source 或 target reader 均未构造，没有生成 certificate output 目录。工程收据位于 `outputs/remote_runs/source_cohort_engineering_20261014/engineering_receipt.json`。

本次没有 GPU 操作，没有真实目标标签或完整真实 test 指标。后续方法比较属于整体模型比较，包含主干、空间传播、输入统计使用和 loss 的差异，不能解释为 gate 的独立贡献，也不能据此声称超过现有 SOTA。
