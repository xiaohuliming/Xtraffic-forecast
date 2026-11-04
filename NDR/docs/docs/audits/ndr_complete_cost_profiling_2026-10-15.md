# NDR 后续完整成本测量入口

日期：2026-10-15，Asia/Singapore。

## 范围

本改动位于独立分支 `codex/ndr-cost-profiling`，基于 `cc38299bff1b4086594c788fa38272af16f2f597`。原八项训练、后继六项校准及其部署文件、顺序和绝对截止保持原身份。本改动尚未部署或启动 CUDA 工作，不产生新的模型性能结论。

三个入口增加显式 `--profile-closeout`：SharedSTAEformer 两源训练入口、SharedGraphNDR 地区内入口和原 STAEformer、BiST 强对照入口。该选项只允许校准，必须覆盖完整 source validation，不允许限制验证 batch，也不读取 Orange 或真实 test。原训练数学及默认校准行为保留。

## 实际测量与产物

先完成原校准的真实优化步，再调用原生产 evaluate 路径计时。该路径包含尾批、实际观测哈希、全局 masked 指标、horizon 归约和预测传输。随后使用原保存路径写出真实模型、Adam、scheduler、随机数和 loader 状态，测量 payload 构造、torch.save、flush、fsync、内容哈希、原子改名和 marker 的整体时间。CUDA 计时前后同步。

新增产物为 `calibration_probe.pt` 和 `calibration_closeout.json`。检查点使用专用成本 schema，标记 `non_resumable_cost_probe`，完整训练 epoch 为零，不伪造 epoch 游标或训练 history。即使改名为 `last.pt`，训练或 resume 也会在 runtime 配置和数据读取前拒绝。

完整生产验证结束后的下一独立进程才执行 `--mode verify`。同 PID 被拒绝，验证结果使用专门的工程状态，不能进入正式训练验收或冻结 test 门。CPU 回执明确 `cuda_budget_eligible=false`，不能填入 5080 训练预算。

目前只实现测量入口。CUDA 成本、外部完整进程耗时、预处理、回传以及未来完整训练 history 的检查点增长仍须实测或单独保留预算，不能用旧前向耗时估算代替完整成本。

## 部署依赖修正

隔离检查实际复现此前 41 文件候选 ZIP 导入失败。`fourier_dual_net/rgdn.py` 修改 sys.path 后导入裸模块 `model`，普通包名 AST 扫描漏记了 tracked 文件 `baselines/GraphWaveNet/model.py`。该候选 ZIP 与旧审计回执保留，不能部署。

三个入口的 source snapshot 已显式纳入该依赖，新包还须包含 `ndr_research/cost_profile.py`。包验收须从独立解压目录移除旧工作区 PYTHONPATH，再运行真实 import、parser 和 validator，不能仅核对成员哈希。

## 后继草案与验收

新草案保留原九种配置，每项成本测量后串行运行独立 verifier，共十八个进程步骤。原八项和六项完整源端与本地验收、原进程句柄退出、严格 GPU owner 和 pristine BiST 身份仍为前置条件。草案不是执行计划，正式预算和截止未冻结。

合成 CPU 验证覆盖真实 Adam 状态、完整尾批、实际保存与哈希、同 PID 拒绝、改名检查点的续训拒绝、独立新进程复验以及默认产物契约。测试回执、命令参数验证、隔离导入反例和新包验收位于原工作区 `outputs/remote_runs/ndr_generalization_20261013/control/`。它们是工程证据，不能代替完整训练、多种子确认或泛化结果。


## 串行控制器完整准备，2026-10-15 04:53

独立控制提交为 `128450ccd88ed09b744472d00f0ef9dcc66102db`。新增十八步串行 runner 和合成测试，最终 runner SHA-256 为 `9c50b1630a43907b1577f0da60f3549cb79165cb5466b3887011609e296d1aee`。已有二十一项检查的 JUnit 为零 failure、error、skip；本轮核对测试与代码字节和独立静态审查，没有重复全仓测试。Provider 符号链接、启动前截止复查和 BiST 目录重叠三项问题已修复。

独立控制 ZIP 只含 runner 和标准库验收 helper 两个文件，20,631 bytes，SHA-256 为 `20f6318e4c0a2e1e2ec121c0dde98376bb41c32648cd69c5ef7e9957b072b2f3`。隔离解压后的真实 CLI help 成功，未冻结草案在 worker、GPU、模型或数据读取前被拒绝。

已将相同两文件准备到 5080 独立控制根 `C:/XTrafficResearch/ndr_cost_20261015/control_128450c_81062a3497`，源端隔离 help 和后续独立回读哈希通过。原四十三文件 producer 与 pristine BiST 在准备前后均保持精确哈希和 clean Git 身份。没有写正式执行计划、注册任务或启动新 CUDA 工作。实际训练工作区继续为 clean cc38299。

第一次内联长命令返回不可解码的 Windows 错误。只读检查确认该新根只有 ZIP，没有解压出的控制文件或完成回执；随后改为上传冻结标准库脚本并在短命令中核对哈希后运行，避免 Windows 命令长度限制。原始失败和部分上传清单保留，没有将失败写成准备成功。

新草案 `control/next_cost_closeout_source_and_control_ready_draft_20261015.json` 记录代码与独立控制根，十八项 timeout、绝对截止和正式预算仍未冻结。完整八项与六项前驱、Mac 完整验收、真实旧句柄退出和严格 idle 条件仍须在实际 dispatch 前满足。控制准备总回执为 `control/cost_control_preparation_closeout_20261015.json`，该状态不产生模型性能、泛化或 SOTA 结论。
