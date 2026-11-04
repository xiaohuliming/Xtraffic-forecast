# 5080 后续成本源码与 BiST 准备

日期：2026-10-15，Asia/Singapore。

已在独立新根 `C:/XTrafficResearch/ndr_cost_20261015/prepared_97440a5_d3cbbfb05a` 准备已验收源码与公开 BiST 锁定包。原八项和六项目录、代码、计划和截止没有改变。此操作只使用 Python 标准库、文件传输和 Git，没有导入或构造模型，没有读取 XTraffic 数据值，没有注册任务或启动 CUDA 工作。

源码 ZIP 保持四十三个文件，SHA-256 为 `24d822ee3d8aec86bdf6d7dd6bb450cbf82ea860e47a2b4cf45cc601a014e83a`，来源提交为 `97440a52708f2c3f03169ed4a385080c2129ea77`。源端新建的最小 Git 仓库实际提交为 `3b09dbf199372b356889a21ae9cfa8a95de37ff6`，clean 为 true。两者表示来源身份与实际运行仓库身份，不能互相替换。

BiST 从已验收公开 Git bundle 还原，设置 `core.autocrlf=false`，保持提交 `dd94adf7721fcbb9e3feb5d1b44040305199a4cc`、tree `1dc3c4de312ec8cb2eaf2aafc7b680aa247d19be` 和官方 origin，工作树 clean。八个锁定 Python 文件哈希均匹配。没有运行上游 notebook、engine 或数据转换。

准备后又执行独立只读回读，复核源码四十三文件的精确集合、哈希和 Git 身份，以及 BiST 提交、tree、clean 和八文件哈希。准备回执为 `control/source_preparation_host_receipt_d3cbbfb05aa24038b78c4a85ae0782b5.json`，回读证据为 `control/source_preparation_independent_readback_20261015.json`。

新草案记录了实际源端路径和 Git 身份，仍不是执行计划。原八项、原六项完整源端与本地验收、原句柄退出、严格 GPU owner 检查、实际预算和 timeout 冻结后，才可执行后续九种成本配置及串行独立 verifier。当前没有新的模型性能或泛化结论。
