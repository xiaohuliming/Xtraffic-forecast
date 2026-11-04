# SharedSTA 因果历史归一化工程准备

日期：2026-10-16，Asia/Singapore。

新候选 `SharedSTAEformerCausalNorm` 仅替换 SharedSTA 的输入归一化和输出逆变换。当前 12 步中有效历史的 FP32 均值用于中心化，population standard deviation 用于尺度。有效点不足两个，或标准差低于源尺度的 0.05 时，沿用固定源尺度。无有效点时沿用源中心。阈值固定，未根据真实数据或验证分数选择。

原 `canonical_history` 的因果填充与源气候态回退保持不变。原 native backbone、直接预测头、日历输入及 1061172 个参数的 state 布局均不变，没有新增可学习参数、buffer、mask 通道或 last-value skip。leading missing 的源气候态填充及源尺度回退会限制仿射性质，不能声称所有缺失场景都具有全局仿射等变性。

模型与测试已窄提交 `726f09262e28118d3ebc2742f5d81cee7530fb3c`，分支为 `codex/shared-sta-causal-norm`。工作树位于 `/Users/xhlm/.codex/worktrees/shared-sta-causal-norm-sparse/科研实习`，这是有效 CLI Git 工作树，尚未成为 App 管理附件。原成本工作树与原模型文件没有修改。

作者专项 43 项 CPU 检查通过。独立代理另复跑这 43 项，并增加 10 项独立 oracle，独立验证共 53 项，不与作者重复运行相加。检查使用合成历史和真实 native 模型，覆盖有效历史统计、阈值两侧、缺失 sentinel、因果填充、变节点、布局、重载与所有 native 参数的有限非零梯度。未读真实缓存、当前 formal7 部分指标、Orange 或最终 test。

独审回执为 `control/shared_sta_causal_norm_independent_final_review_20261016.json`，SHA-256 为 `aeea4d3db17d6a821366e37f513ee04eec5c7d9ef9180137193e084024668300`。Root 核对两文件与提交 blobs、三个原基线依赖和独立 JUnit 的回执为 `control/shared_sta_causal_norm_root_commit_review_20261016.json`，SHA-256 为 `cea62569ea362ee0a2975d9bdfe2001b765e8bd5583d99b56eb16eaa579f6b41`。这里的 control 均相对 `outputs/remote_runs/ndr_generalization_20261013/`。

由于两类模型刻意具有相同 state 布局，单靠 `strict load_state_dict` 不能拒绝跨模型混用。独立双臂入口与专项测试已窄提交 `0bff2880ee734f8ed9033debb4d3d6ab63b51000`。新 schema 为 `shared_staeformer_norm_transfer_complete_v1`，绑定 class、profile、method、constructor、源统计与数据身份，保存完整 Adam、scheduler、RNG 及 train、val loader。`research_spec` 是可变描述元数据，不能单独作为可信运行身份。

该入口保持原 effective batch 数学、区域顺序与 source reader。static 臂与原 trainer 的真实 native 权重、Adam、cosine scheduler、RNG、两套 loader 与读取顺序在合成 CPU 两轮训练中逐值相同。双臂均通过训练批和部分验证批中断恢复对照。验证累计全有效点与尾批，并保存有序批次 array digest 的独立 collection commitment，不伪装成旧脚本的原始 bytes 串联 SHA。

初版 c064 通过 40 项专项，但独审实际证明缺少一个参数的 Adam 状态仍被接受，selected 逐 horizon 指标及部分验证累计值不一致也可放行。该版本为 failed review，旧 JUnit 与真实反例保留。修正版在反序列化 Adam 自动 cast 前核对完整参数覆盖和 moment 形状、dtype、有限性，恢复时联合核对计数及累计值，独立复验比较全部已报告选中指标。

修正版作者和独审最终各 53 项合成 CPU 检查通过，四个旧实际反例均被拒绝，不与旧 40 项相加。实际 spawn 子进程重算两臂完整合成 source validation 均为零差。独审回执 `control/shared_sta_norm_transfer_independent_final_review_v2_20261016.json` 的 SHA-256 为 `ec25b46cc5147ce88a9aae12034b370921084ebf688a1d6af500d7e1cbe880ec`。Root 已阅读主体和窄修代码，核对原八个依赖、提交 blobs 与全部审查产物，回执 `control/shared_sta_norm_transfer_root_commit_review_20261016.json` 的 SHA-256 为 `f0953ae55f2d4574b2cec6d117c3559027b47311d90f65c9f2d35bf3e250bc76`。Root 未重复最终套件。

条件研究合同 `control/shared_sta_causal_normalization_paired_research_contract_20261016.json` 的 SHA-256 为 `75d66073b42ebe782638e7300d5c9c8a2bde513afbc4bfda6402d170c87b1b1c`。拟配对保持原 pooled SharedSTA 的两源、30 轮、seed42、batch16、microbatch2 和原优化配方，只有 Alameda 与 ContraCosta 的完整有效点 validation MAE 都严格低于 static 对照，才进入匹配 seeds1、2 确认。一个地区的提升不能抵消另一地区的退步。完整双臂验证、Mac 保全与独立预算仍是后续工作。

当前 n8 仍为 pending，没有选入 GPU 训练。合同不是启动计划，没有执行预算、源端准备、CUDA、精度或泛化结果，也不证明真实区域尺度漂移。当前 formal7 配方、截止和部分分数关闭规则保持。

本地静态源码包位于 `control/shared_sta_norm_static_source_0bff2880_20261016/`，含 67 个项目文件、62 个运行静态闭包文件、38 个测试源码闭包文件及全部 26 个 ndr 模块。唯一未检出的 upstreams.lock metadata 从同一 Git commit 的原 blob 加入 payload，未改稀疏工作树。真实隔离导入和 main help 各执行一次，数据、模型、optimizer、CUDA 和网络守卫均无触发，没有复跑测试。

manifest 的 SHA-256 为 `e2a04cb359fd6ac1b64728f77390fe13482cd2363305873c22db805cf01323fe`。Root 逐一核对 67 个 ZIP 成员与同 commit 原 Git bytes、解包精确树、46 项运行 snapshot 成员和全部产物哈希，回执 `control/shared_sta_norm_static_package_root_member_review_20261016.json` 的 SHA-256 为 `b3c4de38d96b2a92ec6fdd763cd3bf4eea5a67c49ee363add69ffe206008f352`。

隔离副本位于主仓库内部，真实 Git 查询检测到祖先 `28172d9d58b051eec7abb42108449c30cc0c1331`、分支 `codex/fdn-benchmark-v1` 和 dirty true，已原样保存，没有设置代码 SHA 覆盖。此身份不是未来部署训练身份，不能使用该 help 结果充当 clean source-host 验收。原候选工作树前后保持 clean0bff，当前包仅完成本地准备，`production_launchable=False`。

随后已在 5080 新根 `C:/XTrafficResearch/shared_sta_causal_norm_20261016/prepared_0bff2880_43a9626ba1` 完成纯代码准备。只上传源码 ZIP、原 manifest、新准备 manifest 与标准库准备脚本，在全新 repo 创建独立最小 Git。67 个文件共 1301958 bytes，逐文件 SHA、Git blob、精确文件集合与 clean 状态通过。实际源端 Git 为 `4180a4645796a4792962e961dddb42e62d2f3aa7`，tree 为 `841aceec4d39aa700ee7a1207d4a84a9bf718b19`，与包来源 0bff 明确分列。

Root 另起只读 SSH 查询重新核对源端全部 67 项、原 manifest bytes、归档和准备脚本及实际 Git。回执 `control/shared_sta_norm_source_host_root_independent_readback_20261016.json` 的 SHA-256 为 `5c47d37b08e807585025aeda31fc37afba4069b1be05c76f3a1bb7d5c2c1eed6`。没有 import 项目、Torch 或 HDF5，没有运行 help，没有读取真实缓存或访问原训练目录、进程与任务。候选仍为 pending，`production_launchable=False`，没有创建训练计划或 GPU 任务。
