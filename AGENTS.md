# XTraffic Agent 入口

开始工作前依次阅读：

1. `CLAUDE.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`

当前 XTraffic 的旧缓存、v2 缓存和既有结果均不得写成已验证论文结论，也不得覆盖旧 summary。v2 已冻结。除原有泄露问题外，复审还确认了重复 forecast origin、active event 跨 split、验证指标累计错误和训练契约不完整。5080 已停止 v2 旧队列，最先完成的 4 个 v2 结果只可视为 `contaminated preliminary`。

当前修复协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`。最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。纳入 `origin/fu` 的合理修复并补齐 Windows 路径测试后，Mac、5080 与 5090 均为 143 项通过，1 个既有 warning。三地区正式缓存已在 `outputs/dist_net/region_data_v3_noleak_dedup` 落盘，并通过本地两次、5080 一次及 5090 一次严格复验。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。

2026-07-20 22:23 新加坡时间起，正式运行口径由“5090 同机 45 项”改为双机分层矩阵。两台机器保持独立结果根、独立 `config_hash`、独立 postflight 和运行时 provenance。最终只能称为 `host-stratified distributed 45`，不得称为 homogeneous 5090 matrix。固定分片不交叉且并集严格为 45 项，所有直接 A/B 比较组保持在同一台机器。

2026-07-22，用户批准的 27/18 重分片已于 22:49 新加坡时间完成原子切换。5080 分片共 27 项：tasks001-004、007-010、013-016、019-030、040-042。5090 分片共 18 项：tasks005-006、011-012、017-018、031-039、043-045。两分片互斥且并集严格为 canonical 45，所有直接比较组保持同机。task039 完整落盘后通过 `BOUNDARY_RESULT_OK`，relay ready marker 已发布且保持完好。

5080 的旧 24 项队列已于 2026-07-22 18:13:06 新加坡时间完成 24/24，skipped valid 为 4，runner 退出码为 0。旧 24 项 source-host postflight 与 134 文件 allowlist 已通过。task007 的首个 attempt 因两条遗留诊断 PowerShell 扫描耗尽主机内存而失败，没有 summary 或正式任务目录，第二个 attempt 已完整重跑并验收。这是主机运维故障，不是模型失败。旧 task005 迁移记录不进入正式导出 allowlist。

截至 2026-07-27 11:19:49 新加坡时间，runner 层已完成或重新验收 45/45。5090 为 18/18，5080 为 27/27。`rtx5090_18` 与 `rtx5080_27` 均已在结果服务器完成逐文件复验和正式原子提升。只读 composite index 尚未完成，论文分析仍暂停。

task040 旧失败的根因是 producer float32 `.mean()` 与 runner float64 audit 的精度语义不同。新控制层保留 float64 audit，只对 graphwavenet、fourier_dual_net、staeformer 与 staeformer_icsf 精确复现 producer，summary 必须处于 2 ULP 内。恢复路径只接受 prior queue、failed task state、最新 zero-exit attempt、canonical path、非空日志和完整 artifact 全部通过的候选。修复提交为 `dfb6b7f` 与 `33fae6c`，本地全仓为 247 passed 和 1 个既有 warning，独立复审未发现 P0 或 P1。

部署探针曾发现 CMD 环境变量展开问题。Python preflight 拒绝了合成 task ID，没有 trainer 启动，但 queue 被短暂改写。异常快照已保存，SHA-256 为 `bdcb4718859b8dfbaf52d588e2d5af2789dcec42659f8927879807fd120869ab`。queue 随后根据多份落盘证据原子恢复，task state 与 artifact 未改变。最终实现由 Python 直接读取环境变量，CMD 不再展开其值。

task040 已于 11:23 新加坡时间从 `attempt_0001` 完成可审计 recovery，正式 task state 为 completed，没有 `attempt_0002`。summary、ckpt、prediction artifact 与 train log 哈希均与恢复前一致。

task041 在完成 30 个 epoch 与测试推理后，因旧 summary 临时路径达到 282 字符而触发 Windows 传统路径限制。checkpoint、prediction artifact 与 train log 完整，失败不是模型或数据问题。提交 `e7bf755` 缩短临时名并增加 incident-specific repair 与 completed state 幂等保留，本地全仓为 249 passed 和 1 个既有 warning。5080 dry-run 完成正式 cache 全量对照后，task041 于 17:08 从原 `attempt_0001` 原子恢复，staging 与正式目录均通过 runner 验收，没有 `attempt_0002`。summary 与 decomp mask SHA-256 为 `115c0cfba5d6a256a27011a6169797ac48606cb90a914a942919854e399e27ac` 与 `55ae377b62287d5098a895c5608cb8050d79520df2a5936f0754c0a18bef5d5a`，原三份 artifact 哈希未变。

2026-07-26 17:10，runner 以 `KEEP_COMPLETED` 保留 tasks040-041 的恢复状态并启动 task042 `attempt_0001`。17:15 的落盘日志为 epoch 1、batch 1500/4598，GPU 使用 10,799 MiB，利用率 97%，温度 63 摄氏度。runner、trainer 与 finalizer 当时各有 1 个进程。

task042 于 2026-07-26 23:32:54 新加坡时间完成并通过 runner 验收。all 与 unaffected masked MAE 均为 `10.780716896057129`，affected 为 `null`，best val masked MAE 为 `10.51922192820698`，有效点数为 `103270922`，参数量为 `1536372`。5080 队列由此达到 27/27，runner 层达到 45/45。

5080 finalizer 首次执行结果服务器目录树复验时遇到 PowerShell 至 Windows native ssh 的引号传递问题。该故障不影响训练结果、source-host postflight 或已上传文件。提交 `c48f232` 完成修复，本地全仓为 250 passed 和 1 个既有 warning，PowerShell 5.1 AST、Windows native argv 与真实 incoming 文件树复验均通过。finalizer 复用 152 个已上传文件，没有重训、重传或改写任务结果。`rtx5080_27` 于 2026-07-27 11:19:49 新加坡时间正式发布，transfer marker SHA-256 为 `ddf097ddd206efbd0d487e21e608c688945c7dc2dadd3f307d739e1508221c06`。

首次 composite 复验暴露结果服务器内存峰值风险。提交 `5f3113b` 在 float64 audit 后原位复用 prediction 缓冲计算 float32 producer error，并在首次指标校验前释放 actual。提交 `93038ec` 跳过 full-window 中未使用的事件分组归约，同时完整保留 `all` 指标契约。两项修复不改变 artifact、summary、训练 checkout 或任务 `config_hash`。本地全仓为 252 passed 和 1 个既有 warning，独立复审未发现 P0 或 P1。

下一步仅生成并验证只读 composite index，再执行同机配对与多种子统计。不得在 composite 与统计完成前恢复论文结论。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。
