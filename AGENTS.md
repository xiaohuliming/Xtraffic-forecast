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

截至 2026-07-22 22:56 新加坡时间，runner 层完成或重验 39/45。5090 新队列为 15/18，skipped valid 为 15，正在 task043。task043 已进入 epoch 2 batch 1100/2299，GPU 利用率为 83%，显存为 5111/32607 MB。5080 新队列为 24/27，skipped valid 为 24，正在 task040。task040 已进入 epoch 3 batch 1200/2299，GPU 利用率为 70%，显存为 3212/16303 MB。

5090 新 runner PID 为 241437，新 finalizer PID 为 241585。旧 runner、旧 finalizer、边界 watcher 与 transition 均已按设计退出。新 5090 分片继承 15 项，从 task043 开始，上传名为 `rtx5090_18`。5080 新 runner PID 为 17260，新 finalizer PID 为 6152，边界等待器已正常退出。新 5080 分片继承 24 项，从 task040 开始，上传名为 `rtx5080_27`。5080 成功上传后保持开机。

控制提交依次为 `60405a9`、`e191c27` 与最终退出码文件实现 `30bcb81`。训练代码 SHA 仍为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。本地全仓测试为 230 passed、1 个既有 warning。5080 的正式 PowerShell 已通过 SHA-256、AST 和真实 relay 冒烟。用户授权处理容量后，结果服务器仅清理了未使用的 Docker build cache，Build Cache 从 12.25 GB 降为 0B，可用空间增至 19,434,708,992 字节。4 个业务容器不受影响，relay ready marker 保持完好。两个上传分片尚未出现，这是训练未完成时的正常状态。5080 relay 私钥只保存在工作站用户目录。任何训练、复验、空间、传输或哈希错误都保持相应机器开机。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。
