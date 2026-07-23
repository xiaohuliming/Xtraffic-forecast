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

截至 2026-07-23 13:50 新加坡时间，runner 层正式完成或重验 42/45，剩余 tasks040-042。5090 新队列已完成 18/18，`rtx5090_18` 已回传并通过 marker 复验。source-host postflight 于 2026-07-23T04:40:34Z 验证通过，导出包含 97 个 source files、10,071,444,446 source bytes 和 18 个 task dirs。5090 已按计划关机，SSH 不可达。

5080 task040 已完成 30 epochs 和 inference，但 summary 的 float32 MAE 11.305081367492676 与 runner 使用 float64 重算的 11.305101714317631 相差 2.03468e-5，因此被 runner 拒绝。队列以 failed 24/27 和退出码 4 结束，没有正式 task040 目录，也没有 postflight 或上传。`valid_count` 为 103270922，artifact 身份、mask 与 finite 检查均一致。这是指标精度契约问题，不是数据泄露或模型失败。

正式 task040 尚未产生，不得擅自写成已修复或已验收。完整 staging 包含 summary 7,895 字节、ckpt 1,306,955 字节、test_predictions 483,524,632 字节和 train.log 3,420 字节。确认无协议进程后，已写入 `C:\Users\ASUS\protocol_v3_5080_shutdown_state.json`。5080 于 13:49 按用户要求正常关机，SSH 和 ping 均不可达。结果服务器当前可用空间为 9,322,172,416 字节，`rtx5080_27` 不存在。5080 relay 私钥只保存在工作站用户目录。下一步是审查并修复控制层浮点精度验证与 staging recovery，重新开启 5080 后复用 task040 artifact 并运行 tasks041-042。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。
