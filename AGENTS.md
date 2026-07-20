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

5080 分片共 24 项：tasks001-004、007-010、013-016、019-030。runner 已重新验收并跳过 tasks001-004，当前队列为 4/24、skipped valid 为 4，正在训练 task007。task007 的 `attempt_0001` 在 epoch 9 数据装载阶段因主机内存耗尽触发 NumPy `ArrayMemoryError`。根因是两条遗留诊断 PowerShell 扫描占用约 48 GB 私有内存。该 attempt 没有 summary 或正式任务目录，失败记录原样保留。只停止这两条诊断进程后内存恢复，runner 已用 `attempt_0002` 重启 task007，训练与 GPU 均正常。这是主机运维故障，不是模型、数据协议或训练数值失败。截至 2026-07-21 00:02 新加坡时间，`attempt_0002` 已完成 epoch 7，train 为 12.0424，val 为 12.6845，用时 359 秒，GPU 利用率为 92%。旧 task005 的迁移失败记录继续保留，但不在 5080 分片导出 allowlist 中。上述 4 个既有 v3 结果与冻结的 4 个 v2 `contaminated preliminary` 结果必须明确区分。

5090 分片共 21 项：tasks005-006、011-012、017-018、031-045。task005 与 task006 已完成并通过 runner 验收，新分片重新验收后跳过这两项。当前队列为 2/21、skipped valid 为 2，正在训练 task011。截至 2026-07-21 00:02 新加坡时间，task011 已完成 epoch 8，train 为 12.3250，val 为 12.3937，用时 659 秒，并保存当前 best checkpoint。task005 的 all、affected、unaffected MAE 分别为 12.025650、17.932862、11.720226。task006 分别为 11.740712、17.583309、11.438629。两项都只有 seed42，只能作为方向性单种子证据。边界切换时旧队列为 task007 创建了零字节日志和未完成 staging，未产生 summary 或正式结果，postflight 与上传均明确排除。

5090 自动收尾脚本为 `/root/autodl-tmp/protocol_v3_finalize_5090_shard.sh`，当前 PID 为 106466。它只在 5090 分片达到 21/21、skipped valid 为 2、source-host postflight、严格导出 allowlist、结果服务器逐文件 SHA-256 和原子提升全部通过后，将结果保存到 `root@8.134.12.5:/root/xtraffic_results/protocol_v3_noleak_dedup/distributed_45_5080_5090/shards/rtx5090_21` 并关闭 5090。5080 收尾进程 PID 为 12524，它等待 5090 上传标记后上传 `rtx5080_24`，成功后保持实验室工作站开机。两个进程都已通过独立进程检查和持续 WAIT 日志复验。任何训练、复验、空间、传输或哈希错误都保持相应机器开机。

结果服务器当前约有 6.3 GB 空闲，双分片粗估约 11.6 GB，因此容量仍被阻塞。服务器另有 12.25 GB 未使用 Docker build cache 可回收，尚未获得明确清理授权。5080 使用的专用 relay SSH 私钥只保存在工作站用户目录，不进入仓库。调度、分片、postflight、组合索引和收尾扩展加入后，本地全仓测试为 186 passed、1 个既有 warning。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。
