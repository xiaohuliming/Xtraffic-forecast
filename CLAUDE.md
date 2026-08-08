# XTraffic 项目导航与协作约定

最近更新：2026-08-08 18:55，Asia/Singapore

## 新会话先读

1. `README.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`
6. `docs/README.md`

`XTraffic项目完整汇报.md` 是研究历史、实验数字和出处的唯一权威长报告。数据和结果能否用于论文，以两份泄露审计和 `result_validity.csv` 为准。本文只保留当前快照和操作入口，不再追加已被长报告承接的逐日流水。

## 不可混用的协议边界

- 旧 XTraffic 缓存与结果为 `contaminated`，只读保留。
- v2 已冻结。最先完成的 4 项只能标记为 `contaminated preliminary`。
- 当前无泄露协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。
- IGSTGNN Table 4 复现使用独立协议 `igstgnn_table4_official_v1`，有效性标签为 `official_protocol_with_known_overlap`。
- Table 4 官方协议结果不得与 v3 无泄露结果混合。
- NDR-Net 扩展、BiST-NDR 配对和 native-100 都是项目扩展，不是 IGSTGNN 原论文方法行。
- 所有数字必须来自落盘文件。未通过 source-host postflight、目标端逐文件复验和 composite 的结果，不能写成完整矩阵结论。

## 当前有效性快照

| 对象 | 当前状态 | 主要身份或产物 |
|---|---|---|
| v3 正式缓存 | `verified` | bundle digest `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5` |
| v3 最小判别矩阵 | `verified_host_stratified_45` | composite index `8cf92dc2e931e3e9c3ca18cafb81a794fe68fbd0556dec3401b8650f827899fe` |
| v3 统计分析 | `descriptive_analysis_complete` | `outputs/protocol_v3_noleak_dedup/analysis/` |
| IGSTGNN Table 4 公开材料复现 | `verified_official_protocol_with_known_overlap_39` | index `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2` |
| NDR-Net Table 4 扩展 | `verified_same_protocol_project_method_extension_9` | index `8f4666c45b7708c47d1ee90f3a81444eaf46fee5ba1011bc0242f96c20e0983a` |
| BiST 与 BiST-NDR 30 epoch 配对 | `runner_completed_unanalyzed_18_of_18` | 尚未生成配对性能结论 |
| BiST 与 NDR-Net native-100 | `verified_native100_dual_host_18` | index `27746081f2450d5f8ae7fe950000a23d147bd117ac8b8a5fa3ac18fdfc12314c` |
| v3 full-window baseline | `stopped_supplementary_1_of_33` | 只保留 Alameda HL 补充结果 |
| v3 event baseline | `running_0_of_33_on_5090x4` | 全 33 项在单台 4 卡 5090 重跑，matrix `protocol_v3_event_full33_5090x4_v1` |
| v3 event 全方法扩展 | `implementation_verified_waiting_for_gpu` | BiST、IGSTGNN 与 NDR-Net 三地区共 9 项，尚未启动正式训练 |

本地全仓回归为 589 passed 和 2 个既有 warning。

## v3 正式缓存与训练身份

- 缓存构建代码：`29159f626494de52c69b36fe5bba9854f13cb56c`
- 最小判别矩阵训练代码：`3a4952837aaf6fbda1403d50b610b1bf40c6098a`
- cache bundle ID：`bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`
- cache bundle digest：`854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`
- 缓存目录：`outputs/dist_net/region_data_v3_noleak_dedup`
- 结果根：`outputs/protocol_v3_noleak_dedup`

| 地区 | 总样本 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

v3 缓存已通过本地两次、5080 一次和 5090 一次严格复验。缓存身份与某次训练代码身份必须分开记录。

## 正在运行的 v3 event baseline

主矩阵为 11 种方法乘 3 地区，共 33 项。当前固定 seed 42、上限 30 epoch、patience 6、batch 32、target-disjoint 70/15/15 切分，使用 v3 去重事件样本。

- canonical manifest：`config/protocol_v3/table4_event_baselines_phase1_v1.json`
- manifest SHA-256：`f9cdc1afa5f9f126a6a8094d4d4fc38bc565cf3de277dc4a7742f85a0d2f92ef`
- 当前正式单主机矩阵：0/33 completed，0 failed
- 5080 的 10 项和旧 Orange 的 4 项只读保留，不进入当前矩阵
- 尚未形成完整 baseline 表，BiST、IGSTGNN 和 NDR-Net 的 v3 event 适配已通过本地验证，尚未运行正式 9 项

### 5080 运行快照

2026-08-06 15:28 新加坡时间，恢复队列为 1/13 completed，正在 `v3e011_event_d2stgnn_alameda_seed42`。task011 已到 epoch 6、batch 500/1001。GPU 持续满载，最近显存约 15.9 GB，温度约 45 摄氏度。该任务每个 epoch 约 15,943 秒，是当前瓶颈。

5080 的 22 项声明结果分布在三个受控结果根。前两个 queue 因已修复缺陷按设计遇错停止，不是当前再次失败。

| 任务 | 结果根 | 代码 SHA |
|---|---|---|
| tasks001-005 | `D:\t4\v3_event_phase1_5080_v2\results` | `c16dc7a8208f834984e76a40a7b6f47f523c210b` |
| tasks006-009 | `D:\t4\v3_event_gwnfix_v1\results` | `e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae` |
| tasks010-022 | `D:\t4\v3_event_dgfix_v1\results` | `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2` |

task010 Alameda DGCRN 的 H3、H6、H12 和 Average MAE 为 `11.284030`、`12.563134`、`14.453520` 和 `12.577300`。summary SHA-256 为 `49dcb9734a0f74e7aed6613045d087c2ab27dd7d1752bf05facff5484f113aec`。

### Orange 主机与灾备

8 卡 RTX 4090 D 主机的端口 `11218` 当前 TCP 可达后由远端立即关闭 SSH，无法区分余额不足、实例停止或平台回收。不得写成已确认的具体原因。未验收 staging 保留在原数据盘，恢复 GPU 后需逐项检查 checkpoint 恢复契约。

Orange 4 项灾备已在群晖 NAS 通过 25/25 文件复验和原子改名。正式目录为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/orange_4090d8_partial_4_v1`，共 26 个 regular files 和 `1652693542` bytes。`files.sha256` 的 SHA-256 为 `6297b0403746faea2b9c8da0c174952db670cfbd2bdb8f20bea28289118c118a`。该树只是 `verified_partial_disaster_backup_4`，不是正式发布。

## 5080 训练后自动 NAS 回传

持久计划任务 `XTraffic_ProtocolV3_Event_NAS_Finalize_v1` 当前为 Running，每 60 秒检查一次。它只在以下条件全部满足后上传：

1. tasks010-022 恢复队列 13/13 completed。
2. runner 退出码为 0。
3. 没有 runner、baseline trainer 或 GraphWaveNet trainer 残留。
4. tasks001-022 的 `config_hash`、summary、prediction artifact、必需产物和 completed state 逐项复验通过。
5. NAS 容量、逐文件 SHA-256、精确 allowlist 和目录树全部通过。

任何失败都 fail closed，保留源结果并保持 5080 开机。最终 NAS 目标为 `/var/services/homes/xhlm/Drive/XTraffic/results/protocol_v3_noleak_dedup/event_baseline_phase1/hosts/event_rtx5080_alameda_contra`。当前正式目录和 incoming 都不存在，这与等待训练完成的状态一致。

- 固定计划：`config/protocol_v3/event_baseline_5080_nas_backup_v1.json`
- 计划 SHA-256：`738bc46122f9af6466c1b605013b57c2cc05874aadcda6682cdf08fb4ed964c8`
- postflight SHA-256：`b8605c1725fe9de5ffc74d19815f460b19be812dd495f9814fe41a92eb5b1bf0`
- finalizer SHA-256：`7295346daf7eb972ead64db27706f4234157ae783e68c69421097e6787349c70`
- 5080 专用 NAS 公钥指纹：`SHA256:Nx5BbjtFe9eRbSJh4hV/YN+reR2pw4hZuFLa9PLKMGc`

私钥只存在 5080 的 `D:\t4\v3_event_nas_control\id_ed25519`，不进入 Git。postflight 核心是跨平台 Python，未来 5090 可复用同一契约，但必须使用独立 host ID、分片计划、NAS 目录、SSH 密钥和 Linux 持久包装。

## Table 4 官方协议复现与扩展

- 39 项公开材料最大忠实复现已完成。清单 SHA-256 为 `d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78`，index SHA-256 为 `3e5f6bee6eb0cd667f28c287f9538136b31c4312652673453e3f56e446edbbe2`。
- 复现与论文报告值的 Average MAE 绝对差均值为 `2.450150`。该差异必须保留，不得写成作者数值的完全复现。
- NDR-Net 扩展 9/9 已复验。三地区 Average MAE 的三种子均值与样本标准差为 Alameda `10.759524 ± 0.040862`、Contra Costa `11.413717 ± 0.024537`、Orange `11.569331 ± 0.031664`。
- native-100 双机分层矩阵 18/18 已完成两次只读 composite。最终 48 行表位于 `outputs/reproduction/igstgnn_table4_official_v1/table4_with_native100_verified_v1/`，CSV SHA-256 为 `4034b49080c223f1cc28a730b1790701cdcbf0e827e4eb982cb36baddbf024d6`。
- BiST 与 BiST-NDR 的统一 30 epoch 配对队列已 18/18 runner completed，但尚未完成配对统计和性能结论。

完整数字、每份 artifact 哈希、主机分片和 incident 请查看 `XTraffic项目完整汇报.md` 与 `docs/audits/igstgnn_table4_*.md`。

## v3 event 全 33 项在 4 卡 5090 上运行中

2026-08-08 08:56 UTC 启动。完整记录见 `docs/audits/protocol_v3_event_cc_orange_multi_gpu_2026-08-07.md`。

代码 SHA `d23a991a570fd1c59e3d2ac8315a5e4063e8097f`，plan_digest `23ed8507a1a301daef27bb613bc88d2070ec7052858de7d219e9c845c4d2cb82`。三个地区全部同机，最终表不再是主机分层矩阵。5080 上已验收的 10 项 Alameda 与 4090 D 上的 4 项 Orange 保持只读，不进入本矩阵。原 18.7 小时是基于探针标定的排产估计，当前正式四路并发的首轮 epoch 慢于标定，完成时间须按后续实测更新。

实测显示只有 Orange 的 D2STGNN 与 STTN 在 32 GB 上放不下 batch 32，两者使用 microbatch 16 并独占 GPU3。全程 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，该设置已写入 plan 并参与 plan_digest。

回传路径已解决。Hostinger `31.97.147.71` 当前约有 49.6 GB 可用空间，已启用 4 GB swap，xray 保持 active。5090 上的 finalizer PID 为 `17206`，正在等待 33/33。成功后会执行 source-host postflight、隐藏 incoming 上传、逐文件 SHA-256、原子提升、远端只读 composite 和论文列结构表格生成，然后自动关闭 5090。任一环节失败都保持主机开机。

远端 composite 在 systemd 中运行，`MemoryMax` 为 3 GB，`MemorySwapMax` 为 4 GB。控制提交为 `314d45714b61465bce57e8a33c824ce6c47891fd`。composite、表格生成器和 post-publish 包装的 SHA-256 分别为 `2da92291801e516f18ac335fd9a330e75561ce77b16e7897dbc47b99789e0d13`、`890006ec3f1132eeb1aa87cd42d4d668a0160b16c68de069d7304a5230a09993` 和 `28585a7d269218083e2ef83e440d5acf7cc7f0c03498c9d780ce3dd3400a4183`。

18:55 新加坡时间复验仍为 0/33 completed、0 failed。Orange DGCRN、DCRNN、DSTAGNN 与 D2STGNN 分别完成 epoch 3、4、6 与 3。finalizer PID `17206` 持续正常等待。

## v3 无泄漏全方法 42 行表已完成代码准备

最终表的固定范围为 11 个已在运行的 baseline，加 BiST、IGSTGNN 和 NDR-Net，乘三地区，共 42 行。全部使用 `protocol_v3_noleak_dedup`、cache schema 3、seed 42、30 epoch 上限、target-disjoint event split、显式观测 mask 和全部有效观测点累计的 validation masked MAE。各模型保留公开配方的 batch、optimizer 和 scheduler，表中会明确记录这些差异。

新增 canonical manifest 为 `config/protocol_v3/table4_event_extensions_v1.json`，精确包含 9 项。数据适配器不物化 object-pickle split，不把 `affected_mask` 放入任何训练或验证 batch。NDR-Net 的 normalization 和周内时段中位数只在唯一 raw train timestamps 上拟合，并使用 v3 显式 `flow_mask`。IGSTGNN 静态支持文件仅在 `sensors.csv order == region_idx` 且 `station_id == node_order[region_idx]` 逐项完全一致后加载。

新增部署、真实 CUDA 探针、LPT 多卡控制、source-host postflight、严格回传、发布后 composite 与 42 行合表器。两个发布树必须分别验证，且协议、cache bundle ID 和 digest 全部相同，才允许输出 CSV、Markdown、LaTeX 和可追溯 manifest。正式 9 项尚未训练，不得填入任何预测数字。

实现提交 `16e90baab74bb767a4f8d7a46f45eef6d87af44d` 已推送到 `main`。Hostinger 的 `/opt/xtraffic/v3_event_repo` 已原子切换到该提交，旧 `314d457` checkout 保留为回滚。`/opt/xtraffic/bin/event-extension-post-publish` 已安装，SHA-256 为 `80f05af5b0bb87a221e703479fc034f3b717acf07fff5748b583a1a518388edc`。切换后 xray 仍为 active，swap 仍为 4 GB，根分区约有 44.1 GB 可用。临时 bundle 已清理。

## 历史：v3 event 剩余分片的多卡主机计划

2026-08-07 布防记录。

八卡 4090 D 实例已在无卡模式恢复，数据盘完好，Orange 4 项正式结果与 6 项 staging 都在。但代码复核确认 `train_baseline_zoo.py` 与 event runner 都没有 checkpoint 续跑路径，那 6 项 staging 必须从 epoch 0 重跑。

计划把 Contra Costa 全部 11 项与 Orange 全部 11 项共 22 项放到同一台新多卡主机。Orange 重跑 11 项而不是 7 项，因为原 4 项从未做过 source-host postflight，混用会让 Orange 比较组跨主机。Alameda 留在 5080。

关键实测：Orange DGCRN 在 24 GB 卡上每 epoch 20,587 秒，单项占剩余工作量的 38%，是唯一的关键路径。按实测标定 3 卡与 4 卡墙钟相同；按 32 GB 投影标定 4 卡几乎零空转。卡数必须等探针实测后再定，投影标定标记为 `projection_not_measured`，不得当证据引用。

入口顺序为 `protocol_v3_event_deploy_host.sh` 到探针到 `protocol_v3_event_multi_gpu.py --plan-only` 到人工复核 plan 到 `protocol_v3_event_launch_host.sh`。回传复用已审计的 `protocol_v3_event_host_postflight.py` 契约，Linux finalizer 为 `protocol_v3_event_nas_finalize.sh`。

## 下一步顺序

1. 继续监控单主机 33 项矩阵，不在任务中途改动分片或训练契约。
2. 等 finalizer 完成 source-host postflight、Hostinger 发布、只读 composite 和 33 行表格生成。
3. 租用新多卡 5090 主机，先运行 9 项扩展的真实 CUDA 探针和 plan-only，人工复核 plan digest 后再启动正式训练。
4. 最终表中明确标注 v3 无泄漏协议、seed 42 与单次运行口径，不与官方重叠协议的 Table 4 数字混用。

## 实验纪律

- 保留诚实负结果，不为追求正结果反复改停止线。
- 种子约定为 42、1、2。单种子 baseline 与三种子均值只能做描述性比较。
- 指标使用 masked MAE，验证指标必须按全部有效观测点累计。
- `affected` 只是事后分组指标，除非实验明确声明标签监督。
- 新结果必须记录协议、split、normalizer 与 climatology 拟合范围、缓存哈希、Git SHA、dirty 状态和 schema version。
- 不得把 Windows 5080 与 Linux 主机的绝对路径、`config_hash` 或运行时 provenance 混成同机结果。
- 报告类正文不用破折号，不用括号内插说。代码注释保持最少，commit 信息使用英文。

## 项目方法和仓库约定

- 项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。
- `scripts/README.md` 是脚本分类入口。
- `outputs/README.md` 说明历史结果和本地产物的关系。
- `docs/archive/` 保存被主报告吸收的旧文档。
- 不批量移动仍被历史入口引用的脚本或结果。
- 本地原始数据位于根目录 `archive/`，不进入 Git。
- 远程运行包和日志放入被忽略的 `outputs/remote_runs/`。

## 凭据与远程运行

实验室共享主机、地址和密码集中明文维护在 `config/lab_hosts.env`。脚本可以直接加载该文件，也允许环境变量覆盖。私钥、个人令牌和个人账号凭据仍不进入仓库。

## 历史追溯

以下内容不再在本文重复，均已有完整落盘记录：

- v2 冻结、去重与 active event 隔离问题：`docs/audits/protocol_v3_review_2026-07-19.md`
- 原始数据泄露根因：`docs/audits/data_leakage_2026-07-18.md`
- 45 项 v3 分片、task040、task041 与 composite incident：`XTraffic项目完整汇报.md`
- IGSTGNN Table 4 九卡恢复、task034 和三源 composite：`docs/audits/igstgnn_table4_4090d_launch_2026-07-29.md` 与主报告
- native-100 双机重分片与替代 5090：`docs/audits/igstgnn_table4_native100_dual_host_rebalance_2026-08-01.md`
- v3 event Orange 八卡运行与恢复：`docs/audits/protocol_v3_event_4090d8_launch_2026-08-04.md`
