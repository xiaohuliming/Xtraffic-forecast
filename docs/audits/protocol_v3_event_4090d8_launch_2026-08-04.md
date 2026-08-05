# protocol v3 event baseline 八卡启动记录

## 范围

2026-08-04，用户提供了新的 SeetaCloud 主机。硬件复验确认该实例不是单张 RTX 5090，而是同一容器中的 8 张 RTX 4090 D，每张可用显存约 24 GB。容器主机名为 `autodl-container-5ee0448de2-ca306372`，正式数据盘为 `/root/autodl-tmp`。

本轮仍使用 `protocol_v3_noleak_dedup`、cache schema 3、target-disjoint 切分和 event 样本口径。Orange 的 11 个 baseline 全部保留在同一台主机，只在主机内按 GPU 分片。因此 Orange 的直接方法比较不跨主机。RTX 5080 继续执行 Alameda 与 Contra Costa。最终结果必须使用明确的主机分层 composite，不得称为同质矩阵。

## 数据与环境门

三地区正式缓存和图文件已传入新主机。runner 按正式 `cache_bundle_manifest.json` 逐文件复验通过。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。

基础镜像原先只有 PyTorch 2.12.1 和 CUDA 13.0。安装 h5py、pandas、scipy 和 pytest 后，远端首轮专项门为 11 passed。八卡控制器实现提交为 `2303557dc7fc3f5e87760bae0b31d407b544e5f5`，对应 Git bundle SHA-256 为 `4916a2b918cd3f00e8b4135ee93125da8bc23ab2f5df5423d5418cfbaec4c418`。

## 八卡分片

| GPU | 任务 |
|---:|---|
| 0 | DCRNN |
| 1 | DGCRN |
| 2 | D2STGNN |
| 3 | DSTAGNN |
| 4 | STTN |
| 5 | ASTGCN，STGCN |
| 6 | GraphWaveNet，AGCRN |
| 7 | LSTM，HL |

持久控制器 PID 为 `2429`，矩阵 ID 为 `protocol_v3_event_orange_4090d8_v1`。正式根为 `/root/autodl-tmp/v3_event_orange_4090d8`。每张 GPU 都有独立的 shard manifest、results root、queue status 和日志。启动后 8 个子 runner 和 8 个 trainer 均实际出现。

## 24 GB 显存 incident 与恢复

首轮真实训练暴露三个容量事件。DGCRN 在额外申请 120 MiB 时已使用 23.44 GiB。D2STGNN 需要再申请 2.80 GiB。STTN 反向需要再申请 5.61 GiB。三项均没有正式任务目录，traceback 保留在各自失败 attempt 中。

提交 `845804de1586eb6620496298d264c789c6a140c9` 增加等效 batch 32 的显存恢复。训练按全 batch 的有效观测点数归一化梯度，对多个 microbatch 累积反向，然后只执行一次优化器更新。验证和测试使用同一观测集合分块推理。全仓回归为 517 passed 和 2 个既有 warning，远端恢复专项门为 14 passed。

DGCRN 使用 microbatch 8，STTN 使用 microbatch 16，两者已通过首个完整有效 batch。D2STGNN 在 microbatch 16 时仍占用 23.50 GiB 并安全失败。提交 `5bced13246ee3eb1e903726f0e9c60c839f46209` 将 D2STGNN 降为 microbatch 8。独立重试 PID 为 `7214`，首个完整有效 batch 的 loss 为 `141.838`，当时显存占用约 19.5 GiB。

## 当前状态

2026-08-04 21:19 新加坡时间，Orange 的 DCRNN、DSTAGNN、ASTGCN、GraphWaveNet 和 LSTM 使用首轮 checkout 训练。DGCRN 和 STTN 在独立恢复 checkout 训练。D2STGNN 在第二个独立恢复 checkout 训练。后续轻量任务会由同一 GPU 的子队列串行接续。

RTX 5080 的 event 队列已验收 2/33，当前执行 Alameda DCRNN。Alameda LSTM 的 Average MAE 为 `12.950932743865955`，H3、H6 和 H12 MAE 为 `11.472908787131686`、`12.867387335945871` 和 `15.258082126124597`，summary SHA-256 为 `cfe37cfb481623214afa4516125100f8e6d05d518fe71cf84a895203b50f9452`。

当前只确认启动、首个完整 batch 和已有的两份 Alameda 正式结果。Orange 尚无本轮新的正式性能结论。所有结果只有在逐任务五文件验收、源主机 postflight、结果服务器逐文件复验和最终 composite 全部通过后才可入表。

## 初始化 incident 与修正恢复

后续复验确认 DSTAGNN 首轮从第一个 batch 起 loss 即为非有限值。runner 在测试预测门安全失败，没有正式任务目录。对照上游 LargeST engine 后确认，v3 baseline factory 遗漏了 ASTGCN 与 DSTAGNN 的显式参数初始化。ASTGCN 首轮数值虽然有限，但同样不满足绑定上游实现契约，因此在没有正式结果时停止。

提交 `d8760d1a48c30d7d87130aef1712c85b8aa76359` 恢复上游参数初始化。所有大于一维的参数使用 Xavier uniform，一维参数使用 uniform。AGCRN 改为复用同一入口，其数值语义不变。Mac 全仓回归为 519 passed 和 2 个既有 warning，远端专项门为 12 passed。部署 bundle SHA-256 为 `7b4433425ffb9c05641ade2a09aabc1847d3db588651bdd48e0d9d4f16d77b76`。

独立恢复矩阵 ID 为 `protocol_v3_event_orange_upstream_init_recovery_v1`，代码 checkout 为 `d8760d1a48c30d7d87130aef1712c85b8aa76359`。DSTAGNN 使用 GPU 3，ASTGCN 与后续 STGCN 使用 GPU 5。首个完整 batch 的 loss 分别为 `198.078` 与 `200.413`，均为有限值。

## 最新状态

5080 event 队列已验收 3/33，当前执行 Alameda AGCRN。Alameda DCRNN Average MAE 为 `14.215668450826508`，H3、H6 和 H12 MAE 为 `12.167831568642377`、`14.011284383846979` 和 `17.427471336632582`，summary SHA-256 为 `806a6ab4f91ffca55609b908c82d6e5b028d5b26421a06e7f7336dfac09b2801`。

Orange 的 DCRNN、GraphWaveNet、LSTM、DGCRN、D2STGNN、STTN、DSTAGNN 与 ASTGCN 正在训练或恢复。Orange 仍为 0/11 正式验收。首轮 ASTGCN 与 DSTAGNN attempt 均不得进入最终 composite。

## GraphWaveNet 测试标签 incident 与恢复

5080 上的 Alameda GraphWaveNet 完成 30 epoch 后，在测试推理访问 `batch["affected_mask"]` 时触发 `KeyError`。训练 best validation masked MAE 为 `12.934727422724507`。runner 返回码为 1，没有正式 task006。checkpoint 与完整 train log 保留为 incident 证据。

根因是 GraphWaveNet 使用独立训练入口，未复用 baseline zoo 已修正的 split 级 RegionData 参数。提交 `edf4c1e` 增加同等契约。event 训练集与验证集不加载分析标签，只有 event 测试集加载 `affected_mask`。full-window 行为保持不变。提交 `e1bea79` 增加 GPU 6 的 Orange GraphWaveNet 与 AGCRN 恢复模式。本地全仓为 521 passed 和 2 个既有 warning。Windows 专项门为 12 passed，Linux 专项门为 16 passed。

5080 原结果根已正式验收 5 项，当前保持只读。新 continuation 结果根为 `D:\t4\v3_event_gwnfix_v1\results`，分片为 tasks006-022，共 17 项，checkout 为 `e1bea79d41be4c923fa874d20f0e9fcbf4ed5dae`。首次启动因 manifest 的 `default_results_root` 是绝对路径而被配置门拒绝，没有 trainer。失败日志与退出码已移入 incident 目录。字段改为相对路径后，计划任务 `XTraffic_ProtocolV3_Event_Baseline_Continuation_v1` 正常运行，当前执行 Alameda GraphWaveNet。

Orange 原 GPU 6 GraphWaveNet 在 epoch 25 停止，没有正式结果。新 controller PID 为 `12996`，子 runner PID 为 `13065`，matrix ID 为 `protocol_v3_event_orange_4090d8_gwn_recovery_v1`。恢复根为 `/root/autodl-tmp/v3_event_orange_4090d8/gwn_recovery_results`。它先从头执行 task028 GraphWaveNet，之后执行 task026 AGCRN。bundle SHA-256 为 `e810c8c56b9715e2ff4e8b286c48d4ae0c0f92ba049698e66850bcbe36761359`。

## 最新正式进度

全矩阵为 7/33。5080 为 Alameda 5 项，Orange 为 2 项。Orange HL 的 H3、H6、H12 与 Average MAE 为 `15.407020772641456`、`18.471071562965424`、`24.89280260874014` 与 `19.025463577015767`。Orange LSTM 对应为 `12.191570568539724`、`13.618700471002564`、`16.221245969669557` 与 `13.72444114753531`。其余任务继续训练或恢复，不得提前形成完整性能结论。

## 2026-08-05 主机不可达状态

12:06 新加坡时间，端口 `11218` 连续三次 TCP 建连成功，但 SSH 均由远端立即关闭。该行为不是普通网络超时。结果服务器未发现本轮 v3 event 结果目录。现有证据不足以区分余额不足、实例手动停止与平台回收，因此审计状态为 `suspected_provider_stop_or_balance_exhaustion`，不得写成已确认欠费。

最后在线证据确认 Orange HL 与 LSTM 已通过 runner 验收，另外多个任务处于训练或恢复中。两项正式结果尚未回传，当前滞留在不可达主机。重新开启同一实例并挂载原数据盘后，第一步必须只读复验并回传这两项，再审计其他 attempt。若平台返回的是全新空盘容器，未回传结果不得凭日志或旧数值重建。

## 5080 DGCRN partial batch incident

5080 继续在线，并在 GraphWaveNet 修复根中完成 tasks006-009。task010 Alameda DGCRN 在 epoch 1 的最后一个 batch size 1 上触发 `IndexError`。根因是 vendored DGCRN 对 `x[..., i]` 使用无维度 `torch.squeeze`，单样本时误删 batch 维度。失败 attempt 没有正式任务目录。

提交 `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2` 去除该 squeeze，并增加单样本前向回归。Mac 全仓为 522 passed 和 2 个既有 warning。5080 专项门为 13 passed。部署 bundle SHA-256 为 `a2c6e990eb9883884e064ed2d5b5c1c681393e96ca7af192749d76dccbb3d902`。

独立恢复分片固定为 tasks010-022，共 13 项。checkout 为 `9dd7cac140d6af3ea0d504962f3ce4c99b4f8be2`，结果根为 `D:\t4\v3_event_dgfix_v1\results`，计划任务为 `XTraffic_ProtocolV3_Event_Baseline_DGCRNFix_v1`。首个 Alameda DGCRN batch loss 为 `137.815`，GPU 显存约 10.6 GiB。runner 状态为 running。

## 无卡模式恢复与 partial export

2026-08-05 12:37 新加坡时间，端口 `11218` 恢复 SSH。容器主机名仍为 `autodl-container-5ee0448de2-ca306372`，证明这是原实例而不是空数据盘替代容器。`/root/autodl-tmp` 保留原有结果、staging、日志和控制文件，剩余空间约 47 GB。实例当前没有 GPU，`nvidia-smi` 不可用，没有 trainer、runner 或 finalizer 进程，因此未尝试继续训练。

只读复验确认停机前实际完成 4 个 Orange 正式任务：

| 任务 | 方法 | H3 MAE | H6 MAE | H12 MAE | Average MAE |
|---|---|---:|---:|---:|---:|
| v3e023 | HL | `15.407021` | `18.471072` | `24.892803` | `19.025464` |
| v3e024 | LSTM | `12.191571` | `13.618700` | `16.221246` | `13.724441` |
| v3e028 | GraphWaveNet | 待从 prediction artifact 汇总 | 待汇总 | 待汇总 | `12.950535` |
| v3e030 | STTN | `11.697110` | `12.646399` | `13.804152` | `12.585341` |

源主机上的 4 个正式任务目录保持原位。另以硬链接建立 `/root/autodl-tmp/v3_event_orange_4090d8/partial_export_4_v1`，纳入 20 个任务 artifact、state、queue snapshot、计划和 export 元数据，共 25 个文件。源 artifact 总字节数为 `1652698342`，`files.sha256` 的 SHA-256 为 `6297b0403746faea2b9c8da0c174952db670cfbd2bdb8f20bea28289118c118a`。

partial export 已复制到 Mac 的 `outputs/protocol_v3_noleak_dedup/remote_recovery/partial_export_4_v1`，本地执行清单复验为 25/25 通过。该副本是停机风险下的内容级保护，不构成结果服务器正式发布，也不替代最终 composite。

结果服务器 `/root/xtraffic_results` 所在 40 GB 分区为 100% 已用，可用空间为 0。没有可用的替代结果盘。本轮没有删除历史实验、Docker 或系统文件，正式回传暂缓。必须先扩容或经明确审核释放足够空间，再执行远端逐文件复验与原子提升。

未验收 staging 的最后落盘 epoch 为 D2STGNN 4、AGCRN 4、DSTAGNN 11、ASTGCN 12、DGCRN 2 与 DCRNN 19。这些只说明中间状态存在，不能视为正式结果。恢复 GPU 模式后需先检查 checkpoint、optimizer、scheduler、验证状态与代码 provenance 是否满足恢复契约，不满足时必须从头重跑。

5080 继续运行 task010 Alameda DGCRN。12:37 时 queue 为 running，tasks010-022 中 0/13 新任务完成。训练已完成 epoch 2，epoch 3 到 batch 600/1001。综合两台源主机与 Mac 备份，全矩阵已有 13/33 项通过 runner 验收并具备可访问的哈希保护 artifact。

## 结果服务器容量恢复

用户于 2026-08-05 授权清理 40 GB 结果服务器分区。清理前分区为 100% 使用且可用空间为 0。只读盘点确认 `/root/xtraffic_results` 为 21 GB，正式实验数据不在删除范围。

`/root/xtraffic_stage` 共 12 个文件和约 1.7 GB，时间均为 2026-07-07。六个 HDF5、三个 config 和三个图文件的 SHA-256 分别与 Mac 的旧 `outputs/dist_net/region_data` 和 `outputs/region_graphs` 完全一致。该缓存属于审计前旧污染数据，在服务器上没有其他引用或打开文件。删除前记录了文件哈希，随后只删除该冗余 staging 副本。

同时将 systemd journal 从 800.3 MB 清理到 75.7 MB，删除可再生成的 APT lists 与 pkgcache。一个从 2026-08-04 起卡住的 `apt-get update` 被正常停止。Docker 活跃容器、Docker 镜像、`/www`、`/opt` 和正式结果均未删除。

ext4 `/dev/vda3` 原有 468496 个保留 block，约占 4.5%。将保留比例调为 1% 后仍保留 104340 个 4096 字节 block，约 407.6 MiB。13:09 最终复验为 40 GB 总量、36 GB 已用、`3267715072` bytes 可用、93% 使用率。`/root/xtraffic_results` 仍为 21 GB，所有一级正式目录存在，活跃 Docker 容器仍在线。

当前空间足以接收 1.65 GB 的四项 partial export，但上传后余量不足，且完整 Orange 11 项预计无法安全容纳。为避免再次将根分区推至临界状态，本轮不上传 partial export。Mac 的 25 文件哈希验证副本继续作为灾备，结果服务器正式发布等待扩容或新增结果盘。
