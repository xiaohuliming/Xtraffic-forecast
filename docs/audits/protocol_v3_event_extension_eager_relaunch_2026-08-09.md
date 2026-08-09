# protocol v3 event 扩展预载加速与重启审计

更新时间：2026-08-09 12:10，Asia/Singapore

## 目标

最终目标不变：在 `protocol_v3_noleak_dedup` 下完成 33 项 baseline 与 BiST、IGSTGNN、NDR-Net 三地区 9 项扩展，生成可用于论文的 42 行无数据泄露对比表。

本次只优化 9 项扩展的数据读取和排产。33 项 baseline 的训练契约、结果根和控制进程不做修改。

## 根因证据

旧扩展计划只用 CUDA 优化步标定任务耗时，没有计入逐样本 HDF5 读取。runner 任务状态时间戳给出的已完成任务实测耗时如下：

| 任务 | 实测耗时 |
|---|---:|
| IGSTGNN Contra Costa | 5.96 h |
| BiST Alameda | 10.51 h |
| IGSTGNN Alameda | 12.26 h |

按旧分片实测速度外推，四张卡的总时长约为 31.2、43.7、23.4 和 44.7 小时。最慢分片比最快分片长近一倍，旧 LPT 标定不再成立。

Orange IGSTGNN 的 20 个真实训练 batch 数据构建对照如下：

| 模式 | 初始化 | 20 batch 构建 | 每 batch |
|---|---:|---:|---:|
| lazy HDF5 | 3.862 s | 28.5172 s | 1.42586 s |
| eager memory | 11.244 s | 0.270774 s | 0.0135387 s |

两条路径的 batch stream SHA-256 均为 `af7f99015966b930571b3ddd9f111e60f694a537112cdcfe56dd4f76095fddb1`。数据构建提速约 105 倍，支持逐样本 HDF5 I/O 是主要瓶颈的判断。

## 实现与门禁

提交 `c78b87a` 已推送到 `main`，包含以下变更：

1. canonical manifest 显式固定 `traffic_loading: eager`。
2. trainer 命令、invocation config、summary 和 config hash 记录同一加载模式。
3. 新增三区域 eager 与 lazy 等价性审计。训练集先使用固定 seed shuffle，验证集和测试集保持顺序读取，逐 batch 做字节级 SHA-256 对照。
4. CUDA 探针分别记录 batch 构建时间和完整优化步时间，正式 task calibration 使用两者之和。
5. LPT 依据新 task calibration 重新装箱。
6. loading audit 文件及其 SHA-256 写入正式 plan 和 NAS 回传计划，source-host postflight 会再次复验该文件。

本地专项测试先按 TDD 形成失败门，再完成实现。最终全仓结果为 606 passed 和 2 个既有 warning。Python 编译、shell 语法与 `git diff --check` 均通过。

## 旧结果处理

旧扩展运行最后一次有证据的快照为 3/9 completed、0 failed。三项已完成结果和其他中间 attempt 不删除。切换时应生成逐文件 SHA-256 incident 清单，并标记为 `superseded_by_eager_loader_relaunch`。旧结果不得与新代码的 6 项混合，因为 extension composite 要求所有 9 项使用唯一 code SHA。

## 当前远端状态

2026-08-09 12:10 新加坡时间，本机对两个 AutoDL 入口的 TCP 连接均成功，但 SSH 在鉴权前由远端关闭：

| 入口 | 用途 | 观测 |
|---|---|---|
| `connect.westc.seetacloud.com:50989` | 9 项扩展四卡机 | TCP 成功，SSH 立即关闭 |
| `connect.westd.seetacloud.com:36571` | 33 项 baseline 四卡机 | TCP 成功，SSH 立即关闭 |

从 Hostinger 结果服务器解析到真实入口后再次探测，跨境 TCP 超时。本机直连真实 IP 时仍表现为 TCP 成功后 SSH 关闭。现有证据只能说明两个容器当前不可通过 SSH 访问，不能断言具体是余额、手动停机、无卡启动还是平台回收。

本轮没有在远端执行停止、删除、覆盖、上传或关机操作。最后可确认的 baseline 进度为 6/33，0 failed，扩展进度为 3/9，0 failed。它们不是当前实时状态。

## 恢复后的执行顺序

1. 重新读取两台主机的 hostname、queue、task state、runner、trainer、finalizer 和 GPU 状态。
2. 如果扩展旧数据盘恢复，先为 3 项正式结果和全部 attempt 生成只读 incident 清单。
3. 使用独立目录 `/root/autodl-tmp/v3_event_extensions_eager_v2` 部署提交 `c78b87a`，不覆盖旧 checkout 和结果根。
4. 运行三区域 loading equivalence gate。
5. 停止旧扩展 controller 和 finalizer，不动 baseline。
6. 依次运行 9 项 loader-aware CUDA 探针，生成新 calibration 和 plan-only。
7. 人工复核 4 个 shard 的 task union、microbatch、预计墙钟与 plan digest。
8. 启动新的 9 项正式队列，重新武装 Hostinger 严格回传和成功后自动关机。

在第 7 步完成前，不启动新正式训练。
