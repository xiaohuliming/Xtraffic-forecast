# NDR 改进与跨区域泛化研究

日期：2026-10-13，Asia/Singapore

## 目标与当前证据

用户授权自行设计模型或插件，在 XTraffic 上使用 RTX 5080 训练，目标是提高预测性能和泛化能力，并与现有强方法公平比较。

当前已复验的 v3 event 45 行表中，NDR-Net 的三个地区 Average MAE 分别为 10.875715、11.627960、11.639792，均为当地最低。全窗口九模型表未含 NDR-Net，BiST 的三个地区 Average MAE 分别为 10.257884、10.549751、10.626994。这些表都是单种子和固定 30 epoch 的描述性比较，不能证明充分收敛后的 SOTA。

旧 event CSV SHA-256 为 `44afc9c2c8d425dae9be486d071282c9c1ac8fce67e86f7da970f650d478b68c`。本轮重新核验全窗口 165 个文件，共 13,641,954,611 bytes，零缺失和零哈希不符。旧文件保持不变。

## 冻结的方法假设

原 NDR 通过常态气候态锚点与持续锚点重组两个残差预测头，门控只利用残差绝对幅度和最后两步差值。不同符号的恢复趋势被折叠为相同门控信号。显式历史观测 mask 在原适配器内丢弃，门控无法辨别真实末观测与因果填充值。

首轮拆成四种独立对照：

1. `baseline` 精确复用原 STAEformerDeseason s0g。
2. `projected_oracle` 只改变训练门控 teacher，采用两锚点凸组合的逐点 MAE 投影解。
3. `sanitized_control` 仅按窗口内真实观测做因果重建，缺少此前观测时使用 train-only climatology。
4. `mask_signed` 在第 3 种历史处理上，给原单调门控 logit 添加节点共享的小型 MLP correction。特征包括带符号残差、恢复趋势、有效率、观测年龄、未来锚点差和预测步长。

共享路由器末层零初始化。完整有效历史下，候选初始输出与原 NDR 一致。原 STAEformer 骨干、双残差头和全局标准化保持匹配。历史可靠持续锚点是单独开关，默认关闭，不能与路由器机制混为一个因素。

共享路由器不含 node ID 或 region embedding。原骨干仍含地区节点专属 embedding，因此本阶段只能验证路由器的节点数适用性，不能据此声称整个模型严格零样本跨区域迁移。

## 评测分工与晋级门槛

开发只使用 Alameda 与 ContraCosta 的 train 和 validation。开发主指标为累计全部有效目标点的 raw-flow masked MAE，辅助记录每预测步的 MAE、RMSE、MAPE、参数量、实际时间和峰值分配显存。保留完整尾 batch，不读取 affected 标签。

Orange、seeds 1 和 2 以及最终 test 作为冻结后的独立评测。候选进入扩大验证需在两个开发地区相对同机原 NDR 的完整 validation MAE 都下降，且相对 sanitized control 的差值可辨。不能只选择最好的地区。最终同协议同时比较原 NDR 与真实 BiST，并检查足够长训练后的收敛。

最终性能声明必须依赖三地区 seeds 42、1、2，独立 checkpoint 推理，逐文件回传复验，日块或事件块配对统计，以及未参与开发的评测。已知重叠的 IGSTGNN 官方协议不进入本研究主比较。

泛化后续分为严格零样本和目标历史校准迁移。严格零样本不拟合目标区域训练段统计；校准迁移可以用目标历史前缀拟合气候态和尺度，但模型权重不使用目标监督训练。两者分开报告。

## 主机与历史队列核对

本轮实时连接到 RTX 5080。CUDA 可用，Python 3.13.3、PyTorch 2.10.0.dev20251203+cu128、NumPy 2.2.6、h5py 3.14.0。检查时无 Python 训练任务，约 21 GiB 可用内存，C 盘约 46 GB 空闲，D 盘约 8.5 GB 空闲。新代码和结果使用 C 盘独立路径，复用正式缓存，不删除旧结果。

9 月局部基线队列已是 failed，完成 2/6 后报告 GPU busy or free memory insufficient。不是当前运行队列，不自动重启。

新队列保持开机，不自动租卡、不上传 DMIT、不改旧任务或永久电源设置。每阶段先校准、冻结代码和计划，再启动训练。中断保留 best 和 last，只有保存 optimizer、scheduler、RNG、loader 游标且身份一致的 last 才允许恢复。

## 持续研究状态

假设树、远端探测回执、部署包和后续结果位于主工作区 `outputs/remote_runs/ndr_generalization_20261013/`。研究代码在隔离分支 `codex/ndr-generalization`，工作区为 `/Users/xhlm/.codex/worktrees/ndr-generalization/科研实习`。

目前只证明候选实现与工程契约，不证明性能提升。正式训练、独立完整 validation 验收和最终测试结果将在落盘后追加。

## 第二候选与外部强对照

第二候选 `SharedGraphNDR` 默认有 123,832 个参数，使用共享时序编码、两层正反道路邻域均值传播、八个全局 latent token、双残差预测头和共享历史路由器。它没有随节点数变化的可训练参数。图传播直接使用边列表，latent attention 使用 K 乘 N 与 N 乘 K，不生成完整 N 乘 N attention。

不同节点数量严格加载同一权重、图排列等变、缺失值隔离、有限梯度和输出单位检查的 17 项测试已通过。这个结果是工程证据，尚无精度或跨区域测试结果。动态节点尺度不是参数，目标区域尺度若用目标历史训练段拟合，应称历史校准迁移。

当前公开文献与代码检索确认以下可适配强对照，尚未全部在本仓库部署：

- [FlashST，ICML 2024](https://proceedings.mlr.press/v235/li24bw.html)，[公开代码](https://github.com/HKUDS/FlashST)。
- [DualCast，IJCAI 2025](https://www.ijcai.org/proceedings/2025/366)，[公开代码](https://github.com/suzy0223/DualCast)。其中 DualCast-S 是相关双分支插件对手。
- [STGformer](https://arxiv.org/abs/2410.00385)，[公开代码](https://github.com/Dreamzz5/STGformer)。原文仍含节点专属位置嵌入，适合作预测与效率对照。
- [GraphSparseNet](https://arxiv.org/abs/2502.19823)，[公开代码](https://github.com/PolynomeK/GSNet)。原文仍含节点 embedding。
- [HyperD，AAAI 2026](https://doi.org/10.1609/aaai.v40i18.38599)，[公开代码](https://github.com/ll121202/HyperD)。
- [OpenCity](https://arxiv.org/abs/2408.10269)，[公开代码](https://github.com/HKUDS/OpenCity)。预训练材料包含 Bay Area 与 Los Angeles，必须先审计年份与站点重叠再用于独立泛化结论。

[节点自适应嵌入局限研究](https://arxiv.org/abs/2411.11448)提供了设计动机，其 PCA 目标嵌入使用目标域 validation 子集，不能直接当成 source-only 零样本方法。检索有边界，不代表已覆盖所有 2026 方法。

外部适配顺序为原 NDR、原 STAEformer、真实 BiST，随后增加一个可扩展骨干和 DualCast-S。性能胜出必须以同数据、同 split、真实重训与独立复验为依据，公开论文数值不能与本项目 v3 数值直接拼表。

## 2026-10-13 16:10 首轮正式开发队列已启动

本地研究提交为 `7f26800`，分支为 `codex/ndr-generalization`。部署包为不含凭据与原始数据的 440 文件源码快照。包 SHA-256 为 `95b6445b69032cf3d695c9756aa674fa90aa49b18e2939c882f52baa13f22752`。5080 部署采用独立 clean snapshot commit `8260e6ca62d9f01c48b3f40d4c185ed818541af7`，每个源码文件与 Mac 逐一一致。两个 SHA 表示不同 Git 树的真实身份，不混称同一 checkout。

Mac 相关验收为 59 passed，Ruff、编译和 diff 检查通过。Windows 首批模型、控制器与既有 deseason 测试为 26 passed。四变体在 521 节点上完成真实 CUDA 前后向有限性验收，mask_signed 只比原 NDR 多 193 个参数。

独立 CUDA 校准覆盖 Alameda 与 ContraCosta 的原 NDR 和 mask_signed，effective batch 16、microbatch 8、FP32、三线程、三次预热和二十个计时 batch。Alameda 的每项 30 epoch 外推为 4.75 至 5.02 小时，ContraCosta 为 2.26 至 2.43 小时。峰值分配显存约为 4.19 GiB 与 3.86 GiB，不能当成整卡占用。其余两种变体采用同地区较慢校准项作为预算估计，没有把估计写成该变体实测速度。

原 NDR 和 mask_signed 都在真实 Alameda 缓存完成两次优化更新，并由独立进程从 best checkpoint 完整复算 validation。每项覆盖全部 6954 个样本和 35,937,114 个有效点，MAE 复算差为 0。它们仍标为 `verified_smoke_checkpoint_complete_validation`，不是训练精度结论。

首轮固定八项，Alameda 与 ContraCosta 分别包含原 NDR、mask_signed、sanitized_control、projected_oracle，全部 seed 42、固定 30 epochs、不早停。启动前冻结顺序为两地区原 NDR、两地区 mask_signed、两地区 sanitized control、两地区 projected oracle。数据、源码、batch、优化器和选择口径相同。完整 validation 的全局有效点 MAE 选 best，同时保存十二步指标与步长均值。

计划 digest 为 `ad530e29fdcf2bd0cdabdd739349814e805f12e11cfab6341e9180651807672b`。外推八项约 29.77 小时，乘 1.5 余量再加一小时约 45.65 小时，内部总上限为 48 小时，单项上限为 12 小时。48 小时截止为 2026-10-15 16:10:22，Asia/Singapore。到达内部停止点保留自有 worker 和 checkpoint 的真实状态，不清理他人进程，不自动重启或延长预算。

远端根为 `C:/XTrafficResearch/ndr_20261013/stage1_event_gate_seed42`。后台计划任务为 `XTraffic_NDR_EventGate_Stage1_20261013`，无触发器，pythonw 运行，SSH 断开后保持运行。新任务没有自动回传与关机。最初直接 pythonw SSH 子进程在会话结束后消失，没有写出控制状态或启动训练，随后改为无触发器计划任务。注册时参数转义错误在开始校准之前修正，实际训练采用已验收的后台路径。

16:10 启动时 supervisor PID 6232，首项 Alameda 原 NDR worker PID 18316。首轮 epoch 1 已落盘 batch 100/2001、updates 100，两个句柄都实测存活，GPU 整卡显存约 6.7 GiB。队列完成数仍为 0/8，尚无性能胜出结论。

每项训练后由新进程严格重建 best checkpoint，完整复算 validation，并验证样本、目标、mask、缓存、源码和运行时身份。整队通过后才写 source postflight 与逐文件哈希清单。后续仍需将结果回传 Mac 并复验，再进行晋级判断。

## 下一阶段实现边界

SharedGraphNDR 的动态 node_scale 需要专门的 loss 适配，不能给模型添加地区专属 sd_node 可训练状态来迁就旧入口。完整 effective batch 的有效计数、teacher confidence 总和和 specialization 权重均值必须先固定，再分 microbatch 反传。

PyTorch 2.10 官方源码对 deterministic algorithms 模式的 CUDA index_add 有确定性路径，不应预防性关闭严格确定性或替换传播算子。5080 nightly 的实际完整 forward、backward、optimizer step 与重复梯度仍需在后续独占 CUDA probe 验收。

严格 source-only 迁移需要新建白名单数据读取器。当前 RegionData 初始化会加载地区 baseline，get_sample 也会构造该地区的基线，因此不宜直接作为严格目标域适配器。新读取器只读取当前 forecast origin 前的 flow、observed mask、时间、已冻结 origin/split 身份和静态道路图，不读取目标域 train 统计或气候态。源端 pooled weekly template 与尺度只从两个源区域唯一 raw train timestamps 拟合，并在源 train、source val 和目标推理始终采用同一种定义。使用目标历史前缀校准的迁移另列结果，不能混为 source-only。

## 首项完整基线已保全

Alameda 原 NDR 在当前 5080 配方下完成 30 epochs，best 为 epoch 14。完整 validation 的全局有效点 MAE 为 `10.745627126021652`，覆盖 6954 个样本和 35,937,114 个有效点。源端由独立进程从 best checkpoint 复算，MAE 差为 0，verification SHA-256 为 `88e07501487e714964abd653d53e3169845e3443e3781ac475c2e23503d0d856`。

全部 11 个文件共 39,001,946 bytes 已取回 Mac，精确文件集合、大小和逐文件 SHA-256 匹配，传输前后源文件清单不变。本地 best 与 last checkpoint 严格结构加载和有限性验证通过，last 保留 optimizer、scheduler、RNG 和 loader 状态。本地没有再次进行完整 validation 推理。

best checkpoint SHA-256 为 `da9b32cff7c27c880f517753f24c132ba1e776328f70672c72d2f1fdff138a41`，last 为 `dc87218a40d80d5737782424fd5b166485d0fb56afad85670c985eb435afbfc6`。证据位于原工作区 `outputs/remote_runs/ndr_generalization_20261013/partial_verified/alameda_baseline_seed42/` 与 `alameda_baseline_local_verification.json`。

它只建立当前开发配方的原 NDR 参照，不与历史 test MAE 比较，不代表整矩阵完成或候选获胜。当时队列为 1/8，接续 ContraCosta 原 NDR，原固定八项顺序不变。

## 两地区原 NDR 基线已保全

最新源端进度回执 `stage1_poll_17.json` 确认 2/8 completed，当前训练项为 `alameda_mask_signed_seed42`，supervisor 和当前 worker 均存活。原固定顺序、30 epochs 与 48 小时截止保持不变。

| 原 NDR 地区 | 最佳 epoch | 完整 validation global MAE | 样本数 | 有效目标点 |
|---|---:|---:|---:|---:|
| Alameda | 14 | 10.745627126021652 | 6954 | 35,937,114 |
| ContraCosta | 17 | 11.27520497996187 | 3472 | 17,308,933 |

两项均完成 30 epochs，源端独立 checkpoint 复算的 MAE 差为 0。新版回传工具按冻结 plan 取回精确文件集合，验证 config、数据、源码、checkpoint sidecar、完整 validation、best 和 last 模型结构以及已完成 epoch 游标。源端传输前后文件清单完全相同。

新版一致布局为原工作区 `outputs/remote_runs/ndr_generalization_20261013/stage1_copied/`，共 22 文件、77,124,280 bytes。逐项回执在其 `receipts/`，两项摘要在同级研究根 `control/baseline_pair_partial_verified.json`。早期 Alameda 副本继续保留。ContraCosta best SHA-256 为 `5bdaf421a4636b7a520e2bae4c9bbcf6fd54ceb11b1aa32a7995d2a8035cf350`，last 为 `eb3bcfbf4528b833c21f04828ed821b42f4e8b8918349de5cf0083e6ff99da0c`，verification 为 `f97eeb62727eb5369092ddf46639f5cb3e4c8d901540c0a87b8a29e4b04a4914`。

本地只进行结构重载与文件复验，没有再次遍历完整 validation，也没有评估 test。这个结果仍是两项参照，候选没有完成，没有整矩阵性能判断。

首项 Alameda 的后续曲线从 epoch 14 到 epoch 30 的 validation MAE 上升约 2.5247%，训练 MAE 继续降低。该单项没有显示 validation 仍改善而被 30 轮截断的明显证据，当前不改变停止线。若后续预声明 100 epoch 配方，应让所有指定方法从 epoch 0 使用相同的新调度周期，不能修改旧 30 epoch checkpoint 的冻结 epochs 并称为严格续训。readiness 记录在 `control/stage1_decision_readiness_review.json`。
