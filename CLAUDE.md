# XTraffic 项目导航与协作约定

最近更新：2026-07-19

## 新会话先读

1. `README.md`
2. `docs/audits/protocol_v3_review_2026-07-19.md`
3. `docs/audits/data_leakage_2026-07-18.md`
4. `docs/audits/result_validity.csv`
5. `XTraffic项目完整汇报.md`
6. `docs/README.md`

`XTraffic项目完整汇报.md` 是研究历史、实验数字和出处的唯一权威长报告。数据是否仍可用于论文，以 v3 复审、2026-07-18 泄露审计和有效性清单为准。

## 当前状态

审计前的研究线已经完成：

- RADS 在 GWN 与 STAEformer 两个骨干、三个区域上完成验证。
- RADS-O 增加逐 horizon 门控与训练期 oracle 蒸馏。
- NDR-Net 在 RADS-O 上增加正常态与扰动态双残差头。
- 三者是包含关系：`RADS ⊂ RADS-O ⊂ NDR-Net`。
- 审计前记录的最低历史 MAE 为 Alameda 的 NDR-Net 11.052。
- Orange 的 STAEformer RADS 验证已经完成。README 中曾写迁移仍在进行，属于过期状态。
- P2 六锚点库实验失败，方法加注线已经关闭。
- baseline zoo 已启动并有本地回传结果，但尚未形成权威诊断汇总。

以上是历史实验进度，不是当前可发表状态。

## 当前协议状态

2026-07-18 审计确认旧 XTraffic 管线存在 P0 数据问题：

1. 全年 `np.interp` 会把未来观测写入历史输入。
2. RADS 气候态在拟合前使用事故标签排除数据，严格 label-free 主张不成立。
3. severity、affected 与最终 duration 含事后信息，部分模型却将其用于输入或损失。
4. 事件协议和 full-window 协议存在切分边界目标重叠。
5. 多个脚本的 scaler 或残差统计越过训练边界。

因此，所有旧 XTraffic 绝对数值和配对结论均暂停进入论文。旧结果只读保留，不能删除或原地覆盖。

v2 随后修复了原始泄露问题，但全面复审又确认了四类影响实验有效性的缺陷：

1. 同一 forecast origin 可能因多个事故重复进入训练或测试，指标会被重复样本扭曲。
2. active event identity 仍可能跨越 split，按锚定事故分组不足以保证严格隔离。
3. 部分验证 MAE 按 batch 或样本均值累计，没有按全部有效观测点统一计算。
4. 多个训练入口缺少一致的 mask、图支持、缓存身份、输出隔离和恢复契约。

因此 v2 已冻结，5080 队列已停止。队列最先完成的 4 个结果只能标记为 `contaminated preliminary`，不能用于论文、模型选择或恢复旧结论。v2 的三地区样本数、远端测试记录和启动时 GPU 状态保留为历史运维记录，不再视为正式协议验收证据。

当前修复版本为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`。它不是最终训练代码 SHA。纳入 `origin/fu` 的合理修复后，完整测试结果为 143 项通过，1 个既有 warning。最终训练代码 SHA 等 main 合并后锁定。默认缓存目录为 `outputs/dist_net/region_data_v3_noleak_dedup`，默认结果根目录为 `outputs/protocol_v3_noleak_dedup`。

三地区 v3 正式缓存已在 `outputs/dist_net/region_data_v3_noleak_dedup` 落盘，并通过两次严格复验。正式 bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。

| 区域 | 正式样本数 | train | val | test |
|---|---:|---:|---:|---:|
| Alameda | 45760 | 32001 | 6954 | 6805 |
| ContraCosta | 23337 | 16414 | 3472 | 3451 |
| Orange | 60015 | 42372 | 9098 | 8545 |

缓存状态现为 `verified`。v3 训练仍未启动，当前没有任何可恢复论文结论的 v3 结果。实时状态和验收记录见 `docs/audits/protocol_v3_review_2026-07-19.md`，实验族机器可读状态见 `docs/audits/result_validity.csv`。

## v2 冻结记录

2026-07-19 曾从提交 `bf3045ac` 构建 v2 三地区缓存。事件样本 train、val、test 数分别为 Alameda 43973、9450、9366，ContraCosta 19159、4026、4040，Orange 75395、15727、14368。full-window 三地区曾记录为 73561、10501、21013，远端曾有 28 项测试通过。45 项队列于 00:39 启动，启动判活时 GPU 利用率为 91%，显存使用约 6.0 GiB。

这些数字只用于追溯 v2 运维历史。它们不能证明 v2 数据和指标有效，也不能写入论文结果。

## 下一步顺序

1. 把已验证的 v3 cache bundle 传输到 5080。
2. 在远端复验 archive、bundle manifest 和三地区缓存文件哈希。
3. main 合并后锁定最终训练代码 SHA，从该精确 SHA 创建 clean clone，并核对 worktree、代码 SHA、bundle ID 与 bundle digest。
4. 使用 `protocol_v3_noleak_dedup` 的新输出目录启动 45 项最小判别矩阵。
5. 队列完成后根据新结果决定论文主张，再评估是否恢复 baseline zoo。

## 最小判别矩阵

- 三地区 STAEformer s0a 与 s0c
- Alameda 与 ContraCosta 的 s0f 与 s0g
- 三地区 ICSF A/B
- 三地区 full-window GWN、FDN 与 STAEformer

## 实验纪律

- 所有汇报数字必须来自已落盘文件，严禁估算填充。
- XTraffic 旧缓存与旧 summary 只读冻结。
- v2 缓存和最先完成的 4 个远程结果同样只读冻结。
- 新结果必须记录协议、split、normalizer 与 climatology 拟合范围、缓存哈希、Git SHA、dirty 状态和 schema version。
- 等参数比较优先，多种子约定为 42、1、2。
- 指标使用 masked MAE，并按需要报告 all、affected 与 unaffected。
- affected 只能作为事后分组指标，除非实验明确声明使用标签监督。
- 负结果如实保留，不为追求正结果反复改停止线。
- 报告类正文不用破折号，不用括号内插说明。
- 代码注释保持最少，commit 信息使用英文。

## 仓库维护

- `scripts/README.md` 是脚本分类入口。
- `outputs/README.md` 说明历史结果和本地产物的关系。
- `docs/archive/` 保存被主报告吸收的旧文档。
- 不批量移动仍被历史入口引用的脚本或结果，避免破坏导入与路径。根目录只保留项目级文档和配置。
- 本地原始数据位于根目录 `archive/`，不进入 Git。
- 远程运行包和日志放到被忽略的 `outputs/remote_runs/`。

## 凭据与远程运行

实验室共享主机、地址和密码集中明文维护在 `config/lab_hosts.env`。运行脚本可以直接 `source` 该文件，也允许环境变量覆盖。私钥、个人令牌和个人账号凭据仍不进入仓库。
