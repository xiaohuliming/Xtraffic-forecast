# XTraffic 项目导航与协作约定

最近更新：2026-07-19

## 新会话先读

1. `README.md`
2. `docs/audits/data_leakage_2026-07-18.md`
3. `docs/audits/result_validity.csv`
4. `XTraffic项目完整汇报.md`
5. `docs/README.md`

`XTraffic项目完整汇报.md` 是研究历史、实验数字和出处的唯一权威长报告。数据是否仍可用于论文，以 2026-07-18 审计和有效性清单为准。

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

## 当前最高优先级

2026-07-18 审计确认旧 XTraffic 管线存在 P0 数据问题：

1. 全年 `np.interp` 会把未来观测写入历史输入。
2. RADS 气候态在拟合前使用事故标签排除数据，严格 label-free 主张不成立。
3. severity、affected 与最终 duration 含事后信息，部分模型却将其用于输入或损失。
4. 事件协议和 full-window 协议存在切分边界目标重叠。
5. 多个脚本的 scaler 或残差统计越过训练边界。

因此，所有旧 XTraffic 绝对数值和配对结论均暂停进入论文。旧结果只读保留，不能删除或原地覆盖。

无泄露 v2 代码已经完成：共享协议位于 `dist_net/protocol.py`，正式构建入口为 `scripts/build_full_county_cache.py`，数据测试位于 `tests/data/`。当前完整测试结果为 28 项通过。仓库已同步远端 main 的组员修复，可用部分已经并入正式 `scripts/` 入口，根目录重复脚本和远程运行产物已经归位。

2026-07-19 已在 Mac 上从完整原始数据生成三地区正式缓存与事故关系缓存，并传到 5080。缓存 provenance 指向 main 的 `bf3045ac`，构建工作区为 clean。事件样本 train、val、test 数分别为 Alameda 43973、9450、9366，ContraCosta 19159、4026、4040，Orange 75395、15727、14368。三地区 event group 与目标时刻集合均无交集，full-window 三地区均为 73561、10501、21013，远端 28 项测试全部通过。

5080 的 Tailscale 地址在线。旧的 contaminated baseline zoo 队列已停止并保留日志。45 项 v2 最小判别队列于 2026-07-19 00:39 启动，严格串行，首项为 Alameda STAEformer s0a seed 42。启动判活时 GPU 利用率为 91%，显存使用约 6.0 GiB。当前还没有新 summary，因此旧 XTraffic 结论仍不可恢复。

## 下一步顺序

1. 三地区 `region_data_v2_noleak` 与事故关系缓存已生成。
2. cache smoke test、目标集合互斥检查和 provenance 检查已通过。
3. 正在用 `protocol_v2_noleak` 新结果目录重跑最小判别矩阵。
4. 队列完成后根据新结果决定论文主张，再恢复 baseline zoo。

## 最小判别矩阵

- 三地区 STAEformer s0a 与 s0c
- Alameda 与 ContraCosta 的 s0f 与 s0g
- 三地区 ICSF A/B
- 三地区 full-window GWN、FDN 与 STAEformer

## 实验纪律

- 所有汇报数字必须来自已落盘文件，严禁估算填充。
- XTraffic 旧缓存与旧 summary 只读冻结。
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
