# 文档索引

文档按用途分为当前入口、审计、方法说明、历史设计和归档记录。不要仅按文件修改日期判断权威性。

## 当前入口

| 文件 | 用途 |
|---|---|
| `../README.md` | 仓库总览与当前有效性状态 |
| `../CLAUDE.md` | 新会话入口、工作顺序与协作约定 |
| `../XTraffic项目完整汇报.md` | 完整研究历史与审计前落盘结果 |
| `audits/protocol_v3_review_2026-07-19.md` | v2 冻结原因、v3 修复范围与当前协议状态 |
| `audits/data_leakage_2026-07-18.md` | 原始数据泄露问题与验收基础 |
| `audits/result_validity.csv` | 各实验族的机器可读有效性状态 |

`XTraffic项目完整汇报.md` 仍是研究历史和数字出处的权威文档，但其中的 XTraffic 数值在 v3 复跑前均属于历史结果。当前是否可用于论文，以 v3 协议复审、数据泄露审计和有效性清单为准。

当前协议为 `protocol_v3_noleak_dedup`，cache schema 为 3。正式缓存构建代码 SHA 为 clean commit `29159f626494de52c69b36fe5bba9854f13cb56c`，最终训练代码 SHA 为 `3a4952837aaf6fbda1403d50b610b1bf40c6098a`。纳入 `origin/fu` 的合理修复并补齐 Windows 路径测试后，Mac 与 5080 均为 143 项通过，1 个既有 warning。v3 正式缓存已在 `outputs/dist_net/region_data_v3_noleak_dedup` 落盘，并通过本地两次及 5080 一次严格复验。bundle ID 为 `bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4`，bundle digest 为 `854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5`。正式样本数与 train、val、test 切分为 Alameda 45760 和 32001/6954/6805，ContraCosta 23337 和 16414/3472/3451，Orange 60015 和 42372/9098/8545。缓存状态为 `verified`，45 项 v3 队列运行中，尚无完成并验收的新结果。v2 和最先完成的 4 个远程结果均已冻结为无效历史记录。

## 当前方法说明

| 文件 | 用途 |
|---|---|
| `RADS_NDR_汇报讲稿.md` | RADS、RADS-O 与 NDR-Net 的层级、机制、无效历史数值和新颖性定位 |

## 历史设计文档

以下文件记录已结束的研究方向。它们用于追溯设计决策，不代表当前实现计划。

| 文件 | 状态 |
|---|---|
| `dist_net_design.md` | 旧 DIST-Net 设计，包含事故特征与双支路线 |
| `impact_guided_start.md` | 旧事故影响引导方向起点 |
| `新架构方向_残差双支笔记.md` | 旧 RGDN 方向简记 |
| `superpowers/specs/` | 已执行或停止的设计规格 |
| `superpowers/plans/` | 已执行或停止的实施计划 |

实验室共享连接配置统一维护在 `config/lab_hosts.env`。历史计划中的旧地址只用于追溯，实际运行先加载这份共享配置。

## 已归档进度文档

`archive/` 只保存已被主报告吸收的旧交接与旧总结。目前包括：

- `项目总结报告_2026-06-10.md`
- `去季节化方向_交接_2026-06-26.md`
- `事故影响感知交通流预测研究方案.md`

新增归档文档必须在文件顶部写明归档日期、替代文档和保留原因。

## 文档维护约定

- 项目状态只在 README、CLAUDE、完整汇报和审计文件中维护。协议状态先看 `audits/protocol_v3_review_2026-07-19.md`。
- 一个数字只认落盘来源，不用估算补齐。
- 历史结果不静默删除。有效性变化通过审计和清单标记。
- 实验室共享地址与密码只在 `config/lab_hosts.env` 维护一份。私钥、个人令牌和个人账号凭据不得写入文档。
- 论文稿引用 XTraffic 数值前，必须检查 `audits/result_validity.csv`。
