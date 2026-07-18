# XTraffic Agent 入口

开始工作前依次阅读：

1. `CLAUDE.md`
2. `docs/audits/data_leakage_2026-07-18.md`
3. `docs/audits/result_validity.csv`
4. `XTraffic项目完整汇报.md`

当前 XTraffic 旧缓存已被标记为 `contaminated`。不得把旧结果写成已验证论文结论，不得覆盖旧 summary。数据修复必须使用新 schema、新缓存版本和新输出目录。

无泄露 v2 代码已经实现，默认缓存目录为 `outputs/dist_net/region_data_v2_noleak`，默认结果根目录为 `outputs/protocol_v2_noleak`。完整缓存和最小判别矩阵尚未重跑。

项目方法层级为 `RADS ⊂ RADS-O ⊂ NDR-Net`。完整研究史和数字出处见主报告，当前有效性见审计文件。

所有数字必须来自落盘文件。保留诚实负结果。种子约定为 42、1、2。指标采用 masked MAE。

实验室共享主机、地址和密码统一明文维护在 `config/lab_hosts.env`，脚本直接加载该文件。私钥、个人令牌和个人账号凭据仍不得进入仓库。
