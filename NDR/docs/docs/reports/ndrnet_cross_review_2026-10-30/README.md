# NDR-Net 稿件交叉评审记录，2026-10-30

本目录是对 Overleaf 项目 6aa7d46b74bb397c87e2010f 提交 ba2d5c3 的交叉评审，用 peer-review 与 citation-management 两个 skill 的本地脚本完成，重点是把稿件中的每个数字与研究仓库的核验产物逐项对照。稿件正文没有发送到外部服务，CrossRef 只用于核对参考文献的 DOI。

- `private-review.md`：评审正文，含数字交叉核对台账、5 条 major 与 11 条 minor 意见，`lint_review.py` 通过。
- `intake.json` 与 `intake_report.json`：授权与范围声明，状态 READY_FOR_LOCAL_REVIEW。
- `study_profile.json` 与 `guideline_selection.json`：报告指南选择，交通预测无内置指南匹配。
- `claim_matrix.csv` 与 `claim_matrix_report.json`：25 条声明的证据映射，14 条 supported、9 条 partly_supported、2 条 unsupported。
- `stats_checklist.json` 与 `stats_checklist_report.json`：22 项方法、统计与可复现性清单，9 项 verified_present、10 项 partly_documented、1 项 missing、2 项不适用。
- `table_regeneration_check.txt`：用 `scripts/paper/build_ndrnet_paper_assets.py` 重新生成六个表格，与稿件逐字节一致。
- `doi_crossref_check.txt`：37 条参考文献中 25 条 DOI 的 CrossRef 核对结果，全部一致，12 条无 DOI。
- `citation_validation_report.json`：citation-management 的字段完整性报告，0 错误，13 条缺推荐字段的警告。

两条 unsupported 声明分别是 Section 4.12 关于原生长预算迁移尚未测量的表述，以及 Section 4.1 与 Table 2 关于 IGSTGNN 只接收起报时间与已持续时间的描述，详见 private-review.md 的 M1 与 M2。

## 作者处置与落实情况，2026-10-30 晚

作者决定：不在稿件中披露 BiST 原生配方 100 轮的开发集负结果，即 M1 的披露要求不采纳；不在摘要与 Highlights 加单种子与 30 轮限定，即 M3 不采纳。其余意见已落实到 Overleaf 克隆的本地提交，尚未 push。

- M1：只把 Section 4.12 末句从"尚未测量"改为范围表述，即本文不评估插件在骨干原生长预算下的表现，不再对该实验作任何事实性断言。
- M2：Section 4.1 与 Table 2 改为"起报时即已知的属性：事故类型、位置、起报时间与已持续时间"。
- M4：生成脚本中 Table 4 标题加 day-type 入口说明，NDR-Net 行标签改为 "NDR-Net, day-type entry"。
- M5：Section 4.12 改为 15 项消融加 9 项迁移共 24 项窗口级检验；Table 5 标题注明 p 值偏乐观、末列按一致性计数解读。block bootstrap 未做。
- m1 摘要加 absolute；m2 Section 4.10 两处措辞；m3 改为 75% 与 71%；m4 改为原生 5 分钟分辨率；m5 补 ε 取值与官方 split 零流量视为缺失；m6 LargeST 改为基准措辞；m7 Figure 3 改为四面板各自刻度；m8 删除引言脚注、加入 cache digest、利益冲突与生成式 AI 声明草稿并留作者确认 TODO，CRediT 留 TODO；m9 删除 5 个模板残留图与重复的初版 PDF；m10 为 8 条文献补经核实的 URL、3 条 arXiv 条目补 DOI，Lopez-Paz 2016 的 OpenReview 页面未能核实故未加；m11 不采纳：ADNet 是本组自己的上一个模型，作者决定不作为基线加入。
- 修改涉及研究仓库的 `scripts/paper/build_ndrnet_paper_assets.py`，该文件仍在工作树中未提交。重新生成后 Table 4 与 Table 5 的 .tex 仅标题与行标签变化，其余表格与三张未改动的图保持原字节。

## 补充修改，2026-10-31

作者另转来四条针对第 3.4、3.5 节与代码一致性的意见，经对照 `scripts/train_staeformer_deseason.py` 核实后全部落实：初始门控改写为 σ(2 − ln2·(d_n + u_n)) 并说明只有落在周基线上的传感器为 0.88；u_n 的描述改为"最近一步变化的幅度、不含方向"，第 3.4 节 "still-growing" 与第 4.8 节 "instantaneous trend" 一并修正；特化权重均值改为"近似为 1"并注明 ḡ* 的分母含 ε；式 (16) 分母补 +ε。主 MAE 式 (11) 与代码一致，未改。

## Overleaf 提交 3a71fce 的审阅与补回，2026-10-31

协作者在 Overleaf 上精简了第 3 节，技术陈述经核对无误。两处因删减断裂的信息已补回并本地提交：图 1 说明恢复 σ_r、B、P、g、Y、M 的定义与"省略其余损失输入连接"的说明；3.3 节补回自适应嵌入按历史位置与传感器联合索引、先时间后空间注意力的描述，以支撑 Table 7 对 Orange 参数量增长的解释。
