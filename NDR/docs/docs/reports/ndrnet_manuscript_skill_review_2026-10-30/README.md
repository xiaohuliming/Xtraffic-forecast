# NDR-Net 稿件 skill 审查记录，2026-10-30

本目录是用 peer-review、ml-paper-writing 与 citation-management 三个 skill 对 Overleaf 项目 6aa7d46b74bb397c87e2010f 的 NDR-Net 稿件做的结构化自查，供作者核对。所有脚本均在本机运行，稿件正文没有发送到外部服务；OpenAlex 与 CrossRef 只用于查询参考文献的标题与 DOI。

- `intake.json`：审查范围与授权声明，`validate_review_intake.py` 通过，仅有能力边界披露的提示。
- `claim_matrix.csv`：14 条核心声明与证据映射，`validate_claim_evidence.py` 输出 11 条 supported、3 条 partly_supported。
- `stats_checklist.json`：22 项方法、统计与可复现性清单，9 项 verified_present、10 项 partly_documented、1 项 missing、2 项不适用。
- `private-review.md`：按 Location、Observation、Evidence、Why、Requested action 写的审稿意见，含 5 条 major 与 8 条 minor。
- `citation_validation_report.json`：38 条参考文献的字段完整性与 DOI 解析报告，0 错误。

已直接写进稿件的修改见 Overleaf 提交记录；未做的修改在 private-review.md 的 Requested action 中标明。
