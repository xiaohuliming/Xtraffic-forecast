# Peer-review working draft — REVIEW-NDRNET-CROSS-20261030

> Private working document prepared at the authors' request with the peer-review and citation-management skills. Human review, policy checks, and factual verification are required. This draft makes no editorial decision. Manuscript: NDR-Net, Overleaf project 6aa7d46b74bb397c87e2010f, commit ba2d5c3 of 2026-10-30, local clone identical to origin/main.

## Intake record

- Reviewer capacity: `author_requested_reader`
- Peer-review model: `single_anonymized`，目标期刊尚未确定，按 Elsevier CAS 模板暂定
- Declared processing plan: `approved_ai_assistance`
- Manuscript text is not embedded by the generator.
- 本次为交叉评审：以稿件 ba2d5c3 为对象，把正文、表格、图注中的每个数字与研究仓库的核验产物逐项对照，用生成脚本重新生成全部表格做字节比对，读取训练代码核对方法描述，并用 CrossRef 核对全部 DOI。没有重新训练或复算任何预测文件。

# Comments to authors

## Evidence-bounded summary

稿件提出 NDR-Net，用共享的 STAEformer 编码器同时修正两个参考水平：训练期拟合的周气候态与最近一次观测，再用只依赖历史偏离的单调门控逐传感器逐步长融合，训练时用事后的锚点偏好蒸馏门控并加分支专门化损失。评估在 TraffiDent 的 Alameda、Contra Costa、Orange 三个地区上进行，协议为目标不相交、按事故锚定、无泄露的 v3 协议，主比较为 14 种方法、seed 42、最多 30 轮；消融为三种子阶梯加窗口级配对检验；另有门控分析、按偏离分位的误差分布、BiST 骨干上的插件迁移，以及单独列出的公开 split 对照。25 条核心声明的证据映射见 `claim_matrix.csv`。

## Cross-check ledger

下表列出本次逐项核对的结果。权威来源均为仓库内已通过核验门禁的落盘文件。

| 核对项 | 稿件位置 | 稿件数值 | 权威来源 | 结果 |
|---|---|---|---|---|
| 六个生成表格 | Tables 3 到 8 | 全部单元格 | `scripts/paper/build_ndrnet_paper_assets.py` 重新生成 | 逐字节一致，见 `table_regeneration_check.txt` |
| 最优列计数 | Abstract、Sec 4.5、Sec 5 | 33 of 36，其余 3 列第二 | `verified_table/table.csv` 与 `asset_report.json` | 一致，Contra Costa 三个 RMSE 列由 BiST 领先 |
| 相对改善 | Sec 4.5 | 3.66、7.83、1.43、7.03、1.42、4.72，MAPE 4.82、0.77、2.07 | `asset_report.json` relative | 一致 |
| 步长差距 | Sec 4.6 | 0.36 到 0.58、0.11 到 0.26、0.16 到 0.23 | `asset_report.json` horizon 与 NDR-Net summary 的 mae_h | 一致 |
| 消融阶梯 | Sec 4.7、Table 4 | 0.412、0.291、0.283；0.131 到 0.020；0.032、0.068；0.543、0.497；4.8%、4.1% | `analysis/multiseed_summary.csv` | 一致 |
| 门控逐种子增益 | Sec 4.7 | Alameda 0.064 到 0.160，Contra Costa 0.090 到 0.185 | `analysis/matrix_45.csv` 逐种子 s0c 减 s0f | 一致，实际值 0.160、0.074、0.064 与 0.090、0.185、0.139 |
| 双头逐种子 | Sec 4.7、Table 5 | 五升一降，一降为 0.008；5 of 6 显著 | `analysis/paired_seed_level.csv` | 一致，Alameda seed 1 为 -0.0081 |
| 受影响传感器 | Sec 4.7 | 0.773、0.774 对 0.530、0.484；相对 4.1% 到 4.8% | `multiseed_summary.csv` | 一致 |
| 门控参数与均值 | Sec 4.8 | 1.41、1.29、约 0.01；0.537、0.525 对 0.642、0.625；RADS 0.80 对 0.88 等 | `asset_report.json` gate | 一致 |
| 分位 MAE | Sec 4.8、Fig 4 | 0.82 到 21.10；1.31 到 21.11 | `asset_report.json` bucket_mae_mean | 一致；两高分位占比为 74.5% 与 70.6% |
| 迁移结果 | Sec 4.9、Table 6、Fig 5 | 7.54%、8.76%、6.44%；97% 到 99%；0.53 到 0.62 与 1.11 到 1.51 | `bist_ndr_pair_retrieved_20261020/analysis/*.csv` 与 10-20 报告 | 一致 |
| 插件参数 | Sec 4.9 | 37,692，约 2.6% | 10-20 报告 | 一致 |
| 匹配契约 | Sec 4.9 | batch 16、30 轮、Adam 1e-3、wd 1e-4、clip 5、cosine | `bist_ndr_pair_profile_v1.json` 与 10-20 报告 | 一致 |
| BiST 公开配方 | Sec 4.9 | 500 轮早停、lr 2e-3、batch 64 | `.third_party/.../bist_dd94adf/src/utils/args.py` 与 `profiles.public_material_v2.json` | 一致，patience 20 |
| 公开 split 数值 | Sec 4.10、Table 8 | 10.76、11.41、11.57；10.73、11.35、11.69；12.76、13.76、13.29；10.50、11.30、11.14 对 10.94、11.58、11.77 | `table4_with_native100.csv` | 一致；NDR 100 轮 Alameda MAE 标准差为 0.0566 |
| 参数量与最佳轮 | Sec 3.6、Sec 4.11、Table 7 | 21,936，1.4%；1.58、1.56、2.03 M；最佳轮 10、22、11 对 28 到 30 与 27 到 28 | `asset_report.json` params 与 best_epoch | 一致 |
| 数据集统计 | Table 1 | 521、496、990；20,937、9,082、35,171；窗口与切分计数 | 完整汇报第 419 行；`cache_bundle_manifest.json` event_count；协议审计表 | 一致 |
| 去重计数 | Sec 4.1 | 17,021、3,884、45,464 | `cache_bundle_manifest.json` duplicates_removed；协议审计 | 一致 |
| 切分比例 | Sec 4.1 | 70%/15%/15% | NDR summary 的 train_end 73584、val_end 89352，全年 105120 槽 | 一致 |
| 填补与气候态拟合 | Sec 4.1 | 因果前向填充；仅原始训练时间戳 | summary 的 imputation_method 与 weekly_climatology_fit | 一致 |
| 主机与代码版本 | Sec 4.3、Sec 4.12 | 一台四卡 RTX 5090、两个代码版本；消融 5080 与 5090 | `verified_table/manifest.json` host_id 与两个 code_sha；`multiseed_summary.csv` host 列 | 一致 |
| 方法超参数 | Sec 3.3、3.4、3.5、4.3 | 24/24/24/80，3+3 层，4 头，FFN 256，dropout 0.1；a=2，b=c=0；β=0.25；λ 1 与 0.3；Adam 1e-3，wd 1e-4，clip 5，batch 16，cosine 到 1e-5，不早停 | `scripts/train_staeformer_deseason.py`、`ndr_extension_profile_v1.json`、`baselines/STAEformer/STAEformer.py` | 一致 |
| 损失公式 | Eqs. 10 到 17 | 置信度、Huber、分母、专门化权重 | `train_staeformer_deseason.py` compute_loss 与 normalizers | 一致，分母为整批常数以保证微批梯度等价 |
| 官方 split 气候态 | Sec 3.2 | 保留窗口位置重数 | `ndr_extension_runner.py` fit_training_climatology | 一致，另有零流量视为缺失的规则稿件未提 |
| IGSTGNN 输入 | Sec 4.1、Table 2 | 只暴露起报时间与已持续时间 | `scripts/build_igstgnn_data_from_h5.py` 头部说明 | 不一致，见 M2 |
| 原生长预算迁移 | Sec 4.9、Sec 4.12 | 尚未测量 | `docs/reports/BiST_NDR同预算对照结果_2026-10-21.md` | 不一致，见 M1 |
| 参考文献 | References | 37 条 | CrossRef 核对 25 条 DOI；12 条无 DOI | 25 条题名、年份、页码全部一致，见 `doi_crossref_check.txt` |

## Strengths

- 全部六个结果表由脚本从核验产物生成，本次重新生成后与稿件逐字节一致，正文中的每个数字都能追溯到落盘文件。
- Section 4.1 与 Table 2 逐条列出泄露来源、影响与补救，并给出去重与跨切分清理计数，透明度高于多数事故预测评估。
- Table 3 报告全部 36 列，包括 BiST 领先的三个 Contra Costa RMSE 列，正文也如实写出。
- 消融使用三种子、同窗口配对检验与受影响分层，把机制与动机问题连接起来；Section 4.8 给出学到的门控系数与模型无关的分位分析。
- Section 4.9 与 4.10 明确说明迁移结果与公开 split 结果的适用边界。

## Major comments

### Major comment M1

- Location: Section 4.9 末段第二条 caveat；Section 4.12 最后一句 "the behavior of the NDR head under each backbone's native long schedule remains to be measured"；Section 5 第二段关于 BiST 迁移的句子；CX-012
- Observation: 稿件称插件在骨干原生长预算下的表现尚未测量。研究仓库中 2026-10-21 的报告记录了一次在 v3 无泄露协议、Alameda 与 Contra Costa、seed 42、完整验证集上的锁定原生 BiST 配方对照：30 轮时插件改善 3.69% 与 1.21%，100 轮时 Alameda 只剩 0.50%，Contra Costa 退化 1.20%，预注册判据 R1 在 100 轮失败。上一轮审查记录已指出这项负结果被删去并建议披露或恢复，当前稿件没有恢复，反而写成尚未测量。
- Evidence or criterion: `docs/reports/BiST_NDR同预算对照结果_2026-10-21.md` 结果表与判据节；`docs/reports/ndrnet_manuscript_skill_review_2026-10-30/private-review.md` 的 Ethics 段；COPE 与 ICMJE 关于完整报告负结果的要求。
- Why it matters: 该句与仓库记录不符，读者会把匹配预算下 6% 到 9% 的改善外推到原生 BiST；结论段关于迁移性的表述会被解读为对充分训练 BiST 的优势。这是声明与证据对齐和选择性报告问题，比任何数字误差都更可能在审稿中被追问。
- Requested action: 把 Section 4.12 该句改写为：一次开发集初步检查显示，在 BiST 原生 100 轮配方下插件的验证集改善在 Alameda 缩小到 0.5% 并在 Contra Costa 反转为 1.2% 的退化，该检查为 seed 42、仅验证集、未读测试集，因此插件在原生长预算下的测试集表现尚待测量。在 Section 4.9 第二条 caveat 中用一句话引用同一结果，并把 Section 5 的迁移句限定为短预算下的一致早期收益。可选：把 STAEformer 加 ICSF 事故特征的零增益结果放回或在 limitations 中提及，它直接支持引言关于事故输入价值可能被高估的论点。

### Major comment M2

- Location: Section 4.1 "Missing Data and Statistics" 段 "namely the incident start and the elapsed time since it"；Table 2 第三行 Remedy 列 "Only the incident start and the elapsed time at the forecast origin are exposed"；Section 4.2 关于 IGSTGNN 接收可部署事故特征的句子；CX-015
- Observation: v3 导出给 IGSTGNN 的输入包含事故类型 one-hot、起报时已持续时间、经纬度与里程桩、三种距离核、节假日标记，以及由类型映射得到的描述码。稿件只写了起报时间与已持续时间。
- Evidence or criterion: `scripts/build_igstgnn_data_from_h5.py` 文件头部对 incident_feat 维 0 到 11、event_features 与 event_distances 的说明；`docs/audits/data_leakage_2026-07-18.md` 第 254 行关于可用信息的规定，即事故类型、位置、起报时间与当前已持续时间。
- Why it matters: 这段描述决定读者如何判断对 IGSTGNN 的比较是否公平。少写了类型与位置会让人以为事故感知基线被削弱到只剩计时信息，反而削弱稿件的公平性主张。
- Requested action: 改写为：事故条件化基线只接收起报时即已知的属性，即事故类型、位置、起报时间与截至预测时刻的已持续时间；最终持续时间、严重度与事后标注的受影响传感器集合被扣留。Table 2 第三行同步修改，Section 4.2 保持引用即可。

### Major comment M3

- Location: Abstract 倒数第三句；Highlights 第四条；Section 1 倒数第二段；Section 5 第二段；CX-001
- Observation: 摘要与 Highlights 给出 33 of 36 与三地区最低 Average MAE，没有写明这是 seed 42 单次训练与 30 轮预算下的结果。Contra Costa 与 Orange 对 BiST 的 Average MAE 优势为 1.43% 与 1.42%，而 Table 4 中 NDR 阶梯的种子标准差在 0.015 到 0.131 之间，BiST 在 v3 协议下的种子方差未知。上一轮审查记录写明摘要已加限定，但当前摘要文本没有。
- Evidence or criterion: Table 3 与 Table 4；Section 4.5 末段自述为 controlled descriptive comparison；Section 5 limitations 段。
- Why it matters: 头条声明在重新播种后可能改变，摘要与正文的谨慎程度不一致会被审稿人视为夸大。
- Requested action: 在摘要与 Highlights 加入 single-seed 与 30-epoch 的限定词；或在投稿前按研究计划为 NDR-Net、BiST 与 IGSTGNN 在 v3 协议下补 seeds 1 与 2，并把 Table 3 的这三行改为三种子均值与标准差。

### Major comment M4

- Location: Table 4 caption 与行标签 "+ dual residual heads (NDR-Net)"；Section 4.3 第三段；Section 4.12；CX-008
- Observation: Table 4 的 NDR-Net 行 10.872 与 Table 3 的 10.88 来自不同训练入口，前者使用按工作日与周末区分的 2 乘 288 气候态，后者使用 7 乘 288 的周气候态。正文已披露，表格本身没有标注。
- Evidence or criterion: `scripts/train_staeformer_deseason.py` 使用 `baseline_median[day_kind, tod]`；`scripts/protocol_v3_event_extension_data.py` 拟合 7 乘 288 基线；上一轮审查 M2。
- Why it matters: 读者会直接对照两表，并把成分归因当成对最终模型的测量。
- Requested action: 在 Table 4 caption 加一句说明该阶梯使用 day-type 气候态入口，行标签改为 "NDR-Net, day-type entry"。更完整的做法是用周气候态重跑阶梯并补齐 Orange 的后两行，可作为修订轮的工作。

### Major comment M5

- Location: Section 4.7 与 Table 5；Section 4.9 窗口级检验句；Section 4.12 "The fifteen paired tests"；CX-022
- Observation: 稿件共报告 15 项消融窗口级检验与 9 项迁移窗口级检验，均以 p 小于 0.01 判定且无多重比较校正，相邻窗口在时间上重叠。Section 4.12 只计入 15 项。
- Evidence or criterion: Table 5 的 15 个 seed 与地区对；Section 4.9 的 9 个 seed 与地区对；Section 4.12 对 p 值乐观性的自述；`stats_checklist.json` 中 analysis.multiplicity 与 analysis.clustering_repeated_measures 为 partly_documented。
- Why it matters: 显著性陈述强于依赖结构所能支持，计数不准会被视为疏漏。
- Requested action: 把 fifteen 改为 twenty-four 或分别写出两类检验的数目；对两步消融与迁移至少补一个按日块或按事故块的配对 bootstrap，或把窗口级结果改写为方向一致性的描述性计数。

## Minor comments

### Minor comment m1

- Location: Abstract "with the largest reductions on incident-affected sensors"；Section 5 第二段同一表述；CX-007
- Observation: 受影响传感器上的绝对降幅最大，相对降幅与其他传感器相近，Section 4.7 已写清楚，摘要与结论没有写 absolute。
- Evidence or criterion: Table 4 的 0.773、0.774 对 0.530、0.484，相对 4.1% 到 4.8%。
- Why it matters: 省略 absolute 会让读者误以为机制在事故处有额外的相对增益。
- Requested action: 摘要与结论改为 largest absolute reductions。

### Minor comment m2

- Location: Section 4.10 第三与第四句；CX-013
- Observation: "below every public-recipe baseline except BiST" 在 Orange 不成立，NDR-Net 30 轮的 11.57 低于 BiST 公开配方的 11.69；"seed standard deviations below 0.06" 只对 MAE 成立，Alameda 的 MAE 标准差为 0.0566，RMSE 标准差为 0.14。
- Evidence or criterion: Table 8 与 `table4_with_native100.csv`。
- Why it matters: 两处措辞与表格不符。
- Requested action: 改为 "below every public-recipe baseline in Orange and below all but BiST in Alameda and Contra Costa"，并把标准差一句限定为 MAE。

### Minor comment m3

- Location: Section 4.8 第二段 "roughly three quarters"；CX-010
- Observation: 两个最高分位的误差占比为 Alameda 74.5% 与 Contra Costa 70.6%。
- Evidence or criterion: `asset_report.json` bucket_mae_mean。
- Why it matters: 四分之三对 Contra Costa 偏高。
- Requested action: 改为 "about 71% to 75%"。

### Minor comment m4

- Location: Section 4.1 第一段 "aggregate the traffic measurements into 5-minute intervals"；CX-017
- Observation: TraffiDent 的 PeMS 数据原生即为 5 分钟分辨率，缓存构建脚本只筛选 Mainline 传感器与前向填充，没有聚合步骤。
- Evidence or criterion: `scripts/build_full_county_cache.py` 第 771 行与填充相关行。
- Why it matters: aggregate 暗示了一个不存在的预处理步骤。
- Requested action: 改为 "use the benchmark's native 5-minute resolution"。

### Minor comment m5

- Location: Section 3.2 与 3.5，Eqs. 10 到 14
- Observation: ε 未给出数值，代码为 1e-6；Section 3.2 写出官方 split 保留窗口重数，但没有写该 split 把零流量视为缺失并用非零中位数拟合；气候态空格子取零的规则没有给出受影响格子的数量。
- Evidence or criterion: `train_staeformer_deseason.py` 的 1e-6；`ndr_extension_profile_v1.json` 的 median_of_nonzero_flow 与 empty_cell_fallback zero。
- Why it matters: 复现者需要这些常量与规则。
- Requested action: 在正文给出 ε 的值，在 Section 3.2 补一句零流量视为缺失的规则，可选报告空格子比例。

### Minor comment m6

- Location: Section 2.1 末句 "BiST proposes a lightweight bidirectional model that scales to large networks~\citep{ma2025bist,liu2023largest}"
- Observation: LargeST 是基准数据集，不是 BiST 的方法来源。
- Evidence or criterion: 两条参考文献的题名。
- Why it matters: 读者会误以为两篇文献都提出了 BiST。
- Requested action: 改为 "and was evaluated on the LargeST benchmark" 之类的措辞。

### Minor comment m7

- Location: Figure 3
- Observation: 左右双坐标轴把全部传感器与受影响传感器画在不同尺度上，两条曲线的视觉斜率不可比。
- Evidence or criterion: 图注与渲染后的图像。
- Why it matters: 双轴图容易被解读为相对变化相同。
- Requested action: 改为两个子图，或画相对骨干的百分比变化。

### Minor comment m8

- Location: main.tex 作者块注释、`\RenewDocumentCommand\printorcid`、被注释的 `\printcredits`、Section 5 的两处 TODO、Section 1 脚注的代码链接
- Observation: 作者顺序仍标注为占位；Elsevier 要求的生成式 AI 声明、CRediT 与利益冲突声明尚未加入；数据与代码可用性段缺 release tag 与 cache digest；引言脚注与 Section 5 重复给出同一仓库链接；仓库是否公开尚未确认。
- Evidence or criterion: 源文件中的 TODO 注释；Elsevier 投稿要求。
- Why it matters: 投稿前必须补齐，否则会被编辑部退回。
- Requested action: 逐项补齐，只保留一处代码链接，确认仓库公开并写入 cache bundle digest 854237ec 与发布标签。

### Minor comment m9

- Location: 编译日志与 Overleaf 项目文件
- Observation: 本地 pdflatex 加 bibtex 编译通过，20 页，无未解析引用；`\maketitle` 处有一个 117 pt 的 Overfull hbox，对应 Highlights 页作者行无空格的渲染；Table 2 的 p 列有 6 处 Underfull；BibTeX 对 3 条 ICLR 条目提示 empty pages。项目目录中残留模板示例文件 Fig1.pdf、Fig2.pdf、Fig3.pdf、grabs.pdf、pic1.pdf 与旧稿 `ndrnet第一版.pdf`。
- Evidence or criterion: `main.log`、`main.blg`、目录列表。
- Why it matters: 排版警告无碍审稿，但残留文件会随投稿包上传。
- Requested action: 清理残留文件；Overfull 与 Underfull 可交由出版社排版处理。

### Minor comment m10

- Location: References
- Observation: 25 条带 DOI 的条目经 CrossRef 核对题名、年份与页码全部一致。12 条无 DOI，分别为 DCRNN、AGCRN、STTN、XTraffic 预印本、STL、Autoformer、FEDformer、RevIN、Distilling、Lopez-Paz 等人、Attention Is All You Need 与 Hinton 蒸馏。XTraffic 与 TraffiDent 为同一数据集的预印本与正式版。
- Evidence or criterion: `doi_crossref_check.txt` 与 `citation_validation_report.json`。
- Why it matters: 无 DOI 条目缺少可解析标识，预印本与正式版并存需要说明。
- Requested action: 为 arXiv 条目补 10.48550 前缀的 DOI 或 URL，为 ICLR 条目补 OpenReview 链接，在首次引用处说明 XTraffic 是 TraffiDent 的预印本名称。

### Minor comment m11

- Location: Section 4.2 基线列表
- Observation: 仓库已有同协议、同 seed 的 FDN 即 ADNet 三地区核验结果，组成 45 行表，稿件为避免引用尚未见刊的 TAAI 论文而移除了该行。
- Evidence or criterion: `outputs/protocol_v3_noleak_dedup/event_full45_fdn_20260812/verified_table/table.csv`；Overleaf 提交 2512327。
- Why it matters: 该方法在三个地区的 Average MAE 均高于 NDR-Net，加入不会改变结论，TAAI 论文公开后审稿人可能询问为何缺席。
- Requested action: 可选。TAAI 论文可引用后恢复该行与引用。

## Methods, statistics, and reproducibility

按 `stats_checklist.json` 评估：9 项 verified_present，10 项 partly_documented，1 项 missing，2 项不适用。缺失项为预先指定：稿件没有写明分析计划，而研究仓库为 BiST-NDR 对照保留了预注册的停止规则。部分记录项集中在单种子主比较、重叠窗口的依赖、无多重比较校正、代码版本与主机的定性描述、可用性声明的 TODO。统计工具建议对依赖结构与多重比较再次确认，本次评审认为常规配对方法在作者能力范围内，不需要外部专家。代码核对方面，门控初始化、Huber 阈值、置信度公式、两个辅助损失的归一化、λ 权重、优化器、学习率计划、梯度裁剪、批大小、编码器维度与主对照 profile 全部与正文一致，损失分母在代码中以整批常数实现，以保证微批训练的梯度与整批公式等价。六个结果表重新生成后与稿件逐字节一致。

## Ethics, transparency, figures, tables, and citations

数据为公开的高速公路传感器与事故记录，不含个人信息。透明度方面的唯一实质问题是 M1：两项开发阶段负结果未写入稿件，且其中一项被表述为尚未测量。图使用色盲友好的 Okabe-Ito 调色板与不同标记，图注自足；Figure 3 的双轴问题见 m7。表格报告十二步合并平均并在 caption 中定义。引文核对见 m10。Elsevier 的生成式 AI 声明、CRediT 与利益冲突声明尚待作者加入。

## Limitations of this review

没有重新训练或从预测文件复算任何指标，数字核对以仓库内已核验的汇总文件为准。训练代码只做静态阅读，没有执行。目标期刊未定，版面、篇幅与引用格式未检查。起草稿件的助手与本次评审属同一工具谱系，本次以独立落盘证据为依据逐项反查，但不能视为完全独立的外部评审。CrossRef 只接收了参考文献的 DOI，稿件正文没有发送到任何外部服务。

# Confidential comments to editor

> Not applicable to an author-requested internal cross-review. Retained for channel separation only.

## Reviewer disclosures

- Conflicts and editor clearance: none identified; the review was requested by the authors.
- Competence limits or specialist review needed: no retraining or reproduction was executed; transportation-operations impact was not assessed; no specialist review requested.
- Assistance or tools used and required disclosure: local deterministic skill scripts for intake, claim-evidence mapping, statistics checklist and lint; the project's own asset-building script for table regeneration; CrossRef lookups of DOIs only.
- Confidentiality or retention issue: manuscript text stayed on the authors' machine and Overleaf project; review artifacts are retained in the research repository under the authors' control.

## Editorial-process or integrity concerns

None beyond the reporting item raised to the authors as M1.
