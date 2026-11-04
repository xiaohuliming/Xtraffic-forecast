# Peer-review working draft — REVIEW-NDRNET-LOCAL-20261030

> Private working document prepared at the authors' request with the peer-review, ml-paper-writing and citation-management skills. Human review, policy checks, and factual verification are required. This draft makes no editorial decision. Manuscript: NDR-Net, Overleaf project 6aa7d46b74bb397c87e2010f, commit 711d5e2 plus the local edits recorded in this folder.

## Intake record

- Reviewer capacity: `author_requested_reader`
- Peer-review model: `single_anonymized` (provisional; the target journal has not been chosen)
- Declared processing plan: `approved_ai_assistance`
- Manuscript text is not embedded by the generator.
- Reconfirm conflicts, competence limits, tool use, confidentiality, and deletion or retention obligations before submission.

# Comments to authors

## Evidence-bounded summary

The manuscript proposes NDR-Net, a dual-anchor residual forecaster for traffic flow around incidents. A shared STAEformer encoder predicts residual corrections to a training-fitted weekly climatology and to the last observation, and a monotone gate driven by historical deviation combines the two per sensor and horizon; the gate is distilled toward an oracle anchor preference and the branches receive a specialization loss during training. The evaluation uses three TraffiDent regions under a leakage-free, target-disjoint, incident-anchored protocol, compares fifteen methods with seed 42 and a 30-epoch cap, adds a three-seed ablation with window-level paired tests on two regions, analyses the learned gate and the error by deviation quintile, transfers the plug-in to BiST under a matched contract on the public split, and reports the public-split comparison separately. Principal claims are listed in `claim_matrix.csv` (CLAIM-001 to CLAIM-014).

## Strengths

- Table 2 and Section 4.1 state each leakage source, its effect and the remedy, with deduplication and purge counts in Table 1; this is more transparent than most incident-forecasting evaluations.
- Table 3 reports all 36 region-horizon-metric columns, including the three Contra Costa RMSE columns in which BiST leads, and Section 4.5 states them in the text.
- The ablation (Tables 4 and 5, Figure 3) uses three seeds, paired window-level tests on identical windows, and an affected/unaffected stratification that connects the mechanism to the motivating problem.
- Section 4.8 reports learned gate coefficients and a model-independent deviation-quintile analysis, which makes the mechanism inspectable rather than asserted.
- Section 4.10 states the scope of the transfer result explicitly, including that the matched contract truncates BiST.

## Major comments

### Major comment M1

- Location: Abstract; Section 4.5; Table 3 (CLAIM-001, CLAIM-002)
- Observation: The fifteen-method comparison uses one seed and a 30-epoch cap. The Contra Costa and Orange average-MAE margins over BiST are 1.4 percent, which is of the same order as the seed standard deviations observed for the NDR-Net ladder in Table 4 (0.015 to 0.131).
- Evidence or criterion: Table 4 seed standard deviations; Section 4.13.
- Why it matters: The column-count claim and the per-region ranking could change under reseeding of the strongest baselines.
- Requested action: Before submission, add seeds 1 and 2 for at least NDR-Net, BiST and IGSTGNN in the leakage-free comparison, or keep the single-seed qualifier in the abstract and the first sentence of Section 4.5. The abstract now carries the qualifier.

### Major comment M2

- Location: Section 4.3; Section 4.7 (CLAIM-003, CLAIM-012)
- Observation: The ablation training entry derives the periodic baseline from a weekday/weekend (2 x 288) climatology, whereas the model of Section 3.2 and the main-comparison NDR-Net use a seven-day (7 x 288) climatology.
- Evidence or criterion: Verified against `dist_net/data.py` (baseline_median shape 2 x 288 x N x 3) and `scripts/protocol_v3_event_extension_data.py` (7 x 288); disclosed in Section 4.3.
- Why it matters: The component attribution is measured on a close variant of the described model, not on the model itself.
- Requested action: Either rerun the ladder with the weekly climatology, or keep the disclosure and add the point to the threats to validity. The disclosure and the threats sentence are now in the manuscript.

### Major comment M3

- Location: Section 4.7; Table 5; Section 4.13 (CLAIM-003)
- Observation: Window-level paired tests treat consecutive overlapping windows as independent, and fifteen tests are reported at the 1 percent level without a multiplicity adjustment.
- Evidence or criterion: Section 4.13 acknowledges the optimism of the p-values; the statistics checklist marks `analysis.multiplicity` and `analysis.clustering_repeated_measures` as partly documented.
- Why it matters: Significance statements are stronger than the dependence structure supports, even though the direction agreement across seeds and regions is informative.
- Requested action: Report a day-block or incident-block paired bootstrap for the two ablation steps, or restate the window-level results as descriptive agreement counts. The manuscript now names this in the threats to validity.

### Major comment M4

- Location: Section 4.10; Table 6; Figure 5 (CLAIM-007)
- Observation: The transfer to BiST is shown only on the public split with a 30-epoch matched contract under which BiST reaches 11.74 average MAE in Alameda, against 10.73 for its public recipe in Table 8.
- Evidence or criterion: Table 8 BiST public-recipe row; Section 4.10 second caveat.
- Why it matters: A reader may read the 6 to 9 percent reductions as superiority over BiST, which the design does not test.
- Requested action: Keep the explicit scope statement, and add a leakage-free-protocol transfer result when one is available. No change beyond the existing caveat was made.

### Major comment M5

- Location: Section 4.1; Section 4.3 (CLAIM-014)
- Observation: Protocol enforcement, cache identity and code revisions are described qualitatively, but the manuscript has no data and code availability statement, repository link, or digest identifiers.
- Evidence or criterion: Elsevier journals require a data availability statement; the statistics checklist marks `reproducibility.data_materials_access` and `reproducibility.provenance_versions` as partly documented.
- Why it matters: Reviewers cannot locate the protocol implementation or confirm that every number traces to a verified artifact.
- Requested action: Add a data and code availability section with the repository and the verification manifests. A section with the public repository used in the ADNet paper has been added; the authors should confirm the link and add the cache digest and release tag before submission.

## Minor comments

### Minor comment m1

- Location: Section 4.8 (CLAIM-005)
- Observation: The learned gate coefficients and the gate means by sensor group appear only in the text.
- Evidence or criterion: Values come from the verified run summaries; the claim matrix flags the mechanistic claim for expert reading.
- Why it matters: A compact table would let readers check the mechanism without parsing prose.
- Requested action: Optional: add a two-row table of gate coefficients and gate means per region.

### Minor comment m2

- Location: Section 4.2
- Observation: STWave, STDN and HyperD are discussed as the closest decomposition-based traffic models in Section 2.2 but are absent from Table 3.
- Evidence or criterion: Related work versus baseline list.
- Why it matters: Reviewers from that line of work will ask why.
- Requested action: State that these models were not run under the protocol and that ADNet represents the family. A sentence has been added.

### Minor comment m3

- Location: Section 1, paragraph 3
- Observation: The sentence stated that each listed leakage source favours models that react to incident records, which the current manuscript no longer tests directly.
- Evidence or criterion: Claim matrix CLAIM-009, alignment issue causal_language.
- Why it matters: Causal language without supporting evidence invites a reviewer objection.
- Requested action: Softened to "several of which can inflate the apparent value of incident inputs".

### Minor comment m4

- Location: Sections 2.2, 3.1, 3.3, 3.5, 3.6
- Observation: The two anchors were named inconsistently (periodic reference, weekly baseline, recurrent weekly level, persistence reference).
- Evidence or criterion: ml-paper-writing consistency guidance; terminology scan in this folder.
- Why it matters: Readers must map several names to two objects.
- Requested action: Unified to "periodic baseline" and "persistence anchor" in the methods and evaluation; the explanatory terms "recurrent pattern" and "current level" remain in the introduction.

### Minor comment m5

- Location: Section 4.11
- Observation: "The cap is therefore not binding for NDR-Net but may be for BiST" treated a best epoch near the cap as evidence of continued improvement.
- Evidence or criterion: Best-epoch values alone do not establish convergence.
- Why it matters: Overstatement.
- Requested action: Rephrased as a suggestion and pointed to the 100-epoch native comparison.

### Minor comment m6

- Location: Tables 3, 4, 6 and 8; Figures 2, 3 and 5
- Observation: Metric headers lacked direction indicators and the figures used a default palette.
- Evidence or criterion: ml-paper-writing table and figure guidance (direction symbols, colourblind-safe palettes, grayscale readability).
- Why it matters: Accessibility and quick reading of tables.
- Requested action: Added lower-is-better arrows to metric headers and switched figures to the Okabe-Ito palette with distinct markers and line styles.

### Minor comment m7

- Location: References
- Observation: All 38 entries were checked. Every DOI resolves through CrossRef. OpenAlex or CrossRef confirmed title, authors and venue for STID, LargeST, Pan et al. 2013, ST-Norm, IGSTGNN, BiST, TraffiDent, STDN, HyperD, Bergmeir and Benitez 2012, Vapnik and Vashist 2009, Williams and Hoel 2003, Jacobs et al. 1991, Kapoor and Narayanan 2023, Cerqueira et al. 2020, Jiang and Luo 2022, DCRNN, STTN and Lopez-Paz et al. 2016. TraffiDent and LargeST now carry the proceedings DOI and pages.
- Evidence or criterion: `validate_citations.py --check-dois` report and OpenAlex lookups recorded in this folder.
- Why it matters: Citation errors are a common cause of desk rejection.
- Requested action: The ADNet entry (TAAI 2026) remains "accepted" until the proceedings appear; RevIN (ICLR 2022) has no DOI by design and was confirmed by venue knowledge only.

### Minor comment m8

- Location: Section 4.3
- Observation: Training wall-clock times are not reported.
- Evidence or criterion: ml-paper-writing compute-reporting guidance.
- Why it matters: Readers expect compute cost next to parameter counts.
- Requested action: The manuscript now states that wall-clock times were not recorded in the verified summaries; record them in future runs.

## Methods, statistics, and reproducibility

Assessed from `stats_checklist.json`: 9 items verified present, 10 partly documented, 1 missing, 2 not applicable. The missing item is prespecification: the manuscript does not state a prespecified analysis plan, although the project kept preregistered stop rules for the ablation ladder and the transfer study; stating those rules in a sentence would close the gap. The partly documented items concern the single-seed main comparison, the dependence between overlapping windows, the absence of a multiplicity adjustment, the qualitative description of code revisions and the lack of an availability statement. No specialist review is needed; the central methods are standard spatiotemporal deep learning and paired comparison.

## Ethics, transparency, figures, tables, and citations

The data are public freeway sensor measurements and incident records without personal information. Two negative results (incident-feature injection into STAEformer and the BiST native-recipe 100-epoch comparison) were removed from this version at the authors' request; the authors should decide before submission whether to restore them or to mention their existence in the limitations, since selective omission of negative results is a reporting concern. Figures use a colourblind-safe palette with distinct markers; captions are self-contained. Tables report the pooled twelve-horizon average with the definition in the caption. Citations were verified as described in m7. An Elsevier declaration of generative AI use in the writing process and a CRediT statement must be added by the authors.

## Limitations of this review

The reviewer is the assistant that drafted most of the manuscript text, so this is a structured self-check rather than an independent review. No experiment was reproduced; numbers were compared against the verified result files in the research repository, not recomputed from predictions. The venue is not yet chosen, so page limits, reference style and venue-specific requirements were not checked. The transportation-operations impact of the error reductions was not assessed.

# Confidential comments to editor

> Not applicable to an author-requested internal review. Retained for channel separation only.

## Reviewer disclosures

- Conflicts and editor clearance: none identified; the review was requested by the authors.
- Competence limits or specialist review needed: transportation-operations domain impact was not assessed; no specialist review requested.
- Assistance or tools used and required disclosure: local deterministic skill scripts for intake, claim-evidence, statistics and citation checks; OpenAlex and CrossRef lookups of bibliographic titles and DOIs only.
- Confidentiality or retention issue: manuscript text stayed on the authors' machine and Overleaf project; review artifacts are retained in the research repository under the authors' control.

## Editorial-process or integrity concerns

None identified.
