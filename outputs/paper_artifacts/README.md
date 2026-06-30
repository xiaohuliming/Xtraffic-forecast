# Paper Artifacts

Generated from existing experiment outputs.

## Tables

- `tables/main_result_table.md`
- `tables/component_ablation_table.md`
- `tables/seed_robustness_table.md`
- `tables/temporal_decay_group_table.md`
- `tables/horizon_decay_table.md`
- `tables/gate_branch_interpretability_table.md`
- `tables/gate_case_study_table.md`
- `tables/gate_case_study_mixed_table.md`

LaTeX snippets are also written next to the Markdown tables as `.tex` files.

## Figures

- `figures/method_architecture.png`
- `figures/method_architecture.pdf`
- `figures/horizon_decay_gain_pct.png`
- `figures/horizon_affected_mae.png`
- `figures/ablation_affected_mae.png`
- `figures/severity_recovery_decay_gain.png`

Additional interpretability figures:

- `figures/gate_branch_ablation_mae.png`
- `figures/gate_selection_alignment.png`
- `figures/gate_by_horizon.png`
- `figures/gate_by_event_group.png`
- `figures/case_studies/case_01_sample_192208.png`
- `figures/case_studies/case_02_sample_56226.png`
- `figures/case_studies/case_03_sample_184513.png`
- `figures/case_studies/case_04_sample_184542.png`
- `figures/case_studies_mixed/case_01_success_sample_192208.png`
- `figures/case_studies_mixed/case_02_success_sample_56226.png`
- `figures/case_studies_mixed/case_03_neutral_sample_195028.png`
- `figures/case_studies_mixed/case_04_neutral_sample_187753.png`
- `figures/case_studies_mixed/case_05_failure_sample_88134.png`
- `figures/case_studies_mixed/case_06_failure_sample_60576.png`

## Draft

- `method_experiment_draft.md`
- `manuscript_draft_en.md`
- `manuscript_draft_zh.md`

## Experiment Audit

- `../impact_guided_next_stage/experiment_audit/status_zh.md`
- `../impact_guided_next_stage/dual_branch_gate_interpretability/report_zh.md`
- `case_study_report.md`
- `case_study_report_mixed.md`
- `confidence_gate_experiment_report_zh.md`
- `hard_mining_experiment_report_zh.md`
- `tables/source_gated_residual_extension.csv`
- `tables/source_selected_gate_case_studies.csv`
- `tables/source_selected_gate_case_studies_mixed.csv`

## Writing Helpers

- `paper_outline_zh.md`
- `section_draft_zh.md`
- `related_work_draft_en.md`
- `method_diagram.md`

## LaTeX

- `latex/main.tex`
- `latex/main.pdf`
- `latex_ieee/main.tex`
- `latex_ieee/main.pdf`
- `latex_zh/main.tex`
- `latex_zh/main.pdf`

## References

- `references_todo.bib`

Source script:

```bash
python3 scripts/assemble_paper_artifacts.py
```
