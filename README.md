# XTraffic Incident Impact Forecasting Snapshot

This repository is a lightweight handoff snapshot for the XTraffic incident-impact
forecasting experiments.

It intentionally tracks code, notes, and selected experiment summaries only.
Raw XTraffic data, HDF5 caches, model checkpoints, large sample-level CSVs, plots,
and PDFs are excluded from GitHub.

Current model-side handoff:

- Main candidate: `anomgate05`
- Previous backup: `balanced_default`
- Core idea: freeze the group-aware dual-branch ST-TIS source model, then learn a
  local impact correction adapter.
- Key signal: branch disagreement `|incident branch - normal branch|` is used as
  a latent incident-impact gate for correction confidence.

Start reading:

1. `outputs/impact_guided_next_stage/incident_branch_ablation_summary_zh.md`
2. `outputs/impact_guided_next_stage/impact_correction_final_candidates/summary.md`
3. `scripts/train_impact_correction_adapter.py`
4. `scripts/evaluate_impact_correction_adapter.py`
5. `scripts/diagnose_impact_correction_alignment.py`
