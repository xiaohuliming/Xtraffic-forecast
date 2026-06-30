# Xtraffic-forecast

A reproducibility and application study on **XTraffic 2023** (California mainline sensors:
Alameda 521 nodes, Contra Costa 496, Orange 990; 12-step → 12-step flow forecasting at
5-minute granularity). XTraffic is the largest public incident-annotated traffic dataset,
released to support **incident-aware** forecasting.

## Main finding: incident labels give no measurable gain

On XTraffic, using incident labels does **not** improve flow forecasting. Any label-free
model — from our lightweight FourierDualNet to the community-standard SOTA STAEformer —
matches or clearly beats the incident-aware model IGSTGNN that uses the full label set.

Strongest evidence: **STAEformer (ICCV'23, label-free) beats incident-aware IGSTGNN on all
three regions**, all below IGSTGNN's own reported numbers, and faithfully porting IGSTGNN's
incident module (ICSF) into both GraphWaveNet and STAEformer yields **zero gain**.

| Region | STAEformer (no labels) | IGSTGNN self-reported (uses labels) |
|---|---:|---:|
| Alameda | **11.391** | 12.69 |
| Contra Costa | **12.116** | 13.43 |
| Orange | **12.500** | 13.13 |

## Positive result: adaptive de-seasonalization (v0c)

Sustained digging on the decomposition direction produced the project's first clean,
reproducible, equiparameter, multi-seed, **label-free** architectural gain after six failed
enhancement attempts.

De-seasonalization predicts only the deviation from a cached climatology baseline and adds
the baseline back. Its weakness is incident nodes: the future baseline assumes a return to
the periodic norm, which is wrong while an incident persists. **v0c** fixes this with a
per-node, per-step weight `alpha = exp(-relu(r - r0) / tau)` from the recent residual
magnitude `r`: normal nodes keep the periodic baseline as their anchor, incident nodes
switch the anchor toward persistence (the last observed level). The residual head is
unchanged; only the anchor moves. `alpha = 1` reduces exactly to plain de-seasonalization.

| Region | v0b plain de-season | v0c adaptive | Δ |
|---|---:|---:|---:|
| Alameda | 11.681 | **11.452** | −0.229 |
| Contra Costa | 12.453 | **12.297** | −0.157 |

Seeds 42/1/2 means, equiparam single GraphWaveNet (+2 params), zero per-seed overlap.
An ablation splits the gain ~half global persistence blend, ~half adaptive anomaly
weighting. v0c reaches within noise of STAEformer on Alameda using one fifth the
parameters. Third region (Orange) and a STAEformer-backbone version are in progress.

## Full report

**[XTraffic项目完整汇报.md](XTraffic项目完整汇报.md)** is the single authoritative report and
progress doc — full narrative, every number with its on-disk source, evolution path, the
IGSTGNN audit, honest publishability assessment, and next steps. Superseded docs are kept
under `docs/archive/`.

## Repo layout

```
fourier_dual_net/   FDN (FFT decomposition + 2 GraphWaveNet backbones) and rgdn.py
                    (de-seasonalization + AdaptiveAlpha: v0a/v0b/v0c/v0d variants)
dist_net/           event-anchored + full-window data pipeline (region cache, loaders)
baselines/
  GraphWaveNet/     single-branch GraphWaveNet
  STAEformer/       vendored ICCV'23 STAEformer (strongest label-free baseline)
  IGSTGNN/          vendored KDD'26 IGSTGNN with our dataloader threading-bug fix
scripts/            training, evaluation, plotting, and data-prep scripts
tests/              model and de-seasonalization unit tests
docs/               design docs; docs/archive/ holds superseded progress docs
outputs/            result tables, diagnostics, and paper artifacts (no raw predictions)
```

## Key scripts

- `scripts/build_full_county_cache.py` — build the h5 flow cache and climatology baseline.
- `scripts/train_rgdn.py` — de-seasonalization variants (v0a raw, v0b de-season, v0c adaptive,
  v0d const-alpha, plus the older RGDN dual-branch variants).
- `scripts/train_staeformer_xtraffic.py` / `scripts/train_staeformer_deseason.py` — STAEformer
  baseline and the de-seasonalization wrapper around it (backbone-agnostic test).
- `scripts/train_fourier_dual_net.py` / `scripts/train_graphwavenet.py` — FDN and GWN.
- `scripts/significance_tests.py` — paired significance and seed noise bands.

## Data

Raw XTraffic h5 caches, climatology baselines, model checkpoints, and full
`test_predictions.npz` arrays are not in this repo (multiple GB). The pipeline regenerates
them from the scripts above; defaults match the numbers in the full report.

## Notes on the IGSTGNN baseline

Our IGSTGNN numbers use IGSTGNN's official code with a one-line fix to
`src/utils/dataloader.py`: the per-thread inner loop indexes `batch_samples[i-start_idx]`,
collapsing every thread's view of the batch to the first `chunk_size` samples. With
`batch_size=48` and `num_threads=24`, the effective batch was 2 unique samples repeated 24×.
The correct index is `batch_samples[i]`. All our comparisons use the fixed version, which is
the setting most favorable to IGSTGNN.

## License

The FDN code, scripts, and result artifacts in this repo are released under the same terms as
the underlying baselines. Vendored baselines keep their original licenses.
