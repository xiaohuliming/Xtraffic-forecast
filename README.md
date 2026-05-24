# Xtraffic-forecast

Traffic forecasting on the XTraffic 2023 California dataset (Alameda, Contra Costa, Orange).
Headline method: **FourierDualNet (FDN)** — an FFT-decomposed dual-branch GraphWaveNet
that splits the historical flow into a low-frequency *Main* component and a
high-frequency *Pert* component, runs a dedicated backbone on each, and sums the
two predictions. The model never reads incident labels.

This repo also vendors the head-to-head baselines:
- `baselines/GraphWaveNet/` — single-branch reference.
- `baselines/IGSTGNN/` — KDD'26 incident-aware SOTA (with our **dataloader threading-bug fix**;
  see *Notes* below).

## Results

| Setting | Region | Model | Needs labels | MAE all ↓ | MAE affected ↓ | MAE unaffected ↓ |
|---|---|---|:-:|---:|---:|---:|
| B | Alameda | GraphWaveNet | No | 12.40 | 18.23 | 12.04 |
| B | Alameda | **FourierDualNet (learnable)** | **No** | **11.98** | **17.76** | **11.62** |
| A | Alameda | IGSTGNN (bug-fixed) | Yes | 13.07 | 19.19 | 12.63 |
| A | Alameda | **FourierDualNet (matched-window)** | **No** | **12.30** | **17.90** | **11.89** |
| B | ContraCosta | GraphWaveNet | No | 13.45 | 19.77 | 13.14 |
| B | ContraCosta | **FourierDualNet (learnable)** | **No** | **13.13** | 19.57 | **12.82** |
| A | ContraCosta | IGSTGNN (bug-fixed) | Yes | 13.72 | 19.78 | 13.31 |
| A | ContraCosta | FourierDualNet (matched-window) | No | 13.75 | 19.92 | 13.33 |
| B | Orange | GraphWaveNet | No | 13.01 | **18.21** | 12.76 |
| B | Orange | **FourierDualNet (learnable)** | **No** | **13.00** | 18.28 | **12.74** |
| A | Orange | IGSTGNN (bug-fixed) | Yes | 13.77 | 19.14 | 13.47 |
| A | Orange | **FourierDualNet (matched-window)** | **No** | **13.30** | **18.30** | **13.02** |

**Setting B** = no incident labels (deployment-friendly), evaluated on the full FDN test set.
**Setting A** = incident labels available, evaluated on the *intersection* of IGSTGNN and FDN
prediction windows so the two models predict the same 12-step future.
FDN never reads incident labels in either setting — the only difference is the eval window.

Full tables: [outputs/setting_a_artifacts/master_comparison_table.md](outputs/setting_a_artifacts/master_comparison_table.md).

## Repo layout

```
fourier_dual_net/        FDN model (FFT decomp + 2 GraphWaveNet backbones, optional cross-attn + conditioned mask)
dist_net/                Earlier dual-branch design (kept for reference and ablation)
baselines/
  GraphWaveNet/          single-branch GraphWaveNet
  IGSTGNN/               vendored KDD'26 IGSTGNN with our dataloader bug-fix
scripts/                 training, evaluation, plotting, and data-prep scripts
docs/                    design docs
outputs/                 result tables, plots, and breakdown JSONs (no raw predictions — see Data)
```

## Data

Raw XTraffic h5 caches, model checkpoints, and full `test_predictions.npz` arrays are not in
this repo (totals ~3.6 GB of NPZ + 14 GB of converted IGSTGNN data). The pipeline can regenerate
all of them:

1. `scripts/build_full_county_cache.py` — build the h5 flow cache for a county.
2. `scripts/train_fourier_dual_net.py` — train FDN; defaults match the numbers in this README.
3. `scripts/train_graphwavenet.py` — train the single-branch baseline.
4. `scripts/build_igstgnn_data_from_h5.py` — adapter to convert our h5 cache into the
   dict-format IGSTGNN expects.
5. `scripts/build_paper_artifacts.py` + `scripts/build_setting_a_artifacts.py` +
   `scripts/build_final_master_table.py` — regenerate the markdown tables under `outputs/`.

## Notes on the IGSTGNN baseline

Our reported IGSTGNN numbers use IGSTGNN's official code with a one-line fix to
`src/utils/dataloader.py`: the per-thread inner loop indexes `batch_samples[i-start_idx]`,
which collapses every thread's view of the batch to the first `chunk_size` samples. With
`batch_size=48` and `num_threads=24`, the effective batch was 2 unique samples repeated 24×.
The correct index is `batch_samples[i]`. After the fix, MAE improves by 0.28 / 0.65 on
Alameda / ContraCosta. Diff is in `baselines/IGSTGNN/src/utils/dataloader.py`.

## License

The FDN code, scripts, and result artifacts in this repo are released under the same terms
as the underlying baselines. The IGSTGNN vendor copy keeps its original `LICENSE`.
