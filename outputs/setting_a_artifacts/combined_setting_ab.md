# Final Comparison Summary

## Setting B — No incident labels (deployment-friendly)

See `outputs/fourier_dual_net/paper_artifacts/setting_b_main_table.md` for the full table.
FourierDualNet `learnable_K3` beats single-branch GraphWaveNet on Alameda (-0.42 all-MAE) and
ContraCosta (-0.32 all-MAE); essentially tied on Orange (-0.02).

## Setting A — With incident labels (same pipeline, matched windows)

# Setting A — FourierDualNet vs IGSTGNN (with incident labels)

Both models trained + evaluated on our `.h5` cache pipeline (event-anchored, imputed).
**IGSTGNN was trained with our fix for the official-code dataloader threading bug** (see paper Section X).
Comparison is done on **matched prediction windows** — for each IGSTGNN sample at adapter `_t_idx = s0`, we find the FDN sample with `sample_start = s0+11` (which predicts the same 12-step future window). Only windows shared between both models are included.

Δ columns = FourierDualNet minus IGSTGNN. Negative = FDN wins.

| Region | n_match / n_IGS | match | IGS all | IGS aff | IGS un | FDN all | FDN aff | FDN un | Δ all | Δ aff | Δ un |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Alameda | 4561/9360 | 48.7% | 13.074 | 19.194 | 12.632 | 12.298 | 17.896 | 11.894 | -0.776 | -1.297 | -0.738 |
| ContraCosta | 1138/4032 | 28.2% | 13.720 | 19.780 | 13.310 | 13.748 | 19.920 | 13.331 | +0.028 | +0.140 | +0.021 |
| Orange | 9704/14352 | 67.6% | 13.765 | 19.142 | 13.465 | 13.296 | 18.301 | 13.016 | -0.469 | -0.840 | -0.448 |

**Findings:**
- Alameda + Orange: FourierDualNet **outperforms** IGSTGNN on all 3 metrics by 0.4–1.3 MAE.
- ContraCosta: essentially **tied** (Δ within ±0.14).
- FourierDualNet's advantage is **larger on affected nodes** than overall, suggesting FFT decomposition
  captures incident-induced anomalies better than IGSTGNN's explicit incident modeling on this pipeline.

## Discovered IGSTGNN bug

During this work we identified a threading-related indexing bug in IGSTGNN's official dataloader
(`batch_samples[i-start_idx]` should be `batch_samples[i]`, in `src/utils/dataloader.py`).
This bug causes each batch of size B to only use the first 2 unique samples (replicated B/2 times),
reducing effective batch size from 48 to 2. Our reported IGSTGNN numbers use the **fixed** code;
we observed +0.28 / +0.65 / unknown MAE improvement (Alameda / CC / Orange) over the buggy code.