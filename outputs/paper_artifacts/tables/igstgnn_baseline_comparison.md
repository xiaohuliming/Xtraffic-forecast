# IGSTGNN Baseline Reproduction (KDD'26) — Our Run vs. Paper Reported

Setup: same XTraffic dataset, same 70/15/15 time-based split, same 12→12 horizon, same hyperparameters (warm_epoch disabled, bs=24 on Mac MPS / bs=4 on Windows RTX 5080 due to VRAM limits, otherwise paper defaults).

## Average over horizons 1–12 (raw flow units)

| Region | Metric | Paper | Ours | Δ | Rel |
|---|---|---:|---:|---:|---:|
| Alameda | MAE | 12.69 | 12.86 | +0.17 | +1.3% |
| Alameda | RMSE | 21.73 | 22.56 | +0.83 | +3.8% |
| Contra Costa | MAE | 13.43 | 13.31 | **−0.12** | −0.9% |
| Contra Costa | RMSE | 22.50 | 22.53 | +0.03 | +0.1% |
| Orange | MAE | 13.13 | 13.53 | +0.40 | +3.0% |
| Orange | RMSE | 23.35 | 23.11 | **−0.24** | −1.0% |

**复现验证**: 三个 region MAE 全部在 paper 数字 ±3% 以内，RMSE 在 ±4% 以内，确认 IGSTGNN 复现成功。Contra Costa MAE 与 Orange RMSE 实际略好于 paper 报告。

## Per-horizon MAE — Alameda

| H | Paper | Ours | Δ |
|---|---:|---:|---:|
| 3 | 11.80 | 11.70 | −0.10 |
| 6 | 12.64 | 12.77 | +0.13 |
| 12 | 14.21 | 14.48 | +0.27 |
| Avg | 12.69 | 12.86 | +0.17 |

## Per-horizon MAE — Contra Costa

| H | Paper | Ours | Δ |
|---|---:|---:|---:|
| 3 | 12.50 | 12.30 | −0.20 |
| 6 | 13.44 | 13.33 | −0.11 |
| 12 | 14.81 | 14.89 | +0.08 |
| Avg | 13.43 | 13.31 | −0.12 |

## Per-horizon MAE — Orange

| H | Paper | Ours | Δ |
|---|---:|---:|---:|
| 3 | 12.28 | 12.48 | +0.20 |
| 6 | 13.16 | 13.56 | +0.40 |
| 12 | 14.35 | 15.15 | +0.80 |
| Avg | 13.13 | 13.53 | +0.40 |

## Subgroup analysis (ours, average over horizons 1–12)

### By incident type (test set, raw flow units MAE)

| Type | Alameda n / MAE | Contra Costa n / MAE | Orange n / MAE |
|---|---|---|---|
| Hazard | 1794 / 12.88 | 1252 / 13.25 | 3435 / 13.68 |
| UnknInj | 1037 / 12.72 | 689 / 13.40 | 1030 / 13.33 |
| NoInj | 851 / 13.28 | 456 / 13.73 | 1009 / 13.89 |
| 1141 | 306 / 12.47 | 230 / 12.89 | 412 / 13.15 |
| Other | 255 / 12.97 | 245 / 13.16 | 268 / 12.23 |
| Fire | 136 / 11.54 | 97 / 13.27 | 68 / 12.98 |
| AHazard | 85 / 12.55 | 91 / 12.41 | 57 / 12.29 |
| CarFire | 48 / 13.53 | 36 / 14.41 | 65 / 12.76 |

### By incident duration (MAE)

| Duration | Alameda | Contra Costa | Orange |
|---|---:|---:|---:|
| <30 min | 12.86 | 13.25 | 13.53 |
| 30–120 min | 12.86 | 13.41 | 13.70 |
| ≥120 min | 12.74 | 13.32 | 12.38 |

**观察**: 三个 region 中 IGSTGNN 的 overall MAE 几乎与事故时长无关（短/中/长事故差异 < 0.5）。这印证了 overall 网络平均会稀释局部事故影响——长事故的 affected sensor 集合虽小但严重，被未受影响节点的良好预测平均掉了。**这正好凸显了我们的 affected/tail 评估补充 IGSTGNN 这种 overall-only 报告的价值**。

### Holiday vs non-holiday (MAE)

| | Alameda | Contra Costa | Orange |
|---|---:|---:|---:|
| Non-holiday | 12.84 | 13.31 | 13.51 |
| Holiday | 13.32 | 13.23 | 14.23 |

Holiday 整体略差（特别是 Alameda 和 Orange），与 IGSTGNN 中 holiday embedding 的设计动机一致。

## Reproduction notes

- Trained on per-region incident-anchored windows (same window definition as paper: 12 history + 12 forecast steps, anchored at incident time t)
- Mac MPS: Alameda 40 epochs, Contra Costa 40 epochs (bs=24). RTX 5080: Orange 13 epochs (bs=4, training crashed at epoch 14 due to console close — used best checkpoint at epoch 9, val MAE 13.39)
- Bug fixes applied to released IGSTGNN code (8 distinct bugs in dataloader/engine/model — see `baselines/IGSTGNN/` repo diff). Without these the released code does not run.
