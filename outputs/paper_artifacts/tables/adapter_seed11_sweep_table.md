# Seed-11 Sweep: Filter Selectivity × Weight Pareto

固定 anomgate05 anomaly gate（floor=0.25, threshold=0.5, temperature=0.25），扫描 selective regret filter 的 magnitude quantile q 与 weight λ。

| 配置 | Overall affected Δ | High_and_long affected Δ |
|---|---:|---:|
| anomgate05 (no regret) | -0.000518 | **+0.000176** |
| q=0.95, λ=0.10 | -0.000503 | +0.000040 |
| q=0.90, λ=0.05 (主候选) | -0.000507 | +0.000058 |
| q=0.90, λ=0.10 | -0.000478 | -0.000025 |
| q=0.80, λ=0.10 | -0.000422 | -0.000056 |
| q=0.90, λ=0.20 | -0.000357 | -0.000093 |

观察：filter selectivity (q) 与 weight (λ) 在 Pareto 上几乎完全可互换——q=0.95 + λ=0.10 ≈ q=0.90 + λ=0.05，q=0.80 + λ=0.10 ≈ q=0.90 + λ=0.20。这说明 selective regret 实质由"regret pressure"这个一维标量决定。
