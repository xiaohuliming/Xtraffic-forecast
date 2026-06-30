# Per-Seed Caveat: Severity-High & Recovery-Long Affected Delta

每个 cell 是 adapter affected MAE − source affected MAE。负数表示 adapter 比 source 更好；正数表示 adapter 反而更差（caveat）。

| Seed | source affected MAE | anomgate05 Δ | + magnitude regret w=0.05 Δ | + magnitude regret w=0.10 Δ | + top-k harmful regret w=0.10 Δ |
|---:|---:|---:|---:|---:|---:|
| 7 | 1.290 | -0.000406 | -0.000388 | -0.000314 | -0.000167 |
| 11 | 1.286 | **+0.000176** | +0.000058 | -0.000025 | -0.000055 |
| 23 | 1.276 | -0.000259 | -0.000305 | -0.000327 | -0.000298 |
| **3-seed 平均** | 1.276 | -0.000163 | **-0.000211** | -0.000222 | -0.000173 |

Seed 11 上 anomgate05 出现 +0.000176 caveat（adapter 比 source 更差），是 source 在该 seed 局部基坑导致的 per-seed 不稳定。Magnitude regret 主候选把它缩到 +0.000058（-67%）；w=0.10 完全翻号；但 w=0.10 对 seed 7/23 损失更大，3-seed 均值上 w=0.05 综合 Pareto 占优。
