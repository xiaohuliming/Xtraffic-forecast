# Table 4. Where temporal decay helps

## Severity groups

| Group | Samples | No-decay affected MAE | Decay affected MAE | Gain (%) | H6 gain (%) | H12 gain (%) |
| --- | --- | --- | --- | --- | --- | --- |
| severity <= 1.557 | 9167 | 0.8539 | 0.8553 | -0.16 | -0.16 | -0.30 |
| 1.557 < severity <= 2.578 | 9167 | 0.9826 | 0.9811 | 0.15 | 0.16 | 0.06 |
| severity > 2.578 | 9165 | 1.3345 | 1.3206 | 1.05 | 1.10 | 1.22 |

## Recovery groups

| Group | Samples | No-decay affected MAE | Decay affected MAE | Gain (%) | H6 gain (%) | H12 gain (%) |
| --- | --- | --- | --- | --- | --- | --- |
| recovery < 30 min | 12563 | 0.9591 | 0.9572 | 0.19 | 0.16 | 0.27 |
| 30 <= recovery < 90 min | 3380 | 1.0520 | 1.0545 | -0.24 | -0.16 | -0.51 |
| recovery >= 90 min | 11556 | 1.2339 | 1.2224 | 0.94 | 0.98 | 1.05 |
