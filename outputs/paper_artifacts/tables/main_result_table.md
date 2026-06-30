# Table 1. Main forecasting results

| Model | Normal branch | Incident residual inputs | All robust MAE | All gain (%) | Affected robust MAE | Affected gain (%) |
| --- | --- | --- | --- | --- | --- | --- |
| statistical normal + residual STGNN | statistical blend | statistical residual | 0.8735 -> 0.7378 | 15.54 | 1.3888 -> 1.1659 | 16.05 |
| learned normal | learned normal STGNN | future residual target only | 0.8328 -> 0.7579 | 8.99 | 1.2938 -> 1.1646 | 9.99 |
| + normal_delta | learned normal STGNN | normal_delta | 0.8328 -> 0.7434 | 10.73 | 1.2938 -> 1.1620 | 10.19 |
| + dual historical residual | learned normal STGNN | normal_delta + dual history | 0.8328 -> 0.7254 | 12.90 | 1.2938 -> 1.1380 | 12.04 |
| + disagreement proxy | learned normal STGNN | normal_delta + abs(normal_delta) + dual history | 0.8328 -> 0.7248 | 12.97 | 1.2938 -> 1.1381 | 12.04 |
| + temporal decay head | learned normal STGNN | normal_delta + abs(normal_delta) + dual history + temporal gate | 0.8328 -> 0.7239 | 13.07 | 1.2938 -> 1.1308 | 12.60 |
| residual temporal decay no-aux | learned normal STGNN | normal_delta + abs(normal_delta) + dual history + temporal gate; no aux labels | 0.8328 -> 0.7221 | 13.30 | 1.2938 -> 1.1290 | 12.74 |
| dual-branch gate | learned normal STGNN | normal-style residual branch + incident graph branch + gate | 0.8328 -> 0.7203 | 13.51 | 1.2938 -> 1.1283 | 12.80 |
| dual-branch gate no-aux | learned normal STGNN | normal-style residual branch + incident graph branch + gate; no aux labels | 0.8328 -> 0.7181 | 13.78 | 1.2938 -> 1.1234 | 13.17 |
| ST-TIS incident branch no-aux | learned normal STGNN | normal-style residual branch + ST-TIS-style incident branch + gate; no aux labels | 0.8328 -> 0.7135 | 14.32 | 1.2938 -> 1.1217 | 13.30 |
| ST-TIS gate-head fine-tune no-aux | learned normal STGNN | freeze residual branches, fine-tune gate head only; no aux labels | 0.8328 -> 0.7116 | 14.56 | 1.2938 -> 1.1189 | 13.52 |
