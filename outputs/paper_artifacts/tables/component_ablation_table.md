# Table 2. Learned-normal and gated-residual component ablation

| Variant | Added signal | All MAE | Affected MAE | Step gain all (%) | Step gain affected (%) | Best beta |
| --- | --- | --- | --- | --- | --- | --- |
| learned normal | future residual target only | 0.7579 | 1.1646 | - | - | 1.0000 |
| + normal_delta | normal_delta | 0.7434 | 1.1620 | 1.92 | 0.22 | 0.9500 |
| + dual historical residual | normal_delta + dual history | 0.7254 | 1.1380 | 2.43 | 2.07 | 1.0000 |
| + disagreement proxy | normal_delta + abs(normal_delta) + dual history | 0.7248 | 1.1381 | 0.08 | -0.01 | 1.0000 |
| + temporal decay head | normal_delta + abs(normal_delta) + dual history + temporal gate | 0.7239 | 1.1308 | 0.12 | 0.64 | 1.0000 |
| residual temporal decay no-aux | normal_delta + abs(normal_delta) + dual history + temporal gate; no aux labels | 0.7221 | 1.1290 | 0.25 | 0.16 | 1.0000 |
| dual-branch gate | normal-style residual branch + incident graph branch + gate | 0.7203 | 1.1283 | 0.25 | 0.07 | 1.1000 |
| dual-branch gate no-aux | normal-style residual branch + incident graph branch + gate; no aux labels | 0.7181 | 1.1234 | 0.31 | 0.43 | 1.0500 |
| ST-TIS incident branch no-aux | normal-style residual branch + ST-TIS-style incident branch + gate; no aux labels | 0.7135 | 1.1217 | 0.64 | 0.15 | 1.0000 |
| ST-TIS gate-head fine-tune no-aux | freeze residual branches, fine-tune gate head only; no aux labels | 0.7116 | 1.1189 | 0.27 | 0.25 | 1.0000 |
