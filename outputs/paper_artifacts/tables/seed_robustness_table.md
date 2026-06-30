# Table 3. Seed robustness of the dual-branch gated residual no-aux model

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | Best beta |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.7181 | 1.1234 | 0.5279 | 1.1338 | 1.3648 | 1.0500 |
| 11 | 0.7198 | 1.1296 | 0.5276 | 1.1398 | 1.3742 | 1.0500 |
| 23 | 0.7166 | 1.1188 | 0.5279 | 1.1289 | 1.3566 | 1.0000 |

## Mean and standard deviation

| Metric | Mean +/- std | Gain (%) |
| --- | --- | --- |
| All candidates | 0.7182 +/- 0.0016 | 13.77 +/- 0.20 |
| Affected candidates | 1.1240 +/- 0.0054 | 13.13 +/- 0.42 |
| Unaffected candidates | 0.5278 +/- 0.0002 | - |
| Affected horizon 6 | 1.1342 +/- 0.0055 | - |
| Affected horizon 12 | 1.3652 +/- 0.0088 | - |

# Table 3b. Seed robustness of the ST-TIS-style incident branch no-aux model

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | Best beta |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.7153 | 1.1240 | 0.5236 | 1.1328 | 1.3678 | 1.1000 |
| 11 | 0.7161 | 1.1203 | 0.5265 | 1.1290 | 1.3662 | 1.0000 |
| 23 | 0.7135 | 1.1217 | 0.5220 | 1.1314 | 1.3638 | 1.0000 |

## ST-TIS Mean and standard deviation

| Metric | Mean +/- std | Gain (%) |
| --- | --- | --- |
| All candidates | 0.7150 +/- 0.0013 | 14.15 +/- 0.16 |
| Affected candidates | 1.1220 +/- 0.0019 | 13.28 +/- 0.14 |
| Unaffected candidates | 0.5240 +/- 0.0023 | - |
| Affected horizon 6 | 1.1311 +/- 0.0019 | - |
| Affected horizon 12 | 1.3659 +/- 0.0020 | - |

# Table 3c. Seed robustness of the ST-TIS gate-head fine-tune no-aux model

| Seed | All MAE | Affected MAE | Unaffected MAE | H6 affected MAE | H12 affected MAE | Best beta |
| --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.7150 | 1.1239 | 0.5232 | 1.1325 | 1.3690 | 1.0500 |
| 11 | 0.7138 | 1.1189 | 0.5238 | 1.1278 | 1.3627 | 1.0000 |
| 23 | 0.7116 | 1.1189 | 0.5205 | 1.1290 | 1.3599 | 1.0000 |

## Gate-head fine-tune mean and standard deviation

| Metric | Mean +/- std | Gain (%) |
| --- | --- | --- |
| All candidates | 0.7135 +/- 0.0017 | 14.33 +/- 0.21 |
| Affected candidates | 1.1206 +/- 0.0029 | 13.39 +/- 0.22 |
| Unaffected candidates | 0.5225 +/- 0.0017 | - |
| Affected horizon 6 | 1.1298 +/- 0.0024 | - |
| Affected horizon 12 | 1.3639 +/- 0.0046 | - |
