# Impact Correction Final Candidates

Negative delta means the adapter is better than its source checkpoint.

## Recommendation Table

| candidate             | role                                 | evidence_scope   |   overall_all_delta |   overall_affected_delta |   overall_unaffected_delta |   high_and_long_all_delta |   high_and_long_affected_delta |   high_and_long_unaffected_delta | decision                                                                                    |
|:----------------------|:-------------------------------------|:-----------------|--------------------:|-------------------------:|---------------------------:|--------------------------:|-------------------------------:|---------------------------------:|:--------------------------------------------------------------------------------------------|
| anomgate05            | new all-oriented main candidate      | 3-seed mean      |           -0.000441 |                -0.000484 |                  -0.000421 |                 -0.000369 |                      -0.000163 |                        -0.000510 | Promote as strongest all/unaffected candidate; document high-risk affected per-seed caveat. |
| balanced_default      | previous balanced default            | 3-seed mean      |           -0.000204 |                -0.000273 |                  -0.000172 |                 -0.000140 |                      -0.000153 |                        -0.000131 | Keep as simpler/stabler backup; weaker than anomgate on 3-seed mean.                        |
| max05_conservative    | low-correction conservative ablation | 3-seed mean      |           -0.000113 |                -0.000131 |                  -0.000105 |                 -0.000090 |                      -0.000070 |                        -0.000104 | Stable but weaker than balanced; keep as ablation, not backup.                              |
| highfocus_affected    | affected-oriented ablation           | seed23           |           -0.000224 |                -0.000546 |                  -0.000072 |                 -0.000043 |                      -0.000321 |                         0.000148 | Not default; use only to show affected-vs-unaffected tradeoff.                              |
| max12_not_recommended | rejected strong-correction ablation  | seed23           |           -0.000170 |                -0.000639 |                   0.000050 |                  0.000262 |                      -0.000134 |                         0.000534 | Reject as default; high-risk all/unaffected regress.                                        |

## Anomgate05 3-Seed Mean

| group                  |   samples |   all_delta |   affected_delta |   unaffected_delta |   source_affected_mae |   adapter_affected_mae |
|:-----------------------|----------:|------------:|-----------------:|-------------------:|----------------------:|-----------------------:|
| overall                |     27499 |   -0.000441 |        -0.000484 |          -0.000421 |              1.104865 |               1.104381 |
| severity_low           |      9167 |   -0.000290 |        -0.000521 |          -0.000233 |              0.836312 |               0.835791 |
| severity_mid           |      9167 |   -0.000603 |        -0.000833 |          -0.000498 |              0.960233 |               0.959400 |
| severity_high          |      9165 |   -0.000397 |        -0.000233 |          -0.000505 |              1.288906 |               1.288673 |
| recovery_short_lt30    |     12563 |   -0.000356 |        -0.000533 |          -0.000302 |              0.937698 |               0.937165 |
| recovery_mid_30_90     |      3380 |   -0.000634 |        -0.001167 |          -0.000411 |              1.028430 |               1.027263 |
| recovery_long_ge90     |     11556 |   -0.000455 |        -0.000332 |          -0.000533 |              1.193583 |               1.193251 |
| severity_high_and_long |      8029 |   -0.000369 |        -0.000163 |          -0.000510 |              1.276361 |               1.276198 |

## Balanced Default 3-Seed Mean

| group                  |   samples |   all_delta |   affected_delta |   unaffected_delta |   source_affected_mae |   adapter_affected_mae |
|:-----------------------|----------:|------------:|-----------------:|-------------------:|----------------------:|-----------------------:|
| overall                |     27499 |   -0.000204 |        -0.000273 |          -0.000172 |              1.104865 |               1.104592 |
| severity_low           |      9167 |   -0.000184 |        -0.000311 |          -0.000152 |              0.836313 |               0.836002 |
| severity_mid           |      9167 |   -0.000273 |        -0.000399 |          -0.000215 |              0.960241 |               0.959842 |
| severity_high          |      9165 |   -0.000157 |        -0.000175 |          -0.000146 |              1.288900 |               1.288725 |
| recovery_short_lt30    |     12563 |   -0.000221 |        -0.000323 |          -0.000189 |              0.937695 |               0.937372 |
| recovery_mid_30_90     |      3380 |   -0.000261 |        -0.000498 |          -0.000162 |              1.028462 |               1.027965 |
| recovery_long_ge90     |     11556 |   -0.000178 |        -0.000208 |          -0.000158 |              1.193578 |               1.193370 |
| severity_high_and_long |      8029 |   -0.000140 |        -0.000153 |          -0.000131 |              1.276355 |               1.276202 |

## Max05 Conservative 3-Seed Mean

| group                  |   samples |   all_delta |   affected_delta |   unaffected_delta |   source_affected_mae |   adapter_affected_mae |
|:-----------------------|----------:|------------:|-----------------:|-------------------:|----------------------:|-----------------------:|
| overall                |     27499 |   -0.000113 |        -0.000131 |          -0.000105 |              1.104865 |               1.104734 |
| severity_low           |      9167 |   -0.000096 |        -0.000150 |          -0.000083 |              0.836313 |               0.836163 |
| severity_mid           |      9167 |   -0.000145 |        -0.000197 |          -0.000121 |              0.960241 |               0.960044 |
| severity_high          |      9165 |   -0.000096 |        -0.000080 |          -0.000107 |              1.288900 |               1.288820 |
| recovery_short_lt30    |     12563 |   -0.000110 |        -0.000147 |          -0.000099 |              0.937695 |               0.937548 |
| recovery_mid_30_90     |      3380 |   -0.000147 |        -0.000259 |          -0.000101 |              1.028462 |               1.028204 |
| recovery_long_ge90     |     11556 |   -0.000107 |        -0.000099 |          -0.000111 |              1.193578 |               1.193479 |
| severity_high_and_long |      8029 |   -0.000090 |        -0.000070 |          -0.000104 |              1.276355 |               1.276285 |

## Seed23 Variant Affected Delta

| group                  |   anomgate05 |   balanced_default |   highfocus_affected |   max05_conservative |   max12_not_recommended |
|:-----------------------|-------------:|-------------------:|---------------------:|---------------------:|------------------------:|
| overall                |    -0.000525 |          -0.000530 |            -0.000546 |            -0.000316 |               -0.000639 |
| severity_low           |    -0.000515 |          -0.000563 |            -0.000544 |            -0.000326 |               -0.000836 |
| severity_mid           |    -0.000808 |          -0.000793 |            -0.000806 |            -0.000467 |               -0.001131 |
| severity_high          |    -0.000334 |          -0.000339 |            -0.000369 |            -0.000208 |               -0.000239 |
| recovery_short_lt30    |    -0.000611 |          -0.000655 |            -0.000675 |            -0.000355 |               -0.001019 |
| recovery_mid_30_90     |    -0.001040 |          -0.000910 |            -0.000875 |            -0.000570 |               -0.001336 |
| recovery_long_ge90     |    -0.000389 |          -0.000402 |            -0.000427 |            -0.000250 |               -0.000338 |
| severity_high_and_long |    -0.000259 |          -0.000292 |            -0.000321 |            -0.000184 |               -0.000134 |

## Seed23 Variant Unaffected Delta

| group                  |   anomgate05 |   balanced_default |   highfocus_affected |   max05_conservative |   max12_not_recommended |
|:-----------------------|-------------:|-------------------:|---------------------:|---------------------:|------------------------:|
| overall                |    -0.000321 |          -0.000183 |            -0.000072 |            -0.000187 |                0.000050 |
| severity_low           |    -0.000221 |          -0.000201 |            -0.000133 |            -0.000149 |               -0.000160 |
| severity_mid           |    -0.000443 |          -0.000285 |            -0.000188 |            -0.000230 |               -0.000151 |
| severity_high          |    -0.000287 |          -0.000068 |             0.000093 |            -0.000178 |                0.000425 |
| recovery_short_lt30    |    -0.000288 |          -0.000280 |            -0.000217 |            -0.000187 |               -0.000271 |
| recovery_mid_30_90     |    -0.000364 |          -0.000171 |            -0.000067 |            -0.000176 |                0.000016 |
| recovery_long_ge90     |    -0.000340 |          -0.000098 |             0.000058 |            -0.000191 |                0.000353 |
| severity_high_and_long |    -0.000266 |          -0.000024 |             0.000148 |            -0.000167 |                0.000534 |
