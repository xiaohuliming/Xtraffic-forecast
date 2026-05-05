# Impact Correction Seed23 Targeted Sweep

Negative delta means the adapter is better than the source.

## Affected Delta

| group                  |   highfocus_finalselect |   highfocus_loss |   tailtarget_balanced |
|:-----------------------|------------------------:|-----------------:|----------------------:|
| overall                |               -0.000546 |        -0.000428 |             -0.000530 |
| severity_low           |               -0.000544 |        -0.000396 |             -0.000563 |
| severity_mid           |               -0.000806 |        -0.000600 |             -0.000793 |
| severity_high          |               -0.000369 |        -0.000320 |             -0.000339 |
| recovery_short_lt30    |               -0.000675 |        -0.000435 |             -0.000655 |
| recovery_mid_30_90     |               -0.000875 |        -0.000703 |             -0.000910 |
| recovery_long_ge90     |               -0.000427 |        -0.000372 |             -0.000402 |
| severity_high_and_long |               -0.000321 |        -0.000295 |             -0.000292 |

## All Delta

| group                  |   highfocus_finalselect |   highfocus_loss |   tailtarget_balanced |
|:-----------------------|------------------------:|-----------------:|----------------------:|
| overall                |               -0.000224 |        -0.000213 |             -0.000294 |
| severity_low           |               -0.000215 |        -0.000178 |             -0.000273 |
| severity_mid           |               -0.000384 |        -0.000311 |             -0.000445 |
| severity_high          |               -0.000091 |        -0.000150 |             -0.000176 |
| recovery_short_lt30    |               -0.000325 |        -0.000215 |             -0.000368 |
| recovery_mid_30_90     |               -0.000305 |        -0.000300 |             -0.000389 |
| recovery_long_ge90     |               -0.000130 |        -0.000190 |             -0.000216 |
| severity_high_and_long |               -0.000043 |        -0.000132 |             -0.000133 |

## Unaffected Delta

| group                  |   highfocus_finalselect |   highfocus_loss |   tailtarget_balanced |
|:-----------------------|------------------------:|-----------------:|----------------------:|
| overall                |               -0.000072 |        -0.000112 |             -0.000183 |
| severity_low           |               -0.000133 |        -0.000124 |             -0.000201 |
| severity_mid           |               -0.000188 |        -0.000178 |             -0.000285 |
| severity_high          |                0.000093 |        -0.000038 |             -0.000068 |
| recovery_short_lt30    |               -0.000217 |        -0.000148 |             -0.000280 |
| recovery_mid_30_90     |               -0.000067 |        -0.000132 |             -0.000171 |
| recovery_long_ge90     |                0.000058 |        -0.000075 |             -0.000098 |
| severity_high_and_long |                0.000148 |        -0.000020 |             -0.000024 |

Conclusion: `tailtarget_balanced` is the balanced choice. `highfocus_finalselect` slightly improves affected deltas, including high/long groups, but weakens unaffected deltas on high-risk groups.
