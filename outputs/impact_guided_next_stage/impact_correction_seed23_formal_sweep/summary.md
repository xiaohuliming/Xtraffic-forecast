# Impact Correction Seed23 Formal Sweep

Negative delta means the adapter is better than the source.

## Smoke Test Metrics

| run                   |      all |   affected |   unaffected |   max_correction |   target_margin | selection   |
|:----------------------|---------:|-----------:|-------------:|-----------------:|----------------:|:------------|
| max12                 | 0.706688 |   1.101313 |     0.521818 |         1.200000 |        0.020000 | loss        |
| highfocus_finalselect | 0.706614 |   1.101369 |     0.521684 |         0.800000 |        0.020000 | final_loss  |
| balanced_default      | 0.706547 |   1.101386 |     0.521578 |         0.800000 |        0.020000 |             |
| balanced_finalselect  | 0.706547 |   1.101386 |     0.521578 |         0.800000 |        0.020000 | final_loss  |
| margin0               | 0.706551 |   1.101393 |     0.521580 |         0.800000 |        0.000000 | loss        |
| margin005             | 0.706598 |   1.101497 |     0.521601 |         0.800000 |        0.050000 | loss        |
| max05                 | 0.706606 |   1.101584 |     0.521572 |         0.500000 |        0.020000 | loss        |

## Full Group Affected Delta

| group                  |   balanced_default |   highfocus_finalselect |     max05 |     max12 |
|:-----------------------|-------------------:|------------------------:|----------:|----------:|
| overall                |          -0.000530 |               -0.000546 | -0.000316 | -0.000639 |
| severity_low           |          -0.000563 |               -0.000544 | -0.000326 | -0.000836 |
| severity_mid           |          -0.000793 |               -0.000806 | -0.000467 | -0.001131 |
| severity_high          |          -0.000339 |               -0.000369 | -0.000208 | -0.000239 |
| recovery_short_lt30    |          -0.000655 |               -0.000675 | -0.000355 | -0.001019 |
| recovery_mid_30_90     |          -0.000910 |               -0.000875 | -0.000570 | -0.001336 |
| recovery_long_ge90     |          -0.000402 |               -0.000427 | -0.000250 | -0.000338 |
| severity_high_and_long |          -0.000292 |               -0.000321 | -0.000184 | -0.000134 |

## Full Group All Delta

| group                  |   balanced_default |   highfocus_finalselect |     max05 |     max12 |
|:-----------------------|-------------------:|------------------------:|----------:|----------:|
| overall                |          -0.000294 |               -0.000224 | -0.000228 | -0.000170 |
| severity_low           |          -0.000273 |               -0.000215 | -0.000185 | -0.000294 |
| severity_mid           |          -0.000445 |               -0.000384 | -0.000305 | -0.000461 |
| severity_high          |          -0.000176 |               -0.000091 | -0.000190 |  0.000160 |
| recovery_short_lt30    |          -0.000368 |               -0.000325 | -0.000227 | -0.000447 |
| recovery_mid_30_90     |          -0.000389 |               -0.000305 | -0.000292 | -0.000382 |
| recovery_long_ge90     |          -0.000216 |               -0.000130 | -0.000214 |  0.000085 |
| severity_high_and_long |          -0.000133 |               -0.000043 | -0.000174 |  0.000262 |

## Full Group Unaffected Delta

| group                  |   balanced_default |   highfocus_finalselect |     max05 |     max12 |
|:-----------------------|-------------------:|------------------------:|----------:|----------:|
| overall                |          -0.000183 |               -0.000072 | -0.000187 |  0.000050 |
| severity_low           |          -0.000201 |               -0.000133 | -0.000149 | -0.000160 |
| severity_mid           |          -0.000285 |               -0.000188 | -0.000230 | -0.000151 |
| severity_high          |          -0.000068 |                0.000093 | -0.000178 |  0.000425 |
| recovery_short_lt30    |          -0.000280 |               -0.000217 | -0.000187 | -0.000271 |
| recovery_mid_30_90     |          -0.000171 |               -0.000067 | -0.000176 |  0.000016 |
| recovery_long_ge90     |          -0.000098 |                0.000058 | -0.000191 |  0.000353 |
| severity_high_and_long |          -0.000024 |                0.000148 | -0.000167 |  0.000534 |

## Decision

- `balanced_default` remains the recommended default because it improves all/affected/unaffected on every group.
- `max05` is the conservative option: weaker affected gains, but better high-risk unaffected/all stability than the stronger adapters.
- `max12` is not recommended as default: it improves affected overall but hurts high-risk unaffected and turns high-risk all deltas positive.
- `highfocus_finalselect` is affected-oriented only: slightly better affected deltas, but high-risk unaffected becomes worse.
