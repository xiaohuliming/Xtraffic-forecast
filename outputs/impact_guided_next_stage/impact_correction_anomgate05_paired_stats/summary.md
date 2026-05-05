# Impact Correction Paired Sample Stats

Delta is adapter sample MAE minus source sample MAE; negative is better.
The confidence interval is a normal approximation over event-level sample deltas, so it is supporting evidence rather than a full seed-level significance test.

## Seed-Mean Focus

| group                  | target     |   seeds |   seed_mean_delta |   seed_std_delta | all_seeds_improved   |   mean_improve_rate |
|:-----------------------|:-----------|--------:|------------------:|-----------------:|:---------------------|--------------------:|
| overall                | all        |       3 |         -0.000385 |         0.000184 | True                 |            0.562820 |
| overall                | affected   |       3 |         -0.000396 |         0.000063 | True                 |            0.541220 |
| overall                | unaffected |       3 |         -0.000403 |         0.000271 | True                 |            0.562467 |
| severity_high          | all        |       3 |         -0.000361 |         0.000140 | True                 |            0.542826 |
| severity_high          | affected   |       3 |         -0.000167 |         0.000361 | False                |            0.519355 |
| severity_high          | unaffected |       3 |         -0.000511 |         0.000472 | True                 |            0.560041 |
| recovery_long_ge90     | all        |       3 |         -0.000430 |         0.000218 | True                 |            0.555757 |
| recovery_long_ge90     | affected   |       3 |         -0.000246 |         0.000180 | True                 |            0.528233 |
| recovery_long_ge90     | unaffected |       3 |         -0.000541 |         0.000470 | True                 |            0.565837 |
| severity_high_and_long | all        |       3 |         -0.000337 |         0.000149 | True                 |            0.537593 |
| severity_high_and_long | affected   |       3 |         -0.000093 |         0.000408 | False                |            0.513763 |
| severity_high_and_long | unaffected |       3 |         -0.000520 |         0.000521 | True                 |            0.556693 |

## Per-Seed Focus

| label   | group                  | target     |   samples |   mean_delta |   ci95_low |   ci95_high |   improve_rate |   p_normal_approx |
|:--------|:-----------------------|:-----------|----------:|-------------:|-----------:|------------:|---------------:|------------------:|
| seed7   | overall                | all        |     27499 |    -0.000225 |  -0.000253 |   -0.000197 |       0.544456 |          0.000000 |
| seed11  | overall                | all        |     27499 |    -0.000585 |  -0.000619 |   -0.000551 |       0.598240 |          0.000000 |
| seed23  | overall                | all        |     27499 |    -0.000344 |  -0.000376 |   -0.000311 |       0.545765 |          0.000000 |
| seed7   | overall                | affected   |     26205 |    -0.000325 |  -0.000384 |   -0.000265 |       0.541347 |          0.000000 |
| seed11  | overall                | affected   |     26205 |    -0.000441 |  -0.000517 |   -0.000365 |       0.539477 |          0.000000 |
| seed23  | overall                | affected   |     26205 |    -0.000423 |  -0.000492 |   -0.000354 |       0.542835 |          0.000000 |
| seed7   | overall                | unaffected |     27393 |    -0.000190 |  -0.000222 |   -0.000157 |       0.530172 |          0.000000 |
| seed11  | overall                | unaffected |     27393 |    -0.000708 |  -0.000741 |   -0.000675 |       0.625744 |          0.000000 |
| seed23  | overall                | unaffected |     27393 |    -0.000310 |  -0.000345 |   -0.000274 |       0.531486 |          0.000000 |
| seed7   | severity_high          | all        |      9165 |    -0.000254 |  -0.000317 |   -0.000191 |       0.539989 |          0.000000 |
| seed11  | severity_high          | all        |      9165 |    -0.000519 |  -0.000597 |   -0.000441 |       0.556028 |          0.000000 |
| seed23  | severity_high          | all        |      9165 |    -0.000310 |  -0.000384 |   -0.000237 |       0.532460 |          0.000000 |
| seed7   | severity_high          | affected   |      9162 |    -0.000427 |  -0.000549 |   -0.000306 |       0.544968 |          0.000000 |
| seed11  | severity_high          | affected   |      9162 |     0.000245 |   0.000085 |    0.000405 |       0.484610 |          0.002629 |
| seed23  | severity_high          | affected   |      9162 |    -0.000318 |  -0.000464 |   -0.000173 |       0.528487 |          0.000017 |
| seed7   | severity_high          | unaffected |      9141 |    -0.000170 |  -0.000245 |   -0.000096 |       0.511760 |          0.000007 |
| seed11  | severity_high          | unaffected |      9141 |    -0.001051 |  -0.001127 |   -0.000974 |       0.642380 |          0.000000 |
| seed23  | severity_high          | unaffected |      9141 |    -0.000314 |  -0.000393 |   -0.000234 |       0.525982 |          0.000000 |
| seed7   | recovery_long_ge90     | all        |     11556 |    -0.000257 |  -0.000311 |   -0.000203 |       0.543787 |          0.000000 |
| seed11  | recovery_long_ge90     | all        |     11556 |    -0.000674 |  -0.000741 |   -0.000607 |       0.578834 |          0.000000 |
| seed23  | recovery_long_ge90     | all        |     11556 |    -0.000358 |  -0.000420 |   -0.000296 |       0.544652 |          0.000000 |
| seed7   | recovery_long_ge90     | affected   |     11541 |    -0.000360 |  -0.000464 |   -0.000256 |       0.543280 |          0.000000 |
| seed11  | recovery_long_ge90     | affected   |     11541 |    -0.000038 |  -0.000176 |    0.000099 |       0.505156 |          0.587283 |
| seed23  | recovery_long_ge90     | affected   |     11541 |    -0.000339 |  -0.000458 |   -0.000220 |       0.536262 |          0.000000 |
| seed7   | recovery_long_ge90     | unaffected |     11531 |    -0.000196 |  -0.000260 |   -0.000133 |       0.517041 |          0.000000 |
| seed11  | recovery_long_ge90     | unaffected |     11531 |    -0.001077 |  -0.001143 |   -0.001011 |       0.648079 |          0.000000 |
| seed23  | recovery_long_ge90     | unaffected |     11531 |    -0.000350 |  -0.000418 |   -0.000282 |       0.532391 |          0.000000 |
| seed7   | severity_high_and_long | all        |      8029 |    -0.000229 |  -0.000298 |   -0.000160 |       0.535185 |          0.000000 |
| seed11  | severity_high_and_long | all        |      8029 |    -0.000507 |  -0.000593 |   -0.000420 |       0.549010 |          0.000000 |
| seed23  | severity_high_and_long | all        |      8029 |    -0.000274 |  -0.000354 |   -0.000194 |       0.528584 |          0.000000 |
| seed7   | severity_high_and_long | affected   |      8029 |    -0.000396 |  -0.000529 |   -0.000264 |       0.540042 |          0.000000 |
| seed11  | severity_high_and_long | affected   |      8029 |     0.000370 |   0.000197 |    0.000543 |       0.478142 |          0.000029 |
| seed23  | severity_high_and_long | affected   |      8029 |    -0.000254 |  -0.000408 |   -0.000100 |       0.523104 |          0.001210 |
| seed7   | severity_high_and_long | unaffected |      8011 |    -0.000148 |  -0.000230 |   -0.000065 |       0.503308 |          0.000445 |
| seed11  | severity_high_and_long | unaffected |      8011 |    -0.001115 |  -0.001200 |   -0.001030 |       0.642367 |          0.000000 |
| seed23  | severity_high_and_long | unaffected |      8011 |    -0.000296 |  -0.000383 |   -0.000208 |       0.524404 |          0.000000 |
