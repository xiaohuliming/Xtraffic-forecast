# Impact Correction Paired Sample Stats

Delta is adapter sample MAE minus source sample MAE; negative is better.
The confidence interval is a normal approximation over event-level sample deltas, so it is supporting evidence rather than a full seed-level significance test.

## Seed-Mean Focus

| group                  | target     |   seeds |   seed_mean_delta |   seed_std_delta | all_seeds_improved   |   mean_improve_rate |
|:-----------------------|:-----------|--------:|------------------:|-----------------:|:---------------------|--------------------:|
| overall                | all        |       3 |         -0.000199 |         0.000087 | True                 |            0.599295 |
| overall                | affected   |       3 |         -0.000238 |         0.000182 | True                 |            0.558647 |
| overall                | unaffected |       3 |         -0.000182 |         0.000074 | True                 |            0.591209 |
| severity_high          | all        |       3 |         -0.000152 |         0.000041 | True                 |            0.581597 |
| severity_high          | affected   |       3 |         -0.000142 |         0.000132 | True                 |            0.538601 |
| severity_high          | unaffected |       3 |         -0.000165 |         0.000103 | True                 |            0.588995 |
| recovery_long_ge90     | all        |       3 |         -0.000175 |         0.000059 | True                 |            0.589448 |
| recovery_long_ge90     | affected   |       3 |         -0.000168 |         0.000139 | True                 |            0.545447 |
| recovery_long_ge90     | unaffected |       3 |         -0.000173 |         0.000102 | True                 |            0.590640 |
| severity_high_and_long | all        |       3 |         -0.000136 |         0.000036 | True                 |            0.574916 |
| severity_high_and_long | affected   |       3 |         -0.000120 |         0.000115 | True                 |            0.533856 |
| severity_high_and_long | unaffected |       3 |         -0.000150 |         0.000121 | True                 |            0.584654 |

## Per-Seed Focus

| label   | group                  | target     |   samples |   mean_delta |   ci95_low |   ci95_high |   improve_rate |   p_normal_approx |
|:--------|:-----------------------|:-----------|----------:|-------------:|-----------:|------------:|---------------:|------------------:|
| seed7   | overall                | all        |     27499 |    -0.000107 |  -0.000113 |   -0.000100 |       0.585767 |          0.000000 |
| seed11  | overall                | all        |     27499 |    -0.000208 |  -0.000215 |   -0.000201 |       0.661733 |          0.000000 |
| seed23  | overall                | all        |     27499 |    -0.000281 |  -0.000311 |   -0.000251 |       0.550384 |          0.000000 |
| seed7   | overall                | affected   |     26205 |    -0.000125 |  -0.000136 |   -0.000113 |       0.569357 |          0.000000 |
| seed11  | overall                | affected   |     26205 |    -0.000141 |  -0.000155 |   -0.000127 |       0.559015 |          0.000000 |
| seed23  | overall                | affected   |     26205 |    -0.000447 |  -0.000503 |   -0.000391 |       0.547567 |          0.000000 |
| seed7   | overall                | unaffected |     27393 |    -0.000103 |  -0.000111 |   -0.000096 |       0.565582 |          0.000000 |
| seed11  | overall                | unaffected |     27393 |    -0.000250 |  -0.000258 |   -0.000242 |       0.680575 |          0.000000 |
| seed23  | overall                | unaffected |     27393 |    -0.000193 |  -0.000226 |   -0.000159 |       0.527471 |          0.000000 |
| seed7   | severity_high          | all        |      9165 |    -0.000105 |  -0.000116 |   -0.000094 |       0.585597 |          0.000000 |
| seed11  | severity_high          | all        |      9165 |    -0.000175 |  -0.000187 |   -0.000162 |       0.633715 |          0.000000 |
| seed23  | severity_high          | all        |      9165 |    -0.000178 |  -0.000240 |   -0.000115 |       0.525477 |          0.000000 |
| seed7   | severity_high          | affected   |      9162 |    -0.000116 |  -0.000135 |   -0.000098 |       0.574984 |          0.000000 |
| seed11  | severity_high          | affected   |      9162 |    -0.000025 |  -0.000048 |   -0.000003 |       0.513971 |          0.029572 |
| seed23  | severity_high          | affected   |      9162 |    -0.000285 |  -0.000394 |   -0.000176 |       0.526850 |          0.000000 |
| seed7   | severity_high          | unaffected |      9141 |    -0.000102 |  -0.000116 |   -0.000089 |       0.560989 |          0.000000 |
| seed11  | severity_high          | unaffected |      9141 |    -0.000284 |  -0.000297 |   -0.000271 |       0.697517 |          0.000000 |
| seed23  | severity_high          | unaffected |      9141 |    -0.000109 |  -0.000181 |   -0.000036 |       0.508478 |          0.003384 |
| seed7   | recovery_long_ge90     | all        |     11556 |    -0.000107 |  -0.000117 |   -0.000098 |       0.585324 |          0.000000 |
| seed11  | recovery_long_ge90     | all        |     11556 |    -0.000203 |  -0.000214 |   -0.000192 |       0.651177 |          0.000000 |
| seed23  | recovery_long_ge90     | all        |     11556 |    -0.000216 |  -0.000269 |   -0.000163 |       0.531845 |          0.000000 |
| seed7   | recovery_long_ge90     | affected   |     11541 |    -0.000110 |  -0.000126 |   -0.000093 |       0.571008 |          0.000000 |
| seed11  | recovery_long_ge90     | affected   |     11541 |    -0.000068 |  -0.000088 |   -0.000047 |       0.533143 |          0.000000 |
| seed23  | recovery_long_ge90     | affected   |     11541 |    -0.000327 |  -0.000418 |   -0.000235 |       0.532190 |          0.000000 |
| seed7   | recovery_long_ge90     | unaffected |     11531 |    -0.000107 |  -0.000119 |   -0.000095 |       0.565866 |          0.000000 |
| seed11  | recovery_long_ge90     | unaffected |     11531 |    -0.000290 |  -0.000302 |   -0.000279 |       0.698465 |          0.000000 |
| seed23  | recovery_long_ge90     | unaffected |     11531 |    -0.000121 |  -0.000183 |   -0.000059 |       0.507588 |          0.000130 |
| seed7   | severity_high_and_long | all        |      8029 |    -0.000100 |  -0.000112 |   -0.000088 |       0.578403 |          0.000000 |
| seed11  | severity_high_and_long | all        |      8029 |    -0.000172 |  -0.000186 |   -0.000159 |       0.628472 |          0.000000 |
| seed23  | severity_high_and_long | all        |      8029 |    -0.000135 |  -0.000202 |   -0.000068 |       0.517873 |          0.000087 |
| seed7   | severity_high_and_long | affected   |      8029 |    -0.000107 |  -0.000127 |   -0.000088 |       0.571179 |          0.000000 |
| seed11  | severity_high_and_long | affected   |      8029 |    -0.000011 |  -0.000035 |    0.000013 |       0.509403 |          0.357426 |
| seed23  | severity_high_and_long | affected   |      8029 |    -0.000241 |  -0.000356 |   -0.000125 |       0.520986 |          0.000044 |
| seed7   | severity_high_and_long | unaffected |      8011 |    -0.000099 |  -0.000114 |   -0.000084 |       0.556235 |          0.000000 |
| seed11  | severity_high_and_long | unaffected |      8011 |    -0.000288 |  -0.000302 |   -0.000273 |       0.696417 |          0.000000 |
| seed23  | severity_high_and_long | unaffected |      8011 |    -0.000062 |  -0.000142 |    0.000017 |       0.501311 |          0.123513 |
