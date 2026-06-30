# Latent Impact Correction Adapter — 3-Seed Mean

负数表示 adapter 优于 source（同 seed 比较）。Source 为 ST-TIS gate-head fine-tune no-aux（§5.4 主候选）。

| Variant | Overall all | Overall affected | Overall unaffected | High_and_long all | High_and_long affected | High_and_long unaffected |
|---|---:|---:|---:|---:|---:|---:|
| anomgate05 (基线 adapter，无 regret) | -0.000441 | -0.000484 | -0.000421 | -0.000369 | -0.000163 | -0.000510 |
| **anomgate05 + magnitude regret w=0.05 (主候选)** | **-0.000418** | **-0.000465** | **-0.000396** | **-0.000376** | **-0.000211** | **-0.000489** |
| anomgate05 + magnitude regret w=0.10 | -0.000374 | -0.000417 | -0.000353 | -0.000354 | -0.000222 | -0.000445 |
| anomgate05 + top-k harmful regret w=0.10 | -0.000302 | -0.000327 | -0.000290 | -0.000294 | -0.000173 | -0.000378 |
