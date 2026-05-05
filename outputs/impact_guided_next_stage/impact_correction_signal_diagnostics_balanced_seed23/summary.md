# Impact Correction Signal Diagnostics

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_tailtarget_smoke_seed_23`

Signals are node-level averages over forecast horizon and channels where applicable.

| group                  | node_type   | signal           |   count |     mean |      q10 |      q25 |      q50 |      q75 |      q90 |
|:-----------------------|:------------|:-----------------|--------:|---------:|---------:|---------:|---------:|---------:|---------:|
| overall                | affected    | node_prob        |  138418 | 0.372786 | 0.174572 | 0.243401 | 0.340943 | 0.473354 | 0.626324 |
| overall                | affected    | branch_delta_abs |  138418 | 0.945517 | 0.229631 | 0.364630 | 0.613098 | 1.063077 | 1.959843 |
| overall                | affected    | correction_abs   |  138418 | 0.014124 | 0.003533 | 0.005626 | 0.010070 | 0.018599 | 0.030090 |
| overall                | unaffected  | node_prob        |  415659 | 0.186302 | 0.052942 | 0.072538 | 0.172411 | 0.261444 | 0.348831 |
| overall                | unaffected  | branch_delta_abs |  415659 | 0.436801 | 0.077470 | 0.138836 | 0.310492 | 0.574972 | 0.921859 |
| overall                | unaffected  | correction_abs   |  415659 | 0.009042 | 0.002672 | 0.004094 | 0.006275 | 0.010928 | 0.019101 |
| severity_high          | affected    | node_prob        |   68874 | 0.407830 | 0.190641 | 0.264412 | 0.372300 | 0.524038 | 0.699172 |
| severity_high          | affected    | branch_delta_abs |   68874 | 1.099906 | 0.247435 | 0.394346 | 0.669827 | 1.224425 | 2.420366 |
| severity_high          | affected    | correction_abs   |   68874 | 0.017156 | 0.004353 | 0.007237 | 0.013196 | 0.022834 | 0.035081 |
| severity_high          | unaffected  | node_prob        |  146481 | 0.195279 | 0.054610 | 0.075276 | 0.181965 | 0.274175 | 0.364729 |
| severity_high          | unaffected  | branch_delta_abs |  146481 | 0.461538 | 0.076954 | 0.143114 | 0.329728 | 0.605617 | 0.971502 |
| severity_high          | unaffected  | correction_abs   |  146481 | 0.011297 | 0.003062 | 0.004837 | 0.008158 | 0.014722 | 0.023810 |
| recovery_long_ge90     | affected    | node_prob        |   84403 | 0.402902 | 0.193166 | 0.265675 | 0.369770 | 0.513726 | 0.678030 |
| recovery_long_ge90     | affected    | branch_delta_abs |   84403 | 1.056097 | 0.243841 | 0.388944 | 0.660283 | 1.186178 | 2.274450 |
| recovery_long_ge90     | affected    | correction_abs   |   84403 | 0.016634 | 0.004285 | 0.007057 | 0.012781 | 0.022079 | 0.034061 |
| recovery_long_ge90     | unaffected  | node_prob        |  188349 | 0.196433 | 0.054733 | 0.075872 | 0.183975 | 0.275534 | 0.365643 |
| recovery_long_ge90     | unaffected  | branch_delta_abs |  188349 | 0.461890 | 0.077767 | 0.144341 | 0.330691 | 0.607899 | 0.974341 |
| recovery_long_ge90     | unaffected  | correction_abs   |  188349 | 0.011146 | 0.003068 | 0.004840 | 0.008075 | 0.014450 | 0.023428 |
| severity_high_and_long | affected    | node_prob        |   62902 | 0.414974 | 0.196809 | 0.270887 | 0.379895 | 0.532914 | 0.706786 |
| severity_high_and_long | affected    | branch_delta_abs |   62902 | 1.118156 | 0.249936 | 0.398183 | 0.677037 | 1.246674 | 2.480427 |
| severity_high_and_long | affected    | correction_abs   |   62902 | 0.017658 | 0.004579 | 0.007653 | 0.013803 | 0.023481 | 0.035738 |
| severity_high_and_long | unaffected  | node_prob        |  129151 | 0.197893 | 0.055078 | 0.076217 | 0.185220 | 0.277692 | 0.368741 |
| severity_high_and_long | unaffected  | branch_delta_abs |  129151 | 0.466553 | 0.077481 | 0.143919 | 0.332884 | 0.612070 | 0.982712 |
| severity_high_and_long | unaffected  | correction_abs   |  129151 | 0.011711 | 0.003211 | 0.005063 | 0.008616 | 0.015340 | 0.024481 |
