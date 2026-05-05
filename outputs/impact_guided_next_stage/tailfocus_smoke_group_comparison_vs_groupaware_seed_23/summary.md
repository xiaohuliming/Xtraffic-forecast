# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_tailfocus_smoke`

Negative delta means the candidate is better.

| group                  |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:-----------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall                |     27499 |       0.707528 |            0.707309 |   -0.000219 |            1.102129 |                 1.102510 |         0.000381 |              0.522390 |                   0.521889 |          -0.000501 |
| severity_low           |      9167 |       0.569966 |            0.569205 |   -0.000762 |            0.836268 |                 0.835146 |        -0.001122 |              0.503807 |                   0.503135 |          -0.000672 |
| severity_mid           |      9167 |       0.661907 |            0.661323 |   -0.000583 |            0.960090 |                 0.959530 |        -0.000560 |              0.524273 |                   0.523679 |          -0.000594 |
| severity_high          |      9165 |       0.834081 |            0.834520 |    0.000439 |            1.283541 |                 1.285044 |         0.001502 |              0.536201 |                   0.535936 |          -0.000265 |
| recovery_short_lt30    |     12563 |       0.606516 |            0.605907 |   -0.000610 |            0.937863 |                 0.937363 |        -0.000500 |              0.504731 |                   0.504087 |          -0.000643 |
| recovery_mid_30_90     |      3380 |       0.669732 |            0.668852 |   -0.000880 |            1.027625 |                 1.026231 |        -0.001394 |              0.520288 |                   0.519623 |          -0.000665 |
| recovery_long_ge90     |     11556 |       0.790827 |            0.791059 |    0.000232 |            1.189193 |                 1.190304 |         0.001111 |              0.539110 |                   0.538786 |          -0.000324 |
| severity_high_and_long |      8029 |       0.837585 |            0.838115 |    0.000530 |            1.270579 |                 1.272196 |         0.001617 |              0.540289 |                   0.540073 |          -0.000216 |
