# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
Candidate: `dual_branch_sttis_incident_ft_seed_23_final_convexgate_tail_ft_smoke`

Negative delta means the candidate is better.

| group                  |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:-----------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall                |     27499 |       0.707528 |            0.709091 |    0.001563 |            1.102129 |                 1.103074 |         0.000945 |              0.522390 |                   0.524242 |           0.001852 |
| severity_low           |      9167 |       0.569966 |            0.570853 |    0.000886 |            0.836268 |                 0.834959 |        -0.001309 |              0.503807 |                   0.505239 |           0.001432 |
| severity_mid           |      9167 |       0.661907 |            0.662553 |    0.000646 |            0.960090 |                 0.958620 |        -0.001470 |              0.524273 |                   0.525896 |           0.001623 |
| severity_high          |      9165 |       0.834081 |            0.836865 |    0.002784 |            1.283541 |                 1.286856 |         0.003314 |              0.536201 |                   0.538634 |           0.002433 |
| recovery_short_lt30    |     12563 |       0.606516 |            0.607266 |    0.000750 |            0.937863 |                 0.936603 |        -0.001260 |              0.504731 |                   0.506098 |           0.001368 |
| recovery_mid_30_90     |      3380 |       0.669732 |            0.669973 |    0.000241 |            1.027625 |                 1.024863 |        -0.002761 |              0.520288 |                   0.521782 |           0.001494 |
| recovery_long_ge90     |     11556 |       0.790827 |            0.793315 |    0.002488 |            1.189193 |                 1.191823 |         0.002630 |              0.539110 |                   0.541508 |           0.002398 |
| severity_high_and_long |      8029 |       0.837585 |            0.840662 |    0.003077 |            1.270579 |                 1.274284 |         0.003704 |              0.540289 |                   0.542935 |           0.002646 |
