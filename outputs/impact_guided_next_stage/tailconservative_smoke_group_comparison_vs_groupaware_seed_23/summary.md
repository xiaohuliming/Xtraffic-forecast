# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_tailconservative_smoke`

Negative delta means the candidate is better.

| group                  |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:-----------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall                |     27499 |       0.707528 |            0.707583 |    0.000055 |            1.102129 |                 1.102532 |         0.000403 |              0.522390 |                   0.522282 |          -0.000108 |
| severity_low           |      9167 |       0.569966 |            0.569478 |   -0.000488 |            0.836268 |                 0.835382 |        -0.000886 |              0.503807 |                   0.503418 |          -0.000389 |
| severity_mid           |      9167 |       0.661907 |            0.661653 |   -0.000254 |            0.960090 |                 0.959781 |        -0.000309 |              0.524273 |                   0.524045 |          -0.000229 |
| severity_high          |      9165 |       0.834081 |            0.834747 |    0.000666 |            1.283541 |                 1.284841 |         0.001299 |              0.536201 |                   0.536447 |           0.000246 |
| recovery_short_lt30    |     12563 |       0.606516 |            0.606144 |   -0.000372 |            0.937863 |                 0.937397 |        -0.000466 |              0.504731 |                   0.504387 |          -0.000343 |
| recovery_mid_30_90     |      3380 |       0.669732 |            0.669258 |   -0.000474 |            1.027625 |                 1.026727 |        -0.000897 |              0.520288 |                   0.519991 |          -0.000297 |
| recovery_long_ge90     |     11556 |       0.790827 |            0.791327 |    0.000500 |            1.189193 |                 1.190230 |         0.001037 |              0.539110 |                   0.539270 |           0.000160 |
| severity_high_and_long |      8029 |       0.837585 |            0.838351 |    0.000766 |            1.270579 |                 1.271999 |         0.001420 |              0.540289 |                   0.540606 |           0.000317 |
