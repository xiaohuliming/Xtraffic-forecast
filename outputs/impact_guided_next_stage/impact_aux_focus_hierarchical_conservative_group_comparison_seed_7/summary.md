# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_7_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_7_normal_veto_hierarchical_conservative_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.710039 |            0.710019 |   -0.000020 |            1.107855 |                 1.107723 |        -0.000133 |              0.523391 |                   0.523424 |           0.000033 |
| severity_low        |      9167 |       0.570740 |            0.570794 |    0.000054 |            0.836230 |                 0.836410 |         0.000180 |              0.504783 |                   0.504805 |           0.000022 |
| severity_mid        |      9167 |       0.663488 |            0.663549 |    0.000061 |            0.961539 |                 0.961646 |         0.000107 |              0.525916 |                   0.525956 |           0.000040 |
| severity_high       |      9165 |       0.838493 |            0.838357 |   -0.000136 |            1.294022 |                 1.293627 |        -0.000395 |              0.536592 |                   0.536628 |           0.000036 |
| recovery_short_lt30 |     12563 |       0.607619 |            0.607662 |    0.000043 |            0.938625 |                 0.938729 |         0.000104 |              0.505939 |                   0.505963 |           0.000024 |
| recovery_mid_30_90  |      3380 |       0.670855 |            0.670904 |    0.000049 |            1.029136 |                 1.029184 |         0.000047 |              0.521249 |                   0.521299 |           0.000049 |
| recovery_long_ge90  |     11556 |       0.794713 |            0.794630 |   -0.000083 |            1.197924 |                 1.197653 |        -0.000272 |              0.539934 |                   0.539971 |           0.000037 |
