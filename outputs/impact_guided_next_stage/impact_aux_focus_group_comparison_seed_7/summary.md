# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_7_normal_veto_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_7_normal_veto_impact_aux_focus_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.710000 |            0.710039 |    0.000038 |            1.108568 |                 1.107855 |        -0.000713 |              0.523000 |                   0.523391 |           0.000391 |
| severity_low        |      9167 |       0.570170 |            0.570740 |    0.000570 |            0.835441 |                 0.836230 |         0.000789 |              0.504266 |                   0.504783 |           0.000516 |
| severity_mid        |      9167 |       0.663133 |            0.663488 |    0.000355 |            0.961310 |                 0.961539 |         0.000229 |              0.525502 |                   0.525916 |           0.000414 |
| severity_high       |      9165 |       0.839066 |            0.838493 |   -0.000573 |            1.295856 |                 1.294022 |        -0.001833 |              0.536329 |                   0.536592 |           0.000263 |
| recovery_short_lt30 |     12563 |       0.607239 |            0.607619 |    0.000381 |            0.938541 |                 0.938625 |         0.000084 |              0.505467 |                   0.505939 |           0.000472 |
| recovery_mid_30_90  |      3380 |       0.670064 |            0.670855 |    0.000791 |            1.027652 |                 1.029136 |         0.001485 |              0.520748 |                   0.521249 |           0.000502 |
| recovery_long_ge90  |     11556 |       0.795113 |            0.794713 |   -0.000400 |            1.199410 |                 1.197924 |        -0.001485 |              0.539648 |                   0.539934 |           0.000285 |
