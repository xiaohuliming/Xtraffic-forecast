# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707490 |            0.707466 |   -0.000025 |            1.102705 |                 1.102669 |        -0.000036 |              0.522064 |                   0.522044 |          -0.000019 |
| severity_low        |      9167 |       0.569328 |            0.569319 |   -0.000008 |            0.834985 |                 0.835004 |         0.000019 |              0.503328 |                   0.503313 |          -0.000015 |
| severity_mid        |      9167 |       0.661401 |            0.661395 |   -0.000006 |            0.959358 |                 0.959385 |         0.000027 |              0.523873 |                   0.523851 |          -0.000022 |
| severity_high       |      9165 |       0.834827 |            0.834776 |   -0.000051 |            1.285602 |                 1.285506 |        -0.000096 |              0.536076 |                   0.536056 |          -0.000021 |
| recovery_short_lt30 |     12563 |       0.606031 |            0.606019 |   -0.000012 |            0.937252 |                 0.937264 |         0.000012 |              0.504284 |                   0.504265 |          -0.000020 |
| recovery_mid_30_90  |      3380 |       0.668898 |            0.668907 |    0.000008 |            1.025867 |                 1.025958 |         0.000091 |              0.519840 |                   0.519814 |          -0.000026 |
| recovery_long_ge90  |     11556 |       0.791315 |            0.791273 |   -0.000042 |            1.190740 |                 1.190659 |        -0.000081 |              0.538928 |                   0.538911 |          -0.000017 |
