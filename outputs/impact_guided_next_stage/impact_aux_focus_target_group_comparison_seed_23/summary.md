# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_target_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707490 |            0.707345 |   -0.000145 |            1.102705 |                 1.102536 |        -0.000169 |              0.522064 |                   0.521930 |          -0.000134 |
| severity_low        |      9167 |       0.569328 |            0.569298 |   -0.000030 |            0.834985 |                 0.835076 |         0.000091 |              0.503328 |                   0.503269 |          -0.000060 |
| severity_mid        |      9167 |       0.661401 |            0.661335 |   -0.000066 |            0.959358 |                 0.959401 |         0.000043 |              0.523873 |                   0.523756 |          -0.000117 |
| severity_high       |      9165 |       0.834827 |            0.834540 |   -0.000287 |            1.285602 |                 1.285204 |        -0.000398 |              0.536076 |                   0.535862 |          -0.000214 |
| recovery_short_lt30 |     12563 |       0.606031 |            0.605994 |   -0.000037 |            0.937252 |                 0.937319 |         0.000067 |              0.504284 |                   0.504216 |          -0.000069 |
| recovery_mid_30_90  |      3380 |       0.668898 |            0.668828 |   -0.000071 |            1.025867 |                 1.025950 |         0.000083 |              0.519840 |                   0.519705 |          -0.000135 |
| recovery_long_ge90  |     11556 |       0.791315 |            0.791072 |   -0.000243 |            1.190740 |                 1.190418 |        -0.000322 |              0.538928 |                   0.538735 |          -0.000194 |
