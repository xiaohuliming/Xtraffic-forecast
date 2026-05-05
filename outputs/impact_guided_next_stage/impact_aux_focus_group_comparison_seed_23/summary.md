# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707490 |            0.707456 |   -0.000035 |            1.102705 |                 1.102596 |        -0.000109 |              0.522064 |                   0.522064 |           0.000000 |
| severity_low        |      9167 |       0.569328 |            0.569347 |    0.000019 |            0.834985 |                 0.835089 |         0.000104 |              0.503328 |                   0.503326 |          -0.000002 |
| severity_mid        |      9167 |       0.661401 |            0.661429 |    0.000027 |            0.959358 |                 0.959460 |         0.000102 |              0.523873 |                   0.523866 |          -0.000007 |
| severity_high       |      9165 |       0.834827 |            0.834704 |   -0.000123 |            1.285602 |                 1.285280 |        -0.000322 |              0.536076 |                   0.536085 |           0.000009 |
| recovery_short_lt30 |     12563 |       0.606031 |            0.606037 |    0.000006 |            0.937252 |                 0.937301 |         0.000049 |              0.504284 |                   0.504278 |          -0.000007 |
| recovery_mid_30_90  |      3380 |       0.668898 |            0.668967 |    0.000069 |            1.025867 |                 1.026124 |         0.000257 |              0.519840 |                   0.519831 |          -0.000009 |
| recovery_long_ge90  |     11556 |       0.791315 |            0.791224 |   -0.000091 |            1.190740 |                 1.190491 |        -0.000249 |              0.538928 |                   0.538938 |           0.000010 |
