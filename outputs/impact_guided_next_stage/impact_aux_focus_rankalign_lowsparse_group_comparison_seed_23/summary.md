# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_rankalign_lowsparse_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707456 |            0.707426 |   -0.000029 |            1.102596 |                 1.102603 |         0.000008 |              0.522064 |                   0.522018 |          -0.000046 |
| severity_low        |      9167 |       0.569347 |            0.569330 |   -0.000017 |            0.835089 |                 0.835091 |         0.000001 |              0.503326 |                   0.503305 |          -0.000021 |
| severity_mid        |      9167 |       0.661429 |            0.661412 |   -0.000017 |            0.959460 |                 0.959475 |         0.000015 |              0.523866 |                   0.523833 |          -0.000033 |
| severity_high       |      9165 |       0.834704 |            0.834657 |   -0.000047 |            1.285280 |                 1.285284 |         0.000004 |              0.536085 |                   0.536004 |          -0.000081 |
| recovery_short_lt30 |     12563 |       0.606037 |            0.606021 |   -0.000017 |            0.937301 |                 0.937312 |         0.000011 |              0.504278 |                   0.504253 |          -0.000025 |
| recovery_mid_30_90  |      3380 |       0.668967 |            0.668946 |   -0.000022 |            1.026124 |                 1.026093 |        -0.000031 |              0.519831 |                   0.519813 |          -0.000018 |
| recovery_long_ge90  |     11556 |       0.791224 |            0.791184 |   -0.000040 |            1.190491 |                 1.190504 |         0.000013 |              0.538938 |                   0.538863 |          -0.000074 |
