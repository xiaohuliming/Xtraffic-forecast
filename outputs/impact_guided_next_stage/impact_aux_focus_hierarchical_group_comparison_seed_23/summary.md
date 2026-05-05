# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_impact_aux_focus_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707456 |            0.707427 |   -0.000028 |            1.102596 |                 1.102612 |         0.000017 |              0.522064 |                   0.522015 |          -0.000049 |
| severity_low        |      9167 |       0.569347 |            0.569312 |   -0.000034 |            0.835089 |                 0.835077 |        -0.000012 |              0.503326 |                   0.503286 |          -0.000040 |
| severity_mid        |      9167 |       0.661429 |            0.661395 |   -0.000034 |            0.959460 |                 0.959449 |        -0.000011 |              0.523866 |                   0.523822 |          -0.000044 |
| severity_high       |      9165 |       0.834704 |            0.834685 |   -0.000019 |            1.285280 |                 1.285324 |         0.000044 |              0.536085 |                   0.536024 |          -0.000061 |
| recovery_short_lt30 |     12563 |       0.606037 |            0.606011 |   -0.000026 |            0.937301 |                 0.937321 |         0.000020 |              0.504278 |                   0.504237 |          -0.000041 |
| recovery_mid_30_90  |      3380 |       0.668967 |            0.668896 |   -0.000072 |            1.026124 |                 1.025987 |        -0.000137 |              0.519831 |                   0.519787 |          -0.000044 |
| recovery_long_ge90  |     11556 |       0.791224 |            0.791206 |   -0.000019 |            1.190491 |                 1.190535 |         0.000044 |              0.538938 |                   0.538879 |          -0.000058 |
