# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_rankalign_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707456 |            0.707439 |   -0.000017 |            1.102596 |                 1.102614 |         0.000018 |              0.522064 |                   0.522030 |          -0.000034 |
| severity_low        |      9167 |       0.569347 |            0.569340 |   -0.000007 |            0.835089 |                 0.835092 |         0.000002 |              0.503326 |                   0.503317 |          -0.000009 |
| severity_mid        |      9167 |       0.661429 |            0.661422 |   -0.000007 |            0.959460 |                 0.959481 |         0.000021 |              0.523866 |                   0.523846 |          -0.000019 |
| severity_high       |      9165 |       0.834704 |            0.834672 |   -0.000032 |            1.285280 |                 1.285301 |         0.000021 |              0.536085 |                   0.536018 |          -0.000068 |
| recovery_short_lt30 |     12563 |       0.606037 |            0.606032 |   -0.000006 |            0.937301 |                 0.937321 |         0.000021 |              0.504278 |                   0.504264 |          -0.000014 |
| recovery_mid_30_90  |      3380 |       0.668967 |            0.668959 |   -0.000009 |            1.026124 |                 1.026098 |        -0.000026 |              0.519831 |                   0.519830 |          -0.000001 |
| recovery_long_ge90  |     11556 |       0.791224 |            0.791197 |   -0.000027 |            1.190491 |                 1.190516 |         0.000026 |              0.538938 |                   0.538877 |          -0.000061 |
