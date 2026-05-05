# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_conservative_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707456 |            0.707412 |   -0.000043 |            1.102596 |                 1.102588 |        -0.000008 |              0.522064 |                   0.522004 |          -0.000060 |
| severity_low        |      9167 |       0.569347 |            0.569307 |   -0.000040 |            0.835089 |                 0.835082 |        -0.000007 |              0.503326 |                   0.503278 |          -0.000048 |
| severity_mid        |      9167 |       0.661429 |            0.661390 |   -0.000038 |            0.959460 |                 0.959443 |        -0.000017 |              0.523866 |                   0.523817 |          -0.000048 |
| severity_high       |      9165 |       0.834704 |            0.834655 |   -0.000050 |            1.285280 |                 1.285279 |        -0.000001 |              0.536085 |                   0.536004 |          -0.000082 |
| recovery_short_lt30 |     12563 |       0.606037 |            0.606006 |   -0.000031 |            0.937301 |                 0.937317 |         0.000016 |              0.504278 |                   0.504232 |          -0.000046 |
| recovery_mid_30_90  |      3380 |       0.668967 |            0.668880 |   -0.000087 |            1.026124 |                 1.025976 |        -0.000148 |              0.519831 |                   0.519769 |          -0.000062 |
| recovery_long_ge90  |     11556 |       0.791224 |            0.791183 |   -0.000041 |            1.190491 |                 1.190499 |         0.000009 |              0.538938 |                   0.538865 |          -0.000073 |
