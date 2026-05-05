# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707412 |            0.707352 |   -0.000060 |            1.102588 |                 1.102451 |        -0.000137 |              0.522004 |                   0.521979 |          -0.000024 |
| severity_low        |      9167 |       0.569307 |            0.569307 |    0.000001 |            0.835082 |                 0.835120 |         0.000038 |              0.503278 |                   0.503269 |          -0.000009 |
| severity_mid        |      9167 |       0.661390 |            0.661363 |   -0.000027 |            0.959443 |                 0.959395 |        -0.000048 |              0.523817 |                   0.523800 |          -0.000018 |
| severity_high       |      9165 |       0.834655 |            0.834527 |   -0.000128 |            1.285279 |                 1.285026 |        -0.000253 |              0.536004 |                   0.535960 |          -0.000044 |
| recovery_short_lt30 |     12563 |       0.606006 |            0.605993 |   -0.000013 |            0.937317 |                 0.937299 |        -0.000018 |              0.504232 |                   0.504221 |          -0.000011 |
| recovery_mid_30_90  |      3380 |       0.668880 |            0.668881 |    0.000001 |            1.025976 |                 1.026007 |         0.000031 |              0.519769 |                   0.519757 |          -0.000012 |
| recovery_long_ge90  |     11556 |       0.791183 |            0.791073 |   -0.000110 |            1.190499 |                 1.190278 |        -0.000221 |              0.538865 |                   0.538825 |          -0.000040 |
