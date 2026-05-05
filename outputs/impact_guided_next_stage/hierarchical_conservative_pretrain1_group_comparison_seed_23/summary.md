# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain1_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707412 |            0.707390 |   -0.000022 |            1.102588 |                 1.102492 |        -0.000096 |              0.522004 |                   0.522016 |           0.000013 |
| severity_low        |      9167 |       0.569307 |            0.569328 |    0.000021 |            0.835082 |                 0.835120 |         0.000038 |              0.503278 |                   0.503295 |           0.000017 |
| severity_mid        |      9167 |       0.661390 |            0.661396 |    0.000005 |            0.959443 |                 0.959433 |        -0.000010 |              0.523817 |                   0.523830 |           0.000012 |
| severity_high       |      9165 |       0.834655 |            0.834582 |   -0.000073 |            1.285279 |                 1.285082 |        -0.000197 |              0.536004 |                   0.536013 |           0.000010 |
| recovery_short_lt30 |     12563 |       0.606006 |            0.606012 |    0.000006 |            0.937317 |                 0.937305 |        -0.000012 |              0.504232 |                   0.504244 |           0.000012 |
| recovery_mid_30_90  |      3380 |       0.668880 |            0.668932 |    0.000051 |            1.025976 |                 1.026080 |         0.000103 |              0.519769 |                   0.519799 |           0.000030 |
| recovery_long_ge90  |     11556 |       0.791183 |            0.791122 |   -0.000061 |            1.190499 |                 1.190329 |        -0.000171 |              0.538865 |                   0.538873 |           0.000008 |
