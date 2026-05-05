# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_23_normal_veto_hierarchical_pretrain_afffocus3_groupaware`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.707412 |            0.707528 |    0.000116 |            1.102588 |                 1.102129 |        -0.000459 |              0.522004 |                   0.522390 |           0.000386 |
| severity_low        |      9167 |       0.569307 |            0.569966 |    0.000660 |            0.835082 |                 0.836268 |         0.001186 |              0.503278 |                   0.503807 |           0.000529 |
| severity_mid        |      9167 |       0.661390 |            0.661907 |    0.000516 |            0.959443 |                 0.960090 |         0.000647 |              0.523817 |                   0.524273 |           0.000456 |
| severity_high       |      9165 |       0.834655 |            0.834081 |   -0.000574 |            1.285279 |                 1.283541 |        -0.001738 |              0.536004 |                   0.536201 |           0.000197 |
| recovery_short_lt30 |     12563 |       0.606006 |            0.606516 |    0.000510 |            0.937317 |                 0.937863 |         0.000546 |              0.504232 |                   0.504731 |           0.000499 |
| recovery_mid_30_90  |      3380 |       0.668880 |            0.669732 |    0.000852 |            1.025976 |                 1.027625 |         0.001648 |              0.519769 |                   0.520288 |           0.000520 |
| recovery_long_ge90  |     11556 |       0.791183 |            0.790827 |   -0.000356 |            1.190499 |                 1.189193 |        -0.001306 |              0.538865 |                   0.539110 |           0.000245 |
