# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_11_normal_veto_impact_aux_focus_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_11_normal_veto_hierarchical_conservative_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.708575 |            0.708556 |   -0.000018 |            1.105219 |                 1.105177 |        -0.000041 |              0.522477 |                   0.522470 |          -0.000008 |
| severity_low        |      9167 |       0.569495 |            0.569505 |    0.000010 |            0.834179 |                 0.834233 |         0.000054 |              0.503737 |                   0.503737 |          -0.000001 |
| severity_mid        |      9167 |       0.661415 |            0.661415 |    0.000000 |            0.957863 |                 0.957885 |         0.000021 |              0.524582 |                   0.524572 |          -0.000010 |
| severity_high       |      9165 |       0.837419 |            0.837366 |   -0.000052 |            1.291913 |                 1.291799 |        -0.000114 |              0.536203 |                   0.536192 |          -0.000012 |
| recovery_short_lt30 |     12563 |       0.605883 |            0.605885 |    0.000002 |            0.935898 |                 0.935914 |         0.000016 |              0.504506 |                   0.504504 |          -0.000002 |
| recovery_mid_30_90  |      3380 |       0.668833 |            0.668850 |    0.000017 |            1.025356 |                 1.025438 |         0.000082 |              0.519961 |                   0.519950 |          -0.000010 |
| recovery_long_ge90  |     11556 |       0.793589 |            0.793547 |   -0.000042 |            1.195546 |                 1.195455 |        -0.000090 |              0.539602 |                   0.539590 |          -0.000012 |
