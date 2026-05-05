# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_11_normal_veto_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_11_normal_veto_impact_aux_focus_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.708593 |            0.708575 |   -0.000018 |            1.105411 |                 1.105219 |        -0.000192 |              0.522414 |                   0.522477 |           0.000063 |
| severity_low        |      9167 |       0.569424 |            0.569495 |    0.000071 |            0.834037 |                 0.834179 |         0.000142 |              0.503684 |                   0.503737 |           0.000053 |
| severity_mid        |      9167 |       0.661346 |            0.661415 |    0.000068 |            0.957773 |                 0.957863 |         0.000090 |              0.524524 |                   0.524582 |           0.000058 |
| severity_high       |      9165 |       0.837569 |            0.837419 |   -0.000150 |            1.292405 |                 1.291913 |        -0.000492 |              0.536127 |                   0.536203 |           0.000077 |
| recovery_short_lt30 |     12563 |       0.605851 |            0.605883 |    0.000032 |            0.935919 |                 0.935898 |        -0.000021 |              0.504458 |                   0.504506 |           0.000048 |
| recovery_mid_30_90  |      3380 |       0.668679 |            0.668833 |    0.000153 |            1.025004 |                 1.025356 |         0.000352 |              0.519890 |                   0.519961 |           0.000070 |
| recovery_long_ge90  |     11556 |       0.793687 |            0.793589 |   -0.000098 |            1.195918 |                 1.195546 |        -0.000372 |              0.539527 |                   0.539602 |           0.000075 |
