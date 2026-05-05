# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_11_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_11_normal_veto_hierarchical_pretrain1_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.708556 |            0.708587 |    0.000030 |            1.105177 |                 1.105231 |         0.000054 |              0.522470 |                   0.522489 |           0.000019 |
| severity_low        |      9167 |       0.569505 |            0.569527 |    0.000022 |            0.834233 |                 0.834262 |         0.000029 |              0.503737 |                   0.503757 |           0.000021 |
| severity_mid        |      9167 |       0.661415 |            0.661430 |    0.000015 |            0.957885 |                 0.957894 |         0.000010 |              0.524572 |                   0.524589 |           0.000017 |
| severity_high       |      9165 |       0.837366 |            0.837415 |    0.000049 |            1.291799 |                 1.291890 |         0.000092 |              0.536192 |                   0.536212 |           0.000020 |
| recovery_short_lt30 |     12563 |       0.605885 |            0.605904 |    0.000020 |            0.935914 |                 0.935961 |         0.000047 |              0.504504 |                   0.504515 |           0.000011 |
| recovery_mid_30_90  |      3380 |       0.668850 |            0.668870 |    0.000021 |            1.025438 |                 1.025424 |        -0.000014 |              0.519950 |                   0.519985 |           0.000035 |
| recovery_long_ge90  |     11556 |       0.793547 |            0.793587 |    0.000040 |            1.195455 |                 1.195525 |         0.000069 |              0.539590 |                   0.539612 |           0.000022 |
