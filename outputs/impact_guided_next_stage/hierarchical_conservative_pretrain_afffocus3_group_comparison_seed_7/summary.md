# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_7_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_7_normal_veto_hierarchical_pretrain_afffocus3_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.710019 |            0.709953 |   -0.000066 |            1.107723 |                 1.108420 |         0.000698 |              0.523424 |                   0.523000 |          -0.000425 |
| severity_low        |      9167 |       0.570794 |            0.570203 |   -0.000591 |            0.836410 |                 0.835562 |        -0.000847 |              0.504805 |                   0.504278 |          -0.000527 |
| severity_mid        |      9167 |       0.663549 |            0.663120 |   -0.000429 |            0.961646 |                 0.961307 |        -0.000339 |              0.525956 |                   0.525485 |          -0.000471 |
| severity_high       |      9165 |       0.838357 |            0.838937 |    0.000580 |            1.293627 |                 1.295523 |         0.001896 |              0.536628 |                   0.536335 |          -0.000293 |
| recovery_short_lt30 |     12563 |       0.607662 |            0.607227 |   -0.000435 |            0.938729 |                 0.938496 |        -0.000234 |              0.505963 |                   0.505466 |          -0.000497 |
| recovery_mid_30_90  |      3380 |       0.670904 |            0.670084 |   -0.000820 |            1.029184 |                 1.027764 |        -0.001420 |              0.521299 |                   0.520729 |          -0.000570 |
| recovery_long_ge90  |     11556 |       0.794630 |            0.795022 |    0.000392 |            1.197653 |                 1.199166 |         0.001514 |              0.539971 |                   0.539654 |          -0.000317 |
