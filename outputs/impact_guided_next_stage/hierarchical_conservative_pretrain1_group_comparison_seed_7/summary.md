# Dual-Branch Group Comparison

Base: `dual_branch_sttis_incident_ft_seed_7_normal_veto_hierarchical_conservative_quickgrid`
Candidate: `dual_branch_sttis_incident_ft_seed_7_normal_veto_hierarchical_pretrain1_quickgrid`

Negative delta means the candidate is better.

| group               |   samples |   base_all_mae |   candidate_all_mae |   all_delta |   base_affected_mae |   candidate_affected_mae |   affected_delta |   base_unaffected_mae |   candidate_unaffected_mae |   unaffected_delta |
|:--------------------|----------:|---------------:|--------------------:|------------:|--------------------:|-------------------------:|-----------------:|----------------------:|---------------------------:|-------------------:|
| overall             |     27499 |       0.710019 |            0.709979 |   -0.000040 |            1.107723 |                 1.108502 |         0.000779 |              0.523424 |                   0.523000 |          -0.000424 |
| severity_low        |      9167 |       0.570794 |            0.570200 |   -0.000595 |            0.836410 |                 0.835516 |        -0.000894 |              0.504805 |                   0.504285 |          -0.000520 |
| severity_mid        |      9167 |       0.663549 |            0.663110 |   -0.000439 |            0.961646 |                 0.961268 |        -0.000377 |              0.525956 |                   0.525488 |          -0.000468 |
| severity_high       |      9165 |       0.838357 |            0.839015 |    0.000658 |            1.293627 |                 1.295728 |         0.002101 |              0.536628 |                   0.536328 |          -0.000299 |
| recovery_short_lt30 |     12563 |       0.607662 |            0.607228 |   -0.000434 |            0.938729 |                 0.938485 |        -0.000244 |              0.505963 |                   0.505470 |          -0.000492 |
| recovery_mid_30_90  |      3380 |       0.670904 |            0.670067 |   -0.000837 |            1.029184 |                 1.027698 |        -0.001486 |              0.521299 |                   0.520733 |          -0.000566 |
| recovery_long_ge90  |     11556 |       0.794630 |            0.795079 |    0.000448 |            1.197653 |                 1.199317 |         0.001665 |              0.539971 |                   0.539650 |          -0.000320 |
