# Impact Correction Alignment Diagnostics

- adapter_dir: `/Users/xhlm/Desktop/Study/科研实习/outputs/impact_guided_next_stage/groupaware_impact_correction_adapter_highfocus_anomgate05_seed_11`
- split: `test`

| group                  | target     |   target_abs_mean |   correction_abs_mean |   mean_improvement |   sign_match_rate |   beneficial_rate |   harmful_rate |
|:-----------------------|:-----------|------------------:|----------------------:|-------------------:|------------------:|------------------:|---------------:|
| overall                | all        |          0.709210 |              0.011572 |           0.000689 |          0.515584 |          0.512184 |       0.487813 |
| overall                | affected   |          1.104695 |              0.018323 |           0.000518 |          0.513683 |          0.509789 |       0.490208 |
| overall                | unaffected |          0.523656 |              0.008404 |           0.000769 |          0.516476 |          0.513308 |       0.486689 |
| severity_high          | all        |          0.837098 |              0.017139 |           0.000603 |          0.514593 |          0.509926 |       0.490071 |
| severity_high          | affected   |          1.289442 |              0.024724 |          -0.000083 |          0.508788 |          0.503765 |       0.496233 |
| severity_high          | unaffected |          0.537308 |              0.012113 |           0.001058 |          0.518441 |          0.514010 |       0.485987 |
| recovery_long_ge90     | all        |          0.793633 |              0.016296 |           0.000732 |          0.516366 |          0.511842 |       0.488156 |
| recovery_long_ge90     | affected   |          1.193803 |              0.023578 |           0.000195 |          0.512108 |          0.507194 |       0.492804 |
| recovery_long_ge90     | unaffected |          0.540775 |              0.011695 |           0.001071 |          0.519057 |          0.514779 |       0.485219 |
| severity_high_and_long | all        |          0.841093 |              0.018093 |           0.000588 |          0.514616 |          0.509710 |       0.490288 |
| severity_high_and_long | affected   |          1.277056 |              0.025799 |          -0.000176 |          0.508697 |          0.503440 |       0.496558 |
| severity_high_and_long | unaffected |          0.541759 |              0.012802 |           0.001112 |          0.518680 |          0.514014 |       0.485983 |
