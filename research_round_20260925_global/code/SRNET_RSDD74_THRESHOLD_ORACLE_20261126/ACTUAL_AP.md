# SRNet reserve mask AP by IoU threshold

The audit reused the exact frozen SRNet N/F/I candidates and original author-reference scorer on the 74-chip reserve and unchanged nested 42-chip view. It reproduced all original aggregate COCO and continuous geometry results. An independent reconstruction from stable descending scores, TP/FP status, ignore flags and recall interpolation reproduced all 101 precision-envelope entries per arm and threshold. [Registered design](PRELOCK.md) · [Original comparison](../SRNET_SSDD_E24_RSDD74_PAIRED_REPLAY_20261126/ACTUAL_REPLAY.md)

| IoU threshold | 74-chip native AP | 74-chip fixed AP | 74-chip independent AP | 74-chip I−F | Nested 42 I−F |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 78.503 | 78.503 | 78.503 | 0.000 | 0.000 |
| 0.55 | 76.184 | 76.184 | 76.184 | 0.000 | 0.000 |
| 0.60 | 73.828 | 73.828 | 73.828 | 0.000 | 0.000 |
| 0.65 | 53.472 | 53.472 | 53.472 | 0.000 | 0.000 |
| 0.70 | 33.368 | 33.368 | 34.995 | 1.627 | 2.179 |
| 0.75 | 12.480 | 12.480 | 12.480 | 0.000 | 0.000 |
| 0.80 | 4.317 | 4.492 | 4.317 | -0.176 | 0.000 |
| 0.85 | 0.438 | 0.438 | 0.438 | 0.000 | 0.000 |
| 0.90 | 0.064 | 0.064 | 0.064 | 0.000 | 0.000 |
| 0.95 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

On all 74 chips, I−F is +0.145 mask AP points. At .70, one original candidate changes from false positive to true positive, and one fixed native pair crosses the raster threshold upward. AP at .70 rises by 1.627 points. At .80, one candidate gains true-positive status and one loses it. The total number of true positives stays constant, while the ranked precision envelope gives a 0.176-point decrease. The fixed-pair raster IoU shows one upward and one downward crossing at .80. The other eight AP thresholds are unchanged. Averaging all ten threshold differences reproduces the pooled result.

On the nested 42-chip view, one false positive becomes a true positive at .70, and the 2.179-point increase at that threshold gives the full +0.218 mask AP change. Retained evaluator true positives preserve their reference assignment in both I−F views. At .90, one eligible original fixed pair crosses the continuous IoU threshold in each view. Raster AP at .90 is unchanged. The threshold crossing counts describe observed events. AP also depends on the ordering of matches and on precision interpolation. [Full threshold and transition tables](AP_AGGREGATE_SAFE.json) · [Observed score closure](AP_TERMINAL.json)

The public result lists cohort-level counters. The original observer receipt binds a private result whose auxiliary per-arm status field uses candidate image keys. The release projection derives TP, FP and ignore totals from the verified transition matrices, checking the row and column marginals across all three comparisons. It retains per-threshold AP, aggregate transitions and aggregate crossings. [Aggregate projection implementation](project_ap_release.py)
