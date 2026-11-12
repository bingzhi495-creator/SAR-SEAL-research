# D-FINE FIT109 source-calibrated guard for strict original matches

A whole-candidate swap guard was tested on the same accepted, detector-training-visible HRSID FIT109 capture and immutable point pairs. For each outer source fold, the independent rendered-intensity point head trained for 200 AdamW steps using the three folds outside the outer and next calibration fold. Its natural strictly positive swap set was retained only when the predicted mean gain exceeded the clipped 90% source-maximum residual order statistic. The 1,000 point-optimizer steps used 64 sampled pairs per step. There were no detector forwards or updates. All 10,900 masks retained their original ranks, scores and areas.

| Arm | Mask AP | AP75 | AP90 | AR100 | Source-equal continuous IoU | Fixed strict .90 |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 49.3251 | 47.3883 | 1.5337 | 53.1250 | 80.0492 | 7 |
| Nested full head, unguarded | 49.9528 | 48.4738 | 1.5501 | 53.6290 | 80.4985 | 6 |
| Same head, guarded | 49.3995 | 47.3883 | 1.5337 | 53.1653 | 80.0604 | 7 |
| Same-pair truth-only geometry oracle | | | | | 81.1885 | 11 |

The unguarded head edited 4,406 candidates with 19,837 swaps. It retained five original strict matches after edits, lost two and made one newly strict. The guard retained 34 candidate edits (0.77% of the unguarded edits) and 254 swaps (1.28%). All were in one outer fold. There were two fixed-valid edited candidates, both with positive actual mean gain, and 32 unmatched edits. The seven raw strict .90 matches all remained untouched. There were no newly strict matches.

| Outer fold | Calibration sources / eligible | Residual cutoff | Outer source exceedances / sources | Kept candidate edits |
|---|---:|---:|---:|---:|
| 0 | 22 / 20 | 0.7670 | 0 / 22 | 0 |
| 1 | 22 / 19 | 0.3276 | 6 / 22 | 34 |
| 2 | 22 / 19 | 1.0582 | 0 / 22 | 0 |
| 3 | 21 / 18 | 0.6578 | 2 / 22 | 0 |
| 4 | 22 / 17 | 0.5699 | 3 / 21 | 0 |

The observed exceedance count was 11/109 outer sources or 11/93 sources with eligible actions. Guarded versus raw gained +0.0744 AP and +0.0112 fixed IoU points with paired 95% interval [0.0000, 0.0305] across 412 fixed matches and 93 groups. Both gains were below the predeclared +0.50 AP and +0.20 IoU thresholds, and the interval lower endpoint was zero. The gate stopped despite preserving AP75 and all seven original strict matches. This preservation came with near-total abstention. The observed FIT rates provide no distribution-free coverage claim. Earlier paired, gray and common-count stop decisions remain in force. DEV, official TEST and RD were outside this experiment.
