# HRSID FIT109 simple-point boundary repair

Simple-point constraints kept the foreground component count and hole count of every edited mask. They did not improve mask AP or continuous IoU against either equal-count comparison. The original five source-held-out D-FINE point-head fits were reproduced once, with 200 optimizer steps per fold and zero detector forwards or updates. They supplied the same point scores to every policy.

| Policy | Swaps | Edited candidates | Component or hole count changed | Mask AP | AP75 | Continuous IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unconstrained, original natural count | 36,571 | 6,043 | 848 | 53.294 | 57.646 | 81.455 |
| Unconstrained, simple-point count | 36,351 | 6,030 | 806 | 53.279 | 57.498 | 81.461 |
| First-layer removal preferred, simple-point count | 36,351 | 6,030 | 646 | 53.279 | 57.498 | 81.461 |
| Simple-point exchanges | 36,351 | 6,030 | **0** | 53.279 | 57.498 | 81.458 |

The comparison preserves the original 10,900 native ranks, scores and masks before editing. The three equal-count arms exchange the same number of pixels per candidate, including zeros. Simple-point feasibility reduces the count for 181 candidates and defers 220 swaps. Among the equal-count controls, hole count grows on 335 unconstrained masks and 163 masks with first-layer removal priority. Each of the three matched-count arms retains the seven original strict 0.90 fixed matches and gains five.

The continuous IoU difference is **−0.0035** points versus each equal-count control (95% paired source-group interval −0.0101 to +0.0015, 412 fixed original matches in 93 groups). Both prelocked benefit thresholds and the positive lower-bound requirement fail. AP75 retention and strict-match nondegradation pass. Component and hole count preservation is a measured property of this method. The controlled accuracy results give no basis to expand that claim to improved SAR boundary accuracy.

On the distinct historical PSeg TEST232 evaluation, whole-candidate hole counts and true-hull error have different denominators. In the 293 matched eligible candidates, the natural policy deletes four cells into enclosed background and removes three pixels of their true polygon area. Its native-probability same-count comparator deletes eleven such cells, corresponding to ten pixels of true polygon area. These post-hoc results explain why a higher overall hole count need not imply worse true hull overlap. [Localized PSeg geometry](results/PSEG_TEST232_ENCLOSED_DELETION_GEOMETRY_SAFE_20261121.json)

The actual run exited 0. An independent terminal-and-result check confirmed closure, budget matching, the four historical baselines and the unpassed decision. The detector had seen the FIT images. Only the point heads held out sources by fold.

- [Prelocked design and earlier work](FIT109_BOUNDARY_SIMPLE_POINT_PILOT_PRELOCK_20261121.md)
- [Actual aggregate](results/FIT109_BOUNDARY_SIMPLE_POINT_ACTUAL_SAFE_20261121.json)
- [Closure check](results/FIT109_BOUNDARY_SIMPLE_POINT_READBACK_SAFE_20261121.json)
- [Replay code](code/FIT109_BOUNDARY_SIMPLE_POINT_PILOT_20261121/)

