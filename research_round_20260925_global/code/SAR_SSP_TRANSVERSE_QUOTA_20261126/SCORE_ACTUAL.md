# Actual transverse quota test on the historical SAR-SSP reserve

The same frozen D-FINE carrier and source-trained occupancy head retained all 232 images and 23,200 candidates. Native mask, confidence, rank, signed point pool and original candidate budget were fixed. The observed render and independent label-free readback agreed on all archived B and F actions, all new RLEs, and every signed count. Each edited arm made 147,118 flips across 10,393 positive candidates. A total of 4,024 candidates passed the locked native geometry check and 3,959 had a reduced matching menu. Outside geometry eligibility T, A and L retained F. The original paired mask IoU evaluation reproduced native, B and F exactly before reporting new contrasts.

| Arm | Mask AP | AP50 | AP75 | AP90 | APs | AR100 | Fixed native-match IoU |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Native N | 13.344 | 32.503 | 6.933 | 0.000 | 15.193 | 28.921 | 54.831 |
| Balanced B | 13.338 | 31.745 | 7.409 | 0.000 | 15.234 | 29.065 | 55.048 |
| Free F | 13.199 | 31.745 | 7.283 | 0.000 | 14.943 | 28.777 | 54.836 |
| Transverse T | 13.332 | 31.745 | 7.409 | 0.000 | 15.225 | 29.065 | 54.978 |
| Area control A | 13.257 | 31.745 | 7.409 | 0.000 | 15.034 | 28.993 | 54.927 |
| Long-axis control L | 13.315 | 31.745 | 7.283 | 0.000 | 15.129 | 28.993 | 54.944 |

The all-hull scope includes 139 author instances in 23 hull-bearing parent groups. Stock matched 88 hulls at raster IoU 0.50. The 51 misses contributed zero IoU in every arm. Of the fixed matches, 39 in 19 groups had eligible geometry and 38 had a strictly reduced menu. The nonedge sensitivity contains 91 hulls and 66 original matches in 20 groups.

| Contrast | Mask AP change | All-hull IoU change [95% interval] | Changed eligible hulls / groups |
|:---|---:|---:|---:|
| T minus F | +0.133 | +0.143 [−0.185, +0.487] | 36 / 19 |
| T minus B | −0.006 | −0.070 [−0.248, +0.036] | 2 / 2 |
| T minus A | +0.075 | +0.052 [−0.183, +0.270] | 36 / 19 |
| T minus L | +0.017 | +0.035 [−0.159, +0.182] | 29 / 17 |

Scores and differences are percentage points. The 4,096 paired intervals resample the original parent groups. The full-cohort T−F contrast fell below both practical improvement thresholds and its IoU interval crossed zero. T−B also missed coverage among geometry-eligible changed matches. The two specificity controls passed the exposure, false-foreground and AP-tolerance requirements. Both IoU intervals crossed zero. The combined specificity, mitigation and recovery decisions did not pass.

Relative to F, T removed 1.105 points [−1.669, −0.607] of false foreground and increased missed hull area by 0.597 points [+0.282, +1.028]. It reduced total predicted foreground by 1.702 points [−2.522, −0.983] and reduced true intersection by 0.597 points [−1.028, −0.282], normalized to the original author hull area. These changes retain paired covariance through per-hull differences. The same sign pattern appears against both A and L. Nonedge T−F IoU was +0.097 points [−0.261, +0.489]. The strategy reduced false foreground, with part of the change coming at the cost of true hull pixels. This historical result leaves practical transfer unconfirmed.

| Arm | Added cells | Removed cells | Expanded candidates | Shrunk candidates | Balanced candidates |
|:---|---:|---:|---:|---:|---:|
| B | 73,559 | 73,559 | 0 | 0 | 10,393 |
| F | 80,493 | 66,625 | 6,316 | 3,356 | 721 |
| T | 73,215 | 73,903 | 4,518 | 3,256 | 2,619 |
| A | 80,748 | 66,370 | 6,245 | 3,315 | 833 |
| L | 77,214 | 69,904 | 5,491 | 3,160 | 1,742 |

The readback and scoring children exited naturally with a verified footer. There was no detector forward or target fitting. [Registered protocol](PRELOCK.md) · [Readback](VERIFY_SCORE_SAFE.json) · [Complete score](TRANSVERSE_SCORE_SAFE.json) · [Witnessed closures](SCORE_TERMINALS_SAFE.json)

