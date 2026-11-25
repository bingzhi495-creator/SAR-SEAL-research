# Actual native pixel-footprint averaging on SRNet

The observed capture recovered the 608-by-608 probability plane after the rotated RoI paste and before the final 512-by-512 resize. The complete original image list and 89 candidates, including three images with no candidates, stayed in order. Independent capture readback reproduced all original native FP32 probabilities bit for bit. Both editing runs used the exact original 14 features, scalar head, confidence, signed endpoint sets and candidate masks. The independent editing process reproduced the new edit ledger byte for byte.

I is the original independent writer. A uses the mean of the reconstructed pre-resize field over a native pixel footprint in the direct logit prior. S applies the same area-average rule to the already native grid. All three retain the original learned residual and exact signed count from F. All positive-budget candidates used eight additions and eight removals, with 45 edited candidates and 360 writes on each side per arm. Across their 2,880 endpoint entries, the mean absolute prior change from I was 0.00924 for A and 0.01640 for S. A changed 21 candidate edits relative to I, S changed 27, and A differed from S on 23. These counts precede reference scoring.

| Full 74 images | AP | AP75 | AP90 | APs | AR100 |
|:---|---:|---:|---:|---:|---:|
| Native N | 33.265 | 12.480 | 0.064 | 33.720 | 42.222 |
| Fixed distance-paired F | 33.283 | 12.480 | 0.064 | 33.735 | 42.222 |
| Independent I | 33.428 | 12.480 | 0.064 | 33.937 | 42.346 |
| Source-grid integral A | 33.265 | 12.480 | 0.064 | 33.720 | 42.222 |
| Native-grid integration S | 33.265 | 12.480 | 0.064 | 33.720 | 42.222 |

| View and contrast | AP change | Continuous IoU change [95% interval] | Changed eligible anchors / parents / sensor-date groups |
|:---|---:|---:|---:|
| Full 74, A−I | −0.1627 | +0.0082 [−0.0163, +0.0322] | 15 / 9 / 9 |
| Full 74, A−S | 0.0000 | +0.0027 [−0.0286, +0.0322] | 17 / 9 / 9 |
| Full 74, S−I | −0.1627 | +0.0055 [−0.0136, +0.0277] | 20 / 10 / 10 |
| Nested 42, A−I | −0.2179 | −0.0060 [−0.0308, +0.0190] | 10 / 6 / 6 |
| Nested 42, A−S | 0.0000 | −0.0041 [−0.0373, +0.0322] | 12 / 6 / 6 |
| Nested 42, S−I | −0.2179 | −0.0019 [−0.0254, +0.0288] | 13 / 6 / 6 |

Scores and differences are percentage points. The blocked geometry intervals use the original sensor-date groups and 4,096 draws. Full and nested geometry retained all 32 and 22 eligible original matches, respectively. All preregistered directional contrasts are inconclusive. Neither full-view primary comparison met the practical AP threshold, coverage requirement or strictly positive continuous interval. The perturbation changed the realized rankings while offering no verified advantage specific to reconstructing the pre-resize probability field on this historical cohort. Full small-object, missed-area and excess-area metrics for both original views are included in the aggregate receipt.

[Preregistered scope](PRELOCK.md) · [Replay instructions](REPRODUCE_INTEGRAL_REPLAY.md) · [Safe score](INTEGRAL_SCORE_SAFE.json) · [Observed edit and score closure](INTEGRAL_REPLAY_TERMINALS_SAFE.json)

