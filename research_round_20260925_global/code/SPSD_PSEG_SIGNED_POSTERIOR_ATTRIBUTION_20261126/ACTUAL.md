# Signed posterior attribution on source development

The frozen variable signed allocation improved all-hull IoU over the fixed allocation by **0.993 points** under the PSeg author masks and **0.580 points** under the SPSD author masks. The paired 95% intervals were [0.747, 1.244] and [0.298, 0.877]. The incremental overlap error did not concentrate on cells outside the original native mask on these reused development images. The outside-minus-inside estimates were −0.024 and −0.021 points, with both intervals crossing zero.

## Frozen comparison

The diagnostic reused the original seed-5297 candidate-balanced BCE head, 183 source-development images in 180 decoded groups, 18,300 recorded detections, and 8,785 positive-budget candidates. Both allocation policies used exactly 115,384 total flips. The policy masks differed on 8,217 positive-budget candidates. The scorer reproduced the frozen masks, raw signed overlap forecasts, and estimated IoU gains before opening either set of author annotations. The all-hull IoU intervals and native, fixed, and variable mask AP reproduced the closed score. The analysis used no new detector forwards, parameter updates, FIT labels, historical TEST data, or external acquisitions.

The all-hull denominator includes 411 PSeg and 410 SPSD hulls, of which 311 and 308 matched their native prediction at the original criterion. The fixed and variable masks differed on 224 and 227 matched hulls in 109 and 113 groups. Within the separate shared-hull audit, 410 author-to-author pairs were assigned, 370 reached IoU 0.50, 276 shared their native match, and 206 had a positive common quota.

## Attribution

Let F and V be the archived fixed and variable masks. They use the same detector coordinates, posterior p, static anchor, and edit budget. For an author mask with occupancy y, area T, and overlap I_F with F, define δ = 1_V − 1_F and ΔA = |V| − |F|. The true and predicted changes in intersection are g = Σδy and ĝ = Σδp. The true signed IoU numerator is g − I_F·ΔA/(|F|+T). The scorer substitutes its frozen expected overlap and author-area proxy in the predicted numerator. Thus predicted-minus-true signed numerator error partitions exactly into the occupancy term ĝ − g and the error in the area-slope term. The scorer reports the difference of the two ordinary IoU values separately.

The table reports 100 times the signed quantities divided by each author hull's area T. The two IoU rows are ordinary IoU percentage points. Hulls contribute equally within an image, images equally within their decoded group, and the 180 groups equally. Native misses contribute zero paired effect. Intervals are percentiles of 4,096 paired group draws conditional on the frozen allocations.

| Quantity | PSeg estimate [95% interval] | SPSD estimate [95% interval] |
|---|---:|---:|
| Signed area change | 1.795 [1.457, 2.144] | 2.300 [1.820, 2.791] |
| Predicted intersection change | 1.422 [1.149, 1.708] | 1.524 [1.194, 1.861] |
| True intersection change | 1.333 [1.068, 1.598] | 1.292 [1.003, 1.600] |
| Occupancy contribution to signed numerator error | 0.089 [−0.017, 0.210] | 0.231 [0.049, 0.426] |
| Cells outside native mask | 0.032 [−0.032, 0.108] | 0.105 [−0.005, 0.220] |
| Cells inside native mask | 0.056 [0.005, 0.112] | 0.126 [0.039, 0.224] |
| Outside minus inside | −0.024 [−0.075, 0.029] | −0.021 [−0.105, 0.060] |
| Area-slope contribution to signed numerator error | −0.088 [−0.105, −0.071] | −0.105 [−0.128, −0.083] |
| Total signed numerator error | 0.001 [−0.108, 0.123] | 0.127 [−0.060, 0.320] |
| Predicted IoU delta | 1.273 [1.021, 1.538] | 0.853 [0.655, 1.056] |
| True IoU delta | 0.993 [0.747, 1.244] | 0.580 [0.298, 0.877] |

Variable allocation produced net increases in false-positive area on both sides, even though IoU increased. Under PSeg, the outside and inside changes were +0.241 and +0.221 points, with raw net counts of +268 and +272 pixels. Under SPSD they were +0.410 and +0.597 points, with raw net counts of +341 and +529 pixels.

| Changed endpoint subgroup | PSeg raw cells | PSeg p − y, pp [95% interval] | SPSD raw cells | SPSD p − y, pp [95% interval] |
|---|---:|---:|---:|---:|
| Outside native mask, present only in V | 1,417 | −1.51 [−5.58, 2.90] | 1,434 | 0.62 [−5.20, 6.61] |
| Outside native mask, present only in F | 246 | −3.70 [−7.72, 1.23] | 264 | −10.23 [−21.65, 0.59] |
| Inside native mask, retained only in V | 1,243 | 1.22 [−3.73, 6.07] | 1,251 | 2.42 [−3.43, 8.39] |
| Inside native mask, retained only in F | 72 | 9.04 [−2.55, 19.35] | 81 | −1.26 [−13.65, 11.72] |

The subgroup p − y column is the ratio of the group-weighted residual sum to the group-weighted exposed count. The rows describing cells present only in F receive a negative sign in the signed numerator. In particular, the observed error in posterior occupancy on new outside additions crosses zero under both references. The area-slope term partially offsets the occupancy error. These decompositions describe decisions already made on exposed development data. Causal attribution and transfer of any correction to an independent domain remain untested by this diagnostic.

## Evidence and replay

The [aggregate result](score/RESULT_SAFE.json) is bound to result SHA-256 `5088ea86a48a418988bff78fbb29e342a65991949f4bd985eb48960dd72d3cdc`. The source worker exited naturally with code 0, and an independent wrapper verified its result footer before recording terminal SHA-256 `f15695aa4e84270096923fa588ea24663274f8b5698c856f974d95423cb613e0`.

The [prescore protocol](PRELOCK.md), [source worker](audit_source.py), [aggregation helper](aggregate.py), and [synthetic aggregation checks](test_aggregate.py) identify the reproduction procedure. The source worker takes paths for the closed original render/score, archived same-cohort controls, source development-only annotations, immutable training head, points and proof files, and shared signed arithmetic. Running the full reproduction requires the original admitted artifacts. The archived [fixed/variable score](../SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/ACTUAL.md) contains the original mask-level comparison.

