# Fixed-total variable signed mask edits

**Completed matched development result.** Allowing different numbers of add and drop pixels passed both preregistered advancement criteria. Against archived BCE robust, the fixed-match IoU improvements were +0.990 points with 95% interval [+0.746, +1.242] on PSeg and +0.581 points [+0.300, +0.877] on SPSD. Against the same estimated IoU objective with the original equal add/drop split, the improvements were +0.993 [+0.747, +1.244] and +0.580 [+0.298, +0.877]. The respective mask AP gains were +2.461 and +1.782 points for either comparator.

## Equal-total design

The 183 previously used source-development images share one frozen D-FINE carrier, candidate-balanced BCE endpoint posterior and original native add/drop support. An existing positive quota k always implies exactly 2k actual pixel flips. The fixed writer allocates k to add and k to drop. The variable writer evaluates every legal split of the same 2k among the same signed endpoint pools. Both select the finite candidate action with greatest minimum estimated IoU improvement over the static writer, using exactly the same local foreground proxy and side-ranked subset construction. The near-fixed writer permits at most one extra add or drop from its original split. Mean-IoU and raw minimum estimated intersection controls vary the objective with the same variable support. The frozen candidate roster included all 18,300 original-ranked slots, original scores, 8,821 supported slots, 8,785 positive quotas, and 982,269 enumerated action plans. There were 115,384 actual flips for every new writer. Original masks and scores, all nine archived writer outcomes, endpoint support, and every positive slot's actual signed count independently matched the locked provenance before opening labels.

Each of the two advancement contrasts changed 202 of the 206 common positive-quota native-associated hulls across 106 decoded groups. Both passed the +0.15 IoU margin, positive paired lower bound, AP noninferiority and change thresholds under both annotations. The variable robust writer also improved on archived BCE mean and static under both references.

Mask AP uses all original scores and candidate slots. IoU uses fixed native score-ordered COCO IoU .50 matches, with unmatched hulls at zero, then equal weighting within images, among images within each decoded group, and among the 180 groups. The reference sets contain 411 PSeg hulls and 410 SPSD hulls. Intervals use 4,096 paired decoded-group resamples shared across references.

## All-slot scores

| Writer | PSeg mask AP | PSeg fixed IoU | SPSD mask AP | SPSD fixed IoU |
| --- | ---: | ---: | ---: | ---: |
| Native | 20.701 | 59.118 | 20.789 | 58.963 |
| Archived dynamic | 20.864 | 59.280 | 21.096 | 59.092 |
| Archived static | 20.891 | 59.314 | 21.147 | 59.172 |
| Archived BCE robust | 20.996 | 59.541 | 21.797 | 59.666 |
| Archived BCE mean | 20.947 | 59.476 | 22.312 | 59.940 |
| Archived BCE cell-pessimistic | 20.910 | 59.460 | 22.043 | 59.723 |
| Archived action-regressed robust | 21.042 | 59.559 | 21.905 | 59.717 |
| Archived action-regressed mean | 20.884 | 59.482 | 22.245 | 59.934 |
| Archived action-regressed cell-pessimistic | 20.929 | 59.482 | 22.124 | 59.758 |
| Fixed split, estimated robust IoU | 20.996 | 59.539 | 21.797 | 59.667 |
| Variable split, estimated robust IoU | 23.457 | 60.531 | 23.579 | 60.247 |
| Variable split, estimated mean IoU | 23.955 | 60.843 | 23.536 | 60.174 |
| Variable split, raw minimum overlap | 23.849 | 60.760 | 23.483 | 60.107 |
| Near-fixed split, estimated robust IoU | 21.451 | 59.760 | 22.233 | 59.768 |

| Variable robust minus | PSeg Δ mask AP | PSeg Δ IoU [95%] | SPSD Δ mask AP | SPSD Δ IoU [95%] |
| --- | ---: | ---: | ---: | ---: |
| Archived BCE robust | +2.461 | +0.990 [+0.746, +1.242] | +1.782 | +0.581 [+0.300, +0.877] |
| Fixed split, estimated robust IoU | +2.461 | +0.993 [+0.747, +1.244] | +1.782 | +0.580 [+0.298, +0.877] |
| Archived BCE mean | +2.510 | +1.056 [+0.810, +1.303] | +1.268 | +0.307 [+0.056, +0.569] |
| Archived static | +2.565 | +1.217 [+0.977, +1.458] | +2.433 | +1.075 [+0.807, +1.361] |
| Near-fixed split, estimated robust IoU | +2.005 | +0.771 [+0.569, +0.980] | +1.346 | +0.479 [+0.247, +0.728] |
| Variable split, estimated mean IoU | −0.498 | −0.312 [−0.423, −0.209] | +0.044 | +0.073 [−0.012, +0.163] |
| Variable split, raw minimum overlap | −0.392 | −0.228 [−0.306, −0.149] | +0.096 | +0.140 [+0.048, +0.242] |

The near-fixed restriction recovered some of the improvement. The wider split improved both reference scores further. The mean-IoU and raw intersection variants both had higher AP and IoU on PSeg and lower point estimates on SPSD than the variable robust variant. Thus the volume freedom is supported against strong and objective-matched fixed baselines. The robust objective trades performance between references, and the present evidence gives no advantage under both references over all variable alternatives.

## Area and true overlap

The same native-matched hulls separate the measured IoU change into a true-intersection term and a mask-area term. For baseline 0 and treatment 1, use true area G, mask areas M and true intersections I, with U=M+G−I. The symmetric exact decomposition is ΔIoU = [((M0+M1)/2+G)ΔI − ((I0+I1)/2)ΔM] / (U0U1). The two terms sum to the measured IoU change in every group. Area terms include the direct algebraic denominator effect and are reported separately from absolute area error against the annotation. Area changes and intersection gains are measured in raster pixels.

| Reference | Comparator | Δ IoU [95%] | Overlap term [95%] | Area term [95%] | Δ true intersection [95%] | Δ absolute area error [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PSeg | Fixed split, estimated robust IoU | +0.993 [+0.747, +1.244] | +2.243 [+1.797, +2.692] | −1.250 [−1.500, −1.015] | +4.195 [+3.543, +4.882] | −4.705 [−5.568, −3.838] |
| PSeg | Archived BCE robust | +0.990 [+0.746, +1.242] | +2.241 [+1.794, +2.691] | −1.251 [−1.500, −1.015] | +4.190 [+3.542, +4.877] | −4.705 [−5.568, −3.838] |
| SPSD | Fixed split, estimated robust IoU | +0.580 [+0.298, +0.877] | +2.041 [+1.592, +2.518] | −1.461 [−1.758, −1.169] | +3.469 [+2.850, +4.113] | −1.520 [−2.580, −0.470] |
| SPSD | Archived BCE robust | +0.581 [+0.300, +0.877] | +2.042 [+1.593, +2.518] | −1.461 [−1.758, −1.169] | +3.469 [+2.853, +4.113] | −1.520 [−2.580, −0.470] |


Against the fixed-IoU rule, the positive overlap term more than offset the area term under both references. Estimated robust variable actions grew 6,029 masks, shrank 2,188 and reproduced the exact original equal-split choice in 568. They exchanged 67,777 adds and 47,607 drops, on the same total budget as fixed. The realized split grew 199 and shrank 3 of 206 common matched positive-quota hulls, with four unchanged. True intersection increased for both references in 182 of 206 hulls, decreased in at least one in five, and the lower of the two differences was zero in 19. This is consistent with recovering original foreground omissions. The result is a single seed on a repeated source-development view. Both author masks describe the same image group partition, so this evidence supports annotation-reference robustness within the view and requires fresh acquisitions for a transfer claim.

## Verified records

The first scoring attempt stopped in the geometry audit before opening labels. Its fixed-split numerical bound on expected intersection gain was invalid for unequal signed counts. The subsequent score uses the exact bound given by the two masks' foreground set differences, with the same original protocol, renderer and actions. The independent checks covered every actual signed endpoint, archive parity and all five budgets. See the prescore erratum.

- Scoring protocol SHA-256 `5866e98058edf8728dfb072062ad5d2ca1c8ef6b2f17609e9c54854b0182ad89`.
- Rendering protocol SHA-256 `90151f99805ec9baaf8a08236b8f9981f86a43247943bc7a03ecf7ce3bc25fa4`.
- Immutable renderer output SHA-256 `d12a193bf80e6fb24b3c853c8da7d342f6408aa22e9f3920b873f2d5eb3de1e5`.
- Natural render terminal SHA-256 `cb84320a93db76c5d11156c40aca0b2b1bd4d0fa2bd746a7d68c6c86ccaef9d2`.
- Accepted actual score SHA-256 `f86390623ffd658e91c17e8a7b99e0859bcc3918b9fce0afda2ff332d72d023a`.
- Natural score terminal SHA-256 `a3ad0ea5274ecfd7c11c7baf1d4ad2047b1926790871268922c5b90fc97edeb4`.

[评分汇总](../../results/SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125_SAFE.json) · [无标签写回](../SPSD_PSEG_GATE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125/RENDER_ACTUAL.md)
