# Matched selective adoption of fixed-budget BCE edits

**Completed source-development result.** The learned ranking outperformed the raw posterior margin when both chose the same number of learned edits and the same quota composition per image. Both author references gave positive paired fixed-IoU intervals and higher mask AP. The PSeg effect was smaller than the prespecified +0.15-point margin. The selective policy also underperformed always adopting the original BCE action on both references and failed the locked overall advancement rule.

## Locked comparison

The original candidate-balanced BCE head provided the unchanged feasible robust edit for each candidate. A separate nine-feature ridge model was fitted on 847 distinct feasible edits from 447 FIT images in 439 groups, using previously closed group-out-of-fold candidate-balanced BCE predictions. Its target was the smaller signed intersection gain under the two author masks, measured against static and divided by twice the budget. No gate masks entered the fit. At evaluation, it estimated the same target from image-side features. The comparator used the original BCE head's own normalized predicted minimum on exactly the same robust proposal. Neither policy generated a new action.

An image with n valid distinct actions contributes floor(n/3) robust adoptions per policy, allocated over its original edit budgets by largest remainder. Both select from the same candidate roster and have identical counts and budget composition per image. Equal scores break by original rank. The 183 images have 8,706 eligible proposals and 2,844 adoptions for each policy across 173 nonempty images. The policies agreed on 2,166 adoptions. Other positive slots retained the static edit. Zero-quota slots retained native masks. Every writer kept the original 18,300 slots and scores, 8,785 positive quotas and 57,692 additions and removals. Every original archive RLE, score and fixed IoU was replayed exactly.

The gate used the same 183 previously used source-development images and 180 decoded groups. It had no external acquisition or historical test. Mask AP uses all detections. Fixed IoU uses native COCO .50 detection associations, zero for native misses, all 411 PSeg and 410 SPSD hulls, then group-equal scores. Paired intervals use 4,096 group resamples. The fit, label-free render and score reached natural exit0 and independently verified result footers.

## All-slot scores

| Writer | PSeg mask AP | PSeg fixed IoU | SPSD mask AP | SPSD fixed IoU |
| --- | ---: | ---: | ---: | ---: |
| Native | 20.701 | 59.118 | 20.789 | 58.963 |
| Dynamic | 20.864 | 59.280 | 21.096 | 59.092 |
| Static | 20.891 | 59.314 | 21.147 | 59.172 |
| Always BCE robust | 20.996 | 59.541 | 21.797 | 59.666 |
| BCE mean | 20.947 | 59.476 | 22.312 | 59.940 |
| BCE cell-pessimistic | 20.910 | 59.460 | 22.043 | 59.723 |
| Archived signed-action robust | 21.042 | 59.559 | 21.905 | 59.717 |
| Archived signed-action mean | 20.884 | 59.482 | 22.245 | 59.934 |
| Archived signed-action cell-pessimistic | 20.929 | 59.482 | 22.124 | 59.758 |
| Raw BCE margin-ranked | 20.901 | 59.368 | 21.188 | 59.270 |
| FIT ridge-ranked | 20.930 | 59.452 | 21.371 | 59.423 |

| Learned ranking minus | PSeg Δ mask AP | PSeg Δ IoU [95%] | SPSD Δ mask AP | SPSD Δ IoU [95%] |
| --- | ---: | ---: | ---: | ---: |
| Always BCE robust | −0.066 | −0.090 [−0.167, −0.015] | −0.426 | −0.243 [−0.422, −0.089] |
| Raw BCE margin-ranked | +0.030 | +0.084 [+0.006, +0.209] | +0.183 | +0.154 [+0.026, +0.364] |
| Same-head BCE mean | −0.017 | −0.024 [−0.124, +0.077] | −0.940 | −0.517 [−0.737, −0.323] |
| BCE cell-pessimistic | +0.021 | −0.008 [−0.105, +0.088] | −0.671 | −0.300 [−0.501, −0.127] |
| Static | +0.039 | +0.137 [+0.036, +0.274] | +0.225 | +0.251 [+0.101, +0.469] |

Learned-vs-margin passed the AP guard, positive lower bound for both references, and change exposure with 31 common positive-quota hulls across 26 groups. It missed the +0.15-point materiality cutoff on PSeg. Against always robust, the learned policy changed 165 such hulls across 95 groups, but showed lower IoU on both labels and missed the SPSD AP guard. It was also below the same-head mean. Performance relative to static retained positive paired effects under both annotations.

## True lower-reference edit gain

Across 206 exact shared native-association, positive-quota eligible hulls, the table gives realized minimum signed intersection gain over the two author masks. A fallback contributes zero. Policies adopt equal counts across the original all-slot cohort. The subset of author-associated hulls has different adoption counts.

| Policy | Adopted / 206 | Negative | Zero | Positive | Group-equal minimum gain, cells | Group-equal normalized gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static | 0 | 0 | 0 | 0 | +0.000 | +0.00000 |
| Always BCE robust | 206 | 76 | 85 | 45 | −0.022 | −0.00058 |
| Raw margin-ranked | 34 | 11 | 12 | 11 | +0.083 | +0.00481 |
| FIT ridge-ranked | 41 | 9 | 16 | 16 | +0.230 | +0.01429 |

Group-equal realized lower gain improved by +0.146 [+0.008, +0.331] cells relative to the same-budget margin control. The normalized effect was +0.00948 [+0.00079, +0.02119]. The observed harmful fraction among adopted eligible hulls was 9/41 (22.0%) for learned ranking and 11/34 (32.4%) for raw margin. The raw lower-gain difference from always adopting was +0.251 [−0.00039, +0.491] cells. These paired gains localize an effect in the ordering at fixed coverage. The adopted subgroup result does not replace all-hull IoU and all-slot AP, where the fixed one-third policy fell below the original writer.

## Verified records

- Gate protocol SHA-256 `4eba0ba9d289ba7544df104aadb899a6056af9e04d04e13bc02a5f284dd2a85e`.
- Closed selective FIT SHA-256 `31540b54793100849988159f66e732121f24d6d08673b428573bd6ea4dacfd26`.
- Independent FIT readback SHA-256 `5016e4d3f02c80193a23c28bb124c3cc3027cf1b8728839157d36def56b1d7ea`.
- Archived BCE score SHA-256 `b93e0f4d355f538ae751b8622bf359541dd5705f25d1dfe6eca836984521b052`.
- Label-free render SHA-256 `ef158fdab2cf6222bd752b0dd0cfb0432a6abae54e485e8daff525a8363747cc`.
- Actual score SHA-256 `f3e07c567133a51617399fe4758ff383793354e190ae5001d5a57309ead6e15a`.
- Natural render terminal SHA-256 `9af049ab3c5a3a3398e4fba9ae771cc021b276fee424c2ed164b778677bc4de0`.
- Natural score terminal SHA-256 `dc23766ceed88011c4907708bbf609ef486f5bbf4c69eaaccbd6ca3618732924`.

[重放汇总](../../results/SPSD_PSEG_GATE_SELECTIVE_BCE_ADOPTION_SEED5297_20261125_RENDER_SAFE.json) · [评分汇总](../../results/SPSD_PSEG_GATE_SELECTIVE_BCE_ADOPTION_SEED5297_20261125_SAFE.json)
