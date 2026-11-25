# Matched D-FINE intermediate endpoint features on the source development gate

**Completed source-development result.** Adding spatially aligned D-FINE intermediate endpoint vectors increased mask AP against archived regression and the equal-width scramble under both references. The paired localization comparisons failed both locked primary tests. Spatial alignment did not yield a demonstrated localization advantage within this feature projection and short training recipe.

## Capture and matched fit

The original frozen detector supplied its 256-channel, 160 × 160 mask feature map at the existing add/drop endpoint coordinates. The acquisition used a fixed six-dimensional signed projection and replayed all 915 author training images. Raw archived top-100 probabilities, candidate audits and author detector exports matched exactly. The observed capture required 915 detector forwards. A separate readback verified the archive and coordinate correspondence, including all 183 gate images, 8,821 supported candidates and 265,943 endpoint pairs. It accessed no annotation or historical test. The gate's render and score used the saved vectors without a detector forward or training update.

Each 1,346-parameter head received 1,000 final updates on the same draws and targets, reusing a previously closed 5,000-update group-out-of-fold action bank. Aligned and scrambled heads replaced the same six duplicate features and retained the first 22 inputs and their moments. They shared cue scaling, loss and supervision. The scramble retained each side's exact six-value vector multiset and permuted locations inside each original proposal and side. Original regression tensors and all three regression writer masks and scores matched their archives exactly.

Both references use all 18,300 original slots and original scores for COCO mask AP. Fixed IoU uses original score-ordered COCO IoU 0.50 matches, zero IoU for native misses, 411 PSeg reference hulls and 410 SPSD hulls, and equal weighting within images and exact decoded groups. All 8,785 positive slots retain 57,692 original additions and removals per writer. Intervals use 4,096 paired resamples over the 180 decoded image groups. The 183 source-development images have been used in earlier design iterations.

## All-slot scores

| Writer | PSeg mask AP | PSeg fixed IoU | SPSD mask AP | SPSD fixed IoU |
| --- | ---: | ---: | ---: | ---: |
| Native | 20.701 | 59.118 | 20.789 | 58.963 |
| Dynamic | 20.864 | 59.280 | 21.096 | 59.092 |
| Static fixed budget | 20.891 | 59.314 | 21.147 | 59.172 |
| Regression robust, exact archive | 21.059 | 59.550 | 21.774 | 59.644 |
| Regression mean, exact archive | 20.915 | 59.458 | 22.386 | 59.916 |
| Regression cell-pessimistic, exact archive | 20.985 | 59.474 | 21.995 | 59.736 |
| Aligned feature robust | 21.080 | 59.498 | 21.899 | 59.603 |
| Aligned feature mean | 20.926 | 59.450 | 22.247 | 59.908 |
| Aligned feature cell-pessimistic | 20.997 | 59.434 | 22.170 | 59.729 |
| Scrambled feature robust | 21.025 | 59.508 | 21.539 | 59.524 |
| Scrambled feature mean | 20.986 | 59.490 | 22.294 | 59.836 |
| Scrambled feature cell-pessimistic | 20.926 | 59.441 | 21.952 | 59.620 |

| Aligned robust minus | PSeg Δ mask AP | PSeg Δ IoU [95%] | SPSD Δ mask AP | SPSD Δ IoU [95%] |
| --- | ---: | ---: | ---: | ---: |
| Regression robust | +0.020 | −0.053 [−0.109, −0.002] | +0.125 | −0.041 [−0.127, +0.040] |
| Scrambled robust | +0.054 | −0.010 [−0.056, +0.032] | +0.361 | +0.078 [−0.016, +0.176] |
| Same-head mean | +0.154 | +0.048 [+0.001, +0.096] | −0.348 | −0.305 [−0.432, −0.188] |
| Same-head cell-pessimistic | +0.082 | +0.063 [+0.015, +0.116] | −0.271 | −0.127 [−0.227, −0.032] |
| Static | +0.188 | +0.183 [+0.084, +0.293] | +0.753 | +0.430 [+0.221, +0.681] |

Both primary contrasts pass AP noninferiority and exposure. Aligned masks changed relative to regression on 205 of 206 shared positive-quota hulls and relative to scramble on all 206, across 107 groups in each contrast. Neither contrast reached +0.15 IoU points with positive paired lower limits in both references. At SPSD, the aligned mean and cell-pessimistic rules had higher IoU than the aligned robust rule. Relative to static, the aligned robust writer retained positive IoU effects in both references.

## Fixed-menu mechanism

Label-free rendering enumerated 460,093 legal actions across 8,785 slots. The 206 common positive-quota matches use 822 shared direct and 17,819 shared full-menu actions. Measure signed intersection gain relative to static against each original author mask. The common gain is the midpoint of the two gains, the half-gap is half their absolute difference, and the lower gain equals common gain minus half-gap. Each head was also allowed a diagnostic selection using the exact same full support, with a static-first tie order. All numbers in the table use raw cell gains and paired group intervals.

| Mechanism | Regression | Aligned | Scrambled | Aligned − regression [95%] | Aligned − scrambled [95%] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full shared-menu common-gain MAE | 1.216 | 1.225 | 1.268 | +0.008 [−0.046, +0.060] | −0.043 [−0.097, +0.012] |
| Full shared-menu half-gap MAE | 0.756 | 0.759 | 0.763 | +0.003 [−0.015, +0.021] | −0.003 [−0.021, +0.014] |
| Minimum realized gain, own support | −0.077 | −0.305 | −0.336 | −0.227 [−0.457, +0.004] | +0.031 [−0.189, +0.251] |
| Minimum realized gain, shared support | −0.083 | −0.295 | −0.336 | −0.212 [−0.442, +0.019] | +0.040 [−0.180, +0.261] |

Using identical full support changed the selected action for only 3 regression, 1 aligned and 1 scrambled cases. Under the shared support, 72, 76 and 85 cases respectively selected edits with negative realized lower-reference gains. The aligned head's point estimates against scramble are favorable on full-menu errors and realized lower reference gain, with intervals crossing zero. The stricter control gives no clear evidence that the location information carried by this projection improved action selection. The AP signals and learned-versus-static effects support continued study of the fixed-budget pipeline, scoped to this single-seed, repeatedly used source-development cohort.

## Verified records

- Gate protocol SHA-256 `23de6ce51b03ea31e6951d0a4e08e814d16adfbc65a60665b52036f7d57ade0c`.
- Closed native cue capture SHA-256 `e60b218556cca7fc3b6a069d5fccce63dc8f69f4eb2f93453cb32e35e0fc2cec`.
- Independent cue readback SHA-256 `bb9644c4605ef238aa8779028c3b2d77451a66657767b56b05214a1f73148b1e`.
- Closed FIT SHA-256 `9c9545b9b1974e4048f44bea3a6665e75e1113ec6dec08cefd56aed60a64650b`.
- Label-free render SHA-256 `f26e653800b449cc72af68bff0224614e2911823af0ef10449e6b0a347dd55d3`.
- Actual score SHA-256 `8ff9db309fc81d201ee2ba6525aec31de9098fdeb3cabda09e8234a68435fdba`.
- Natural render terminal SHA-256 `5cb83b479593036740fd07e753baed2d44a37dbd9a46374f9145980deedc9e4e`.
- Natural score terminal SHA-256 `68928cc11652b21938249ba52aff3fd9270ca69a902c5758bdc8b552023a7712`.

[重放汇总](../../results/SPSD_PSEG_GATE_MATCHED_SPATIAL_CARRIER_CUES_SEED5297_20261125_RENDER_SAFE.json) · [评分汇总](../../results/SPSD_PSEG_GATE_MATCHED_SPATIAL_CARRIER_CUES_SEED5297_20261125_SAFE.json)
