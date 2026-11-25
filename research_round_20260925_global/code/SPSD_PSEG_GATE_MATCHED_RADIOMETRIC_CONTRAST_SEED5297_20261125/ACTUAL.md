# Matched radiometric feature replay on source development chips

**Actual result, 25 November 2026.** Joint within-candidate radiometric features did not pass the locked advancement criteria. Both references showed higher mask AP than the exact regression and side-wise feature baselines, while fixed-match IoU differences were small and their paired intervals crossed zero. Exact capacity, supervision, candidate support, and edit-budget controls isolate the effect of this feature replacement within the specified short-budget recipe.

## Matched fit and replay

The three heads share the frozen detection and mask construction path, original 14 input features, 1,346-parameter head, candidate support, and final training recipe. The replacement inserts five radiometric cues into duplicate input slots. They comprise centered midranks of intensity, its local 3 × 3 mean, and intensity minus the local 5 × 5 mean, together with robust standardized intensity and residual. Joint scaling uses both sides of each candidate, and the equal-width control computes the same cues on each side separately. These cues use the supplied rendered grayscale intensities.

The same group-out-of-fold action bank was reused. Each head received 1,000 final updates with the same sampled candidate packs, seed 5297, BCE and signed-action regression. The original regression tensors matched exactly, and all three regression writer streams matched the archived stream in exact encoded masks and scoring results. Rendering and scoring both exited naturally with verified result footers. Gate execution used no detector forwards, training updates, fit labels, historical test, or external acquisitions. Rendering supplied all 18,300 immutable candidate slots on 183 previously used development chips and 180 decoded-image groups. It preserved native ordering and scores, with 8,821 supported candidates, 8,785 positive quotas, and 57,692 exact additions and removals per writer.

Fixed matching under each author's reference uses native IoU 0.50, zero for unmatched hulls, and equal group weights after equal image weighting. There were 411 PSeg reference hulls (311 native matches) and 410 SPSD reference hulls (308 native matches). IoU intervals below are percentile intervals from 4,096 paired decoded-group resamples. Mask AP uses all original slots.

## Exact all-slot scores

| Writer | PSeg mask AP | PSeg fixed IoU | SPSD mask AP | SPSD fixed IoU |
| --- | ---: | ---: | ---: | ---: |
| Native | 20.701 | 59.118 | 20.789 | 58.963 |
| Dynamic area baseline | 20.864 | 59.280 | 21.096 | 59.092 |
| Static fixed-budget baseline | 20.891 | 59.314 | 21.147 | 59.172 |
| Regression robust, exact archive | 21.059 | 59.550 | 21.774 | 59.644 |
| Regression mean, exact archive | 20.915 | 59.458 | 22.386 | 59.916 |
| Regression cell-pessimistic, exact archive | 20.985 | 59.474 | 21.995 | 59.736 |
| Joint radiometric robust | 21.111 | 59.568 | 21.901 | 59.646 |
| Joint radiometric mean | 20.964 | 59.528 | 22.364 | 59.916 |
| Joint radiometric cell-pessimistic | 21.081 | 59.518 | 21.951 | 59.718 |
| Side-wise radiometric robust | 21.074 | 59.537 | 21.632 | 59.634 |
| Side-wise radiometric mean | 20.919 | 59.530 | 22.329 | 59.923 |
| Side-wise radiometric cell-pessimistic | 21.009 | 59.512 | 21.984 | 59.754 |

| Joint robust minus | PSeg Δ mask AP | PSeg Δ IoU [95%] | SPSD Δ mask AP | SPSD Δ IoU [95%] |
| --- | ---: | ---: | ---: | ---: |
| Regression robust | +0.051 | +0.018 [−0.018, +0.053] | +0.126 | +0.002 [−0.082, +0.090] |
| Side-wise robust | +0.037 | +0.032 [−0.005, +0.070] | +0.269 | +0.012 [−0.047, +0.074] |
| Same-head mean | +0.147 | +0.040 [−0.008, +0.092] | −0.463 | −0.270 [−0.412, −0.142] |
| Same-head cell-pessimistic | +0.030 | +0.050 [−0.001, +0.106] | −0.050 | −0.072 [−0.169, +0.022] |
| Static | +0.220 | +0.254 [+0.137, +0.382] | +0.754 | +0.474 [+0.274, +0.706] |

Both learned-head contrasts passed the AP and exposure thresholds. All 206 common positive-quota hulls across 107 groups changed relative to raw regression, and 202 across 106 groups changed relative to side-wise normalization. Both contrasts missed the prespecified +0.15 IoU-point margin and positive lower interval on each reference. Positive comparisons with static reproduce a benefit for the fixed-budget learned pipeline under both author masks. The SPSD score favored the same-head mean over its robust rule.

## Feasible-action diagnostics

The label-free replay enumerated 385,093 distinct feasible actions across 8,785 candidates. The diagnostic was held to the 206 common positive-quota hulls, with 820 shared direct actions (static and three distinct deployed choices where available) and 16,896 shared full-menu actions. Both author masks supplied target add/drop intersection gains for each fixed mask. The common gain `m` is the arithmetic midpoint of reference gains, and the half-gap `h` is half their absolute difference. The smaller signed reference gain is `m − h`. Normalized gains divide by twice the exchange quota. Raw errors below are in cells.

| Diagnostic | Regression | Joint | Side-wise | Joint − regression [95%] | Joint − side-wise [95%] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full shared-menu raw `m` MAE | 1.2277 | 1.2255 | 1.2430 | −0.0022 [−0.0501, +0.0424] | −0.0175 [−0.0481, +0.0099] |
| Full shared-menu raw `h` MAE | 0.7637 | 0.7723 | 0.7688 | +0.0086 [−0.0088, +0.0260] | +0.0035 [−0.0075, +0.0156] |
| Group-equal realized minimum raw gain | −0.0773 | −0.1134 | −0.1374 | −0.0361 [−0.2339, +0.1596] | +0.0240 [−0.1202, +0.1705] |
| Deployed writes with negative minimum gain / 206 | 70 | 69 | 72 | | |

Full shared-menu results show no clear decrease in calibration error. The smaller direct-choice subset showed lower common-gain MAE for joint than side-wise (−0.0274 cells, 95% interval [−0.0577, −0.0003]). The effect is conditional on the deployed-choice subset and does not establish calibration across the shared menu. Under label-conditioned substitution inside each head's prespecified legal menu, substituting true `m` into the joint selector raised the group-average minimum raw gain +0.936 cells [0.748, 1.123], substituting only true `h` raised it +0.662 [0.503, 0.827], and substituting both raised it +1.023 [0.840, 1.211]. These diagnostics measure remaining feasible-action headroom and use source author masks unavailable at inference.

This completed single-seed source-development run supports a narrow mask-AP signal and confirms the learned pipeline's advantage over static. The paired data do not establish a localization gain attributable to joint radiometric contrast under this design. The repeated-development scope and short head training budget remain explicit.

## Verified records

- Gate protocol SHA-256 `d25a4ff51bc266bdab598a2fb013ff96a401c8368f2aacbcf68873f32e8cbd63`.
- Closed fit aggregate SHA-256 `c3e554b905cbacd0f08dc8107e0ea8ad1d32babb9639fee51791f5b4527bf039`.
- Label-free render aggregate SHA-256 `57b8264a7993df70a4e263e57f59ae5c1084da397a91e085b00533d33dffc2c4`.
- Scoring aggregate SHA-256 `47d73e8b8ee6b721fececfb4c012258d13689ee1ee8d9c330b21d512b1c02d69`.
- Natural render and scoring terminal SHA-256 digests `e39ecbd941211f6dfa3b3367413c0de8ecd839dec392c9237272cad820245ba5` and `241d21f0eeb39faa72f942e2707b1d46d3d1ea802c142a2d9678b4eea79206a1` respectively.

[重放汇总](../../results/SPSD_PSEG_GATE_MATCHED_RADIOMETRIC_CONTRAST_SEED5297_20261125_RENDER_SAFE.json) · [评分汇总](../../results/SPSD_PSEG_GATE_MATCHED_RADIOMETRIC_CONTRAST_SEED5297_20261125_SAFE.json)
