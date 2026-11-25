# HRSID DEV705 learned bounded pairing

The historical DEV705 comparison reproduced both archived comparators and the original native metric values. It used the same RF-DETR 5202 archive and all three frozen occupancy-head fits. The fixed arm (F) uses original distance-first pairs. The bounded arm (L) optimizes the same estimated occupancies and exact signed count under each candidate's maximum selected F displacement. Independent selection (I) uses the same signed count with no displacement cap. The fixed signed budget keeps area constant. All 67,386 original mapped candidate ranks and scores enter COCO scoring, together with 1,782 historical author instances. The continuous match analysis uses 1,461 fixed native .50 matched hulls across 27 acquisition prefixes. [Protocol](../PRELOCK.md) · [Score and fixed-match evidence](SCORE_SAFE.json)

| Frozen head seed | F mask AP | L mask AP | I mask AP | L−F continuous IoU | I−L continuous IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5217 | 56.056 | 56.481 | 56.904 | +0.274 | +0.341 |
| 5218 | 55.974 | 56.434 | 56.898 | +0.265 | +0.376 |
| 5219 | 55.987 | 56.527 | 56.934 | +0.308 | +0.355 |
| Three-head mean | 56.006 | 56.481 | 56.912 | +0.282 | +0.357 |

The native mask AP is 56.072. All metric differences in the table are points. The continuous geometry calculation holds the original matched objects fixed, averages paired effects across the three heads on each hull, then gives each acquisition prefix equal weight. Both AP changes preserve direction with the four empty raster annotations omitted. [Complete scores](SCORE_SAFE.json)

| Comparator | Δ mask AP | Δ continuous IoU [95% paired interval] | Changed matched hulls / prefixes | Preregistered gate |
| --- | ---: | ---: | ---: | --- |
| Independent minus bounded | +0.431 | +0.357 [+0.296, +0.429] | 1,438 / 27 | positive |
| Bounded minus fixed | +0.475 | +0.282 [+0.219, +0.344] | 1,436 / 27 | positive |
| Independent minus fixed | +0.906 | +0.639 [+0.548, +0.743] | 1,443 / 27 | positive |

Re-pairing under the original maximum displacement improved both mask AP and fixed-match shape agreement. Independent selection retained a further gain on this cohort. The bounded arm missed the preregistered practical recovery tolerances of ±0.15 AP points and a continuous IoU interval fully within ±0.15 points. The cap controls maximum individual displacement. Mean displacement and spatial layout may vary. Differences in group-equal continuous false positives and false negatives are equal because the writes preserve area on each hull. L−F reduces both by 0.589 continuous pixels per hull, while I−L reduces both by 0.719. [Decision and geometry](SCORE_SAFE.json)

The label-free replay reproduced the native scores, original F and I masks, all archived per-candidate signed coordinates and counts, and original posterior-based action sums. Under the original cap, I's optimal endpoint set was feasible on 1,746 of 40,316, 1,297 of 40,512, and 1,615 of 40,571 positive-budget candidates for seeds 5217, 5218, and 5219, respectively. These counts describe feasible matching under the inherited cap. AP and shape agreement use a separate historical author scoring phase. New detector forwards and optimizer updates were zero. These three heads share the same historical images and acquisition prefixes. [Replay counts](RENDER_SAFE.json) · [Prescore checks](PRELABEL_SAFE.json) · [Observed replay exit](RENDER_TERMINAL.json) · [Observed score exit](SCORE_TERMINAL.json) · [Reproduction instructions](REPRODUCE.md)
