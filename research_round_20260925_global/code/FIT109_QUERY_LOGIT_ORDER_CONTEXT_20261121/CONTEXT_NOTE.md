# Point ranking and query-logit context on HRSID FIT109

Adding ordinary same-side pairwise ranking to pointwise binary cross-entropy (BCE) increased mean mask AP by **0.078 points** across three point-head seeds. The AP difference changed sign across seeds. A separate single-seed test found that retaining the angular order of nearby query logits reduced AP by **0.437 points** relative to sorting the same values within each ring.

## Paired training objectives

Both objectives used the same 14 endpoint features, continuous pixel-occupancy targets, five source-group folds, and 200 optimizer steps per fold. The pairwise term had weight 0.25. Within each seed, both heads shared their initialization and training draws. The frozen detector had trained on FIT109. The folds held sources out of point-head fitting.

| Head seed | BCE AP | Pairwise AP | AP difference | Continuous IoU difference |
| --- | ---: | ---: | ---: | ---: |
| 5217 | 55.740 | 55.912 | +0.172 | +0.082 |
| 5218 | 55.791 | 55.757 | −0.033 | +0.029 |
| 5219 | 55.568 | 55.662 | +0.095 | +0.018 |
| Mean | **55.699** | **55.777** | **+0.078** | **+0.043** |

Both trained heads exceeded native RF-DETR (54.844 AP) at every seed. Mean continuous IoU was 84.431 for BCE, 84.474 for pairwise training, and 83.934 for the native masks. The three-seed spread measures sensitivity to point-head initialization and sampling on a single detector. At seed 5217, the decision-focused SPO+ objective reached 55.624 AP and failed its prespecified advancement criteria. The two additional seeds repeated the ordinary pairwise control and matched BCE.

## Does angular order help?

At seed 5217, three equal-width heads used the pairwise objective and the same training draws. Each had 32 additional input slots. The ordered arm sampled signed logits on square rings of radius 1 and 3 in the native 144 × 144 query grid, giving 8 and 24 values. The sorted arm retained exactly those values and sorted each ring independently. The zero-padded arm reproduced the 14-feature pairwise result. Feature extraction used archived query planes and finished before labels were attached.

| Input | Mask AP | AP75 | AP90 | Continuous IoU | Original strict matches lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero padding | 55.912 | 60.906 | 7.056 | 84.489 | 3 |
| Sorted within each ring | 56.246 | 60.888 | 7.196 | 84.517 | 2 |
| Ordered within each ring | 55.809 | 60.755 | 6.745 | 84.475 | 4 |

Ordered minus sorted was −0.041 continuous IoU points (paired 95% interval −0.151 to +0.061). Ordered minus zero was −0.103 AP points and −0.014 continuous IoU points (−0.109 to +0.080). The ordered arm failed the prespecified comparison against both controls. The sorted arm's +0.334 AP over zero padding is an exploratory single-seed observation.

All AP values pool the 10,488 mapped candidates across the 109 FIT images with their original confidence and order. Each trained arm used the same per-candidate quota across 6,738 area-eligible candidates, adding 48,982 pixels and removing 48,982. Continuous IoU weights the 92 contributing source groups equally across 392 original fixed matches. Strict losses count identities whose native continuous IoU exceeded 0.90 and whose edited mask fell to or below 0.90. The paired intervals quantify source-group variation in continuous IoU for the specified seed. Generalization to new detector fits or test images requires separate evaluation.
