# Crossing writable support and total flips on archived Instance-RSDD 74

## Primary question and exact population

Does imposing the old per-candidate flip count substantially recover the performance lost by whole-image Phoenix threshold writes when the scores, whole-image support, native proposals and eligible candidates stay fixed? At an equal small count, the [previous area experiment](RSDD74_PHOENIX_AREA_ABLATION_ACTUAL_20261123.md) finds limited evidence for balancing additions and removals. The [original whole writer](RSDD74_PHOENIX_WHOLEMASK_ACTUAL_20261123.md) simultaneously changed support, count and eligibility. The new experiment separates the first two using a common active population.

Retain the frozen RF-DETR 5202 output on the already evaluated 74 chips with all 7,346 mapped native candidates, their scores, native order, classes and masks. Exactly the original 4,107 positive-budget candidates may change in any of the four arms. All other candidates retain native output. No training or labels enter prediction. The full historical LVIS output on 6,079 nonempty masks is a reference with different eligibility. The nested date and sensor views remain those previously specified.

## One score field, two supports, two flip rules

Use the exact admitted Phoenix author commit 13ee3996ef3d0292fab144316cc8e03b69c0ecfa, the official EfficientViT-XL1 instance weights, and the old original RGB and binary prompts. Apply the same author five-iteration, smooth-kernel-1 path with binary prompt and multimask selection. Use seed 5202 per image and the existing DGX four CPU workers, two threads and chunks of two masks. Recompute the fifth posterior only for the 4,107 original positive-budget candidates. Require the posterior values at all 259,242 recorded native endpoints to agree exactly with the old probability ledger. Do not store the full posteriors. Return only the resulting masks and aggregate execution and verification summaries.

Define narrow as exactly the old scored add and drop endpoints, and whole as every native pixel. On the whole grid, signs are determined by the original binary mask. At a location with posterior p, the signed utility is p minus 0.5 to put in foreground and 0.5 minus p to remove foreground. The fixed-count rule chooses exactly 2k total flips, with free signed split, where k is the original dynamic-attraction quota for that particular candidate. It maximizes the sum of signed utility with no zero or negative filter. Equal-utility splits choose minimal absolute signed imbalance, then fewer additions. Within a sign, ties prioritize included original narrow endpoints in their original ordinal order, then other cells in native row-major order. Full support therefore includes narrow without changing its tie convention. The threshold rule writes every available difference between the author binary output and the native mask. This equals applying the positive signed utility threshold apart from exact p=0.5 outputs, where the author's native binary decision determines the mask. Count such cases, and require equality to the posterior threshold away from ties.

| Policy | Writable support | Flip rule |
| --- | --- | --- |
| NC | Narrow | Exact original 2k |
| NT | Narrow | Threshold |
| WC | Whole | Exact original 2k |
| WT | Whole | Threshold |

Require NC to exactly reproduce all 4,107 archived free-quota narrow writes. Require WT to reproduce all 4,107 already computed full LVIS masks at the same original candidates. Verify that NT is the projection of the WT differences onto the narrow set. Independently check all three anchors and the realized counts and preservation of untouched candidates. Both fixed-count arms must spend exactly 59,800 flips. Summarize positive and forced nonpositive flips, sign distribution, support use and exact label-free parity. Validate the observed completion and ordered masks before labels open.

## Estimation and decision

Evaluate complete-stream COCO AP, AP75, AP90, small AP and AR100, author hull rematches and original matched hull retention. Use the original raster-IoU-0.50 fixed matches for continuous IoU and reference-normalized intersection, false positive and false negative area. Retain the original valid area-eligible population, including unchanged pairs. Use the same 4,096 paired sensor/date resamples with seed 5217 and equal parent weights. Report full, date-separated, both sensor and decoded-deduplicated views. AP effects are pooled point estimates. Report the conditional support and threshold/count contrasts and the algebraic interaction (WC minus WT) minus (NC minus NT). Report a paired interval for interaction geometry.

The sole primary contrast is WC minus WT. A recoverable benefit at the original count needs at least +1.00 AP, a positive lower paired 95% continuous-IoU bound, and at least 20 original fixed matches with different written masks across eight or more date blocks. The mirrored rule detects a loss. Secondary contrasts are NC minus WC, NC minus NT and NT minus WT, with descriptive paired 95% intervals and a symmetric +0.30 AP gate at the same exposure floor. Describe what these policies do jointly. Thresholding and an exact count change the realized set and total, so the budget contrast concerns these two operational rules and does not imply a universal monotonic budget effect. This retrospective result will not choose target thresholds or qualify a newly acquired test cohort.
