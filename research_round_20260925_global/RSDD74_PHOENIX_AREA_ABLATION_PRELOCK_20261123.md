# Signed edit counts at equal total flips on archived Instance-RSDD 74

## Primary question

Does preserving the original mask area improve Phoenix boundary editing when total flips, writable locations, candidate eligibility and ranking inputs are identical? The archived complete-mask comparison changes all of these jointly and overextends candidate masks. This follow-up isolates the signed quota using the already evaluated 74 chips and the exact archived Phoenix endpoint probabilities. It does not measure a new-dataset replication or establish a property of all refiners. [Whole-mask result](RSDD74_PHOENIX_WHOLEMASK_ACTUAL_20261123.md) · [Closed endpoints and original signed-quota result](INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md)

## Population and three locked writers

Use all 7,346 original RF-DETR 5202 candidates in their original order, with identical score, class and original mask. Only the same 4,107 candidates with positive dynamic rule quota k may be edited. Every arm makes exactly 2k flips at the same original candidate's endpoints. Across all candidates, this gives 59,800 flips per arm. The old signed support and all full-precision Phoenix probabilities, at 129,621 sampled foreground/background pairs, are reused without inference. Candidates with zero edits in the archive retain their complete original masks. No outcome-based change to k, candidate set, original polygon or ranker.

For an addition with probability p, use utility p minus 0.5. For a deletion use 0.5 minus p. These score the expected gain in pixel correctness up to a common factor if p is calibrated. They provide the same within-side ordering as the archived Phoenix policy. They are not an assumption of cross-sensor calibration. Prefix sums on the sorted sides give the maximizing signed split, with no need to approximate search over masks or reject negative gains. Sort equal utilities by original ordinal within each sign. At identical total utility use the smallest absolute imbalance, then the smallest addition count.

1. Balanced fixes exactly k additions and k deletions. Verify complete equality to each archived Phoenix edited mask.
2. Relaxed permits net foreground area change of at most two pixels either way. The exact total stays 2k, so it adds only the two neighboring signed splits to the balanced feasible set when available.
3. Free permits any available signed split at the same total 2k on the original 2k-or-larger union of writable endpoints.

Always verify stock identities, archived endpoint signatures, chosen sets, exact flip counts and closed writer receipts before opening labels in the independent scoring stage. The larger set guarantees a nondecreasing score objective only. It may increase or decrease real overlap. Whole writes and source-trained occupancy heads are unchanged historical context. The primary contrast is balanced minus free. Relaxed minus balanced is the designated secondary contrast. Report raw and archived matched-arm parity. No post-score redefinition of this comparison.

## Measures and decision

Report all 74 chips as primary, with the archived date-separated, Gaofen-3, TerraSAR-X and decoded-deduplicated views. Score pooled full-candidate COCO AP, AP75, AP90, APs and AR100. Evaluate continuous IoU, reference-normalized FP and FN on the same original valid, area-eligible fixed matches. Average matches within each parent, then weight parents equally. Estimate paired 95% geometry intervals with the frozen 4,096 paired sensor/date draws, seed 5217. AP differences are pooled point estimates. Retain all views and both signs.

For each ordered contrast, eligibility to decide requires 20 original eligible fixed hulls with different written masks across at least eight sensor/date blocks. The primary success threshold is at least +0.30 AP and a strictly positive lower continuous-IoU interval bound. The mirrored rule defines an empirical loss. For relaxed minus balanced apply the same thresholds as a separately identified secondary comparison. A contrast that fails exposure or has mixed outcomes remains descriptive. A failed comparison stops advancing the corresponding same-budget quota claim for this frozen recipe. The no-target-data writer and this retrospective matched experiment do not establish a general optimal quota or constitute a novel optimization primitive.
