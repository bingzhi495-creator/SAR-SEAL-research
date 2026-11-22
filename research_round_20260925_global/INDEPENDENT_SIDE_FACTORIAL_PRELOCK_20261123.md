# Separate the addition and deletion constraints in equal-budget refinement

Frozen independent selection improves on fixed pairing with the same candidates, scores and per-instance edit counts on HRSID and Instance-RSDD. These comparisons changed additions and removals together, leaving open whether releasing one side can retain the benefit. This experiment tests the contribution of each native side within the standalone small-head policy. [Original test](INSTANCE_RSDD_FROZEN_HRSID_ACTUAL_20261122.md) · [Additional parents](INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_20261123.md) · [Related patch refinement](https://arxiv.org/abs/2104.05239)

Use the existing frozen fixed-pair mask F and its matched-count independent mask I. Both use the same small occupancy head and native support. Relative to the same native mask N, let A be the additions and D the removals. The first letter indicates the policy for additions and the second indicates the policy for removals.

| Arm | Background pixels added | Foreground pixels removed |
| --- | --- | --- |
| FF | A(F) | D(F) |
| IF | A(I) | D(F) |
| FI | A(F) | D(I) |
| II | A(I) | D(I) |

All arms retain exactly k additions, k removals and area(N) per candidate, including zero-count candidates. Require exact candidate identity, confidence, native masks and support endpoints, along with reproduction of FF and II. Mask preparation uses the closed 740-image author-test and 74-image additional-parent archives. It opens no annotations and performs no new detector, point-head or PointRend inference, fitting or threshold selection. Use frozen seeds 5217 and 5218 for the 740, and seed 5217 on the 74. They share a detector within cohort.

Primary evaluation uses the historical 583 Gaofen-3 images in the author test. Replicate on the 74 additional-parent images from the same release. Use the 42 sensor/date-separated images from the 74 for sensitivity, and retain other archived views. These cohorts have supported prior evaluations. Score the closed masks against the original human polygon annotations, with original reindexing, duplicate handling and COCO rules. Report all 13 COCO metrics. Continuous geometry uses the original valid, area-eligible native IoU 0.50 fixed matches, including no-ops. Use 4,096 paired PCG64(5217) draws with the original acquisition group or sensor/date resampling and parent-equal aggregation.

The conjunction of II minus IF and II minus FI is the new decision. In each contrast require all available seeds to improve mask AP, a strictly positive lower pooled paired 95% continuous-IoU interval, and true location changes for at least 20 eligible original matches and eight groups in at least one seed. Apply the same decision separately to the primary view and the 74-image replication. Report IF minus FF, FI minus FF and II minus FF as secondary controls. Report the descriptive AP interaction without imposing additivity. Keep each failed condition and all arms.

For polygon true intersection J, J(II) + J(FF) - J(IF) - J(FI) equals zero up to 1e-7 square pixels. This is because the four masks start from the same native stock and the two signed edit sets are disjoint. Match-level area stays fixed. Verify both properties on the original fixed matches. The IoU ratio and COCO AP can be nonadditive. A weak side limits the necessity claim while retaining the earlier joint result. This closed study does not select a target-tuned policy or assert geographic independence.

