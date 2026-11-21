# FIT109 simple-point boundary repair

The natural PSeg-SSDD refinement retains foreground area and increases hole count on 1,544 of 10,466 edited masks. A matched native-probability policy does so on 519 masks. This observation motivates an HRSID FIT development test of topology-preserving point edits. The known mathematical precedents are digital simple points and area-preserving exchange. The hypothesis concerns expected boundary accuracy on the frozen SAR endpoint. [PSeg failure analysis](PSEG_SSDD_TEST232_DFINE_FROZEN_ACTUAL_20261121.md) · [Area-preserving exchange](https://doi.org/10.1016/S0031-3203(00)00105-9) · [Digital topology](https://doi.org/10.1109/TPAMI.2003.1201824)

## Frozen comparison

- HRSID FIT109, D-FINE E24 trained on the source domain, original 10,900 candidate ranks and scores. Recreate the original five source-held-out point-head fits once using 200 updates per fold, the original schedule and frozen endpoint support. Use the same point probabilities in every arm. No additional detector forward or target data access.
- Reproduce the original raw, fixed-pair, equal-count and natural-policy rows. Their results and edit counts verify shared inputs and score calculation.
- For each candidate, take the natural-policy count as the cap. In descending original addition score, tentatively add the first eligible background point. Among original foreground points in ascending removal score, keep the first positive-gain swap whose two flips are simple in sequence. Recheck the current mask at each flip. Preserve eight-connected foreground and four-connected background, with exterior background padding. Use original ordinals for ties, prohibit endpoint reuse, and stop when no positive simple pair remains.
- Measure the realized per-candidate count, including zeros. Force both comparator arms to that same count and preserve the candidate roster. One uses ordinary independent selection. The other uses the original independent addition ranking, with first-layer removal points preferred before lower-priority second-layer points. Within each layer use original removal probability and ordinal. Keep the original natural policy as a separate utility reference.

## Evaluation and decision

Report COCO mask AP, AP75 and AP90, candidate coverage, total and per-candidate realized edit counts, topology changes, original fixed-match continuous IoU with paired source-prefix intervals, added and removed overlap with the reference polygons, and strict-0.90 identity gains and losses. Use all five original folds and count all no-ops. First reproduce the historical rows before interpreting candidate effects.

For a follow-up topology experiment, require gains of at least 0.30 AP points and 0.20 continuous IoU points over both equal-count controls, a positive lower 95% IoU interval bound, AP75 at least as high as both controls and no additional original strict-0.90 losses. Preserve all outcomes and effect estimates. This study describes the FIT109 population exposed to detector training, with held-out sources for the point heads. Any effect requires a separate frozen validation before a broader claim.

