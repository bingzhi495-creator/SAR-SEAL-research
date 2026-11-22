# FIT109 native rival-mask ranking pilot

DEV705 retrospective checks found that independent equal-count endpoint selection improves the matched vessel's overlap and also increases intrusion into nearby annotated vessels. The source-side hypothesis is that occupancy evidence from another credible native candidate can reduce this intrusion at the same edit count, while preserving the ability to repair the matched hull.

## Frozen intervention

Use D-FINE E24 FIT109, all 109 source groups and all 10,900 original top-100 masks. Pin the accepted native predictions, original point support and original labels. Replay all four old arms exactly, including their AP, counts, fixed matches and threshold transitions. Refit only the original point head in each of its five 200-step source-held-out folds with the original initialization, features and training. Each new arm reads the same held-out vector. There are no new detector forwards or updates.

Before reading the labels, compute a vector for all original addition endpoints. A rival is an original same-image candidate with a nonempty mask and score at least 0.50. Drop the query's own candidate and any candidate whose native intersection covers at least 50% of the smaller foreground. Use the maximum score among eligible other masks covering the addition cell, or zero. This is a confidence-weighted mask indicator rather than a calibrated instance probability. The overlap test is a fixed duplicate heuristic. It can also suppress true crowded objects.

- **Independent** ranks the original addition score and reuses the number of edits realized by the fixed-pair arm, exactly as in the historical comparison.
- **Native rival** uses addition score q × (1 − rival score).
- **Permutation** uses the same operation, with the original candidate's rival scores permuted within up to four equal-sized blocks of the order by held-out addition probability and original ordinal. Use NumPy PCG64 seeded with the first 16 little-endian bytes of SHA-256 of the literal `FIT109_NATIVE_RIVALS_Q_QUARTILES_V1|image_id|rank`. Here `image_id` and 1-based `rank` are decimal strings.

All arms keep the same original candidates, scores, feasible endpoints, cap and realized edit counts including zeros. The two new arms keep identical deletion ordinals to Independent and vary the selected additions only. Ties retain original endpoint ordinal. No threshold or coefficient is selected from FIT outcomes, DEV or TEST.

## Measures and decision

Measure COCO mask AP, AP75 and AP90 on the whole cohort and use the same 412 valid original fixed matches for geometry as in the point-head replay. The near subset includes every such match with continuous polygon distance at most four native pixels to another valid polygon in the same image. Measure added area intersecting the union of the other valid polygons after subtracting the matched polygon, and continuous IoU against the original matched polygon. The unchanged deletions cancel from every paired net-intrusion contrast. The primary near-subset estimate includes unchanged and unexposed objects. Average objects within source-prefix groups and weight groups equally. Record near coverage, truly exposed matches and groups, permutation-displaced penalty vectors, and actual ranking changes in every arm comparison.

A truly exposed match has at least one nonzero rival score among its available additions and a positive replayed edit count. The paired intervals require at least 12 such matches from at least eight source-prefix groups. Otherwise report coverage and descriptive effects without confidence bounds. Use 4,096 paired source-prefix draws with PCG64 seed 5217. For each of Independent and Permutation, require a strictly positive lower 95% bound on control minus Native-rival intrusion area, a lower 95% bound above −0.20 points for Native-rival minus control own IoU, and a whole-cohort AP loss of less than 0.30 points. Advance the hypothesis only if all conditions hold against both controls. Report losses and intervals regardless of whether the gate passes. The preserved D-FINE continuous 0.90 threshold includes equality.

The detector saw FIT109. Only the point head holds out sources. The native score rule can mistake duplicated or overlapping predictions for different vessels. A positive result would justify a separate frozen development evaluation with a second detector. It would not establish novelty or deployment value by itself.
