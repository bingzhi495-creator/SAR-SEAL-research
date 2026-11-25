# PointRend crossover on the Instance-RSDD reserve

The frozen RF-DETR detector, its original occupancy head and the accepted PointRend adaptation were evaluated on the existing 74-chip reserve and nested 42-chip sensor–date view. All six edited arms inherit the original occupancy F count and maximum realized single-edge displacement per candidate. Each arm makes 30,841 additions and 30,841 removals across the complete 7,346-candidate stream. The paired geometry denominator is 47 original eligible matches in 12 parents and 11 sensor–date blocks for the full reserve, and 32 in eight parents and eight blocks for the nested view. The original adapted PointRend replay was separately reproduced at its older 31,391-swap budget.

| View and contrast | Pooled mask AP difference, points | Fixed-match continuous IoU difference and 95% block interval, points | Changed hulls / blocks |
| --- | ---: | ---: | ---: |
| Full, PointRend I − L | +0.095 | −0.037 [−0.172, +0.086] | 46 / 11 |
| Full, occupancy I − L | +0.479 | +0.116 [+0.008, +0.242] | 46 / 11 |
| Full, writer × score interaction | +0.384 | +0.153 [−0.008, +0.339] | 46 / 11 |
| Nested, PointRend I − L | +0.271 | −0.104 [−0.235, +0.050] | 32 / 8 |
| Nested, occupancy I − L | +0.890 | +0.132 [−0.020, +0.299] | 32 / 8 |
| Nested, writer × score interaction | +0.619 | +0.236 [+0.056, +0.434] | 32 / 8 |

I is independent selection and L is exact maximum-displacement constrained re-pairing. The interaction is occupancy I−L minus PointRend I−L. Intervals use shared 4,096 paired sensor–date draws and equal parent weights. AP differences are pooled point effects.

The full PointRend contrast reached the required support but failed the prespecified direction criterion. The full-reserve interaction interval crosses zero. The nested result favors the occupancy scorer in continuous overlap and remains a sensitivity within the same acquisition cohort. At one and two pixels, all PointRend I−L Boundary IoU intervals cross zero in both views. The original occupancy result over bounded pairing was reproduced. This experiment supports reporting the observed endpoint advantage with its tested occupancy scorer and does not establish the same effect under the PointRend adaptation.

The independent label-free verifier reconstructed each full RLE from original signed support, checked complete native candidate coverage, same original F row indices and cap, and solved perfect matching feasibility separately for both bounded arms. It observed natural exit zero and verified the footer in 6.25 s. The scoring phase observed natural exit zero and verified its footer in 11.89 s. It recalculated original author COCO metrics and all learned fixed-geometry contrasts before releasing the new comparison. The archived native Boundary IoU point estimates and sensor–date intervals reproduce. Parent-only intervals use the parent order of the crossover scorer for the resample array. Both widths and interval conventions are separately available in the safe score. No new detector inference or updates were used.

[Protocol](PRELOCK.md) · [Verification receipt](VERIFY_SAFE.json) · [Score](SCORE_SAFE.json)

[Editable crossover figure](writer_score_crossover.svg) · [Vector PDF](writer_score_crossover.pdf) · [Plot source](plot_crossover.py)
