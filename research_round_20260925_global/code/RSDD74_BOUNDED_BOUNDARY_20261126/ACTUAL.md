# Native boundary comparison against exact limited-distance re-pairing

The exact 74-image Instance-RSDD reserve and unchanged nested 42-image view retain original native matches, reference polygons, candidate confidence and edit counts. The four arms share the frozen RF-DETR 5202 detector and occupancy head 5217. The unedited prediction is N. The original distance-first writer is F. Global same-budget re-pairing under F's longest chosen edge is L. Independent signed selection is I. All 4,092 positive-budget candidates in the 7,346-slot stream preserve 30,841 additions and 30,841 removals per edited arm.

The program verified the original render and scoring receipts before opening the previously evaluated author TRAIN references. It reused exact masks, recovered original native raster 0.50 matches and reproduced the archived N, F and I boundary estimates within \(10^{-12}\) points. Parity included both widths, the parent and sensor-date bootstrap intervals, the original pretreatment non-saturated subsets and the all-author zero-delta sensitivities. All eligible matches retained interior foreground in both original masks at both widths. This evaluation made no detector forwards, wrote no masks and computed no AP.

| Cohort and width | Contrast | Mean | 95% parent interval | 95% sensor-date interval |
|:---|:---|---:|:---|:---|
| Full 74, 1 px | I minus L | −0.034 | [−0.373, +0.278] | [−0.380, +0.287] |
| | L minus F | +0.735 | [+0.207, +1.243] | [+0.298, +1.217] |
| | I minus F | +0.701 | [+0.180, +1.233] | [+0.231, +1.243] |
| Full 74, 2 px | I minus L | +0.056 | [−0.234, +0.328] | [−0.229, +0.362] |
| | L minus F | +0.561 | [+0.152, +0.992] | [+0.215, +0.988] |
| Nested 42, 1 px | I minus L | −0.090 | [−0.557, +0.345] | [−0.570, +0.353] |
| | L minus F | +0.825 | [+0.169, +1.453] | [+0.166, +1.453] |
| Nested 42, 2 px | I minus L | +0.089 | [−0.311, +0.477] | [−0.316, +0.475] |

All values are Boundary IoU percentage points. Both intervals use 4,096 paired draws with equal parent weighting. Width one is primary. The full cohort has 81 author hulls and 47 valid area-eligible fixed matches from 12 parents and 11 sensor-date blocks. I and L differ on 46 of those matches. The nested 42 images have 47 author hulls and 32 eligible matches from eight parents and eight sensor-date blocks. The all-author zero-delta one-pixel I minus L estimate in the full view is −0.073 points, with parent interval [−0.350, +0.153].

The primary result is inconclusive despite adequate non-saturated coverage. L improves one-pixel boundary overlap over F under both grouping conventions and in the nested sensitivity. The one-pixel evidence does not establish an advantage of I over L. The earlier full-cohort AP and continuous area IoU evidence for I against L reports different endpoints and remains valid. The comparison concerns a previously evaluated reserve under frozen detector and point weights.

[Preregistered scope](PRELOCK.md) · [Reproduction](REPRODUCE.md) · [Safe aggregate](RESULT_SAFE.json) · [Witnessed closure](TERMINAL_SAFE.json)

[Boundary IoU interval plot, editable SVG](boundary_effects.svg) · [Vector PDF](boundary_effects.pdf) · [Plot source](plot_boundary_effects.py)
