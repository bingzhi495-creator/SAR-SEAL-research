# Same-count relocation of SegRefiner-LR deletion positions

## One question

The completed Instance-RSDD signed analysis finds adverse deletion in both contexts of the stock writer. Here we test whether selecting shallower positions improves its specific deletion set when the number of added and removed cells remains unchanged. The operation is a geometric control of a frozen binary writer. Keep the same author COCO checkpoint and eligibility gate, stock additions, detector candidates, confidences, original grid, matches and earlier groups. This method has no new trainable parameters, image features, calibration or extra model forwards. Instance-RSDD is retrospective and the protocol precedes inspection of this intervention and the PSeg stock metrics.

## Frozen policies

Use exactly the admitted raw mask M and stock output S. A = S minus M and D = M minus S. Hold A and d = |D| for every candidate. If d is zero, both relocation arms equal S.

Periphery policy P chooses d native foreground cells ordered by increasing squared Euclidean distance to the closest original background cell, including a one-cell pad outside the image. Break depth ties by increasing squared distance to D, which first preserves existing deletions inside that shell. Break any remaining tie by native row-major pixel order. Its output is (M minus P) union A.

Let C = P intersect D, E = D minus C, k = |E|. The proximity-control deletion set Q keeps exactly C and fills the remaining k places from M minus D in increasing squared distance to E. Break distance ties by native row-major pixel order. Its output is (M minus Q) union A. Thus it moves exactly the same count of deletions, preserves the same old deletion support, and prioritizes locality without the depth criterion. In both policies d and A are checked exactly, before the scorer opens labels. If k is zero, both arms equal stock. Do not exclude unchanged or area-gate bypassed candidates.

Retain raw M and stock S with exactly the original COCO and fixed-match geometric metrics, as verified against the qualified score. The head, writer, and all masks in the closed stock run are immutable.

## Evaluation and threshold

Evaluate the four arms for 74/42 Instance-RSDD and 232/222 PSeg-SSDD under the existing all-candidate COCO and original fixed .50 geometry protocols. Geometry uses the same 4,096 paired PCG64 5217 draws and previous aggregation over parent scenes and sensor-date blocks on Instance-RSDD, or decoded-image groups on PSeg. Inspect the complete fixed matches and separately its exact original 512-gate-exposed subset. Only use PSeg after original stock and signed results close and qualify. Preserve the nested sensitivity labels.

Record all four absolute AP/AP75/AP90/small AP/AR100 scores, original/independent .50 coverage, number of writes and changed masks, shifted deletions and unchanged budgets, gate-exposed hull and grouping support, IoU intervals, and continuous FP and FN area per truth area. The two primary contrasts are periphery minus stock and periphery minus proximity control. A contrast passes if full-candidate mask AP improves by at least 0.30 points, the lower paired continuous-IoU interval is positive in the gate-exposed original geometry subset, and it changes at least 20 such hulls across at least eight original clusters. Both contrasts are required in a complete cohort to advance the geometric prior claim. Replication requires both complete cohorts. Direction, reduced support and negative results are reported without reselection. The proximity control fixes the surviving author deletes and relocation size, so this comparison identifies location choice at the given deletion budget, not the global benefit of choosing that budget.

The conditioning evidence and original authored baseline are the [signed component analysis](SEGREFINER_SIGN_ACTUAL_20261123.md) and [stock transfer](SEGREFINER_STOCK_ACTUAL_20261123.md).
