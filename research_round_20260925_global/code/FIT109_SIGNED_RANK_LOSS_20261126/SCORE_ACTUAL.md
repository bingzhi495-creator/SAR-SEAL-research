# FIT109 repeated within-side ranking loss

The matched experiment completed on the original 109 HRSID FIT images. The frozen detector had seen these source images. Only the point heads were trained and evaluated across five group-disjoint folds. For each of seeds 5217, 5218 and 5219, the calibration BCE and auxiliary within-side ranking heads used matched initializations, features, candidate draws and 200 updates per fold. They had identical per-candidate write budgets from the archived legacy distance-first selector.

The complete output contains 10,488 original candidate slots and 6,421 supported candidates. Each writer made 46,581, 46,888 and 46,925 additions and the same number of removals for the three respective seeds. All-hull discrete IoU uses 496 hulls and 109 prefixes, including 43 unmatched hulls with contribution zero. Fractional-occupancy IoU uses the same frozen native matches for all arms and 392 supported anchors across 92 prefixes. The intervals use paired bootstrap over prefixes with 4,096 draws after seed averaging.

| Comparison | Mask AP difference by seed, points | Mean AP difference, points | Fixed-match fractional IoU difference, points (95% interval) |
|---|---|---:|---:|
| BCE + rank vs matched BCE | +0.019 / +0.083 / +0.120 | +0.074 | +0.013 [−0.007, +0.033] |
| BCE + rank vs archived independent | +0.012 / +0.097 / +0.090 | +0.066 | +0.025 [−0.013, +0.059] |

The changed-anchor coverage requirement passed for both comparisons in all three seeds. Both comparisons fell short of the prespecified +0.30-point AP requirement and the fractional-IoU intervals crossed zero. Both were inconclusive under the directional gate, so the loss did not advance. The secondary prefix-equal raster IoU contrast against archived independent selection was +0.034 [0.002, 0.066] points. The matched BCE comparison on that metric was +0.003 [−0.019, 0.026]. Evaluation is limited to held source groups for the point stage.

## Execution receipts and recovery

- The first renderer stopped at exact posterior reconstruction using a different Torch version. Renderer and independent readback completed with CPU Torch 2.10.0+cu130, the same build and per-candidate arithmetic as point training. Whole edit-ledger bytes matched.
- The first scoring process stopped at a local counter initialization defect. Its immediate retry stopped at the legacy metric guard because the annotation adapter omitted the prior COCO projection of the author annotation ID zero to a fresh positive ID. The canonical rescoring worker uses the original published projection and maps the synthetic identifier back to the raw ID for fractional anchor identity. A separate legacy-only run verified all N/F/I COCO fields exactly and all legacy discrete IoUs within 1.43e−14 points before final scoring. Neither failed attempt produced a scored result.
- Final scoring observed natural child exit zero and verified the aggregate footer digest. The achieved gate and phase hashes are in the corresponding safe JSON receipts. Private labels, masks, edits, weights and worker logs are excluded.
