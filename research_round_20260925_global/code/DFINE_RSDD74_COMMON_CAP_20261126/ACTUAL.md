# D-FINE equal-budget re-pairing on the Instance-RSDD reserve

The frozen HRSID-trained D-FINE E24 detector and independently accepted 1,321-parameter FIT109 occupancy head were replayed on all 74 historical reserve chips and the original nested 42-chip view. F is the original distance assignment followed by positive-gain selection. L chooses the exact best k pairs under the maximum displacement inherited from F. I independently ranks the two sides at the same k. Within D-FINE the three writers share all 7,400 original candidate slots, scores, signed support and realized edit counts.

There are 2,614 positive-budget candidates. All three edited arms add exactly 12,037 native cells and remove exactly 12,037. The fixed and independent masks were checked against the original D-FINE readout while the same original binary32 endpoint scores drove the bounded solver. A separate CPU process repeated the entire label-free replay and matched the complete tape byte for byte. The solver enforced its exact predicted-gain ordering and native squared-distance cap. The 2026-11-26 UTC render and replay completed in 23.25 and 23.00 seconds. The scorer then confirmed the signed masks again and naturally exited after 5.12 seconds. Each observer verified the output footer.

| View and contrast | Mask AP difference, points | Continuous matched IoU difference and 95% block interval, points | Changed eligible hulls / sensor–date blocks |
| --- | ---: | ---: | ---: |
| Full, I − L | +0.344 | +0.133 [+0.018, +0.251] | 42 / 11 |
| Full, L − F | +0.253 | +0.252 [+0.151, +0.362] | 43 / 11 |
| Full, I − F | +0.597 | +0.385 [+0.213, +0.567] | 43 / 11 |
| Nested, I − L | +0.604 | +0.151 [+0.021, +0.290] | 30 / 8 |
| Nested, L − F | +0.000 | +0.225 [+0.114, +0.351] | 31 / 8 |
| Nested, I − F | +0.604 | +0.377 [+0.179, +0.570] | 31 / 8 |

On the full reserve, AP is 48.551, 48.673, 48.926 and 49.270 for the native, F, L and I masks. The original D-FINE native 0.50 assignment covers 78 of 81 author hulls. Of these, 44 masks meet the continuous geometry eligibility in 11 represented parents and sensor–date blocks. The nested view covers 47 of 47 author hulls, with 32 eligible masks from eight parents and blocks. The paired all-author continuous-IoU sensitivity for I−L is +0.074 [+0.005, +0.151] points on the full reserve and +0.107 [+0.002, +0.213] on the nested view. The sensitivity assigns zero effect where the original native masks cannot participate.

The full I−L contrast satisfies the prespecified AP, continuous IoU and coverage conditions. The nested contrast has the same direction and retains its status as a related sensitivity. L−F has positive continuous effects, with AP under the +0.30-point condition in both views. At one native pixel, full I−L Boundary IoU changes by +0.256 [−0.176, +0.697], while full L−F changes by +0.639 [+0.271, +1.052]. The two-pixel I−L interval also crosses zero. The observed independent-selection effect is supported in AP and continuous area geometry on these fixed D-FINE detections. Its narrow Boundary IoU result remains uncertain.

The earlier positive RF-DETR reserve result and this result come from independent source-trained detector-and-point-head pipelines evaluated on the same historical target cohort. The comparison is a second-carrier sensitivity. The two pipelines share target acquisitions and change the detector and head together. Prospective acquisition evaluation remains separate. No optimizer updates or target adjustment occurred. New detector forwards occurred only in the separately admitted 74-image frozen capture.

[Protocol](PRELOCK.md) · [Replay](REPRODUCE.md) · [Frozen render](RENDER_SAFE.json) · [Independent replay](VERIFY_SAFE.json) · [Aggregate metrics](SCORE_SAFE.json)

[Editable pipeline comparison](full_reserve_carrier_effects.svg) · [Vector PDF](full_reserve_carrier_effects.pdf) · [Plot source](plot_carrier_effects.py)
