# FIT-only calibration of native endpoint pair dependence

The original group-held BCE predictions reproduced exactly at all 848 common positive-budget candidates from 447 images in 439 decoded groups. The greedy four-neighbor graph and the deterministic sign-matched rewired graph each contain 21,826 disjoint pairs. The control changed 21,774 edges (99.762%) and kept every paired vertex and sign-stratum count. All 439 groups supplied at least two pairs. The mean Manhattan distance was 1.000 pixel in the local graph and 15.745 pixels in the rewired graph.

Each fold fitted the two dependency coefficients on the other four folds' original OOF marginals. The three policies were then scored on the same held local pairs and reference labels. Lower four-class Brier values indicate better pair predictions. Point estimates weight decoded groups equally, candidates with pairs equally within each group, and pairs equally within each candidate.

| Author reference | Independence | Local fit | Rewired fit | Local minus independence [95% interval] | Local minus rewired [95% interval] |
|---|---:|---:|---:|---:|---:|
| PSeg-SSDD | 0.434380 | 0.376578 | 0.400548 | −0.057802 [−0.062654, −0.053064] | −0.023970 [−0.026480, −0.021450] |
| SPSD | 0.552664 | 0.463598 | 0.493451 | −0.089066 [−0.093668, −0.084253] | −0.029853 [−0.032219, −0.027353] |

On the held rewired pairs, the local coefficient increased Brier loss relative to independence under each author reference. The coefficient fitted on rewired pairs lowered Brier loss relative to independence. The geometry and Brier requirements in the [published protocol](../PRELOCK.md) passed. This qualifies a fixed source-development replay and leaves the effect on action choice and final masks to that experiment.

The comparison is conditional on the original FIT packet admission and the five existing group-held BCE fits. The dependency calibration did not repeat the BCE fits in a nested outer-fold design. This FIT execution accessed no source-development labels, transfer labels, historical test input or new detector forwards.

The [public statistical aggregate](FIT_AGGREGATE_SAFE.json) retains pooled and fold-level outcomes. The exact observed FIT result is bound by `4f80ed3a7d7792ec7a132db3c5203f8faaf47a38efb332259b3077cd29406f1f`, and its separately verified natural-exit terminal by `2741b4a4e7d9c2685499cb873eb4d0ae973df186db99a7d87f874e317076d376`. The worker exited with code 0 in 25.23 s. Individual endpoints, pair labels, original OOF predictions and fitted coefficient files remain in the coordinator workspace.
