# Frozen-order quota fitting and DEV705 evaluation

Train one deployable allocation policy on the closed HRSID FIT109 candidate action features and the exact source-only counterfactual AP targets. The point heads for their own source groups are the existing first-level held-fold S fits. Both the detector and the point head remain frozen. Previous use of FIT by the detector limits the training interpretation. The decisive endpoint is the complete separate DEV705 development cohort, with no DEV annotations in target construction, normalization, regression, control selection, or new mask preparation. DEV705 itself has prior project exposure. The sole prespecified initial point-head seed is 5217 on center5202 E24. [Target protocol](code/RFDETR_FIT109_QUOTA_COMPETITION_TARGETS_20261123/PROTOCOL.md) · [Earlier quota result](RFDETR_FIT_QUOTA_OUTCOME_PROTOCOL_20261109.md)

For each candidate, fix the original selected endpoints and their two posterior rankings. Take the leading a additions and 2k−a deletions at every feasible a, where balanced SU already realized k per side. Candidate scores, order, boxes, k and the total XOR stay unchanged in every quota arm. Include the balanced action, tied utilities choose nearest to k, then smaller a. Unsupported candidates retain their original masks.

Use the same 25 action features in every learned arm. Take the five existing balanced-zero features in the verified source action archive (signed and absolute area change, predicted addition and removal masses, and the conditional IoU-ratio change). Form their products with [1, log(native area)/log(1024), detection score, expected net foreground correction divided by native area, k/8]. Compute one shared source-only weighted RMS over all nonbalanced legal actions, with unit scale for a zero-RMS column and no centering. Fit a no-intercept FP64 ridge with weighted mean squared error plus 0.05 times the squared scaled coefficient norm, with weights equal by source group, seed within group, candidate within seed, and nonbalanced action within candidate. Omit single-action no-ops and the exact zero balanced row from the regression, reporting them separately. Keep every other zero target. Use one solve per learned arm and no hyperparameter selection.

| Arm | Decision |
| --- | --- |
| N | Native detector mask |
| SU | Original equal-count, area-preserving independent writeback |
| J | Maximize 2 times estimated overlap change minus signed area change from the same prefixes |
| R | Maximize the existing source feature's conditional IoU-ratio change |
| C | A constant addition offset chosen in [−8,+8] by weighted source AP target and clipped to each candidate's legal counts. Tie by smaller absolute offset then smaller offset. |
| A | Predict mean exact unilateral AP change on the complete source support |
| M | Predict the same AP target using only original valid geometry-matched candidate rows |
| G | Predict original continuous matched overlap change using precisely the same rows, weights and features as M |
| LU=S and LW=S | Original larger-head controls at SU's exact per-candidate total edit count |

Report the original N and SU numerical replay, each arm's actual edits, allocation, changed area, realized cost, 13 official COCO metrics, and the original sensitivity excluding the four empty annotations. On the 1,461 original valid native matches, report matched polygon IoU, truth-normalized FP/FN and strict counts. Use 4,096 PCG64(5217) paired bootstraps over the same 27 source-prefix groups for geometry contrasts. M−G isolates change of target on the same source rows. A−M changes supervision coverage, with equal model capacity, feature coordinates and output constraints. A−SU tests net policy value. A versus the strongest prespecified N, SU, J, R, C, M, G, LU=S and LW=S is the practical decision. The strongest reference is the maximum complete DEV705 primary AP among them, with all reported regardless of rank.

Require at least +1.0 AP point of A over that reference to enter full seed replication. A strictly positive gain under 1 point allows at most one prespecified follow-up seed 5218 under this same recipe. A nonpositive gain closes extension of this fit after verified execution and defect triage. Record all achieved scope, confidence intervals and negative results. The guard is a project decision, with no implied journal criterion. No official HRSID test, PSeg-SSDD or Instance-RSDD evaluation in this initial stage. Do not use the earlier external results for parameter choices.
