# Frozen SRNet point-prior ablation

The two direct inference ablations reproduced all original fixed and independent write bytes. Both added and deleted 360 cells across the same 45 eligible positive-budget candidates in the complete 74-chip evaluation cohort. The independent second replay produced the same edit bytes. COCO metrics for all three original arms reproduced exactly and original continuous geometry reproduced within `1e-12`. This is a retrospective diagnosis on the author-train reserve. The nested 42-chip view is a fixed sensitivity.

P selects endpoints with the FP32 sigmoid of the clipped native logit. R uses the FP32 sigmoid of the frozen network residual read directly before addition. I uses the FP32 sigmoid of their sum. The residual still receives the standardized native probability as an input feature. Every arm uses the original endpoint support, ordinal tie handling, contour area and per-candidate signed quota from the original fixed writer F. The 89 candidate slots and their detection confidence/order stay fixed.

| View | Contrast | Mask AP change (pp) | Continuous IoU change with blocked 95% interval (pp) | Changed anchors / parents / blocks |
| --- | --- | ---: | ---: | ---: |
| 74 chips | I − P | +0.045 | +0.192 [+0.091, +0.290] | 32 / 13 / 12 |
| 74 chips | R − I | −0.163 | −0.029 [−0.072, +0.017] | 32 / 13 / 12 |
| Nested 42 | I − P | +0.090 | +0.196 [+0.048, +0.347] | 22 / 8 / 8 |
| Nested 42 | R − I | −0.218 | −0.012 [−0.055, +0.034] | 22 / 8 / 8 |

On the full cohort, I − P reduced both FP and FN polygon-cell area by 0.132 pp of truth area. Its continuous interval is positive in both views, and the mask AP difference remains below the 0.30-pp directional criterion. Removing the direct additive prior from the frozen head lowered AP in both views and yielded intervals for the continuous change that include zero. Both predeclared directional gates were inconclusive. No detector forward, fitting, threshold tuning or model selection took place. The renderer saw zero reference files. The scoring stage used the original author train labels under the closed prior evaluation scope. [Prelock](PRELOCK.md) · [Aggregate score](PRIOR_ABLATION_SCORE_SAFE.json) · [Reproduction](REPRODUCE_PRIOR_ABLATION.md)

