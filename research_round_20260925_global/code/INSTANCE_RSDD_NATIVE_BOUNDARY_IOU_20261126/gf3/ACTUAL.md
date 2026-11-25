# Gaofen-3 native Boundary IoU on the frozen Instance-RSDD comparisons

The two frozen detectors were evaluated on the same previously used 583 Gaofen-3 images and 631 author polygons from 76 acquisitions. Image identities, original image filenames, decoded pixels and acquisition groups agreed. Each detector retained its own proposals, scores and original nonignored COCO mask IoU 0.50 matches. There were 629 and 628 original matches, with 425 eligible in each and no empty author rasterizations. We used both previously frozen point heads for 5202 and all three for 5203. Original native COCO metrics, absolute continuous IoU and each head's paired continuous-IoU interval were reproduced. Every compared edit preserved the original candidate's signed budget and zero-quota behavior.

| Frozen detector | Eligible acquisition groups | One-pixel Boundary IoU gain [95% interval] | Two-pixel sensitivity | One-pixel prespecified criterion |
|---|---:|---:|---:|---|
| 5202 | 68 | +0.466 [+0.223, +0.702] | +0.467 [+0.278, +0.653] | Pass |
| 5203 | 67 | +0.177 [−0.025, +0.376] | +0.255 [+0.107, +0.404] | Inconclusive |

Values are points (0–100) for independent selection minus distance pairing. Head effects were averaged within each matched hull, then hulls within each acquisition, then acquisitions with equal weights. Intervals use the paired 4,096 acquisition bootstrap draws. At both widths, all 425 eligible hulls for each detector had interiors in the author and native masks. The prespecified non-saturated subset therefore gives the same effects. The all-author sensitivity sets each unmatched or ineligible hull's paired change to zero. Its one-pixel effects were +0.310 [+0.143, +0.479] and +0.154 [+0.016, +0.301] points, respectively. It retains a separate denominator and does not change either primary decision.

One-pixel boundary overlap improves on carrier 5202. Carrier 5203 shows a positive point estimate with an interval spanning zero at the prespecified width. Its two-pixel effect and original continuous overlap effect are positive. These results characterize the observed short-range contour behavior on the same author dataset cohorts. Contribution scope stays at the frozen endpoint decision and does not add independent acquisitions or a new training result.

The observed worker exited 0 in 43.63 seconds, with verified footer and empty error output. The worker reported 43.13 seconds. Five generated tests passed. No optimizer or detector forward ran. The report and output receipt contain only aggregate measurements and artifact identities.

