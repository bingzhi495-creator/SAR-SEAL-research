# Native Boundary IoU on the Instance-RSDD 74-image reserve

Independent selection improves 1-pixel Boundary IoU by **0.701 points** over fixed pairing at the same realized add and remove counts. The parent-paired 95% interval is [0.180, 1.233]. The improvement remains positive at 2 pixels and in the nested date-separated view. The separate occupancy-versus-PointRend comparison is positive at 1 pixel, with an interval that crosses zero on the full reserve.

The frozen reserve contains 74 images and 81 author hulls across 13 parent scenes. All hulls reproduced their original native COCO association at raster IoU 0.50. The primary analysis contains the 47 fixed matches passing the original 32–1,023 native-pixel area rule, across 12 parents and 11 sensor-date blocks. The date-separated view contains 42 images, 47 author hulls and 32 primary matches across eight parents. It remains nested within the 74-image reserve.

The scorer binds the closed native stream, same-count side-factorial replay, and independently reproduced PointRend replay. The 7,346 original candidate masks, identities, ranks and scores agree between both comparisons. Fixed and independent selection exchange 30,841 pairs on the same 4,092 candidates. Occupancy and PointRend exchange 31,391 pairs on the same 4,114 candidates. Both comparisons preserve their own support and signed quotas. The scorer reproduces 104 archived COCO metric values per view before computing Boundary IoU.

| Cohort | Width | Comparison | Mean difference | Parent-paired 95% interval | Sensor-date block 95% interval |
|---|---:|---|---:|---:|---:|
| 74 images | 1 px | Independent − fixed pairing | +0.701 | [0.180, 1.233] | [0.231, 1.243] |
| 74 images | 2 px | Independent − fixed pairing | +0.617 | [0.236, 1.001] | [0.285, 1.013] |
| 74 images | 1 px | Occupancy − PointRend | +0.829 | [−0.056, 1.750] | [−0.067, 1.879] |
| 74 images | 2 px | Occupancy − PointRend | +0.801 | [0.285, 1.370] | [0.247, 1.400] |
| Nested 42 images | 1 px | Independent − fixed pairing | +0.736 | [0.076, 1.436] | [0.059, 1.427] |
| Nested 42 images | 2 px | Independent − fixed pairing | +0.796 | [0.378, 1.218] | [0.363, 1.230] |
| Nested 42 images | 1 px | Occupancy − PointRend | +1.397 | [0.245, 2.420] | [0.288, 2.429] |
| Nested 42 images | 2 px | Occupancy − PointRend | +0.931 | [0.098, 1.713] | [0.136, 1.720] |

Each value is a percentage-point difference. Fixed-width Boundary IoU intersects and unions the inward boundary bands on the original label raster, using the same anchor for every writer. There was no whole-mask band saturation at the two reported widths. The all-author sensitivity gives out-of-area hulls zero change and weights all represented parents equally. On the 81-hull reserve the 1-pixel estimates are +0.508 [0.104, 0.962] and +0.698 [0.056, 1.386], respectively. The primary full-reserve 1-pixel occupancy-versus-PointRend interval remains inconclusive.

This retrospective boundary audit supports a same-count writer effect on the original frozen reserve. It uses pre-existing author annotations and has zero detector forwards and optimizer updates. The fixed pixel widths are specified in the [study protocol](../PRELOCK.md). The boundary-overlap definition is adapted from [Cheng et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html).

The verified [aggregate result](RESULT_SAFE.json) has SHA-256 `a8cae7278bdd3d5485fc6e539632be2e80e0d27615a3e127803fd34477d14c3a`. The final worker and observer exited naturally with code 0 in 3.90 seconds. The terminal SHA-256 is `67ef5f0f99c840c36ce00d378d1dbc776ad83dcc48c12f356edef659665e3499`. The first worker also exited naturally with code 0. Its observer flagged the absent phase in the result envelope. The archived retry added that field only, and all scientific result fields matched the first run exactly.
