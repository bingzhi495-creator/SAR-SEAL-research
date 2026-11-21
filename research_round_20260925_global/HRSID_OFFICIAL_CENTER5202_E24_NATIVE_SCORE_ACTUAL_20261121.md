# Native RF-DETR on the official HRSID PNG test allocation

The unchanged center-aligned RF-DETR seed 5202 detector reached **54.433 mask AP** on the full official PNG test population. The detector was trained for 24 epochs and 17,640 updates on 2,937 official training images. No point head was applied to this export.

| Population | Images | Annotations | Mapped masks | AP | AP50 | AP75 | AP90 | Small AP | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full official allocation | 1962 | 5922 | 187796 | 54.433 | 88.211 | 62.190 | 4.417 | 52.823 | 61.694 |
| Exclude four empty raster targets | 1962 | 5918 | 187796 | 54.469 | 88.295 | 62.225 | 4.418 | 52.869 | 61.735 |
| Exclude three exact-identity images | 1959 | 5891 | 187498 | 54.511 | 88.288 | 62.300 | 4.451 | 52.900 | 61.767 |

Scores use COCO segmentation and at most 100 detections per image. The primary view retains all 5,922 annotation records and all images, including one without annotations. The first sensitivity excludes four annotations whose polygons rasterize to empty masks and retains the prediction stream. The second removes two exact decoded-image matches to the FIT pool and the higher-ID record of an exact duplicate pair within the test split. These views were frozen before the export. The small sensitivity differences quantify those exact records, with no inference about independence of acquisition scenes.

## Execution and independent verification

The initial run was interrupted after 1,134 complete, ordered image blocks. Its process exit value could not be recovered. An independently launched continuation checked three deterministic positions spanning the prefix, reproduced their blocks exactly and evaluated the 828 remaining images. This worker exited successfully. The label-free readback confirmed a single 1,962-image stream with 187,796 mapped masks and the original native ranks, scores, mask pixels and image identities. Only scoring opened the fixed GT. A second CPU implementation rebuilt all three COCO populations and agreed exactly on all 39 values, including AP90. Neither scoring implementation ran the detector or trained a model.

- [Aggregate output](results/HRSID_OFFICIAL_CENTER5202_E24_NATIVE_COCO_ACTUAL_20261121.json)
- [Verification summary](results/HRSID_OFFICIAL_CENTER5202_E24_NATIVE_SCORE_VERIFICATION_20261121.json)
- [Input audit](HRSID_OFFICIAL_PNG_TEST_INPUT_ACTUAL_20261121.md)
- [Scoring code](code/HRSID_OFFICIAL_BASELINE_SCORE_20261121/) and [independent verification code](code/HRSID_OFFICIAL_BASELINE_SCORE_READBACK_20261121/)
- [HRSID author release](https://github.com/chaozhong2010/HRSID)

Historical project checkpoint, seed and fusion selection used this allocation. The numbers describe the fixed current native endpoint on the historical official test split. Evidence for the boundary repair comes from the separate development comparisons. An evaluation of frozen boundary refinement on the same official test inputs would quantify a within-endpoint historical test effect. An independent target population remains necessary for a stronger generalization conclusion.
