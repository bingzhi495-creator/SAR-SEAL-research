# Frozen HRSID D-FINE transfer on PSeg-SSDD TEST232

Native mask AP increases from **17.974 to 18.319** on the full official allocation. The paired gain is **0.344 points**. Excluding all ten test images with byte-exact copies in the project's historical SSDD FIT741 population leaves 222 images and 502 annotations. On that sensitivity population, AP increases from **18.806 to 19.194** (+0.389 points). The D-FINE detector, 1,321-parameter point head, scaler, thresholds and edit policy were carried unchanged from HRSID.

| Mask policy | AP | AP75 | AP90 | Small AP | AR100 | AP, 222 images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw D-FINE | 17.974 | 13.015 | 0.110 | 16.348 | 36.136 | 18.806 |
| Same head, fixed pairing | 17.995 | 13.293 | 0.149 | 16.379 | 36.264 | 18.844 |
| Independent, fixed-policy count | 18.076 | 13.449 | 0.149 | 16.460 | 36.355 | 18.944 |
| Independent, natural count | 18.319 | 14.705 | 0.149 | 16.756 | 36.612 | 19.194 |
| Native ranking, natural-policy count | 18.028 | 13.358 | 0.110 | 16.394 | 36.227 | 18.884 |

All scores are scaled by 100. The full view retains 232 images, 546 original float-polygon annotations, and 23,200 native predictions including empty masks. The excluded-image view retains 22,200 predictions. The native class is consistently mapped to the original category-zero annotation. The five arms share the same detector forward stream, native boxes, ranks and confidences.

## Matched effects

- With the head and realized edit count held fixed, independent selection improves AP by **0.081** over fixed pairing on all 232 images, and by **0.100** on the 222-image sensitivity. Both matched policies swap 29,304 pairs of pixels across 6,404 edited masks on the full view.
- The natural policy swaps 55,040 pairs of pixels across 10,466 masks. Its gain is **0.323** AP over fixed pairing and **0.290** over native ranking with the natural policy's count. Relative to the raw detector, its AP75 gain is **1.690** points. AP90 remains low. The gain over the raw detector falls below the previously used 0.50-point reference, while the fixed-pair contrast is slightly above the 0.30-point reference.
- The result follows the direction and approximate magnitude of the DEV187 frozen transfer. The earlier DEV187 evaluation found +0.336 AP over raw and +0.071 AP for matched independent selection over fixed pairing. The four contrasts here describe realized effects in historically exposed scopes.

## Provenance

All 232 JPEGs and the original test JSON match the file lengths and Git blob hashes of the pinned third-party mirror. Direct author-hosted byte authentication remains unresolved. The initial GPU adapter stopped before any image forward because it rejected an extra provenance field in the already pinned line list. The revised adapter accepts that audited field, preserves the exact image list and did 232 forwards without training. All 54,976 post-NMS binary masks passed the original author comparison; all 23,200 top100 probability records and the five treatment streams passed subsequent closed native-size, rank and edit-support checks. Independent COCO reconstruction obtained exactly the same 130 metrics and 104 paired contrasts. The historical split participated in earlier project evaluation. Source-scene overlap is unresolved.

- [Frozen protocol](PSEG_SSDD_FROZEN_TEST232_PRELOCK_20261121.md)
- [Score output](results/PSEG_SSDD_TEST232_DFINE_FROZEN_SCORE_ACTUAL_20261121.json)
- [Independent scoring summary](results/PSEG_SSDD_TEST232_DFINE_SCORE_INDEPENDENT_20261121.json)
- [DEV187 frozen result](DFINE_HRSID_TO_PSEG_SSDD_DEV187_FROZEN_TRANSFER_20261120.md)
- [Original data release](https://github.com/TianwenZhang0825/Official-SSDD)
