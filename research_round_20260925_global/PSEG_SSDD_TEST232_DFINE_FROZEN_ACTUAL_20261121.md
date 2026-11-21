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

## Matched geometry and cost

Native COCO matching at 0.50 recovers 402 of 546 annotations. The geometric comparison keeps the 293 original matched pairs inside the frozen editing-area range and weights the 143 contributing decoded images equally. With the same edit count, independent selection improves continuous IoU by **0.106 points** over fixed pairing (image-paired 95% interval 0.055 to 0.157). The change is **0.109** (0.058 to 0.162) after excluding the ten historical duplicates. The natural policy improves continuous IoU by **0.228** points over raw (0.104 to 0.364); false-positive and false-negative area each fall by 0.143 points of the true polygon area. At continuous IoU 0.75, eight matches cross upward and two cross downward. None of the 293 fixed pairs reaches 0.90 under any arm. These post-hoc intervals are conditional on the evaluated images, whose acquisition-scene relationships remain unverified. [Geometry results](results/PSEG_TEST232_PAIRED_GEOMETRY_SAFE_20261121.json)

Area-preserving edits can change connectivity. Among the 6,404 equally edited masks, the number with more holes than their original masks drops from 1,739 with fixed pairing to 1,169 with independent selection. At the larger natural count, the independent policy increases hole count in 1,544 of 10,466 edited masks, while the same-count native-probability control does so in 519. Topology uses 8-connected foreground and 4-connected background. These figures count predicted masks with any increase and do not assign correctness to individual holes. [Topology results](results/PSEG_TEST232_DFINE_EDIT_TOPOLOGY_SAFE_20261121.json)

With images, candidate lists and probability planes already in memory, the natural point stage adds **181.2 ms per image on average** versus native top100 pass-through on an i7-13700KF single thread (median 186.5 ms, 95th percentile 331.4 ms). Each image uses one untimed warm-up pair and two timed pairs in alternating order. Scores, ranks and RLE match the sealed outputs on every pass. The original GPU forward and postprocessing took 11.74 seconds over 232 images in a separate run. Timing excludes input reads and decoding, and the CPU host concurrently ran the HRSID evaluation. [Latency results](results/PSEG_TEST232_DFINE_CPU_ADDON_COST_SAFE_20261121.json) · [Independent arithmetic check](results/PSEG_TEST232_DFINE_CPU_ADDON_COST_INDEPENDENT_20261121.json)


A numerically equivalent sampler replaces per-query conversion of the entire 160 by 160 plane with FP16 conversion of the four interpolation neighbors and reusable per-image coordinate maps. In a separate same-run paired test on all 232 images, CPU point-stage mean latency fell from **158.8 to 104.3 ms per image**, a **34.3%** reduction. The 95th percentile fell from 325.5 to 180.8 ms. All 3,297,531 sampled FP16 probabilities matched bit for bit. Ranks, scores and compressed masks matched the frozen implementation on all 23,200 candidate slots. This covers the stage after resident input capture, in a separate run from the above native-pass-through timing. [Paired acceleration result](results/PSEG_TEST232_DFINE_EXACT_COST_ACCELERATION_SAFE_20261121.json) · [Independent timing recomputation](results/PSEG_TEST232_DFINE_EXACT_COST_ACCELERATION_INDEPENDENT_20261121.json)

## Provenance

All 232 JPEGs and the original test JSON match the file lengths and Git blob hashes of the pinned third-party mirror. Direct author-hosted byte authentication remains unresolved. The initial GPU adapter stopped before any image forward because it rejected an extra provenance field in the already pinned line list. The revised adapter accepts that audited field, preserves the exact image list and did 232 forwards without training. All 54,976 post-NMS binary masks passed the original author comparison; all 23,200 top100 probability records and the five treatment streams passed subsequent closed native-size, rank and edit-support checks. Independent COCO reconstruction obtained exactly the same 130 metrics and 104 paired contrasts. The historical split participated in earlier project evaluation. Source-scene overlap is unresolved.

- [Frozen protocol](PSEG_SSDD_FROZEN_TEST232_PRELOCK_20261121.md)
- [Score output](results/PSEG_SSDD_TEST232_DFINE_FROZEN_SCORE_ACTUAL_20261121.json)
- [Independent scoring summary](results/PSEG_SSDD_TEST232_DFINE_SCORE_INDEPENDENT_20261121.json)
- [DEV187 frozen result](DFINE_HRSID_TO_PSEG_SSDD_DEV187_FROZEN_TRANSFER_20261120.md)
- [Original data release](https://github.com/TianwenZhang0825/Official-SSDD)
