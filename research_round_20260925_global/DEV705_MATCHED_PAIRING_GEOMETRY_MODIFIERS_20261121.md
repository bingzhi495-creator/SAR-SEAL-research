# Vessel width and neighboring-instance overlap under equal-count boundary edits on HRSID DEV705

Independent selection of pixels to add and remove reproduced the original continuous IoU gain over fixed endpoint pairing on both detectors. The comparison retained each candidate's original match at raster IoU 0.50 and its edit count. The RF-DETR contrast was SU=F minus SF, with three point-head seeds averaged within each object. The D-FINE contrast was `free_matched_count` minus `fixed`. Both historical source-group means were reproduced to within 10⁻¹² IoU points.

| Original fixed-match comparison | RF-DETR | D-FINE |
|---|---:|---:|
| Valid matches / images / source prefixes | 1,461 / 652 / 27 | 1,447 / 646 / 27 |
| Reproduced IoU gain, points [95% interval] | +0.639384 [0.547752, 0.743265] | +0.529129 [0.455059, 0.596598] |
| Change in gain per doubling of width at fixed area, points [95% interval] | +0.239 [−0.099, 0.592] | +0.061 [−0.225, 0.351] |
| Matches with width ≤4 pixels / represented prefixes | 4 / 4 | 2 / 2 |

Width was the minor side of the minimum-area rectangle around the original reference polygon. We regressed paired IoU gain on log₂ width and log₂ continuous reference area, weighting source prefixes equally and objects equally within each prefix. The 95% percentile intervals use the same 4,096 paired prefix bootstrap draws for both detectors. Both width intervals include zero. Only four RF-DETR matches and two D-FINE matches had width ≤4 pixels. Original reference area was below 1,024 pixels² for 1,448 RF-DETR and 1,435 D-FINE matches, and neither cohort contained a reference of at least 9,216 pixels².

For targets within four pixels of another valid annotated vessel, independent selection improved the matched vessel's IoU while increasing the edited foreground area inside neighboring vessels. For each edit, we measured added minus removed overlap with the union of other valid same-image polygons after subtracting the matched polygon. The table gives independent minus fixed differences, with source prefixes weighted equally.

| Nearest other vessel ≤4 pixels | RF-DETR | D-FINE |
|---|---:|---:|
| Matches / images / source prefixes | 46 / 26 / 14 | 47 / 25 / 14 |
| Matched vessel IoU gain, points | +0.799 | +0.394 |
| Net change in foreground inside other vessels, pixels² | +0.480 | +0.393 |
| Matches with increased / decreased / unchanged neighbor overlap | 15 / 5 / 26 | 17 / 6 / 24 |

This retrospective development analysis concerns originally matched instances on two frozen detectors. The near-neighbor results are descriptive. They identify a case in which target IoU and neighboring-instance overlap move in opposite directions for segmentation quality. Instance separation recall and the behavior of missed vessels remain unmeasured.

## Execution provenance

The CPU worker ran Python 3.10.13 with the exact arguments below. The environment set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=-1`. The worker exited 0 and produced `DEV705_EQUAL_COUNT_WIDTH_AND_NEIGHBOR_MODIFIERS_ACTUAL_V1`.

```text
python.exe -I -B -X utf8 study.py
```

| Artifact | SHA-256 |
|---|---|
| Pre-analysis protocol | `acdcf61fe49d8f3a0a4f03f6627f94bd3a1f6af86229279efc09646a487c6a6f` |
| Analysis source | `ceba56e9cf2f398dba838f77f0a5045574bd6290ce4825a71479d912d36365c6` |
| Aggregate result | `4855ebac1756ed7d4231d5d44a8d4b6618ebfdca5894ef006440646c227ed2e7` |
| Process exit record | `b293592f85ff3488eef56aec95648426ee0529d15d757be2ee63fe9741e81bcf` |

The public entry uses SEAL_DEV705_SOURCE_ROOT to locate the qualified local inputs. The aggregate retains the original worker identity.
