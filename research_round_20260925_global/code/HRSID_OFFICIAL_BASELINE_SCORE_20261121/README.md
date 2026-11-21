# Frozen HRSID official TEST native baseline scorer

This CPU entry scores one unchanged native RF-DETR center seed 5202 E24 export with the same published COCO segmentation adapter used for DEV705. It computes the twelve standard COCO metrics and AP90 on each predeclared cohort. Results are fractions, with the standard COCO value of −1 for undefined metrics.

| Frozen view | Images | Original annotations |
| --- | ---: | ---: |
| Official | 1,962 | 5,922 |
| Empty-raster sensitivity | 1,962 | 5,918 |
| Image-identity sensitivity | 1,959 | 5,891 |

The official view includes all four empty GT rasters and the originally unannotated image. The empty-raster view keeps every image and prediction. The image-identity view keeps all GT and all predictions on its chosen images. A SHA-256 over the ordered per-image prediction block digests permits comparison of the rows used for each score. Historical official TEST use in fusion, seed and checkpoint selection still applies.

## Inputs after original export closure

Use the original native JSONL format of `hrsid_native_dev705.native_rows`. Each ordered official image contributes one record with `image_id`, `raw_query_count` (164), `native_count` (100), `unmapped_slot1_count`, `native_ranks` and `coco`. The scorer keeps sparse rank gaps, all mapped rows, native scores, ties, empty predicted masks and images with zero mapped rows. The separate label-free prediction readback checks the endpoint and coverage. The score worker checks the exact export bytes twice, the row count, image order, rank order, compressed RLE canvas and fixed category.

The scorer requires an actual Root-issued score admission and the independently completed native prediction readback. The caller supplies the admission SHA from its owner. The entry checks immutable bindings. The originating observer supplies the process outcome.

| Record | Required fields |
| --- | --- |
| Score admission | `status=ROOT_ADMITTED_HRSID_OFFICIAL_TEST_NATIVE_BASELINE_SCORE_V1`, `phase=HRSID_OFFICIAL_PNG_TEST_NATIVE_BASELINE_SCORE_ONLY_V1`, `endpoint_sha256`, `cohorts_sha256`, `gt_sha256`, `adapter_sha256`, `score_source_sha256`, `runtime_versions`, `export_readback_sha256`, `predictions_sha256`, `prediction_bytes`, `mapped_rows` |
| Export readback | `status=HRSID_OFFICIAL_TEST_NATIVE_EXPORT_INDEPENDENT_READBACK_ACCEPTED`, `original_process_terminal=true`, `original_exit=0`, `GT_opened=false`, `AP=false`, `images=1962`, `endpoint_sha256`, `predictions_sha256`, `prediction_bytes`, `mapped_rows` |

`runtime_versions` records the exact Python, NumPy and pycocotools versions under the keys `python`, `numpy` and `pycocotools`. Constant identities for the endpoint, signed-off cohort bytes, original GT bytes, input readback and ordinary adapter are pinned in the scorer. Cohort identity and export readback are verified before the worker opens predictions or GT. The score output is created exclusively.

Supply the ordinary published adapter `code/DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py`. It invokes `COCOeval` in segmentation mode with a single category, all retained image IDs and `maxDets=[1,10,100]`. AP90 uses the valid precision entries at IoU 0.90, area `all` and maximum 100 detections. Empty predictions on an image still leave that image in the evaluator. A completed score needs independent terminal readback for a paper claim.

```text
python -I -B -X utf8 score_native.py --admission ORIGINAL_SCORE_ADMISSION --admission-sha256 ORIGINAL_ADMISSION_SHA --cohorts FROZEN_COHORT_JSON --export-readback ORIGINAL_EXPORT_READBACK --gt TEST_ORIGINAL_ANNOTATIONS_PRIVATE.json --predictions ORIGINAL_NATIVE_PREDICTIONS.jsonl --adapter ORDINARY_COCO_WORKER.py --output FRESH_SCORE_SAFE.json
```

## Synthetic verification

Set `NATIVE_SCORE_ADAPTER` to the absolute path of the ordinary published adapter and run the following command from this source directory.

```text
python -I -B -X utf8 -m unittest discover -s . -p test_score_native.py -v
```

The suite compares all thirteen metrics with direct stock COCO evaluation, checks the effect of a high-confidence false positive on an originally unannotated image, covers globally empty and sparse, tied, partly unmapped predictions and verifies failure on a changed phase, cohort, export digest, image order or original process evidence. All fixtures are generated. No official TEST image, label, prediction or score was opened in the synthetic run.
