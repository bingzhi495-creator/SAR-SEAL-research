# Thin CPU COCO adapter for native end-to-end predictions

2026-11-07. **Unexecuted future scientific source.** Preparation used only ordinary/public source and metadata/IO/API stand-ins. No real checkpoint, GT, prediction, bound SOURCE or qualification payload was opened; no real NumPy/pycocotools/Torch import, SSH, model inference or AP computation occurred. The four unit tests are not a scientific COCO run.

## Mathematical path

`evaluate_population(backend,coco_gt,predictions,image_ids)` is deliberately only original API wiring:

1. Deepcopy the entire original prediction list; remove `bbox` only in that private segm-evaluation copy. Do not change scores, map categories again, discard results, select top100 or modify original files/list.
2. Call original `COCO.loadRes` on the complete segm result list. Omitting prediction bbox makes the original API derive detection area from segmentation, rather than bbox area.
3. Construct original `COCOeval(coco_gt,coco_dt,'segm')`, set all187 image IDs, category1 and `maxDets=[1,10,100]`.
4. Call `evaluate()`, `accumulate()` and `summarize()` once for the whole population. Read actual12 stats; do not implement IoU, matching, precision interpolation or AP locally, and do not average shard AP.

For a valid globally empty prediction list, construct an empty result dataset with original `COCO()`/`createIndex()` and the original image/category metadata, then call the same COCOeval path. This avoids releases where `loadRes([])` indexes element0; it does **not** manually fill AP with zero. Per-image0 predictions and>100 predictions are both legal. Prediction records must already have original category1 and original-sized compressed RLE; category0 is rejected rather than silently remapped again.

The native producer used its own1000 candidates and retains its complete postprocessed list. The evaluator has no18700-record constraint. The teacher's fixed100-query/U66 source/population contract remains separate and must not be changed to fit the native model.

## External prerequisites, without a new owner

The worker does not grant itself upstream or final qualification. Root must first finish and actually qualify the original FORMAL37050 endpoint and the complete native187 prediction invocation, including original terminal/source/count/birth/resources and product identities. A raw producer report's `qualified=False` is expected: it required external Root admission. Its `worker_exit_code=0` alone is insufficient. A new **future CPU lease** consumes existing external qualification refs and the exact native producer source reference; do not edit any frozen training/prediction lease or original qualification to authorize evaluation.

Only a pinned original integerGT187/380/category1 is accepted. The code verifies187 unique original GT images and their dimensions against the exact same native image manifest,380 distinct original annotations and category1, preserving all original geometry, area and crowd metadata. The label `ORIGINAL_FIXED_INTEGER_GT187_380` denotes Root's qualified byte identity; the adapter does not reconstruct historical rounding or certify physical truth from that name. This is already-exposed internalDEV, not an untouched official TEST.

All prediction-file rows are validated against those image IDs and original dimensions. Counts are joined to the complete source-qualified native producer report per image, including zero rows. There is no recoding, thresholding or favorable-population selection. The canonical COCO function can later be reused to reevaluate existing native/R references under separately correct source/population admission. This draft does not read those old results or pretend they came from the new end-to-end producer.

## Actual library provenance

Five selected library refs are required: NumPy, `pycocotools.coco`, `pycocotools.cocoeval`, `pycocotools.mask` and the actual compiled `pycocotools._mask` extension. Their imported file bytes/SHAs must match the future spec and Root's explicit canonical COCO source set. `mask.py` and `_mask` refs must also equal the **actual** native prediction encoder refs recorded in the qualified producer report. A common version string is not sufficient evidence of common implementations. Version text is only a recorded fact.

Root must bind the real canonical evaluator/codec sources before execution. This source set can define prospective unified evaluation; its equality alone does not retroactively certify every historical result's runtime. Historical reference reproduction, if needed, requires actual controlled reevaluation with the same canonical sources and intact prediction provenance. Selected refs do not claim full transitive-environment isolation.

## Interface

CLI: `worker.py --spec ABSOLUTE_SPEC --spec-sha SHA256`.

Spec schema `Root-SAR-native-DEV187-COCO-segm-spec-v1`; operation `CPU_SAR_NATIVE_FULL187_FIXED37050_COCO_SEGM_V01`; run_id begins `SAR-native37050-DEV187-COCO-`. Exact keys:

```
schema, operation, run_id, prelock,
source_refs: {worker: THREE_FIELD_REF},
library_sources: {numpy, pycocotools_coco, pycocotools_cocoeval,
                  pycocotools_mask, pycocotools_extension},
image_manifest_ref, evaluation_gt_ref, predictions_ref,
native_prediction_report_ref, native_prediction_qualification_ref,
checkpoint_ref, endpoint_qualification_ref,
root_lease_ref, output_directory
```

Every ref is `{path,bytes,sha256}`. `checkpoint_ref` is provenance metadata only; this CPU worker never opens model weights. The image manifest is the same `Root-DiffSAR-DEV187-original-image-input-v1` used by the native producer; only its identity, order, image IDs/dimensions and advertised GT ref are consumed. Raw image files are not opened in CPU evaluation.

Future lease schema `Root-SAR-native-DEV187-COCO-segm-lease-v1`, statusROOT_GRANTED. It joins operation/run_id/prelock/output_directory and all manifest/GT/prediction/report/qualification/checkpoint refs, worker_ref, current boot_id, single_attemptTrue, prior_project_scientific_groups_reapedTrue and exclusive_project_scientific_leafTrue. `canonical_COCO_library_sources` equals the selected actual library refs. Inline `external_admission` contains:

```
fixed_endpoint_qualified: true
endpoint_updates: 37050
native_prediction_qualified: true
full187_verified: true
GT_variant: ORIGINAL_FIXED_INTEGER_GT187_380
prediction_source_ref: ACTUAL_EXTERNALLY_QUALIFIED_NATIVE_WORKER_REF
predictions_ref, native_prediction_report_ref, native_prediction_qualification_ref,
checkpoint_ref, endpoint_qualification_ref, image_manifest_ref, evaluation_gt_ref:
    EXACT_MATCHES_TO_THE_SPEC
```

These are external premises supplied only after the actual Root audit. Metadata values do not manufacture scientific qualification. This candidate adds no supervisor, daemon, SSH call or new owner; Root's existing bounded CPU-scientific mode must provide actual process-group/deadline/RSS supervision and postchecks.

## Truthful result states and failures

`REPORT.json` keeps independent read/parse and mathematical states:

- `GT_bytes_read`, `prediction_bytes_read`, `GT_JSON_parsed`, `prediction_JSON_parsed`: None before the corresponding attempt; False when attempted but not completed; True after the full read/parse completes. A byte read can be True while identity validation or parsing fails. False does not claim no partial IO occurred.
- `COCO_math_started=False` and `COCO_math_completed=False` initially; start becomesTrue immediately before the API calls. Successful complete calls and finite12 stats set completedTrue and `AP_computed=True`.
- `metrics=None` until actual successful statistics exist. Raw12 fractional stats retain undefined−1; the separately labeled percentage view multiplies defined values by100 and represents undefined values asNone rather than artificial0.
- `qualified=False` and `Root_external_admission_required=True` remain. If source/resource verification fails after computation, the real computed metrics/True flags remain alongside FAILED status; they are not a qualified scientific result. No failed numerical original is upgraded by a later metadata0.

All failures retain the actual error, traceback and reached state, return original1 and do not automatically retry, change GT, alter scores or select a different endpoint. No actual metric/result file is created during these unit tests.

Schema/admission failures before REPORT creation remain original stderr/exit1 rather than a fabricated complete report; Root must retain those originals. There is no AP calculation on that path.

## Resource and completion limits

Prospective defaults mirror Root's existing CPU mode: CPU0/1, OMP/MKL/OPENBLAS2, CUDA_VISIBLE_DEVICES empty, groupRSS16GiB, start-free48GiB, floor16GiB, deadline1800s. No Torch import or GPU numerical execution is permitted. Worker checks supplement rather than replace Root's actual owner/group qualification.

Outstanding before any AP claim: real final37050 qualification, source-qualified full187 native predictions, exact original integerGT and image manifest, actual consistent canonical COCO/codec refs, bound CPU-owner invocation, actual API execution and external Root product/resource/source postchecks. No current training, frozen predictionv1, original qualification, old failure or old reference result was changed.

Own source/test/README may be copied before binding for ordinary publication. Keep raw GT, full predictions, checkpoints, private specs/leases, qualification chains and host locators out of the public repository. Source readiness or a metadata-test pass is not model validity, AP success, superiority or TGRS readiness.
