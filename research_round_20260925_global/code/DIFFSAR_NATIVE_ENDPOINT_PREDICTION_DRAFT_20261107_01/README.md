# Native DiffSAR fixed-endpoint prediction candidate

2026-11-07. **Unexecuted future prediction source.** No checkpoint, GT body, raw image, Torch/CUDA or actual model has been loaded or evaluated while preparing this candidate. Metadata/IO tests are not model or inference validation.

## Narrow role

This worker exports native model predictions only. It may run only after Root has externally qualified the original fixed **37050-update FORMAL** checkpoint. It cannot select a best checkpoint, load a health/resume-validation/partial endpoint for inference, update an optimizer, compute AP or grant itself result qualification. The newly created future evaluation lease references the existing final Root qualification record; neither a lease boolean nor these metadata checks replaces Root's actual terminal/source/count/weight-chain audit. **Do not modify or expand any frozen training lease, original qualification or old scientific source to authorize prediction.**

The author SSDD test YAML is7066 bytes, SHA256 `1220408ebd3b1dbe7c15c4ebaa389481e7447608d6770df342e248983241527b`. It differs from author train YAML only in proposals500→1000. The worker builds the actual1000-proposal R101/SCJE model, strictly loads every endpoint model parameter and persistent buffer with identical shape/dtype, and recomputes the original typed-byte model digest. No optimizer is constructed or loaded into the model; the trusted full checkpoint is read on CPU and its unused optimizer/gradient/RNG data is released.

Inference uses the original `DiffSARShipInstDatasetMapper(cfg,is_train=False)` and exactly one `model([mapped])` call per image. It preserves RGB/min-test512/max-test512, SAMPLE_STEP1, original DDIM initialization, head, threshold0.5 NMS and the model's custom postprocessing. There is no rewritten sampler, extra NMS, TTA, score threshold, teacher mask/box input or module change. The custom postprocessing performs aligned bilinear×2, padding crop, original-size bilinear resize, in-place threshold, box-mask clipping and empty-box removal.

Original postprocessed `pred_masks` may be float32 **values0/1**, because `gt_` is in-place. Do not demand bool dtype or substitute generic ROI-mask pasting. The worker calls original `instances_to_coco_json(instances.to('cpu'),image_id)` for **all** native post-NMS masks, including when there are more than100, then maps contiguous class0→COCO class1 exactly once. It streams all rows to `predictions.coco.json`; it does not sort/select top100 or silently discard invalid outputs. A metadata/IO test keeps137 stand-in rows with their order and scores unchanged, but does not prove the actual model can produce137 or1000 masks.

## Population and RNG

Only one full, ordered **internalDEV187** pass is supported. DEV has already been used for development; it is not a new untouched official TEST. Expected integer-GT population is380 instances, but this prediction worker **never opens or hashes the GT file and does not independently certify that count**. It accepts only a pinned image metadata manifest, not annotations. Native model candidates are independent of the internal teacher's100-query U66 population; do not route this model through U66's100-candidate constraint.

The inference seed40244023 is reset once **after model construction and exact endpoint loading, immediately before creating the native inference mapper and beginning the ordered187 pass**. Python, NumPy, TorchCPU and visible-CUDA streams are seeded together. It is not inherited from the training checkpoint's ending RNG. There is no per-image reseed, shard, range resume or restart RNG protocol. The full187 order is part of the future lease/spec. Do not split this worker into independently reseeded shards and call that the same trajectory. Exact author-paper inference randomness remains unknown.

## Minimal source and future Root interface

Runtime source dependencies are this `worker.py`, the unchanged reviewed config adapter, and the fixed qualified `full_train_worker` used as a standard-library metadata/digest helper. The training helper must byte-match the endpoint's scientific source pin. The helper is never called to train. There is no new qualification daemon or separate admission framework; Root creates one **new future evaluation lease** with an inline admission joined to the actual final qualification reference.

Spec schema `Root-DiffSAR-fixed50-native-prediction-spec-v1`, operation `CUDA_DIFFSAR_FIXED37050_NATIVE1000_DEV187_PREDICTION_V01`. CLI: `worker.py --spec ABSOLUTE_SPEC --spec-sha SHA256`.

Exact spec keys: schema, operation, run_id, source_refs, library_sources, test_config_ref, checkpoint_ref, endpoint_qualification_ref, image_manifest_ref, evaluation_gt_ref, root_lease_ref, output_directory, prelock. run_id begins `DiffSAR-fixed37050-native1000-`. source_refs has worker/config_adapter/full_train_worker. `library_modules(full_train_helper)` exposes the required native module map, adding the native COCO serializer, detection utils/transforms and pycocotools encoder/binary to existing full-worker modules; include the original torchvision extension in library_sources. All modules are imported only by a future scientific invocation, and their actual file references must match the pinned source set.

Lease schema `Root-DiffSAR-fixed50-native-prediction-lease-v1`, statusROOT_GRANTED. It joins operation/run_id/prelock/output_directory/checkpoint_ref/endpoint_qualification_ref/image_manifest_ref/evaluation_gt_ref, worker_ref/full_train_worker_ref, current boot_id, single_attemptTrue, prior_project_scientific_groups_reapedTrue, exclusive_project_scientific_leafTrue and whole_device_exclusiveFalse. Its `endpoint_admission` contains:

```
qualified: true  # supplied only after actual external Root qualification
selection: FIXED37050_NO_BEST
updates: 37050
checkpoint_ref: exact original final checkpoint three-field ref
qualification_ref: exact existing Root final qualification three-field ref
training_protocol: exact qualified checkpoint protocol
training_scientific_pins: exact qualified checkpoint scientific_pins
model_component_digest: actual qualified endpoint checkpoint model digest
```

This admission is a consumer of prior actual qualification. Tests use stand-ins solely to verify rejection behavior and cannot fabricate that qualification. The worker remains `qualified=False/Root_external_admission_required=True` even after its own original0; Root still verifies prediction execution/source/birth/resources and final outputs externally.

Image manifest schema `Root-DiffSAR-DEV187-original-image-input-v1`:

```
population_role: INTERNAL_DEV187_ALREADY_USED_FOR_DEVELOPMENT_NOT_OFFICIAL_TEST
evaluation_gt_ref: same advertised original integerGT ref as the future spec
GT_instances_expected: 380
category_id_map: {"0": 1}
images: [
  {dataset_index: 0, image_id: ORIGINAL_ID, height: H, width: W,
   image_ref: {path: ORIGINAL_ENCODED_IMAGE_PATH, bytes: N, sha256: HASH}},
  ... 187 distinct logical IDs, fixed indices0..186 ...
]
```

The manifest has no annotation coordinates or masks. Source byte verification may deduplicate repeated file references; prediction never deduplicates logical samples. Inputs are original encoded images which the native mapper decodes, not normalized teacher feature tensors.

## Output and unchanged COCO mathematics

`predictions.coco.json` is a complete COCO result list containing all native predictions, bbox/score/RLE, original image IDs and original category1. Each RLE is produced by the author's existing helper using Fortran-order uint8 and UTF-8 counts. The incomplete `.part` is retained on failure and is not a qualified complete prediction file. `REPORT.json` records per-image native/serialized counts, shape/dtype, synchronized native forward time, total mapping/IO/export time, process/source identity, resource observations and final file identity. There are always optimizer_steps0/training_credit0/APfalse.

**Later CPU evaluation is deliberately not implemented here.** Reuse the already executed native/official COCO mathematical path, not a new metric implementation: for segm, remove prediction bbox before `COCO.loadRes` so area bins use mask area; set the locked population/category and `maxDets=[1,10,100]` in COCO evaluation. That evaluator must independently verify the actual original integerGT187/380 identity. Retain the full prediction list; COCO maxDets100 is not a pre-mask model cap. The existing U66 fixed100 serialization/population guard requires a separate input adapter rather than being silently reused for this end-to-end model.

## Resources and outstanding work

Candidate defaults preserve the established CPU0/1, threads2, allocator8GiB, groupRSS16GiB, start-free48GiB/floor16GiB/deadline1800s profile. Root's actual owner must supervise process-group resources and closure. Actual1000-proposal peak memory/time are **unknown**. A resource failure must remain a resource failure, without reducing proposals, discarding masks, changing precision or selecting a smaller favorable population.

Still missing: the actually qualified37050 checkpoint/admission; a source-qualified DEV187 raw-image manifest on DGX; actual strict500-trained→1000-inference loading/forward/complete export; final source/resource/cleanup admission; and the independent CPU evaluator adapter with actual integerGT identity and source references. No intermediate checkpoint may be used to fill these gaps through this endpoint-only worker. No claim of full author-paper initialization, official-test independence, fair preprocessing cost, AP improvement or TGRS readiness follows from this draft.

Prepare a new ordinary prebinding copy before execution; do not publish checkpoints, raw images, GT, predictions, private spec/lease, host locators or original qualification payloads. Publish only own source and accurately labeled ended safe aggregates. Current training and old failures remain unchanged.
