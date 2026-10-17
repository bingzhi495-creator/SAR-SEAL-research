# Fixed E50 DEV execution wrapper — unbound, numerical execution pending

This draft supplies the missing outer execution entry around the existing U40
prediction and complete-population metrics cores. It does not modify the cores,
reader, trained model, scores, evaluation rules or progression decision. No
model, image, weight, cache, annotation or actual qualification was read during
authoring. Only fabricated standard-library metadata tests and syntax checks
have run. This is not an executed evaluation or a new qualification leaf.

`run_eval.py --spec ABSOLUTE_PATH --owner-execute` is for a future original
Root-owned task after final source publication, complete binding and admission.
No binder, launcher, automatic retry or Root receiver is included here.

## Two operations

| Operation | Spec schema | Sole numerical call |
|---|---|---|
| `GPU_U40_FIXED50_FULL_METHOD_DEV_SHARD` | `ROOT-U40-fixed50-full-method-DEV-prediction-spec-v1` | `prediction_core.predict_shard(...)` |
| `CPU_U40_FIXED50_FULL_DEV_COCO` | `ROOT-U40-fixed50-complete-DEV-metrics-spec-v1` | `population_metrics.evaluate_population(...)` |

The endpoint is seed `202610173101`, fixed epoch **50**, **37050 updates per
arm**, with the actual Root-accepted final pair. The current accounting parent
`parent_ROOT_qualified` is separate from that model parent. No old N28 E24
checkpoint, result, threshold or progression gate is substituted.

The four canonical GPU ranges are `[0,47]`, `[47,94]`, `[94,141]`, `[141,187]`.
Each shard calls the existing prediction core once; the core runs PLAIN then
FULL with one GPU arm resident at a time, all 100 original queries per image,
query microbatch four, original scores/classes/export order, and no new teacher
forward. Native tensors retain their natural query axis; the existing core
selects them by the original export query IDs. Original native zero-residual
calibration remains mandatory. Both arms may change the complete segmentation,
without restricting writes to Omega.

CPU calls the existing metrics core once after all four actual Root GPU
acceptances, retaining the full 187-image/18700-query order. It never averages
shard AP. Native and strong R must reproduce the existing complete COCO
references. The fixed native .50 matching is the only geometry-pair selector;
380 is the total DEV GT denominator, not a requirement that all 380 GT be
selected. Original bbox/area fields and the canonical COCO helper behavior
remain unchanged. No extra mask-area AP, new metric, threshold or benefit gate
is introduced.

CPU imports genuine pinned Torch only to check that CUDA remains uninitialized.
It creates no Torch tensors, model or optimizer, restores no neural state and
does not replay model inference. CPU COCO computation uses the existing NumPy
and official COCO modules. Historical prediction bundles may contain target
associated fields; this is not a claim that such historical caches were created
without GT. DEV annotations are consumed by the existing CPU metrics core.

## Declared source and metadata ABI

Common flat spec fields include `entry_source`, `body_source`, `runtime_source`,
`prediction_core_source`, `training_qualified`, `checkpoint_refs` (PLAIN/FULL),
`pair_commit_ref`, `order_ref`, `dev_context`, `training_sources` (17 roles),
`prediction_sources` (17 roles), `original_image_refs` (187 ordinal-ordered
strict references), `prelock`, `parent_ROOT_qualified`, plus phase, admission,
output, resource and dynamic accounting fields.

GPU adds `endpoint_refs` with exactly `training_qualified`, `pair_commit`,
`preparation`, `order`, plus `shard_index` and `dataset_range`. Their model refs
join the common flat fields. CPU adds `metrics_source`, `coco_context`,
`metrics_module_sources` (exactly legacy_eval/population/geometry/official/coco/
cocoeval/masks) and four ordered `accepted_GPU_shards` strict references.

`sources` is the explicit unique strict-reference list. Every directly closed
input and source used by the unchanged cores must be admitted by the future
Root binding, including contexts and their actually read child references,
final endpoint proofs/checkpoints, each shard's selected JPEG/cache inputs, and
the CPU accepted reports, prediction/Omega exports, references and annotation.
The entry checks declared role membership; original `Control.closed` verifies
actual bytes when read. It does not discover or guess undeclared file paths.
All declared Python role sources are closed before Torch/NumPy import and
rechecked at return. This is explicit-source coverage with pinned scientific
versions, not a complete imported-framework byte audit.

Prediction core pin:
`819a2350487394265980cb77f8ff7e89547dbcac8e0d3e3c175a67a2794c9680`.
Metrics core pin:
`f513d59321ffb571f6d823a74f31e3b6fd190086dc73bd77b400fe0500eb489a`.
The original full model's temporary `dsconv`, `head` and `model_full` aliases,
and all dynamically installed wrapper module aliases, are restored on exit.
Canonical pycocotools modules must match their explicitly admitted sources.

All five leaves use one strict prelock reference with schema
`U40-fixed50-complete-DEV-metrics-prelock-v1`. It contains seed/fixed_epoch/
updates_per_arm, images187/GT380/queries18700, GPU_ranges, query_batch4,
binary_rule `>0`, fixed_original_S0 true, GT_used_for_prediction false,
score_update false, query_filter null, NMS false, no_new_training/no_TEST true,
and `original_image_references_metadata_sha256`. The latter is SHA256 of the
canonical JSON for the ordinal 187 strict refs (sorted object keys, compact
separators, UTF-8, no ASCII escaping). CPU additionally requires each accepted
GPU proof's top-level `prelock` to equal this reference and checks its report's
RGB-reference digest. A future Root GPU receiver must preserve that field.

## Resources and output

Original owned admission has the existing 13 fields. Limits remain CPU affinity
0/1, two Torch/inter-op threads, 1800 seconds, RSS at most 16 GiB, available RAM
at least 16 GiB, whole-device observation at most 8 GiB and CUDA allocator at
most 4 GiB. GPU visibility is `0`; CPU visibility is empty. Original supervision
retains the whole-device/resource deadline controls. The existing prediction
core owns initial CUDA setup and sets its allocator cap before GPU tensors.
CPU stays CUDA cold, including failure cleanup. Torch is `2.0.1+cu118`, NumPy
`1.23.5`; the prediction core also enforces its existing interpreter/numerical
contract. GPU deterministic flags are enabled, benchmark and TF32 disabled.

Each core exclusively creates a fresh output directory. Its original result is
returned unchanged. The thin body returns exactly `result`,
`wrapper_report_ref`, and `core_peak_allocated_bytes`; the last is an observed
core resource scalar, not an added scientific result. `WRAPPER_REPORT.json`
records dispatch/source/prelock/RGB provenance separately from the original
`DEV_REPORT.json` or `COCO_REPORT.json`. Core failures propagate, with available
partial reports and original logs retained. No successful Q is written for a
failure. Failure before output creation remains in the original driver log.

The entry performs a second GPU cleanup (GC, synchronization, cuBLAS workspace
clear, empty cache, synchronization) and requires actual allocated/reserved
bytes both zero. Its reported allocator peak is the largest recorded per-arm
measurement and final interval; the core resets measurements, so this is not
claimed as an independent uninterrupted process-wide peak. RSS/free-RAM are
the original controller's sampled observations. No teacher-inclusive FPS claim
is added.

Successful output includes pending `Q.json`, schema
`U40-owned-pending-fixed50-full-method-DEV-v1`, carrying the unchanged core
result and separate wrapper ref. `qualified` and original_actual0_self_certified
remain false; only Root may accept the original terminal identity and observed
exit, held record, fresh FIRST, sources, outputs and accounting. GPU Q has
AP false; CPU Q has AP true; both have new_training false, DEV true, TEST false,
and utility_gate_decided false. Dynamic normal, failed and legacy counts remain
distinct; this wrapper never advances them.

## Verification boundary

`test_metadata.py` contains three fabricated standard-library tests: operation/
resource/endpoint rejection, declared source and original RGB identity closure,
and pending owned result/provenance rejection. They import no Torch, NumPy,
PIL or COCO and execute neither entry main nor body/core. Passing them is not
evidence that model restoration, CUDA prediction, full-population joins, COCO,
dependency loading or resource behavior has executed. Actual fixed E50
evaluation remains pending the original Root-owned tasks.
