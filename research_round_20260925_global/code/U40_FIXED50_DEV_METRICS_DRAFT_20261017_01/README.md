# U40 fixed-E50 full-population CPU metrics — unbound draft

Status: implemented source and three executed synthetic standard-library metadata tests only. The numerical `evaluate_population` path has not run. No Torch/NumPy/PIL/COCO import, actual image/cache/annotation/prediction/result/weight read, network execution, AP or qualification was performed in this task. No runner, binder, state update or additional qualification leaf is included.

## Injection and future contract

```text
evaluate_population(torch, np, c, spec, pre, modules, report) -> pending result
```

`report` is a caller-owned partial dictionary, updated on failure and success. The future admitted Root-owned CPU caller controls Source closure, resource limits, actual terminal evidence and qualification. This function cannot certify itself or advance a training/seed gate. It starts and ends CUDA cold and never deserializes a checkpoint.

`modules` and `spec.metrics_module_sources` each have exactly seven roles: `legacy_eval`, `population`, `geometry`, `official`, `coco`, `cocoeval`, `masks`. Every injected module's `__file__` must equal its explicitly closed source ref. Three reused source pins are fixed in code:

- Original `N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/eval_body.py`, SHA256 `0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf`; only `official_metrics` is called. None of its old `gpu/cpu/run/validate_endpoint`, E24 constants, N26 AP threshold or progression logic is called.
- Original `population_contract.py`, SHA256 `733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5`; its unchanged `merge_predictions` enforces each original non-segmentation JSON field at every ordinal. GPU prediction owns `replace_segmentations`.
- Original `geometry_diagnostic.py`, SHA256 `6f6669eb91d0842f7c23eb09b992624642d16c9abe8d145c258ec9aae5ff36ac`; only its original `diagnose` mathematics is reused.

The `official` source ref must be the original `coco_context.COCO_source`. `coco`, `cocoeval`, `masks` are explicitly injected official modules; the enclosing runtime still owns the complete admitted framework/import/binary closure. The module itself has no scientific-library imports. `spec.metrics_source` must refer to the executing source file, and all metadata/prediction/annotation refs must already be in the admitted Source map.

New proposed CPU schema: `ROOT-U40-fixed50-complete-DEV-metrics-spec-v1`. Prelock schema: `U40-fixed50-complete-DEV-metrics-prelock-v1`, with seed `202610173101`, `fixed_epoch=50`, `updates_per_arm=37050`, `images=187`, `GT=380`, `queries=18700`, `GPU_ranges=[[0,47],[47,94],[94,141],[141,187]]`, `no_new_training=True`, `no_TEST=True`. These schemas describe future wrappers, not an existing execution or Root acceptance.

Besides that fixed endpoint, spec supplies `training_qualified`, `pair_commit_ref`, `checkpoint_refs={PLAIN,FULL}`, `order_ref`, `dev_context`, `coco_context`, four ordered `accepted_GPU_shards`, the original 17-entry `training_sources`, exact 17-entry `prediction_sources`, `prediction_core_source`, the seven metrics-module refs, `metrics_source`, `prelock`, `phase`, and a fresh `output_directory`. This function exclusively creates that directory; a future wrapper must not create it first. Its admitted control supplies `execution_spec_ref`, `execution_admission_ref`, `closed/check/write_json`.

## Endpoint and four-shard joins

The training reference must be actual `ROOT-U40-continuous-training-segment-qualified-v1`, `qualified=True`, `paired_prefix_advanced=True`, final result span `{start:36864,stop:37050}`, seed above and cursor `{global_updates:37050,completed_epochs:50,within_epoch:0}`. Result checkpoint/order/pair refs must equal spec. No model or optimizer reconstruction is claimed by this CPU join.

Each future prediction acceptance must use `ROOT-U40-fixed50-full-method-DEV-shard-qualified-v1`, with `qualified=True`, preserving the GPU result. Its `result.report_ref` closes a `U40-fixed50-full-method-DEV-shard-pending-v1` report. This Root schema is an explicit future interface proposal; worker pending outputs alone never qualify.

The shared result/report fields are `predictions`, `identity_predictions`, `Omega_support`, `dataset_range`, `shard_index`, `images`, `queries`, `seed`, `fixed_epoch`, `updates_per_arm`, `training_qualified`, `pair_commit_ref`, `checkpoint_refs`, `order_ref`, `dev_context`, `training_sources`, `prediction_sources`, `new_training`, `AP`, `DEV`, `TEST`. Calibration booleans and source maps follow the GPU core's declared new E50 contract. All shards must use one exact core, original encoder/mask API/population refs, original training sources and common endpoint. Actual per-arm costs remain under each report's `cost={PLAIN,FULL}` and are retained unchanged.

All four Root proofs are checked before opening the DEV annotation file. Then the original native/R JSON and each full shard are merged at original image/query order, retaining 18,700 candidates and original S0/class/all non-segmentation fields. Identity exports must reproduce entire native dictionaries exactly. Full original `Omega` support is joined by original dataset/image/query order. No shard AP is computed or averaged.

## Unchanged evaluation and honest scope

The original contexts keep their existing `ROOT-N28-fixed24-dense-native-DEV-context-v1` and `ROOT-N28-fixed24-original-COCO-context-v1` schemas solely as cache/annotation association ABIs; no old model endpoint is accepted. CPU COCO opens original DEV187/380/category1 without conversion. Both native and R complete COCO dictionaries must exactly reproduce the accepted legacy reference under `control_results.native_official.COCO` and `control_results.strong_fixed_E24_Omega.COCO`. The E24 text in this R reference key describes the frozen reference, not the new method endpoint.

Results use real names `native_official`, `R`, `PLAIN`, `FULL`. The standard 12 statistics and original per-IoU AP are retained. Differences reuse those same statistics; undefined `-1` entries remain undefined rather than becoming artificial zero differences. No new metric, threshold, second mask-area AP, model-selection rule or progression policy is implemented. Original plan decisions remain with Root.

For geometry only, the old helper's hard labels are explicitly adapted as `SELECTIVE -> FULL`, `WHOLE_NODE -> PLAIN`. Summary and private-record method keys are changed back, the original algorithm schema and alias map are saved, and no old SELECTIVE/WHOLE_NODE model is claimed to have run. Original native IoU .50 / all-area / maxDets100 nonignored matches select the fixed query pairs. The complete denominator remains 380; selected coverage may be lower. All methods use that same original query, with original `abs(native_L)<=2` support only for inside/outside accounting. This is conditional geometry, not full-380 matching, AP replacement or permission to restrict mask writes.

Only segmentation changes. Original bbox/area metadata is retained even if present. The report counts actual native `bbox`, nonempty `bbox`, and `area` fields, which remain equal across methods. The original official `loadRes` helper may use a nonempty bbox for detection area and unmatched-detection area-group ignore; mask IoU remains segmentation-based. The code does not silently strip/recompute fields or revise the locked baseline evaluation.

The output directory receives full merged private predictions, the fixed-pair private record file, and `COCO_REPORT.json` (`U40-fixed50-complete187-COCO-pending-v1`). Individual GT/query identities and paths are not a public summary. A future Root publication step must extract a sanitized summary under the existing publication policy. Costs include new-UNet branch timing from accepted shards and CPU COCO time; cached teacher precomputation is excluded and complete-pipeline FPS is not measured. Historical target-association fields may exist in original prediction/cache sources; the function does not claim those full sources were GT-free.

## Metadata test scope

Exactly three tests cover the fixed accepted paired endpoint, four-shard population/source/acceptance joins, and explicit geometry relabeling with unchanged values and the conditional 380-GT denominator. They use fabricated JSON only, not fake tensors/NN/COCO. Actual scientific imports, source/byte closure, current input existence, masks, official numerical reproduction, geometry and resources remain unexecuted and must pass within the future Root-owned evaluation.
