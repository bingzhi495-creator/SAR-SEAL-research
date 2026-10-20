U67 is an independent, unbound numerical wiring draft. It has not evaluated real
predictions or annotations. Synthetic JSON/stub tests establish interface
behavior only, not AP, model utility, endpoint eligibility, or publication value.
No frozen U40/U58/U59/U66 source is modified.

`population_metrics_core.evaluate_u66_population(np, c, *, shards, native,
strong, reference_COCO, coco, ids, modules)` has no CLI, file I/O, owner,
checkpoint loading, model execution, retry, writer, or Root acceptance logic.
Only stdlib is imported. Root must inject actual admitted libraries/objects and
independently establish source bytes, original exits, accepted E50 provenance,
the four shard reports and reference provenance before scientific execution.
Pending U66 status is an input ABI, never a grant of Root acceptance.

Inputs and injected APIs:

- `shards`: four U66 JSON reports in original ranges `[0,47]`, `[47,94]`,
  `[94,141]`, `[141,187]`. All whole endpoint dictionaries and `source_refs`
  must match exactly, with U57 FORMAL E50 cursor 37050/50/0. Both arm reports
  must have completed prediction status. A U65 input-reader result is rejected.
  Every original per-image full100/zero-calibration proof and each arm's
  one-encode/25-readout/full100 counts must be complete.
  Top predictions/identity/PLAIN Omega/cost must equal their arm copies; only
  the top predictions are merged. Copies are not additional candidates.
- `native`, `strong`: original full18700 JSON prediction dictionaries, with
  every original100 query IDs and original export order retained. Scores,
  class, bbox/area, empty masks and all non-segmentation fields remain exact.
  No sorting, score normalization, query renumbering or silent truncation occurs.
- `reference_COCO`: exactly `{'native_official': native_COCO_dictionary,
  'R': strong_R_COCO_dictionary}`. Root extracts these complete dictionaries
  from the actual original reference, including the strong control historically
  named `strong_fixed_E24_Omega`; this module does not open or qualify it.
  Both complete12-stat dictionaries, including all metadata and undefined -1,
  must reproduce exactly. Matching AP alone is insufficient.
- `coco`, `ids`: injected original COCO object with original ordered187 IDs,
  380 annotations and category1. This module opens no annotation file itself;
  original official/diagnostic helpers use the injected annotation object.
- `c` supplies `check()`; external ownership/resource/exit controls remain the
  caller's responsibility. `np` is an injected genuine NumPy module in any
  future actual run, never imported here.
- `modules` has exactly seven roles: `population_contract`, `eval_body`,
  `official`, `cocoeval`, `mask_api`, `geometry_diagnostic`, `population_metrics`.
  Actual callable presence is checked, not actual source admission. Root must
  close source bytes and inject the original admitted code. Required APIs are
  `merge_predictions`, `official_metrics`, `coco_metrics`, `COCOeval`, `decode`,
  `diagnose`, `relabel_geometry`, `_delta_stats` respectively.

The unchanged original call sequence is:

1. N28 `population_contract.merge_predictions` checks strong_R, PLAIN, FULL
   and identity against the original full candidate population. Both arms'
   identity dictionaries must agree, and the merged identity is byte-significant
   canonical JSON equal to native (including non-segmentation metadata).
2. N28 `eval_body.official_metrics(np,c,official,COCOeval,coco,values,ids)` runs
   once for native_official, R, PLAIN and FULL. No average of shard AP is used.
   The native call's original COCOeval object alone is retained for matching.
3. N28 `geometry_diagnostic.diagnose(np,mask_api,coco,native_evaluation,
   legacy_labels,original_PLAIN_support)` uses native evaluation's original
   fixed nonignored IoU .50/all/maxDets100 matches. Omega is the original
   PLAIN top support joined by dataset/image/query order; no support is derived
   from any changed prediction mask and no new method rematching occurs.
4. Original U40 `relabel_geometry` transparently maps helper SELECTIVE to FULL
   and WHOLE_NODE to PLAIN; it evaluates no legacy model. Its returned old U40
   geometry schema is an inherited mathematical label, not U57/U67 execution
   qualification. Original `_delta_stats` computes the five original pairwise
   COCO12 differences; an undefined original -1 stays `None` in differences.

Return value is exactly `{'summary': JSON_dictionary, 'private_pair_records':
list}`. Summary status is `METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE`, with
`qualified=False`, `ROOT_actual_terminal_self_certified=False`, no utility gate,
no training credit and no TEST. `AP=True`/`DEV=True` in an actual successful
return describe the invoked original evaluation scope, not a source-test result.
The summary contains complete COCO/per-IoU outputs, original deltas, fixed-pair
geometry summary, endpoint metadata and original per-shard arm costs; it has no
COCOeval object or private GT-pair records. Cost dictionaries are preserved
without invented timing aggregation or FPS. Original loadRes bbox/area policy
is unchanged because non-mask fields are retained.

Keep `private_pair_records` private. The diagnostic is conditional on selected
native matched GT (selected/380, potentially fewer than380), not an all-GT
geometry guarantee, AP substitute or bound. The original diagnostic's scope
and interpretation remain intact. No raw reference, source path, private pair
or other operation artifact is automatically approved for publication.

Finite verification uses seven public behavior tests with synthetic JSON, stub
official/geometry functions, and the ordinary original stdlib merge/relabel/
delta helpers. It imports no Torch, NumPy or pycocotools. It never reads actual
data, reports, Ops, qualification, bound SOURCE or weights and performs no real
COCO evaluation, matching, GPU inference or SSH. No old test suite is rerun.
Test fixtures intentionally use non-RLE synthetic tags that no real mask API
ever decodes. An additional AST check verifies stdlib-only imports and the
original helper calls; it cannot establish numerical correctness on DGX.

Run only these synthetic checks with the local stdlib-capable interpreter:

    python -B test_metadata.py

All actual scientific execution, CPU-resource admission, evidence persistence,
Root acceptance and later publication decisions remain outside this draft.
