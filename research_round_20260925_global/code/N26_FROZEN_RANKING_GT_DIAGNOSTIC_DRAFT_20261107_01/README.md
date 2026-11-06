# Frozen N26 ranking: integer / candidate-float GT diagnostic draft

Status: **source prepared, not scientifically executed**. This is an independent CPU diagnostic for four ended, frozen E24 result lists. It neither evaluates the running full DiffSAR training nor consumes a checkpoint. Existing native full-training, native endpoint-prediction and native CPU COCO v1 sources are unchanged.

The scientific question is whether the three frozen N26 confidence-ranking improvements relative to the same frozen R survive a change from the established integer GT geometry to a local candidate floating-point geometry. The candidate float GT is not an official alternative protocol or a demonstration of physical truth. The established integer protocol remains primary. This diagnostic cannot establish a new method's novelty, cross-dataset benefit or TGRS readiness.

## Frozen experiment

Operation: `FROZEN_N26_RANKING_INTEGER_FLOAT_GT_DIAGNOSTIC`.

All cases use the same already-exposed internal DEV187 population, category 1, 380 annotation identities. Four complete frozen prediction arrays contain 18,700 records each, exactly 100 per original image. These are the older fixed-query R/N26 outputs. They must not be confused with the future native model's potentially variable prediction population.

| Role | Frozen integer AP, percentage points |
|---|---:|
| Common R | 68.09938494745545 |
| Seed 202610172601 | 69.42361315215425 |
| Seed 202610172602 | 69.63714218423894 |
| Seed 202610172603 | 69.55134548319735 |

The new Root diagnostic prelock fixes an absolute tolerance of **1e-8 percentage points**. It does not modify any old validator or historical result. The program reevaluates all four integer cases in the table order. All four APs must reproduce within that tolerance before the first candidate-float COCO call. Failure preserves actual completed integer COCO12 statistics, marks the diagnostic failed and leaves every float case unattempted. There is no tolerance expansion, retry, seed selection or score tuning.

If the integer gate passes, all four candidate-float cases run in the same order. For each GT version the report retains the three prespecified seed APs, the same R AP, each paired quality-minus-R delta, the mean of all three deltas and their sample standard deviation (denominator n-1). It does not average per-shard AP. Undefined original summary values remain raw -1 and become `null` only in a clearly labeled percentage view. If any required AP is undefined, the corresponding paired value is null and no partial-seed mean/SD is substituted.

## Original input roles and checks

Root supplied an independent private locator ledger, schema `N26-independent-frozen-numeric-input-locator-ledger-v1`, 17,191 bytes, SHA256 `3e8825926f76fe571e20ed7c905cfe6b5f3edc56e0ba3fa17992d0b70895b9c6`. Only that metadata was read during this preparation; no referenced numeric input body or qualification chain was followed. This draft contains no private input paths or bundled data.

The new execution spec must supply exactly these nine `input_refs`:

- `integer_GT`, `candidate_float_GT`, `R_predictions`;
- `quality_202610172601`, `quality_202610172602`, `quality_202610172603`;
- `score_audit_202610172601`, `score_audit_202610172602`, `score_audit_202610172603`.

Runtime paths may be newly staged on the chosen platform; each size and SHA256 must match its exact original role in the pinned locator. The report records the original ledger refs and execution refs separately. The integer GT content identity is 118,319 bytes / SHA256 `5657600d6ea198f79bd5d557d2a1aadce5bec031d2ecf72a98716248d1183a55`; the local candidate float GT is 445,643 bytes / SHA256 `8d98e6f49200d5390c17bee66697de1f582133d08063259a561ed5f710be7b6c`. The locator is a reference export, not a new scientific qualification.

Before **any** COCO API call, the worker parses the two pinned raw GT JSONs and joins them by image/annotation ID. It checks unique187 images, original dimensions, category1, distinct380 annotations, unchanged image/category/crowd and all other non-geometric fields. Only annotation `segmentation`, `bbox` and `area` may differ; other dataset/image metadata must agree. It never rounds or reconstructs geometry. Each of the eight model/GT cases then builds a new original COCO object from an independent `deepcopy` of the corresponding raw JSON and calls `createIndex`. This prevents `COCOeval._prepare` mutations of segmentation/ignore from leaking between cases or changing the raw comparison.

All four prediction arrays must contain 18,700 records and exactly100 records for every original image, category1, finite frozen scores, original-sized compressed UTF8 RLE and finite four-value boxes. The three quality arrays are compared to common R **at every original record position**, with every field except `score` required to be exactly JSON-identical, including any explicit query fields. No sorting, filtering, score normalization, score multiplication, 0-to-1 remapping or top100 preselection occurs. The original scores are consumed as frozen. Comparison of full records/order does not independently establish implicit query uniqueness.

The three SCORE_AUDIT files are read only to close their pinned byte identities and are not parsed. No product is recomputed with a different dtype. This task does not newly verify qhat inference or qhat multiplication. Historical job-specific source confirms that QUALITY serialization copies R records and changes only score, but that source fact does not replace the future actual byte/record checks.

## Canonical COCO wiring

The worker imports only the **pure `evaluate_population` function** from the frozen ordinary `DIFFSAR_NATIVE_CPU_COCO_ADAPTER_DRAFT_20261107_01/worker.py` (17,615 bytes, SHA256 `f9fc79403d12d8840c3269b04ba22d7584b8eb5e3c592ead2b6359ad7df65748`). It never calls that module's native `main`, validates no native37050 admission and reads no native checkpoint/report/qualification. A native operation is rejected by this new spec interface.

The reused function deepcopies the complete prediction list, removes `bbox` for segm, calls original `COCO.loadRes`, and uses original `COCOeval(..., 'segm')` with all187 image IDs, category `[1]` and `maxDets=[1,10,100]`. It calls `evaluate`, `accumulate`, `summarize` once for the full population. Mask area therefore comes from segmentation. No local IoU, assignment, interpolation or AP implementation is added. All records remain present; maxDets is solely an evaluation setting.

Root must explicitly pin the actual installed source/compiled-file refs for all five `library_sources` keys: `numpy`, `pycocotools_coco`, `pycocotools_cocoeval`, `pycocotools_mask`, `pycocotools_extension`. The worker imports `numpy`, `pycocotools.coco`, `pycocotools.cocoeval`, `pycocotools.mask`, `pycocotools._mask` and compares their actual `__file__` content refs. A package version string is only recorded. The original old encoder's actual source identity is not provided by this locator and remains unverified; integer AP reproduction is the prelocked functional gate for this new canonical reevaluation, not a claim of historical binary/source identity.

## Future Root interface

No owner, SSH workflow, training controller or additional qualification framework is supplied. **A real new Root prelock, newly closed ended-input refs, actual CPU owner/birth and enforced resource budget are external prerequisites**, not evidence granted by this source or its metadata fields. This preparation grants no execution permission and performs no scientific computation. Root chooses the existing DGX parent CPU mode or a separately strict local CPU mode during a real qualified training gap.

There is one new Root-pinned spec, schema `Root-frozen-N26-integer-float-GT-diagnostic-spec-v1`, with exact keys:

```text
schema, operation, run_id, prelock,
source_refs:{worker,canonical_coco_adapter},
library_sources:{the five keys above}, locator_ledger_ref,
input_refs:{the nine roles above}, output_directory,
CPU_budget:{affinity,threads,RSS_limit_bytes,free_RAM_floor_bytes,start_RAM_bytes,deadline_seconds},
external_Root_conditions:{
 new_diagnostic_prelock_granted:true,
 old_frozen_input_refs_newly_closed:true,
 exclusive_CPU_only_owner:true,
 actual_owner_birth_verified:true,
 CPU_budget_enforced_externally:true,
 canonical_COCO_sources_explicitly_selected:true
}
```

`run_id` begins `N26-frozen-ranking-GT-`. `prelock` is the exact module `PRELOCK` object. Every file ref has exactly `{path,bytes,sha256}` with a canonical absolute path for the execution platform. The fresh output directory must not contain any selected input/source and must not already exist. Runtime invocation is `python -B worker.py --spec ABSOLUTE_SPEC --spec-sha ROOT_PINNED_SHA256`. This is an interface example, not a command executed in this task.

The worker checks the pinned spec, own/helper source, locator, actual math libraries and all nine input contents, then rechecks them after computation. It requires `CUDA_VISIBLE_DEVICES` empty and OMP/MKL/OpenBLAS thread variables equal to the Root budget. It never imports Torch. Actual affinity, RSS/free-memory monitoring, original process/birth ownership and hard timeout are enforced by the **external Root CPU owner**; this worker does not fabricate measurements or install a new supervisor. Its own elapsed deadline checks occur at finite boundaries before/after the complete diagnostic. Both Windows and Linux paths are handled through the current platform's `pathlib`; Root must supply the correct staged execution paths and actual compiled-library identity for that platform.

## Report, failure and verification status

Output `REPORT.json` uses `frozen-N26-integer-float-GT-diagnostic-report-v1`; own `qualified` remains false even when `status=DIAGNOSTIC_COMPLETE_PENDING_ROOT` and worker exit0. Root actual postcheck is still required. The report records actual source refs, all input read/parse states, case-level original COCO12 stats, elapsed math costs, reproduction differences, paired summaries and failures. No best case is selected.

Input `bytes_read` / `JSON_parsed` and case `AP_computed` / `COCO_math_completed` follow `null` (unattempted), `false` (attempt begun/incomplete), `true` (completed). `COCO_math_started` identifies an attempted case. Audit `JSON_parsed` always stays null. A completed byte read can be true even when its hash is wrong; parsing does not begin on a wrong identity. A computed AP remains true and its actual metrics survive a later resource or post-source failure, while status becomes `FAILED_PRESERVED_NO_RETRY`. Failure before validated fresh output creation is retained by the external original exit/stderr; it cannot create a valid scientific report. An external hard kill can leave the last atomically saved intermediate report, which the original supervisor must treat as incomplete.

Only stdlib and API/IO stand-ins have been tested here. The tests cover integer-gate failure blocking all float calls; the fixed three-seed paired mean/sampleSD and undefined handling; GT non-geometric identity rejection; all frozen non-score record/order preservation; independent GT working state despite API mutation; input identity/parse failure states; original-role/staged-content closure; rejection of a native endpoint operation; and preserving returned math completion if subsequent summary-format validation fails. Synthetic API return values and the string `stand-in` are test fixtures, not valid RLE or actual COCO/AP evidence.

**Not executed:** actual input-body reads/hash closure, both real GT identity checks, frozen prediction comparisons, actual canonical scientific imports, any of the eight real COCO cases, integer AP reproduction, float AP, true CPU resource qualification or GitHub remote verification. No old numeric result, source, lease or qualification was changed. The only current verified outcome is prepared source and passing finite stdlib/API tests.
