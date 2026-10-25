## 2026-10-25 实际状态更新

固定native匹配几何诊断已实际执行并通过Root接收，374/380配对；结果见../../U57_FIXED_NATIVE_GEOMETRY_DIAGNOSIS_ACTUAL_20261025.md。

以下保留独立发布副本在绑定前的开发说明；其中未执行标签是当时快照。冻结科学原件未改写。

Root actual update: this fixed-native-pair CPU diagnostic has completed and been externally accepted. See [actual conditional result and limits](../../U57_FIXED_NATIVE_GEOMETRY_DIAGNOSIS_ACTUAL_20261025.md). The own source publication copy was saved before execution binding. The following candidate/development status is historical.

# Posthoc native-fixed-pair geometry candidate

UNBOUND / NOT SCIENTIFICALLY RUN. Four stdlib metadata-contract tests pass.
Development opened only closed source definitions and synthetic COCO metadata;
no actual GT/prediction payload, pixels, model, scientific library, SSH or AP.

This is POSTHOC_EXPLORATORY, proposed after observing the completed native/R/SAM
comparison. It cannot establish a unique cause of AP loss, physical ship-contour
truth, current FULL performance, or a core innovation. Keep all original main
training, SAM inference and AP results unchanged.

## Minimal actual computation

The worker reuses the fixed previous CPU worker's real Control and source/identity
helpers, the original N28.official_metrics/official.coco_metrics definitions, and
geometry_diagnostic.select_native_pairs/_decode. It does not call the previous
three-method AP runner, U67/U70, complete geometry.diagnose, or any model.

It recomputes only native COCO once, checks its complete seven-key/12-stat legacy
dictionary exactly, and retains that evaluator. Native IoU.50, area=all,
maxDets100 and nonignored GT matches alone fix image/query/GT pairs. Actual
cocoDt detection IDs are resolved to original query IDs; image traversal order
is not used as a query or dataset ordinal. R and SAM use that exact query even
when their new masks are empty or worse. No alternative query/rematching,
GT-derived prompts, rescoring, threshold tuning or oracle output replacement.

For each pair, decode original-grid predictions and the original annToMask GT,
then compute TP, FP, FN, IoU, prediction area / GT raster area, FP/GT and FN/GT.
The denominator uses raster pixels, never the annotation's continuous area.
Native pixel IoU must satisfy its original .50 match. Pass counts .50/.75/.90/.95
use exact integer TP/union comparisons. These conditional counts are not AP.
No Omega, uncertainty mask or fake method label is constructed.

SUMMARY retains all380 GT in the coverage denominator, showing selected,
ignored and unmatched counts separately. Geometry excludes the latter two;
this native-covered subset cannot represent all380 GT or isolate the effects
of the23 originally empty candidates. Summary reports pooled pixel counts/IoU,
instance mean/median/quartiles, area ratios and paired IoU differences.

## SPEC and execution

CLI: `python -B -u worker.py --spec ACTUAL_SPEC --spec-sha ACTUAL_SHA256`.

Exact top-level keys:

- schema: ROOT-SAMRefiner-full187-fixed-native-geometry-spec-v01
- operation: CPU_SAMREFINER_FULL187_NATIVE_FIXED_PAIR_GEOMETRY_V01
- run_id: exclusive prefix SAMRefiner-full187-geometry-v01-
- source_refs: exactly worker, cpu_helpers, eval_body, official,
  geometry_diagnostic; frozen content pins are the PINS constant.
- library_sources: exactly numpy, psutil, pycocotools, mask_api, mask_extension,
  coco_loader, cocoeval, using the admitted existing COCO overlay.
- input_refs: exactly context, legacy_context, legacy_result, native_predictions,
  R_predictions, DEV, SAM_predictions, SAM_report. Reuse the actual previously
  admitted eight input refs, re-closing them before this independent execution.
- root_lease_ref, runtime_pythonpath_root, prelock, output_directory, sources.

The prelock is an exact copy of the worker's stdlib-only PRELOCK constant.
Sources are the path-sorted deduplicated exact union of all refs above plus
root_lease_ref. Historical nested paths are never followed. Output is a fresh
canonical POSIX directory with an existing parent and contains no input.

Root lease schema is Root-SAMRefiner-full187-fixed-native-geometry-lease-v01,
with status ROOT_GRANTED, matching run_id/worker_ref/prelock/output_directory,
SAM_predictions_ref and SAM_report_ref, and single_attempt=true,
main_scientific_processes_reaped=true, exclusive_project_scientific_leaf=true.
Ordinary JSON alone does not prove those facts; Root establishes actual admission.

Root first closes all other scientific leaves and admits the original SAM and
AP sources/results. CPU execution requires explicit CUDA_VISIBLE_DEVICES='',
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2, actual CPU0/1 affinity,
Linux/aarch64 Python3.12.14, NumPy1.26.4 and psutil6.1.1. The real reused Control
checks CPU01, workerRSS16GiB, availableRAM16GiB and1800s cooperatively. Root must
also provide external live owned-group supervision, finite deadline and actual
birth/group reaping. Nothing here changes the primary training's limits.

## Outputs and claim boundary

- PRIVATE_PAIR_RECORDS.json: actual fixed GT/detection/query IDs and per-pair
  pixel counts/ratios. Private; contains no pixel arrays or output alternatives.
- SUMMARY.json: numeric aggregate and scope without private paths/identities.
  It remains COMPLETE_PENDING_ROOT_ACCEPTANCE, qualified=false, posthoc, DEV,
  no new-method AP and no innovation/TGRS claim.
- REPORT.json: native calibration, actual input/source refs, source/coverage
  stages, private-output refs, worker birth, resource observations and terminal.
  It contains private refs and must not be published wholesale.

All require actual external Root terminal/source/output/process qualification.
Failures retain the stage/traceback and any completed pending output, with no
automatic retry. Native COCO recomputation is marked accurately; the diagnostic
does not pretend that no AP mathematics occurred. New-method AP is not recomputed.

## Limited tests

`python -B -m unittest -v test_contract`

The four tests verify detection-ID/query mapping, ignored/unmatched denominator,
original-grid rejection, and no alternate-query fallback. They exercise the real
closed geometry metadata selector and plain dictionaries, not scientific pixel
code. Set SAM_GEOMETRY_METADATA_SOURCE to the identical closed13722-byte source
when running outside this workspace. Its bytes/SHA are checked before import.
No test outcome is a real geometry result or experimental qualification.
