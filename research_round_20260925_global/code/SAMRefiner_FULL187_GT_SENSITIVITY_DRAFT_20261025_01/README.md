## 2026-10-25 实际状态

已真实执行，原工具/SSH/worker均0、22个worker来源和7个parent来源通过后核，严格出生身份及进程组均已回收；Root已接收。结果见[追加两GT诊断](../../U57_INTEGER_FLOAT_GT_SENSITIVITY_ACTUAL_20261025.md)。首E50 FULL/PLAIN敏感性仍未完成。下方未执行标签保留为绑定前开发快照，冻结科学原件未改。

# Additional paired integer/local-float GT sensitivity candidate

UNBOUND / NOT SCIENTIFICALLY EXECUTED. Five stdlib synthetic metadata/count tests
pass. Development read only the stated policies/provenance summaries and closed
source definitions, not actual GT/prediction payloads, models or weights. No
scientific library, SSH, rasterization or AP was run in this task.

## Scope and fixed comparison

This diagnostic is POSTHOC_ADDITIONAL_GT_SENSITIVITY, added after the base SAM
negative results. It is not the original preregistration and does NOT fulfill or
replace U57_PROSPECTIVE_ANNOTATION_SENSITIVITY_POLICY_20261025.md's promised first
E50 FULL/PLAIN analysis. first_E50_FULL_PLAIN_sensitivity_completed remains false.
Current main training/primary integer GT, saved outputs, scores and prior AP
remain unchanged. No score, sample, annotation version or best output is chosen
after seeing the results.

The three frozen native/R/SAM populations, each18700 candidates, are evaluated
with identical original COCO segm/maxDets[1,10,100] rules against both original
integer GT and ONE separately derived local-candidate float GT. Integer native/R
must reproduce their complete legacy seven-key/COCO12 dictionaries exactly.
Every method's full12 stats/per-IoU results, float-minus-integer differences and
method contrasts under each GT are retained. Undefined official -1 stats remain
raw; differences with undefined endpoints are null.

Float GT must be a full deepcopy of integer GT, preserving187 images/380 IDs,
category, order, dimensions and every nongeometry field. Only segmentation,
bbox and area are restored unchanged from the one exact source member. The
worker independently checks all380 against the one existing int()/bbox-extrema/
shoelace rule. It does not test rounding, vertex permutations or alternative
matching rules, re-read the archive, authenticate it, or change the mapper report.
The original source file's byte identity and the mapper's original execution
must be verified by Root outside this worker.

For all380 annotation pairs, official annToRLE/mask.decode rasterizes both GTs
on the unchanged image grid. Private rows retain raster areas, intersection,
union, XOR, IoU, area ratio and raw continuous-area/bbox/polygon differences.
Empty masks are not dropped: both-empty IoU and zero-integer-area ratios are
explicitly undefined and counted. The safe summary retains the complete380
denominator and defined/undefined counts. Continuous polygon differences are
not assumed to imply pixel changes.

AP sensitivity jointly includes polygon/raster, bbox/continuous-area and COCO
matching/area-bin effects. This is not a unique causal decomposition, physical
ship-contour truth, certified official archive, new TEST, repeated seed or proof
of current FULL improvement/innovation. Integer results remain primary even if
the float values are more favorable.

## SPEC / actual execution interface

CLI: `python -B -u worker.py --spec ACTUAL_SPEC --spec-sha ACTUAL_SHA256`.

- schema: ROOT-SAMRefiner-full187-GT-sensitivity-spec-v01
- operation: CPU_SAMREFINER_FULL187_PAIRED_INTEGER_FLOAT_GT_V01
- run_id: exclusive prefix SAMRefiner-full187-GT-sensitivity-v01-
- source_refs: exactly worker, cpu_helpers, eval_body, official. PINS fixes the
  reused19k CPU Control/identity helpers and two actual official definitions;
  their original outer entry points are not invoked.
- library_sources: exactly numpy, psutil, pycocotools, mask_api, mask_extension,
  coco_loader, cocoeval, using the same admitted selected library sources.
- input_refs: exactly context, legacy_context, legacy_result, native_predictions,
  R_predictions, DEV, SAM_predictions, SAM_report, FLOAT_DEV,
  FLOAT_mapping_report. The first eight are the actual previously admitted refs.
- root_lease_ref, runtime_pythonpath_root, prelock, output_directory, sources.

Prelock is the exact stdlib-only PRELOCK constant. Sources are the path-sorted
deduplicated direct union: four sources + seven libraries + ten inputs + lease,
normally22 distinct files. Root binds actual refs rather than invented values.
Output is a fresh canonical POSIX directory containing no input. The pure
metadata interface is validate_spec(spec,h), where
h=bootstrap(spec['source_refs']['cpu_helpers'], alias).

Lease schema: Root-SAMRefiner-full187-GT-sensitivity-lease-v01. Required fields
are status ROOT_GRANTED, matching run_id/worker_ref/prelock/output_directory,
SAM_predictions_ref/SAM_report_ref/FLOAT_DEV_ref/FLOAT_mapping_report_ref,
single_attempt=true, main_scientific_processes_reaped=true and
exclusive_project_scientific_leaf=true. Actual source/process/mapping admission
is Root's responsibility; ordinary JSON does not self-certify qualification.

## Float mapping report contract

Schema is Root-SAMRefiner-DEV187-local-float-restoration-map-v1. It retains
integer_DEV_ref, float_DEV_ref, source_training_member_ref and the exact rule:

`int() truncation per polygon coordinate; bbox extrema max-minus-min; polygon area abs(shoelace)/2`

Required counts/flags are unique_mappings380, ambiguous0, missing0,
internal_IDs_population_preserved=true, official_archive_authenticated=false,
posthoc_additional_diagnostic=true. The source training member's known byte/SHA
pin is SOURCE_MEMBER; this is not whole-archive official authentication.

Report refs may retain their original Windows source/derived paths. The worker
joins integer_DEV_ref/float_DEV_ref to actual relocated DGX inputs by exact bytes
and SHA256, preserving original path strings and recording the relocations. It
does not rewrite the derived report or recursively follow its source paths.
Root separately verifies the real mapper source, original tool terminal and
generated output before granting the CPU lease.

## Resources, outputs and limited tests

Scientific imports occur only on the main execution path after genuine source
admission. Required runtime: Linux/aarch64 Python3.12.14, NumPy1.26.4/psutil6.1.1,
explicit CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=MKL_NUM_THREADS=
OPENBLAS_NUM_THREADS=2 and CPU0/1 affinity. Reused real Control cooperatively
checks workerRSS16GiB, availableRAM16GiB, noTorch and1800s. Root also provides
external live owned-group monitoring, deadline and actual original terminal/
birth/group reaping. No main training limit or recipe is changed.

- PRIVATE_GT_PIXEL_RECORDS.json: private annotation/image IDs and per-pair counts;
  contains no pixel arrays or full polygons. Do not publish these rows.
- SUMMARY.json: numeric six-evaluation results, paired differences and aggregate
  mask sensitivity, with posthoc scope and pending Root status, no private paths.
- REPORT.json: calibration, original refs/relocations, stage/resource/output refs
  and worker birth/exit. It contains private refs; do not publish wholesale.

All outputs remain pending actual external Root acceptance. Failure preserves
the true stage, possible partial/complete AP, traceback and written pending
artifacts; no automatic retry or result-driven GT selection.

`python -B -m unittest -v test_contract` runs five small synthetic tests for
population/ID preservation, the single rule, nongeometry/dimension invariance,
empty-raster denominators and byte-identical path relocation. Each new behavior
was observed failing then passing. These tests do not verify real mask numerics,
actual annotation lineage, COCO equivalence or scientific execution.
