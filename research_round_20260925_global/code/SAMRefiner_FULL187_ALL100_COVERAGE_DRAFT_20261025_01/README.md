## 2026-10-25 实际状态更新

全380 GT、每图all-100覆盖诊断已实际执行并通过Root接收；结果见../../U57_ALL100_ORACLE_COVERAGE_ACTUAL_20261025.md。GT参与oracle选择，该结果不是AP或部署方法。

以下保留独立发布副本在绑定前的开发说明；其中未执行标签是当时快照。冻结科学原件未改写。

# All100 exported-mask oracle coverage diagnostic candidate

UNBOUND / NOT SCIENTIFICALLY EXECUTED. Four stdlib synthetic-metadata tests pass.
No actual GT/predictions/weights, scientific library, SSH, masks or AP were used
in this candidate's development. Root owns later admission, execution and review.

## Scientific scope

This is ORACLE_DIAGNOSTIC_POSTHOC: GT IS used to choose the maximum-IoU query for
each GT as a diagnostic. GT is not used for model inference. Original exported
predictions, query identities, classes and S0 are neither changed nor reordered.
The diagnostic choices must never replace deployed outputs, formal AP, ranking,
scores or current main training. It is not deployable, not AP and not one-to-one
recall: one candidate may be the diagnostic best for more than one GT.

For each image/method, all100 already exported single-mask RLEs enter the genuine
official mask.iou matrix against all original GT annToRLEs in that image. Each GT
gets its max IoU and earliest original export position when IoUs tie. Separately,
that position's original S0 rank uses descending score, then original export
index for score ties. IoU ties never use S0 to improve the reported best rank.
This only examines one already exported mask per query; it does not inspect the
SAM author's three internal mask branches or bound their potential performance.

All380 GT remain in the main denominator, including native-unmatched GT. The
report separately describes native matched nonignored, native unmatched
nonignored and native effectively ignored groups. Original iscrowd/ignore flags
are read at runtime, counted and retained in private rows; they are not assumed
zero. Official mask.iou receives original per-GT iscrowd flags, so any crowd GT
uses COCO crowd-overlap semantics rather than ordinary union IoU.

Safe aggregates include max-IoU mean/median/quartiles, .50/.75/.90/.95 coverage
counts/fractions and stable best-candidate rank statistics/bins. Zero-max-IoU GT
are counted; their arbitrary earliest-tie rank is not quality evidence. A rank
summary conditioned on maxIoU>=.50 is explicitly separate. These summaries do
not uniquely explain AP loss or establish current FULL improvement/innovation.

## Implementation and SPEC

Only native COCO is recomputed once, with its actual evaluator retained and
complete legacy seven-key/COCO12 result reproduced exactly. No new-method AP is
recomputed. Closed geometry.select_native_pairs supplies the true native IoU.50
match group; geometry.diagnose, Omega, fake method labels and fake shards are
never used. The actual official overlap matrix itself does no score filtering.

CLI: `python -B -u worker.py --spec ACTUAL_SPEC --spec-sha ACTUAL_SHA256`.

Exact SPEC keys:

- schema: ROOT-SAMRefiner-full187-all100-oracle-coverage-spec-v01
- operation: CPU_SAMREFINER_FULL187_ALL100_ORACLE_COVERAGE_V01
- run_id: exclusive prefix SAMRefiner-full187-oracle-v01-
- source_refs: worker, cpu_helpers, eval_body, official, geometry_diagnostic.
  PINS fixes all four reused source identities. cpu_helpers is the real closed
  18971-byte previous CPU worker; only its Control/source/population helpers run,
  never its three-method AP entry.
- library_sources: numpy, psutil, pycocotools, mask_api, mask_extension,
  coco_loader, cocoeval; actual selected COCO-overlay refs.
- input_refs: context, legacy_context, legacy_result, native_predictions,
  R_predictions, DEV, SAM_predictions, SAM_report; the same eight actual prior
  refs, independently re-closed before and after the new job.
- root_lease_ref, runtime_pythonpath_root, prelock, output_directory, sources.

Prelock is an exact copy of the stdlib-only PRELOCK constant. Sources are the
path-sorted deduplicated union of the direct source/library/input/lease refs.
Historical context paths are never followed. Output is fresh and contains no
input. Pure metadata pre-binding interface is validate_spec(spec,h), where
h=bootstrap(spec['source_refs']['cpu_helpers'], alias).

Lease schema: Root-SAMRefiner-full187-all100-oracle-coverage-lease-v01, with
status ROOT_GRANTED, matching run_id/worker_ref/prelock/output_directory,
SAM_predictions_ref/SAM_report_ref and single_attempt=true,
main_scientific_processes_reaped=true, exclusive_project_scientific_leaf=true.
An ordinary lease JSON does not prove actual execution admission; Root does.

Runtime scientific imports occur only inside main after real source admission.
Root provides a fresh Linux/aarch64 Python3.12.14 child, NumPy1.26.4/psutil6.1.1,
explicit CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=MKL_NUM_THREADS=
OPENBLAS_NUM_THREADS=2 and CPU0/1 affinity. Reused Control checks workerRSS16GiB,
availableRAM16GiB, CPU01, noTorch and1800s cooperatively. Root must also provide
external live owned-group supervision, deadline, original terminal and actual
process-birth/group cleanup. No primary training recipe or limit changes.

## Outputs

- PRIVATE_GT_ORACLE_RECORDS.json: per-GT/query identities, original score/rank,
  max overlap, export position, ties and native/crowd/ignore groups. Keep private.
- SUMMARY.json: numeric aggregates, protocol/scope and limitations without raw
  identities or paths. COMPLETE_PENDING_ROOT_ACCEPTANCE / qualified=false.
- REPORT.json: native calibration, source refs, stage/resource and output refs,
  actual worker birth and return code. It has private refs; do not publish whole.

Root must verify actual tool/SSH/worker0, all sources/outputs and owned-process
cleanup before accepting any output. Failure preserves stage/traceback and any
written pending artifacts; no automatic retry or use of oracle-selected masks
as a better experimental method. Native COCO mathematics is reported honestly.

## Limited tests

`python -B -m unittest -v test_contract`

Four red-to-green metadata tests cover all100 inputs, stable S0/tie ranks with
unchanged scores, first-export IoU ties independently of rank, and full380 GT
denominator including six synthetic unmatched cases. They do not verify real
mask.iou numerical behavior, AP, dataset integrity or experimental qualification.
