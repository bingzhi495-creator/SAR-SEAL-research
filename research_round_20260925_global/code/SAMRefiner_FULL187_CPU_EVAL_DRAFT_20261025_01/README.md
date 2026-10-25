Current Root update: original full187 inference and independent native/R/SAM COCO evaluation are now accepted; see [actual result and scope](../../U57_SAMREFINER_FULL187_COMPARISON_EXECUTION_20261025.md). Source was copied before binding. The following candidate/development status is historical.

# Independent full187 native/R/SAM CPU comparison candidate

UNBOUND / NOT SCIENTIFICALLY RUN. Four stdlib dictionary-contract tests pass.
No actual GT, prediction file, model, NumPy, pycocotools, Torch, SSH or AP was
used while developing this candidate. Synthetic dictionaries are not results.

The worker evaluates exactly native_official, fixed R and the full187 SAM output
with unchanged native S0/class/query/list order. It directly executes the closed
N28 `official_metrics` definition and original `official.coco_metrics` helper;
it never enters N28.cpu, U67 or U70 and never invents the four E50 shards those
different experiments require. All methods use the actual original DEV187/380,
COCO segmentation evaluation, category1 and maxDets[1,10,100], without score
cutoffs. Both old native and R complete seven-key/12-stat dictionaries must
reproduce exactly before SAM is evaluated. No tolerance is silently introduced.

## Actual CLI and Root responsibility

`python -B -u worker.py --spec <actual-spec-path> --spec-sha <actual-SHA256>`

Root must first independently admit the original full187 SAM execution and its
source/artifact/process closure, then ensure every other scientific leaf is
reaped. No ordinary JSON or this worker can establish that original qualification.
Root must supervise this CPU child with the real external finite deadline,
CPU0/1, two threads, whole owned-group RSS16GiB/freeRAM16GiB checks and process
birth/group cleanup. The internal Control checks actual psutil resources and
Torch absence at cooperative checkpoints; it is not external live supervision.

Require the explicit environment CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=2,
MKL_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2. Root sets CPU affinity0/1 before launch.
Runtime is the admitted Linux/aarch64 Python3.12.14, NumPy1.26.4, psutil6.1.1.

## Exact SPEC fields

- schema: ROOT-SAMRefiner-full187-CPU-eval-spec-v01
- operation: CPU_SAMREFINER_FULL187_FIXED_NATIVE100_COCO_V01
- run_id: an exclusive string beginning SAMRefiner-full187-CPU-v01-
- source_refs: exactly worker, eval_body, official; each a real path/bytes/SHA256
  ref. eval_body and official content pins are in SOURCE_PINS.
- library_sources: exactly numpy, psutil, pycocotools, mask_api, mask_extension,
  coco_loader, cocoeval. All are real selected module entry/extension refs.
- input_refs: exactly context, legacy_context, legacy_result, native_predictions,
  R_predictions, DEV, SAM_predictions, SAM_report. Root re-closes the existing
  registered original inputs and actual newly admitted SAM outputs before launch.
  The two contexts are joined by content refs; their historical paths are never
  followed. Only original image_ids and explicit references are used, not tensors.
- root_lease_ref: an exclusive Root-created lease ref described below.
- runtime_pythonpath_root: the actual parent of the selected pycocotools package.
- output_directory: a fresh canonical POSIX directory with an existing parent.
- prelock: the exact object below.
- sources: path-sorted deduplicated exact union of source_refs, library_sources,
  input_refs and root_lease_ref. Conflicting refs at one path are rejected.

Prelock:

```json
{
  "limits": {
    "allowed_cpus": [0, 1],
    "threads": 2,
    "maximum_RSS_bytes": 17179869184,
    "minimum_available_RAM_bytes": 17179869184,
    "deadline_seconds": 1800
  },
  "methods": ["native_official", "R", "SAMRefiner"],
  "score_policy": "ORIGINAL_NATIVE_S0_UNCHANGED",
  "annotation_policy": "ORIGINAL_DEV_UNCHANGED",
  "expected_images": 187,
  "expected_annotations": 380,
  "expected_candidates": 18700,
  "GPU_hidden": true,
  "new_training": false
}
```

The Root lease must contain schema Root-SAMRefiner-full187-CPU-eval-lease-v01,
status ROOT_GRANTED, matching run_id, worker_ref, prelock, output_directory,
SAM_predictions_ref and SAM_report_ref, plus single_attempt=true,
main_scientific_processes_reaped=true, exclusive_project_scientific_leaf=true.
Root verifies those facts externally; the worker only checks the exact join.

## Output and limitations

REPORT.json preserves each full COCO12 dictionary and raw per-IoU precision AP.
Differences SAM-minus-native and SAM-minus-R use percentage points. Original
undefined -1 statistics remain in both raw arrays; a difference is null if either
method is undefined. SAM's existing shared-environment cost scalars are retained
with the actual report ref; this CPU evaluation adds no standalone FPS claim.

Output remains COMPLETE_PENDING_ROOT_ACCEPTANCE / qualified=false. Its actual
tool/SSH/worker terminal, input/source/output bytes and owned-process cleanup
still require Root review. Failures retain partial method results and accurately
mark possible partial AP; no automatic retry, rescore, annotation conversion,
filter, best-output selection or innovation/TGRS claim occurs. The report has
private input refs and must not be published wholesale.

## Limited tests

`python -B -m unittest -v test_contract`

Four tests cover changing only segmentation while rejecting score/JSON-type
changes, duplicate native query identities, native-grid changes, and undefined
statistic differences. Each new behavioral guard was observed failing before its
implementation and then passing. These checks do not establish real scientific
execution, COCO equivalence or SAM performance.
