Current Root update: original full187 inference and independent native/R/SAM COCO evaluation are now accepted; see [actual result and scope](../../U57_SAMREFINER_FULL187_COMPARISON_EXECUTION_20261025.md). Source was copied before binding. The following candidate/development status is historical.

# Full187 SAMRefiner candidate worker

Current Root status: ORIGINAL FULL187 EXECUTION STARTED / NOT ROOT ACCEPTED / NO SAM AP. The original source was copied into this ordinary publication directory before execution binding.

Historical candidate status: UNBOUND / NOT EXECUTED / NO SAM AP. This new candidate neither grants a GPU lease nor changes any main-training recipe, frozen source, failed attempt, or prior single-image acceptance. Root owns scientific execution and qualification.

The worker loads author ViT-H FP32 once and visits the historical 187-image roster. Each image submits all original 100 native masks, including any empty masks, in their native export order to one unchanged author five-iteration call. No score filtering, empty passthrough, GT prompts, candidate rematching, score replacement, resizing to RGB400, or local training occurs. Output preserves the complete native list order and all non-segmentation metadata; only each original-grid mask RLE is replaced. The old conservative fixed-candidate adapter is not edited or used.

JPEG bytes are independently admitted by Root. Per image the worker decodes current PIL RGB, saves a lossless PNG, and requires exact current PIL-to-PNG-to-author-OpenCV pixel identity. This does not establish historical teacher RGB400 equivalence. Official pycocotools decodes/encodes masks; no custom compressed-RLE codec is introduced.

Each image must observe one image-encoder pre/post call and five prompt-encoder/mask-decoder pre/post calls. The per-image author return tuple includes GPU auxiliaries; the whole tuple and transient tensors are released after CPU RLE encoding. Reports contain CPU RLE metadata and scalar diagnostics. Final cleanup retains the already tested v03 private PyTorch workspace-clear route and requires allocated/reserved both exactly zero. Private API absence or nonzero cleanup fails normally.

## Worker invocation and SPEC

Run the qualified fresh Linux interpreter with `-B -u worker.py --spec <absolute-path> --spec-sha <full-sha256>` only under Root's finite external supervisor. Module import and the provided tests use only the Python standard library. Scientific libraries are imported only within `main` after admission.

Required top-level fields:

- `schema`: `ROOT-SAMRefiner-full187-shared-worker-spec-v01`.
- `operation`: `GPU_SAMREFINER_FULL187_NATIVE100_SHARED_V01`.
- `run_id`: unique string beginning `SAMRefiner-full187-v01-` with no slash.
- `exclusive_project_scientific_leaf: true`, `whole_device_exclusive: false`; no `exclusive_GPU` field.
- `source_refs`: exact roles `worker`, `author_refiner`, `author_utils`, `sam_init`, `sam_registry`, `sam_model`. The five author roles retain fixed existing SHA256 pins from author commit `4bb7f95738c7a2e805f189210fb09e02a2197557`; Root binds the new worker bytes.
- `library_sources`: exact roles `torch`, `numpy`, `psutil`, `cv2`, `PIL`, `pycocotools_mask`; each binds the actual selected module's `__file__`. This is selected-file closure, not complete Python/native/shared-library provenance.
- `input_refs`: exact roles `native_predictions` and `image_manifest`.
- `checkpoint_ref`, `checkpoint_stage_ref`, `verified_checkpoint_sha256`, `root_lease_ref`, `gpu_background_observation_ref`.
- `output_directory`: fresh canonical absolute POSIX directory, outside all admitted sources/inputs.
- `sources`: the exact path-deduplicated, path-sorted union of the preceding file refs plus all 187 JPEG refs. No extra parent/source roles belong in this worker closure; the external supervisor has its own source closure.
- `prelock`: exact object below.

Every file ref is exactly `{path, bytes, sha256}`, with canonical absolute POSIX path, nonnegative integer byte count, and full lowercase SHA256.

```json
{
  "author_commit": "4bb7f95738c7a2e805f189210fb09e02a2197557",
  "model_type": "vit_h",
  "precision": "float32",
  "amp": false,
  "allow_cpu_fallback": false,
  "params": {"iters": 5, "margin": 0.0, "gamma": 4.0, "strength": 30, "use_samhq": false},
  "limits": {
    "allowed_cpus": [0, 1], "threads": 2,
    "maximum_RSS_bytes": 17179869184,
    "minimum_available_RAM_bytes": 17179869184,
    "deadline_seconds": 7200,
    "allocator_bytes": 17179869184
  },
  "startup_min_available_RAM_bytes": 51539607552,
  "resource_plan": "FULL187_FIXED_NATIVE100_SHARED_V01",
  "empty_policy": "AUTHOR_ALL100_UNFILTERED",
  "expected_images": 187,
  "queries_per_image": 100,
  "seed": 2026102503
}
```

The seed is a prospective Root choice: the validator permits any integer in `[0, 2**32)`, and the example uses `2026102503`. Python/NumPy/Torch seeds are set; bitwise deterministic reproducibility is not claimed. No parameter or budget changes are made after viewing a result. The independent SAM7200-second/16GiB plan does not replace main training's 1800-second/4GiB plan.

`image_manifest` is a direct JSON list of 187 exact records `{ordinal, image_id, shape, jpeg_ref}`. Ordinals are the existing roster order 0–186; `shape` is original `[H,W]`; image IDs are distinct. Native predictions are the unchanged global JSON list of 18,700 dictionaries with original IDs/category/raw S0/compressed RLE. Every roster image must retain query IDs 0–99 exactly once, with original export permutation preserved. Empty RLEs are accepted for direct author processing.

## Closed stage, lease and shared context

Checkpoint bytes/SHA must equal the complete already verified official ViT-H asset: 2,564,550,879 bytes, SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`. `checkpoint_stage_ref` retains the existing closed `SAMRefiner-CHECKPOINT_STAGE-v1`, `ROOT_VERIFIED_COMPLETE`, matching checkpoint ref, verified SHA, `vit_h`, fixed author commit and original Meta URL.

The new lease has `schema: Root-SAMRefiner-full187-shared-lease-v01`, `status: ROOT_GRANTED`, matching `run_id`, operation, current `boot_id`, `worker_ref`, prelock, output directory and background observation ref; also `single_attempt: true`, `main_scientific_processes_reaped: true`, `exclusive_project_scientific_leaf: true`, `whole_device_exclusive: false`, and `scope: full187_frozen_native100_no_GT_no_AP`. No `exclusive_GPU` field is accepted. A lease record does not prove live exclusivity: Root must establish all project numerical leaves are actually reaped and independently supervise the owned process group.

Background observation retains `SAMRefiner-shared-GPU-background-observation-v02`, same boot, original successful nvidia-smi return code, GPU index0, explicit nonexclusivity, UTC and process birth/kernel-name/observed-MiB metadata only. It does not reserve physical GPU capacity, control other services, or establish a speed benchmark.

## Outputs and external acceptance

`REPORT.json` schema is `SAMRefiner-full187-shared-worker-report-v01`. A worker success has status `FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW`, `worker_exit_code: 0`, `inference_completed: true`, `images_completed: 187`, `candidates_completed: 18700`, model-load count1, exact final CUDA allocated/reserved0, and two full output refs:

- `output_predictions_ref`: `refined_predictions.json`, original global list order and non-segmentation fields preserved.
- `output_image_reports_ref`: `image_reports.json`, 187 entries with original roster identity/shape, global candidate indices and query permutation, input/output empty positions, current pixel equality/SHA, derived `PNG_ref`, author call1, five configured rounds, observed1/5/5 hook deltas and shared-runtime diagnostics.

Generated PNGs are under fresh `png/`; their refs are in image reports. `STAGES.jsonl` and report preserve failure stage, partial counts and original errors. No automatic retry, skip or reuse of failed partial outputs occurs. No AP/GT is evaluated. `author_internal_return_order_verified` remains false: shape/metadata contracts plus fixed author-source order inspection must be assessed separately; wrapper IDs alone do not prove correspondence.

The external supervisor must retain its actual original tool/SSH/worker exits, finite7200-second timeout, CPU0/1/two-thread policy, sampled owned-group RSS≤16GiB and available RAM≥16GiB, startup48GiB, birth-aware owned-group cleanup, frozen source/input post-checks and output closure. Worker self-checks are sampled and do not certify those external conditions. Original success and Root qualification remain mandatory before outputs can become evaluation parents.

## Limited tests

Three standard-library synthetic tests cover original native population/permutation preservation, segmentation-only replacement, and one author-boundary call forwarding all100 including empties with fixed defaults. They use no Torch, NumPy, SAM, COCO, PIL, cv2, true images, weights or GT. They provide no neural/runtime/AP evidence. `python -B -m unittest test_population_contract -v` is the finite test entry.
