# Unexecuted real one-image SAMRefiner trial candidate

Status: UNBOUND / FULL MODEL NOT EXECUTED. Only three stdlib metadata-contract tests and source/AST checks have been run. No Torch or NumPy import, checkpoint access, model execution, SAR input access, installation or SSH was performed to prepare this candidate.

This directory contains one thin trial worker, its three stdlib contract tests, and this README. It does not create a scheduler or resource-owner framework. Root must independently review and qualify the final source, prepare a private SPEC and grant a new external lease before any numerical execution.

## Fixed scientific scope

- Original DEV ordinal 0, image_id 4, original shape H=317/W=420.
- Reuse the already checked PNG, native100_masks.npy and native100_candidates.json through exact private file references and the closed first-input REPORT.
- Use fixed author commit 4bb7f95738c7a2e805f189210fb09e02a2197557 and the already tested fixed_candidate_adapter.py / samrefiner_bridge.py. Their SHA256 pins and the critical installed author-package source pins are enforced.
- Registry vit_h, original full100 export permutation, unchanged class and raw S0; default empty policy reject. The checked first image had 100 nonempty masks.
- FP32 CUDA model, eval, inference_mode, AMP disabled, TF32 disabled, no CPU model fallback.
- One batch call through the existing bridge with iters=5, margin=0.0, gamma=4.0, strength=30, point/box/mask and negative prompts enabled, non-HQ, non-DDP, non-training mode.
- No GT input, AP, matching, training, full187 result or accuracy comparison.

The standard same-name SAM distribution is not an acceptable substitute. Root's external binding must retain the previously verified author-installed package snapshot, including its other package files. The worker checks selected admitted files and the actual custom method/registry import origins; it does not establish a complete Python/native/shared-library closure.

## Private CLI and SPEC

CLI, for future Root-owned execution only:

```text
python one_image_trial.py --spec <absolute-private-spec.json> --spec-sha <full-SPEC-SHA256>
```

There is no runnable SPEC, qualified checkpoint, granted lease or actual input payload in this directory. The worker never downloads a weight, invents a verified digest, creates CHECKPOINT_STAGE, grants itself a lease, or selects a different resource plan.

The future SPEC schema is ROOT-SAMRefiner-one-image-trial-spec-v1 with operation GPU_SAMREFINER_ORIGINAL_ORDINAL0_ONE_IMAGE_TRIAL. It must contain:

| Field | Required prospective contract |
| --- | --- |
| run_id | Unique Root-assigned attempt identifier |
| source_refs | worker, resource_control, fixed_adapter, bridge, author_refiner, author_utils, sam_init, sam_registry, sam_model |
| library_sources | Selected torch/numpy/psutil/cv2/PIL module files |
| input_refs | png, masks, candidates, each exact path/bytes/full lowercase SHA256 |
| input_report_ref | Exact private first-input report accepted independently by Root |
| checkpoint_ref | Exact complete official ViT-H checkpoint path, 2,564,550,879 bytes and Root-verified full SHA256 |
| verified_checkpoint_sha256 | The independently verified full checkpoint digest, equal to checkpoint_ref.sha256 |
| checkpoint_stage_ref | Exact external CHECKPOINT_STAGE record |
| root_lease_ref | Exact external Root grant record for this one attempt |
| prelock | Fixed method/image settings and explicit separate resource plan described below |
| sources | Canonical sorted deduplicated union of all admitted references |
| output_directory | Fresh private absolute POSIX path outside admitted inputs |

Each file reference has exactly path, bytes and sha256. Source/input/weight paths are supplied by Root; none are inferred from a live cache or discovered by scanning. The input report must join all three file references and the measured RGB/mask digests from the accepted first-input check.

Prospective normalized CHECKPOINT_STAGE schema:

```text
schema: SAMRefiner-CHECKPOINT_STAGE-v1
status: ROOT_VERIFIED_COMPLETE
author_commit: 4bb7f95738c7a2e805f189210fb09e02a2197557
model_type: vit_h
official_url: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
checkpoint_ref: exactly the SPEC checkpoint_ref
verified_sha256: exactly the independently verified complete SHA256
```

This is a future interface requirement, not a claim that an existing download-stage file already has this schema or has passed. A filename, partial transfer, arbitrary matching self-calculated digest or mutable status text is not independent qualification. Trust rests on Root's prior source/provenance verification and its exact external SPEC/stage pinning.

The prospective Root lease record schema is Root-SAMRefiner-one-image-lease-v1. It must bind ROOT_GRANTED, run_id, operation, worker_ref, the complete prelock, single_attempt=true, exclusive_GPU=true, main_scientific_processes_reaped=true, and scope=one_image_resource_and_function_probe_only. This record cannot prove an external monitor is live: Root must actually implement and independently observe the lease, terminal and owned-group cleanup.

## Separate prospective resource plan

The main PLAIN/FULL experiment and its existing 4GiB qualifications remain unchanged. An external baseline trial uses a separately reviewed prospective plan, not an automatic increase of that limit.

allocator_bytes is a mandatory positive integer read from the private SPEC; the worker has no default GPU cap. Root's proposed independent plan is:

- PyTorch CUDA allocator cap 16GiB.
- CPU affinity 0/1; Torch intraop/interop and OMP/MKL/OpenBLAS settings 2.
- Process RSS limit 16GiB and sampled available-RAM floor 16GiB.
- Startup MemAvailable at least 48GiB for the proposed 16+16+16GiB conservative budget.
- Exclusive GPU and externally supervised deadline 1800 seconds, at most one attempt.

The startup threshold must be explicitly supplied and be at least allocator cap + RSS allowance + free-RAM floor. This sum is a conservative admission rule, not a measured peak-memory guarantee. The actual allocator cap may differ only through a newly reviewed Root SPEC and lease; no precision/model/N100/iteration reduction, retry or resource upgrade occurs automatically.

The worker reuses the reviewed U65 make_control component only for its original CPU/RSS/available-RAM/file checks. Its unchanged CPU LIMITS subset is supplied separately from the new GPU allocator field. It does not invoke or impersonate the full187 U65 run_owner, and it does not modify the old U57 GPU owner or its four-GiB training qualification. External Root supervision remains mandatory.

The PyTorch set_per_process_memory_fraction call limits the caching allocator. It is not a hard certificate for CUDA context, driver, third-party allocations or whole-device/system memory. Sampled RSS/available RAM are not continuous guarantees.

Static risk: Root estimated a 2.34375GiB FP32 lower bound from ViT-H block weights alone, around 1GiB for one global-attention matrix, and at least 4.34375GiB when ordinary eager softmax old/new matrices overlap; decoder N100 adds substantial buffers, with individual estimates around 400–1200MiB. These are unexecuted static estimates. They make the earlier 4GiB plan very likely to OOM and do not prove 16GiB suffices. No OOM may be treated as an accuracy comparison.

## Persistent stages and truthful completion

REPORT.json is updated together with a flushed/fsynced STAGES.jsonl event stream. Stages separately record:

1. External stage/lease record verification, complete checkpoint-byte verification and checked input joins.
2. CPU FP32 model construction return.
3. CPU checkpoint deserialization and strict weight-loading return.
4. Model-to-CUDA return and separate FP32/CUDA/eval verification.
5. image_encoder, prompt_encoder and mask_decoder Python forward pre/post counts.
6. Complete author-refiner return, subsequent CUDA synchronization, output writing and cleanup.

Hooks return None and do not inspect or replace input/output tensors. A post-hook means the Python forward returned; asynchronous CUDA work is certified only by later synchronization. Decoder counts do not measure complete refinement rounds. completed_rounds_observed remains unconfirmed; configured iterations is 5.

Model construction, weight loading, CUDA placement and inference completion flags are set only after the corresponding operations return. On failure/OOM, already reached stages and counters remain visible. The worker clears model/backend references, removes hooks, synchronizes CUDA and empties its cache when initialized, records allocator cleanup observations, and returns failure if cleanup/resource checks fail. Root must still observe actual process termination and an empty owned group.

Successful output is refined100_masks.npy plus candidate_identity.json and private file digests, pending independent Root review. These are not COCO/AP results. Bridge ID labels preserve the adapter contract but cannot independently detect an internal author mask-row permutation.

## Outstanding input-equivalence gap

The accepted first-input check proves the current PIL-decoded RGB array survives PNG and OpenCV round trip, along with the admitted first-image/native100 mapping. It does not prove equality with historical Windows JPEG decoding or the stored historical rgb400 tensor.

The worker reports historical_rgb400_vs_current_PNG_compared=false and historical_input_equivalence_closed=false. A separately registered and bounded reconstruction of rgb400 through the exact reviewed preprocessing, followed by comparison with the historical cache, remains pending. This one-image resource/function probe cannot close that gap or support a fair accuracy claim.

## Actual stdlib validation and boundaries

Three tests use fabricated JSON-like metadata only: completed checkpoint-stage gate, fixed source/precision/resource contract, and original candidate order/class/raw-score preservation including negative zero. No masks, arrays, checkpoint tensors or model are executed by the tests.

Observed incremental checks:

| Change | Actual RED | Actual GREEN |
| --- | --- | --- |
| Complete checkpoint stage gate | ef30ae / exit1, worker missing | 417803 / exit0 |
| Fixed method/source/resource contract | c433ad / exit1, validator missing | abaeaf / exit0 |
| Explicit independent resource cap/startup plan | cca9d4 / exit1, old four-GiB-only rule | 6de30d / exit0 |
| Raw candidate identity preservation | fe14bc / exit1, function missing | 62413c / exit0 |

Appending the driver initially introduced a stray plus syntax error: 31912f / exit1. It was corrected; the three stdlib tests passed b38790 / exit0. The latest verification after the final cleanup edits was 83e61d / exit0: all three tests passed, both Python files parsed as AST, and sys.modules contained neither Torch nor NumPy. This syntax failure is preserved and is not a scientific RED, model execution, integration pass or inference result.

The reviewed candidate worker SHA256 is a9b7910fcd18d53752122b8e983044b9ab4855a0484950cd4e90b88079b2c8b8 (27,920 bytes). The stdlib test source SHA256 is fa5132353f209f05ba010213f328ed32e6effcde7f49c6f55dbcd7ed2c7f4b79 (6,002 bytes). Root must qualify the final file bytes independently before binding any runtime SPEC.

Real Torch/NumPy runtime integration, dependency availability, full ViT-H allocation, the real hooks, actual author inference and cleanup are unexecuted. No old adapter suite or synthetic CUDA operator suite was rerun here.

Private SPEC/lease/stage records, credentials/host configuration, original images, masks, native predictions, refined outputs and weights must not be published. Reviewable source/tests/README are candidates for Root's normal publication process; nothing in this directory is deployed or published by this subtask.

