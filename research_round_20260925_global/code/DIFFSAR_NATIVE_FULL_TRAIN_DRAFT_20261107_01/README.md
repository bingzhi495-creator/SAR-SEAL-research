# Native DiffSAR complete-training worker candidate

2026-11-07. **Unexecuted, not bound and not an AP result.** This candidate follows the actual native32 health stage; its existing single-case failures remain failures. The health32 updates are not reused here. Root owns any scientific execution and qualification.

The scientific worker is self-contained `worker.py`; deploy only that file, the unchanged reviewed `config_adapter.py`, and pinned author/runtime/input files. `runtime_fragment.py.txt`, `main_fragment.py.txt` and the private builder are development aids, not extra runtime dependencies. Top-level imports are standard library only. The standard-library tests verify segment bounds, rejection of incomplete or mismatched continuation state, and delegate-and-skip sample sequence behavior. They do not prove CUDA resume equivalence.

## Proposed fixed full protocol

New formal training begins from native public-code construction initialization at update0, seed40244023; no pretrained model is requested. Paper initialization is still uncertified. Native R101/SCJE, 500 proposals, six heads, original24 losses and mapper, original per-parameter AdamW builder, wd1e-4, baseLR3.5e-5, batch1 and original optimizer-step global clipping1.0 remain. The explicit FIT/50-epoch adaptation changes the original full schedule18560 to37050, milestones29640/35198, warmup20/factor1.0. It does not truncate the full schedule at a segment boundary. Data workers0 and an explicitly seeded native TrainingSampler with shuffletrue are execution adaptations. Do not call this exact author-paper reproduction or initialization-matched to the COCO-initialized project teacher.

Full canonical741 logical image records /1661 instance annotations are kept, even though there are730 unique JPEG byte objects. Native COCO parsing is joined back by image_id and annotation_id; class1 maps to native class0 through an explicit metadata name. Mapped empty instances are not replaced with a more favorable sample. No DEV/TEST/evaluation hooks, AP, best-checkpoint selection or initializer repair occurs.

`FORMAL` spans have0<=start<stop<=37050 and at most256 actual optimizer updates. The last span can be shorter. `RESUME_VALIDATION` permits only0->8 and4->8. A continuation must use a fixed trusted own-checkpoint reference; weights-only and mismatched protocol, purpose or cursor are rejected. Do not mix validation checkpoints into formal training or describe a reconstructed optimizer as continuity.

## Exact continuation design

Each checkpoint stores model state (original dtypes, including float64 diffusion buffers), every named parameter gradient or explicitNone, optimizer, scheduler, Python/NumPy/TorchCPU/all-visible-CUDA RNG states, native sampler seed plus consumed-index cursor, and MapDataset private RNG/fallback candidates. Model/optimizer/gradient tensors are copied to CPU one at a time; no full GPU clone is introduced. This adds host staging and roughly2–3GB per checkpoint, to be measured in actual execution. Checkpoints are never overwritten or deleted by this worker.

The sample order is produced by the original TrainingSampler generator and its original private Torch generator. A sampler proxy burns only already-consumed indices, without fetching or remapping old images; it does not replace the shuffle algorithm. Source/runtime pins and fixed single-process rank/worldsize make this reconstruction meaningful. Actual emitted indices and logical image identities are checked for every batch.

The original native loader wraps MapDataset in ToIterableDataset and DataLoader, followed by AspectRatioGroupedDataset. With worker0/batch1, the latter clears its buckets before yielding. The worker eagerly creates the inner DataLoader iterator **without calling next**, constructs the original AspectRatioGroupedDataset around it, then restores full numerical RNG and MapDataset state before any mapping. This removes the extra resume-time DataLoader base_seed draw from the mathematical trajectory. It does not discard a prefetched augmented sample. Both aspect buckets and the emitted sample cursor must be empty/exact at each checkpoint boundary.

At resume start, the worker captures actual loaded model, gradients, optimizer, scheduler, RNG and loader state and hashes typed original bytes. Those digests must equal the actual checkpoint state digests before training continues. Checkpoint IO restores the captured mathematical RNG after writing, so A's midpoint checkpoint does not silently consume later training randomness. Native forward -> zero_grad -> backward -> optimizer.step -> scheduler.step order is preserved; gradients are not cleared merely to make a segment boundary easier to resume.

## Required actual continuity validation

Root must prelock and execute two original tasks before crediting formal continuity:

1. C: modeRESUME_VALIDATION, start0/stop8. Save update4 full state and continue in the same process to8; save8.
2. S: modeRESUME_VALIDATION, start4/stop8, resume_checkpoint_ref=C4. Save8.

This costs12 actual optimizer updates, not eight. Neither branch is formal training. An independent CPU comparator must inspect the actual C4, C8 and S8 checkpoints (sequential C4 then C8/S8 to bound memory), actual reports and source joins. It must check exact loaded start state; exact subsequent sample/augmented-image/target hashes, counters, static scheduler/optimizer metadata and end RNG/loader state. Final floating tensor/loss differences use only Root's **predeclared** tolerance; report observed maxima and exactness separately, not a universal FP32 error bound. Any violation remains a failed validation; do not relax tolerance after seeing data. This candidate itself does not declare final CUDA comparison success.

`component_digests(torch,state)` hashes model/gradients/optimizer/scheduler/rng/loader. `tree_digest` uses type, original dtype, shape, exact bytes, dictionary key ordering and OrderedDict metadata. It is callable by an independent comparator after pinned import, without changing the training worker.

## Interface

Operation: `CUDA_DIFFSAR_SSDD_NATIVE_FULL_TRAIN_SEGMENT_FP32_V01`.

Spec schema: `Root-DiffSAR-SSDD-native-full-train-spec-v1`. Exact fields are schema, operation, run_id, source_refs, library_sources, config_ref, input_manifest_ref, annotations_ref, input_root, compatibility_receipt_ref, root_lease_ref, output_directory, prelock, segment and resume_checkpoint_ref. source_refs contains worker/config_adapter. Segment is `{start,stop,mode}`. Fresh0 uses resume_checkpoint_ref=None; a positive start requires the explicit fixed reference. run_id starts `DiffSAR-SSDD-native-full-v01-`.

Lease schema: `Root-DiffSAR-SSDD-native-full-train-lease-v1`, statusROOT_GRANTED, with identical run_id/operation/output_directory/input_root/prelock/segment/resume_checkpoint_ref and worker/config_adapter/config/input-manifest/annotations reference joins. It must explicitly grant a single owned scientific leaf after prior groups are reaped. It does not claim whole-device exclusivity.

Root must pin all `LIBRARY_NAMES` entries:36 entries including native common, LR scheduler, Torch DataLoader/fetch and augmentation implementation in addition to health-stage dependencies. Selected pins are not all transitive runtime dependencies or a fully isolated environment.

Runtime CLI is `worker.py --spec ABSOLUTE_SPEC --spec-sha SHA256`. Parent must set CPU affinity0/1, OMP/MKL/OPENBLAS threads2, CUDA_VISIBLE_DEVICES=0 and **PYTHONHASHSEED=40244023 before process startup**. CPU hash seeding cannot be fixed by setting it later inside the Python process. Resource contract remains allocator8GiB, groupRSS16GiB, start-available48GiB, floor16GiB, deadline1800s; Root supervisor verifies group resources/birth/cleanup. The worker never SSHs, loops on failure or starts another task.

## Outputs and admission

REPORT schema: `DiffSAR-SSDD-native-full-train-segment-report-v1`. Raw output includes no AP. Completion status is `NATIVE_FULL_SEGMENT_COMPLETE_PENDING_ROOT`, stage `NATIVE_FULL_SEGMENT_COMPLETE`, all_requested_updates_completed=True, optimizer_steps_completed=stop, new_optimizer_updates=stop-start, samples_consumed_after_segment=stop. Worker qualified remainsFalse until external Root qualification. All required original24 loss terms are finite per step; entire saved model/gradient/optimizer/scheduler tensor state must be finite at a checkpoint. Zero gradients are legal; no nonzero mask-gradient coverage gate is introduced.

Each report step has completed global_update (1..8 for C), dataset_index, image_id, mapped_image_shape, mapped_instances, augmentation_and_target_sha256, losses, total_loss, used_learning_rates and post_scheduler_learning_rates. `optimizer_steps_completed` increases only after the original native step and CUDA synchronization return. Sampler order/repeated exposure follows the native shuffle, not handpicked favorable inputs.

Checkpoints are `CKPT_UPDATE000004.pt`, `CKPT_UPDATE000008.pt` etc in the fresh worker output directory. REPORT.checkpoints entries have updates/ref/component_digests/finite_tensor_state/gradients_saved/original_dtype_buffers_preserved/sampler_consumed/aspect_buckets_empty. S report additionally has resume_start_exact=True and resume_start_component_digests. Actual source/input references are rechecked after numerical work. Final CUDA allocated/reserved must be integer0/0, allocator peak<=8GiB and cleanup path records cuBLAS_workspaces_cleared=True. Original tool/SSH/worker exits and independent group closure remain necessary; a metadata checker0 cannot upgrade a failed numerical original.

No checkpoint, image, fullGT, private execution spec/lease or locator belongs in the public repository. Publish only ended safe aggregates and prebind ordinary source copies. Formal full AP, cost accounting including the teacher, capacity controls, full training repeats, cross-dataset evidence and TGRS claims remain incomplete.
