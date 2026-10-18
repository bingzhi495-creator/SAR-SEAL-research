# U44 external streaming window — unbound, not executed

This is one reusable window of 1–4 consecutive new images, not a teacher loader,
dataset reader, scheduler, evaluator, qualification or launch entry. No scientific
data, weight, actual Source/receipt chain or model has been read or executed while
preparing it. Only standard-library metadata tests and AST checks were run.

## Call contract

```python
run_window(torch, np, Image, c, models, image_rows, *,
    modules, source_refs, native_modules, native_source_refs,
    preprocessing, teacher_provenance, restoration_receipt, report)
```

`models` is exactly `teacher`, `PLAIN`, `FULL`: three distinct already-loaded,
FP32, frozen, eval CPU models. The caller must have built and strictly restored
the two adapters while CUDA was cold, using the fixed E50/37050 paired endpoint
and full seed **202610173101**. The window neither constructs nor loads models,
resets the trained head, restores Adam, nor opens checkpoints. Subsequent windows
reuse these same objects on CPU, with initialized CUDA allowed only at zero
allocated/reserved bytes. `c` supplies the existing `closed`, `check`, `psutil`
ABI and continuous resource enforcement. `report` must be an empty caller-owned
dict, retained by the caller on an exception.

Each `image_rows` element has exactly
`{dataset_index, image_id, image_ref, original_shape}`. Indices must be consecutive
in the caller's original population order; shape is original `[H,W]`. IDs and
strict image refs must be unique within the window. The U41 call receives only
the last three fields; annotations, split metadata and GT are not accepted.
An outer population manifest and admission are still required. Nothing here
selects a split or claims all HRSID/SSDD images are admitted.

`restoration_receipt` maps both arms to the separate U44 CPU restore output
(`U44-external-fixed50-CPU-arm-restore-pending-v1`, `RESTORED_CPU_ONLY`). A receipt
with `qualified=False` is a provenance record, not permission or numerical
qualification. Before GPU use, this window computes the actual current CPU model
digest, matches it to each recorded restored state, checks the actual module and
parameter names, eval/frozen/grad-None state, and both arms' common E50 refs.
It never converts receipt status to a population qualification.

`teacher_provenance` and `preprocessing` are passed unchanged to U41. The former
contains the original loading proof, checkpoint/loading refs and config sources;
the latter must explicitly close the original normalization, channel conversion
and padding configuration. The original `pref.state_proof` hashes CPU contiguous
values (not device), so the same loading proof is compared before transfer, on
GPU and after CPU return. A newly computed proof is never substituted as the
teacher's loading provenance. Teacher construction/loading remains an outer task.

## Explicit numerical dependencies

`modules` is exactly:

- `native_capture`: published U41 `capture_native_image`;
- `image_bridge`: U44 `make_image_item`;
- `prediction_core`: published U40 fixed50 single-image function;
- `image_geometry`, `u36_helpers`, `training_state`, `population_contract`;
- `mask_api`, `encoder`.

`source_refs` has those nine keys plus `PIL.Image`. All are actual module-path
and strict-ref byte checked through `c.closed`; eight historical code/library
identities are pinned by `PINS`. `encoder` is the caller-admitted original module
providing `segmentation(np,masks,binary)`, as in U40's dynamic encoder-source
contract; its identity must be supplied by the outer admission, not guessed from
a filename. The original full zero-render dictionary comparison remains required.

`native_modules` and `native_source_refs` retain U41's exact existing interface:
eight objects (`Compose`, `PackDetInputs`, `pseudo_collate`, `observer`,
`renderer`, `exporter`, `pref`, `masks`) and its ten source refs (`observer`,
`renderer`, `exporter`, `pref`, `masks`, `Compose`, `pseudo_collate`, `formatting`,
`data_preprocessor`, `base_preprocessor`). U41 verifies its own actual dependencies;
the original `pref` is additionally pinned here before using its state/RNG proof.
These selected dependencies do not imply a complete Python/OS/DLL closure.

## Actual dispatch and release boundaries

1. Move only the teacher to GPU. For each image call U41's genuine image-only
   capture, then U44's genuine CPU bridge. Preserve native 100-query export
   dictionaries once. The bridge keeps only CPU e/M/native L/RGB400/geometry and
   metadata; F/C and capture GPU tensors are released before the next image.
2. Return the same teacher to CPU, compare its original proof and require
   allocator **allocated=reserved=0** after the original U40 cleanup.
3. Move only PLAIN to GPU; under `torch.no_grad()` call the original
   `_calibration_and_prediction(..., collect_support=False)` for each carrier.
   It encodes once and predicts 25 groups of four queries, indexing natural
   e/M/L by the original export qids. It renders explicit zero delta and verifies
   the complete native RLE/S0/class/query dictionaries without resetting the
   trained head. Actual prediction is original native L plus rendered residual.
4. Return PLAIN to CPU and hard-zero cleanup; repeat the same process for FULL.
   Both original model/BN digests must survive unchanged. Real forward hooks
   require one backbone and CONV0_4 call per image and zero calls to all four
   old semantic heads. The teacher also has an actual head-forward count.
5. Drop all window dense carriers and RNG snapshots; require final hard-zero
   cleanup. Caller-owned CPU models remain available for the next window.

The wrapper does not call the old `_arm` (which destroys its model) or the old
187-image shard entry. Capture, bridge, item and active-model aliases owned here
are cleared; exceptions retain textual traceback metadata only. External GPU
aliases cannot be cleared for the caller: they cause the hard-zero check to fail,
not a false successful release. A failed stage prevents the next model and has
no automatic retry. The caller must preserve the partial `report`; there is no
successful return for an incomplete window.

## Output and resources

Success returns `{dataset_range, predictions, images, queries, report}`.
`predictions` is exactly `{native: rows, PLAIN: rows, FULL: rows}`, each with
100 rows per image in original export order. Thus the external merge can consume
`{dataset_range, predictions}` without another native artifact. No R result is
invented. All returned payloads are final RLE dictionaries or metadata, not
feature tensors. The report schema is `U44-external-stream-window-pending-v1`;
`status=COMPLETED` means this window completed, while `qualified`,
`numerical_runtime_qualified` and `population_qualification` remain false.

The original FP32 runtime flags, CPU affinity 0/1, two threads, 4 GiB allocator
cap and sampled 16 GiB RSS gate are checked. Per-stage measurements include GPU
transfer, genuine inference, return to CPU, state verification and cleanup.
The outer owner must enforce the unchanged whole-GPU 8 GiB, available RAM
16 GiB and whole-leaf 1800-second hard limits continuously; a window's elapsed
check is not a replacement for that supervisor. No AMP, VJP, training, teacher
recapture qualification, or changed padding/math is introduced.

Reported time includes teacher capture and stage transfers/checks, excludes
initial loading/CPU restoration and outer setup, and is not end-to-end FPS.
`retained_CPU_tensor_bytes` counts only the four specified window-carrier tensor
types, not total RSS or model memory. Resource fit, numerical equivalence,
cross-window stability and real population completion remain **unexecuted**.

## Preparation validation

Two standard-library tests were developed RED→GREEN: exact image-only window
projection with GT/extra-field rejection; and restored-current-state/E50/shared
provenance joins with mismatched digest, cursor and qualification rejection.
They do not import Torch, NumPy, PIL or COCO, and do not simulate tensor/model
behavior. AST validation checks syntax only. Root review, outer source/input
admission, the actual teacher loader and real resource/numerical execution are
still needed before this draft can be used.
