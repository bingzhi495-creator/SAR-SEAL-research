# Frozen matched M200-to-L GPU relationship replay

Status at author handoff: new independent UNBOUND SOURCE, AST/compile only. No
scientific payload was read, hashed, imported or executed while authoring. Frozen
cache, common, failed CPU qualifier and CPU diagnostic originals stay unchanged.

This is one finite relationship diagnostic, not cache qualification. It does not
load a model, teacher, JPEG or GT file, perform matching, generate Q, train a head
or compute AP. Torch packet deserialization necessarily includes unused F and
supervision buffers; their values are not accessed by this program.

## Entry and owner controls

Entry: `replay_gpu.py --spec ABSOLUTE_SPEC --owner-execute`.

The SPEC schema is
`ROOT-N13-frozen-M-to-L-relationship-GPU-replay-spec-v1`.
The report schema is
`N13-frozen-M-to-L-relationship-GPU-replay-report-v1`.

Exact primary reference keys (`{path,bytes,sha256}`):

```text
entry_source
cache_GPU_entry_source
cache_GPU_spec
cache_GPU_report
cache_GPU_RAW
cache_GPU_admission
```

Required additional fields are `phase`, `GPU_admission_file`, `output_directory`,
`normal_RAW_count=38`, `RAW_inputs` of length38,
`prior_failed_CPU_receipt`, `additional_failed_RAW_inputs` of length5,
`terminal_identity_pins` for all44 original receipts, `sources`, `recipe`,
`deadline_seconds=1800`, `allocator_limit_MiB=4096`,
`whole_GPU_limit_MiB=8192`. Original owner driver SHA remains
`ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924`.

Before any SHA or numerical import, all38 actual normal0 and all6 preserved failed1
RAWs must pass their exact original identity, admission SHA, held handle, ROOT
wait/reap and original-birth-absence checks. Source/input closure hashes follow.
All38 historical owners then retain original admission/phase-binding/bootstrap
joins. Current SPEC/admission/entry/environment are bound; the program never
reads its own unfinished terminal receipt. Owner uses CPU[0,1], two threads,
CUDA0, no GUI and no automatic retry; original whole-GPU observation remains the
owner's responsibility, not a self-certification by this program.

The cache GPU is successful normal index37. Its37-normal/four-failure historical
prefix, source/SPEC/report/admission/RAW, packet/carrier order and all741/1661
counts are preserved. Its historical additional failures equal the new first3,
not the extended current5. All741 packet and741 carrier refs from that exact
report must be present in current `sources` and admission bindings. Full actual
Torch2.0.1+cu118, NumPy1.23.5 and psutil module/binary files must be pinned, not
only distribution metadata. Imports and actual binary origins are checked before
GPU work and again at completion.

## Fixed replay recipe

`RECIPE` is the exact static constant in `replay_gpu.py`; ROOT reads it by AST
without importing the SOURCE. It locks full741 images/1661 matched query cohort,
38 normal/six failed inputs, canonical `[1,100,200,200]` FP32 contiguous NCHW,
dummy-zero unmatched slots, original-query index insertion, original two-stage
bilinear renderer, strict FP32 bytes, diagnostic-only subsetG, full-report then
failure on any canonical mismatch, and no Q/GT/model/teacher/AP/retry.

For each original report row, the program independently verifies core/sidecar
schema, original source/SPEC/phase/teacher ref, dataset index, identity, fixed
query IDs, G/raw-GT count, finite FP32 M/L shapes and archived tensor digests.
No new query choice, GT selection or matching occurs. Saved `matched_M200` must
be contiguous `[1,G,200,200]`. Saved `matched_native_L` stays the sole reference.

Construct a zero FP32 CUDA tensor `[1,100,200,200]`, standard contiguous NCHW
stride `[4000000,40000,200,1]`; index-copy the saved matched M channels into their
original unique query positions0..99. Independently render exactly:

```python
padded = interpolate(M, size=(800,800), mode='bilinear', align_corners=False)
crop = padded[0, :, :actual_h, :actual_w]
native = interpolate(crop[:,None], size=(original_h,original_w),
                     mode='bilinear', align_corners=False)[:,0]
matched = native.index_select(0, original_query_ids)
```

The decisive predicate is saved/replayed dtype and shape equality, finite FP32,
and every byte equal in matching raster order. Signed zero is distinguished.
There is no tolerance, allclose, sign-only or sampled-pixel substitute. All741
rows are reported even when bytes differ; after cleanup and full report save,
any canonical mismatch causes exit1. The smallerG GPU renderer is additionally
recorded without affecting this predicate and cannot act as a fallback.

## Report and separate ROOT qualification

Pass status:
`N13_FROZEN_M_TO_L_CANONICAL100_EXACT_BYTES_REPLAY_COMPLETE_NO_CACHE_Q`.
Failure status:
`N13_FROZEN_M_TO_L_CANONICAL100_EXACT_BYTES_REPLAY_FAILED_NO_CACHE_Q`.

Report includes `source/spec/GPU_admission/phase/recipe`, full RAW prefix/failure
and identity pins, `cache_GPU_source/cache_GPU_spec/cache_GPU_report/cache_GPU_RAW/
cache_GPU_admission`, `start`, `trace`, `rows`, `actual_imported_files`, `cleanup`,
peak allocation and deadline. Strong success fields are:

```text
images = 741
matched_queries = 1661
canonical100_all_exact_bytes = true
canonical100_mismatched_images = 0
canonical100_mismatched_queries = 0
canonical100_mismatched_pixels = 0
cleanup.allocated_after = 0
Q_written = false
cache_qualified = false
current_owned_GPU_terminal_self_certified = false
```

Each row preserves `dataset_index/image_id/identity/query_ids/packet/
validation_carrier/input_tensor_digests`, saved M shape/stride,
`canonical100_exact_bytes`, `per_query_canonical100_exact_bytes`, `canonical100`
and `subset_G`. Each comparison contains exact-byte bool, per-query mismatch
counts, byte count, shape/stride and descriptive max absolute error; the latter
is never a pass criterion. `subset_G_all_exact_bytes` and its mismatch totals
remain diagnostic. ROOT must later join the report's actual original owned
terminal0/reaped/birth-absence receipt; this program cannot produce that proof.

## Evidence limits

Unmatched slots are dummy zeros, not restored original all100 M. Original full-M
stride was not archived; standard contiguous NCHW dispatch is an explicitly
conditional reconstruction of the matched-input relationship. The archived
matched M bytes are preserved, and the actual all741 byte comparison is required
to test the condition. Static channel-independent interpolation or original
einsum facts do not certify original layout or actual equality. This diagnostic
does not independently re-run the backbone, validate GT/features/sampling, repair
or erase the original CPU failures, or establish model utility/innovation.
