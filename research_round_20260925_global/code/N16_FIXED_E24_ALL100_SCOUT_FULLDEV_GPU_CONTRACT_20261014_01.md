# N16 fixed E24 all100 FullDEV GPU entry: UNBOUND contract

This is an ordinary, unbound implementation. It has not run against scientific data, has no original terminal record, and confers no GPU/CPU qualification or AP result. ROOT supplies and authenticates the real closed refs, original owners, resource admission and SPEC. No live/private scientific file was read to build it.

Entry: `N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_01.py`, CLI `--spec <ROOT-bound-spec> --owner-execute`. `source_gate(path,spec,cpu=False)` is reusable by the CPU wrapper with `cpu=True`; the CPU caller must independently prove its own `__file__ == spec.entry_source.path`. The CPU wrapper additionally joins the completed new GPU leaf, closes its report/manifest/capture refs and independently checks final outputs.

## Fixed evaluation behavior

The native model is not loaded or forwarded. The entry replays qualified original native captured F256/e0/M200/class fields, uses the original renderer and official score/order/RLE/COCO kernels, and requires exact native reconstruction for all 187 images and 18,700 natural queries. Original teacher/native/selector provenance and the strong seed202610120701 E24 LOG_PR endpoint are preserved. LOG_PR has 8 FP32 tensors, 40,497 parameters and no buffers; new scouts have 4 tensors and 16,547 parameters each. Neither interface substitutes for the other.

The 12 official policies are native; strong fixed E24 original Omega; E24/E8 UTILITY and ERROR atmost/exact; E24 DENSE_ERROR_FLIP and MAGNET atmost. E24 is fixed main and E8 is a fixed curve, with no DEV winner. E8 dense/MagNet records are additional CPU-proof diagnostics only and do not enter AP.

For each query, Omega is original FP32 `abs(L)<=2`, P is `L>0`, and C is Omega union P union centered Chebyshev8 dilation. Empty P returns Omega. The new entry passes an exact CV2 17-rectangle/anchor(8,8)/one iteration/constant-zero facade to the frozen evalbody; it does not edit the frozen body. K0 performs no edit even when C is nonempty. Scores and ties use the frozen NumPy FP64 helper; selected supports are actual C points with exact indices, no GT input, no ranking tolerance.

Each E8/E24 UTILITY/ERROR model independently runs canonical ascending C chunks of 2048 once. Its saved actual GPU logits may be reused for atmost/exact and ERROR/dense diagnostics. There is no projection sharing across models. Original PR runs common C once as a shared action/reference; UTILITY/ERROR selected-only PR runs separately for every endpoint/mode. Selected and original Omega PR must match common C at the same actual inputs with absolute error <=1e-4, rtol0, and exact strict `combined>0` signs. Failure is numerical qualification failure without fallback or tolerance change. The full original strong Omega prediction and COCO12 must reconstruct exactly.

## ROOT SPEC fields

GPU schema: `ROOT-N16-fixed-E24-all100-scout-fullDEV-GPU-spec-v1`; CPU schema: `ROOT-N16-fixed-E24-all100-scout-fullDEV-CPU-spec-v1`. GPU has 109 normal ancestors plus 12 failed ancestors; CPU has 110 plus 12 after the new GPU actual0. All RAW-only actual-terminal/identity checks precede the first Source SHA. Legacy8 and distinct failed9/10/11/controlled12 records remain separate. Controlled12 is held exit1/reaped/birth-absent resource/deadline abort1800..1805, never normal0 or scientific success.

The existing training-CPU prefix metadata, carrier refs, opportunity GO refs, exact locked protocol, all Source/admission/pin/failed-extra fields remain required. New mandatory fields are:

- `training_GPU_source`, `training_GPU_spec`, `training_GPU_report`, `training_GPU_RAW`, `training_GPU_admission_refs`.
- `training_CPU_source`, `training_CPU_spec`, `training_CPU_qualification`, `training_CPU_RAW`, `training_CPU_admission_refs`.
- `ROOTtrain_result`: actual ROOT schema `ROOT-N16-paired-scout-training-qualified-result-v1`, status `ROOT_N16_PAIRED_SCOUT_E8_E24_FULLY_QUALIFIED_FOR_DEV`, with exact refs/checks/epochs/updates/eligibility and original GPU0/CPU0 joins. Bare booleans do not replace these bindings. Training owner slots are GPU normal-2/-1 for the GPU entry and normal-3/-2 for the CPU entry.
- `scouter_source`, `selector_eval_body_source`, `selector_cpu_proof_source`, `labels_source`; `scout_checkpoints` exact keys `8`,`24` must match both training reports and actual ROOT result.
- `strong_fixed_reference_predictions` and `strong_fixed_carrier_control_name`, whose named carrier `control_results[name].predictions` must equal the ref. ROOT supplies the exact historical control name; the entry does not guess or select one.
- `selector_mode='reuse'`, fresh `output_directory`, `deadline_seconds=1800`, `automatic_retry=False`, allocator4096MiB/wholeGPU8192MiB, CPU affinity01 and Torch threads/inter-op2. ROOT must bind actual library imports and CPU/GPU admission metadata.

No missing-field sentinel is eligible to execute. All actual sources are checked again before REPORT. The full 1800 seconds includes Source gates, firstpass, capture, export and COCO; there is no automatic retry.

## Resource budget before CUDA/scout/PR

`capture_budget` is an exact dict of positive integer fields:

`total_original_pixels`, `total_C_points_upper_bound`, `total_K_points`, `maximum_query_C_points`, `maximum_total_raw_capture_payload_bytes`, `maximum_serialized_capture_bytes`, `maximum_capture_files`, `minimum_free_output_bytes`, `required_available_RAM_bytes`, `maximum_process_RSS_bytes`, `auxiliary_output_budget_bytes`, `serialized_chunk_overhead_bytes`, `serialized_query_core_overhead_bytes`, `serialized_shared_image_overhead_bytes`.

ROOT geometry is 187 images, sumN29,404,940, maxN339,796. The CPU-only firstpass reads authenticated original selector L, constructs all100 C/K counts and image unions, checks the old Omega counts and firstpass RSS, and records C time. Actual counts must fit the prelocked bounds. No scout or PR runs before this hard budget gate.

Let U=sum image union sizes, C=sum all100 C sizes, T=sum all100 K and N=sum original pixels. The conservative raw tensor upper bound is

`1032U + 400N + 104C + 240T + 1200*sum(ceil(N_image/8)) + 187*102400` bytes.

This includes one actual F_unique/union index per image, one original natural100 L/e, all100 C indices, four actual scout outputs, common action, selected-only/old-Omega action and indices, and all 12 full masks per query as exact packbits. It does not miscount masks as 12 per image.

With nc=ceil(C_query/2048) if K>0 else0, nk=ceil(K/2048), physical chunk files are bounded by `5*sum(nc)+9*sum(nk)`, query core files by4*18700 and shared files187. Variable core refs are bounded by `sum(14nc+8nk)`, charged8192 bytes each plus65536 per core. Per-file metadata overhead is independently hard-capped; chunk allowance>=65536, shared allowance>=1MiB and core allowance>=65536+8192*max(4nc+2nk). Actual source bindings/path length are covered by those limits.

The firstpass computes max per-image total raw capture <=1GiB, largest shared image, exact component counts, serialization overhead and capture-file bound. Capture plus auxiliary JSON/manifest/prediction/trace output must total <=72GiB; free disk must be >= their prelocked bounds plus16GiB; available RAM>=16GiB and process RSS cap<=16GiB. Shared readback is budgeted with two simultaneous copies and headroom. Each actual capture is saved/read back bitwise, each file/raw/serialized/RSS cap is supervised, and all auxiliary writes/trace bytes are cumulatively capped. These are rejection gates, not runtime/speed promises.

## Physical capture and CPU adapter

Every `.pth` is a real physical ref `{path,bytes,sha256}`. Its wrapper has exact keys `schema,binding,value`, with schema `N16-actual-GPU-streamed-capture-member-v1`. Binding contains source/spec/GPU_admission/phase, training_GPU_report/training_CPU_qualification/ROOTtrain_result, both scout_checkpoints and locked_protocol.

One shared image value, schema `N16-actual-GPU-image-shared-inputs-v1`, stores actual GPU union_flat_indices INT64[U], F_unique FP32[U,256], e0 FP32[100,256], native_L FP32[100,H,W], C_indices[100]/C_counts/K_counts, identity/native packet/selector/export order and actual input markers. GPU F is sampled directly at original grid coordinates; it is never CPU rerendered. Before dropping duplicate chunk F/e/L from disk, the entry compares their actual observed FP32 bytes with this shared capture.

Query core schema `N16-actual-GPU-query-core-and-physical-chunk-refs-v2` has query_id, epoch8/24, evalmode, shared actual image ref, K, common_C_PR_chunks physical refs and controls. Scout and selected chunks carry `scout_epoch` explicitly. Saved chunk roles/mode/query/2048 offsets/model endpoint must match during decoding. Shared actual scout outputs may legitimately be referenced by several diagnostics; there are no fake member refs.

Each control records actual_selected_indices INT64, real chunk refs and actual_mask_packbits UINT8[ceil(H*W/8)], mask_shape=[H,W], mask_bits_count=H*W, mask_bitorder=`little`. Padding bits are zero and CPU decoding rejects altered padding. This lossless storage change does not change action, ranking, mask or RLE.

CPU APIs inside the same ordinary entry:

1. `validate_shared_for_CPU(torch,np,cv2,shared_value)` independently verifies native FP32 all100 C/K and union mapping once per image.
2. `decode_CPU_query_core(torch,np,core_wrapper,shared_wrapper,shared_ref,load_closed_physical,expected_binding,synthetic_only=False)` returns exactly the frozen CPU proof packet. ROOT's loader authenticates actual completed-GPU closure/hash/physical ref before returning each complete wrapper. It does not use virtual refs, CPU renderer/scout forward or approximate CPU values to define ranking. The decoder reconstructs F/e/L solely from actual GPU shared bytes, validates model/query/mode/chunk bindings, and reconstructs the exact full Boolean mask from saved packbits.
3. Call frozen `proof_query`/`proof_all100_queries` with ROOT-closed wrappers named `original_head` and `independent_PR` for the actual 8-tensor PR. Each `(F,e,L)->(u,L+u)` must be own FP32 addition; absolute1e-4/rtol0/strict signs remain mandatory. All18700 indices and masks must be exact, with native unchanged outside selected. Full original score/order/RLE/COCO12 output verification is the CPU wrapper's responsibility.

REPORT schema/status are `N16-fixed-E24-all100-scout-fullDEV-GPU-report-v1` / `N16_FIXED_E24_ALL100_SCOUT_GPU_COMPLETE_PENDING_INDEPENDENT_CPU_PROOF`. `capture_manifest` is an actual ref; manifest schema `N16-streamed-GPU-all18700-point-capture-manifest-v1` contains the same binding, rows187 with shared physical ref and query profile core/oldOmega refs, and `physical_capture_refs` for every real `.pth`. REPORT additionally binds native_packets, selector_manifest, native/strong references, START/trace and all12 control prediction/COCO outputs. Current original terminal is never self-certified.

## Checks completed and remaining limits

AST parsing passed. Ordinary CPU-only inline fixtures exercised shared input capture, 100 queries, CV2==SciPy C, two distinct K0 cases (including nonempty C), four model calls without cross-model sharing, 980 PR calls, and 400 actual-record-shaped CPU proofs across E8/E24 atmost/exact. All reconstructed selected indices/full masks passed. Wrong endpoint, differing F, signed-zero readback differences and mask padding were rejected; raw/metadata budget was checked against synthetic captured bytes. No scientific input, CUDA execution or actual scientific qualification occurred.

These fixtures cannot establish real training-to-deployment batching compatibility, near-zero PR signs, Source ABI or resource duration. ROOT must authenticate and run the real closed leaf once. Numerical failures remain failures; they cannot be cured by score/rank tolerance, alternate renderer, fallback, changing action or posthoc budget changes.

The path measures shared diagnostic runtime on cached native fields. Four actual model forwards, sorting, selected PR, common reference, transfer/import/readback/capture/RLE/COCO are recorded with overlapping-inclusive scopes. Costs are not additive and may not be subtracted to claim standalone efficiency or fresh end-to-end FPS. No new blind TEST, three-seed, convergence, external SOTA, mechanism, innovation or TGRS conclusion follows from this entry.
