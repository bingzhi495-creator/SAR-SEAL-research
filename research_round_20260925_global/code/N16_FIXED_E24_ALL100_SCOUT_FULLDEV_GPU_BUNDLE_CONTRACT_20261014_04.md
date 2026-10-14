# N16 FullDEV GPU04 — per-image bundle storage contract

Ordinary UNBOUND implementation, not scientifically run or qualified. The current actual worker and its Source/payloads were not read or changed. Existing failed13 and normal109 remain unchanged; no failure14 is assumed. ROOT binds any later actual failure separately after original terminal verification. The original methods, forward calls, FP32 inputs, canonical2048 batches, NumPy FP64 rankings, selected-only PR, absolute1e-4/rtol0/strict-sign checks, original scores/order/RLE and 12 policies follow GPU03 unchanged.

The implementation changes storage only: one actual `torch.save`, one exact readback, and one file reference/hash per completed image. Ordinary AST comparison confirms unchanged numerical/evaluation/native/RAW13 functions; the only core-schema substitution is storage metadata. No speed or deadline success is established by synthetic checks. Shared diagnostic costs remain inclusive and nonadditive; this is not fresh native-model FPS.

## Real physical file and internal IDs

Each image has one actual `.pth` bundle. Only the outer manifest stores its true `{path,bytes,sha256}` physical ref. There are 187 physical bundle refs after a complete run. Inside the bundle, references are strict unique strings, such as `i000/SHARED_ACTUAL_INPUTS.pth`, `i000/q000/commonC_c000000.pth`, or `i000/q000/e24_atmost_QUERY_CORE.pth`. These strings identify registered dictionary members; they are not filenames or virtual physical refs. There is no self-referential physical SHA.

Bundle exact keys:

`schema,binding,dataset_index,image_id,members,member_registry,member_count,raw_tensor_payload_bytes`.

Schema is `N16-actual-GPU-image-capture-bundle-v1`. Binding is the unchanged03 source/spec/GPU_admission/phase/training_GPU_report/training_CPU_qualification/ROOTtrain_result/scout_checkpoints/locked_protocol dict. `members` maps ID strings to original captured values, with no per-member physical wrapper. `member_registry` has exactly the same ID keys and describes each value; actual inputs/output tensors are retained unchanged. ID names are derived from actual image/query/endpoint/mode/role/chunk offset, capped at256 UTF8 bytes and unique.

`bundle_descriptor(torch,value,validate_tensors=True)` constructs the exact descriptor keys:

`kind,query_id,epoch,evalmode,control,stage,chunk_first,points,raw_tensor_payload_bytes`.

- `shared_inputs`: all scope fields through points are None. The value uses unchanged `N16-actual-GPU-image-shared-inputs-v1`, one actual F_unique/union mapping, e100, L100 and C100/K100. It is captured from actual GPU inputs with no CPU rerender.
- `query_core`: query_id, epoch8/24 and evalmode are populated; control/stage/chunk_first/points are None. Core schema is `N16-actual-GPU-query-core-and-bundle-member-IDs-v3`. Its shared reference and common/scout/selected chunk lists contain member ID strings.
- `scout_chunk`: stage=`scout_C_logits`, populated actual model epoch/arm/query, evalmode=`shared-atmost-exact`, integer chunk_first and points=len(flat_indices).
- `PR_chunk`: stage=`full_C_PR`,`original_Omega_PR` or `selected_PR`. Common/oldOmega epoch is None; selected epoch is8/24. All control/mode/query/chunk fields retain03 values.

The 100 queries, four model forwards and all recorded outputs are preserved. Same-model atmost/exact/dense references may point to one saved actual-logits member; models do not share a forward or projection. Actual whole masks remain uint8 little-bitorder packbits with shape/N/zero-padding validation. Before any F/e/L duplicates are dropped, the existing GPU callback checks their observed bytes exactly against the shared input capture.

## Pure CPU API

ROOT authenticates the completed-GPU report, Source/admission/owner/manifest and true whole-bundle physical refs first, then loads each real bundle once. The following functions perform no file I/O or Source operation:

- `validate_CPU_image_bundle(torch,np,cv2,bundle,expected_binding,synthetic_only=False)` checks the full raw/finite bound, registry and member IDs, independent original FP32 all100 C/K/union, complete400 cores, every actual canonical chunk and role/endpoint/query/mode/scope, oldOmega members, and no missing/extra/orphan members. Return values are informational; callers need not depend on them.
- `lookup_CPU_bundle_member(torch,bundle,member_id,expected_binding,member_use=None)` returns the original value after strict ID/binding/registry/role checks. `member_use` is either None or a mutable set of strings, updated by `.add`. This also supports the CPU author's strong-oldOmega reconstruction.
- `decode_CPU_query_core_bundle(torch,np,bundle,core_id,expected_binding,member_use=None,synthetic_only=False)` returns exactly the existing numeric-proof nine-field packet: `schema,query_id,evalmode,native_L,e0,K,C_flat_indices,common_C_PR_chunks,controls`. The schema remains `N16-all100-scout-selection-same-GPU-input-query-capture-v1`; e0 is FP32[256]. F/e/L are reconstructed only from saved actual GPU shared bytes; inputs for identical supports are reused within that query without concatenating entire-C feature tensors. Scout ranking uses the original saved GPU logits, not CPU scout forward or approximate CPU PR values. Padding and endpoint/role mismatches reject.

The CPU wrapper must independently visit the manifest's complete query profile and oldOmega roster, update its own mutable used set through these lookups/decodes, and require `used == members.keys() == member_registry.keys()`. Repeated member references are allowed; missing or unused members are not. The unchanged two original40497/8-tensor CPU PR formulas, strict indices/full masks/native outside support, original strong reference and all12 officialCOCO outputs remain mandatory. These APIs cannot certify actual provenance by themselves.

## Manifest/report ABI

Manifest schema: `N16-GPU-all18700-image-bundle-capture-manifest-v2`. It retains binding, images187/queries18700 and each original identity/native packet/selector/order row. Each row adds `bundle` (the unique real physical ref), `member_locators` (all registered ID strings) and `bundle_capture_cost`, while `shared_actual_image_inputs` and `query_profiles` contain internal member IDs. Top-level `physical_capture_refs`/REPORT `capture_refs` are only the187 real bundle refs; `capture_files` is187 and internal locators are explicitly marked strings.

REPORT schema/status:

`N16-fixed-E24-all100-scout-fullDEV-GPU-bundle-report-v2`

`N16_FIXED_E24_ALL100_SCOUT_GPU_BUNDLE_COMPLETE_PENDING_INDEPENDENT_CPU_PROOF`.

GPU/CPU SPEC schemas remain the03 schemas. Scientific source/admission and owned-original-terminal distinctions are unchanged. CPU must bind this new report/manifest/storage schema and source, with no substitution of an earlier physical-chunk report.

## Resource and checks

The existing firstpass and actual all18700 C/K/union counts determine raw/metadata bounds. Raw tensor formula, per-image <=1GiB, max packet serialization2GiB, RSS16GiB, whole GPU8GiB/allocator4GiB, full1800-second deadline, capture71.5GiB plus auxiliary0.5GiB and free-disk16GiB margin remain unchanged. Header/registry/internal-reference overhead is conservatively charged through the existing metadata allowances; the admitted number of physical capture files is now187. During member registration, cumulative raw and live RSS are checked; before and after full-bundle readback simultaneous RSS is checked. Actual serialized bundle/file/cumulative bounds and all auxiliary JSON/trace caps remain supervised. No threshold is loosened and no numeric output is silently dropped to fit memory.

Saved test: `test_N16_IMAGE_BUNDLE_STORAGE_SYNTHETIC_ONLY_20261014_04.py`. Four tests passed using one tiny ordinary synthetic100-query image, 400 cores, K0 empty/nonempty C, actual spy counts of one save/readback/hash, reconstructed original nine-field CPU packets and complete used-member coverage. Wrong registry role/endpoint, mask padding, physical-ref-shaped internal locator, missing member and binding were rejected. No scientific input or GPU/model execution occurred. Actual duration and compatibility remain for ROOT's later admitted run and independent CPU qualification.
