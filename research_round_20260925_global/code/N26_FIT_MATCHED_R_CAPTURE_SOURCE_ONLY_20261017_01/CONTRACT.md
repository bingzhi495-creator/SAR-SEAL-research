# N26 complete frozen-R FIT quality-input capture

Status: **UNBOUND_NOT_EXECUTED**. No model, tensor, image, GT raster or CUDA execution was used to construct or test this draft. The unit fixtures exercise metadata expressions and scope/chunk rules only. Actual intake and independent CPU verification remain required.

## Caller and files

`capture_body.run(torch, np, spec, c, pre) -> {"report_ref": strictref}`. Importing this module imports only the standard library. Runtime imports are cv2, scipy.ndimage and pycocotools.mask. `capture_setup.py` supplies the original CPU-only frozen model/dataset/head assembly; it is a separate component, authored independently. The only runtime source files in this directory are `capture_body.py` and `capture_setup.py`; `test_capture_body.py` is a synthetic test, never a science entry.

Control must expose `check()`, `closed(ref) -> path`, `ref(path) -> strictref`, `write_json(path, value) -> strictref`, `execution_spec_ref` and **`execution_admission_ref`**. There is no `admission_ref` alias. The caller admits and pins all sources before entry; this body does not replay an inherited gate or certify its own terminal status.

`pre` is the decoded `spec.prelock`, with images=741, GT=1661, dataset_shards=[[0,371],[371,741]], models="frozen_R_E24", no_new_training=true. Required spec keys are `context`, `prelock`, `setup_source`, `quality_source`, `dataset_indices`, `allocator_limit_MiB` (=4096), `phase`, `output_directory`. Exactly one complete prelocked half is permitted, in original dataset order. The output directory must not exist. Parent-directory creation and execution admission belong to the caller.

The decoded context schema is `ROOT-N26-quality-input-context-v1`. It supplies the static-map roles, including `N16_GPU_spec`, `N16_GPU_report`, `original_fit_cache_report`, `existing_packet_refs`, `existing_native_L_refs`, `identities`, `image_ids`, `dataset_indices`, `teacher_checkpoint` and `fixed_PR_checkpoint`, plus the source/config/data refs consumed by setup. All refs must be in the caller's Source closure.

## Capture contract

Each image uses the original image/parser/GT conversion, matching loss seed and one frozen teacher loss observation. All ten matcher/target stages, raw/processed GT correspondence and final natural-query assignment must equal the old N16 evidence. Every original real GT has exactly one matched query; no quality threshold or unmatched zero target is introduced.

Fresh all100 e0/class/native_L and sampled F-center calibration are compared by exact dtype/shape/encoded tensor bytes. Any mismatch stops; there is no tolerance fallback for these carriers. Historical N07 sampled native_L alone retains its old N16-approved rtol/atol. Actual N07 packet provenance is source/spec/phase/teacher_checkpoint: the writer does not put GPU_admission in that packet. N16 packets and L carriers retain source/spec/GPU_admission/phase checks.

For every matched query, all Omega = abs(native_L)<=2 coordinates are ascending flat indices. The original PR factory receives rendered F256, original e0, native_L and zero49 patches, with arm='PR', chunks of 2048 and the actual final tail. Its actual FP32 native_L+residual strict >0 determines R inside Omega; outside Omega and K0 retain native P. This is complete Omega inference, not sampled PR. GT enters only the matched IoU target and audit, never the quality feature. Low-quality and empty predictions remain samples.

The quality input is CPU FP32 original e0[256] followed by the predicted R-mask-only crop/resizing feature[784]. The target is CPU FP64 full-original-grid IoU against the original matched nonempty GT. Stored records retain original image/GT/query identities, crop/empty/area, masks/e0/GT digests, Omega/chunk counts and target. AUDIT additionally retains private compressed R-mask RLE, with an exact encode/decode binary-byte check. No full F or L is saved.

## Outputs and limits

`FIT_QUALITY_INPUTS.pth`: schema `N26-FIT-matched-frozen-R-query-mask-quality-inputs-v1`, `features` CPU FP32[n,1040], `targets_FP64` CPU FP64[n], ordered `records` and current source/spec/ad/phase/context/prelock/setup/quality/teacher/PR bindings. `AUDIT.json` includes the private RLE records and per-image evidence. `CAPTURE_REPORT.json` schema `N26-FIT-frozen-R-complete-mask-quality-capture-report-v1`, status `N26_FIT_SCOPE_CAPTURE_COMPLETE_PENDING_ROOT_AND_INDEPENDENT_CPU`, includes strict `inputs`/`audit` refs and capture checks.

The PTH payload is tensors plus Python built-in containers/scalars/None only; NumPy scalar targets are explicitly converted to Python float before tensor construction and metadata storage. Readback uses `torch.load(..., map_location='cpu', weights_only=True)` supported by the pinned Torch 2.0.1 API. This is a source-level compatibility statement, not a successful real readback claim; actual execution must pass that readback.

Teacher/head states must remain exact and frozen with grad=None. Original CPU/CUDA RNG state is restored. No CUDA tensor escapes an image call; run moves both models to CPU, releases aliases/workspaces and requires allocated/reserved=0, peak<=4096MiB. Wrapper still owns final cleanup/resource enforcement. `qualified`, `ROOT_actual_GPU0_verified`, `independent_CPU_verified`, training and AP flags remain false. These private inputs/RLE/audit are not publication artifacts. No training, deployment metric or AP result is claimed.
