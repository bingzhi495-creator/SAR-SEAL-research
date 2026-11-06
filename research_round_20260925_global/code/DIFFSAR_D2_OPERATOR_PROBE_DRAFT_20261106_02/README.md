# v02: post-numerical cuBLAS workspace cleanup only

This independent candidate preserves v01 unchanged. Root confirmed v01 was not bound or scientifically executed. v02 has also not been bound or numerically executed. No operator failure, observed leak or successful cleanup is claimed.

The need is inferred from version-pinned PyTorch 2.10 sources and the author's DCNv2 GEMM path: its cuBLAS workspaces can remain allocated in persistent handle/stream maps after ordinary tensor collection. Emptying the CUDA caching allocator does not by itself remove the map's live workspace references. Thus v01's final allocated/reserved zero contract needed an explicit post-operation workspace release.

Only after all numerical checks (or on the failure cleanup path), v02 synchronizes CUDA, calls the version-checked private API `torch._C._cuda_clearCublasWorkspaces()`, records entered/completed flags, then empties the allocator cache and requires the original zero/zero plus peak-cap conditions. An unavailable API fails closed. No workspace setting is changed before forward/backward, and no tolerance, case, reference, source pin, admission limit, native math or success criterion is relaxed. The worker still cannot self-qualify execution.

Pinned primary evidence:

- [PyTorch v2.10.0 CUDA notes, lines 942-951](https://github.com/pytorch/pytorch/blob/v2.10.0/docs/source/notes/cuda.rst#L942): persistent cuBLAS workspace behavior and explicit clear API.
- [PyTorch v2.10.0 CublasHandlePool.cpp, lines 98-117](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L98): both handle-stream workspace maps and their explicit clearing.
- [PyTorch v2.10.0 Python binding, lines 1903-1906](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp#L1903): `_cuda_clearCublasWorkspaces` binding.

One additional necessary standard-library stand-in test verifies invocation/flags and rejection of a missing cleanup API. It does not import Torch, call CUDA, measure actual allocations or prove this API works on the installed binary. Actual numerical/cleanup evidence remains pending Root's one bounded run. The following inherited description still defines the unchanged math and interface.

---

# Fixed author Detectron2 CUDA operator probe candidate

Status: **SOURCE-ONLY / UNBOUND / UNEXECUTED**. This candidate author ran only one necessary standard-library source/lease admission test and AST/inert-import checks. No Torch/Torchvision/Detectron2/scientific library, CUDA, SSH, real image/GT, dataset, model or checkpoint was imported, read or executed. Root owns actual execution and qualification.

## Why this operator path

Fixed author DiffSARShipInst commit `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871` enables modulated deformable convolution in all four R101 stages, with convolution groups and deformable groups both 1. The actual `DeformBottleneckBlock.conv2` uses a 3x3 kernel, no bias, and stride 1 or 2 depending on stage. The probe therefore prioritizes the author's `detectron2.layers.deform_conv.modulated_deform_conv`, not ordinary unmodulated DCN or a complete backbone.

The selected built extension is fixed to Root's already accepted build-only artifact: **20097880 bytes**, SHA256 `8b3323891f77ce4d2e9fdc330336cc600e821c0f77b90f398714f4d6dc6398af`. Importing or running it has not been established by build success. The worker rejects a different binary before numerical imports.

Primary author source links:

- [Train configuration](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/configs/DiffSARShipInst-SSDD-train.yaml#L131)
- [Actual functional and compiled forward/backward](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/deform_conv.py#L187)
- [Actual block and mask sigmoid](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/resnet.py#L275)
- [Author zero-boundary bilinear/offset semantics](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/csrc/deformable/deform_conv_cuda_kernel.cuh#L666)

## Two fixed independent comparisons

Both cases use FP32 CUDA, batch 1, input channels 2, output channels 3, kernel 3, padding 1, dilation 1, groups 1, deformable groups 1, and bias `None`:

| Case | Input shape | Stride | Output shape |
|---|---|---:|---|
| modulated_stride1 | 1x2x4x5 | 1 | 1x3x4x5 |
| modulated_stride2 | 1x2x5x7 | 2 | 1x3x3x4 |

Features/weights and output gradients are fixed signed modular values. Offsets contain both signs, have nonzero noninteger magnitude below 0.3 and avoid integer sampling/boundary kinks. Masks lie strictly in (0,1). Input, offsets, masks and weights are separate leaves, with independent clones for the reference. Masks are treated as already sigmoid-produced values; gradients are with respect to those mask values, not sigmoid logits or the offset-prediction convolution. No RNG or case selection occurs.

For kernel point `(ky,kx)`, the author samples at `y=oy*stride-pad+ky*dilation+offset[2*k]`, `x=ox*stride-pad+kx*dilation+offset[2*k+1]`, where `k=ky*kernel_width+kx`. Offsets are interleaved **dy,dx**; the worker does not reorder them based on the upstream variable names `offset_x/offset_y`. Features outside the image contribute zero. Sampling strictly outside `(-1,H)x(-1,W)` contributes zero; partial border neighborhoods use zero for missing neighbors, not border clamping.

The independent reference constructs each of the nine sampling grids with `2*x/(W-1)-1`, `2*y/(H-1)-1`, calls public PyTorch `grid_sample(mode='bilinear', padding_mode='zeros', align_corners=True)`, then explicitly multiplies masks and convolution weights and sums channels/kernel points. It never invokes deformable convolution, the author binary, or Torchvision DCN. This is an independent composition for these shapes and parameters, not a proof for every convolution setting.

The public author functional receives ten positional arguments `(input, offset, mask, weight, None, stride, padding, dilation, 1, 1)`. Stride/padding/dilation are **integers**: this author's modulated functional does not normalize tuple arguments before forwarding them to `_C`.

Outputs and four first derivatives (input, offset, mask, weight) are compared using the same signed nonconstant upstream output gradient. CUDA is synchronized after both native and reference forward/backward paths; shapes, FP32/device and finite values are required. Fixed elementwise tolerance is `abs(actual-reference) <= atol + rtol*abs(reference)`:

| Quantity | atol | rtol |
|---|---:|---:|
| Output | 3e-5 | 3e-4 |
| Input gradient | 8e-5 | 8e-4 |
| Weight gradient | 8e-5 | 8e-4 |
| Offset gradient | 1e-4 | 1e-3 |
| Mask gradient | 1e-4 | 1e-3 |

These pre-execution FP32 bounds allow accumulation/interpolation-order differences, not semantic substitutions. Maximum absolute error and error divided by the fixed allowed error are reported independently. Failures remain failures, with no case change, CPU fallback, tolerance adjustment or automatic retry. Higher derivatives are not requested.

## Delegated normal interface checks

The actual author's ROIAlignV2 wrapper delegates to Torchvision `roi_align`. A single constant interior ROI uses the author's 7x7 output and sampling ratio 2, aligned true. Its analytic expected output is all ones and input-gradient sum is 49. The output and gradient sum use fixed atol/rtol 1e-5. This checks the normal wrapper/backend connection with an analytic invariant; it is **not** an independent ROIAlign implementation comparison or evidence that author `_C` implements ROIAlign.

The author's axis-aligned batched NMS similarly delegates to Torchvision. Three same-class boxes (one exact duplicate, one disjoint) with descending scores must keep indices `[0,2]` at threshold 0.5 on CUDA. This checks the normal interface against an analytic keep set. NMS gradients are not meaningful or requested. Rotated NMS/ROIAlign are outside scope.

## Root spec/lease interface

```text
python -B worker.py --spec <absolute-private-spec.json> --spec-sha <sha256>
```

The eight exact spec fields follow the closed CARAFE probe pattern:

```text
schema = Root-D2-fixed-operator-probe-spec-v1
operation = CUDA_D2_FIXED_FP32_OPERATOR_PROBE_V01
run_id starts with D2-fixed-FP32-v01-
source_refs = {worker: {path, bytes, sha256}}
library_sources = {torch, detectron2, deform_conv, d2_extension, roi_align,
                  nms, torchvision, tv_roi_align, tv_boxes, tv_extension}
root_lease_ref = {path, bytes, sha256}
output_directory = fresh absolute POSIX directory
prelock = exact exposed PRELOCK constant
```

The module source roles identify the actual corresponding Python module `__file__` values: Torch, Detectron2, `detectron2.layers.deform_conv`, `detectron2._C`, `detectron2.layers.roi_align`, `detectron2.layers.nms`, Torchvision, `torchvision.ops.roi_align`, and `torchvision.ops.boxes`. `tv_extension` is the actual Torchvision `_C.so` file loaded via `torch.ops.load_library`; it is verified against `torch.ops.loaded_libraries`, not incorrectly imported as a Python `torchvision._C` module. The D2 extension is a normal Python extension module with its own actual `__file__`.

All selected refs, own worker and lease are hashed before numerical imports, actual imported paths/bytes are joined, and selected sources are rechecked afterward. Only these selected sources/binaries are closed; this is not authentication of every transitive dependency, CUDA library or the whole environment.

Lease schema is `Root-D2-fixed-operator-probe-lease-v1`, status `ROOT_GRANTED`, joined to operation/run/worker/prelock/output. `single_attempt`, `main_scientific_processes_reaped` and `exclusive_project_scientific_leaf` must be true; `whole_device_exclusive` false. A supplied boot ID is checked. Root alone establishes actual admission and process/group absence.

The worker writes fresh `OUTPUT/REPORT.json`. Successful status is `TWO_FIXED_D2_MODULATED_DEFORM_CONV_CASES_PASSED_PENDING_ROOT`; `worker_exit_code=0` while `qualified=false`. Both DCN comparisons and delegated normal interface checks must complete. Partial native progress, failures and resource/cleanup errors are retained. Root must separately observe original-tool/SSH/worker zero, postverify sources and reap/check every owned birth/group; self-reported zero grants no qualification.

## Resource and claim limits

CPU0/1, all Torch/OMP/MKL/OpenBLAS threads 2, allocator at most 1 GiB, worker RSS at most 4 GiB, startup available RAM 48 GiB and runtime floor 16 GiB, external hard deadline 120 seconds, one attempt. Linux aarch64/Python3.12.14/Torch2.10.0+cu130 with CUDA build13.0 are checked. Detectron2/Torchvision actual versions and selected library identities are reported. The dedicated venv may reuse base site-packages; no fully isolated runtime is claimed.

Successful exit also requires synchronized CUDA and allocated/reserved both zero after cleanup. Worker samples do not certify whole-device memory or group RSS; Root's external supervisor owns the group/hard-timeout contract and birth cleanup. Imports or compilation alone are not operator evidence. Passing these tiny cases would still not establish full R101 channels, all shapes/groups/dtypes, a complete model forward/train step, a full detector reproduction, training speed, SAR metrics, AP or publication readiness.

The one necessary standard-library test checks inert import and rejection of substituted built-binary/run-lease identities. AST/inert checks and this metadata test do not simulate the numerical reference or prove native execution. Actual numerical validation remains pending Root's bounded run.
