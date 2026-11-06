## 2026-11-06 实际状态

两个固定CUDA案例已真实完成并由Root接收，结果见[实际记录](../../CARAFE_ACTUAL_OPERATOR_PROBE_20261106.md)。下方未执行标签为绑定前快照；冻结科学原件未改。

# Fixed CARAFE FP32 CUDA operator probe candidate

Status: **SOURCE-ONLY / UNBOUND / UNEXECUTED**. The author of this candidate ran only two standard-library metadata admission tests, AST parsing and inert module import. No Torch/MMCV/scientific library was imported; no CUDA, real data, GT, model or checkpoint was read or executed. Test success is not operator success.

This probe is deliberately limited to the installed official MMCV 2.1.0 `CARAFEFunction.apply` on the Root-selected Torch 2.10.0+cu130 / CUDA build 13.0 / Linux aarch64 / Python 3.12.14 stack. Root must independently qualify actual execution, source pins, hard resource supervision and process cleanup. It does not build or alter the environment.

## Fixed operator and reference

The fixed author DiffSARShipInst FPN commit `bdc3362372213bc8c3a1f6d5b7b74dc224d1b871` constructs `CARAFEPack(channels=256, scale_factor=2)` with default reassembly kernel 5 and group 1. Its feature/mask forward and backward extension call signatures were statically matched with MMCV 2.1.0 by the independent compatibility reviewer. The probe uses kernel 5, group 1, scale 2 but **four feature channels**, not the author's complete 256-channel FPN or its learned mask encoder.

For output `(oy, ox)`, the input neighborhood is centered at `(oy // scale, ox // scale)`. A missing feature outside the input is zero. A kernel offset `(ky, kx)` uses mask channel `group_index * kernel^2 + ky * kernel + kx` at the output position, where `group_index = channel // (channels / groups)`. The Function requires `channels % groups == 0` and does not itself softmax masks.

The independent reference is implemented using public PyTorch primitives: `unfold` with zero padding `(kernel-1)//2`, reshape to `(N,G,C/G,kernel^2,H,W)`, repeat only the feature neighborhood H/W axes by `scale`, multiply by masks already shaped on `(H*scale,W*scale)`, then sum kernel offsets. It does not call MMCV/native CARAFE, substitute nearest interpolation for the operator, or normalize boundary masks again.

Primary sources:

- [Author FPN and CARAFEPack](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L790)
- [Official MMCV 2.1.0 Function](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/carafe.py#L142)
- [Official optimized CUDA neighborhood and group indexing](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/ops/csrc/common/cuda/carafe_cuda_kernel.cuh#L125)

## Two predeclared cases and tolerances

1. `non_square_signed`: features `(1,4,3,5)`, signed modular values; positive nonuniform softmax masks `(1,25,6,10)`.
2. `boundary_dominated`: features `(1,4,2,3)` with positive top-left and negative bottom-right channel-specific values; masks `(1,25,4,6)` emphasize two extreme kernel offsets while retaining positive weights elsewhere. The large 5x5 neighborhood on this tiny non-square input exercises zero padding.

Both cases are generated deterministically without RNG. Masks are normalized **before detaching into leaf tensors**. The probe compares derivatives with respect to feature and mask leaves, not mask logits or learned CARAFEPack parameters. Independent clones are used for the reference. Both paths receive the same nonconstant signed output gradient. Each native forward and backward is explicitly CUDA-synchronized, and outputs/gradients must have the prescribed FP32 CUDA shape and finite values.

Elementwise comparison is fixed before execution as `abs(actual-reference) <= atol + rtol*abs(reference)`:

| Quantity | atol | rtol |
|---|---:|---:|
| Output | 2e-5 | 2e-4 |
| Feature gradient | 5e-5 | 5e-4 |
| Mask gradient | 5e-5 | 5e-4 |

These FP32 bounds allow small native accumulation-order differences. They are not bitwise determinism claims. Native failures or out-of-bound errors remain failures; the worker does not retry, fall back to CPU, change cases or loosen tolerances. It reports maximum absolute error and maximum error divided by the fixed allowed error, separately for all three quantities.

## Root interface

The CLI is:

```text
python -B worker.py --spec <absolute-private-spec.json> --spec-sha <sha256>
```

Metadata import is inert and exposes `SCHEMA`, `OPERATION`, `LEASE_SCHEMA`, `PRELOCK` and `validate_spec(spec)` without scientific imports. Exact spec fields are:

```text
schema = Root-CARAFE-fixed-operator-probe-spec-v1
operation = CUDA_CARAFE_FIXED_FP32_FORWARD_BACKWARD_V01
run_id starts with CARAFE-fixed-FP32-v01-
source_refs = {worker: {path, bytes, sha256}}
library_sources = {torch, mmcv, carafe, mmcv_extension}
root_lease_ref = {path, bytes, sha256}
output_directory = fresh absolute POSIX directory
prelock = exact PRELOCK constant
```

The four library refs identify actual `torch.__file__`, `mmcv.__file__`, `mmcv.ops.carafe.__file__` and `mmcv._ext.__file__`. The worker hashes all selected files before scientific imports, compares actual imported paths/bytes against those refs, verifies installed versions and rechecks selected files afterward. This closes only those selected sources/binary, not all imported transitive dependencies or every CUDA/runtime binary.

The Root lease schema is `Root-CARAFE-fixed-operator-probe-lease-v1`, status `ROOT_GRANTED`, joined to the same operation/run/worker/prelock/output directory. It requires `single_attempt`, `main_scientific_processes_reaped` and `exclusive_project_scientific_leaf` true, and `whole_device_exclusive` false. A supplied `boot_id` is checked. Only external Root can establish those admission facts; the worker does not self-authenticate them. The outer supervisor may include private prior birth records, which remain outside this source's authority.

The output is `output_directory/REPORT.json`, initially STARTED and then a terminal report. Success status is `TWO_FIXED_CARAFE_FP32_CUDA_CASES_PASSED_PENDING_ROOT`, `worker_exit_code=0`, while `qualified` remains false. Partial native call progress is retained on failures. Worker, SSH and original-tool outcomes, source postchecks and external group reaping are separate Root requirements.

## Resources and scope

CPU affinity 0/1, Torch/OMP/MKL/OpenBLAS threads 2; PyTorch CUDA allocator at most 1 GiB; observed worker peak RSS at most 4 GiB; free RAM at least 16 GiB; external hard wall deadline 120 seconds; one attempt. Root's parent must enforce the **whole owned process-group** RSS/deadline and verify final births/groups absent. Source alone only samples worker RSS, available RAM and elapsed time between calls; it cannot interrupt a stalled native kernel or certify group totals.

Successful completion also requires CUDA synchronization, allocator allocated/reserved both zero after collection/cache release, and observed peak allocated memory within the cap. Those allocator figures do not certify whole-device memory or exclusivity. Other GPU services may exist. No FPS, training speed or full model memory claim follows from this probe.

Passing two tiny FP32 examples would only support this selected CARAFE forward and first-backward operator path. It would not establish Detectron2 `_C`, deformable convolution, RoIAlign/NMS, all CARAFE shapes/dtypes/groups/scales, second derivatives, the learned CARAFEPack mask encoder, complete DiffSARShipInst training, SAR accuracy, AP or publication readiness.

The two necessary tests are only admission checks: inert import and rejection of altered resource/case contracts; rejection of cross-run/cross-worker Root leases. They were developed red-to-green without scientific execution. The real numerical evidence must come from Root's separately bounded actual CUDA run.
