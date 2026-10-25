# SAMRefiner v03清理修正：已通过限定单图功能探针

当前实际状态：2026-10-25已在共享GPU上完成单次清理修正验证，原工具/SSH/worker均0，原零清理条件保持并实际通过，仅38来源范围及一图100候选功能/资源探针合格。[详细范围](../../U57_SAMREFINER_ONE_IMAGE_FUNCTIONAL_PROBE_ACCEPTED_20261025.md)。无AP、全187图或核心创新结论；v02原失败保留，未作父产物。

## 成稿时的设计与准备记录（历史）

Status: UNEXECUTED / UNBOUND / NO GPU LEASE GRANTED. Cleanup efficacy remains unverified; this is not a successful-fix report. Frozen v02 and its failed original8312 attempt are unchanged. Its tool/SSH/worker1 and strict external closure remain failure evidence; successful author return does not upgrade that attempt to an accepted run.

This independent candidate changes only the worker's finally cleanup block. Everything through the main numerical/output try block is byte-identical to v02. Model, input identity, FP32/noAMP, ViT-H/N100/five rounds, shared scope,16GiB/48GiB/1800s and zero allocated/reserved cleanup thresholds are unchanged. No automatic retry, tolerance increase or alternate resource/precision path is introduced.

## Static mechanism, not an established diagnosis

The observed remaining9,568,256 bytes equals8,519,680 +1,048,576 bytes. Python model/backend/module references and hook handles were already released. Author return includes GPU auxiliary tensors, but the bridge returns plain nested masks and does not intentionally retain those auxiliaries.

PyTorch v2.10.0 official sources provide a plausible process-local backend-cache explanation:

- [CublasHandlePool.cpp L108](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L108): clearCublasWorkspaces clears both cuBLAS and cuBLASLt DataPtr maps.
- [L162](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L162), [L220](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L220), [L253](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp#L253): CUDA Lt defaults to1024KiB, uses CUDACachingAllocator and is cached by handle/stream; environment/unified-workspace choices affect the interpretation.
- [Module.cpp L1581](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp#L1581), [L2029](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/Module.cpp#L2029): Python torch._C._cuda_clearCublasWorkspaces calls that clearing function.

These facts do not establish the actual cause of the earlier residual or prove this correction works. The private API is version-qualified and must be callable; absence remains a failure, not a skipped cleanup.

## Only added cleanup behavior

After existing reference release/GC and CUDA synchronization, finally now persists:

1. Current CUBLAS_WORKSPACE_CONFIG/CUBLASLT_WORKSPACE_SIZE/TORCH_CUBLASLT_UNIFIED_WORKSPACE values and before allocated/reserved counts.
2. A returned call to torch._C._cuda_clearCublasWorkspaces.
3. empty_cache, CUDA synchronize, and after allocated/reserved counts.

The original allocated==reserved==0 and peak<=explicit allocator cap checks still apply. An unavailable API, clear error, nonzero remainder or cleanup error fails normally. Clearing is confined to this fresh child process's backend maps, not background user services. It occurs after model/output computation and does not alter those results.

parent.py changes only its worker byte/SHA pin and records actual parent_identity=owner.proc_identity(os.getpid()), persisted for later external birth verification. Existing timeout, signal handling, sampled-resource supervision, owned-group cleanup, output closure and pending-Root status are unchanged.

## Future binding

Existing shared-v02 schemas can be retained, but Root must use a new unique run_id, new exact worker/parent refs, fresh outputs, a new explicit one-attempt source-correction lease and fresh background observation. Old failed leases/outputs are not reused. Complete checkpoint verification and all project scientific groups reaped remain required. This candidate does not grant itself a run or modify the active main experiment.

Invocation remains the qualified interpreter with -B parent.py --spec <private-parent-SPEC> --spec-sha <full-SHA256>. Private operational records, observations, model/input/output payloads and weights remain private.

## Preparation limits

Only stdlib AST/import/difference checks are permitted here; there is no fake Torch or mirrored cleanup test. No old lifecycle suite is rerun to suggest new scientific evidence. The real private-API call and before/after allocation observations remain unexecuted and require Root's separately authorized single validation attempt.

