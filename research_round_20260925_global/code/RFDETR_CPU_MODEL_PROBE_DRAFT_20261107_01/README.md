# RF-DETR SegNano CPU probe — ordinary draft

This small library prepares one real CPU model check. **No real checkpoint or
Torch model has been executed while preparing this draft.** Execution authority,
source binding, resource limits, process isolation, leases and closure belong to
the caller. This directory provides none of those controls.

## API

```python
from probe import run_probe
result = run_probe(checkpoint_path, seed=5201)
```

`checkpoint_path` must be a local path to an externally qualified, original
official COCO SegNano checkpoint. Importing `probe` imports only the standard
library. Calling `run_probe` lazily imports Torch and the caller-bound RF-DETR
1.12.0 source. The caller must install compatible project core dependencies and
SciPy; the library does not install anything or download weights.

The function:

- safely loads with `weights_only=True` on CPU; its sole optional retry permits
  only `argparse.Namespace` and `types.SimpleNamespace` in a scoped safe-globals
  context, still using `weights_only=True`;
- requires the original checkpoint's `model` dictionary; does not normalize
  wrappers, rename/remove keys, interpolate positions or resize classification;
- builds the original 90-config-class, 312px, 3-channel SegNano through the low
  level builder and `force_no_pretrain=True`;
- uses `load_state_dict(strict=True)`, then verifies every key's CPU device,
  shape, dtype, finiteness and `torch.equal` value against the original state;
- fixes the CPU generator seed and Torch intra-op threads to **2**, restoring
  the caller's previous CPU RNG state and thread count in `finally`;
- performs one eval/inference-mode native forward on a constant synthetic image,
  normalized using the official RGB mean/std; checks dense tensors of shape
  `[1,100,91]`, `[1,100,4]`, `[1,100,78,78]` and finite float32 CPU outputs.

It does not train, compute loss, postprocess masks, change heads, read real images,
save weights, optimize/export/compile the model, create a high-level inference
snapshot, authorize itself or retry a failed scientific probe. The caller should
use a dedicated process: CPU RNG and thread settings are process-wide, so this
function is not intended for concurrent calls in a shared training process.

## Returned evidence

The JSON-safe dictionary **always has `qualified=False`**. Success is
`status='ok'`, with `strict_original_state_load=True`, `strict_state_equal=True`,
`forward_executed=True`, `variant='RFDETRSegNanoConfig'`, `resolution=312`, seed,
CPU thread count, state-tensor counts, output shape/dtype/device/finiteness and
phase/total times. Counts concern all state tensors, not only trainable parameters.

Ordinary failures return `status='failed'` and `error={stage,type,message}`;
checkpoint-path text is redacted from that message. `cleanup_errors`, when
present, also force failure. A caller must treat failure as failure. Process
interruptions and exits propagate after cleanup. Even `status='ok'` does not close
the external qualification or demonstrate any SAR segmentation accuracy.

Raw strict incompatibility is an allowed research outcome. Do not switch to a
permissive loader or rewrite the checkpoint to make this probe pass.

## Synthetic control tests

```text
python -B -m unittest -v test_probe
```

Six stdlib tests cover strict missing-key refusal, bounded safe retry with no
pickle fallback, dtype/nonfinite/equality refusal, native output-contract refusal,
JSON-safe unqualified success and invalid-path refusal. Their tensor stubs contain
only shapes and tokens; they neither simulate model accuracy nor allocate tensor
images. Runtime imports are replaced at the external-module boundary. No Torch,
RF-DETR, checkpoint bytes or real data are imported/read by these tests.

Preparation used red→green checks for safe rejection, strict missing keys, the
full control path and invalid path handling. Final result: **6 tests passed**.
This is control-code evidence only; real strict loading and real forward were
unexecuted at preparation.

Actual update: the first raw-strict CPU attempt has now failed and its resources
have been closed; no forward or SAR AP was produced. The frozen original library
and test source are unchanged. See the repository actual-result record.
