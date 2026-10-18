# U50 independent DGX paired pilot body — SOURCE ONLY / NOT EXECUTED

<!-- U50_ORIGINAL_DGX_ACTUAL_ACCEPTED_20261018_01 -->

**后续真实执行状态：** 原模型、入口和Linux监督器已在DGX一次实际任务中完成两臂各3次真实更新，原任务退出0并经Root接收；2图/28不同匹配实例，6次可丢弃更新，约45秒。无AP，主训练仍2816/37050；不是原Windows检查点跨环境续训，也不证明全量、加速或效用。见[实际核验及下一步](../../U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。下文SOURCE ONLY是开发形成时状态；冻结执行原件未修改。


Only two tiny standard-library order/recipe tests and AST were run here. No
Torch/NumPy import, tensor/package/model execution, remote connection, data or
weight access, training, resource measurement or AP was performed.

```text
run_pilot(torch, np, c, *, portable_refs, modules, source_refs,
          prelock, pilot_source_ref, report) -> partial/completed report
pilot_contract() -> exact proposed new-environment prelock
```

`portable_refs` maps the integer original ordinals 54 and 710 to two admitted
absolute **DGX-local** file strictrefs. `c.closed(ref)` must verify their actual
bytes/size/SHA, and `c.check()` must belong to the actual owner. `c.psutil` supplies
real CPU/process observations. This helper reads only those two data files using
`torch.load(map_location='cpu', weights_only=True)`. There is no allowlist,
unrestricted-pickle fallback, original JPEG/annotation/checkpoint read, or new
export/transfer claim. Root must independently authenticate actual U49 export,
CPU roundtrip and transport; U49's internal pending qualified=False is not a
self-issued execution qualification.

The genuine U49 schema is `U49-matched-CPU-portable-item-v1`, with metadata's
nine keys dataset_index/image_id/G/query_ids/ann_ids/identity/rgb_receipt/geometry/
provenance. Exactly five CPU FP32 tensors are required: RGB400/e/M/native_L/targets.
The tensor key is `L` and its receipt explicitly names N16 native_L. The body
checks raw-byte SHA/shape/dtype/logical bytes, finite values, independent compact
storage and original matched GT/query order. U49's byte digest excludes headers;
shape/dtype are checked separately. M/e/L are **already matched and ordered**;
they are never indexed again by natural query id. Targets remain [G,H,W], not
lists or a union. F is absent because it is not an input of U39._update; this is
not complete compatibility with the old F-validating reader.

`modules` and `source_refs` contain exactly eleven reviewed roles:

```text
shared_initialization
full_legacy full_dsconv full_head full_feature
plain_legacy plain_feature instance_adapter
image_geometry u36_helpers u39_training_ops
```

Each actual module file is byte/SHA-closed against the original published pin.
`pilot_source_ref` similarly admits this exact new helper. The actual loader must
bind the full model's `dsconv` and `head` imports to those admitted objects;
U34 checks object identity. Original provenance refs inside packages remain
Windows lineage metadata and are not opened or silently rewritten on DGX.

The numeric path is deliberately narrow: original
[U34 build_pair](../U34_FULL_PLAIN_INIT_DRAFT_20261017_01/shared_initialization.py)
and original
[U39 _run_arm / _update](../U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_DRAFT_20261017_01/resource_probe.py).
The old probe entry, 741-image reader gate and old runtime assertions are never
called. A tiny reader maps only 54/710, preserves their full identities and
rebuilds the original geometry object from the saved original/content sizes.
It does not invent the other739 records or provide an F placeholder.

Within this new environment, U34 recreates fresh seed202610173101 initialization,
copies common104 states (68 parameter tensors +36 BN buffers) and equal adapter
states. Old U40 weights, optimizer and RNG are never loaded. Same seed does not
prove old/new environment initial tensors are equal. The old b4b612 DSConv is
retained; U46 is not silently substituted.

Both arms use original FP32/B1/400/query1, encoder-only CPU-saved activations,
AdamW3e-4/wd1e-4/betas(.9,.999)/eps1e-8/foreachFalse/fusedFalse, and full-G equal
1/G BCE+softDice with smooth1. U39's exact update/loss is called, not reimplemented.
Sequence is [54,710,54], G=[1,27,1], three real updates per arm, 29 GT visits per
arm and six disposable updates/58 visits total. The two distinct images contain
28 distinct matched instances. Initial zero residual, original gradient/BN/
optimizer continuity checks and old semantic-head bypass remain U39's checks.

The proposed environment is Python3.12.14/Torch2.10.0+cu130/NumPy1.26.4/Linux
aarch64. Caller must set two Torch threads/CPU01 and explicit TF32False,
cuDNN benchmarkFalse/deterministicTrue before first CUDA. No AMP/compile/VJP or
recomputation is introduced. Cold CPU U34 construction precedes a real 4GiB
allocator cap, followed by one GPU arm at a time. Each arm uses a fresh pilot CUDA
stream seeded inside fork_rng; original U40 RNG is not involved. Caller CPU RNG
and post-initialization CUDA RNG preservation are observed. Final allocated and
reserved memory must both be zero. Offload/cleanup interface availability is
checked; actual Torch2.10 compatibility and gradients remain unexecuted.

CPU01/two threads, RSS<=16GiB, availableRAM>=16GiB and <=1800s remain proposed
owner limits. U39 retains its per-image allocated peak<=4GiB assertion. Each
stage's original record is retained; the report's allocator peak takes their
maximum rather than only the last small image. GB10 memory is shared UMA: a
whole-GPU unsupported reading stays None. The old Windows/WDDM whole8GiB gate
is explicitly **not** certified, and cumulative saved-tensor transfer events
are not unique live activation bytes or measured speedup.

Missing outer pieces are intentional: actual admitted U49 export/transport,
DGX-local safe module loader and original import aliases, entry setting the
runtime/backend/affinity, continuous resource/whole-UMA supervision, output and
terminal ownership/Root acceptance. This file implements none of those controls.
Caller must preserve partial step counts and reject any status other than
COMPLETED_PENDING_ROOT_REVIEW. No automatic retry or checkpoint saving exists.
Even completed three-step execution establishes no AP, convergence, full741
stability, acceleration, structural mechanism or publishable innovation.
