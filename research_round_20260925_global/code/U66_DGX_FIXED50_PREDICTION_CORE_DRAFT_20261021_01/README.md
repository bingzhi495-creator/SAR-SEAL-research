# U66 future DGX fixed-E50 prediction core

Unbound source draft; **no real Torch/NumPy, GPU prediction, checkpoint/data
read, SSH or ML execution has occurred in this task**. Only new files here were
edited. Frozen U40/U58/U59 and the running U57 Source remain unchanged.

```python
predict_shard(torch, np, c, *, endpoint_metadata, entries, modules,
              source_refs, shard_index, dataset_range, report)
```

This is the missing numerical connection, not an owner/CLI/binder/metrics/Q
implementation. Root alone supplies genuine admitted libraries, source-loaded
modules and live `c.closed(ref)` / `c.check()` / `c.psutil` supervision. Root must
separately source-close U66 itself. A path-returning stub is not real admission.

`modules` has exactly 19 roles: the original U58 twelve
`training_state`, `training_entry`, `u36_helpers`, `u39_training_ops`,
`shared_initialization`, `full_legacy`, `full_dsconv`, `full_head`, `full_feature`,
`plain_legacy`, `plain_feature`, `instance_adapter`, plus `restore_state`,
`portable_reader`, `image_geometry`, `prediction_core`, `encoder`, `mask_api`,
`population_contract`. `source_refs` has those 19 plus `dev_reader` and
`dev_rgb_reader`; those last two are closed as code and not executed as readers.

`prediction_core` is the original ordinary U40 prediction module, **not U66**.
The directly checked ABIs are:

* U34 `shared_initialization.build_pair(torch, seven_modules, seven_refs)`
  returns `(adapters, fresh_receipt)`. Its seven constructor roles exclude
  `shared_initialization` itself.
* U58 `restore_arm(torch,np,c,adapter,arm,endpoint_metadata,fresh_receipt,
  modules=twelve_modules,source_refs=twelve_refs,report=restore_report)` is used
  for production FORMAL E50, never the SMOKE path or old U40 endpoint validator.
* U59 `prepare(torch,c,entries,image_geometry=...,source_refs=three_refs)`;
  `load(i)` returns `(item,row,originals)`. The three refs are `dev_reader`,
  `dev_rgb_reader`, `image_geometry`. Row keys are `dataset_index/image_id/qids`.
* Original U40 `_calibration_and_prediction(torch,np,c,adapter,item,row,
  originals,{encoder,mask_api,population_contract},cost,collect_support)` returns
  `(predictions,identity,support,proof)`.

Endpoint metadata is U58's exact `refs/qualified/spec/pair_commit/order` ABI.
U58 authenticates the actual endpoint documents/full checkpoint signatures,
complete trained head/BN/inactive state, and actual final segment ending37050.
There is no guessed final start36864 or Windows/U40 metadata substitution.

The CPU constructor runs once; **both arms restore while CUDA is cold before
any GPU initialization**. Only after both genuine CPU restores and reader
metadata preparation does U66 initialize CUDA and apply the old U40 allocator
cap. Arms are popped and processed serially, in frozen eval and `no_grad`.
No optimizer is constructed/restored; U58 does load and authenticate original
CPU optimizer/RNG containers as part of complete checkpoint verification.

U66 passes original `rgb400/e/M/L`, geometry, row and all100 native dictionaries
directly to the unchanged per-image function. It neither moves/reorders item
tensors nor computes a residual, resets the trained head, changes scores or
reconstructs output masks. The original helper owns one encode/image,25 groups
of4 natural query indices, exact zero/native RLE calibration including empty
masks, genuine trained `adapter.predict`, and native `L+render_delta` output.
All original score/class/query/bbox/area/order fields retain their semantics.
PLAIN alone collects original Omega support; FULL support remains unrequested.
No new renderer, sampling math, threshold or configurable option is introduced.

Original hooks, whole-state digest and U36 measurement ABI are reused to check
one backbone/feature exit per image, no old semantic head calls, unchanged
trained state/BN and no gradients. Parameter metadata is copied from U58's
already-verified `parameter_report` before inference freeze; U39's training-only
`_partition` requires trainable parameters and is not called on frozen models.
Its historical manifest flags are not current inference grad flags. Models are
never temporarily unfrozen or repartitioned. The old four ranges remain
`[0,47]`, `[47,94]`, `[94,141]`, `[141,187]`. Full187 entries are metadata-checked;
only the selected shard is actually loaded for each arm. This is not all187
prediction success. Identity predictions must agree across both arms.

The supplied report is retained on failures and returned as JSON-compatible
pending evidence on success. It contains per-arm predictions, identities,
Omega support, original proofs/cost, restore reports and resources, plus PLAIN
identity/Omega and the original cost summary. Successful status is
`PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE`; `qualified=False`, `AP=False`,
`TEST=False`, zero training credit. `DEV=True` denotes DEV prediction work;
`DEV_metrics_computed=False` and no COCO annotations/GT rasterization are used.
Reader preparation or successful restore alone cannot produce this status.

Models, hooks and current-image aliases are released on success/failure; both
per-arm and final CUDA allocator allocated/reserved must be zero. PLAIN must
complete that release before FULL transfers to CUDA. Original prediction errors
propagate, even if hook cleanup also fails, with cleanup errors separately
recorded. Unknown ABIs fail rather than selecting a fallback. U57 private
cleanup is not assumed: final cleanup reuses the explicit original U40 CUDA
sequence. Root still owes admitted hard deadline/whole-GPU/RSS/free-RAM control,
runtime/RNG/alias/source accounting and actual terminal acceptance. Sampled U36
RSS/allocator observations are not an OS peak/cap or standalone FPS evidence.

Windows provenance inside packages remains historical metadata and is not
revalidated as raw input here; historical export readers may have opened GT
containers. No claim of a historically GT-untouched end-to-end flow is made.

Finite test command (stdlib Python, `-B`, in this directory):
`C:/Users/houbingzhi/anaconda3/envs/seal_gpu/python.exe -B -m unittest test_metadata -v`.
Real RED/GREEN records:

* Initial public ordering/identity test: `8abf8e`, exit1 (module missing), then
  `9ee120`, exit0 (one test).
* Prediction-error plus hook-cleanup-error test: `ecceb4`, exit1 (cleanup masked
  original error), then `7f8524`, exit0 (two tests).
* Review added the missing between-arm hard-zero check: `187dae`, exit1 (no
  cleanup between PLAIN/FULL), then the same public test passes after adding
  the original explicit CUDA sequence to each arm as well as final cleanup.
* ABI review restored U39's trainable-only partition precondition in the stub:
  `3f9410`, exit1 on the frozen restored model. GREEN uses U58's existing JSON
  partition report; no temporary unfreeze or new parameter math is performed.
* The final four public stub tests additionally check second CPU-restore failure
  never touches GPU/reader and malformed reader ABI cannot become prediction.

All model/tensor/RLE/resource objects in tests are explicit synthetic stubs.
Their masks/proofs are invented and must never be counted as real calibration,
training, restoration or predictions. No old numerical or geometry suite is
rerun. Real DGX/FP32/zero-calibration/nonzero trained prediction remains unrun.
