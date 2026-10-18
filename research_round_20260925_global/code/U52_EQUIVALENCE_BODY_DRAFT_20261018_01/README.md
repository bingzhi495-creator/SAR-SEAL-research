# U52 standalone equivalence body — unbound, not numerically executed

This is an engineering diagnostic for the separate shared-encoder-gradient candidate. It does not modify U40, change the original DSConv source, establish a new method, award formal training updates, or evaluate AP. Only ordinary source files were read. No actual input package, checkpoint, scientific chain, model, numerical library or remote machine was opened/run while preparing it.

`equivalence_body.run_equivalence(torch, np, c, *, context, source_refs, prelock, body_source_ref, report)` returns the same JSON-compatible report dictionary. `report` starts empty. The caller provides real imported Torch/NumPy under a newly admitted Linux owner; `c` supplies `closed(strictref)`, `check()` and `psutil`. No entry, binder, launcher or owner is implemented here. The old U50 entry `main`, pilot `run_pilot`, and U39 whole probe are never called.

`context` has exactly two fields:

- `portable_refs`: string keys `"54"` and `"710"`, each an admitted DGX-local strict file reference. These must be the original accepted/transferred U49 two-image packages, not U51 batches or invented tensors. Existing U50 `_load_item` performs the real future weights-only CPU load and all original shape/dtype/identity/digest/geometry checks. Windows provenance is retained without reopening those references.
- `scratch_directory`: a new canonical absolute POSIX directory. It must be separate from all inputs and sources. It stores private CPU audit snapshots, not U40 checkpoints or published weights. Failed/partial files are retained and never retried or overwritten.

`source_refs` contains exactly the 14 pinned roles in `PINS`: the original U50 eleven numeric dependencies, `u50_loader`, `u50_pilot`, and `candidate`. The candidate SHA is `80e6b777ccd1b41cbdc5bd4ef34f4e4d3d429b827f039c830b78bc08a847eed9`. Only a future call loads those files. It reuses the original loader's alias/order handling, restores all prior aliases, and directly calls the published original `build_pair`, `_load_item`, `_PortableReader`, `_partition`, `_update`, head-buffer checks and resource samples. Original full DSConv `b4b612…` remains unchanged; U46 is not substituted. This comparison therefore does not certify the mathematical correctness of that original DSConv implementation.

## Fixed numerical contract

The caller must supply exactly `equivalence_contract()`. Environment is Linux/aarch64, Python 3.12.14, Torch 2.10.0+cu130 and NumPy 1.26.4; CPU0/1, two threads, FP32, deterministic cuDNN, no TF32/AMP/compile/checkpoint recomputation. The actual cold U34 pair is constructed once with seed `202610173101`, its 104 common states/68 parameters/36 BN buffers and shared adapter verified by the original function. Models are operated one arm at a time, PLAIN then FULL. Original RGB400/query1/BCE+Dice/G/AdamW/offload rules remain unchanged.

For each arm:

1. Original U39 `_update` performs one real G1/index54 warmup at Adam step1. It must produce a nonzero final head, so the next test is not solely the initially zero encoder-gradient case. A private CPU snapshot records the complete model, including BN and inactive states; complete Adam state/groups; parameter-name mapping; and Python/NumPy/Torch CPU/CUDA RNG.
2. `BASE_A`, `BASE_B`, and `CANDIDATE` each restore that same warm model/BN/Adam/RNG, with every restored tensor and parameter-group mapping checked exactly. Each then performs index710/G27 at step2 followed by index54/G1 at step3. Step3 carries its own trajectory's model, BN and Adam state from step2; it is not reset to baseline between comparisons.
3. BASE_A provides reference snapshots for steps2/3. BASE_B must be bitexact for all gradients, model parameters/buffers, Adam tensors/groups, RNG and scalar losses. If this control fails, candidate execution stops and the other arm is not started; no retry or tolerance relaxation occurs.
4. Candidate runs the actual separate `update_shared_encoder`. A local AdamW subclass observes all gradients immediately before invoking the unchanged parent AdamW step. It adds no backward hooks, changes no gradient, and performs no extra optimizer step. This is necessary because both original and candidate functions clear gradients in their `finally` blocks.

The upper budget is 7 actual disposable optimizer updates per arm, 14 total. Each image has the original forward hooks: one backbone and CONV0_4 call, no legacy full forward, and zero old semantic-head calls. Unused-head BN states remain unchanged. Effective gradients must be present/finite and inactive gradients None through the original helper. Every parameter, including inactive parameters, and every registered buffer is compared; original BN behavior is not rewritten.

Prelocked candidate tolerances (`|candidate-reference| <= atol + rtol*|reference|`) are:

| Field | rtol | atol |
|---|---:|---:|
| Gradients | 1e-4 | 1e-6 |
| Model parameters and floating buffers | 1e-5 | 1e-7 |
| Adam exp_avg | 1e-4 | 1e-7 |
| Adam exp_avg_sq | 3e-4 | 1e-10 |
| Scalar loss/BCE/Dice | 1e-6 | 1e-7 |

Integer buffers, actual Adam step, names/shapes/dtypes, parameter groups, inactive None and RNG are exact. These tolerances allow FP32 accumulation-order differences over 27 query contributions while retaining tighter absolute checks for parameters and squared moments. They are diagnostic thresholds, not a guarantee that either implementation will pass; Adam's near-zero-gradient sensitivity can still produce a real failure. No threshold depends on measured results. Each tensor reports finite status, element count, violations, bitexactness, maximum absolute/relative difference, error/reference/current L2 and relative L2. Relative L2 is null for a nonzero error against zero norm, and zero when both are zero. Pointwise relative error uses a declared 1e-30 denominator floor for display only.

## Lifecycle, partial output and scope

One arm's GPU model/Adam is active at a time. All model, query, optimizer and failed-traceback aliases are released before allocated/reserved-zero checks between trajectories and arms. CPU snapshots are loaded one comparison step at a time; both comparison steps' full tensor copies are not retained in RAM. The same arm object is reused with strict warm-state restoration. At most six complete private files are retained: warm plus two BASE_A comparison snapshots for each arm. Their actual sizes are recorded through strict refs; no exact disk/RSS estimate is claimed. The future caller must admit private writable scratch and enough disk. A write attempt can leave a partial file; `private_audit_snapshot_write_attempted` distinguishes this from a completed saved warm state.

Allocator4 GiB is applied before model/input CUDA allocation. RSS≤16 GiB/free RAM≥16 GiB/CPU0,1/two threads and hard1800 seconds must be continuously enforced by the new actual owner; `c.check()` and original samples also run throughout. Whole-card GB10 memory N/A remains `None`, never zero or proof of the old Windows 8 GiB global contract. The body restores caller RNG/aliases and reports real cleanup, but does not claim the parent was reaped. Owner terminal, process identity, resource journal and Root acceptance are separate outstanding work.

Successful execution returns `status=COMPARISON_COMPLETED_PENDING_ROOT` with `verdict=WITHIN_PRELOCKED_TOLERANCES` or `CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES`; the latter is a completed negative diagnostic, not a passing comparison. A nondeterministic baseline produces `BASELINE_CONTROL_STOPPED`/`BASELINE_CONTROL_MISMATCH`. Operational, finite-gradient, source, resource or cleanup errors produce `FAILED` with retained partial counters/errors. Returned optimizer steps are counted by the original update immediately, even if later auditing fails. `new_training=True` means these real disposable updates occurred; `main_training=False`, formal budget credit0, qualifiedFalse and AP/DEV/TESTFalse remain fixed. There is no automatic retry, U40 state adoption, whole-population claim or convergence claim.

Snapshot copies, comparisons and disk I/O perturb execution timing. Resource/timing observations are therefore diagnostic costs, not a fair speed benchmark. Passing two images on this runtime would support only this prelocked local equivalence check, not full741 resource stability, all epochs, cross-runtime bitexactness, or improved accuracy.

## Preparation evidence and remaining execution work

Three tiny stdlib metadata tests: missing-module RED `d3bd3c`, then GREEN `429fcf` (3 tests, 0.001s). They check the immutable warm/control/carried-step/tolerance contract, two-private-input context boundaries, and failure-aware verdict selection. They contain no fake tensor, Torch/NumPy import or real input access. Final source AST/pins are checked separately; no numerical GREEN is claimed.

Static review identified that real `Module.state_dict()` returns `OrderedDict`. The CPU-copy dispatcher accepts dict subclasses and preserves iteration order while returning plain dictionaries. One additional pure stdlib regression (`9b4e22`, 1 test, 0.000s) checks nested OrderedDict dispatch and key order using only a Tensor **type sentinel**, with no tensor instance/arithmetic or restore claim. Thus four small metadata tests have passed; the earlier three were not unnecessarily rerun after this narrow check.

Before any actual run, Root still needs a new exact input/source spec and entry/owner mapping for this 14-role body, the two authenticated DGX package refs, exclusive scratch, and resource/terminal acceptance. The old U50 19-ref spec and six-step success gate cannot be used unchanged. The body is ordinary unbound source only and has not been transferred, frozen, launched or qualified by this task.
