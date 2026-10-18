# U52 independent DGX equivalence runtime — unbound, not executed

This directory provides the Linux parent and worker for the separately prepared U52 shared-encoder gradient equivalence body. It does not change the candidate, original numerical helpers, U40 training, or any existing execution source. The engineering question is whether the candidate reproduces the specified original two-image update trajectories within the body's prelocked FP32 tolerances. It is not a method novelty, AP, convergence, or speedup result.

Only three small standard-library metadata tests and AST checks have run during preparation. No Torch/NumPy import, tensor, model, portable package read, SSH, actual spec, owner launch, source freezing, or numerical comparison has run for this draft.

## Actual invocation interface

After independent Root review, transfer and explicit binding of the real Linux file references, the future launch is:

```text
<DGX-python> linux_owner.py --spec <absolute-spec.json> --spec-sha <SHA256>
```

The owner invokes the admitted `entry.py` with the same two arguments. The worker is not an alternative unowned execution path. Both modules import only the standard library at module load; the actual parent imports the admitted psutil at CLI execution, and the actual child imports the admitted psutil/Torch/NumPy after source verification. The runtime pins Python 3.12.14, Torch 2.10.0+cu130, NumPy 1.26.4 and psutil 6.1.1 through the inherited actual runtime checks. Source admission for these libraries covers the three selected top-level module files, not all binary dependencies, shared libraries, Python, or the OS.

The spec has exactly these 13 keys:

```text
schema, operation, phase,
entry_source, body_source, owner_source,
source_refs, library_sources, context, prelock,
output_directory, owner_output_directory, sources
```

`schema` is `ROOT-U52-DGX-equivalence-spec-v1`; `operation` is `GPU_U52_DGX_SHARED_ENCODER_EQUIVALENCE`. No optional input-provenance field is accepted. Each file reference is exactly `{path, bytes, sha256}` with an absolute canonical Linux path and the actual file identity; no model or data references are invented here.

`body_source` must identify the 32,259-byte `equivalence_body.py`, SHA256 `82ae34c912696bff1cabe0d88a89c8a50757a4f3e75322378cc7c0a91854339a`. `source_refs` contains the exact fourteen body roles and fixed source hashes: the eleven original initialization/model/geometry/training helper roles plus `u50_pilot`, `u50_loader`, and `candidate`. The entry's `SOURCE_PINS` is the explicit map. `library_sources` is exactly `psutil`, `torch`, `numpy`.

`context` is exactly:

```text
portable_refs: {"54": <original U49 private package ref>, "710": <original U49 private package ref>}
scratch_directory: <new absolute private directory inside output_directory>
```

The original two U49 packages are required; this is not a claim that other U51 packages are interchangeable. `prelock` must equal the reviewed body's `equivalence_contract()` exactly. It includes the unchanged six resource limits, trajectories, indices, optimizer, tolerances and numerical protocol. The `sources` list is the path-sorted exact list of 22 distinct references: three runtime infrastructure files, fourteen numerical/source roles, three selected library files and two private packages.

Owner and worker output directories must be distinct and neither may contain the other. Neither output may contain admitted inputs or sources. Their parent directories must already exist; both output directories and the private scratch directory must be new. The owner creates its output; the worker creates its own output; the body exclusively creates scratch. Reuse/overwrite and automatic retry are rejected.

## Execution and result semantics

The entry delegates numeric imports and computation to the frozen body:

```text
run_equivalence(torch, np, c, context=..., source_refs=..., prelock=...,
                body_source_ref=..., report=body_report)
```

`c` is the new owner's `make_control(spec, psutil)` object and provides the existing `closed(ref)`, `check()`, `psutil` and `summary()` ABI. No original full-population entry is invoked. The body performs one original warmup per arm, then BASE_A, BASE_B and CANDIDATE each carry G27 step 2 into G1 step 3 from the same exact warm state. A complete comparison has seven returned updates per arm, fourteen total. Counts refer to real disposable optimizer updates; `new_training=True`, `pilot_only=True`, `main_training=False`, and formal training budget credit remains zero.

The wrapper requires the body's `COMPARISON_COMPLETED_PENDING_ROOT` status, exact baseline controls in both arms, fourteen entered and returned updates, the expected read/call sequence, unchanged caller RNG, restored module aliases, and actual GPU allocated/reserved zero at the recorded release points. It preserves detailed original body results and partial counters.

Two numerical verdicts are valid completed diagnostics:

- `WITHIN_PRELOCKED_TOLERANCES`: both candidate comparisons meet the body's fixed tolerances.
- `CANDIDATE_OUTSIDE_PRELOCKED_TOLERANCES`: at least one candidate comparison fails them, while the baseline controls and execution contract pass.

The second verdict remains a valid exit-0 diagnostic, explicitly **not** an equivalence pass. Baseline-control mismatch, incomplete updates, operational exception, source mismatch, resource violation, failed cleanup or failed invariant produces worker `FAILED` and a nonzero exit. Neither result retries, changes tolerance, adopts a model, advances U40, or produces AP. Auditing and checkpoint I/O are included in timing; timing is not a speed benchmark.

## Linux ownership and resource scope

The parent retains the reviewed U50 process lifecycle: real `Popen` child, new session/process group, CPU affinity `[0,1]`, two OMP/MKL/OpenBLAS threads, `/proc` PID/start-ticks and session/group identity, resource polling, actual `wait()` returncode/reaping and final birth/group absence. The owner is a small single-threaded monitor and moves itself off child CPUs when other allowed CPUs exist.

Limits remain: 1,800 seconds, child process-group RSS at most 16 GiB, available system RAM at least 16 GiB, and body-enforced 4 GiB Torch allocator cap before model transfer. Whole-GB10 memory usage is not fabricated from unavailable `nvidia-smi` readings: the receipt records no WDDM whole-card 8 GiB certification and no numerical zero for an unavailable memory counter. This independent UMA scope is not interchangeable with the earlier Windows resource contract.

On a limit violation or timeout, the owner signals only its verified original process group, then escalates and reaps. SIGTERM, SIGHUP and SIGINT enter controlled cleanup. Linux `PR_SET_PDEATHSIG=SIGKILL`, with a parent-PID race check, covers abrupt owner death for the direct worker; it is not a general descendant-containment guarantee. The ordinary worker does not intentionally launch another scientific worker. If process identity/group absence cannot be established, the receipt does not claim clean completion. An external caller must still observe the actual owner terminal; the owner cannot certify its own exit.

The owner does not import ML, open audit weights, or modify any Windows account/prefix. It stores an independent Linux journal, not Windows RAW/held/FIRST or a fabricated Windows process handle.

## Artifacts and remaining actual work

The worker writes `REPORT.json`, schema `U52-DGX-equivalence-worker-report-v1`, with `status` COMPLETE/FAILED, the complete `body_report`, actual counters, numerical verdict, baseline/candidate booleans, resource observations, source identities and restoration flags. The owner writes `stdout.log`, `stderr.log`, `RUN_STATE.json` and exclusive `OWNER_RECEIPT.json`, schema `U52-Linux-owned-equivalence-receipt-v1`. Its completion label is `OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW`; final acceptance still requires Root's actual terminal and receipt review. Candidate outcome is recorded separately from execution success.

The body's scratch files include model/BN/Adam/gradient CPU snapshots. They are private scientific outputs, remain after failure, and must never be included in GitHub publication. The full report may contain private paths. Only ordinary source and a separately reviewed, appropriately scoped summary are publication candidates.

Root still must review these sources, close actual Linux package/library/source references, prepare and hash a fresh exact spec and prelock, confirm resource/serialization headroom, launch once under this owner without overlap, observe the original owner terminal, and review the actual numerical result. This draft itself grants no execution or scientific qualification.

## Preparation checks

Three tiny standard-library tests first failed against the unadapted U50 copies and then passed after this implementation (3 tests, 0.003 s). They check the exact 22-reference spec and scratch scope, complete positive and negative diagnostic metadata with baseline/count/RNG rejection, and required real terminal/reap/birth/resource conditions. They neither mimic tensor arithmetic nor certify GPU equivalence. AST comparison confirmed the original source allowlist, resource predicates, Linux process identity, group cleanup, parent-death preparation and JSON journal helpers were unchanged.
