# Independent one-image SAMRefiner supervisor — frozen unexecuted candidate

## Current archive status (2026-10-25)

**V01 PARENT NOT EXECUTED.** Only the standard-library preparation tests described below were previously run. No real Linux GPU supervision, model execution, or qualified result is established for this parent. The private frozen source is unchanged, and the two published Python files are byte-identical copies.

This is an archival implementation and contract description, not a runnable execution grant. No SPEC or lease instance, original image, mask, third-party source, checkpoint, or actual user-process/host identity is included. Runtime references below describe interfaces rather than published private operational material. This archiving task did not run the tests, Torch, SSH, or the model.

Status: UNBOUND / LINUX GPU SUPERVISION NOT EXECUTED. The original 27,920-byte model worker is unchanged. This parent is a single-attempt supervisor, not another model wrapper.

This version requires a truthful literal exclusive_GPU=true admission. The archived parent has not been executed as a Linux GPU attempt; this publication includes no runtime grant or background-process identity.

## Private invocation

```text
<qualified-python> -B parent.py --spec <absolute-private-parent-SPEC.json> --spec-sha <full-SPEC-SHA256>
```

Root must first verify complete checkpoint bytes/provenance, close and reap all main scientific processes, establish the literal GPU exclusivity required here, then create a new private worker SPEC, normalized CHECKPOINT_STAGE and Root lease. No runnable grant or private input is included.

The parent SPEC schema is ROOT-SAMRefiner-one-image-supervisor-spec-v1, with operation GPU_SAMREFINER_ORIGINAL_ORDINAL0_ONE_IMAGE_TRIAL. It contains run_id, parent_source, owner_source, worker_source, worker_spec_ref, root_lease_ref, checkpoint_stage_ref, timeout_ref, library_sources.psutil, worker_python, output_directory and sources. Every file reference is exact path/bytes/full SHA256. sources is their canonical sorted deduplicated union, normally eight refs. Parent and worker outputs must be separate fresh directories.

worker_python must be an explicitly selected absolute interpreter path resolving to the same base executable as this parent; it may retain its virtualenv symlink path. The child itself verifies its selected scientific library imports.

Pinned dependencies:

- Model worker: 27920 bytes, SHA256 a9b7910fcd18d53752122b8e983044b9ab4855a0484950cd4e90b88079b2c8b8.
- Reviewed U57 race-fixed helper: 37956 bytes, SHA256 6c6e2867bc7a36624f09e6483eadaaf6cd2f8340001e839000b7d85c5ac6659e.
- The child's U65 CPU/RSS control: 33056 bytes, SHA256 763c097e77c1ec5faa4c69adde5512feced9e52ad8b3e942c5337027434ef373.

The parent imports only the reviewed model worker's stdlib metadata validators, never main or Torch. It consumes the same external checkpoint-stage certificate and Root lease as the worker. It does not open the checkpoint or scientific input arrays.

## Lease binding and actual monitoring

In addition to the original worker lease fields, this version requires parent_source_ref, parent_output_directory, worker_output_directory, boot_id and prior_scientific_groups. The latter is a nonempty list of Root-reaped original group-leader identities with pid/start_ticks/process_group/session; PID, PGID and SID must match. Parent verifies same boot and absence of those prior births/groups. This consumes Root's reap evidence; it cannot wait on prior processes that belonged to another parent.

The worker plan must explicitly be allocator16GiB/startup48GiB, with the original CPU0/1/thread2/RSS16GiB/freeRAM16GiB/1800s contract. Old PLAIN/FULL four-GiB training and its qualification are unchanged.

The actual launch is one Popen(start_new_session=True), using the original _prepare_child(expected_parent_pid), wrapping admitted GNU timeout --signal=KILL within the remaining1800-second parent budget. The captured identity belongs to the timeout leader, with PID=PGID=SID; it is not replaced by the model worker PID. There is no retry or budget/precision/model adjustment.

Existing functions are reused exactly: proc_identity(pid), birth_absent(original,current), _group_members(psutil_module,original), _kill_owned_group(psutil_module,child,original,receipt), _prepare_child(expected_parent_pid). The original run_owner and training-specific terminal_success are not invoked.

Parent samples whole-owned-group RSS, available RAM, affinity and elapsed time, persisting CONTROL.json and bounded observations. Samples and peaks are observations, not continuous RAM guarantees or whole-device allocator certification. The child applies its explicit PyTorch allocator limit. The parent never grants itself GPU exclusivity.

SIGTERM/SIGHUP/SIGINT become controlled exceptions that enter owned-group cleanup; previous handlers are restored. Signals during cleanup are recorded without interrupting cleanup and block success. PDEATHSIG covers only the direct timeout child, not all descendants. SIGKILL cannot run Python finally. An external Root observer must therefore still independently verify the parent's actual terminal and owned-group absence after interruption or connection loss.

Cleanup uses actual wait/reap, same-boot verification, original birth absence and a resolved empty group scan. A leader exiting with surviving descendants is a violation even if cleanup later removes them. Unknown identity, unresolved scans, cleanup errors, nonzero worker returns or resource violations cannot become success. Bounded cleanup observation may extend after the scientific deadline; it grants no additional model work.

Successful pending status also requires the exact model worker's completed report, report boot/PGID/birth join, author return and CUDA synchronization, zero allocated/reserved cleanup, and closed output bytes. Parent output is CONTROL.json plus stdout/stderr; transient CONTROL.next.json supports atomic metadata replacement. qualified remains false and the caller must independently observe this parent exit.

## Actual finite tests

Only standard-library lifecycle tests have run locally:

- Real subprocesses returning0 and7 were actually waited and retained their return codes: RED87c5fa/exit1 (module absent), GREENa14ef7/exit0.
- A real sleeping subprocess reached a short deadline and returned control for cleanup; the test then killed/reaped its own child: REDd28a85/exit1, GREEN75695b/exit0.
- After supervisor implementation: 3e137e/exit0; after signal/cleanup changes: 9a4a20/exit0, both tests passed and AST parsed with no Torch/NumPy import.

These local tests did not execute the Linux prctl, birth/PGID/group-cleanup paths, psutil resource supervision, GNU timeout integration, GPU, model, checkpoint or real data. The test's direct child cleanup is not evidence that Linux production group cleanup has run. No fake Torch or fabricated scientific result was used.

Root must separately review/bind this frozen candidate before any real attempt. Private SPECs, leases, source-binding metadata, operational details, checkpoint material, images/masks and model outputs are not publication payloads.

