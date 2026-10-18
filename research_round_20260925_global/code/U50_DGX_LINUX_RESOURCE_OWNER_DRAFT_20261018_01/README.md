# U50 independent Linux resource owner — unbound, not executed

<!-- U50_ORIGINAL_DGX_ACTUAL_ACCEPTED_20261018_01 -->

**后续真实执行状态：** 原模型、入口和Linux监督器已在DGX一次实际任务中完成两臂各3次真实更新，原任务退出0并经Root接收；2图/28不同匹配实例，6次可丢弃更新，约45秒。无AP，主训练仍2816/37050；不是原Windows检查点跨环境续训，也不证明全量、加速或效用。见[实际核验及下一步](../../U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。下文SOURCE ONLY是开发形成时状态；冻结执行原件未修改。


This directory provides the concrete owner for the separate DGX paired pilot.
It is not U40 continuation, a Windows RAW/FIRST adapter, or a scientific
qualification. No remote command, model, Torch/NumPy/psutil import, process
launch, actual source/data access, or numerical test was performed during
development. Three tiny standard-library tests cover actual temporary-file
allowlisting, inclusive resource boundaries, and Linux identity/terminal
metadata. Linux process control, signals, prctl and resource observations have
**not** been executed by those tests.

## Entry and child API

Root supplies an actual canonical DGX spec and its SHA, then explicitly invokes:

```text
/home/hou/venvs/sar-gb10/bin/python linux_owner.py --spec ABS_SPEC --spec-sha SHA256
```

`validate_spec(spec)` accepts the exact thirteen fields agreed with the remote
entry: schema, operation, phase, entry_source, pilot_source, owner_source,
source_refs, library_sources, portable_refs, prelock, output_directory,
owner_output_directory, sources. Schema is `ROOT-U50-DGX-paired-pilot-spec-v1`;
operation is `GPU_U50_DGX_PAIRED_DISPOSABLE_PILOT`. There is no optional metadata
field. The eleven numeric sources, three selected library files
(`psutil/torch/numpy`), two U49 packages and three infrastructure sources form
the exact sorted finite source list, with only identical-reference deduplication.
Linux path comparisons are case-sensitive; actual references must use canonical
absolute local paths. Windows provenance inside the portable packages is not
opened by this owner.

The selected psutil source is verified before import, with actual runtime
version **6.1.1** required. The actual module path, source bytes, Python version
and interpreter are recorded. This is a footprint of selected source/module
files, not complete Python, native binary, shared-library or OS closure.

`make_control(spec, psutil_module)` returns the exact U50/U39 interface:

- `c.psutil`: the real injected module.
- `c.closed(strictref)`: exact finite allowlist membership, canonical path,
  streaming size/SHA verification with file identity/mtime/ctime stability;
  resource checks are made during hashing. No discovery or arbitrary fallback.
- `c.check()`: actual process RSS, available RAM, affinity and elapsed time;
  violations persist and raise. Missing/nonfinite observations never become 0.
- `c.summary()`: `U50-Linux-child-control-observations-v1`, elapsed_seconds,
  samples, peak_RSS_bytes, minimum_available_RAM_bytes, violations.

The child entry creates its fresh `output_directory`. The owner creates a
distinct, nonnested fresh `owner_output_directory`; both parents must already
exist. Existing outputs are rejected, so a completed, failed or interrupted
attempt cannot be silently restarted. There is no retry loop.

## Actual ownership and limits

The stdlib parent imports real psutil only on actual CLI invocation, remains
free of scientific imports, and requires Linux/aarch64. It remains single
threaded and, when another allowed CPU exists, pins itself to one CPU outside
0/1. Its child uses a new POSIX session/process group and real CPU affinity0/1,
with OMP/MKL/OPENBLAS thread environment2 and CUDA_VISIBLE_DEVICES=0. Entry owns
the two Torch thread settings and backend policy. Before exec, the child sets
`prctl(PR_SET_PDEATHSIG,SIGKILL)` and checks its parent PID, then applies affinity.
This kills the **direct worker** if its owner dies, including owner SIGKILL.
The setting is not automatically inherited by future descendants; ordinary
timeout and signal cleanup additionally checks the entire owned group.

The parent observes PID, `/proc/PID/stat` startticks, process group/session,
boot_id, actual return code and subprocess wait/reap. It samples the owned
group every0.1s and persists scalar/identity observations about every5s. Group
RSS is the sum of member RSS; shared pages may therefore be counted more than
once if descendants exist. This conservative aggregate is explicitly separate
from the child's own RSS. CPU0/1, RSS<=16GiB, available RAM>=16GiB, and1800s are
enforced by the owner. The parent budget starts before source preflight, so
preflight time is included; the child independently measures its own control
lifetime. The original pilot body enforces the4GiB CUDA allocator cap before
model/input transfer and retains original per-image peak checks.

On a resource violation, timeout, observer exception, or SIGTERM/SIGHUP/SIGINT,
only the original owned process group is signalled: SIGTERM with2s grace,
then SIGKILL with5s observation, followed by a bounded1s direct-child wait.
PID reuse or uncertain group identity is reported and never used to signal an
unrelated process. A remaining unkillable process is recorded as alive/unreaped;
it cannot produce success. A normally exited leader with remaining group members
also fails and triggers owned-group cleanup. Post-termination evidence writing
is separate from the1800s child deadline; this is not a promise that arbitrary
filesystem stalls have a hard total parent deadline.

The owner catches SSH disconnect SIGHUP where delivered. If the owner is
unconditionally killed or the machine fails, it cannot write a terminal receipt;
Root must inspect actual terminal/process identity evidence. The parent does
not claim its own exit before returning. The direct-child kernel death signal
reduces the risk of a detached worker surviving that case; it is not a claim of
complete crash-proof descendant supervision.

GB10 whole-card used/total may be N/A. This owner records whole_GPU as **None**
and `whole_GPU_WDDM8GiB_contract_certified=False`, never0 or a passed old Windows
gate. It does not query or stop any user service. Actual available-RAM/RSS and
Torch allocator evidence belong to this independent UMA experiment.

## Outputs and Root boundary

`RUN_STATE.json` is an atomic current observation. `OWNER_RECEIPT.json` is written
exclusively at terminal return, alongside stdout.log/stderr.log. The receipt
contains real child identity, signals, wait/reaped result, whether the original
birth is absent, any remaining owned group, sampled resources, selected runtime,
and byte references for logs and the worker REPORT.json. It does not deserialize
tensors or replay gradients.

The worker outer report is `U50-DGX-paired-pilot-worker-report-v1` with
status `COMPLETE`; the preserved body_report must have status
`COMPLETED_PENDING_ROOT_REVIEW`. Success additionally joins phase/operation/spec
and requires both reports to record six entered and six returned optimizer
updates, `new_training=True`. The original U39 update sets that true when real
disposable training begins. Body `main_training=False`, `pilot_only=True` and
formal_training_budget_credit0 distinguish those updates from formal U40 work.
Exit0 alone is insufficient. Original birth absent, direct-child reaped, empty
resolved group and no observed violations are required for
`OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW` and parent CLI exit0.

All other outcomes return1 with partial evidence where the process could write
it. If the child died before writing its report, update counts/new_training may
remain unknown; no zero updates are invented. Root still must observe the actual
owner terminal, verify the source/spec/transport and private report references,
inspect the six-step/gradient/BN/resource evidence, and record a separate DGX
pilot acceptance. This code never updates U40 frontier, optimizer/RNG, Windows
normal/failed/legacy counters, or its existing qualification prefix. No AP,
convergence, full741 stability, speedup or cross-environment equivalence is claimed.

## Development evidence

First temporary-file allowlist test: missing-module RED `a8f167`, then a real
unknown-path exception discrepancy `2a1f25`, corrected GREEN `895ce0`.
Resource boundary test: RED `db8985` then GREEN `99580a`.
PID birth/remaining-group/terminal test: RED `ac3718` then GREEN `ef0e09`.
Final three-test stdlib run after the narrow signal/prctl changes: `e7554c`,
3 tests in0.008s. These checks deliberately do not pretend to exercise Linux
signals, resource monitoring, real waitpid or model execution on this Windows host.
