# Shared-GPU one-image function probe v02 — executed and failed archive

## Current archive status (2026-10-25)

**V02 ACTUALLY EXECUTED; STRICT EXTERNAL CLOSURE FAILED.** The original tool, SSH, and worker each returned 1. This is not a successful or qualified parent result, and its outputs must not be admitted as a qualified parent for subsequent results.

Root's completed external closure established:

- The real FP32 ViT-H refinement call returned outputs for the fixed 100 candidates, and CUDA synchronization completed.
- Observed peak allocated CUDA memory was 7,011,852,288 bytes.
- The final cleanup gate failed because allocated memory remained 9,568,256 bytes and reserved memory remained 14,680,064 bytes instead of zero. Root identified this as the failure criterion; functional return does not override the original nonzero terminal.
- All 38 source references and the recorded output bytes were verified. The original same-boot child birth was absent, its owned process group was resolved empty, and worker/parent processes were no longer present.
- No AP, GT-based evaluation, or training was performed. This is a resource/lifecycle qualification failure, not a measured accuracy failure or evidence that the refinement method is ineffective.

The private v01/v02 originals remain frozen. Published Python files retain their exact original bytes; preparation-time wording inside those files is historical and does not override the executed-failed status above. The remaining preparation and interface descriptions are retained for auditability, not as evidence of production success.

This archive includes no SPEC or lease instance, runtime observations identifying user processes/hosts, original image, mask, third-party source, model output, or weight. This archiving task did not rerun tests, Torch, SSH, or the model.

## Archived design scope

This version changes admission and reporting contracts only; it neither stops user services nor claims whole-device exclusivity.

worker.py is copied from the original27920-byte worker. FP32/noAMP, ViT-H, original inputs and100 candidates, five iterations/gamma4/strength30, eval/inference_mode,16GiB allocator, RSS16GiB/freeRAM16GiB/startup48GiB,1800 seconds and one attempt remain unchanged. There is no automatic retry, precision/model/iteration reduction or resource upgrade. AP/GT/training remain outside this scope.

## New private interfaces

- Worker SPEC: ROOT-SAMRefiner-one-image-shared-probe-spec-v02.
- Parent SPEC: ROOT-SAMRefiner-one-image-shared-supervisor-spec-v02.
- Operation: GPU_SAMREFINER_ORIGINAL_ORDINAL0_SHARED_FUNCTION_PROBE_V02.
- run_id must start with SAMRefiner-shared-v02- and be independently assigned by Root.
- Worker resource_plan: INDEPENDENT_SHARED_GPU_FUNCTION_PROBE_V02.
- Lease: Root-SAMRefiner-one-image-shared-lease-v02 / ROOT_GRANTED.
- Lease scope: one_image_shared_GPU_resource_and_function_probe_only.

Both SPECs and the lease require exclusive_project_scientific_leaf=true and whole_device_exclusive=false. The old exclusive_GPU field is rejected, including a mixed v02 document that retains exclusive_GPU=true. Old v01 schemas/operation/run cannot qualify.

Both SPECs require gpu_background_observation_ref as an exact absolute path/bytes/full SHA256 reference. It is added to both source-closure unions, must equal the lease reference, and is retained in both reports. Other private input, checkpoint-stage and Root-reference requirements are inherited unchanged. The child has normally22 distinct refs, the parent9; deduplicate by exact canonical path.

The parent pins this directory's new worker bytes/SHA rather than the original worker pin. Source role paths and actual digests are supplied through a separately controlled runtime SPEC; no actual operational SPEC is included here.

## Background observation contract

Required exact JSON fields:

```text
schema: SAMRefiner-shared-GPU-background-observation-v02
boot_id: canonical kernel boot UUID
observed_UTC: timezone-aware UTC ISO8601 timestamp, e.g. ...Z
original_nvidia_smi_returncode: integer0 from the original successful observation
gpu_index: integer0
whole_device_exclusive: false
processes:
  - pid: positive integer
    start_ticks: positive integer from /proc/<pid>/stat
    name: exact kernel /proc/<pid>/comm, without final newline
    used_memory_MiB: nonnegative integer from the original GPU observation
```

processes must be nonempty and have unique PIDs. Each process object permits exactly those four fields. No cmdline, executable path or model/weight path is admitted. Root records the actual nvidia return and observations; candidate code does not fabricate them or invoke nvidia-smi.

Before launch, parent verifies same boot, each observed PID/start_ticks still matches a non-zombie process, and kernel comm still matches name. It only reads stat/comm; it never reads user cmdline or weights. It also retains actual verification that Root's prior scientific group births are absent and their groups are empty. Stale identity fails without automatic refresh.

The observation is a snapshot, not a workload/memory lock or a continuous inventory. Background load can grow, shrink, stop or change after the check. The reported MiB is observed metadata rather than a current reservation. Resource failure, OOM and timing variation under sharing are not evidence of worse method accuracy; duration/cost is diagnostic, not a speed benchmark.

## Invocation and supervision

```text
<qualified-python> -B parent.py --spec <private-parent-SPEC.json> --spec-sha <full-SPEC-SHA256>
```

Root must finish weight/source verification, reap all project scientific processes, qualify this independent shared scope and establish external oversight before invoking it. A metadata field alone cannot prove a live Root controller.

The existing parent lifecycle, actual wait/reap, GNU timeout, SIGTERM/HUP/INT handling, same-boot birth/PGID checks and owned-group cleanup are unchanged. Cleanup targets only the newly owned timeout session/group, never listed background services. SIGKILL cannot run Python finally; external Root terminal/group observation remains required.

Reports use SAMRefiner-one-image-shared-probe-report-v02 and SAMRefiner-one-image-shared-parent-control-v02, explicitly whole_device_exclusive=false, project-leaf exclusivity and speed_benchmark=false. The worker preserves persistent module-stage hooks and CUDA synchronization/cleanup. Decoder counts are not completed refinement rounds. The known first-image RGB400 mismatch remains independent external evidence; no full187 or historical full-resolution equivalence is claimed.

## Historical preparation evidence and differences

The metadata-contract tests use synthetic process-identity fixtures. Those fixtures are not observations of actual user processes or GPU execution, and do not establish scientific or lifecycle qualification.

Before the failed actual attempt, two stdlib metadata-contract tests ran: background schema/boot/UTC/returncode/allowed-field checks and rejection of v01/falsely exclusive grants. RED8ee24e -> GREEN775f04; RED9b74aa -> GREEN9d3849; SPEC-level ambiguity RED813595 -> GREEN02f047. All final GREEN observations exited0. Those preparation tests did not execute any scientific library, model, data, SSH or GPU; the later actual attempt is reported separately above.

The worker tail beginning at modules={} is required to be byte-identical to v01: imports, input validation, FP32 model/weight loading, forward hooks, complete author call, output logic, CUDA synchronization and cleanup are unchanged. Parent wait_child/need/reference/load_source ASTs and its exception-cleanup prefix are unchanged. Only schema/operation/run/resource-plan labels, shared lease/background closure and report/identity fields differ.

The later real Linux/model integration attempt was externally closed as failed, as reported above; it must not be described as unexecuted or successful. Preparation tests cannot establish its scientific or lifecycle qualification. No tests or numerical payload were rerun for this publication. Actual observations, SPEC/lease instances, user-service identities, private inputs, masks, outputs, and weights are excluded.

