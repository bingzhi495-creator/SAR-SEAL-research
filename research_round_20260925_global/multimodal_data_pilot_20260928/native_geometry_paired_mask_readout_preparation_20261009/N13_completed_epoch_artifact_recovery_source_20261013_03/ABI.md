# N13 completed-epoch artifact recovery CPU ABI — 2026-10-13

Status: new SOURCE, statically authored and not executed. The original v02 six files are immutable. This entry is restricted to the original failed seed202610120701 E01–08 GPU attempt, not a general recovery tool. It does not implement v03 training.

## Entry and schemas

Entry: recover_train_cpu.py --spec <absolute ROOT spec> --owner-execute, through the existing single owned finite CPU driver only.

- SCHEMA: ROOT-N13-completed-epoch-artifact-recovery-CPU-spec-v3
- REPORT_SCHEMA: N13-completed-epoch-artifact-recovery-CPU-report-v3
- Q_SCHEMA: ROOT-N13-completed-epoch-artifact-recovery-CPU-qualification-v3
- Q_STATUS: ROOT_N13_COMPLETED8_FAILED_GPU_ENDPOINT_CPU_CHECKED_PENDING_ROOT_TERMINAL

The Q status means checks completed inside the reader. Actual owned CPU exit0, reap and original birth absence must subsequently be checked by ROOT; the entry does not authorize resume itself.

## Exact required spec fields

schema, phase, entry_source, sources, RAW_inputs, normal_RAW_count=40, prior_failed_CPU_receipt, additional_failed_RAW_inputs (exact6 three-key receipt/phase/returncode rows), terminal_identity_pins (exact47), CPU_admission_file, output_directory, report_path, Q_path, recovery_protocol, preregistered_protocol, artifacts, recipe, validation_protocol, numeric_protocol, seed=202610120701, first_epoch=1, last_epoch=8, resume=null, deadline_seconds=1800, allocator_limit_MiB=4096, whole_GPU_limit_MiB=8192, and all19 REF_KEYS below.

Normal owner rows have exact receipt/phase/admission/phase_binding/bootstrap keys. Normal40 are actual0; the primary plus additional6 failures are actual1. The last additional failure is the original failed GPU701 attempt. No failed owner is included in the normal list. Every47 original held exit/reap/birth absence and exact admission/identity pin is checked before the first SHA, SOURCE load, numerical import or artifact read. Complete closure hashing occurs only afterwards. The complete current gate is repeated before final integrity checks.

REF_KEYS (19):

- entry_source
- training_entry_source
- training_spec
- training_GPU_admission
- training_GPU_RAW
- training_GPU_phase_binding
- training_GPU_bootstrap
- training_failure_ROOT_record
- original_CPU_validator_source
- region_common_source
- cache_contract_source
- point_heads_source
- cache_CPU_qualification
- relationship_entry_source
- relationship_spec
- relationship_report
- relationship_GPU_admission
- relationship_GPU_RAW
- relationship_ROOT_qualification

Every ref is exact {path,bytes,sha256}, absolute and sources-bound. All original training sources and protocol are inherited exactly. The original CPU validator must be the frozen qualify_train_cpu.py beside the original training source, present in its original sources. Relationship six refs are identical to the historical original training spec. recovery_protocol and preregistered_protocol are additionally sources-bound; the latter must already be in original training sources.

NUMERIC is the unchanged old CPU literal: endpoint_BCE_rtol=1e-4, endpoint_BCE_atol=1e-6, point_BCE_arithmetic_rtol=1e-4, point_BCE_arithmetic_atol=1e-6. Recipe is copied exactly from the original v2 spec and checked against the bound original train.RECIPE. No optimizer schedule, budget, seed, head or region-selection change occurs.

## Exact original artifact mapping

artifacts uses the same nine filename keys and pinned values as the ROOT failure-preservation record:

- START.json
- TRACE.jsonl
- UPDATE_TRACE.jsonl
- DECISION.json
- INITIAL_LOADED_STATE.pth
- HEAD_LOG_PR_SEED202610120701_EPOCH08.pth
- HEAD_LOG_PR_SEED202610120701_EPOCH08.PROOF.json
- HEAD_GEO_PR_SEED202610120701_EPOCH08.pth
- HEAD_GEO_PR_SEED202610120701_EPOCH08.PROOF.json

Each file retains its path under the original training output_directory, and is hashed in the closure before any torch.load. The original output_directory/REPORT.json must remain absent before, during and after recovery. The fresh separate output_directory must not exist or lie inside the original directory. report_path and Q_path equal its REPORT.json and QUALIFICATION.json. No original payload or metadata is written or renamed.

## Preserved failed-owner join

training_GPU_RAW remains actual1, with its own original admission, PHASE_BINDING and bootstrap AST checked through common.owner_join. The old train.source_gate is called only with the unchanged original spec and owner_environment=False, after the new47 terminal gate. It validates historical40 normal plus6 failures; it does not qualify its own failed current owner. Original train.run, original CPU source_gate and original CPU qualify are never called.

training_failure_ROOT_record has schema ROOT-N13-training-attempt02-failure-preservation-v1 and status ORIGINAL_GPU_EXIT1_E01_08_ARTIFACTS_PRESERVED_NOT_QUALIFIED. Its top-level GPU_spec/GPU_admission/GPU_RAW/phase_binding/bootstrap/artifacts must join exactly; stdout/stderr are also closure-bound. Its full_source_checked_count matches original source count8719, normal40/failed7 and fixed failure flags. RAW actual1 and all47 identity pins remain authoritative. Preserved elapsed/whole-GPU peak/minimum RAM must equal the original RAW values exactly and retain ROOT owned-observation scope; minimum RAM is at least16GiB. Original unsaved allocator peak, allocated_before/after, imported-library vector and original report elapsed value remain unknown.

## Reused independent checks and explicit difference

The bounded original CPU reader supplies finite_state, load_and_check_states, imported_files and endpoint_check; its independent_logits and optimizer_check are reached by those helpers. Its source_gate and qualify are never used. Explicit gate contains original historical cache ABI, new CPU admission and an artifact-derived compatibility view.

The new view is labelled ARTIFACT_DERIVED_COMPATIBILITY_VIEW_NOT_ORIGINAL_GPU_REPORT. START and unchanged checkpoints supply provenance, paired cold proofs, history, gradient counters and the exact checkpoint/proof refs. It never receives a GPU REPORT schema or fabricated cleanup fields.

records_check copies the original recorded permutation/update/gradient-count formulas. Its only substantive difference removes the final original-GPU-REPORT scope/cleanup assertion, replacing it with exact complete8/13288-count assertions. This is documented because original cleanup counters were never saved. All update records, per-epoch online means, checkpoint histories, paired query/annotation visits, private RNG reconstruction and gradient presence/nonzero counters remain checked.

The reader validates eight finite FP32 tensors/40497 scalars, identical per-seed cold weights, fresh INITIAL state, all13288 paired update records, cumulative AdamW steps/moments, unchanged proof digests, final full1661-instance BCE and editable BCE/counts/empty denominators. It constructs cold heads only; it never creates an optimizer or executes an update. It does not replay gradient/AdamW arithmetic or independently recompute intermediate epoch losses.

## Outputs and next use

New REPORT.json records new entry/spec/admission, all refs and nine artifacts, labelled derived view, independently recomputed endpoints, identity/count checks and unknown original GPU fields. New QUALIFICATION.json binds that report and the same adoption inputs/checkpoints and checks. Both retain original_GPU_actual_returncode=1, original_GPU_report_written=false, original_exit_is_not_relabelled=true, training_rerun=false and current_owned_CPU_terminal_self_certified=false.

CPU resources: actual affinity[0,1], Torch intra/inter threads2, CUDA_VISIBLE_DEVICES empty and CUDA never initialized; original Torch2.0.1+cu118/NumPy1.23.5 and imported actual files must be closure/admission-bound. The entire leaf deadline is1800 with no automatic retry. Failure or timeout preserves another failed attempt; it does not declare method failure or license a tolerance change.

Future adoption requires ROOT actual CPU0 qualification and a separately registered v03 adoption-aware resume branch. Its original E08 checkpoint/optimizer/private RNG remains unchanged. Original GPU1 is never relabelled or added to normal history. After recovery CPU0, the next cold seed702 and703 must still be completed before seed701 block2; no seed borrows another seed's initialization. This source does not prove training efficacy, convergence, novelty or TGRS readiness.
