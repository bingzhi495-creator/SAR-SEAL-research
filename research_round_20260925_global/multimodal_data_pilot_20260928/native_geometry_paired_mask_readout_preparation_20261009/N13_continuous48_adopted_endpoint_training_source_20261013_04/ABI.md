# N13 continuous48 qualified/adopted endpoint training v4 ABI

Status: **UNBOUND SOURCE / STATICALLY AUTHORED / UNEXECUTED**, 2026-10-13. Files: `train.py`, `qualify_train_cpu.py`, this ABI. Authored from independent public-release v3/D02 SOURCE copies. No original SCI, SPEC, Q, cache, GT or model was read, hashed, imported or executed during authorship. Original v2/v3 files and all qualified artifacts remain immutable.

This version implements only the remaining complete eight-epoch blocks of the original study. It adds explicit metadata qualification for successful v3/v4 parents and for the one separately CPU-adopted original failed seed701 E08 endpoint. It does not alter scientific formulas, training points, heads, optimizer settings, seeds, epoch budget or evaluation rules.

## Schemas and execution

GPU: `train.py --spec <absolute spec> --owner-execute`, under the existing finite ROOT-owned driver.

- SCHEMA: `ROOT-N13-sameK-region-point-training-spec-v4`.
- REPORT_SCHEMA: `N13-sameK-region-point-training-report-v4`.
- STATUS: `N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION`.
- CP_SCHEMA: `N13-sameK-region-PR-FL91-head-checkpoint-v4`.
- START_SCHEMA: `N13-sameK-region-point-training-start-v4`.
- LOADED_SCHEMA: `N13-sameK-region-point-training-loaded-state-v4`.
- CPU_SPEC_SCHEMA: `ROOT-N13-sameK-region-point-training-CPU-spec-v4`.
- CPU_Q_SCHEMA: `ROOT-N13-sameK-region-point-training-CPU-qualification-v4`.
- CPU_Q_STATUS: `ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V4`.

CPU: `qualify_train_cpu.py --spec <absolute spec> --owner-execute`, under the existing CPU-only owned driver. Written REPORT/Q files do not certify their own held terminal. ROOT must separately verify actual exit, reap, original birth absence and exact admission/identity before accepting a leaf.

GPU normal_RAW_count is at least45; CPU at least46. Each equals the complete prelocked actual normal list length. Additional failed rows remain exact6 plus the primary failure, totaling seven. Original failed seed701 GPU stays in that failed chain. Exact identity pins cover every normal and failed RAW; all held exits, reap/birth absence and pins are checked before the first SHA, SOURCE load or numerical import. Counts increase with actually closed interleaved leaves rather than a seed-local formula.

All original owner/bootstrap/phase-binding checks, complete source closure, pinned libraries, no-retry rule and current admission checks are retained. GPU allocator4096MiB, whole-GPU8192MiB, CPU affinity[0,1], two threads, original Torch2.0.1+cu118/NumPy1.23.5 and deadline<=1800 remain. CPU must keep CUDA uninitialized.

## Unchanged main SPEC fields

GPU fields retain the v3 contract: schema, phase, entry_source, sources, RAW_inputs, normal_RAW_count, prior_failed_CPU_receipt, additional_failed_RAW_inputs, terminal_identity_pins, GPU_admission_file, output_directory, deadline_seconds, allocator_limit_MiB, whole_GPU_limit_MiB, recipe, validation_protocol, seed, first_epoch, last_epoch and resume.

The original23 GPU REF_KEYS remain:

`entry_source, region_common_source, cache_contract_source, point_heads_source, original_PR_point_heads_source, PR_factory_CPU_qualification, cache_entry_source, cache_spec, cache_report, cache_GPU_admission, cache_GPU_RAW, cache_CPU_qualification, cache_CPU_source, cache_CPU_spec, cache_CPU_admission, cache_CPU_RAW, FIT, relationship_entry_source, relationship_spec, relationship_report, relationship_GPU_admission, relationship_GPU_RAW, relationship_ROOT_qualification`.

CPU fields retain the original contract: schema, phase, sources, RAW_inputs, normal_RAW_count, prior_failed_CPU_receipt, additional_failed_RAW_inputs, terminal_identity_pins, CPU_admission_file, Q_path, deadline_seconds, recipe, numeric_protocol, validation_protocol, seed, first_epoch and last_epoch. It inherits the current GPU SPEC's resume contract through its bound training_spec.

The original16 CPU REF_KEYS remain:

`entry_source, training_entry_source, training_spec, training_report, training_GPU_RAW, training_GPU_admission, region_common_source, cache_contract_source, point_heads_source, cache_CPU_qualification, relationship_entry_source, relationship_spec, relationship_report, relationship_GPU_admission, relationship_GPU_RAW, relationship_ROOT_qualification`.

Old region_common/cache_contract/PR factory refs remain the exact previously qualified refs. Their cache/relationship qualification uses its actual historical failure prefixes; this entry never edits old specs or truncates the current seven-failure list to imitate an older list.

Seeds are exactly202610120701/202610120702/202610120703. first_epoch must be one of9,17,25,33,41; last_epoch equals first_epoch+7. This entry has no cold-start branch and accepts no null resume. The final48 budget, fixed8/24/48 curves and no-best-epoch rule remain.

## Typed resume contract

Every resume is an exact dictionary with these11 common keys:

`type, training_entry_source, training_spec, training_GPU_RAW, training_GPU_admission, CPU_entry_source, CPU_spec, CPU_qualification, CPU_RAW, CPU_admission, checkpoints`.

All values except type/checkpoints are exact absolute `{path,bytes,sha256}` refs in the current sources closure. checkpoints is exactly `{LOG_PR: <checkpoint_ref>, GEO_PR: <checkpoint_ref>}`. Both arms must belong to the same seed and the exact completed epoch preceding first_epoch.

### Successful v3/v4 parent

type is `QUALIFIED_SUCCESSFUL_GPU_ENDPOINT`. Add exactly one key, `training_report`, giving12 keys. A v3/v4 GPU report and the corresponding v3/v4 CPU qualification are required; original v2 success reports are not an accepted alternative.

The metadata checks explicitly map parent train schema v3 to report/checkpoint/CPU/Q v3 and parent train schema v4 to those v4 schemas. They verify same recipe/seed/cache/factory/relationship, actual report and Q paths, complete nine artifacts, both checkpoint/proof refs, endpoint epoch/update counts, independent CPU checks and evidence boundaries.

The parent GPU0 and CPU0 are found by exact receipt/source/spec/admission refs in the full current normal history. Their historical indices gi and ci must satisfy ci=gi+1. The parent GPU SPEC RAW list equals current[:gi]; the parent CPU SPEC and Q RAW lists equal current[:ci]. Both historical pin sets must be exact current subsets. **The parent is not required to be the latest current ancestor**, allowing three seeds to interleave without loading another seed's endpoint.

### Adopted original failed seed701 parent

type is `CPU_ADOPTED_FAILED_GPU_ENDPOINT`. Add exactly `recovery_report` and `ROOT_adoption_record`, giving13 keys. **There is no training_report key and no invented original GPU report.**

This branch is restricted to seed202610120701 first_epoch9 last_epoch16, the original v2 E08 two checkpoints and the one known ROOT adoption record. Its pinned record ref must have bytes11137 and SHA256 `a53834223f6a59b925a11d2be1e0aa63d57febbb4a9084220be30c86e1043b32`.

Record schema: `ROOT-N13-completed8-failedGPU-endpoint-adoption-v1`.

Record status: `ROOT_ACTUAL_CPU0_COMPLETE_E08_ADOPTED_ORIGINAL_GPU_EXIT1_PRESERVED`.

The record's CPU_spec/CPU_admission/CPU_RAW/CPU_phase_binding/CPU_bootstrap/CPU_report/CPU_qualification refs must join the separately successful recovery CPU at normal index40. Its original_GPU_failure_record/original_GPU_RAW must join the recovery Q and the original failed GPU. The recovery CPU spec/Q/report/source/admission, exact nine original_artifacts, checkpoint_refs and final_endpoints must all agree. Full recovery-source count8740 and historical40-normal+7-failure pins are preserved; ROOT's post-close record has normal41, failures7, seed701, completed8, updates13288, fixed48 and original session56178.

The recovery schemas/status remain explicit v3 recovery schemas, distinct from ordinary training CPU Q. The Q may still carry its reader's pending-terminal status: adoption additionally requires the immutable ROOT record and the actual CPU0 normal receipt. The reader's pending Q alone never authorizes adoption.

Original GPU actual1 remains the seventh failed RAW, with its exact source/spec/admission, phase binding, bootstrap and pins. It is never inserted into normal history. Original training SPEC inherits its six-failure prefix; recovery SPEC/Q inherit all seven. No original SPEC is rewritten or replaced with the current full history.

The recovery view must be labelled `ARTIFACT_DERIVED_COMPATIBILITY_VIEW_NOT_ORIGINAL_GPU_REPORT`, original exit1, original report written=false. Original output/REPORT.json must remain absent at parent admission and after current GPU/CPU computation. All original checkpoints/proofs/artifacts remain source-bound and are never modified. Recovery record flags preserve no optimizer update/retraining/replayed trajectory/AP/efficacy and continuation-only adoption.

This is the sole path accepting parent checkpoint schema v2. Generic future failed runs, a recovery Q without ROOT closure, another seed, another endpoint or an ordinary v2 parent are rejected.

## Uniform state continuity and current outputs

parent_gate returns a checked parent type, original parent SPEC/report-or-labelled-view/CPU SPEC/Q, selected CP_schema, exact checkpoint summaries and historical owner indices. GPU make_pair uses that selected CP_schema. It loads parent weights, endpoint history, gradient activity and private image-order RNG without changing numerical operations. Existing AdamW construction then loads the exact archived state; its schedule, moments and step counts are preserved.

CPU independently checks current INITIAL_LOADED_STATE against both archived parent checkpoints: exact weights/optimizer/history/gradient counts/order/private RNG, original source/spec/GPU admission and parent epoch/seed/factory/cache. Parent checkpoint v2 is admitted only with the adoption type, seed701 and first_epoch9. Successful parent types accept only v3/v4. Current final checkpoints must always be v4.

Current START/INITIAL_LOADED_STATE/checkpoint metadata includes parent_endpoint_type and parent_checkpoint_schema. Current REPORT/Q includes resume_parent_type, resume_parent_checkpoint_schema and resume_parent_CPU_qualification. These fields disclose the adoption path without changing or relabelling an older file.

All later successful v4 endpoints may be regular parents, including v4 endpoints whose bound history started from the adopted v2 E08 endpoint. Their complete provenance retains that earlier failed GPU and recovery qualification. No seed borrows another seed's initialization.

## Numerical equality and evidence limits

RECIPE is text-identical to the independent published v3/D02 recipe: same40497 PR head, LOG_PR/GEO_PR, same per-seed paired initialization, private image permutations and GT order, frozen points, external region gate, constant AdamW3e-4/weight_decay1e-4/original betas and epsilon, no accumulation/clip/augmentation, instance-mean point BCE and unchanged editable denominators. No new head, sampler or mathematical feature is added.

Training functions digest/cpu_tree/same_tree/check_head/optimizer_proof/cache_packet/endpoint/stability/train_epochs/check_state/imported_files are statically text-identical to published v3. make_pair only changes the admitted parent schema; save_checkpoint/save_loaded/run add metadata and the original-missing-report integrity check. Training forward/backward/update/RNG operations remain unchanged.

Independent CPU finite_state/optimizer_check/records_check/independent_logits/endpoint_check/digest/same_tree/resources/imported_files are text-identical. load_and_check_states changes only qualified parent provenance/schema and explicit current metadata checks. NUMERIC stays endpoint BCE rtol1e-4/atol1e-6 and point BCE arithmetic rtol1e-4/atol1e-6.

Each new leaf has13288 updates per arm and no repeated earlier epochs. Complete48 remains79728 updates per arm per seed. CPU checks final full-FIT point BCE and editable BCE, all recorded visits, paired initialization, loaded continuity, private order RNG, finite AdamW state/steps and proof digests. It does not independently replay gradients/AdamW arithmetic or recompute prior/intermediate epoch losses. Those losses remain bound GPU records, including the adopted original E01–08 records.

GPU/CPU result construction uses gate['common'].RELATION_REF_KEYS; the two old bare-common report failures are not reintroduced. Fresh output/Q paths and no automatic retries remain. There is no DEV/TEST/AP, native-model update or method-efficacy/novelty/TGRS claim in this entry.

## Static validation and handoff

Two entries passed stdlib-only Python3.8 AST parsing and compile, without executing SOURCE. A static text check confirmed the unchanged numerical functions and RECIPE noted above. No numerical toy test, model forward or original scientific payload access occurred. Once all three files are STOP, one independent focused static review precedes ROOT binding, publication and any actual execution.
