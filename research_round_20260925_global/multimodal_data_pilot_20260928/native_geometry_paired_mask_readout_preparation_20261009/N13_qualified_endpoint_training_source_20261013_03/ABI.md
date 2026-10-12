# N13 qualified endpoint training v3 — cold-only ABI, 2026-10-13

Status: new SOURCE, statically authored and not executed. Created from independent github_release D02 copies. Original D02 files, factory, cache adapter and region common stay unchanged and retain their original exact references. This v3 entry admits only cold seed202610120702 or202610120703 E01–08 with resume=null. It rejects every resume, including any recovered failed-GPU endpoint. It does not implement adoption-aware continuation.

## Schemas and entry

GPU train.py --spec <absolute spec> --owner-execute uses the existing finite owned driver.

- SCHEMA: ROOT-N13-sameK-region-point-training-spec-v3
- REPORT_SCHEMA: N13-sameK-region-point-training-report-v3
- STATUS: N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION
- CP_SCHEMA: N13-sameK-region-PR-FL91-head-checkpoint-v3
- START_SCHEMA: N13-sameK-region-point-training-start-v3
- LOADED_SCHEMA: N13-sameK-region-point-training-loaded-state-v3
- CPU_SPEC_SCHEMA: ROOT-N13-sameK-region-point-training-CPU-spec-v3
- CPU_Q_SCHEMA: ROOT-N13-sameK-region-point-training-CPU-qualification-v3
- CPU_Q_STATUS: ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V3

Independent CPU qualify_train_cpu.py uses the corresponding CPU schemas and the same explicit owner flag. A written GPU report or CPU Q does not certify that process's actual held terminal; ROOT closes actual exit/reap/birthabsence separately.

## Exact differences from D02

Only version constants, seven-failure RAW gates and cold-only admission change. Training normal_RAW_count has minimum41; CPU has minimum42. Each equals its prelocked actual RAW list length. Additional failed rows are exact6 with receipt/phase/returncode1; plus primary failure gives seven, distinct from normal RAWs. terminal_identity_pins are exact count+7, so the first cold702 GPU has48 and its CPU49. Counts for later seed703 reflect all actual completed predecessors, not a guessed formula.

All actual terminal/identity checks precede SOURCE hashing/imports. Original failed seed701 remains in failures, never normal. The ROOT-qualified recovery CPU can be a successful provenance ancestor; no recovered checkpoint is loaded by this cold-only source.

parent_gate is replaced by an explicit cold-only check for seed702/703 first_epoch1 last_epoch8 resume=null. The old generalized parent/resume admission has been removed from this entry. Both result constructions now use gate['common'].RELATION_REF_KEYS, fixing the two report-time unbound-common references. All old numerical functions remain unchanged. The CPU loaded-state helper retains an unreachable later-epoch branch, but its entry and the GPU source gate jointly reject all later-epoch/resume inputs.

## Spec and ref contracts

GPU fields are unchanged from the original v2 contract except schema/source/versioned products and current RAW chain: schema, phase, sources, RAW_inputs, normal_RAW_count, prior_failed_CPU_receipt, additional_failed_RAW_inputs, terminal_identity_pins, GPU_admission_file, output_directory, deadline_seconds, allocator_limit_MiB4096, whole_GPU_limit_MiB8192, recipe, validation_protocol, seed, first_epoch1, last_epoch8, resume=null and all23 REF_KEYS.

GPU REF_KEYS (23):

- entry_source
- region_common_source
- cache_contract_source
- point_heads_source
- original_PR_point_heads_source
- PR_factory_CPU_qualification
- cache_entry_source
- cache_spec
- cache_report
- cache_GPU_admission
- cache_GPU_RAW
- cache_CPU_qualification
- cache_CPU_source
- cache_CPU_spec
- cache_CPU_admission
- cache_CPU_RAW
- FIT
- relationship_entry_source
- relationship_spec
- relationship_report
- relationship_GPU_admission
- relationship_GPU_RAW
- relationship_ROOT_qualification

CPU fields preserve the original v2 contract with v3 schemas/current RAW chain: schema, phase, sources, RAW_inputs, normal_RAW_count, prior_failed_CPU_receipt, additional_failed_RAW_inputs, terminal_identity_pins, CPU_admission_file, Q_path, deadline_seconds, recipe, numeric_protocol, validation_protocol, seed, first_epoch1, last_epoch8 and all16 REF_KEYS.

CPU REF_KEYS (16):

- entry_source
- training_entry_source
- training_spec
- training_report
- training_GPU_RAW
- training_GPU_admission
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

CPU inherits exactly training RAWs followed by the actual new GPU0 owner. It requires the actual GPU report, exact v3 checkpoint/start/loaded schemas, all nine artifacts and the same cache/factory/relationship recipe and refs. It does not reuse original failed seed701's nonexistent report. Existing seed701 recovery outputs must be included by ROOT in complete provenance closure when making its successful ancestor row; they do not initialize these seeds.

Old cache_contract_source and region_common_source references remain exact D02/D01 respectively. The unchanged adapter validates historical cache/relationship prefixes using the original additional_failed_RAW_inputs[:5], not the new full6. The original N12/cache GPU prefixes and factory synthetic qualification are unaffected.

## Scientific recipe and validation remain unchanged

RECIPE text is byte-for-byte identical to the independent published D02 copy: LOG_PR/GEO_PR, same40497 PR factory, same per-seed paired cold weights, private image permutation and original GT order; external native_L+point_in_region*ungated_PR_residual; per-region768 plus common256 fixed points; constant AdamW3e-4/weight_decay1e-4, original betas/epsilon, no accumulation/clip/augmentation. The original factory is referenced, not rewritten or renamed.

Each cold leaf has8epochs,13288updates per model,26576optimizer steps across the pair, with741images/1661instances per epoch. The whole preregistered study remains48epochs/79728updates per model/three seeds, fixed8/24curves and48main endpoint; this entry only implements the two remaining first blocks. It does not replace48 with8 or choose a best DEV epoch.

Static text equality confirmed unchanged train functions: digest, cpu_tree, same_tree, check_head, optimizer_proof, save_loaded, make_pair, save_checkpoint, cache_packet, endpoint, stability, train_epochs. CPU functions finite_state, optimizer_check, load_and_check_states, records_check, independent_logits and endpoint_check also remain text-identical. GPU run numerical body is unchanged; only its relation-ref report expression changes scope.

CPU NUMERIC remains endpoint_BCE_rtol1e-4/atol1e-6 and point_BCE_arithmetic_rtol1e-4/atol1e-6. Its checks include paired cold state, complete image/update order,13288 steps and optimizer structure, exact proof/state/RNG, and independent final FIT full/editable BCE with empty counts. It does not replay intermediate gradients/AdamW or claim scientific efficacy.

Resources remain affinity[0,1]/threads2, original Torch2.0.1+cu118/NumPy1.23.5, bound import sources, allocator4096MiB/whole-GPU8192MiB, leaf deadline<=1800 and no retry. CPU must not initialize CUDA. Original output paths are immutable; each new leaf requires fresh output/Q paths.

## Follow-on boundary

ROOT must qualify actual seed702 GPU0 and independent CPU0, then seed703 likewise before moving to block2. Later resume requires a separately registered version with qualified-parent and recovery-adoption contracts; this source refuses it. No checkpoint from another seed, old N10 endpoint or original failed owner is a cold initializer. No dense DEV/AP evaluation, new convergence claim, novelty claim or TGRS claim is performed here.
