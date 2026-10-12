# N13 qualified canonical-relation training v02 ABI

Status: UNBOUND SOURCE / NOT EXECUTED. 2026-10-13. This directory is an independent version. The original v01 GPU cache, common, failed CPU qualifier and unexecuted v01 training files remain frozen. ROOT alone admits, executes and closes actual owned terminals; this source cannot self-qualify.

## Files and scope

- point_heads.py: byte-for-byte ordinary staged copy of the original N10 PR-FL91 factory (40,497 scalars; eight tensors); no new architecture.
- cache_contract.py: stdlib-only small adapter; forwards original common packet/region/input/logit helpers and replaces cache-admission metadata only. It also contains an explicit CPU historical gate derived from the released common; it never calls the old common.source_gate with a trimmed current SPEC.
- qualify_cache_cpu.py: original independent CPU regions, private sampling, hard GT/e/C/query identity, alias and feature checks; changes whole-L CPU allclose to recorded diagnostics under the new validation protocol.
- train.py / qualify_train_cpu.py: metadata/version/closure changes only; fixed 48-epoch paired training and independent endpoint checks retain v01 numerical recipes.

region_common_source remains the original v01 common reference. cache_contract_source is the new adapter reference. Nothing rewrites the original GPU SPEC/common/source ref. The original packet and validation-carrier schemas, cache GPU schema/report/status and recipe/regions_recipe/numeric_protocol are inherited exactly. Training still loads only core packets; sidecars are bound in the closure but never torch-loaded during epochs.

## New schemas

Cache CPU SPEC: ROOT-N13-sameK-region-FIT-cache-CPU-spec-v2
Cache Q: ROOT-N13-sameK-region-FIT-cache-CPU-qualification-v2
Cache Q status: ROOT_N13_SAMEK_REGION_FIT_CACHE_CPU_QUALIFIED_V2
Training SPEC/report/CP/START/LOADED and training CPU SPEC/Q are the corresponding v01 names with suffix -v2. Training STATUS remains N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION. Training CPU Q status is ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED_V2. These schema changes describe new admission evidence; they do not change the head or optimization.

## RAW-first and historical joins

Every entry checks ALL normal terminal0 and all six original terminal1 receipts, unique locators, held_handle WAIT_OBJECT_0, reaping, identity-specific original birth absence and resources before its FIRST SHA or source/numerical import. terminal_identity_pins must exactly cover the complete normal plus failure locator set; each pin has exactly phase/returncode/original_identity/admission_sha256 and must equal its original RAW values. Additional failed rows are exactly receipt/phase/returncode=1, five rows plus the separate primary prior_failed_CPU_receipt. No failed row is incorrectly interpreted as a normal owner row.

New cache CPU requires 39 normal inputs (indices0..38), six failures and45 pins. Original N13 GPU owner is index37 with its original 37-row ancestry and primary+additional[:3] historical failure list. N12 GPU/CPU remain indices35/36, and the N12 Q also inherits only additional[:3]. Original N07 GPU/CPU indices17/18 and context closure remain unchanged. The separate relation owner is index38.

First training leaf needs all40 normal ancestors including successful cache CPU, plus six failures and46 pins. Subsequent leaves bind normal_RAW_count to their exact complete historical list; seed leaves may interleave. The successful cache CPU owner is located through its exact source/spec/admission/RAW join; its input prefix is exactly the CPU SPEC's39 original ancestors. Training CPU needs the complete training prefix plus that leaf's actual GPU0 (first41 normal), with exact pins and sources. Parent resume checkpoints must have independently closed GPU+CPU ancestry and preserve the parent pins as exact subsets.

## Relationship refs and ROOT record

All cache CPU, training and training CPU SPECS contain these six pinned path/bytes/sha256 references:

- relationship_entry_source
- relationship_spec
- relationship_report
- relationship_GPU_admission
- relationship_GPU_RAW
- relationship_ROOT_qualification

The first five point to the frozen relationship GPU entry, its SPEC/report/admission and actual normal38 RAW. The last points to ROOT's post-terminal record, schema ROOT-N13-frozen-M-to-L-relationship-GPU-qualification-v1, status ACTUAL_GPU_TERMINAL0_CANONICAL100_ALL_MATCHED_BYTES_EXACT_NOT_CACHE_Q.

The adapter independently joins actual normal38 original RAW/owner. ROOT record actual_identity must exactly map original_identity.pid/birth_unix/birth_filetime_100ns and the row phase/admission SHA, plus actual0/held/reaped/birthabsent. The record's five refs must exactly match the SPEC. Record booleans never replace actual RAW verification.

The relation SPEC/report bind the same original cache GPU37 refs and complete38-row ancestry with all six failures. The successful report requires741 images/1661 matched queries, canonical100_all_exact_bytes=True and image/query/pixel mismatches0, cleanup allocated_after0 and Q_written/cache_qualified/current-self-terminal=False. Every row is tied to the same query IDs, image identity, packet/carrier refs and input tensor digests, and all matched replay bytes are exact. canonical100 means a contiguous [1,100,200,200] carrier with unmatched slots filled by dummy zeros, followed by the original renderer and then matched-query selection. It neither recovers discarded unmatched M200 nor repeats teacher inference. subset_G results are diagnostics only.

## Cache CPU evidence protocol

The inherited numeric_protocol remains the original values. The separate new validation_protocol must exactly equal this adapter literal:

```python
VALIDATION_PROTOCOL=dict(
 canonical_GPU_M_to_L_exact_bytes_ROOT_qualification_required=True,
 canonical_GPU_relation_images=741,canonical_GPU_relation_matched_queries=1661,
 native_L_CPU_full_allclose='diagnostic only, original v1 rtol/atol unchanged',
 native_L_CPU_rtol=1e-4,native_L_CPU_atol=1e-4,
 sampled_F_rtol=1e-4,sampled_F_atol=1e-4,sampled_F_allclose_is_strong_gate=True,
 exact_saved_GPU_L_is_sole_selector_and_training_logit=True,
 saved_GPU_L_regions_draws_labels_CPU_exact_replayed=True,
 CPU_replay_never_substitutes_saved_GPU_L_P_B_K=True,
 unmatched_M200_recovered=False,native_teacher_forward_replayed=False,
 certified_GPU_error_bound=False,historical_failures_preserved=True)
```

The CPU uses saved GPU L to independently replay P, exact LOG/GEO same-K regions, tie order, private draws, GT labels and region flags. Sampled F reconstruction retains the original rtol/atol1e-4 strong gates, including the original cache sampled F relation; original e/C and query correspondence are exact. The original sampled-L-to-old-cache practical gate is also retained. Full cross-device native-L allclose is computed with the unchanged v01 thresholds and finite CPU output, but no longer admits/rejects this new Q. Actual per-image pass/fail, failing pixels and max absolute error are recorded; the aggregate may be False. No observed-error-derived tolerance is introduced.

New Q flags canonical_GPU_M_to_L_exact_bytes_ROOT_qualified / saved_GPU_L_geometry_and_sampling_CPU_exact_replayed / hard_targets_CPU_checked / F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked / sampled_F_original_two_stage_CPU_replayed are True only after their gates. native_L_CPU_full_allclose_is_qualification_gate=False. Ambiguous old full_native_L_geometry_and_sampling_CPU_replayed and practical_native_L_CPU_replayed claims are omitted. current_owned_CPU_terminal_self_certified=False; ROOT must close this new CPU0 before training.

## Required cache CPU refs

Original common REF_KEYS are all inherited (21 refs):

- entry_source
- region_common_source
- context_GPU_spec
- context_CPU_qualification
- FIT
- original_cache_entry_source
- original_cache_spec
- original_cache_report
- original_cache_CPU_qualification
- original_cache_CPU_source
- original_cache_CPU_spec
- original_cache_GPU_admission
- original_cache_GPU_receipt
- original_cache_CPU_admission
- original_cache_CPU_receipt
- prior_N12_GPU_report
- prior_N12_CPU_qualification
- prior_N12_CPU_source
- prior_N12_CPU_spec
- prior_N12_CPU_admission
- prior_N12_CPU_receipt

Add cache_entry_source/cache_spec/cache_report/cache_GPU_admission/cache_GPU_RAW, cache_contract_source and the six relationship refs (33 refs total). entry_source is this new CPU entry; cache_entry_source is original GPU v01. Other ordinary SPEC controls: schema/phase/sources/RAW_inputs/prior_failed_CPU_receipt/additional_failed_RAW_inputs/terminal_identity_pins/CPU_admission_file/Q_path/deadline_seconds/allocator_limit_MiB=4096/whole_GPU_limit_MiB=8192/recipe/regions_recipe/numeric_protocol/validation_protocol/image_root/matching_loss_seed_base/region_rank_seed_base/uniform_seed_base. Deadline is a positive ROOT-locked integer; cache CPU is not automatically limited to the training1800-second leaf.

## Exact training REF_KEYS

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

entry_source/point_heads_source/cache_contract_source refer to new v02 files. original_PR_point_heads_source and PR_factory_CPU_qualification retain original N10 qualified source/Q refs at historical CPU28; new point_heads bytes/SHA must match the old factory although paths differ. region_common_source/cache_entry_source/cache_spec/cache_report/cache_GPU_admission/cache_GPU_RAW remain original GPU/common v01 refs. cache_CPU_source/spec/qualification/admission/RAW refer to the successful new v2 qualification; FIT is inherited. All sources, packet/carrier refs and library implementations are in the full closure.

Additional required training controls: schema/phase/sources/RAW_inputs/normal_RAW_count/prior_failed_CPU_receipt/additional_failed_RAW_inputs/terminal_identity_pins/GPU_admission_file/output_directory/deadline_seconds/allocator_limit_MiB/whole_GPU_limit_MiB/recipe/validation_protocol/seed/first_epoch/last_epoch/resume. First leaf: seed=202610120701, first_epoch=1, last_epoch=8, resume=None, cold paired weights. This is13288 updates per arm /26576 model optimizer steps /13288 paired UPDATE_TRACE records. No previous N10 or v01 endpoint is accepted.

## Exact CPU training REF_KEYS

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

Training CPU controls mirror complete current ancestry, terminal pins and original training recipe/validation_protocol/seed/first_epoch/last_epoch; add numeric_protocol equal qualify_train_cpu.NUMERIC, CPU_admission_file/Q_path/deadline_seconds. It accepts only the bound v2 GPU training ABI, full current artifacts and the same relationship refs. CPU step/optimizer replay is not claimed; final FIT and editable BCE, exact cold/resume continuity, private RNG and recorded update/optimizer counts remain independently checked.

## Exact unchanged training RECIPE

ARMS=('LOG_PR','GEO_PR'); SEEDS=(202610120701,202610120702,202610120703). The literal recipe expression remains identical to v01:

```python
RECIPE=dict(arms=list(ARMS),seeds=list(SEEDS),fixed_final_epochs=48,
    updates_per_epoch_per_model=1661,images_per_epoch=741,points_per_GT=1024,
    final_updates_per_model=79728,instance_mean_BCE=True,one_update_per_GT_instance=True,
    no_accumulation=True,AdamW_lr=3e-4,AdamW_weight_decay=1e-4,AdamW_betas=[.9,.999],
    AdamW_eps=1e-8,constant_lr=True,gradient_clip=None,fixed_cached_points_all_epochs=True,
    no_augmentation=True,head_parameters=40497,head_parameter_tensors=8,
    paired_seed=True,paired_initialization=True,paired_image_order=True,
    raw_GT_order_within_image=True,external_region_gate=True,
    empty_region_updates='K0 has zero data residual gradient; ordinary AdamW step incl momentum/decay is retained and counted',
    logits='native_L + point_in_region * ungated_PR_residual',
    training_inference_regions_identical=True,GT_used_for_region_selection=False,
    supervision='768 draws in each arm region plus common256 global draws; fixed cache',
    treatment='support-domain and sampled-supervision joint comparison',
    endpoint_BCE='full1661 instance mean over1024 cached points after every epoch',
    editable_BCE='mean of within-region point means over instances with >=1 editable sampled point',
    convergence_windows=[list(range(37,43)),list(range(43,49))],
    editable_relative_drop_threshold=.01,convergence_rule='relative drop >1% flags not stable; does not extend48',
    curve_epochs=[8,24,48],best_epoch_selection=False,native_weights_updated=False,
    resume='only qualified own N13 complete-epoch checkpoints; preserve optimizer and private order RNG')
```

Resource gates retain CPU affinity[0,1], threads2, CUDA_VISIBLE_DEVICES='0' for GPU / '' for CPU, whole GPU8192MiB, allocator4096MiB, one seed pair at up to8 complete epochs per leaf, deadline positive <=1800sec, Torch2.0.1+cu118/NumPy1.23.5, no retry. The recipe is fixed48epochs /79728 updates per arm /three paired seeds. Leaves resume exact optimizer/private shuffle RNG; windows37–42 and43–48 flag editable loss drop>1% as not stable without extending48. Empty-editable instance counts and denominator are retained.

This metadata repair does not establish training efficacy, convergence, novelty, SOTA or TGRS readiness. Failed v01 L checks remain failed under their original criteria. New Q is a separately registered evidence protocol, not a retroactive passing label.
