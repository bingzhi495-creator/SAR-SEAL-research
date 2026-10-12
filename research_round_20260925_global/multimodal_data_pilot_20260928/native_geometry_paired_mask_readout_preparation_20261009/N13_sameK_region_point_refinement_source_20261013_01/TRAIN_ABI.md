# N13 paired region training SOURCE ABI — 2026-10-13

Status: **UNBOUND / NOT EXECUTED**. This document describes the authoring contract; it is not a registration, terminal receipt, training result or independent qualification. ROOT owns binding and all actual execution. No scientific inputs, numerical libraries or toy runs were used in authoring.

## Files and exact head reuse

This author owns only point_heads.py, train.py, qualify_train_cpu.py and this document. region_common.py and the new cache/CPU cache reader belong to the cache author. Existing bound SOURCE is untouched.

point_heads.py retains the independent staged N10 PR-FL91 factory text exactly: make_point_head(torch, seed), head.forward(F,e,L,patch,arm='PR'), eight named tensors, 40,497 scalars. Shapes in order: hidden1 weight(91,257)/bias(91), hidden2 weight(91,92)/bias(91), hidden3 weight(91,92)/bias(91), final weight(1,92)/bias(1). LN(F256) is non-affine, eps1e-5. c=clamp(L,-8,8)/8 is concatenated before every linear layer; three ReLU hidden layers; the final layer is zero at cold initialization. e and zero patch49 remain validated ABI inputs, without a residual dependence on their values.

**The old head's returned |L|<=2 logits are discarded.** Both arms call region_common.region_logits(torch,head,F,e,L,regionflag), which requests the ungated residual and returns L+u inside the supplied bool flag and L outside. Thus geometry-only editable pixels are not accidentally restricted by the old head's internal band. The CPU endpoint reader independently writes the LN/linear/ReLU/flag formula and never calls head.forward or region_logits for its loss replay. No new architecture primitive or official full PointRend replication is claimed.

A new point_heads_source path is byte/SHA joined to original_PR_point_heads_source. The prior N10 PR_factory_CPU_qualification retains its original source references; it is not relabelled as a qualification of the new path. Its actual held CPU ancestor is index28. The new pair's exact cold weight equality is additionally checked at execution and independently by the new CPU reader.

## Locked per-model recipe and leaves

Seeds: 202610120701, 202610120702, 202610120703. Arms: LOG_PR, GEO_PR. Both arms of each seed have identical cold weights and private image order. Across arms the cache sampling support differs, so sampled logits/labels and initial sampled BCE need not match.

Fixed primary endpoint: 48 epochs, 1,661 instance updates/epoch/head, 79,728 updates/head. No native weights are updated. Each leaf runs one seed pair and at most eight complete epochs: the planned leaves are 1–8, 9–16, 17–24, 25–32, 33–40, 41–48. ROOT may register smaller fixed complete-epoch segments if needed, preserving the same48 endpoint; a segment cannot straddle epoch8 or24 without ending there, to retain those curve checkpoints. Three seeds may be interleaved by common epoch block. A leaf's positive deadline is <=1,800 seconds, never dynamically increased by this entry. Actual initial throughput is unmeasured; a terminal failure is preserved, not automatically retried.

AdamW: LR3e-4, weight_decay1e-4, betas(.9,.999), eps1e-8, constant LR; amsgrad/maximize/capturable/differentiable=False, foreach=None, no clipping, no accumulation. One mean1024-point BCE and one AdamW step per GT instance, in raw GT order within each image. Paired image permutations come from private CPU Generator(seed), uninterrupted across leaves. Fixed cache, no augmentation or resampling, no DEV/TEST or best-epoch selection. Checkpoints and whole optimizer state remain cumulative; N10 checkpoints are explicitly ineligible as N13 parents.

The cache has per arm768 within-region draws plus256 common global draws, with replacement, duplicates retained; K=0 shares1024 global draws. Same-K region selection is GT free, from frozen original native L/P; matching and hard labels remain FIT supervision. This treatment changes **support and sampled supervision jointly**. It cannot isolate support alone. Training and eventual inference must use exactly the same per-query region definition; DEV AP would need all100 native queries, not only the earlier diagnostic's selected pairs.

K=0 contributes no data residual gradient, but the ordinary AdamW step still counts and can change parameters through retained moments/weight decay. Full BCE includes its constant native loss. Editable BCE averages within-region point means over GT instances with at least one editable sampled point; each epoch reports editable_instances, empty_editable_instances and editable_points. Empty editable population yields None, never zero or a stability conclusion.

After every epoch, full1661 FIT losses are recomputed for both arms. At epoch48, editable-BCE means for37–42 and43–48 yield a relative drop; >1% flags NOT_STABLE_GT1PCT_DROP. Other cases only mean the predeclared drop flag is absent, not that convergence or generalization is proven. There is no automatic extension beyond48. Epoch8/24 checkpoints are curve evidence, not selectable winners. ROOT's separate preregistered utility/stop rule and DEV history remain unchanged by these sources.

## Shared cache interface

region_common.load_packet(torch,report,index) reads only the core packet and performs packet_validate once. The packet has matched query identities, query_embeddings[G,256], original matched_native_L[G,H,W], P/region RLE/K, and arms[arm] with F_samples[G,1024,256], native_logits[G,1024], targets(bool), point_indices(int64), point_in_region(bool). instance_inputs returns F,e,L,y,flag on the requested device. Cache rows carry tensor digests and query_ids.

The separate validation_carrier_{index}.pth contains frozen F256/M200 for independent cache CPU replay. All741 carrier references remain in closure and are checked against packet/row references, but neither training epochs nor training endpoint replay loads a carrier, JPEG, GT mask, teacher/native checkpoint or native model. Full native original L stays in the core packet for qualified selection identity; this adds cache I/O compared with N10. Teacher/cache compute is separate from head training.

validate_cache_input(spec,closure,owners) returns GPU_SPEC, REPORT, CPU_Q. It joins actual cache GPU/CPU at indices37/38, validates their original37/38 ancestor prefixes and the completed new cache qualification. It is not the cache entry's fixed37 pre_gate. New training uses its own dynamic RAW count and preserved failures.

## Training SPEC keys

schema = ROOT-N13-sameK-region-point-training-spec-v1.

Control values: phase, sources(list unique byte/SHA/path refs), RAW_inputs, normal_RAW_count, prior_failed_CPU_receipt, additional_failed_RAW_inputs, GPU_admission_file, output_directory, deadline_seconds, allocator_limit_MiB=4096, whole_GPU_limit_MiB=8192, recipe=the literal train.RECIPE, seed, first_epoch, last_epoch, resume.

Primary refs: entry_source, region_common_source, point_heads_source, original_PR_point_heads_source, PR_factory_CPU_qualification; cache_entry_source, cache_spec, cache_report, cache_GPU_admission, cache_GPU_RAW, cache_CPU_qualification, cache_CPU_source, cache_CPU_spec, cache_CPU_admission, cache_CPU_RAW, FIT.

normal_RAW_count equals the actual prelocked RAW_inputs length, >=39 on the first leaf. It is **not** derived as39+2*n per seed: interleaved other seed leaves remain global ancestors. All normal rows use receipt/phase/admission/phase_binding/bootstrap and all refs are sources-bound. There are four failures: prior_failed_CPU_receipt plus exactly3 additional rows, each exactly {receipt,phase,returncode:1}. These failed rows do not acquire fictitious admission/bootstrap fields. All normal/failure held exits, reaping and original birth absence are checked before the first SHA or SOURCE/numerical import. Actual owner joins follow the existing original bootstrap/admission protocol. Neither entry can qualify its own current RAW.

resume is None iff first_epoch=1. Otherwise it is exactly a dict containing training_entry_source, training_spec, training_report, training_GPU_RAW, training_GPU_admission, CPU_entry_source, CPU_spec, CPU_qualification, CPU_RAW, CPU_admission, checkpoints. checkpoints is {LOG_PR: ref, GEO_PR: ref}. The parent must be an actual independently CPU-qualified N13 endpoint, same seed, recipe, factory/cache and last_epoch=first_epoch-1. The full parent GPU/CPU specs/source closures and their true ancestor prefixes join to the global current list. State, optimizer, history, gradient counters and private RNG loaded before the first update are saved and CPU-compared exactly to this parent; loading alone is not accepted as continuity proof.

The actual Torch/NumPy/psutil files, extension binaries, nn/optimizer/serialization/math source files are checked against sources and admission's bound_source_sha256 before and after numerical work. Torch2.0.1+cu118/NumPy1.23.5, actual CPU affinity[0,1], intra/inter-op threads2, GPU visible0, TF32off/cudnn deterministic/allocator4096MiB/wholeGPU8192MiB follow the admitted existing driver. The cache's numeric_protocol concerns renderer replay and is not substituted for this imported-library check.

## GPU outputs and checkpoint contract

START.json: N13-sameK-region-point-training-start-v1, binding source/spec/admission/phase, seed/range/recipe, cache/factory refs, paired cold proofs, resume parent Q, reset=False declarations, and initial_loaded_state.

INITIAL_LOADED_STATE.pth: N13-sameK-region-point-training-loaded-state-v1. It is written/read back before the first update and contains actual per-arm state_dict, optimizer_state_dict, incoming history/gradient counters, cumulative epoch_orders and private_order_state.

TRACE.jsonl: one row per new epoch, paired order SHA,1661 updates, means of recorded online BCE and fullFIT/ editable endpoint dictionaries. UPDATE_TRACE.jsonl: exactly1661 rows per new epoch; each paired GT update records seed/epoch/image_index/GT_index/annotation_id/query_id/global_step, per-arm BCE and eight gradient present/nonzero flags. These are actual bound records, not independent gradient replay.

HEAD_<ARM>_SEED<SEED>_EPOCH<XX>.pth: N13-sameK-region-PR-FL91-head-checkpoint-v1 in meta; state_dict / optimizer_state_dict / private_order_state / meta. Meta holds cumulative epochs/updates/visits, recipe, exact source/spec/admission/cache/factory, parent Q, initial paired cold proofs, complete cumulative per-epoch orders/loss history/gradient counts, and stability arithmetic. training_resume_eligible=True is limited to separately qualified own N13 complete-epoch parents, never a formal native initializer. Its .PROOF.json stores actual state/optimizer/RNG digests, cumulative counts and readback equality.

DECISION.json retains final48, current leaf epoch, no extension/DEV/test selection and per-arm stability flags. REPORT.json schema N13-sameK-region-point-training-report-v1, status N13_SAMEK_REGION_POINT_TRAINING_LEAF_COMPLETE_PENDING_ROOT_QUALIFICATION, two arm checkpoints, all artifacts/histories/cold/gradient data, current leaf and cumulative updates, cleanup and explicit evidence limits. This is pending independent CPU and ROOT actual-terminal closure.

## CPU SPEC and Q

CPU schema = ROOT-N13-sameK-region-point-training-CPU-spec-v1.

CPU primary values: phase, sources, RAW_inputs, normal_RAW_count (GPU count+1, exact GPU prefix plus the just-finished GPU RAW), prior_failed_CPU_receipt, additional_failed_RAW_inputs, CPU_admission_file, Q_path, deadline_seconds (<=1800), recipe, seed, first_epoch, last_epoch, numeric_protocol=qualify_train_cpu.NUMERIC.

CPU primary refs: entry_source (CPU reader), training_entry_source, training_spec, training_report, training_GPU_RAW, training_GPU_admission, region_common_source, point_heads_source, cache_CPU_qualification. Complete GPU/cache/parent sources and all output/proof refs are in closure. The GPU source_gate is rechecked with owner_environment=False, preserving its original GPU admission identity without pretending this CPU process has GPU environment. The actual current CPU admission requires CUDA_VISIBLE_DEVICES='', threads2, CPU01 and no CUDA initialization.

CPU Q schema ROOT-N13-sameK-region-point-training-CPU-qualification-v1, status ROOT_N13_SAMEK_REGION_POINT_TRAINING_CPU_QUALIFIED. It contains source=training_entry_source, reader_source=CPU entry, spec=CPU SPEC, CPU_admission, phase, training_spec/report/GPU_RAW/GPU_admission, recipe, seed/range/final48, exact RAW/failure inputs, cache/factory/common refs, checkpoints, cold_state_proofs and final_endpoints.

True checks: cold_paired_weights_CPU_checked; resume_loaded_state_CPU_checked; private_image_order_and_endpoint_RNG_CPU_reconstructed; optimizer_endpoint_structure_and_stepcounts_CPU_checked; full_update_visit_sequence_CPU_checked; final_fullFIT_endpoint_BCE_CPU_recomputed; final_editable_BCE_CPU_recomputed. Resources: CPU_affinity=[0,1],threads=2,CUDA_initialized=False. False limits: intermediate_epoch_losses_independently_recomputed; per_step_gradient_or_AdamW_replayed; current_owned_CPU_terminal_self_certified. Fullvisit records and cumulative step/moment structures are checked, not every optimizer transition.

Final CPU losses use explicit PR arithmetic and stable BCE over all741/1661 qualified core packets. Full and editable means compare to GPU endpoint at rtol1e-4, atol1e-6; editable population counts compare exactly. Stable point BCE is compared to functional BCE with the same numeric tolerance. No data-dependent fallback or looser post-hoc tolerance exists. The37–48 window arithmetic uses bound intermediate GPU loss records, so its independent-convergence field remains False.

The CPU reader verifies all cumulative private randperm(741) orders and endpoint generator state from the seed; actual pre-update loaded weights/full optimizer/RNG/history equal cold initialization or the prior qualified checkpoint. Optimizer checks include eight finite moment shapes, nonnegative exp_avg_sq and cumulative1661*epoch steps. It does not execute optimizer.step or load the native model. ROOT must still obtain actual held exit0, reap and original birth absence for this CPU process and separately close its scientific qualification before another leaf resumes.
