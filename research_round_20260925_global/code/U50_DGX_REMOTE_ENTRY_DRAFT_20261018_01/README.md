# U50 DGX Linux worker entry — SOURCE ONLY / NOT EXECUTED

<!-- U50_ORIGINAL_DGX_ACTUAL_ACCEPTED_20261018_01 -->

**后续真实执行状态：** 原模型、入口和Linux监督器已在DGX一次实际任务中完成两臂各3次真实更新，原任务退出0并经Root接收；2图/28不同匹配实例，6次可丢弃更新，约45秒。无AP，主训练仍2816/37050；不是原Windows检查点跨环境续训，也不证明全量、加速或效用。见[实际核验及下一步](../../U50_DGX_PAIRED_PILOT_ACTUAL_AND_FULL_INPUT_NEXT_20261018.md)。下文SOURCE ONLY是开发形成时状态；冻结执行原件未修改。


`entry.py --spec ABS --spec-sha SHA256` is a new, independent Linux/aarch64 worker. It calls the unchanged `pilot_body.py` (20338 bytes, SHA256 `003641b7c81e198bb70502073aba22f623f9dac82554ca9ce90ba059ac20625b`). Authoring ran only two tiny standard-library metadata tests and AST checks: no Torch/NumPy import, SSH, real input access, model execution or remote trial. This directory does not modify U40 or earlier U49/U50 sources.

The original Linux parent resource owner is a separately admitted `linux_owner.py`, providing `make_control(spec, psutil)`: `c.closed(ref)`, `c.check()`, `c.psutil`, and `c.summary()`. The worker does not replace it with Windows held handles, WAIT_OBJECT_0, FIRST prefixes, Q files, or a self-issued Root qualification. Parent-owned actual process/session/start identity, deadline termination, process-group cleanup and Root review remain mandatory outside this worker.

Spec schema is `ROOT-U50-DGX-paired-pilot-spec-v1`, operation `GPU_U50_DGX_PAIRED_DISPOSABLE_PILOT`, with these thirteen required keys:

- `schema`, `operation`, `phase`;
- `entry_source`, `pilot_source`, `owner_source` — actual DGX-local strict refs;
- `source_refs` — exactly the eleven original component roles below;
- `library_sources` — actual canonical `psutil`, `torch`, `numpy` module-file refs, selected files only;
- `portable_refs` — string keys `54`, `710`, each the actual transferred U49 CPU package ref;
- `prelock` — exactly the original source-closed `pilot_contract()` dictionary;
- `output_directory`, `owner_output_directory` — distinct absolute POSIX directories, neither inside the other;
- `sources` — the full nineteen refs above, sorted by case-sensitive POSIX path, without duplicates or undeclared payloads.

There are no optional spec fields. Root retains original export/transfer evidence outside the worker spec. The worker does not open historical Windows references inside packages, reinterpret them as Linux files, or independently certify the old export. Root must authenticate the accepted U49 export and byte-identical transfer. The existing transfer manifest's `pilot_source_ref` maps directly to spec `pilot_source`; its `source_refs` and `portable_refs` maps are reused with actual transferred paths. No new teacher, JPEG, annotation file, F carrier, checkpoint or full-population cache is required.

The eleven actual source aliases load in this order, derived from original published U39 wrapper and U34 dependency checks:

| Role | Module alias |
| --- | --- |
| full_dsconv | dsconv |
| full_head | head |
| full_legacy | model_full |
| full_feature | feature_backbone |
| plain_legacy | plain_model1 |
| plain_feature | plain_feature |
| instance_adapter | full_instance_adapter |
| image_geometry | image_geometry |
| shared_initialization | shared_initialization |
| u36_helpers | _U50_U36_HELPERS |
| u39_training_ops | _U50_U39_TRAINING_OPS |

All eleven hashes retain original U50 body pins. In particular, DSConv remains original b4b612…; this is not a U46 replacement. `full_legacy.DSConv/CoordConv` must be the exact admitted dependency objects. The source-closed original U34 constructs the shared104/adapter CPU initialization, and original U39 `_run_arm` owns all training operations. No alternate model, simplified loss, F placeholder, second query reorder or replica of the training loop is introduced. Every temporary source alias, including the pilot and owner aliases, is restored in `finally`.

Before numerical imports, the worker verifies exact CLI spec SHA and all nineteen admitted files, including its own entry and both portable files. Stable-file checks exclude access time, which may legitimately change on reads. It then verifies actual imported library paths/bytes and the original prelocked environment: Python3.12.14, Torch2.10.0+cu130, NumPy1.26.4, Linux/aarch64. The owner must launch with CPU affinity0/1, `CUDA_VISIBLE_DEVICES=0`, and OMP/MKL/OPENBLAS thread variables2. Torch threads are set to2; CUDA remains cold through component loading and CPU initialization. TF32 is false for matmul/cuDNN, benchmark false, deterministic true, before first CUDA. FP32/noAMP/noCompile/noVJP/no recompute, original400 geometry, query_chunk1 and both fresh three-step AdamW pilots are unchanged. Actual 4GiB allocator setup and per-arm GPU release stay in the original body.

The child checks the original owner control during loading/training and retains `c.summary()` even after a refusal. The Linux parent must independently enforce the whole1800-second deadline including imports, group RSS≤16GiB and availableRAM≥16GiB. GB10 UMA whole-GPU observation remains `None`; the old WDDM whole8GiB contract is explicitly not certified. This is not a claim that the new Torch save_on_cpu hooks, pinned transport, private CUDA cleanup, AdamW state or NumPy interoperability have passed numerically; those interfaces must pass in the actual single owned trial or fail without a fallback/retry.

The worker exclusively creates `output_directory/REPORT.json`, schema `U50-DGX-paired-pilot-worker-report-v1`, outer status `COMPLETE` or `FAILED`. Original complete/partial detail is under `body_report`, whose successful status is `COMPLETED_PENDING_ROOT_REVIEW`. Top-level `successful_optimizer_updates`, `optimizer_steps_entered`, and `new_training` are always copied from actual body observations, including failures. Successful `new_training=True` means six real disposable optimizer updates; main training remains false and U40 budget credit zero. COMPLETE requires exactly ten named boolean flags all true, both arms passing `[54,710,54]`, G=`[1,27,1]`, three returned updates each, total6, 29GT visits per arm/58 total, caller CPU and post-initialization CUDA RNG preservation, and each arm/final CUDA allocated/reserved0. Qualified/main/AP/DEV/TEST remain false. Any partial body, source drift, resource or cleanup failure produces a FAILED report and nonzero child exit. No checkpoints or weights are written.

The report also retains exact sources, selected package audits, actual flags/backend restoration, versions, child sampled resources and costs. It is private and may contain original provenance/GT identities from the two accepted packages; do not publish it verbatim. Parent stdout/stderr and owner receipt are separate. Root may later publish a scalar-only reviewed summary.

Missing-entry RED `e066bc` preceded initial GREEN `51f01f`. An additional false-training completion refusal was RED `c1da4b` then GREEN `e60a84`; still only two small tests (0.004s), no tensor fixture or scientific import. These checks establish metadata rejection and loader ordering, not actual DGX compatibility, resource feasibility, AP, speedup, innovation, cross-environment bit equivalence or U40 continuation.
