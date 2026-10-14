# UNBOUND_DRAFT — N14 stage24 full historical DEV GPU runner

未绑定、未执行。只开发普通源码，不读取实际 SCI、RAW/SPEC/Q、数据、缓存、权重或运行输出。ROOT 负责真实字段审查、完整 Source、预算、绑定和单次实际执行。搭配稳定 N14_FIXED_DEV_EVALUATION_ADAPTATION_UNBOUND_DRAFT_20261014_02.py。

GPU schema：ROOT-N14-stage24-fixed-fullDEV-GPU-spec-v1；CPU replay schema：ROOT-N14-stage24-fixed-fullDEV-CPU-spec-v1。共享 source_gate(path,spec,cpu=False) 返回 source/spec_ref/admission_ref/admission/closure/owners，及 context/evaluation_spec/lineage/common/parent/cache_report/prior/prior_q/packets/selector/adapter/stage24_gate/stage24_GPU_report/stage24_GPU_spec/stage24_CPU_Q/stage24_CPU_details/stage_origin。cpu=True 将 own_gate entry_file设为自身spec.entry_source.path、CPU_admission_file与cuda empty；CPU入口另外核自己的__file__，并在回放前闭合 completed GPU DEV 的真正终止、来源、报告和捕获链。

完整新SPEC至少包含：

- sources、RAW_inputs、normal_RAW_count、expected_normal_RAW_count。GPU此次最新normal101，随后CPU102。原failed8与terminal_identity_pins保留；additional_preserved_independent_CPU_failure单列第9失败；total_failed_RAW_count=9、total_terminal_identity_pin_count=normal+9。先第9 RAW-only，再全部normal+failed8 RAW/pin，之后才首SHA。
- entry_source、phase、GPU_admission_file（CPU模式CPU_admission_file）、output_directory、deadline_seconds=1800、automatic_retry=false、allocator_limit_MiB=4096、whole_GPU_limit_MiB=8192。actual ref dict必须exact；单独locator用解析后的真实绝对路径，不要求slash拼写相同。
- carrier_spec、carrier_evaluation_source、evaluation_adapter_source=v02、memory_source、seed=202610120701、memory_seed=原module_seed。carrier是原qualified E24 fullDEV evaluationSPEC；其 export/renderer/reader/native/DEV/selector/endpoint/cache refs完整继承和Source闭合。旧endpoint_gate仅资格原PR/cache/native provenance。
- stage24_GPU_source/spec/report/admission/RAW，stage24_CPU_source/spec/admission/RAW/qualification，stage24_CPU_endpoint_numeric_protocol，stage24_checkpoints两arm refs。GPU24 report.checkpoints == CPU24 spec.stage24_checkpoints == CPU24 Q.checkpoints == 新spec.stage24_checkpoints。
- original_parent_epochs=24、fixed_stage_epochs=stage_epochs_completed=24、selector_mode=reuse、point_chunk=2048、controls=[native_official,BASE_CONTINUATION,MEMORY_ADDON]、images187、raw_GT380、queries_per_image100、predictions_per_control18700、support_formula='abs(actual_original_GPU_native_L)<=2'、fixed_native_scores_class_order=true、best_DEV_selection=false、no_TEST=true。

canonical_stage24_gate(torch,gate,spec)具体核验实际CPU24证据及当前checkpoint/order/history/三leaf trace，随后填新adapter合同：8/14 tensors、40497/42698参数、24私有741顺序和最终状态、累计39864paired trace、1661/epoch在线均值、AdamW exact39864steps，保留stage_origin_START。它不重放训练或优化器轨迹。adapter.load_stage24_heads用原base factory与memory factory构建独立两臂、严格加载受训状态；不能将MEM套旧LOG/GEO8tensor资格。训练CPU24 endpoint numeric JSON另绑定当前GPU24报告，after-GPU-before-CPU锁定、非AP scope，不称GPU训练前预注册。

ROOT须锁定capture_budget的14整数：total_original_pixels、total_LOG_points、union_points_upper_bound、raw_tensor_payload_upper_bound_bytes、maximum_image_pixels、maximum_image_LOG_points、maximum_image_union_points_upper_bound、maximum_image_raw_tensor_payload_bytes、serialized_capture_overhead_budget_bytes、auxiliary_output_budget_bytes、minimum_free_output_bytes、required_available_RAM_bytes、capture_readback_peak_RAM_upper_bound_bytes、maximum_process_RSS_bytes。N/T来自原DEV metadata及row.band_counts_all100；本稿保守U_upper=N。payload上界1032ΣU+28ΣT+187×102400，每图上界1032N_i+28T_i+102400。保存源包与完整exactreadback至少2×最大图payload，另锁总process RSS与序列化/辅助输出磁盘预算；至少2GiB磁盘余量、16GiB可用RAM。这不等于总进程峰值证明。

CaptureSink.write直接保存v02已CPU detached packet，不clone第三份。每图仅1个IMAGE_CAPTURE.pth真正ref，shared含sampled_F/e0/union_flat_indices/metadata，queries[100]显式flat_indices/native_L/两arm GPU_residual/GPU_refined_logits，加GPU_predictions_by_arm。没有虚拟member refs或另存完整100L；原完整selector来自row.carrier .pth，其容器dict.native_L。

CAPTURE_MANIFEST schema=N14-stage24-fixed-DEV-capture-manifest-v2。rows[187]包含dataset_index/image_id/capture（实际ref）/U/T/raw_tensor_payload_bytes/packet/selector/natural_queries100/K_natural_query_order。manifest绑定本次source/spec/GPU_admission/phase和stage24_GPU_report/stage24_CPU_qualification。REPORT.capture_manifest为真正JSON ref，capture_refs为187真正包refs。所有回放输入为同一GPU point features、L、e和索引，无CPU特征重渲染。

REPORT schema=N14-stage24-fixed-fullDEV-GPU-report-v1，status=N14_STAGE24_FIXED_DEV_GPU_COMPLETE_PENDING_INDEPENDENT_CPU，evidence_status=GPU_AP_PENDING_INDEPENDENT_CPU_REPLAY。包含source/spec/GPU_admission/phase、完整RAW/pins、stage24 refs/checkpoints、原native/packet/selector refs、parameters/head proofs、三control predictions refs与control_results官方COCO12、START/TRACE/IMAGE_DIAGNOSTICS refs、capture manifest/refs/budget、actual imported files、耗时/allocator峰值/cleanup。完整187图380GT与all100原score/class/order/S0/旧LOG K；无挑seed、bestDEV、TEST、NMS、scorefilter、native大网或训练optimizer。不能自认证自身terminal。

新CPU DEV spec约定 completed_GPU_DEV_source/spec/report/admission/RAW，GPU_runner_source==completedGPUsource，以及CPU_DEV_numeric_protocol。后者JSON与训练均值协议独立，其8keys由v02严格核：schema='N14-fixed-DEV-same-GPU-point-input-CPU-numeric-protocol-v1'、same_actual_GPU_point_inputs=true、point_chunk=2048、mask_RLE_exact=true、locked_before_CPU_execution=true、bound_completed_GPU_evaluation_report=本次完整GPUDEV report ref、atol=rtol=1e-4。

CPU逐两arm2048chunks独立公式，sameGPUinput residual/logit allclose + 完整mask/RLE exact。门不符只记FAILED_CPU_REPLAY，ROOT逐字检查实际GPU captured logits与GPU预测，不后验放宽容差；CPU computational portability与GPU self-consistency/AP有效性分开判断。GPU报告保持pending independentCPU；不宣称AP收益、机制归因、外部SOTA、容量控制、方法总体失效或创新合格。开发期间未执行任何科学载荷；实际外部字段与绑定由ROOT审阅。
