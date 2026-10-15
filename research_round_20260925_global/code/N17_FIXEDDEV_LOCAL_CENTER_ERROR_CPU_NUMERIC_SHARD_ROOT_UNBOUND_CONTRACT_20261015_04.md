# N17 CPU04：UNBOUND 剩余scope七段适配

由普通CPU03 78439B/SHA3c16d8385c0b164ae7091382241fe6fd019550b3211a8c891f176b49a5b606d7独立复制；旧冻结源/前三报告不改、不重跑。未读取科学链/数据/权重、未科学执行或Git。仅分段执行，总walltime可增加，科学方法/训练/容差/排名/指标/停止门不变，不实现cache优化。

当前CPU_SCHEMA为 `ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v4`；report schema/status沿原report-v1。新入口只允许全局cpu_shard_id3..6、ownnormal=expected=cpu_shard_owner_index≥130且大于映射GPUowner，failed21/pinsnormal+21；typed21及1..20前门原AST保留。GPU `SHARD_RANGES` 保留五段，legacy仍旧四段；新增CPU_RANGES=[[0,47],[47,70],[70,94],[94,117],[117,141],[141,164],[164,187]]，CPU_PARENT_GPU_IDS=[0,1,2,3,3,4,4]。spec必填INT `CPU_parent_GPU_id`，新四段对应3/3/4/4；GPU_shard仍原完整父十键descriptor及原父range，不伪造子GPU记录。

`parent_GPU_gate(spec,gate,gpu)` 除父id从显式字段取，其完整父来源/REPORT/manifest/10predictions/全100/19或20计数等门不变，返回原全父keys：parent_GPU,parent_GPU_report,parent_GPU_manifest,parent_GPU_START,parent_GPU_binding,expected_predictions,expected_prediction_groups。随后 `scope_parent_GPU(spec,gate) -> gate` 只新增五keys：

- scope_rows：父完整manifest中本CPU半开range的原row引用；严格完整父覆盖后取contiguous slice，逐图保100和原qids。
- scope_expected_predictions：policy→本scope完整顺序100×images预测列表。
- scope_expected_prediction_groups：policy→本scope image_id→原组。
- scope_capture_refs：本scope原真实bundle refs，原binding不改。
- scope_image_ids：本scope原顺序image IDs。

不替换/改写原parent manifest/report/context/expected_*，不猴补frozen模块。run仅iter scope_rows，内部逐图/逐query/四模型/八profile/两PR公式/FP64排序/mask/RLE语句AST与03完全相同；最终对scope refs/preds精确核验。

新 `CPU_execution_partition_supplement` strict3ref必须SOURCE闭合，`expected_CPU_execution_partition_supplement`仅spec完整预锁JSON，实际JSON逐型/逐值canonical全等；额外ROOTrefs/detail允许但也须expected一致。schema `N17-fixedDEV-remaining-CPU-seven-partition-execution-supplement-v1`，必需键逐Root合同：CPU_ranges、CPU_parent_GPU_ids、reuse_completed_CPU_ids=[0,1,2]、new_CPU_ids=[3,4,5,6]、completed_CPU_owners=[127,128,129]、completed_CPU_descriptors、global_images187/global_GT380/global_queries18700/policies10、all_scientific_rules_unchangedTrue/reuse_failed_partialsFalse/one_scientific_leafTrue/deadline_seconds1800/automatic_retryFalse/total_evaluation_wall_time_may_increaseTrue/CPU_cache_optimization_implementedFalse。

`validate_CPU_execution_partition_supplement(spec,gate)`返回parsedJSON，source_gate存入gate['CPU_execution_partition_supplement']。它不检查当前own schema或当前cpu_shard_id，可供final/post元数据门复用；只校锁、refs、前三已完成CPU03 descriptor及CPUempty实际owner join/原spec与pending报告完整范围。completed_CPU_descriptors严格十键为 `cpu_shard_id,cpu_dataset_ordinal_range,source,spec,report,RAW,admission,phase_binding,bootstrap,cpu_shard_owner_index`，0..2/127..129，旧schema独立字面numeric-v3，failed21。旧PRIOR_CPU_SCHEMA numeric-v2只用于失败21，不改。前三数值不重执行。

新REPORT保原capture_manifest全父ref、GPU_shard全父descriptor；capture_refs/image_proofs/control_results[name].predictions及scope_dataset_indices/scope_image_ids/images/queries只本CPU范围。新增top CPU_parent_GPU_id与CPU_execution_partition_supplement strictref（不写expectedJSON）；GPU原binding不加CPU字段，decoder仍用完整父binding。最终7段合并/统一COCO/AP/实际资格由ROOT及后续final处理，当前输出仍pending/qualifiedFalse。

单项stdlib fake元数据RED→GREEN：四新范围全部精确cover94:187、不重叠，父映射错/父row缺失/前三新运行拒绝；旧gate内容不变，原数值循环body/局部formula/check/encode AST完全同03。只有validate_cpu_header、validate_parent_record、source_gate、run原函数AST变动，新增scope与CPU补充校验两函数；inherited_source_gate/typed21全部AST同03。最终1 test passed/exit0（0.225秒），未跑任何真实或合成数值forward。STOP EDIT，Root绑定资源/SOURCE/实际refs及一轮定向核验后自行执行。
