# N20 训练 tail 的最小 front 接线清单（未执行）

仅核对普通 tail `N20_PAIRED_TRAIN_GPU_RUN_TAIL_UNBOUND_DRAFT_20261015_01.py`，SHA256 `af590e05f27303a164a0168fbc0e182ee5d45b707cc3f28211a5cbc3ca1b05e6`。不读取实际资格链，不构成训练准入。

1. 当前 GPU 分支建议字面 `ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1`，只替换当前 own-schema 分支。历史 `SCHEMA='ROOT-N17-fixedDEV-local-center-error-GPU-shard-spec-v1'` 必须保留供失败18–20校验；typed21/先 RAW 后首 SHA 的顺序不改。仅缓存 CPU 原 actual0 且 Root 闭合之后，才可用本首轮 `normal=expected_normal=140 / failed21 / pins161`，不能预先声明完成。own entry 为最终拼接文件 `__file__`，新 GPU admission/CUDA0，禁止复用旧 admission。

2. 新增并严格闭合 `N20_root_cache_qualified_result`、`expected_N20_root_cache_qualified_result`：实际 JSON 与 expected 整体相等，再核 schema `Root-N20-fullFIT-true-feature-neighbor-cache-post-actual-qualified-v1`、status `FULLFIT_TRUE_FEATURE_NEIGHBOR_CACHE_QUALIFIED`、`qualified=True`、`original_GPU_CPU_actual0_verified=True` 及 741/3035136/1786820。逐个 `gpu_record`/`CPU_record` 以其显式 owner_index、原 spec/prefix/source/admission/phase/bootstrap/RAW 做实际 owner join；GPU CUDA0、CPU CUDAempty，CPU 输出键为 `Q`。GPU record 的输出键应按 Root 精确 envelope 原字段读取，不能套用 CPU 的 `Q` 或臆造键。Root 需把两记录全部物理 refs、GPU report、CPU Q/图证明及741新旧 packets 纳入 Sources。

3. 新增 strictref `N20_initialization_lock` 和 `expected_N20_initialization_digest`；首 seed 为 `initialization_seed=202610150201`。Root 消息给定新共同 digest `a13802fc0bf4f23a8ab368b11b15a24b60552baadf1ab8264daf242173854a68`，tail 自身不硬编码它。front 将 lock 的实际 schema/seed/source 字段与其精确元数据核对后再放行；未获普通字段合同，不能假定旧 N17 metadata schema/字段。tail 会独立按 sorted name/shape/dtype header + FP32 raw bytes 算五张量 whole digest，并核两个臂相等、storage 独立、初始 spatial_weight 为零。

4. 以下四个 spec strictrefs 均在 import 前 closure 检查：`scout_module_source`=N20 create_head 模块；`base_scout_source`=旧 N16 make_scout 模块；`paired_training_body_source`=N20 五张量 body；`original_paired_training_body_source`=旧 N16 body。保留旧 `locked_protocol`，新增设计锁仍为 `N20_locked_protocol`。N20 body 的 base_factory 实参必须传模块对象，不能传裸 make_scout 函数。无 teacher/PR/full-native-L 加载。

5. `source_gate(path,spec)` 的实际最小返回键：`closure, source, spec_ref, admission_ref, admission, common, lineage, parent, parent_cache_report, parent_cache_report_ref, neighbor_report, neighbor_report_ref, training_binding_refs`。`parent_cache_report` 是已解析原 N16 cache report，ref 可沿用当前 `spec.cache_GPU_report`；`neighbor_report` 是已解析新 N20 capture GPU report，ref 沿用已资格 CPU gate 的同一真实 report ref。两者均需含 `images/sampled_points/rows/packet_refs`；旧 report 另需 `teacher_checkpoint/fixed_PR_checkpoint`。tail 会将 decoded report 与其 closed JSON 再核相等。`protocol` 可由 front 保留用于锁校验，但 tail 实际不读取 `gate['protocol']`，不是必要返回键。

6. **parent 实际被使用一次**：tail L319 调 `gate['lineage'].final_endpoint_check(gate['parent'])`。因此必须原样保留 inherited gate 经 `lineage.endpoint_gate(context,closure,owners)` 返回的 `parent`，不可替换为 cache report、N20 Root envelope 或 GPU/CPU record。若当前 front 未传该对象，这是明确缺项；无需重新加载模型。`common` 只用于两次 `imported_files(torch,np,cv2,masks,COCO,closure,admission,ndimage)`；应在 membership/SHA 门后 `load(spec.region_common_source,...)`，不能把 native common、训练 body 或 lineage 模块冒充该 ABI。

7. `training_binding_refs` 是非空 `name -> strictref` 映射，建议至少包含 `N20_root_cache_qualified_result` 和 `N20_initialization_lock`；均须 closed。tail 原样写入 START/checkpoint metadata/REPORT，禁止覆盖自身 source/spec/admission/phase、模型 refs、两个 cache report refs 或初始化/冻结 checkpoint 键。缓存 CPU Q 等独立记录可另加入该映射，名称由 Root 确认，无需改 tail。

8. 资源/输出 spec 必填 `output_directory, phase, entry_source, GPU_admission_file, sources` 及动态三计数；`deadline_seconds=1800, automatic_retry=False, allocator_limit_MiB=4096, whole_GPU_limit_MiB=8192, maximum_process_RSS_bytes=17179869184`。环境 CUDA0/CPU01/四线程环境变量均2；tail 设置 Torch inter/intra2、冷 CUDA 后限额再建模型，检查可用 RAM16GiB/RSS16GiB，末尾全 Sources/spec/admission/RNG/library 不变和 CUDA 清理0。只产 E8/E24、每臂17784更新/72843264曝光，REPORT 仍 pending independent CPU，`qualified=False`。

STOP EDIT：仅接线备忘；当前未启动训练、未读取私有数据、未测试或改冻结源码。
