# U65 full187 DEV portable-reader CPU validation body

**D盘独立新稿 / UNBOUND / UNEXECUTED。** 原E盘不完整草稿不改、不删、不用于绑定。本目录的 body/tests/本说明是普通研发源码；未读取真实 manifest、plan、包、原数据、Ops、资格链或权重，没有 SSH、Torch/NumPy 导入或科学执行。有限测试采用可弱引用的空token、合成JSON和stub；它们不是数值张量、真实包或真实187图验证。

该 body 最小适配旧 U54 CPU验证循环，接入未修改的 U59 `package_io`。只验证实际187包内的四个CPU字段及原metadata，不构造模型/教师/优化器，不做forward、匹配、GT栅格化、mask decode、RLE/零残差校准、AP或TEST，不增加训练额度。

## 精确调用和原reader ABI

`run(torch,c,manifest,*,reader_module,image_geometry,source_refs,prelock,report,record_image)`。

caller保留同一个空report，并提供真实Torch、已闭合来源的U59 reader_module、原image_geometry、Control.check及JSONL记录callback。entry/owner负责实际source-byte闭合、平台/库/CPU资源、调用者RNG/dtype/线程等外状态、真实运行及退出；body不新增owner、binder或资格控制。

manifest使用真实U64 `manifest['entries']`187项。body调用
`reader_module.prepare(torch,c,manifest['entries'],image_geometry=...,source_refs=...)`；source_refs只向U59转交 `dev_reader/dev_rgb_reader/image_geometry` 三键，caller其他来源角色不传入。U59负责完整索引/唯一身份/自然和导出query轴、四字段CPU FP32 proof、原native字典及geometry构造验证。

每次 `reader.load(index)` 是真实三元组 `(item,row,originals)`，不是旧matched reader的单字典/last_audit接口：

- item恰 `rgb400/e/M/L/identity/query_ids/geometry`；没有F、targets、G或GT字段。
- row恰 `dataset_index/image_id/qids`，与reader.records及原entry身份/置换连接。
- originals为原100条native字典，保持原序；body仅连接image/query和计算metadata摘要，不重排或改写score/mask。

e/M/L保持自然100query轴；query_ids是独立原导出置换。四字段实际shape/dtype/finite/字节proof及原identity/native RLE结构由未修改U59内置验证，body不另造替代field proof。

body自己维护访问/失败计数；U59没有 `visits/failed_loads/last_audit`，不假定这些旧属性存在。每图清除返回tuple及所有item/row/originals别名，del语义/GC与四个weakref检查通过后才记录并进入下一张；只保留小JSON摘要，不缓存整个人口tensor。

## Prelock

`prelock_contract()`精确生成schema `U65-DEV187-CPU-reader-validation-prelock-v1`：

- environment：Python3.12.14、Torch2.10.0+cu130、NumPy1.26.4、Linux/aarch64。
- dataset_indices0..186，images187，queries18700，CUDA_visible_devices空。
- limits：CPU0/1、threads2、RSS16GiB、可用RAM下限16GiB、deadline1800秒。
- no_new_training/no_forward/no_matching/no_AP/no_DEV/no_TEST/no_GT_rasterization均True，DEV_technical_input=True。

`no_DEV/DEV=False`表示未进行DEV性能评价；**DEV_technical_input=True**明确任务确实属于DEV技术输入读取。它不是无DEV数据接触声明，也不代表某次失败已经读完包。实际读取程度由尝试/验证/失败计数与原实际终态决定。

## Report和callback ABI

body report schema `U65-DEV187-CPU-reader-validation-body-report-v1`，状态STARTED/FAILED/COMPLETE_PENDING_ROOT。`completed_population`要求attempted_images、validated_images、released_images、recorded_images各为187，queries18700，reader_visits为原序0..186，reader_failed_loads0，remaining_returned_tensor_objects0，formal_training_budget_credit0。

五个FLAGS必须恰为全部True：CUDA_uninitialized_before、CUDA_uninitialized_after、metadata_population_verified、all_field_proofs_verified、all_returned_tensor_objects_released。all_field_proofs_verified依赖真实源码闭合的U59全部load返回；它不是stub测试已经证明数值的声明。

qualified/new_training/AP/DEV/TEST/main_training/model_constructed/model_forward_executed/optimizer_created/GT_matching_recomputed/GT_rasterization_executed必须False，DEV_technical_input必须True。另强核
`original_metadata_population={images:187,queries:18700,tensor_population_revalidated:False}`、original_annotation_records_parsed=False、raw_provenance_inputs_revalidated=False。原metadata范围描述不替代187次真实字段核验计数。

成功图传给record_image的JSON恰10键：
`dataset_index/image_id/queries/status/source_package/original_record_sha256/native_predictions_metadata_sha256/query_ids_metadata_sha256/CUDA_initialized/returned_tensor_objects_released`。
其中queries=100、status=VALIDATED、source_package为该真实entry的package_ref；后三项状态/释放只在对应检查实际通过后形成。callback拿到的没有tensor、geometry对象或完整native RLE。metadata摘要采用默认ensure_ascii=True、排序紧凑JSON及UTF-8 SHA256，与U59算法一致。

失败保留调用者report和已返回计数；一次reader.load异常只增加对应失败计数，不跳图/重试或授予全人口完成。finally释放当前返回值、记录残留weakref数与body耗时；entry仍须在返回后核验完整report、外状态与原任务实际退出。COMPLETE_PENDING_ROOT不会自授原始退出0或Root接收。

## 历史来源边界

DGX会实际读取并验证portable包与本地源码。包内Windows JPEG/bundle/packet/教师等provenance路径只是历史metadata；U59不重新打开或核验这些raw input字节，不能把本次CPU验证表述为DGX重新审计了原raw数据。原上游bundle流程可能接触GT关联容器的历史说明仍须保留；本body不新读annotation或栅格化GT。

`python -B test_body_metadata.py`仅运行少量新ABI delta测试：187个空token load/释放/记录、精确三角色prepare参数、GT字段/置换拒绝、部分load失败及全人口guard。AST仅检查普通源码/导入；未重复旧整套数值或元数据测试。真实187包、资源、环境、field proof、外状态和终态仍全部 **UNEXECUTED**。
