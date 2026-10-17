# U40 固定 E50 最薄 GPU 预测核心

普通、未绑定、未执行数值。只有 `prediction_core.py` 函数和三项标准库元数据测试，没有 runner、binder、模型/真实权重读取、GT读取或 GPU 启动。不得评价当前中途 checkpoint。后续 Root 必须提供原 E50 实际接收、封闭来源、真实资源监督及新预测叶；本代码不能自行授予资格。

## 明确 API

```python
validate_final_endpoint(metadata) -> endpoint_identity

predict_shard(torch, np, c, *,
    endpoint_refs, dev_context_ref, original_image_refs,
    modules, source_refs, prediction_core_ref,
    shard_index, dataset_range, output_directory, limits, report) -> result
```

`metadata` 为 `{refs, training_qualified, pair_commit, preparation, order}`，`refs` 精确含同名四项 strict ref（path/bytes/sha256）。纯 `validate_final_endpoint` 不读文件、不认证其字节；真实 `predict_shard` 先通过 `c.closed` 与显式字节哈希读取这四份 JSON，再调用纯函数。要求 Root proof 顶层 qualified/paired_prefix_advanced=True、seed202610173101、最终 `[36864,37050)`、cursor37050/50/0，两臂 CP/pair/实际ORDER引用和原17训练source身份逐项相等。result 中历史 worker 的 pending False 不作为未接收。实际 ORDER 另调用原训练 core.validate_order；不生成新顺序。

`c` 必须提供 `closed(ref) / check() / write_json(path,value) / psutil`。`report` 是调用者保留的空 dict，过程中不断填写 partial；失败不重试。`output_directory` 必须是独占的新科学输出子目录，由本函数 mkdir。Root 外层负责 CPU01、threads2、GPU0、连续整卡≤8GiB/RSS≤16GiB/freeRAM≥16GiB以及1800秒硬结束与回收。核心只核固定设置、设4GiB allocator、调用check、实际采样/汇总资源和严格清理，不能替代外层硬监督。只支持原 CPython3.8.17/Torch2.0.1+cu118/NumPy1.23.5 FP32 合同。

## 来源与实际已核 ABI

`modules`/`source_refs` 精确17角色：`shared_initialization, full_legacy, full_dsconv, full_head, full_feature, plain_legacy, plain_feature, instance_adapter, u36_helpers, u39_training_ops, image_geometry, training_state, dev_reader, dev_rgb_reader, population_contract, encoder, mask_api`。每个注入模块 __file__ 与闭源路径及 SHA 相等。旧训练同名角色必须原 ref 复用，training_state 必须是原 core_source；原完整17训练引用另存 report.training_sources。预测 core 自身 ref 单独闭源。原encoder由已核DEV context.encoder_source精确固定；不是另写RLE算法。mask_api固定当前pycocotools.mask.py Python入口，底层扩展及其他第三方库完整准入仍由后续Root wrapper保持。

已逐一核普通源码真实接口：U34 `build_pair(torch,modules,refs)` 返回 `{PLAIN,FULL}` adapters 和实际104共享state/adapter回执；U39 `_partition(adapter)` 返回effective/inactive sets及参数表；U40训练core `_equal/expected_trace/_cleanup` 沿原数学和状态编码；U36 `_digest/_measure` 沿原张量摘要与采样；DEV桥 `prepare(torch,np,c,context,original_image_refs,dev_reader=...,image_geometry=...,source_refs=...)`；adapter `encode(rgb)`、`predict(encoded,e,M)`；geometry `render_delta(delta) -> [Q,1,H,W]`；原encoder `segmentation(np,mask_api,uint8mask)`；原population `replace_segmentations(...)`。未用假Torch镜像这些数值接口，也未运行它们。

## 恢复、预测与释放

先在 CUDA 冷态调用 U34 一次，构造相同 CPU pair。临时共同初始化只作骨架/谱系检查。每臂由 callee 从 pair 字典 `pop`，先按原 CPU load ABI读取该臂完整checkpoint，核完整训练identity、cursor、effective/inactive名称、模块树、实际/请求trace、原初始化回执完全相等，再strict load全model。所有浮点state需finite FP32；CPU恢复和转GPU后均逐张量相等，包括BN及inactive。不会在恢复后再次common copy或重置head。

原 checkpoint 反序列化不可避免带出 CPU Adam/RNG字典，但不构建、不恢复优化器，也不恢复训练RNG用于继续训练；核模型后立即释放该checkpoint。没有两个完整checkpoint同时驻留。模型eval、requires_gradFalse、no_grad。前后全model/buffer摘要相同、参数grad全None、旧四semantichead调用0；两臂各每图一次backbone/CONV0_4计数。所有callee别名/临时hooks释放后强制allocated/reserved双0，才进入下一臂；外层finally再核双0。失败保存error/partial报告并抛错，绝不写成功结果。

全人口由原reader核187图/18700原始候选。范围固定 `[0,47),[47,94),[94,141),[141,187)`，PLAIN后FULL，一臂GPU、一图encode、query固定4。每块使用实际export qids建立CPU int64 index，对自然轴 e/M/L统一index_select后上卡。F仍由原reader核验，绝不送入新主干；原class/score从未送入模型。全100不筛score、不截断、不NMS。

每臂每图进行完整显式zero400→原geometry校准，要求渲染零、native_L逐元素相同、100原RLE完全相同，再经原replace_segmentations确认整份native字典完全相同。零校准不修改训练head。真实预测只执行 `native_L + render_delta(trained_delta)[:,0] > 0`，仅替换segmentation。原S0、类别、query/image顺序和全部其他JSON字段保留。Omega为原native `abs(L)<=2`，只供CPU既定几何分账，不限制写入。两臂都完整校准；只保存一份identity和Omega（PLAIN遍历所得、与模型无关）。

未向模型提供新DEV COCO标注或历史目标关联字段。原冻结bundle/packet整体load可能带历史GT派生关联信息，来源完整保留，不能宣称整个Source不含GT。尚未准入的187 JPEG引用由Root未来提供并由DEV桥逐图精确join；本任务不补采teacher/数据或资格叶。

## report/result 与 CPU 交接

report schema=`U40-fixed50-full-method-DEV-shard-pending-v1`，成功status=COMPLETED，但qualified=False。输出独占 `PLAIN_PREDICTIONS.json / FULL_PREDICTIONS.json / IDENTITY_PREDICTIONS.json / OMEGA_SUPPORT.json / DEV_REPORT.json`。所有预测是原导出顺序的字典列表，无新标签。

返回result包含：`report_ref, predictions{PLAIN,FULL}, identity_predictions, Omega_support, dataset_range, shard_index, images, queries, seed, fixed_epoch=50, updates_per_arm=37050, training_qualified, pair_commit_ref, checkpoint_refs, order_ref, dev_context, training_sources, prediction_sources, new_training=False, AP=False, DEV=True, TEST=False, qualified=False`。Root外层未来实际接收后，拟 `ROOT-U40-fixed50-full-method-DEV-shard-qualified-v1` 原样携带该result；CPU不得只凭worker pending报告开始评价。四shard齐全后才merge18700，不平均shard AP。

report校准字段：`native_all_RLE_and_identity_prediction_dictionaries_exact=True`, `zero_residual_native_L_exact=True`, `original_full_S0_class_order_preserved=True`, `original_teacher_forward=False`, `GT_used_for_prediction=False`, `DEV_COCO_annotation_opened=False`, `historical_target_fields_may_be_in_bundle=True`, `model_BN_RNG_unchanged=True`。report额外prediction_core_source、构造receipt、每臂restore/model前后SHA/调用数/cleanup/proofs和cost。`cost={PLAIN:...,FULL:...}`与arms内cost同值，分开encode/readout/render/RLE/zero校准/CPU缓存JPEG和恢复时间。采样RSS不是OS精确峰值。包含新UNet骨架成本、不含teacher预计算，不宣称完整pipeline FPS。强R及官方COCO计算留独立CPU实现；不携带旧E24 gate或效用结论。

## 已做与尚未做

三项标准库测试先MissingModule/AttributeError RED后GREEN：端点与源身份拒错、自然query到导出顺序的元数据映射、完整人口及四分片。AST检查、测试不导入Torch/NumPy/PIL，不使用假Torch。没有任何真实恢复、RGB读取、GPU前向、zero校准、峰值或AP验证；不能把metadata PASS写成数值或资源PASS。完成本草稿即STOP，后续仅Root可绑定与执行固定端点。
