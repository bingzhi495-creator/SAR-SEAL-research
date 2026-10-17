# U40 固定 E50 完整 DEV 评价：未绑定、未执行的最小设计

范围：本次仅静态阅读普通公开源码，不读当前训练的 Source/RAW/Q/Ops/H/SCI、模型权重、图像、GT或预测。没有模型恢复、真实 RGB 读取、零残差校准或 AP 执行；不增加资格叶、不补 teacher、不提前评价当前正式段。未来只评价首种子 `202610173101` 的预定 E50 端点。

## 1. 唯一端点与恢复

入口须先核 Root 实际接收的 `ROOT-U40-continuous-training-segment-qualified-v1`，要求 qualified、paired_prefix_advanced 均 true；result 的 `paired_checkpoint_prefix` 为 `{global_updates:37050, completed_epochs:50, within_epoch:0}`，两臂 checkpoint refs、PAIR_COMMIT、原 ORDER ref 与 Root共同前缀逐项相等。最终 span 应为 `[36864,37050)`；全程共同 order 摘要为 `2afadef62eac9f797c125fa105a3a28f94cdbdd9db4b5b5a2b018d4f0d5f194e`。文件 SHA 与 order 内部摘要是两个不同字段，不能混用。任何中途点或仅 worker pending pair 都拒绝。

每个预测叶在 CUDA 冷态调用固定 U34 `build_pair` 一次，构造同一 CPU 骨架；随后逐臂、逐个 checkpoint 用原 Torch2.0.1 CPU load ABI 恢复。核 arm、cursor、完整 model 参数/BN/inactive state、来源/recipe、effective/inactive 名称及初始化谱系；strict load 后逐张量相等，再记录 model-state 摘要。临时共同初始化不能在恢复后覆盖训练权重或重置输出 head。

评价不构造 Adam、不恢复训练 RNG来继续抽样、不执行 backward/step。现有完整训练文件反序列化时可能连带读出 Adam CPU 字典，但不载入优化器，恢复模型后立即释放其余 checkpoint 内容；不同时持有两臂完整 checkpoint 快照。`eval()`、`requires_grad_(False)`、`no_grad()`；前后模型/BN 摘要相同、梯度 None，保留原 inactive 架构分配和 full 显式关闭 Transformer 的资源版本。

## 2. 原 DEV 与 RGB 接口

普通公开目录 `code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/dev_reader.py` 可保持原样：完整187图、18,700候选；原 packet F/e/M/class 字段及 GPU-native L 身份、digest和原 capture association均照旧核验。其返回 `e[100,256] / M[100,200,200] / L[100,H,W]` 是**自然 query 轴**，`query_ids` 是原导出置换。不能用导出 ordinal 直接切自然 tensor。

RGB桥采用 n23 已交普通设计：

```
prepare(torch, np, c, context, original_image_refs,
        *, dev_reader, image_geometry, source_refs) -> reader
reader.load(index) -> 原 item 对象值 + rgb400 / geometry / rgb_receipt
```

原 DEV item 的 F/e/M/L/identity/query_ids 原样保留。Root 未来另准入187个有序原 JPEG strict refs，加载后与真实 packet `identity.image` 逐项 join；不猜目录、不用 GT补身份、不构造“G=100”伪GT字段。后续已在原N07报告与四份N16 JSON采集清单中实际提取有序187个原JPEG候选引用，逐项核dataset/image/shape/bundle/query/native_packet/capture_binding关系，并调用普通RGB桥真实metadata函数检查通过；[准备摘要](results/U40_FIXED50_DEV_METADATA_PREPARATION_ACTUAL_20261017.json)。尚未打开/哈希JPEG或tensor，正式Source闭合、像素与数值校准仍待固定端点，不写成数据数值资格已齐。

桥不调用锁定 FIT 的 U37.prepare/_RGBReader；只沿 U37 已核 JPEG/EXIF/RGB处理与同一 image_geometry 数值路径：JPEG尺寸等于原shape，EXIF absent/1，RGB uint8→FP32/255，原800内容整数几何→右/下零pad→400；不重建 teacher mean/std。原 `image_geometry.make_geometry(...,800,400)` 同时承担最终增量渲染。桥设计见 [RGB桥设计](U40_FIXED50_DEV_RGB_BRIDGE_DESIGN_20261017.md)，SHA `09fbd437a383b760f8115b043b8086a2b21411b654d7cfdb06e1cd44285cbe20`。

## 3. 固定预测路径与资源

保留已有四个固定范围 `[0,47),[47,94),[94,141),[141,187)`，每叶依次 PLAIN、FULL；同一时刻只有一臂GPU模型、一图编码与当前query块。callee内部取得活动模型，完成该臂后释放全部模型/参数/临时hooks/输出tensor引用，清理后allocated/reserved双0，再转另一臂；不照搬旧N28两臂同时上卡。继续FP32/400/B1、4GiB allocator及既有整卡/RSS/RAM/1800s监督，无OOM自动减块或重试。

每图每臂 `encoded=adapter.encode(rgb400)` **一次**；按原 `qids` 每4个构造CPU int64 indices，对 e/M/L 各做 `index_select(0, indices)`，再上卡。共25次readout、100查询；全DEV每臂187次encode、4,675次readout和18,700输出，不筛score、不截top20、不NMS、不重新分类。

```
delta400 = adapter.predict(encoded, e_selected, M_selected)  # [Q,1,400,400]
corrected = L_selected + geometry.render_delta(delta400)[:,0]
binary = corrected > 0
```

零校准用显式 `[Q,1,400,400]` 零增量，**不把已训练head改零**。每个原查询均要求渲染为零、相加后的native L逐元素相等、原RLE字典相等；使用原 encoder，并以 `population_contract.replace_segmentations` 校准后的整份 native prediction 字典逐项相等。类别、S0 score、query、图像、原顺序及全部非segmentation元数据一律保持。两臂只替换 segmentation；空预测也保留。原F继续由 reader验证，但新UNet分支不把F送入模型，不重新运行teacher。

每叶预测完成、输出及校准/成本回执封存并被 Root 实际接收后，才可成为CPU终点评价输入。所有四范围×两臂齐全、无重叠/缺项、身份及封存字节再核后，CPU评价阶段才读取本次DEV COCO标注文件，该文件不作为预测输入。既有冻结bundle/packet保留完整来源，可能携带历史目标关联字段；预测仅访问原DEV reader明确返回的自然查询字段与图像，不将这些历史目标关联字段用于预测、RGB桥或模型输入。未核验bundle所有成员，不能宣称整个Source完全没有GT派生信息。

## 4. 复用边界与结果

初版只含接口设计；现已有未绑定、未执行数值的 [GPU预测函数](code/U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01/DESIGN.md) `validate_final_endpoint(metadata)` / `predict_shard(...)` 和 [CPU完整人口评价函数](code/U40_FIXED50_DEV_METRICS_DRAFT_20261017_01/README.md) `evaluate_population(torch,np,c,spec,pre,modules,report)`。两核心各3项纯元数据测试通过，真实恢复、RGB、GPU、COCO和资源均未验证；187 JPEG候选引用的元数据接线已完成；[最薄数值外层](code/U40_FIXED50_DEV_ENTRY_DRAFT_20261017_01/README.md)已完成源码、3项纯元数据测试与一次独立静态接口审阅；实际文件闭合、绑定接收及E50数值执行仍待完成。沿原 `population_contract.replace_segmentations/merge_predictions` 合并完整人口，不平均四份 shard AP。

旧 `eval_body.py` 的 `gpu/cpu/run/validate_endpoint` 不能整体调用：其中固定 E24、旧seed/模型schema/参数数、SELECTIVE/WHOLE_NODE与旧gate。只复用其 `official_metrics` 数值函数及已核官方COCO helper，保留标准 segm 12指标、maxDets `[1,10,100]`、原crowd/ignore语义及未定义值。完整DEV为187图/380GT；native和强R原预测同时报告，全部12指标必须重现其原接受结果。原R参考不重新训练或改分数。

固定匹配几何继续用原 native .50 / area=all / maxDets100 的非ignore匹配选择，同一原query比较各方法，保留完整380分母和条件子集范围；不是新方法GT-best或AP替代。`geometry_diagnostic.diagnose` 目前硬四个方法标签，可用明确披露的薄适配 `SELECTIVE↔FULL、WHOLE_NODE↔PLAIN` 调用其原数学，然后将摘要/私有记录改回真实名称并记录alias，或另立仅标签参数化副本；不能暗称执行了旧方法。Omega仍由原native `abs(L)<=2`定义，只用于内外误差分账，不能限制本方法全域写入。

最终主要报告 FULL−PLAIN、FULL/PLAIN各自相对native和强R的完整 mask AP/AP75/AR及既定几何统计。效用判断只按U40冻结主计划；**不复制旧 E24 eval_body 中 N26_AP 常数和 seed推进代码**，不新增阈值、不用中途结果换端点。N26若按主计划作为额外参考，必须单列其不同来源/预算，不能当本轮配对对照。

成本分别计CPU恢复/缓存检查/JPEG+RGB400、UNet encode、query读出、render/RLE及完整叶walltime。新UNet主干成本计入，但冻结teacher预计算成本不在本次前向中；不能沿旧报告 `end_to_end_backbone_cost_included=False` 或宣称完整pipeline FPS。评价无训练、不写新权重，真实恢复/全部校准/资源/COCO结果均待固定端点实际执行。当前任务到此STOP，不开启额外检查叶。


## 5. 原COCO字段语义的窄范围核对

普通原helper直接将prediction的deepcopy交给`loadRes`，未删除bbox/area。官方[coco.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)会优先按非空bbox计算检测面积；[cocoeval.py](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)的segm IoU仍由掩膜计算，检测面积参与未匹配预测的面积范围忽略。实际原预测是否含bbox尚未读取，不能据此认定当前AP错误。新核心保留锁定规则和所有非segmentation字段，实际执行时报告bbox/area字段计数，native/R全部12项仍须精确复现；不静默剔除字段、不另加第二AP或改选择门槛。

私有E50执行控制草稿现已补齐prepare/bind、原调用登记和Root终态接收；两组共6项纯stdlib元数据行为检查及一次独立静态互接审通过。最小收口只将Root接收范围限定为源码、必要JSON和实际输出核验，输入JPEG/cache/checkpoint字节闭合仍由原driver/worker负责；科学来源、数学、资源和原numerical core均未改变。这些私有运维助手不公开发布，也未执行真实prepare/bind/接收。只有原E50/37050实际接收、全部原叶已reap后才可准入，不能以草稿通过代替真实恢复/RGB/COCO结果。
