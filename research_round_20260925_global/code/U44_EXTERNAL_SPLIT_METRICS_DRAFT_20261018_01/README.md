# U44：原 HRSID TEST 的最薄合并与 COCO 适配

2026-10-18。**UNBOUND SOURCE / NOT SCIENTIFICALLY RUN。** 只实现完整人口 JSON 元数据合并与逐方法调用原 COCO 函数。未实现外部 runner、teacher/模型恢复、窗口调度、Root 片接收或资格链；没有真实图像、GT、RLE、权重读取，没有科学库导入、数值推理或 AP。任何实际 Source 字节校验、端点、原始输入身份、片 actual0/Root 接收和资源控制均由外层实际 owner 负责；本模块不接受“已接收”布尔自证，也不创建资格 schema。

## 1. 完整合并

```python
merge_population(manifest, method_names, ranges, shards, population_contract)
# -> {method: 完整196200条预测列表}
```

- `manifest`：原 TEST 顺序的 1962 个最小 image-only 记录，每条恰为 `{dataset_index: ordinal, image_id: 原整数ID}`。ID 保持原值且在 TEST 内唯一；TRAIN/TEST 外层命名空间分离，不重编号、不合并两套裸 ID。窗口原四字段记录由外层只投影这两字段，不带 GT。
- `method_names`：事前固定的唯一方法名列表/元组，必须含 `native/PLAIN/FULL`。若协议要求强 R，须在本参数中预先列入并为每片提供真实 R；本源码不实现 R，不用 native 或旧内部 RLE 伪填，也不按 AP 跳过。外层之后必须逐一评价所有预锁方法。
- `ranges`：预锁的非空、连续、无缺重原 ordinal 半开区间，覆盖 `[0,1962)`；不硬设片数。这里不决定分片大小或动态重试。
- 每个 shard 至少含 `dataset_range` 和 `predictions={method: list}`。`predictions['native']` 是同片唯一原预测参照，不另保存重复 native artifact。每图100条、完整自然 query 0..99，排列允许是原导出排列；其余方法须逐 ordinal 使用同一 image/query 排列且所有非 segmentation 字段精确相同。保留空预测、零分候选、bbox、area、类别、原 S0 与额外元数据，不加 NMS/score cutoff。
- `population_contract` 显式注入原已发布 `N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/population_contract.py`，由外层 Source 关闭；只复用 `_check_image/_non_geometry_signature`。旧 `merge_predictions` 硬187，完全不调用。原 `replace_segmentations` 可由窗口按原 ABI 直接使用，本模块不复制它。

本函数只证明输入元数据的完整性与一致性，不证明 native 来源、RLE 可解码、像素正确或该片已经 Root 接收。返回各方法独立深复制的完整列表，调用方输入不改。

## 2. 单方法全人口评价

```python
evaluate_one_method(method, method_names, manifest, merged[method],
    COCO=COCO, COCOeval=COCOeval, official=official,
    module_sources={'official': official_ref, 'coco': coco_ref, 'cocoeval': cocoeval_ref},
    test_ref=original_TEST_ref, closed_ref=control.closed,
    population_contract=original_population_contract)
# -> 原 coco_metrics 返回字典；stats 为原比例，-1 未定义项保持 -1
```

`official` 必须是原 `spatial_position_preserving_common24_evaluation_source_20261009/qualify_evaluation_cpu.py` 模块，49923 B，SHA256 `2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916`，实际调用其626–644行 `coco_metrics(coco, predictions, ids, COCOeval)`，不调用旧 main。函数必须实际定义在该模块；三个实际 module/class 来源均与显式关闭的引用接合。路径可由未来 Source 副本给定，不写死私有路径。这里只核来源接线；实际文件字节关闭、运行库及二进制闭包仍须由 owner/runtime 完成，不能把任意 callable 注入或本模块返回值当原数学已认证。

只打开显式原 TEST ref；锁1962图/5922 annotation/category1，保留原整数ID和原 GT 所有字段，包括空栅格项、crowd/ignore、area、bbox与多组件 segmentation。一次完整 method 仅调用原底层函数一次，不平均 shard AP；并检查调用后原 GT dataset 不变。没有 GT-best 匹配、几何指标、内部380GT/187图限制、旧 native/R AP 数字复现门或模型选择规则。旧 `legacy_eval.official_metrics` 硬187，不调用。

不剔除预测 bbox/area，也不做 mask-area 重写。原底层深复制后交 `COCO.loadRes`；其检测面积与面积分层语义按同一已准入原库保留，不能据“原字段保留”宣称 loadRes 副本完全未生成字段。官方 segm/maxDets1/10/100/12指标、原 score 与全部候选不变；TEST 四项空栅格敏感性若将来预锁，应另实现明确协议变体，不能混入本原 benchmark 包装。

## 有限测试范围

恰三项标准库行为测试：完整1962图合并保留原整数ID/非平凡qid排列/零分/bbox/area；固定范围、预锁R缺失及qid或字段漂移拒绝；未预锁方法/错误原函数指纹/来源未关闭时在任何 COCO 构造之前停止。预测 segmentation 仅为**不可栅格化的 opaque 元数据 token**，不伪造像素或掩膜正确性。评价测试故意停在 source-closure gate，不制造 COCO/AP 成功。真实数值路径及外部全部方法均未执行。

实际逐项 RED→GREEN 已执行：缺模块 RED、未拒绝额外片范围 RED、缺评价接口 RED，各自最小实现后 GREEN；最终三项16.754秒 PASS，并通过 AST 与未导入 Torch/NumPy/PIL/COCO 的检查。最后仅修正模块说明文字，未改执行逻辑。

本源码需与原 U41、U44窗口及外层源/资源/接收逻辑接线后才可能运行；不是已完成外部实验或 TGRS 证据。
资源范围补记（2026-10-18）：上述完整人口合成 fixture 会构造并深复制数十万预测字典，未采集该测试的绝对 UTC 时刻或 RSS 峰值。开发期间主训练发生可用内存保护中断，现有证据不能确认该测试与中断的因果关系，也不能排除干扰。后续科学训练运行期间只执行小型元数据 fixture；完整人口合成验证须安排在无科学叶运行、且有明确资源余量时。本文件保留已完成的行为测试记录，不把它当作模型或 AP 验证。
