# U44 新图单图输入薄桥

普通未绑定草稿；**真实张量、图片解码、几何和模型数值均未执行**。仅两项标准库合成元数据测试与 AST 检查。没有 population、runner、binder、模型构造/恢复、AP 或新资格叶。

```python
result = make_image_item(
    torch, np, Image, capture, dataset_index, image_reference,
    image_geometry=image_geometry, source_refs=source_refs,
    closed_ref=closed_ref, check=check,
)
```

`capture` 为原 U41 `capture_native_image` 返回字典；只接受其精确14键和原 receipt，不接受 GT 或额外关联字段。`source_refs` 精确为 `image_geometry`、`PIL.Image` 两个调用者预准入的 **canonical** 严格引用。核实际模块 `__file__.resolve()`、闭合路径及独立 bytes/SHA；不把 C/E junction 路径差异静默视为同一严格引用。Torch/NumPy 实际库来源与版本由外层闭合。

返回三键 `item/row/originals`。`row` 仅含外层指定 `dataset_index` 和真实 `image_id`，无187人口限制，无 split、G、GT 或伪 matcher。`originals` 保留原 U41 输出对象与全部原字段/排列，不重写 score、category 或 RLE。此版本精确沿原 exporter 的 image_id/query_id/category_id/score/segmentation 五键 ABI；未来该源若改变，须显式重新适配，不能忽略新增字段。

`item` 仅有 CPU `e[100,256]`、`M[100,200,200]`、`L[100,H,W]`、原 identity/query_ids、`rgb400`、geometry 和少量 receipt。e/M 仅去原 batch0，e/M/L detach 后独立复制到 CPU，不按导出名次重排。输出查询顺序由 `export_query_ids` 单独保留；现有 `U40.prediction_core._calibration_and_prediction:175–239` 按 qids 索引自然 e/M/L，可直接接入这个 item/row/originals，逐图一次 encode、查询4一组、原图 L+render_delta、阈值>0、原生 RLE 与显式零残差校准不变。

RGB 处理沿已发布 U37 `rgb_matched_reader.py:103–138` 与 DEV `dev_rgb_reader.py:167–195`：原 JPEG/尺寸、EXIF absent或1而不旋转、原 RGB 转换、uint8→FP32/255。U37 来源 SHA 固定记在代码 provenance；只复用其 JPEG 逻辑，不导入 FIT reader、G/GT 合同。真正几何直接调用原 `image_geometry.make_geometry/prepare_rgb`，源码 SHA `65db107d…00abf7`，保持整数800补边与400分支。该 RGB 是新支路输入约定，不是 teacher 归一化重建。

桥不保存 F/C/完整 capture/teacher 引用，**不代表调用者已释放原 capture 或 teacher**。返回后调用者必须自行删除所有原 GPU 别名、hooks/临时量和 teacher，按原资源协议实际清到 allocated/reserved 双0，才把 PLAIN 或 FULL 上卡。若后续强R还需要F，须在调用者明示小窗口资源与生命周期；本 item 不长期保存它。两臂模型严格恢复、teacher预处理与来源、完整窗口释放/成本、每叶限额及最终人口完整性仍由未来 Root-owned 调用方负责。

公开 `prepare_metadata` 只核图像身份、100自然查询与真实导出排列；标准库测试不能证明 CPU复制、真实图片、e/M/L、渲染、RLE或AP通过。该桥不为 HRSID/任何人口授予数值、盲测、语义对齐或评价资格，未触碰已有源码或发布副本。
