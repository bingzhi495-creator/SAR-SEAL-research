# U43 官方 HRSID 两 split 完整人口像素/栅格检查草稿

**未绑定，未执行真实数据检查。** 本目录只写人口函数和三项标准库元数据测试；没有数据发现、runner、binder、模型、AP 或自动重试。未导入 PIL/NumPy/COCO/Torch，未读实际原图、标注、缓存、权重或资格链。

```python
run_population(Image, ImageFile, np, masks, context, checker,
               *, closed_ref, check, record_image, report)
```

未来原拥有的调用方注入真实运行库及最终 U42 `check_one_image` 原函数、原 Source/资源检查。`report` 必须是调用方持有的空字典，函数逐步填入并返回同一对象。`closed_ref(strict_ref)` 独立核原引用后返回路径；`check()` 资源/中断异常原样传播。核心不写文件。`record_image(result)` 由未来 body 每图写入完整私有 JSONL；对象是 U42 原结果加 `split`、`split_image_ordinal`、`global_image_ordinal`，核心不保存完整结果全集。调用方应立即写入，不累计像素、mask 或逐图结果。

context 精确四字段：

- `schema: U43-HRSID-official-split-pixel-mask-check-context-v1`。
- `image_root`：原 5604 JPG 所在绝对目录，不猜文件名或后缀。
- `annotations`：精确 train/test 两个严格 path/bytes/sha256 引用。
- `image_refs_by_filename`：原 5604 文件名到严格 JPG 引用的精确映射。

两原 JSON 经 `closed_ref`、实际文本字节/SHA 和拒重复键 JSON 解析。TRAIN 固定 3642 图/11047 原实例，TEST 固定 1962 图/5922 原实例，总计5604/16969；不用 merged16951 文件。每个 split 内 image/annotation ID 唯一，跨 split 裸 ID 复用合法，绝不把两 JSON 按裸 ID 合并。保留原 images 顺序和原 annotations 内相同 image 的相对顺序；多 polygon 组件仍是一个原实例。类别1、声明800×800，文件名跨 split 无交集，并集精确对应5604引用；路径须与显式 root/原文件名一致，不发现或重映射数据。

每图直接调用原 U42，不重写解码、栅格或 matcher。空标注图合法；不按 crowd 丢弃。U42 明确返回的图像/栅格/标注失败会保留、汇总并继续后续图；未尝试实例也保留。资源检查、Source、审计回调及非 U42 已处理的异常直接传播，partial report 标 `INTERRUPTED`，不能授完整通过。不吞掉中断，不自动重试。输入原文和标注不改写，逐图临时结果释放；完整明细仅经 callback。输入 JSON/引用索引是必要元数据，核心不持有像素或 mask。

报告 schema 为 `U43-HRSID-population-pixel-mask-check-report-pending-v1`。`split_summaries.train/test` 和 `totals` 的计数字段一致：declared_images、declared_annotations、processed_images、recorded_images、original_annotations_seen、image_checks_passed/failed、pixel_decode_passed/failed_images、empty_annotation_images、raster_attempted/passed/failed_annotations、annotation_metadata_failed_count、not_attempted_annotations。

每 split 另有 EXIF_policy_counts（absent_no_transform_applied、declared_1_no_transform_applied、no_policy_recorded、EXIF_rejection_or_read_failure_images）、geometry_difference_statistics 和 out_of_canvas_diagnostics。area/bbox 差值统计仅覆盖成功栅格实例，报告 count/sum/absolute_sum/minimum/maximum/mean/absolute_mean/nonzero_count，不把连续/离散差值当相等门槛。越界计数按 U42 原规则只诊断，不裁剪、修复或据此认定错标；无可用诊断的失败行单独计数。

`duplicate_encoded_SHA_diagnostics` 仅按原 strictref SHA 分组，报告跨 split、split 内重复组和出现次数，所有图仍保留。它不独立重哈希图像、不比较 PNG、像素或母景，不推断重复组的标注语义是否相同；实际图像 Source 字节检查仍由每图原 U42/closed_ref 完成。不自动去重、改划分或挑标签。

只有全5604图实际 U42 PASSED、全16969原实例栅格通过、零 pixel/raster/metadata 失败、零未尝试、全部5604 callback 写入返回，才标 `INPUT_CHECK_COMPLETE_PENDING_ROOT` 和 `input_check_complete: true`。仅访问完但有失败则 `INPUT_CHECK_FAILED_PENDING_ROOT`；partial 或异常不能冒充完整。full_image_visitation、full_private_audit_coverage、annotation_count_conserved 与计数分开保留。所有状态始终 qualified/population_qualified/semantic_alignment_proven/JPG_equals_PNG_pixels_proven/blind_TEST_proven/new_training/AP 为 false；它不创建原运行、数据资格、语义对齐、盲测、模型效果或外部 AP 证据。

三项已执行测试仅验证 fabricated metadata/counters：split ID 命名空间及原实例分组；错误计数、split 内重复 ID、JSON 重复键拒绝；失败或未覆盖计数不授完整状态。没有调用 run_population、U42、真实或伪造成功的 PIL/NumPy/mask。真实全人口科学执行与资源行为仍待 Root-owned 原任务。
