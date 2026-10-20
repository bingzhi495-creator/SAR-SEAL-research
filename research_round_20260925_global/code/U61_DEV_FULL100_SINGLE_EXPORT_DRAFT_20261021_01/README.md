# U61 原 DEV ordinal0/full100 CPU 导出 — UNBOUND / UNEXECUTED

本目录仅含新 worker、有限标准库元数据测试及本说明。数值路径、真实数据读取、Torch/NumPy/PIL 导入、实际保存重载和资源观测均未执行。没有真实 SPEC、owner、admission、控制器、账户修改或资格决定。Root 后续独立绑定及监督；本 worker 的 `COMPLETE` 不能自授原始退出0或 Root 接收。

任务固定为原始 DEV **ordinal0 的完整100候选**，不训练、不构造教师/模型/优化器、不计算 AP、不读取正式新 DEV annotation 或重新栅格化 GT。完整187图/18700候选只做既有 metadata 的身份/顺序连接。原 bundle 反序列化可能接触历史 GT 关联容器，不能称整个上游流程从未接触 GT。

## 调用和固定限额

`entry.py --spec ABSOLUTE_PRIVATE_SPEC.json`。原 Windows owner 由 Root 按 `admission.entry/entry_args` 绑定；本 worker 不创建或修改 owner。CPU0/1、两线程、CUDA_VISIBLE_DEVICES 空、RSS≤16GiB、可用 RAM≥16GiB、1800秒不变。整卡边界、原进程退出、期限硬终止及回收仍是原 owner/Root 责任。来源哈希阶段同样在 owner 期限内；内部采样在选定 psutil 导入及 CPU 验证后开始，不声称覆盖此前 worker 内部采样。

## 精确 spec ABI（恰13键）

| 键 | 内容 |
| --- | --- |
| `schema` | `U61-DEV-full100-single-CPU-export-spec-v1` |
| `operation` | `CPU_U61_DEV_FULL100_SINGLE_EXPORT` |
| `phase` | Root 明确的新独立任务字符串 |
| `expected_versions` | `{"torch":"2.0.1+cu118","numpy":"1.23.5"}` |
| `limits` | 与源码 `LIMITS` 完全一致：allowed_cpus `[0,1]`，threads `2`，maximum_RSS_bytes 和 minimum_available_RAM_bytes 均 `17179869184`，deadline_seconds `1800`，CUDA_visible_devices `""` |
| `dataset_indices` | 严格整数列表 `[0]`，不接受其他索引、布尔值或批次 |
| `source_roles` | 恰 `entry_source/export_source/dev_reader_source/dev_rgb_reader_source/geometry_source` 五个 refs |
| `library_sources` | 恰 psutil、torch、numpy、PIL、PIL.Image、PIL.ImageFile、pycocotools、pycocotools.coco、pycocotools.mask、pycocotools._mask 十个实际规范模块文件 refs |
| `context_ref` | 既有 `ROOT-N28-fixed24-dense-native-DEV-context-v1` 原 DEV context 的 ref，不是 FIT/U38 context，也不是装有该 ref 的候选清单 |
| `original_image_refs` | 原始 ordinal 顺序的完整187个 distinct JPEG refs，不能只传一张、猜文件名或填占位符 |
| `input_sources` | 下述精确去重 union，不包含 context_ref 自身 |
| `sources` | 五 source_roles、十 library_sources、context_ref、input_sources 的精确去重 union；没有额外 owner、父资格、账户、教师权重或未选图载荷 |
| `output_directory` | 原 owner 允许的全新绝对 D: 目录；禁止已存在目录或覆盖输出 |

每个 ref 恰含 `path/bytes/sha256`，路径绝对，bytes 为非负严格整数，SHA 为64位小写十六进制。不同角色引用同一路径时必须完全一致；`sources/input_sources` 本身不得重复。Root 控制链属于独立绑定/控制器，不向此13键 spec 或 worker `sources` 塞控制字段。

`export_source` 必须是未修改的 U59 `package_io.py`：20627字节，SHA256 `8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473`。三个原 DEV/RGB/geometry pin 见 `SOURCE_PINS`。本 worker 不改动这些原模块。

`input_sources` 恰为以下 refs 按实际路径去重后的集合（顺序不构成科学选择）：

1. 原 context 的 `legacy_context/native_predictions/R_predictions/encoder_source/legacy_ROOT_result`。
2. 原 `context.rows[0]` 的 `original_capture_manifest/bundle/dense_packet`。
3. `original_image_refs[0]`。

先只校验/读取原 context，重建该 union，拒绝误加的其他图或权重，再哈希其余明确 Sources。因此完整187 JPEG refs 只作为 metadata 检查及摘要输入，不会自动打开/哈希其余186图、bundle 或 packet。原 reader 的全187引用枚举不构成载荷读取授权。`teacher_checkpoint` 仅由原 reader 比较历史身份，不递归闭合或加载教师权重。

## 原接口连接及实际执行边界

worker 沿用 U53 的显式来源/规范库文件校验、Source成员与稳定文件身份缓存、CPU资源采样、调用者 CPU/NumPy/Python RNG、dtype/grad/线程状态和临时模块别名检查。仅选择的显式文件闭合，不声称整个框架/OS/binary 的全闭包。模块用 compile/exec 加载，避免在不可变 Source 中写 `__pycache__`。

实际调用原 `dev_rgb_reader.prepare`，传入完整187 refs 与原 dev_reader/image_geometry。原 reader 自己验证 legacy context/cache acceptance、187行、原 capture manifest、bundle/shared locator/native packet 身份与自然查询字段，再仅执行 `reader.load(0)`。十库沿用 U53 检查范围；导入 pycocotools 不创建 COCO 对象或调用匹配、栅格化、评价。

原 native JSON 必须完整18700条；每个 ordinal 的连续100条需与原 row 的 image_id/qids、类别1、有限原分数逐项连接。保留全部原字段和空 mask，不排序、不筛选、不改 score/bbox/area/RLE。将 `native[0:100]` 原样交 U59；未知 selected native 字段由 U59 严格拒绝，不能删除字段后自动重试。

U59 调用为 `export_item(torch, control, 0, item, originals, image_geometry=..., source_refs=..., input_refs=...)`。其中三个 source_refs 恰为 `dev_reader/dev_rgb_reader/image_geometry`；五个 input_refs 恰为 `dev_context/native_predictions/original_capture_manifest/bundle/dense_packet`。worker 核验四个原字段与导出副本的 torch.equal、原字节 SHA、独立紧凑存储，以及完整原 native 字典 canonical JSON 一致，再调用 U59 `save_export` 的真实 CPU `weights_only=True` 重载。输出只含 CPU FP32 `rgb400/e/M/L` 和 U59 最小白名单；F、targets 或整个历史 bundle 不进入包。

自然轴 shape：rgb400 `[1,3,400,400]`，e `[100,256]`，M **`[100,200,200]`**，L `[100,H,W]`。原 dense packet 的 M 首轴已由原 reader 去除，不能再添加。原 `query_ids` 是独立导出置换，不得按排序名次重排自然张量。

本单图任务不调用 U59 全人口 `prepare`，不伪造其余186包，也不执行 renderer/零残差 RLE 校准。保存重载技术成功不等于原 native RLE 已完成数值校准、推理、AP 或训练收益。

## 输出与接收接口

全新目录最多形成 `DEV_FULL100_0.pt` 和 `REPORT.json`。所有真实包、报告及 provenance 保持私有。失败保留已有 partial 文件及失败报告，不覆盖、不跳过、不自动重试。

报告 schema 为 `U61-DEV-full100-single-CPU-export-report-v1`，终态 `COMPLETE/FAILED`。关键字段：

- operation/phase/spec_ref、原 source_roles/library_sources/context_ref/input_sources/limits/dataset_indices；库路径及实际版本；
- `requested_batch_images=1`、`actual_completed_images=0或1`、`actual_completed_queries=0或100`、`batch_export_complete`；计数在真实 save/roundtrip 返回后立即更新，后续 cleanup 失败不能抹掉已返回次数，但不授资格；
- `original_metadata_population={images:187,queries:18700,tensors_revalidated:False}`；
- `package_refs={"0":ref}` 和 `export_entry={dataset_index:0,image_id,query_ids,package_ref}`，未保存成功时为空字典/None；
- `images[0]` 含 dataset_index/image_id/queries/status/reader_load_returned/package_written，成功后带 summary；
- `qualified/new_training/AP/TEST/whole_population_exported/full_binary_closure_claimed/model_constructed/teacher_constructed/teacher_weights_loaded/optimizer_created/model_forward_executed/loss_computed/matching_recomputed/new_GT_rasterization_executed/COCO_dataset_object_constructed/DEV_annotation_opened` 均 False，`DEV=True`，`historical_GT_container_may_have_been_loaded=True`；
- 资源观测、错误和清理错误、原 U53 13个 flags、source hash cache 与 cost。只有所有 flags、资源、最终计数与清理通过才报告 COMPLETE。

summary 精确字段为 `images/queries/fields/metadata_sha256/field_proof_schema/native_predictions_metadata_sha256/original_dictionaries_exact/query_order_exact/natural_query_axes_retained/weights_only_CPU_roundtrip_verified/package_bytes/package_sha256`。images=1、queries=100；fields 恰 rgb400/e/M/L，每项恰 `shape/dtype/sha256/source_bitexact/independent_storage/compact_storage`；field_proof_schema=`U59-full100-DEV-field-proof-v1`，四个保留/roundtrip布尔值与三个字段核验布尔值只有实际通过后为 True。

报告的 `original_image_references_metadata_sha256`、summary 的原 native 字典摘要及 U59 metadata 摘要均沿 U59：JSON sort_keys=True、separators=(',',':')、ensure_ascii=True、allow_nan=False，再取 UTF-8 SHA256。不能把旧 ensure_ascii=False 的摘要直接等同。

## 有限测试

本目录执行 `python -B test_metadata.py`，仅测试合成 ref/187行/18700字典 metadata、原序与身份、不允许额外载荷、角色/输入闭合、摘要白名单。测试不调用 main、不导入科学库、不打开真实 context/index/图像/packet/bundle/权重/Ops/资格链、不触碰进程或网络。AST 检查只证明普通源码可解析和没有顶层科学库导入；真实字节、原库路径、张量等值、storage、实际读取、weights-only 重载、资源及原退出均 **UNEXECUTED**，必须由 Root 后续独立任务证明。
