# U41 单张新图原生100查询采集草稿

状态：普通、未绑定、未执行数值。仅提供函数与三个标准库接口测试；没有 runner、人口框架、prepare/binder、模型构造、权重加载、训练或输出文件写入。本稿不授予 HRSID1962、SSDD232、实际像素读取或 GPU 运行资格，不改 U40。

## 接口

```python
capture_native_image(
    torch, np, model, image_metadata,
    modules=..., source_refs=..., preprocessing=..., provenance=...,
    closed_ref=..., check=...,
)
```

`model` 必须已由调用者验证并加载为冻结 `eval`、无参数梯度的 FP32 CUDA0 原 teacher。函数不负责加载、来源资格核验、初始化 CUDA、设置资源上限或释放 caller 持有的 model。调用者负责现有资源与生命周期边界，不能仅凭本函数返回视为新的运行许可。

`image_metadata` 精确三个键：`image_id:int`、`image_ref:{path,bytes,sha256}`、`original_shape:[H,W]`。不接受 annotations、parser、dataset 或 GT 字段。`closed_ref(ref)` 必须由未来调用者执行既定来源闭合并返回同一绝对路径；`check()` 是调用者的期限/资源检查。这里没有 fallback 或推断文件名。

`modules` 的实际必需对象为 `Compose`、`PackDetInputs`、`pseudo_collate`，以及模块对象 `observer`、`renderer`、`exporter`、`pref`、`masks`。`source_refs` 精确十项：前三个普通模块 `observer/renderer/exporter`、`pref/masks`、`Compose/pseudo_collate`、`formatting`（PackDetInputs）、`data_preprocessor`（实际 DetDataPreprocessor 类）、`base_preprocessor`（其实际 ImgDataPreprocessor 基类）。检查实际已加载文件路径与闭合引用；六个直接数学/兼容性源码 SHA 固定在 `PINS`。完整 Torch/MMCV/扩展二进制及 teacher 模型源码闭合仍属于未来调用者，不能把本十项表当作整个运行环境闭包。

`preprocessing` 必须为原配置闭合得到的解析值，包含原 `mean/std/bgr_to_rgb/rgb_to_bgr/pad_value/pad_size_divisor/pad_mask/batch_augments`，不补光度默认值。实际 loaded preprocessor 的 mean/std buffer、通道转换、divisor、固定补边参数再次核对。当前仅接受原三通道、divisor32、固定800、图像零padding、pad_mask=True、pad_seg=False 的分支。`rgb_to_bgr=False` 必须明确提供。

`provenance` 精确包含：`teacher_checkpoint`、`loading_receipt` 两个严格引用，`config_sources` 原配置依赖引用列表，`resolved_preprocessing` 与传入解析配置完全相等，`teacher_state_proof` 为调用者已核定模型状态摘要。函数只检查前两者引用格式，不打开 checkpoint 或加载回执，不自证其 Root 接收资格；未来调用者必须先完成这一来源关联。函数对实际 model 前后状态摘要与给定摘要做相等检查。

## 原始数值函数与 e/h 边界

以下路径相对于普通发布副本 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/`：

- `N07_frozen_E24_original_detail_point_refinement_source_20261012_01/cache_fit.py:205–228` 的 `observed_fields`：实际 pixel_decoder 输出 F、mask_embed **输出** e、head 最终 M/C；原函数检查 FP32、finite、自然轴形状及 `einsum(e,F)==M`。不是相邻 `native_ordinal.../train.py` pre-hook 捕获的 h。
- `frozen_feature_linear_readout_diagnostic_source_20261010/diagnose.py:200–209` 的 `render_original`：M200→800双线性、裁实际内容、再回原 H/W，均 `align_corners=False`。
- `shared_readout_objective_control_source_20261010/train.py:367–387` 的 `export_original`：原单类 all100 fusion、原类别概率与掩膜质量乘积、RLE编码和完整 export query顺序。

控制流程来自 N07 `evaluate.py:317–334`：只一次 extract_feat/head 前向；另外以完整 `fusion.predict(...,rescale=True)` 校准掩膜、分数、类别。单图私有 head seed 沿原式 `100000+image_id`，前后检查 Torch CPU/CUDA RNG 与模型状态不变；无 loss/matcher/优化器。

## 无 GT 的实际 padding 接线

旧 loader `native_ordinal_two_mode_mask_adapter_learning_preparation_20261008/train.py:246–290` 读取标注并剥离后再给 head，不能直接作为新测试图的 image-only loader。本稿只执行其影像步骤：`LoadImageFromFile(to_float32=True)`、`Resize(scale=(800,800),keep_ratio=True)`、原 PackDetInputs 元信息键；不使用 LoadAnnotations 或 annotation_mapping。

2026-10-17 本轮只读实际安装库（未导入）：

- `mmdet/datasets/transforms/formatting.py:92–117`：没有 annotation 输入时仍创建空 InstanceData。它没有 masks，因此不能直接交给原 training=True 固定补边器。
- `mmdet/models/data_preprocessors/data_preprocessor.py:358–362`：BatchFixedSizePad 在 sample 非 None 且 pad_mask=True 时无额外 guard 访问 masks；直接删除 LoadAnnotations 后照抄旧调用会失败。
- 同文件 `DetDataPreprocessor.forward:132–151` 和 `BatchFixedSizePad.forward:344–375`：`data_samples=None` 跳过目标字段分支，图像 F.pad 仍按原式执行。
- `mmengine/model/base_model/data_preprocessor.py:50–53,255–279,306–308`：None 原样保留；图像通道转换、normalize、stack/pad 不依赖 sample。

因此本稿保持原 `training=True`、pad_mask=True、原 augment 和光度值，传入只含 inputs 的 pseudo_collate batch，显式 `data_samples=None`；不造 GT/空 masks，不关闭目标 padding 配置。用原 `_get_pad_shape` 获取 divisor 对齐尺寸，再按固定补边器真实 early-return 条件和最终 inputs.shape 恢复 metadata：需要固定补边时原库同时更新 batch_input_shape/pad_shape；无需补边时保留原 divisor 尺寸。当前固定800约束下二者最后都必须与实际输出相符。Pack产生的空目标容器只被删除，不交给 preprocessor/head。

这是源码层面的调用关系与代数隔离，**尚未证明真实图像预处理 tensor 与旧标注路径 bitexact**，未执行前向或跨集几何验证。原库分支及上游配置任何变化均必须重新明确，不能凭本注释放行。

## 返回与范围

返回自然轴 CUDA tensors：`F[1,256,200,200]`、`e[1,100,256]`、`M[1,100,200,200]`、`C[1,100,2]`、`L[100,H,W]`；同时返回 `natural_query_ids=0..99`、几何身份、`export_query_ids`、原 `native_predictions`、自然轴 `S0_by_query/score_factors_by_query/category_ids_by_query` 和原 class_packet。只有导出时以真实 qids 索引，自然 F/e/M/C/L 不重排。S0 是原 class probability × mean sigmoid(L) over L>0（分母加1e-6）；保留所有查询及空掩膜，无 score cutoff/NMS。调用者持有返回 tensors，必须按图释放，不应据此把整个人口常驻显存。

返回 `receipt` 仅为单图函数检查记录，`qualified=False`、`numerical_runtime_qualified=False`、`population_qualification=False`。不含标注/GT语义声明之外的新数据资格；HRSID版本、SSDD测试人口、模型历史接触和来源依旧待调用者独立闭合。

## 本轮检查

三项标准库测试先在模块缺失时真实 RED，之后 GREEN：纯影像白名单与 annotation/parser 毒丸不访问，data_samples=None/training=True/实际pad元信息接线；缺光度或固定补边配置拒绝；真实捕获接口选择最终 e 而不是 h、保留自然顺序。注入 stub 只用于这些调用/元数据行为，不模拟 Torch、图片或数值算子，不称预处理/模型数值通过。AST 检查单独进行。没有调用任何科学 main 或读取实际状态、图像、GT、权重、缓存、SOURCE 或资格链。
