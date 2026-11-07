# RF-DETR canonical COCO mask export draft

2026-11-07。状态为普通 source-only 协议适配候选。7 项标准库合成测试通过，真实 Torch PostProcess、实际 COCO codec、图像身份和 canonical GT 对齐仍待独立资格验证。本适配属于评估接口准备，研究创新和真实 AP 证据须由后续研究提供。

## 固定输入与输出

`export_coco_mask_predictions(predictions, *, image_id, original_hw, label2cat, encode_mask)` 每次处理一幅图。`predictions` 是已完成原生 PostProcess 的 `labels`、`scores`、`masks` 三个 Tensor 各自 `.detach().cpu().tolist()` 后组成的字典。三字段按同一行对齐。其他字段如 boxes 可以存在，导出器只读取这三项。

- `labels` 为长度 K 的 Python 整数列表，当前固定单类双 logit 布局，K≤100。
- `scores` 为对应的有限 sigmoid 分数，取值 [0,1]。包括0和小于0.5的分数都保留原值。
- `masks` 的 Python 嵌套列表形状为 `[K,1,original_H,original_W]`。保留行的像素必须已经是 bool。
- `image_id` 为调用方提供的 canonical 整数身份，`original_hw` 为正整数原始 H/W。该函数只校验类型和形状，身份与尺寸的来源由调用方的输入资格链证明。
- `label2cat` 必须显式为整数映射 `{0: 1}`，缺省、空映射及 bool 混用均拒绝。
- `encode_mask` 是调用方注入的固定 COCO 编码器包装函数，接收 `[H,W]` bool 列表，返回压缩 RLE 字典。

返回值是可以拼接成 COCO prediction JSON 的列表。每项只含 `image_id`、`category_id`、`score`、`segmentation`。`segmentation.size` 必须等于原 H/W，`counts` 统一为非空 ASCII 字符串。合法全零预测 mask 及其 RLE 仍保留。导出过程保持行顺序，函数读取的输入限于上述预测和显式参数。

## 原生顺序与类别边界

一般的连续训练标签应通过显式逆映射恢复原 category ID，未映射标签的处理需要预先定义。

本候选进一步固定为单类、2-logit 头。只接受标签0和1，0恢复为 category_id=1，1作为唯一预期的未映射槽丢弃。负数及≥2的整数标签均报错，以阻止错误模型头或错误映射被当成可用结果。禁止以 `label2cat.get(label, label)` 回退到原数值。

调用方必须先执行作者的全部 query×class sigmoid top100。随后本函数按逆映射过滤槽1，最终输出可能少于100条。函数没有剩余 query、logits 或再选择入口。调用方必须记录该顺序，不能在过滤前改成只对 label0 重取 top100，也不能在过滤后补入未被作者 top100 选中的实例。

原生 PostProcess 的 `score_threshold` 必须为 `None`，`num_select` 必须为100，`upsample_masks_to_image_size` 必须为 True，`target_sizes` 必须为真实原 H/W。导出器能拒绝超过100行，却无法从一份已被外部阈值截断的列表恢复丢失预测，也无法证明其上游选择历史。调用方须独立记录并核实这些参数。官方 `predict()` 演示入口的默认0.5阈值不适用于此契约。

固定官方源码 `models/postprocess.py` 的 `_select_topk` 在130–153行选择全部 query/class 对。`_postprocess_masks` 在223–319行按照选定 query 获取 mask logits，再以 bilinear、`align_corners=False` 插值到目标原尺寸，最后按 **logit>0** 得到 `[K,1,H,W]` bool mask。该语义由调用方的原生 PostProcess 提供。本适配仅接收其原尺寸二值结果，拒绝把较小网格、浮点 logits 或整数0/1列表直接冒充该输入。参考固定提交 `639eac4555ceb89ef399ccda3fde716bbcfe34cb`，本机该普通源码 SHA256 为 `0e9a13190e559e517d84b2239ab034fa7bed5e1f6372b0a608c2a095c1b67b82`。[官方 PostProcess](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py)

## 编码器注入与 canonical 评估

正式调用方应从冻结的 package inventory 取得实际 pycocotools 版本和来源，注入该实例 `mask.encode` 的包装器。包装语义应为把二维 bool mask 转成 `uint8` Fortran-contiguous 数组，再交给该固定 encoder。以下仅描述调用契约，`np` 和 `mask_utils` 由已获资格的调用方提供。

```python
def encode_fixed_coco(mask_2d):
    return mask_utils.encode(np.asfortranarray(mask_2d, dtype=np.uint8))
```

导出器拒绝 coder 返回错误尺寸、未压缩 list counts、空 counts 或非 ASCII counts。它验证 RLE 外层契约，实际压缩编码正确性仍须使用冻结 codec 的 decode 做原像素 round-trip，并核对 JSON 读回后的 size 与像素。在这轮合成测试中，encoder 是只识别给定2×2 mask的注入 fixture，实际 pycocotools/NumPy 编解码资格仍待验证。

主评估由外部 canonical evaluator 消费这些预测，沿用原始 mask GT、原 image/category ID、crowd/area 和预定 maxDets100。导出函数的参数中没有 GT。在线 callback 的 GT 经缩放再放大问题应通过原 GT 独立评估通道处理，主 GT 的读取、重采样及修改均属于该函数接口之外。既有 `RFDETR_SAR_DATA_INTERFACE_REVIEW_20261107_01.md` 和 `RFDETR_BASELINE_AND_DIAGNOSTIC_PLAN_20261107.md` 提供协议背景。

当前选择 Python 列表接口用于标准库隔离测试。真实原尺寸 K×H×W `.tolist()` 的内存和速度开销需后续测量，当前候选尚未获得吞吐或资源资格。

## 合成验证与后续门槛

执行环境为 Python3.8.17。使用 `python -m unittest -v test_canonical_export` 验证公共导出接口，7项测试最终通过。每个新增行为先出现 RED，再作最小实现进入 GREEN。首个黄金测试的 RED 为 NotImplementedError，后续 RED 分别暴露未映射键 KeyError、错误/缺失映射、mask尺寸/类型、行对齐/分数/身份、coder结构及越界标签，最终 suite exit0。

覆盖黄金低分 JSON 输出、作者 top100 后槽1删除且不补选、缺映射拒绝、原尺寸单通道 bool mask、身份/行数/有限分数、压缩 RLE 外层守卫与全零 mask、双 logit 布局以外的标签拒绝。测试数据全为手写合成结构。

真实资格阶段须核实实际模型类别数、原生 PostProcess 参数、Tensor→Python的逐行一致性、真实 image_id/HW 来源、冻结 codec 的像素 round-trip，以及在既定原 GT/maxDets100 下的独立评估。只有这些实际完成后，才可报告真实导出验证和 AP。当前交付范围为源码、合成测试与本说明。

代码 SHA256 为 `b2e99e581acb0236f6cee5542321bb3e01b2c620f2c727062b28b71eedfd1820`，测试 SHA256 为 `ae52df5af6f513b0deb498551166733db3d6fa1e15de18805ac3ff9a7178e6a6`。
