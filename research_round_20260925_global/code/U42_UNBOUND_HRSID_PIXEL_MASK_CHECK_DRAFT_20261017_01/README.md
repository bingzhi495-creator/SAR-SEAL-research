# U42 单图真实像素与 COCO polygon 栅格兼容检查草稿

**未绑定，未执行真实解码或栅格化。** 本目录仅有单图函数、短说明和三项标准库测试；没有人口框架、数据发现、runner、binder、模型、AP 或自动重试。未读取实际图像、标注、权重、缓存或资格链，

接口：

```python
check_one_image(Image, ImageFile, np, masks, closed_ref, check,
                image_ref, image_record, annotations)
```

运行库必须由未来实际拥有的调用方注入：真实 Pillow `Image/ImageFile`、NumPy、官方 `pycocotools.mask`。模块顶层只导入标准库。`closed_ref(strict_ref)` 负责独立核对原编码文件的严格路径/字节/SHA 引用并返回路径；`check()` 负责原资源与中断约束，异常直接向上传递。函数不发现或猜测路径，不把测试注入对象当作科学验证。

函数强制 `Image.load()` 完整解码，不能用 `verify()` 替代；要求全局 `LOAD_TRUNCATED_IMAGES` 原本为 False，不修改该全局开关。原文件必须为 800×800、RGB JPEG，解码数组必须为 uint8 `[800,800,3]`；不静默转灰度、转 RGB、缩放、翻转或旋转。EXIF 读取成功且方向 tag 缺失时记录 `absent_no_transform_applied`，按存储坐标处理，**不声称方向 tag=1 已验证**；明确 int1 正常，非1、非法值或 EXIF 读取异常拒绝。缺失 tag 不证明标注语义对齐，策略拒绝也不能一概解释为影像损坏。

每个原 annotation 的所有 polygon 组件一次传入官方 `frPyObjects`，再 union `merge`、`decode` 到原 800×800 画布。一个 annotation 始终是一个实例；crowd 值保留，不以它丢弃实例。检查二值 uint8、非空性及官方 RLE area 与像素和一致。坏元数据、空/坏 mask 均保留独立失败行，整图失败，不静默过滤或修复。图像解码失败时原实例行标为未尝试；空标注图合法，但仍须完整真实像素解码。

polygon 只先检查组件结构与数值有限性，有限越界坐标仍原样传入官方画布栅格器。每个原实例记录 `out_of_canvas_coordinate_count` 和 `out_of_canvas_component_count`；坐标小于0或大于800计越界，恰为800不计。计数仅诊断，不构成拒绝门，不裁剪、修复或据此认定错标。官方栅格结果仍按原二值/非空判据检查。

原 annotation area/bbox 与栅格测量值只记录差值，不要求连续 polygon 面积等于离散像素面积。`geometry_metadata_difference` 是纯标量记账辅助，不执行栅格化，也不裁定几何/语义有效性。输入及标注不改写；不按裸 ID 合并 train/test，不使用有版本差异的 merged JSON 替代原 split。

函数只返回标量、简短元数据和摘要。原像素在摘要后释放，每个 annotation 的 mask/RLE 临时对象逐项释放；不返回或写入像素、mask、polygon 或权重。返回的 `PASSED` 仅指本次单图解码和原画布栅格兼容检查，没有独立 Source/运行资格、人口资格、语义对齐、JPG/PNG 像素等价、盲 TEST、跨集效果或 AP 含义。输入来自哪个 split 由调用方明确登记，函数不推断。

已有普通代码中可核对官方调用链：`protocol/seal_eval.py:37–41` 的 polygon→RLE，`protocol/boundary_metric.py:20–27` 的 merge/decode。本目录不导入或运行旧评价器，也不借用旧 PIL ImageDraw 栅格器。20261015 已核文件头/编码与 split 语义的结果仍保留，本函数不重复下载、扫描或把 `verify` 追认为真实像素成功。

三项已运行测试仅覆盖：load 失败不可被 verify 绕过及 EXIF 元数据分支；多组件仍保留为一个原实例的失败回执；给定标量的 area/bbox 差值不成为相等门槛。所有解码注入都在像素产生前故意失败，NumPy/mask API 被禁止访问，**没有伪造任何数值成功**。真实解码、官方 raster 算法、释放资源与源字节核验仍待未来 Root-owned 实际任务。
