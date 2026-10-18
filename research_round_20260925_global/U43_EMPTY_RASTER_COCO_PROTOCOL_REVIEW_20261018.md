# HRSID 空栅格 GT 与 COCO 评价协议核查

状态：公开一手源码静态核查；检索日 2026-10-18。未执行评价、未读取本机图像、GT、运行日志、资格链或模型。Root 提供的运行记录为：全部 5604 图解码完成，16969 个 annotation 全保留，其中 13 个官方栅格结果为空（TRAIN 9、TEST 4）；本笔记不独立核实这些实数，失败记录已由 Root 接收并保留；该接收只确认终态、来源、输出标量与记账，不授予全部实例非空、语义对齐或模型/AP资格。

COCO 默认行为：在已读上游 `cocoeval.py`（master）的 `_prepare/evaluateImg/accumulate` 中，没有“栅格面积为零即忽略”的规则。若该 GT 非 crowd、声明的 annotation area 落在当前面积范围内，它仍计入正例分母；官方 `maskApi.c:rleIou` 在交集为零时给 IoU 0，空—空也不会得到 IoU 1。因此该 GT 无法在正 IoU 阈值下匹配，会限制召回；不能据数量直接推算 AP 损失。具体 13 项的 crowd/area/bbox 属性本轮未读，不把条件推断写成其逐项事实。[官方 COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)、[官方 maskApi.c](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c)

HRSID 作者公开 README 与可见仓库说明中，未找到“自动删除空栅格 GT”的规范或专用评价实现；这只是已查公开材料的证据边界，不证明所有作者实验均未过滤。README 的实例总数与本次 Root 报告的实际 JSON 计数不同，也不能凭 README 自动删项凑数。[作者仓库与 README](https://github.com/chaozhong2010/HRSID)

MMDetection v3.1.0：`CocoDataset.parse_data_info` 会按 ignore、bbox、声明 area 和类别筛项，但不检查解码后掩膜面积；`LoadAnnotations` 对多边形结构无效与“结构合法但栅格为空”也不是同一判据。`FilterAnnotations(by_mask=True)` 才提供可选掩膜面积过滤，默认 by_mask=False。训练数据过滤不等于原 TEST 评价过滤；`CocoMetric(ann_file=...)` 直接从该注释文件构造 COCO GT。[CocoDataset](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/mmdet/datasets/coco.py)、[加载与可选过滤](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/mmdet/datasets/transforms/loading.py)、[CocoMetric](https://github.com/open-mmlab/mmdetection/blob/v3.1.0/mmdet/evaluation/metrics/coco_metric.py)

后续最小协议建议（仅方案，未执行）：

- 主 benchmark 保持原 TEST 1962 图、5922 annotation、原 area/bbox/crowd、全部 100 queries 和同一官方评价口径；不删图、不修多边形、不改变当前冻结 Source。U43 原“每实例非空”预锁未满足的失败照实保留；“解码完成”和“全部实例非空”分开陈述，不把该失败扩大为数据不可用。
- 在查看任何 AP 前，预锁 TEST 四项的 split-local annotation ID、源文件指纹及原栅格实现，另设 nonempty-raster GT 敏感性视图：仅排除这四个 annotation，1962 图与所有方法的同一份预测完全不变，另报分母与指标差异。它是协议变体，不能替代主结果；TRAIN 九项不混入测试敏感性，也不据此改当前训练。
- 不仅添加 `ignore=1`：上游 segm `_prepare` 会用 iscrowd 重写 ignore；更不能伪标 crowd。若实施敏感性，使用单独、明确标注的 annotation 子集。不得看 AP 后择优协议，不把空栅格诊断称为方法创新或语义对齐证明。

范围：仅上述公开版本/分支代码；未核本机安装版本与其字节一致，未证明具体空栅格成因，也未证明 JPG/PNG 标注语义一致。
