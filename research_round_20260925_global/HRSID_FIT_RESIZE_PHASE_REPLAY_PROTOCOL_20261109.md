# HRSID FIT 训练增强中的掩膜采样相位

在冻结 HRSID E24 的留出图组中，框偏差与常量对照都主要产生向上 1 像素的平移，净增严格匹配同时损失了 15 个原严格配对。此处只检验一个可能来源。原生掩膜最近邻与图像像素中心约定不同，是否在实际训练增强的目标中引入可复现的纵向偏移。重放冻结的 114 张 FIT 图，109 个代理组各一张主图，另有 5 张压力图，不接触 DEV、TEST 或模型。每图 4 个固定种子，按原冻结清单顺序使用 520198000 + 4×图序号 + 重复序号，序号从零开始。两臂只改掩膜 Resize 的采样位置。 [框偏差信号结果](HRSID_FIT_BOX_SHIFT_PROTOCOL_20261109.md)

原生臂执行完整 576 训练增强，包括原 RandomSelect、Resize、RandomSizedCrop、水平翻转和归一化。对照臂使用同一原生逻辑，在每次掩膜 Resize 上显式对普通掩膜张量使用 `nearest-exact`。逐视图确认随机状态、裁剪和翻转轨迹及归一化图像逐字节相同，共有原标注行的框、类别和其他非掩膜字段相同。裁剪逻辑依赖掩膜非空性，因此两个臂的保留行可能不同，单独报告差异、空掩膜及最终可监督行。原始实例索引穿过几何变换，用于对齐和核对，输出只保留聚合。锁定 torchvision 0.25 的 Mask 分发强制旧最近邻，直接传插值枚举会被忽略。 [原生 RF-DETR 训练增强](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/datasets/coco.py#L1037-L1088) · [TorchVision Mask Resize](https://github.com/pytorch/vision/blob/v0.25.0/torchvision/transforms/v2/functional/_geometry.py#L377-L395)

共同参考沿相同 Resize、裁剪和水平翻转的反向几何，在每个最终像素中心一次性读取原 FIT 掩膜。坐标使用像素边界约定，整格平分采用原坐标下取整。它用于衡量顺序重采样相对几何基准的偏移，不替代实例标注。只在两个臂与参考均保有非空掩膜的同一原行比较质心与 IoU，将质心差乘反向尺度报告为原图像素单位，符号沿增强图坐标。报告纵横两轴、原 COCO 面积、直缩或裁剪、水平翻转及行存活。主指标为原面积小于 1,024 平方像素的主层目标纵向有符号偏差。先在每个主图组内对固定视图和实例求均值，再对图组等权平均。

至少 30 个主组有共同小目标，原生纵向偏差至少 +0.25 像素，中心臂使纵向平均偏差的绝对值下降至少 0.25 像素，并且该下降的 95% 图组配对区间下界大于零，才视为可进入下一项。使用 4,096 次重采样，固定种子 5219。中心臂的组均参考 IoU 不得下降超过 0.005，失去的原生非空训练行不超过主层全部原生非空行的 0.5%。压力层和其余分层仅描述。只有通过才锁定权重训练的公平配对，主要变量仍为掩膜 Resize，完整报告行选择变化。未通过保留负证据，区分相位、裁剪选择及几何参考误差。无前向、训练更新或 AP，现有 E24 模型的偏移因果尚待训练对照。

旧最近邻的像素错位、TorchVision Mask 插值选项及 HRSID 框裁剪细化的半像素调整均已有来源。真实同种子标签流重放能回答当前实现的问题，普通中心采样本身不构成新的方法贡献。 [PyTorch 插值定义](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.interpolate.html) · [像素偏移议题](https://github.com/pytorch/pytorch/issues/34808) · [TorchVision 修正](https://github.com/pytorch/vision/pull/9497) · [RSI4EO 预印本附录](https://arxiv.org/html/2610.18845v2#A1.SS3)
