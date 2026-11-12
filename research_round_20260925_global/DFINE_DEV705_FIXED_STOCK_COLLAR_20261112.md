# D-FINE 固定原掩膜的框外补全上界

在已使用的 HRSID DEV705 上重放 D-FINE E24 的 1,329 个固定小型细长船匹配、27 个文件名前缀来源代理组。前序结果在检测框向外扩 2 个原生像素后允许重选全部像元，确定 134 个原框排除的目标在 IoU 0.90 上理论可达。本次固定同一匹配和原掩膜，进一步区分框外补全与框内误差。两种原掩膜与连续真值相交面积实现的最大差为 2.84×10⁻¹³。未运行模型、训练、计算 AP 或访问官方 TEST。 [前序原框结果](DFINE_DEV705_NATIVE_BOX_CEILING_20261112.md) · [冻结协议](code/DFINE_DEV705_FIXED_STOCK_COLLAR_20261112/PROTOCOL.md)

针对连续 IoU 阈值，以单位像元的真值覆盖计算增删对达标裕量的贡献。将可编辑范围分成三个互斥集合，框外两像素带可新增、原前景可删除、原框内背景可新增。每项只取能提高裕量的像元，相加给出相应权限下的精确几何上界。三个集合合并的上界逐样本复现扩框内自由选择结果。 [计算源码](code/DFINE_DEV705_FIXED_STOCK_COLLAR_20261112/score_stock_collar.py)

| 参照范围 | 明确未达标的原掩膜 | 只补框外达到阈值 | 补框外并可删除原前景 | 补框外并可新增框内像元 | 扩框内自由增删 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IoU 0.90，原框排除且扩框可达 | 134 | 0 | 26 | 17 | 134 |
| IoU 0.90，全部固定匹配中的明确未达标 | 1,327 | 5 | 645 | 294 | 1,255 |
| IoU 0.75，原框排除且扩框可达 | 14 | 2 | 8 | 8 | 14 |
| IoU 0.75，全部固定匹配中的明确未达标 | 441 | 76 | 409 | 245 | 438 |

在 134 个严格阈值样本中，117 个即使任意补入也需删除部分原前景，108 个即使可删除部分原前景也需补入某些框内像元。由两集合下界，至少 91 个同时需要这两类框内修正。放宽边界确实解除了一项几何约束，高 IoU 增益还依赖框内误差能否修正。上表使用真值选像元且无固定编辑量，数值仅为几何上界。 [实测](results/DFINE_DEV705_FIXED_STOCK_COLLAR_ACTUAL_SAFE_20261112.json) · [读标注前回执](results/DFINE_DEV705_FIXED_STOCK_COLLAR_PRELABEL_SAFE_20261112.json)

作者 D-FINE 推理先恢复原分辨率、以 0.50 二值化，再裁到检测框。其掩膜训练损失使用真实框范围。YOLACT 的实际推理裁剪在上采样前默认带 1 个原型格点的余量，单位和执行顺序均与本试验不同。BAIS 已训练可以越出候选框的掩膜表示。这里检验固定候选框与真实 SAR 轮廓上的编辑作用范围。 [D-FINE 推理](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/infer/torch_model.py#L295-L313) · [D-FINE 损失](https://github.com/ArgoHA/D-FINE-seg/blob/b782ece206487792f36c165ca65548154cd12094/dfine_seg/model/dfine_criterion.py#L473-L519) · [YOLACT 原型裁剪](https://github.com/dbolya/yolact/blob/master/layers/output_utils.py#L62-L84) · [YOLACT 裁剪余量](https://github.com/dbolya/yolact/blob/master/layers/box_utils.py#L293-L334) · [BAIS](https://arxiv.org/abs/1612.03129)

