# SSDD 原生掩膜网格的同点梯度抵消诊断

已闭合的切向灰度对应探针低于固定轴对照。新假设针对原生空间读出。小船附近的真阳性补入与背景移除会向粗网格同一节点传递相反方向的梯度，独立细网格可能降低这种局部抵消。固定 E24 的原候选、原匹配、原图二像素带及真实有符号错分标签。复用 111 张原 FIT 留出图，比较 248 个原匹配小目标及其中前十候选里的 229 个，逐图配对。 [上轮结果](RFDETR_E24_FIT_CONTOUR_CORRESPONDENCE_PROTOCOL_20261109.md)

在相同稀疏位置和预测 logit 计算 BCE 所需更新，按原生 `align_corners=False` 半像素映射和边缘夹紧构造四点插值。比较 144×144 原生空间与 288×288 独立节点的反事实。固定原图点和标签，先以插值转置聚合更新，再把网格更新读回同一位置。抵消率等于 1 减去聚合后绝对值之和与点级绝对值之和的比。报告错分 BCE、单位符号和全支持 BCE，并按补入、移除及原正确点报告响应方向。局部同单元扫描线标签反转仅作计数。 [原生 RF-DETR 掩膜头](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/heads/segmentation.py)

前十范围锁定推进条件如下。

- 至少 20 张图包含同实例双向错分。错分 BCE 抵消率的图像配对降幅至少 0.05 且区间下界大于零。
- 全支持 BCE 下真实错分的非有益响应率下降，图像配对区间下界大于零。补入和移除的分项均不恶化，原正确像素的反向更新率不升。
- 统一使用 2,048 次图像配对重采样和种子 5201。通过后才比较 2×2 子像素残差头与相同四个学习分支池化回原生网格的控制，保持原匹配、相同支持及 BCE/Dice。共享参数仍可能重新引入抵消，需实际训练和同容量比较。

已有 DUpsampling、Mask DINO 和 RefineMask 覆盖学习上采样、高分辨率查询读出和细粒度掩膜。当前检验限于具体 SAR 错分下的同点更新方向。六项合成测试通过。模型、权重、空间张量、DEV/TEST 与 AP 均在本轮诊断范围外。 [DUpsampling](https://openaccess.thecvf.com/content_CVPR_2019/html/Tian_Decoders_Matter_for_Semantic_Segmentation_Data-Dependent_Decoding_Enables_Flexible_Feature_Aggregation_CVPR_2019_paper.html) · [Mask DINO](https://arxiv.org/abs/2206.02777) · [诊断源码](code/RFDETR_E24_FIT_NATIVE_GRID_CANCELLATION_20261109/native_grid_cancellation.py)
