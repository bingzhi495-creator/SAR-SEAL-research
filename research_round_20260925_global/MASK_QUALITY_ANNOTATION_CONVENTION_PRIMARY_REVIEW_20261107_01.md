# Mask质量评分与标注约定：直接先例短综述

2026-11-07。有界primary-source复核；只读原论文和作者代码，未SSH、读科学载荷、复算指标或改训练。输入事实来自Root已核通知：固定R相对原生在整数GT为+1.028 AP，在同一冻结预测的候选浮点GT下为−1.224；N26三下游头整数AP为69.537±0.107，浮点评价尚未取得。这些不是本轮独立复验，也不是完整模型三种子。

| 原始论文（确切题名／venue／年） | 已有机制与对本项目的边界 |
|---|---|
| [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)，CVPR 2019 | 回归预测mask与GT的IoU以修正分类评分。与S0×qhat同属已知质量排序原则，不能仅改载体/名称就当核心创新；本项目S0也不应擅自等同原论文分类分数。 |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)，ICCV 2017 | RoIAlign消除RoIPool量化造成的空间错位，说明精细分割依赖坐标对齐。研究对象是特征采样，不能据此证明本项目GT浮点转换正确或收益反转由同一机制导致。 |
| [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)，CVPR 2021 | 边界指标补足Mask IoU对大目标边缘误差的敏感性不足，并提出Boundary AP。评价几何与整体AP应区分；换边界指标不能替代主COCO AP，也不能认证某套GT为真。 |
| [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.html)，CVPR 2019 | 边界标签松弛使语义分割训练更耐标注噪声与传播伪影。边界不确定性及其训练处理已有先例；该研究不是实例mask评分，也不是SSDD顶点栅格化诊断。 |
| [Noisy Boundaries: Lemon or Lemonade for Semi-Supervised Instance Segmentation?](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Noisy_Boundaries_Lemon_or_Lemonade_for_Semi-Supervised_Instance_Segmentation_CVPR_2022_paper.html)，CVPR 2022 | 对伪mask边界噪声，用低分辨率特征的容噪头及边界保持图分别处理。精细边界监督与噪声有权衡已有先例；伪标签噪声不等于本项目两套人工标注转换，不能直接套用其结论。 |

**实现证据。** Mask Scoring R-CNN作者代码固定提交`0e8fae677b8ab987e4cdd3e894f17c25410388d8`的[后处理L32](https://github.com/zjhuang22/maskscoring_rcnn/blob/0e8fae677b8ab987e4cdd3e894f17c25410388d8/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/inference.py#L32)直接以bbox分数乘预测MaskIoU。COCO官方提交`8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9`的[`rleFrPoly` L162–190](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L162)接受double顶点，显式上采样、离散边界和下采样。先截断顶点与保留浮点属于不同输入约定，不能先验认为栅格结果相同；本轮没有检查本项目实际调用ABI，更不能因此宣称浮点是真值。

**当前可支持的解释与下一证据。** R收益反号支持“效用对评价标注约定敏感”，不唯一证明过拟合、亚像素修复或整数GT错误。质量评分的IoU监督也可能受约定影响，但N26浮点效用仍未知。应保持全部冻结预测和评分、三个既定头、固定人口与评价规则，在主整数GT及候选浮点诊断GT下统一报告原生／R／各N26的AP、AP75和配对差值；几何及评分诊断沿用既有匹配与适用范围，不能把非线性的AP差值拆成唯一因果贡献，或事后挑GT/头作为主结果。

检索覆盖mask-IoU scoring、boundary annotation noise、subpixel alignment与polygon rasterization的CVF论文及COCO作者实现；本轮未核到专门论证本项目这类SSDD整数截断使精修AP收益反号的直接论文。该有限检索不支持“领域无人做过”或“发现普适新问题”。上述已知方法、代码约定和本项目待检验假设应分别陈述；本轮没有新增模块、效用或创新成立结论。
