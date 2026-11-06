# 掩膜排序、边界与栅格化：三项原始来源短记（2026-11-07）

状态：**有界研究假设，未执行新实验。** 只查两篇已发表原论文的公开摘要/作者项目及一个官方作者实现，不下载论文全文；没有读取本项目科学SOURCE、GT、权重、实际Ops或SSH，不新增训练臂、不改评分、不上传。

## 直接事实与研究问题

现有普通结果记录：固定R相对项目原生模型在整数GT下约+1.028 AP点；同一冻结预测在本地候选浮点GT下变为约−1.224点。N26保持R掩膜几何，仅用S0乘预测质量，三个下游头整数AP约69.424/69.637/69.551；浮点结果仍待本次实际诊断，新FULL的端点效用未知。这些数字来自既有[两GT报告](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/U57_INTEGER_FLOAT_GT_SENSITIVITY_ACTUAL_20261025.md)与[N26报告](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/N26_FIXED_R_QUALITY_THREE_SEED_RESULT_20261017.md)，本轮没有重评。

要区分的三件事是：实例轮廓改变；固定候选的分数排序改变；同一轮廓经不同标注表示/评价约定成为TP或FP。AP本身不能完成这三个机制的唯一归因。

## 本轮仅三项 primary source

| 原始来源 | 可支持的事实 | 对本文的约束 |
|---|---|---|
| Huang等，Mask Scoring R-CNN，CVPR2019 | 作者研究分类分数与掩膜IoU不一致，用实例特征和预测mask估计质量，提升高质量实例的排序 | N26已有直接原则先例。其固定几何增益可作为强排序对照，不能把乘分规则或IoU回归改名为创新 |
| Cheng等，Boundary IoU，CVPR2021 | 区域Mask IoU对不同尺度的边界误差响应不同；作者提出Boundary IoU及结合区域/边界质量的Boundary AP | 总AP提高不能直接写成边界改善。边界度量本身也是已知评价工具，仍依赖所选GT，不认证物理真值 |
| COCO API官方作者实现，固定commit `8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9` | polygon转RLE含离散化；segm匹配用RLE IoU并按score排序，area参与ignore/尺度范围 | 标注坐标表示、分数与评价约定必须分别控制，不能将三种作用合并解释成新结构有效 |

引用：Mask Scoring R-CNN [CVF原记录](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)、[作者实现说明](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/README.md)；Boundary IoU [CVF原记录](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)、[作者项目页](https://bowenc0221.github.io/boundary-iou/)；COCO API [固定实现](https://github.com/cocodataset/cocoapi/tree/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9)。本轮CVF HTML直接打开返回403，论文事实仅取CVF索引返回的原摘要并由作者README/项目页交叉核；未声称通读全文。

## 最有竞争力的替代解释

**整数标注量化与R离散输出的相容性，可能改变固定IoU阈值附近的匹配；N26又可能学到了该版本下的质量排序。** 这是已有机制足以解释局部AP收益的假设，不需要假定新边界结构已经有效。官方`rleFrPoly`接收浮点坐标后，在细化网格离散化并构造像素RLE；先逐坐标int截断再走该实现，一般不等于直接保留浮点坐标后再走同一实现。因此连续面积总量接近，不能推出像素轮廓或TP/FP相同。[固定polygon实现](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L162)、[annToRLE入口](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/coco.py#L383)。这是原作者实现的机制推断，不是本轮对项目mask或当前runtime的实测。

R的整数收益和N26三头的排序收益应肯定并保留；现有反转已经限制“跨标注版本稳定改善”的说法，但不能据此断言R总体无效、整数GT错误或浮点候选更接近真实船体。N26若浮点收益仍正，只支持该固定几何上的排序效用更稳健；不会变成几何创新。若不同，则需限定其标签/评价条件。FULL没有端点结果，以上不能替它预告正负。

## 一个最小后续对照：固定mask的bbox/area负对照

**等当前固定主实验终点后，Root再决定是否单独预锁这一项；本轮不执行。** 现有浮点派生同时恢复segmentation/bbox/area。为隔离评价metadata这一个混杂，建立一个仅用于诊断的GT副本：保留原整数segmentation、图像/annotation IDs与顺序、类别、crowd/ignore及其他字段，只从唯一浮点对应恢复bbox和area。原GT与主评价不改；只对同一冻结native与R预测各增加一次canonical segm COCO，不改score、候选、maxDets或RLE，不重训，不增加模型臂。

**可否定的窄假设：** 总体AP不变须先由未来实际任务闭合这些前提：全部GT area以及参与ignore的预测area均合法且处在默认all范围内；GT的ignore/crowd、类别、身份与顺序不变；predictions的完整内容、分数、候选顺序和RLE完全固定；IoU阈值/maxDets固定；实际canonical segm加载路径不受预测bbox残留影响（若既定合同要求加载前去bbox，沿用该合同，不临时改变原结果）。在这些前提下，仅换GT bbox/area，整体AP/AP50/AP75应与原整数评价完全相同。作者实现的segm IoU分支读取segmentation；area用于范围ignore，默认all为0至10¹⁰。尺度AP可以改变，不能把尺度项与总体AP混同。[segm IoU与排序](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L163)、[ignore/匹配](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L235)、[默认范围](https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L502)。

若满足前提仍改变总体AP，就否定“本次反转可只归入segmentation/栅格化路径”的简化解释，应先处理实际评价实现、加载或metadata交互，不能靠调模块解释分数。若总体AP不变，则排除bbox/area单独作用对总AP的解释空间；结合已有完整浮点反转，可把差异更具体限定到mask/匹配路径，仍无法判断物理轮廓真伪，也不能证明R跨表示更优。这里给的是一个定向负对照，不另铺训练矩阵或指标清单。

该预期来自官方固定作者代码，当前项目实际已选runtime是否逐项同义仍须未来独立Source闭合；本轮未读取其绑定科学SOURCE，也未把外部代码阅读当成当前执行资格。所有结果方向和失败均应保留，不选更有利GT、seed或后处理。此短记为自有普通研究文档，暂无新增AP或发表级成功结论。
