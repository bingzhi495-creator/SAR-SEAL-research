# N25 固定掩膜GT质量评分诊断：执行前计划

日期：2026-10-17。状态：**PLANNED_NOT_IMPLEMENTED_NOT_EXECUTED**。本文件只预锁诊断，不是实现、运行或资格证明。N23按原门槛STOP，不增加seed、调参或融合；本诊断不追认为其主结果补救。

## 问题与固定人口

N24已由Root接收：187图380GT均有效，crowd/ignore为0；五组十阈值、top10/100的最大匹配减原贪心匹配均为0。该结论只排除这些切片内的匹配基数损失，不排除TP/FP在score序列中的位置损失，也不说明所有几何错误已解决。N25问：保持所有掩膜不变，显式使用GT质量重评分能回收多少原AP？

- R为主诊断；五组固定为native、冻结强R、A_COMMON、A_ACTIVE、B_C，全部保留，不选最高组作主。使用已完成的同一DEV187、官方380GT，每组18700个自然query。
- 原图像/类别/标注顺序、mask RLE、query身份、预测原始ordinal及原S0冻结。S0指原已导出的完整分数，不重新用精修mask计算S0，也不将其冒称纯分类概率。
- 不读取全量L/F或模型权重，不前向、训练、改选区/预算、去重、NMS、改mask或删query；不把未匹配query当真实背景。Root执行前另锁输入strictrefs和五组原COCO12。

## 质量量与三种评分

对每个预测d，以其原mask与同图同类别的**全部GT**计算原始mask IoU，定义q(d)=max_g IoU(d,g)。无同类GT或预测mask为空时q=0；合法输入下q须有限且在[0,1]。保留argmax GT/并列情况供审计，不能把最大重叠认定为模型语义身份或原matcher。当前crowd/ignore均0；若执行输入不符，该身份差异须先停止，不能静默改GT候选集。

| 名称 | 导出score | 预锁地位 |
|---|---|---|
| ORIGINAL_S0 | 原S0 | 五组共同基线，复现已有指标 |
| GT_PRODUCT | S0×q | 主oracle；首要比较R的GT_PRODUCT−ORIGINAL_S0 |
| GT_IOU_ONLY | q | 次控制，诊断乘积是否仍受S0影响 |

乘积采用明确FP64标量计算；不加epsilon、不额外裁剪、归一化或score过滤。保留原输入列表，使用官方稳定降序排序；分数并列维持原ordinal，不借GT身份重排tie。两种新分数都使用GT，所有输出显式标ORACLE_DIAGNOSTIC。两oracle独立完整报告，禁止取最大、混合或改名为主结果。

## 执行与拒绝条件

1. 核官方GT、全部五组及每图100query身份与原ordinal。每个新预测除score外与对应原字典逐项完全一致，含RLE、image/category/query。RLE可为计算IoU解码，禁止以重编码掩膜替换原输入。
2. 沿原official.coco_metrics/COCOeval运行完整187图；segm、maxDets=[1,10,100]、原面积及crowd/ignore语义、无新增score过滤。三评分×五组均保留完整COCO12和0.50:0.05:0.95十项AP；不能平均分片AP。
3. ORIGINAL_S0五组12项stats必须精确复现原结果，任一不符立即停止并保留失败原因。全部100候选及top100几何IoU矩阵、每GT最佳IoU/可达阈值计数须不变；这些比较按query身份对齐，不能按新排名位置误配。top10组成和AR10可因重评分变化，单独解释，不要求其不变。
4. 不在结果后追加“明显收益”数字门槛、选择新权重或搜索评分函数。实现、独立验证、真实原终态与Root接收另行完成；本计划不自动授权科研结论。

## 结果与下一步解释

输出输入/源码身份、固定mask与query检查、各预测原S0/q/两个oracle分数的小型审计、15组COCO12、逐阈值AP及相对各自原S0的百分点差。R乘积oracle的完整AP差为主；五组同格式，次控制不能挽救主诊断。top100覆盖不变而AP改变，仅是固定掩膜的排序效应，不是几何提升。

若主oracle有回收，仅支持下一步考察标准MaskIoU质量头的可学习性与成本；FIT监督/匹配GT目标、未匹配样本、初始化、预算及独立验证须重新预锁，不在DEV用GT训练/选型。若仅IoU-only有效，提示乘积仍可能受S0影响，不证明唯一原因。若两者有限，也只限制这两种函数，不能断言score无关或必须转整网。GT评分不是可部署指标或数学AP上界，不证明新方法、创新、跨集收益或SOTA；不改变N20/N23停止规则。

## 直接先例与证据边界

- Huang等，Mask Scoring R-CNN，CVPR2019：[官方PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)、[作者库](https://github.com/zjhuang22/maskscoring_rcnn)。作者实现说明以实例特征和预测mask回归mask IoU，处理质量与分数不一致。N25只做GT质量诊断，不宣称已复现该可学习模块；max-any-GT量也不自动等于其训练匹配目标。
- Kirillov等，PointRend: Image Segmentation as Rendering：[原论文](https://arxiv.org/abs/1912.08193)、[作者实现](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)。其自适应点预测用于精修分割；不能把N25固定mask重评分写成新的PointRend几何精修机制。

本轮已取得两作者库说明及PointRend原论文摘要；CVF PDF入口由作者库确认，但本轮返回403，未重新阅读全文。不引用外部数字作为本项目公平比较，也不复制论文或原始标注。
