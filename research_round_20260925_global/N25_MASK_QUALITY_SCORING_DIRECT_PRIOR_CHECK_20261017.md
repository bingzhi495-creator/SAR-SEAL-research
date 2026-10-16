# N25 掩膜质量评分：有限直接先例核查

2026-10-17。状态：有限原始来源核对完成；候选基线未据本文实施或运行。本次未读取科学载荷、权重、缓存或资格链，没有新模型结果。N25实际结果由Root另行接收；本文不能替代其执行证据。

## 1. 已有直接机制

**Mask Scoring R-CNN，Huang等，CVPR2019。** [官方论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf)提出以实例特征和预测mask回归mask IoU，并用分类分数乘预测IoU。已取得原文方法段及[作者库](https://github.com/zjhuang22/maskscoring_rcnn)中以下相关实现，不声称本轮完整通读论文或审计全库：

- `roi_heads/maskiou_head/roi_maskiou_feature_extractors.py`：ROI特征与池化后的预测mask拼接，4层卷积及FC处理。`mask_head/loss.py`返回的输入mask为sigmoid概率；不能把训练target二值化误写成作者代码的输入也必为二值mask。
- `mask_head/loss.py`、`maskiou_head/loss.py`：质量target来自预测二值mask与对应匹配GT；质量损失只对正proposal类别计算L2，空正样本返回连接图的零损失。原文正proposal条件是proposal box与匹配GT box的IoU大于0.5，不是用mask IoU大于0.5筛去坏mask。
- `maskiou_head/inference.py`：分类分数乘预测MaskIoU。负或未匹配proposal不在该质量损失中被统一伪标IoU0；它们的分类监督与质量回归必须分开理解。匹配GT target也不等于N25用于诊断的max-any-GT量。

**Mask2Former。** 已读[作者推理实现](https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py#L314-L345)及同库`modeling/criterion.py`相关方法。原版实例评分已是类别概率乘前景内平均mask概率，后者不是独立监督的IoU头。criterion通过二分匹配确定mask监督对；未匹配query的分类目标为no-object，mask BCE/Dice只取匹配对。本次是作者代码相关路径核对，不是全文或全部配置复现。

**MaskDINO。** 已读[作者模型代码](https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/maskdino.py)和同库`modeling/criterion.py`。实例评分同样包含前景内平均mask概率；`focus_on_box`分支会令该因子为1，不能脱离配置概括。常规分类使用focal目标，未匹配query没有前景正类；mask损失仍取匹配对。所查实现不能被笼统称为独立mask-IoU质量头，box GIoU损失也不是mask质量评分监督。未作全文复审。

**SAR直接先例：EddyDet，Remote Sensing2023，15(19):4752。** [出版社原文](https://www.mdpi.com/2072-4292/15/19/4752)方法§3.1.3已返回并核读：输入ROI特征和预测mask，回归二值mask与匹配GT的IoU，采用L2，分类分数乘质量量。该任务是SAR海洋涡旋，不能把其结果当船舶可比收益；但足以否定“首次将MaskIoU评分用于SAR”的宽泛说法。未核作者代码或其未匹配样本处理，不能由文字补全这些实现细节。

## 2. 当前S0已经包含质量因子

当前使用的S0形式是class概率乘平均mask概率，与上述作者默认评分形式相符；这不证明代码逐字节等同。新增质量头必须与完整原S0比较，不能只胜过裸class分数就声称修复了原评分基线。

同一个学习质量量qhat有两种不同机制：

| 评分 | 含义 |
|---|---|
| class×qhat | 以学习质量替代原平均mask概率，接近标准Mask Scoring分解 |
| S0×qhat | 保留原maskprob因子后再次加权质量，属于追加校准 |

两者需事先指定主次并同时完整保留，不从结果挑最高。必须固定全部100query、原mask/类别/身份和导出规则，只有score改变；不能同时改mask、候选预算、去重或骨干。N25的GT oracle不等于任何质量头已学到的能力，也不是数学AP上界。

## 3. 唯一后续候选与前提

仅当N25实际接收的结果支持质量重评分存在值得检验的回收机会，才考虑一个**冻结R的标准MaskIoU后处理基线**：用推理可得的预测ROI特征和预测mask回归质量；FIT目标使用固定真实matcher对应GT，不以DEV GT训练、选模型或调评分函数。保留低质量匹配，未匹配query不伪造真实质量0；全部100query仍须推理和审计。ROI/空mask处理、损失、预算和评分主次须在训练前另锁。

这是查询载体上的冻结适配，与原Mask Scoring R-CNN端到端联合训练有差别；不能冒称忠实复现全部原配方。标准MaskIoU及其SAR应用已存在，新增head、换数据域或再次相乘本身不构成创新。真实可学习性、部署收益及成本仍待检验，本文不预选更复杂结构，也不改变N20/N23已锁停止规则。

## 核查限制

本次仅定向检索和核对上述直接机制，不是2024–2026全面查新；未确认范围不等于不存在先例。在线作者分支未固定commit，也不能替代本项目未来Source闭包。Codex MCP跨模型审阅工具未提供，状态为pending；不称跨模型审计或正式完整性审计。本文无实验效用、新颖性成立、SOTA或投稿就绪结论。
