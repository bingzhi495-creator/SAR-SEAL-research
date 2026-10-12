# N06 错误分析与方法查新：先例边界

日期：2026-10-12。只记录公开先例与设计约束；本次误差结果尚未读取，候选瓶颈与创新未成立。

| 先例 | 已有内容与当前核验范围 | 对本研究的约束 |
| --- | --- | --- |
| [TIDE, ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480562.pdf) | 原论文及作者工具页说明，可直接用实例分割输出分析误差，并估计修正错误后的AP变化。本轮查原论文公开文本和工具说明，未运行其软件 | N06的固定COCO匹配与数量分类不是TIDE反事实修正，不称AP因果贡献分解 |
| [PointRend, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) | 原论文已有细特征与粗预测结合、自适应不确定点及迭代边界精修；核验公开论文摘要、首节及作者方法说明 | 自适应点和边界精修本身不能作为新颖性，值得进行精修时必须作为强简单替代 |
| [动态关键点SAR实例增强, JSTARS2024](https://napier-repository.worktribe.com/output/3687229/sar-ship-instance-segmentation-with-dynamic-key-points-information-enhancement) | 论文DOI10.1109/JSTARS.2024.3383779；公开全文方法按实例几何动态关键点构造Gaussian attention并动态预测mask。协同查新读公开论文方法；所检描述未明确跨query逐像素互斥，但不能据此证明不存在 | 船体形状和缓解密集mask重叠已是直接先例；须定位具体运算差异 |
| [DiffSARShipInst, ISPRS JPRS2025](https://www.sciencedirect.com/science/article/pii/S0924271625000887) | 出版社摘要和[作者loss.py](https://raw.githubusercontent.com/XiaowoXuSAR/DiffSARShipInst/main/diffsarshipinst/loss.py)、[detector.py](https://raw.githubusercontent.com/XiaowoXuSAR/DiffSARShipInst/main/diffsarshipinst/detector.py)核验；动态1-to-k匹配、query mask预测与可选box NMS。未声称已阅读论文全文或复现作者方法 | 固定候选生成实例mask已有先例；不能把动态1-to-k写成一对一，不能把所检独立mask路径推广为作者绝无实例交互 |
| [Mask2Former作者源码](https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/maskformer_model.py) | 所检instance推理独立阈值化top-k masks，panoptic推理已有score-weighted像素argmax；[matcher](https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/modeling/matcher.py)包含一对一Hungarian | 一对一训练匹配不自动消除推理掩膜重叠；简单像素归属已是必要替代，不能宣称首次跨实例竞争 |

下一设计只能由实际误差决定。若高IoU几何不足占主要可改善部分，检验是否存在可利用细特征、普通PointRend式精修能否解释收益；若实际匹配竞争明显，先检查GT允许重叠情况并比较简单argmax/后处理。漏候选主导时，mask-only设计的效用上限应收缩。以上均是条件建议，不是已证瓶颈、创新或外部SOTA结论。

独立父场景配对不是当前DEV或HRSID官方划分已具备的条件；来源未知时按实际图像/切片身份限定结果。外部测试不用于上述选择。
