# N07 原图局部细节与实例掩膜精修：定点直接先例核查

日期：2026-10-12。范围：截至本日的五项最接近工作，仅依据原论文、作者机构/项目页和作者公开代码。本文是普通文献记录；没有读取本项目科学载荷、缓存、标注、权重或数值结果，也没有启动训练。文中“作者报告”不是本项目的复现结果。

## 对当前问题的约束

N07 的预定对比是在冻结候选、原生分数和原生掩膜支持的条件下，比较额外 49 个通道为零的点头 C 与输入真实 7×7 原 JPEG 灰度 patch 的点头 R。此对比若成立，首先支持的是**在该载体、采样和预算下，额外原图信息相对空通道的效用**。C 和 R 即使名义参数量相同，零通道也不提供同等活跃输入；结果不能单独证明 SAR 专有纹理机制、边界选择规则或新颖性。N07 的实际执行状态和数值结论只以任务追踪及合格作业记录为准。

| 工作与状态 | 已核输入/机制、作者实验范围 | 可复现接口及本项目适配身份 |
| --- | --- | --- |
| **PointRend**，CVPR 2020，正式发表。[官方论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)；[作者 Detectron2 项目](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/README.md)；[作者 mask head 源码](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/mask_head.py)。 | 以粗 mask logit 的接近零程度选择不确定点，结合细粒度特征逐点重判并逐级提高分辨率。作者论文实验覆盖 COCO、Cityscapes 的实例/语义分割。原图信息与不确定点精修都已有先例。 | 作者项目提供配置、训练/评价入口和代码。移入冻结 E24 候选时，只能称 **PointRend 风格的受控适配**；完整作者模型的骨干、RoI、训练和推理不等同于本项目。 |
| **Boundary Patch Refinement（BPR）**，CVPR 2021，正式发表。[官方论文](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)；[作者代码](https://github.com/chenhang98/BPR)。 | 从已有实例预测的边界裁取原图高分辨率小 patch，以后处理网络精修；作者在 Cityscapes 和 COCO 报告边界/掩膜结果。这是 N07“原图局部 patch 修边界”的最直接普通视觉强替代。 | 作者 README 给出 COCO 样式粗分割 JSON 与图像到 patch 数据、训练及 refined JSON 的入口。移植到 SAR 时必须固定原候选、分数、训练人口及预算；测试真值只能用于评价，不能进入掩膜生成。 |
| **SharpContour**，CVPR 2022，正式发表。[官方论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_SharpContour_A_Contour-Based_Boundary_Refinement_Approach_for_Efficient_and_Accurate_CVPR_2022_paper.html)；[作者项目页](https://xyzhang17.github.io/SharpContour/)。 | 从粗掩膜轮廓出发，实例动态参数的点分类器使用高分辨率特征判断采样点对该实例的内/外，沿轮廓法向离散、迭代地更新顶点。作者在 COCO、Cityscapes，以 Mask R-CNN、CondInst、DANCE 等输出试验。实例条件、法向搜索和点级可靠判别均不能作为首次提出。 | 作者项目页截至本轮仍标注 GitHub “Coming soon”；已核到论文/方法说明，**未核到可运行作者源码**。若实现同类控制，应标为依据论文的受控重实现，不能标成官方复现。 |
| **Gao 等，SAR Ship Instance Segmentation With Dynamic Key Points Information Enhancement**，IEEE JSTARS 2024，正式发表。[作者机构页与全文入口](https://napier-repository.worktribe.com/output/3687229/sar-ship-instance-segmentation-with-dynamic-key-points-information-enhancement)，DOI [10.1109/JSTARS.2024.3383779](https://doi.org/10.1109/JSTARS.2024.3383779)。 | 针对密集近岸漏检、掩膜重叠与杂波，将强散射区域编码为动态关键点，并生成每实例二维 Gaussian 注意图辅助动态 mask head；作者实验使用 PSeg-SSDD 与 HRSID。它直接限制“实例条件 + SAR 强散射点改善 mask”的新颖性表述。 | 机构页可追溯出版信息、摘要与论文 PDF。本轮**未核得作者可运行代码**；可以做论文机制比较，不能凭未运行的文字描述把本地模型写成已胜此法。 |
| **DACF-Net**，International Journal of Digital Earth，2026-09-08 在线发表。[出版页](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)；[作者仓库](https://github.com/yang199809/DACF-Net)。 | 多尺度大核聚合、跨层语义引导及几何感知候选构成渐进特征精修；作者在 HRSID、PSeg-SSDD 报告实例分割、消融与效率变体。其主要改的是全模型特征与候选，**不是**固定 query 后的点精修。 | 截至本轮作者仓库只有 README/许可等占位内容，README 说明实现仍在整理，**不能列为已可运行的作者基线**。论文可作同期竞争方法和差异核查；若重实现，要单列适配身份。 |

这五项作者结果与本项目 N07/E24 不能直接排名：数据集与具体标注版本、父场景/切片划分、预训练、骨干/输入分辨率、候选生成、训练预算、分数与后处理、评价实现均未统一。作者方法完整复现与同一 E24 冻结载体上的机制控制应分列；光学数据上的改善不等于 SAR 改善。近期的 [O²Former 2025 预印本](https://arxiv.org/abs/2506.11913) 已有 [IEEE JSTARS 2026 正式刊载记录](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/)（DOI 10.1109/JSTARS.2026.3677088）；它主要增强 query 生成/方向表示，故未列为固定候选点精修的直接控制，不应再称“仅预印本”。

## 唯一条件候选及否证

**只在 N07 完整公平 C/R 端点提供正面信息证据后考虑。**保持 E24 query、原生分数与带外输出不变，在原生掩膜窄带内，以其 logit 梯度给出局部法向；把同一个 7×7 JPEG 灰度 patch 分成法向内/外成对采样，计算局部归一化差异和两个采样尺度的方向一致性。点头输出有界 logit 残差及“保持原值”的可靠性门；只修正有足够一致证据的点。此运算差异是**显式方向配对、局部可靠性判断和允许拒绝修正**，不是再拼接一个原图 patch。JPEG 灰度不等于定标后向散射或复数相位；本假设不预设船体边界必然更亮。

预锁强替代：N07 R 原 patch 点头；使用同样活跃输入维度、参数量、更新/样本访问和 FLOPs 的普通点头/普通门控；按固定候选改造的 PointRend 式不确定点控制与 BPR 式边界 patch 控制。若改为顶点法向位移，还必须直面对 SharpContour 式实例条件点分类。比较时保持全部 query、分数、支持区域及主评价不变，同时报告 mask AP、预锁高 IoU/边界诊断、像素 FP/FN、额外成本与完整种子。只看 C 的零通道对照不足以排除活跃容量。

**否证条件**：普通点头/门控在同预算下追平，或仅改采样/增加活跃容量即可解释收益；高 IoU 几何与主任务效用不共同支持；收益只在单一开发划分或少数种子出现；标注边界歧义足以解释高阈值落差。出现这些结果时，保留相应工程/信息结论，撤回“可靠性机制必要”与新颖性主张。即使条件候选成功，本次定点检索也不足以证明首次提出，仍需投稿前更新直接先例核查和公平外部验证。
