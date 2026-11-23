# 实例分割与边界精修的原文图文抽样

2026 年 11 月 23 日。逐篇打开了 18 篇论文的原始 PDF，每篇至少渲染并检查了一处关键图表，另核对相关正文或图注。页码是所链接 PDF 的页序。每条只提炼可在本稿落实的构图或论证方式。图例、数据、配色和原作者的主张不移植。

## 边界问题、支撑域与对照

1. **PointRend，CVPR 2020。** PDF 第 3–4 页 Fig. 3–5 将粗预测、点选择、点特征和更新分别呈现。Fig. 4 用 4×4 到 8×8 的同一网格留下黑点标明实际重预测位置。第 7 页 Fig. 7 使用同一飞机区域展示分辨率递进，旁接 Table 3 的采样量和分辨率消融。正文先提出稠密网格与局部细节之间的冲突，再分别解释推理和训练抽点。**本稿做法。** 原生栅格模块用同一像素格展示查询与写回，并画清内插只为查询提供特征。 [论文原 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)
2. **Boundary Patch Refinement，CVPR 2021。** 第 2 页 Table 1 先用边界内不同误差范围的真值替换测量可修复余量。第 4 页 Fig. 2 复用同一车实例贯穿选窗、筛窗、分支输入和重组。第 5 页 Fig. 3 与 Table 2 并置，同一个局部按真值、原输出、带掩膜补丁和去掩膜补丁排列。正文随图解释为何去掉掩膜补丁时相邻实例难分。**本稿做法。** 版面上把同预算的实际消融表与同坐标局部对照排在相邻位置。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.pdf)
3. **SegFix，ECCV 2020。** 第 3 页 Fig. 2 把三种分割器的错误像元按到边界的欧氏距离画成并排直方图，横轴统一。随后的相关工作依距离与方向图、水平集、DenseCRF、精修分段，每段说明其处理对象。**本稿做法。** 若稿件有实际的边界距离统计，用不同载体上的同一物理量支撑问题诊断。只有视觉例时，明确样本与选例口径。 [论文原 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf)
4. **SharpContour，CVPR 2022。** 第 3 页 Fig. 2 上排显示法向、移动方向、移动步长和偏移的局部细节，下排维持长链路。第 6 页 Fig. 4 以 DANCE 和 Mask R-CNN 为两行，各样例在同一视野左右给出处理前后结果。§4.2 说明评价指标与更严格标注的使用。**本稿做法。** 机制图先固定一个对象和参考轮廓，局部视图仅增加有用的符号。跨检测载体的定性证据用一行一载体。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_SharpContour_A_Contour-Based_Boundary_Refinement_Approach_for_Efficient_and_Accurate_CVPR_2022_paper.pdf)
5. **RefineMask，CVPR 2021。** 第 3 页 Fig. 2 保留 FPN 和各级掩膜输出，检测头在图注交代，推理专有连线用紫色虚线。第 4 页 Fig. 3 展开四路融合，Fig. 4 用同一猫轮廓定义距离变量。第 8 页 Fig. 6 每列是同一 COCO 示例，上中下对应原方法、精修和参考。**本稿做法。** 整图用图例说明冻结载体与主实验有效通路，细图定义中间量，让同一符号在公式、图和正文一致。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.pdf)

## 把变化在总图、子图和公式上对齐

6. **Mask2Former，CVPR 2022。** 第 7 页 Table 4 的五个小表统一 AP、PQ、mIoU 和 FLOPs 表头，逐组交换解码器部件。Table 5 则交叉匹配用点或掩膜与训练监督用点或掩膜。§4.4 邻接段落分别解释稳定收益与资源代价。**本稿做法。** Phoenix 支撑域与写回规则用同一 2×2 对照，表中每轴只有一个实验变化，等量增删对照独立指明被固定的逐候选量。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)
7. **Segment Anything，ICCV 2023。** 第 1 页 Fig. 1 以任务、模型、数据三幅图回应引言的三个问题。左图直接给提示和有效掩膜，中图交代编码与解码，右图才画训练和标注循环。**本稿做法。** 论文首图给原生粗掩膜、可编辑端点与输出三件可见对象。细部机制图再解释排序器和写回，贡献之间用数据依赖说明。 [论文原 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)
8. **SAM 2，ICLR 2025。** 作者版第 4 页 Fig. 2 对齐同一段视频的四帧和两次操作，以同一框和箭头标记初始提示、传播和补充提示。第 5 页 Fig. 3 另画当前帧、提示与记忆的关系。§4 由单帧预测开始，逐步引入时序条件。**本稿做法。** 固定一张 SAR 船体示例做对照，不同方法共用来图、框和输出坐标，按动作画局部的先后状态。 [作者版 PDF](https://arxiv.org/pdf/2408.00714v2) · [会议版](https://openreview.net/pdf/7c41968163abe4e3700e3e3a15174a9d679fcd52.pdf)
9. **Mask R-CNN，ICCV 2017。** 作者版第 6 页 Fig. 6 同一配对场景上下摆放 FCIS+++ 与 Mask R-CNN，图注明示重叠目标伪影。Table 2 各小块检验不同问题，§4.2 用相邻小段解释这些对照。**本稿做法。** 失败案例用同一个船际重叠或邻近强散射区域，描述哪个船体被写入或漏掉。结果分析按具体错误展开。 [作者版 PDF](https://arxiv.org/pdf/1703.06870) · [会议条目](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
10. **SOLOv2，NeurIPS 2020。** 第 4 页 Fig. 2 将 SOLO 与 SOLOv2 两幅结构画在相同坐标尺度，保持输入和最终掩膜形状相同，用同一位置突显独立产生掩膜特征和卷积核的变化。§2.1 的 Eq. (1) 就地定义核、特征图和实例输出。第 5 页 Eq. (2) 再给训练损失。**本稿做法。** 左右两种固定排序只改变被优化的可行集，复用端点、分数和配额图形，方程紧随变量定义。 [论文原 PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/cd3afef9b8b89558cd56638c3631868a-Paper.pdf)
11. **CondInst，ECCV 2020。** 第 6 页 Fig. 3 用一致颜色区分 FPN 层、检测头和共享掩膜特征，生成的多个实例输出只重复实例相关分支。图注逐一解释特征图与相对坐标符号。正文在图下解释动态实例滤波如何承接目标检测。**本稿做法。** 点头细图只画一份共享特征和一个端点批次，为 (N\times d) 输入、逐点分数及外侧和内侧分流标明尺寸。 [论文原 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460273.pdf)
12. **QueryInst，ICCV 2021。** 第 8 页 Fig. 4 以上下排列比较 FPN 特征和经查询增强的实例特征，末行提供参考掩膜。Tables 4–6 把头部结构、平行监督和共享查询分别检验。正文紧接各表解释效应大小与耗时。**本稿做法。** 机制可视化采用同一对象、同一网格上的分数和参考占据率两列，配上对应的消融表。使用真实分数时标明共享范围和取例规则。 [论文原 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Instances_As_Queries_ICCV_2021_paper.pdf)

## 度量定义和结果图

13. **Boundary IoU，CVPR 2021。** 第 5 页 Fig. 5 将参考轮廓、预测轮廓、保留的窄带及分子分母逐项绘出，Eq. (1) 给出同一运算的符号。同行 Fig. 4 与正文展示不同误差度量随物体大小变化的响应。**本稿做法。** 评价小图把船内交集、船外误分和船内漏分落在同一参考多边形上，图中的区域直接对应连续 IoU 公式。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf)
14. **Mask DINO，CVPR 2023。** 第 5 页 Fig. 1 用蓝色区域画原 DINO，红色线路标注新增分割通路。第 4 页 Tables 1–2 用相同 50 epoch 预算呈现直接加检测头与直接加分割头的局限，然后才解释其联合架构。**本稿做法。** 用冻结分数和原生掩膜作公共底图，把本稿更改的可编辑位置和动作单独高亮。与强基线的直接替换应留下低于原方法的结果。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.pdf)
15. **HQ-SAM，NeurIPS 2023。** 第 10 页 Fig. 6 同图对照原始 token 与 HQ token 的注意区域，Fig. 7 同提示和裁切上下排列，用框标出细结构或破洞。Table 9 同页列不同适配方案的区域指标、边界指标和可训练参数。**本稿做法。** 定性页各列记录相同的候选输入，保留同尺度窗和类似的背景透明度。若绘热点，对齐标尺并提供参考或实际错误，避免把评分热度解释成精度。 [论文原 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/5f828e38160f31935cfe9f67503ad17c-Paper-Conference.pdf)
16. **SegRefiner，NeurIPS 2023。** 作者版第 2 页 Fig. 1 三行分别呈现明显误分、漏分与细节遗漏，四列为原图、粗预测、既有精修和新精修。紧邻正文把三类错误引入多步像元状态更新的研究问题。**本稿做法。** 本稿按新增掩膜污染、船体丢失及端点受限分行安排真实例。图注明确已经核验的真值、变化方向和取例规则。 [作者版 PDF](https://arxiv.org/pdf/2312.12425) · [会议论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Paper-Conference.pdf)
17. **CascadePSP，CVPR 2020。** 第 8 页 Fig. 14 为每组固定四列原图、参考、输入掩膜和精修结果，整图下一行是同窗放大，窗位置在各列可追踪。图注列明每组粗预测模型。**本稿做法。** 优先放每个港口或船体的一张完整背景图，然后逐方法提供同窗放大。用图例分配轮廓和新增、删除色，参考与变化不可混着标。 [论文原 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.pdf)
18. **SAM-REF，CVPR 2025。** 第 8 页 Fig. 4 给出粗掩膜、误差图、修复掩膜的整图和局部两行。Table 4 改写合并规则，正文逐项解释全局影响和局部变化，还指出局部结果仍可能不如期望。**本稿做法。** 精修组件展示同一对象的变化图和输出，保留负效应例，定量消融同时报告全量 AP 与冻结匹配船体的连续变化。 [论文原 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_SAM-REF_Introducing_Image-Prompt_Synergy_during_Interaction_for_Detail_Enhancement_in_CVPR_2025_paper.pdf)

## 用于本稿的版式决策

- 方法图 2 将坐标、正负候选、连续占据目标与点头分数沿同一工作顺序展开。原生栅格上直接标明抽样点，图中同时解释双线性内插的对象和最终写回的栅格。点头尺寸从实现和稿件核对后才上图。
- 方法图 4 用一份 Phoenix 分数场统领支撑域与写回规则的 2×2 设计，第三个狭域同量比较接在矩阵下方。示意图不填实际 AP 或假定效果，结果表另外沿相同对照轴阅读。
- 定性结果的每一行只使用一份原图及固定裁切，沿两种冻结载体共用原生、参考、同量对照和变化图的列位。主图包含一个改善和一个回退实例，并报告预先确定的取例方式。
- 实验叙述依次交代数据、分割指标、可比输入和运行预算，再分别回答整体变化、机制变化和失败情况。地物、阈值、分组和效应量直接进入句子。
