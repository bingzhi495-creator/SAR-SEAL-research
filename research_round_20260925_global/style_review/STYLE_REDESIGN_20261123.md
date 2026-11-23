# 52 篇论文的图表与写作研读及本稿改版

研读样本覆盖 TGRS 的 SAR、遥感检测与分割论文，以及 CVPR、ICCV、ECCV、ICLR、NeurIPS 等来源的视觉和实验方法论文。共核看 52 篇原文的关键版面，其中 13 篇发表于 TGRS。逐篇笔记列出所看的图、表、公式或段落和本稿可采用的做法。采用 51 份研究原稿 PDF 和 1 篇出版社官方全文页面。

| 评阅组 | 样本 | 逐篇记录 |
| --- | ---: | --- |
| TGRS 和相邻遥感论文 | 17 | [雷达、遥感与版面](TGRS_STYLE_SAMPLE_20261123.md) |
| 视觉实例分割与精修 | 18 | [分割机制、定性图和对照](CV_INSTANCES_STYLE_SAMPLE_20261123.md) |
| 检测、稳健性与数据报告 | 17 | [公式、训练配方、统计与数据资格](PROTOCOL_REPORTING_STYLE_SAMPLE_20261123.md) |

## 改版在稿件中的位置

| 现稿问题 | 本轮采用的做法 | 研读依据 |
| --- | --- | --- |
| 总览和局部图重复同一个选点过程，四张大图挤压正文 | Fig. 1 只串联原生掩膜、合法位置、占据率和写回。Fig. 2 用单栏展示训练多边形面积标签及推理特征查询，Fig. 4 用单栏展示固定 Phoenix 响应与编辑规则。用图元形状、实虚线和颜色共同标识增删。 | [Han 等 Fig. 4](https://arxiv.org/pdf/2008.09397#page=5) · [RT-DETR Figs. 4–5](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf) |
| 固定配对与独立选点的差异需要在同一网格上核对 | Fig. 3 保持原生掩膜、端点和每侧编辑次数一致。列出两种选择、占据率得分、原区域与输出，图内数值由同一示意输入计算。 | [PointRend Fig. 4](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf) · [Deformable DETR Fig. 2](https://arxiv.org/pdf/2010.04159#page=5) |
| 面积不变与重叠变化的关系散在多个段落 | 先定义合法端点、排名和写回，再列出原生重叠、净新增交集以及面积变化。一般面积和同面积两种 IoU 差值共用符号，紧邻公式解释分母。 | [DETR Eqs. (1)–(2)](https://arxiv.org/pdf/2005.12872#page=5) · [Boundary IoU Eq. (1)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf) |
| 划分、固定匹配和 AP 分母靠反复查正文确认 | Table I 写出各划分的用途和嵌套关系。Table II 分列评估范围、样本量和效果，区分全候选 AP 点估计和固定匹配上的采集组重抽样区间。Tables III–V 用独立行及横线区分相同预算和各自预算。 | [WILDS Table 1](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf#page=7) · [Mask2Former Tables 4–5](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf) |
| 定性结果需要从同一 SAR 视野追溯到面积与重叠变化 | Fig. 9 同一行的各方法共享底图、裁剪和轮廓尺度。保留正负两例，按各自连续 IoU 变化中位数附近选取。方法列下报告实际测量，公共图的空间聚合在图注明示。 | [BPR Fig. 3](https://openaccess.thecvf.com/content/CVPR2021/papers/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_Segmentation_CVPR_2021_paper.pdf) · [RefineMask Fig. 6](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.pdf) |
| 摘要和结果段缺少可核对的比较对象 | 按 SAR 边界错位、受控问题、同预算效果和外推边界组织摘要。引言量化窄船横移一像素的损失。相关工作按三个技术问题展开。结果段把主要量级放在句首附近，同时写出比较对象、数据划分与已观察的例外。 | [SegFix Fig. 2 与引言](https://arxiv.org/pdf/2007.04269) · [WILDS 实验设置](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf) |

## 交付与核验

- [17 页英文双栏审阅稿](../paper_boundary_refinement_20261120/boundary_refinement.pdf) · [LaTeX](../paper_boundary_refinement_20261120/boundary_refinement.tex) · [五张矢量图及重建入口](../paper_boundary_refinement_20261120/figures/README.md)
- 查对 11 图、10 表、27 条文献及关键版面的图文关系。作者及投稿字段按约定留占位。公开 Fig. 9 使用发布范围内的聚合数据，内部审阅版用原始 SAR 底图和作者轮廓。
- IEEE 双栏版式、图中文字和文件格式参考出版方说明。图形构图与实验呈现来自上表列出的逐项研读。贡献范围及投稿证据见[稿件证据索引](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md)。[TGRS 作者说明](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/) · [IEEE 图件规范](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/)

