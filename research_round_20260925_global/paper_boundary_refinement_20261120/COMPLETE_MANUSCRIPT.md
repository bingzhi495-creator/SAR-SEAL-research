# NORA 精简正文、补充材料与重绘图件

**NORA: Native-Grid Occupancy Ranking and Allocation for SAR Ship Instance Segmentation**

残差占据率头（Residual Occupancy Head，ROH）在正文使用简称。单输出和双参考版本分别写为 scalar ROH 和 dual-reference ROH。

- [论文与图件续作交接](NORA_CONTINUATION_20261126.md)
- [精简正文 PDF](boundary_refinement_concise.pdf) · [补充材料 PDF](boundary_refinement_supplement.pdf) · [正文 LaTeX](boundary_refinement_concise.tex) · [补充材料 LaTeX](boundary_refinement_supplement.tex)
- [扩展审阅稿 PDF](boundary_refinement_complete.pdf) · [扩展稿 LaTeX](boundary_refinement_complete.tex)
- [十张方法图和编辑入口](figures/method_atlas_20261126/README.md) · [结果图件](figures/journal_20261126/README.md) · [方法图构图说明](style_review/NORA_METHOD_FIGURE_STORYBOARD_20261126.md) · [GLDet 全文与全图表研读](style_review/GLDET_FULL_STYLE_REVIEW_20261126.md)
- [问题、完整设计与证据边界](design/DESIGN_RATIONALE_ZH.md) · [复现索引](design/NORA_REPRODUCIBILITY.md) · [实验证据](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md)
- [语言终审](design/PROSE_FINAL_REVIEW_20261126.md) · [作者信息与投稿资料](submission_20261126/AUTHOR_HANDOFF_ZH.md)

## 精简正文（2026-11-26）

采用 IEEE 双栏排版，正文至结论 12 页，含参考文献共 13 页。正文的 9 张图和 7 张表涵盖整体网络、原生网格采样、ROH、等增删数的三种写回策略和固定总量的双参考分配。保留原检测器和匹配预算的 PointRend 对照，按相同写入预算展示增益与退化案例。外域结果和边界 IoU 的不确定性留在正文。

引言按三个问题依次引出 ROH 的分数像元监督、等额增删时的像元选择，以及固定总量下的增删分配。相关工作沿同一顺序追溯已有方法与具体缺口。正文和图中的 GT 标签区分多边形、轮廓和栅格掩膜。Phoenix 的阈值规则以模型名标明。

独立补充材料的公开版为 25 页，保留扩展稿 15 张数据表的表体和编号公式，收录数据划分、完整实现、推导、各子集与模型的消融、拓扑与计算开销。作者本地审阅版采用真实 SAR 裁片，补充材料共 26 页。本次改动沿用已核验实验，署名与投稿元数据仍保留占位。公开版在本目录分别运行三次 pdflatex 编译两份 LaTeX 源码。

## 扩展审阅稿与图件

重绘覆盖方法图、定量图、定性对照及消融图。图内保留必要的操作、符号、单位和图例，方法段落解释真实裁片和操作示意的关系。公开版为 21 图，作者审阅版包括 22 图及真实 SAR 裁片。方法与原理图共 10 幅。原 15 张数据表的表体、公式和实验结果逐项保留。图注和表题按重排精简。

新版扩写引言和相关工作。引言按 SAR 回波、窄船边界、原生像元与掩膜写回展开，以几何示意说明问题。相关工作依次交代旋转区域与 SAR 特征、局部与整掩膜精修、排序与面积约束，并在各节末明确本文处理的未决环节。原引言的 HRSID 参考引导诊断放在定量结果，真实案例的工作流总图归入方法。后续章节用固定变量、对应对照和可支持的结论衔接。新增的 SAR 成像参考已核对。[Moreira 等的 SAR 综述](https://elib.dlr.de/82313/)

新版增加整体网络架构图并重画 ROH 层图。图 3 展示冻结前端、原生响应与二值掩膜的分流、可写端点、占据率估计、写回及候选属性的旁路。图 6 展示 14 维特征、仿射矩阵与 ReLU、解码器 logit 残差和源域监督，单输出与双参考网络分别拟合。

在本目录执行两至三次 pdflatex 编译 boundary_refinement_complete.tex 即可重建公开稿。署名等投稿元数据保留占位，Umbra 的独立标注与评分状态见证据索引。

