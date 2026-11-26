# NORA 完整稿与重绘图件

**NORA: Native-Grid Occupancy Ranking and Allocation for SAR Ship Instance Segmentation**

残差占据率头（Residual Occupancy Head，ROH）在正文使用简称。单输出和双参考版本分别写为 scalar ROH 和 dual-reference ROH。

- [IEEE 双栏英文稿](boundary_refinement_complete.pdf) · [LaTeX 源码](boundary_refinement_complete.tex)
- [十张方法图和编辑入口](figures/method_atlas_20261126/README.md) · [结果图件](figures/journal_20261126/README.md) · [方法图构图说明](style_review/NORA_METHOD_FIGURE_STORYBOARD_20261126.md) · [GLDet 全文与全图表研读](style_review/GLDET_FULL_STYLE_REVIEW_20261126.md)
- [问题、完整设计与证据边界](design/DESIGN_RATIONALE_ZH.md) · [复现索引](design/NORA_REPRODUCIBILITY.md) · [实验证据](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md)
- [语言终审](design/PROSE_FINAL_REVIEW_20261126.md) · [作者信息与投稿资料](submission_20261126/AUTHOR_HANDOFF_ZH.md)

重绘覆盖方法图、定量图、定性对照及消融图。图内保留必要的操作、符号、单位和图例，方法段落解释真实裁片和操作示意的关系。公开版为 21 图，作者审阅版包括 22 图及真实 SAR 裁片。方法与原理图共 10 幅。原 15 张数据表的表体、公式和实验结果逐项保留。图注和表题按重排精简。

新版扩写引言和相关工作。引言按 SAR 回波、窄船边界、原生像元与掩膜写回展开，以几何示意说明问题。相关工作依次交代旋转区域与 SAR 特征、局部与整掩膜精修、排序与面积约束，并在各节末明确本文处理的未决环节。原引言的 HRSID 参考引导诊断放在定量结果，真实案例的工作流总图归入方法。后续章节用固定变量、对应对照和可支持的结论衔接。新增的 SAR 成像参考已核对。[Moreira 等的 SAR 综述](https://elib.dlr.de/82313/)

新版增加整体网络架构图并重画 ROH 层图。图 3 展示冻结前端、原生响应与二值掩膜的分流、可写端点、占据率估计、写回及候选属性的旁路。图 6 展示 14 维特征、仿射矩阵与 ReLU、解码器 logit 残差和源域监督，单输出与双参考网络分别拟合。

在本目录执行两至三次 pdflatex 编译 boundary_refinement_complete.tex 即可重建公开稿。署名等投稿元数据保留占位，Umbra 的独立标注与评分状态见证据索引。

