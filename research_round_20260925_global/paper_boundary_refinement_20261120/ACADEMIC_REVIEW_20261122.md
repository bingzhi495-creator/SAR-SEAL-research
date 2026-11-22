# 英文审阅稿复核记录（2026-11-22）

使用 GPT-6.2 Astra（ultra）开设独立上下文，分别复核外部学术工作流、引用和关键定量主张。核对范围为当前审阅稿及公开实验汇总。作者信息仍为占位符。

| 核对项 | 核对结果 |
| --- | --- |
| HRSID 同头、同量对照 | DEV705 上 D-FINE +0.811 mask AP，两次 RF-DETR 拟合分别 +0.906 和 +0.838。历史官方划分 +1.216。两次 RF-DETR 拟合共享初始化和评价图像。三个点头种子不能计为三次检测器复验。[DEV705 同量对照](../RFDETR_COMMON5203_DEV705_FROZEN_ACTUAL_20261122.md) · [历史官方划分](../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md) |
| 同量动态空间对照 | HRSID DEV705 的学习排序提高 2.428 AP、1.150 连续 IoU。历史 PSeg-SSDD TEST232 的冻结 D-FINE 分别变化 −0.026 和 −0.270。稿件按载体和划分报告。[HRSID DEV705](../HRSID_DEV705_CLASSICAL_ATTRACTION_ACTUAL_20261122.md) · [PSeg-SSDD TEST232](../PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md) |
| PSeg-SSDD 目标训练 RF-DETR | DEV187 上学习排序相对动态规则 +0.845 AP、+0.193 连续 IoU，区间 [−0.036, 0.430]。相对原生下降 0.735 AP、0.404 连续 IoU。连续几何的预锁确认条件未通过。区间只用于固定匹配的连续 IoU。[评分与预锁](../PSEG_RFDETR_FROZEN_SOURCE_HEAD_ACTUAL_20261122.md) |
| 真值上界 | 保持动态规则的交换次数时，同端点独立选点的几何上界高于原生 2.246 连续 IoU 点。减少交换次数与放开面积限制分别在同一上界上提高 0.032 和 1.018 点。稿件分别报告两项差值。[上界计算](../results/PSEG_RFDETR_FROZEN_SOURCE_HEAD_ORACLE_SAFE_20261122.json) |
| 参考文献 | 独立核对 16 条参考文献和 20 次文内引用，可追溯到原始论文、会议记录或数据作者。修订 DEV705 的首次定义，去除相关工作中未就近链接的项目内部消融数值。直接先例支持正文对独立选点、空间吸引力和排序方法的描述。[HRSID](https://ieeexplore.ieee.org/abstract/document/9127939/authors) · [Atkinson](https://www.asprs.org/wp-content/uploads/pers/2005journal/jul/2005_jul_839-846.pdf) · [RankSEG](https://www.jmlr.org/papers/v24/22-0712.html) |
| 代码与版面 | 发布副本的四项合成回放测试和一项评分测试通过。PDF 两次编译为七页，检查了图、表和参考文献布局。图注标明方法示意图和实测结果图。[可编辑图件和数据入口](figures/README.md) · [LaTeX 源稿](boundary_refinement.tex) · [审阅 PDF](boundary_refinement.pdf) |

连续 IoU 的固定匹配计数已注明严格不等式适用的 HRSID RF-DETR 结果，以及包含等号的 D-FINE FIT109 和 PSeg RF-DETR 结果。COCO AP90 独立计算。[COCO 官方评价](https://cocodataset.org/dataset/detection-eval.htm)

从公开 GitHub 技能采用了逐句核对数值及范围、书目和语境分开核验、按最终版面查看图件等方法。以上记录来自独立的同模型家族复核，外部仓库仅作公开方法来源。[ARIS 数值审核](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/4ba94ae2a1c1fbbb14858750dd8764cabf6c28bd/skills/paper-claim-audit/SKILL.md) · [Orchestra 引用规范](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/20-ml-paper-writing/ml-paper-writing/references/citation-workflow.md) · [K-Dense 科学图件](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/scientific-visualization/SKILL.md) · [OpenAI PDF 检查](https://github.com/openai/skills/blob/49f948faa9258a0c61caceaf225e179651397431/skills/.curated/pdf/SKILL.md)
