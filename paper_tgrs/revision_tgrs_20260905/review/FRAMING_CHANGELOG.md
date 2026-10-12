# 摘要、引言、相关工作重写记录

日期：2026-09-05。独占修改文件：`sec_intro.tex`、`sec_related.tex`。依据：本修订副本的 `PAPER_PLAN.md`、前一轮 `figure_redesign_20260905/review/PAPER_AUDIT_CN.md` 及所列原始论文。未修改原目录稿件、实验表格或数值。

## 新的论文主线

主问题调整为：已训练 SAR 实例分割载体产生多个重叠预测后，如何关联候选、选择融合轮廓、局部修正边界并校准排序？三类输出问题构成方法结构，避免把“结构修改均无效”当作研究前提。

摘要明确区分两套载体：HRSID 为 **COCO-initialized Mask2Former-R50 trained with auxiliary mask-quality supervision**；PSeg-SSDD 为使用标准 Mask2Former head 的 COCO 初始化载体。两者都在完成各自训练后冻结。引言同时区分 HRSID carrier 的辅助质量训练与后置质量回归器，避免用推理时关闭质量分数混合来否认辅助监督的存在，也不把两个数据集描述成完全相同的训练头。

## 贡献边界

1. 将 fusion 精确命名为 cluster-support-gated：按分数选种子、mask-IoU 关联、有限候选数区间内投票、区间外保留种子、平局保前景。明确候选数不等于不同视图数。
2. 将 BPR 明确列为继承的局部补丁精修方法；新增内容只定位在本研究的具体输入、形态学写入范围、与融合和重排的接口及评估。删除 BPR 基本构思属于本稿的暗示。
3. 将 weight soup、TTA、mask voting 本身列为既有工具，不申请泛化的“首次融合掩码假设”新颖性。
4. 字段不覆盖只支持可追溯修改范围，不推出可交换、近似可加、保证提高 AP 或宽 TTA 必定安全。
5. 删除“完全未修改模型”“保留规律”“容量不需要”“已隔离 COCO 初始化 +9.8”“总体 SOTA”“所有阶段与载体皆显著”等叙事。

## 数值与协议

摘要和引言保留核心已报告端点：HRSID official 45.6 → 59.4，inshore 27.7 → 44.5；PSeg-SSDD official 60.9 → 67.5。未创造新实验或显著性结论。

HRSID 被明确标为 exploratory，因为测试子集参与配置选择。PSeg-SSDD 被描述为融合/精修设置迁移，并训练该数据集的模型与纠错器；它提供数据集范围内的进一步证据，不等于确认全部 HRSID 设计论断。跨论文数值仅作为文献背景，初始化、选模和预算不一致时不作同条件排行榜。

## 引用变化与原始来源

新增引用键仅有 **`mnc2016`** 和 **`fcis2017`**，由主任务维护 `refs.bib`。其余引用键沿用现有库。

- [MNC 原论文，CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.pdf)：Inference 部分明确给出 box-level NMS 后的分类分数加权 mask voting 和二值化。正文相关工作已引用这一直接前作。
- [FCIS 原论文，CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Fully_Convolutional_Instance-Aware_CVPR_2017_paper.pdf)：第 2.3 节 Inference 明确给出相似 ROI 的分类分数加权逐像素平均与二值化。正文据此承认 mask voting 既有历史。
- [BPR 官方页面，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html)：明确为可作用于已有实例分割输出的边界补丁后处理。官方页码 **13926–13935**；旧 `refs.bib` 的 13921–13930 需要主任务修正。
- [Mask Scoring R-CNN 官方页面，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)：以预测 mask IoU 处理分类置信度与掩码质量失配，支持将质量训练与外部质量重排纳入正确前作脉络。

所有已有 section labels 均保留，包括 `sec:related-peft` 与 `sec:related-aniso`；二者保留为兼容锚点，现在指向边界修正与质量校准小节。引用库中没有实际使用的旧条目无需为保持数量而强行引用。

## 检查说明

最终摘要为 **222 词**（空白分词），满足 150–250 词要求。两个文件在 IEEEtran 类下完成局部 draft-mode 编译，退出码 0，无错误与 Overfull box；有一项普通 Underfull box 警告。六个原有 section labels 全部保留，核心六个数值端点未变。检查时仅新增的 `mnc2016`、`fcis2017` 待主任务补入引用库；局部不包含全稿图件和 bibliography，因此引用与图号未解析警告由全稿编译统一收敛。机器记录见 `review/FRAMING_CHECKS.json`。

全稿的最终编译、引用解析、排版验收由主任务统一完成。当前改写使正文与已见训练记录一致，但不构成新的 AP 复现、全面查新或独立确认性实验。
