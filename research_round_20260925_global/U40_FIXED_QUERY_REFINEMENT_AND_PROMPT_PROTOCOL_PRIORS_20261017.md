# 固定query掩膜精修与点提示方法：直接先例和比较范围补核

2026-10-17。使用research-lit/novelty-check流程作定点补充，不宣称全量查新完成、不增加项目去重论文数。无可调用跨模型审阅后端；本文不认证新颖性。不读当前训练科学载荷、不改正在运行的50轮配方。

## ReSAM：任务和论文版本需要区分

Muhammad Naseer Subhani，ReSAM，CVPR2026。已核[CVF正式论文索引](https://openaccess.thecvf.com/content/CVPR2026/papers/Subhani_ReSAM_Refine_Requery_and_Reinforce_Self-Prompting_Point-Supervised_Segmentation_for_Remote_CVPR_2026_paper.pdf)、[作者项目页](https://mnaseersubhani.github.io/resam_project_page/)及[作者arXiv v1全文§3–4](https://arxiv.org/html/2511.21606v1)。CVF PDF/HTML直连失败，正式版仅核到出版社索引表格与方法片段；方法和划分细节以下明确来自v1及作者代码，不写成已完整读取正式全文。

ReSAM以稀疏点驱动SAM/SAM2，通过清理重叠伪掩膜、由掩膜重建框后再次提示，以及语义队列对齐进行适配。正式索引表3的HRSID-Inshore单点SAM结果为58.40 IoU、70.11 F1；这些不是COCO mask AP。v1§4.1说明其近岸子集为459训练/250测试，不能自动替代官方HRSID完整3642/1962协议。v1的WHU消融终值为73.4 mIoU，正式索引表4为75.8，版本有修改，不能把两版消融数字混成一组实验。

进一步核验作者仓库固定提交[c30ef449ecb3bf010779c4b47f0dd455a2471646](https://github.com/MNaseerSubhani/ReSAM/tree/c30ef449ecb3bf010779c4b47f0dd455a2471646)。[eval_utils.py第84–133行](https://github.com/MNaseerSubhani/ReSAM/blob/c30ef449ecb3bf010779c4b47f0dd455a2471646/utils/eval_utils.py#L84)的验证入口接收GT掩膜，通过提示函数从中生成点，逐实例计算二值IoU/F1；[HRSID配置](https://github.com/MNaseerSubhani/ReSAM/blob/c30ef449ecb3bf010779c4b47f0dd455a2471646/configs/config_hrsid.py)默认单点、LoRA rank4和10轮。仅阅读源码，没有运行作者代码或核验其数据/权重；当前代码也不自动等同论文最终实验版本。GT采样是该点提示协议的输入约定，本笔记不把这一约定指认为结果造假。

**本项目的比较结论是推论：** ReSAM适合作为点提示/弱监督相关工作，当前不能把其IoU/F1数值直接放入自动实例COCO AP优劣表。若将来适配为自动候选提示，必须报告提示来源和训练监督变化，另作受控实验；当前不启动此分支。正式自动实例分割强对照仍需按相同数据协议核验。

## CascadePSP：原图加粗掩膜细化已有直接先例

Ho Kei Cheng等，CascadePSP，CVPR2020，8890–8899。[CVF记录](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.html)与[原文§3.2索引](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.pdf)描述全局低分辨率结构修正、局部高分辨率裁块级联和粗掩膜逐层更新。直连被拒，依据为原出版页和原PDF可检索段落，不作全文审阅声明。

**与本项目关系是推论：** 额外原图输入、卷积精修和全局/局部结合本身已有先例。本法的候选差异是每图共享完整自研UNet++编码，使用固定教师query/自然掩膜条件预测native logit残差，并保留分类分数及候选映射；这一接口组合尚非创新证明。与CascadePSP的逐级全局/裁块细化不同，也不自动说明本法更好。

## 对当前主方法的约束

1. [O2Former/DACF-net补核](U40_DIRECT_PRIOR_UPDATE_20261017.md)已表明方向、多尺度和高层语义调制都有直接先例；本轮不将这些已知原语重新命名为独立创新。
2. 当前锁定方法是**固定候选上的实例掩膜精修**。候选数、类别、分数排序保持不变；掩膜改变仍可改变实例匹配与mask AP/召回，但不能据此宣称新增proposal或检测头贡献。真实预测还须核验既定无新增GT提示接口；历史缓存的目标关联字段范围沿原说明披露，不声称整个Source无GT派生信息。
3. 先完成原full/plain同图像、条件、监督、顺序和每臂37050更新的固定端点。普通plain已是同协议图像主干控制；完整结构比plain更多的容量和计算仍是替代解释。
4. 若主效用得到支持，优先完成已计划的容量/静态方向分支/固定增益替换，再选择能改变结论的最少通用精修对照。完整作者复现、同协议结构适配和预训练直接迁移必须分别注明，不能把训练条件不同的额外模型称完全公平对照。
5. N28首种子路由结果及停止决定保持，不据这轮阅读加模块、改预算或融合。完整主干有效性、直接先例差异、重复稳定性和跨集证据尚未完成，不能宣布SOTA或TGRS投稿依据已成立。
