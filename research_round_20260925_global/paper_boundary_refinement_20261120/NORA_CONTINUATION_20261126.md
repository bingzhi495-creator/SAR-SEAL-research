# NORA 论文与图件续作交接

**现行交接，2026-11-27。** [全文修订及验证](design/NORA_FULLTEXT_REVISION_20261127.md) · [正文、独立补材及图件索引](COMPLETE_MANUSCRIPT.md)。公开稿件为 13 / 27 页。正负案例及冻结迁移证据保留，Umbra 独立评分仍待完成。以下续作 02 及更早信息按当时版本读取。


> 续作 02 已完成稿图修订与四份 PDF 编译核验。当前公开正文 13 页、补材 26 页（20 图）；本机原生补材 26 页（21 图）。[现行入口与修订记录](COMPLETE_MANUSCRIPT.md) · [证据及验收记录](design/CONTINUATION_AUDIT_20261126_02.md) · [五页可编辑核心图](figures/revision_20261126_02/README.md)。下文保留本次交接时的历史快照；Umbra 独立评分仍未完成。

2026-11-26，北京时间。公共 SAR 船舶实例分割，目标期刊 IEEE TGRS。现行标题 **NORA: Native-Grid Occupancy Ranking and Allocation for SAR Ship Instance Segmentation**。当前交付是作者审阅稿。以本页定位材料，从正文叙事和全套图件复核开始续作。

## 当前版本和文件

| 材料 | 入口 |
| --- | --- |
| 精简正文 | [PDF](boundary_refinement_concise.pdf) · [LaTeX](boundary_refinement_concise.tex)。正文至结论 12 页，含参考文献 13 页，9 图、7 表 |
| 独立补充材料 | [PDF](boundary_refinement_supplement.pdf) · [LaTeX](boundary_refinement_supplement.tex)。公开版 25 页、20 图、15 表，含真实裁片的作者审阅版 26 页、21 图 |
| 扩展稿及稿件历史 | [扩展 PDF](boundary_refinement_complete.pdf) · [扩展 LaTeX](boundary_refinement_complete.tex) · [稿件总览](COMPLETE_MANUSCRIPT.md)。10 页、24 页版本可用于追溯叙事和公式 |
| 设计、复现和研究记录 | [设计理由](design/DESIGN_RATIONALE_ZH.md) · [复现编号](design/NORA_REPRODUCIBILITY.md) · [主张与证据](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md) · [研究全史及原目标](../SAR_SEAL_RESEARCH_HANDOFF_20261126.md) · [历史论文索引](../../docs/PAPER_INDEX.md) |
| 作图及投稿交接 | [方法图册及 Draw.io](figures/method_atlas_20261126/README.md) · [定量、定性及消融图](figures/journal_20261126/README.md) · [构图说明](style_review/NORA_METHOD_FIGURE_STORYBOARD_20261126.md) · [投稿资料与权限核对](submission_20261126/AUTHOR_HANDOFF_ZH.md) |

[清稿基础提交 08e91e3](https://github.com/bingzhi495-creator/SAR-SEAL-research/commit/08e91e36d928bc237fd14d4cf2c4fe669e9f3675) 已同步核验。用精简稿与独立补材作为编辑基线，其余长短稿用于取回证据。较早交接中的页数、制作说明和待执行状态以本页为准。署名、单位、ORCID、基金、权限及投稿元数据仍待作者确认。已具备完整审阅材料，正式投稿尚未执行。

## 从引言到结论的主线

开篇从停靠船舶、细长船体和岸边相近回波进入。检测框可能正确而轮廓错位。四像元宽的理想轮廓横移一像元，面积不变，IoU 降至 0.60。再由三类误差逐一引入下列方法。Introduction 讲研究问题、技术缺口和解决办法。数据集、指标、实验设置及成绩放在 Protocol 与 Results。

| 现有问题 | 方法与直接先例 | 正文需要回答的问题 |
| --- | --- | --- |
| 连续标注轮廓跨越原生像元，中心二值标签丢失覆盖比例 | 在冻结检测框内取内外两像元带，每侧至多 32 个端点。基于 14 维端点特征，用 polygon–cell 分数面积监督 ROH 的 decoder-logit 残差。标量头为 28→24→24→1，前两层用 ReLU，1,321 参数。14 维向量的重复用于对齐比较输入宽度 | 对应旋转 RoI、局部/全局特征和边界表达，从 SRNet、DRSNet 到动态关键点、GLDet、ARANet、DACF、DSR。说清 (q_i) 的标注几何含义、原生输出仍为二值，以及强分数来源对照与分数监督消融各自支持的范围 |
| 正确面积仍可能伴随两侧错分，最近配对可能限制被改像元 | 同一冻结候选、端点、ROH 分数及每实例 (k/k) 增删下，F 按平方距离优先配对并确定保留数，L 继承 F **已保留边** 的最大单对距离并在其内优化匹配，I 两侧独立选点。估计目标可行域嵌套，真实 IoU 与 AP 由实测决定 | 对应 Atkinson 的保面积交换和 PointRend、SegFix、BPR 的边界细化。分别解释 L−F 改配对和 I−L 放松单边约束。与原检测器的自然预算比较、同预算比较各用自己的分母。累计位移、实例连通性与窄边界收益另行判断 |
| 面积本身偏大或偏小，等量增删保留该偏差 | 固定总修改数 (2k_s)，允许重新分配增与删。双参考 ROH 为 28→24→24→2，1,346 参数。在两套同图标注构造的共同候选菜单上，最大化两种预测 IoU 增益中的较小值。等分对照共用分数、总量、目标和候选构造 | 对应 RankSEG、RankSEG-RMA 的得分排序和前景量控制，以及 SegRefiner、Phoenix 的整掩膜精修。区分最终前景面积、总修改数和增删符号，指出参考敏感性与冻结迁移失效 |

Methods 以 Fig. 1 整体架构开场，依次展开 Fig. 2–4 的采样、覆盖率和 ROH，Fig. 5 的三种位置策略，Fig. 6 的总量控制。每个公式初次使用时解释变量、索引、单位及适用条件。Protocol 报告冻结前端、数据划分、同图标注、指标和分组重采样。Results 先报绝对性能和等预算比较，再报可变分配、正负定性、消融与外部 refiner。Discussion 交代近岸细边界、可编辑候选覆盖、误差与成本边界。Conclusion 归结三项技术设计及实际测到的范围。

## 可直接继承的结果

以下 AP 与 IoU 差值均为百分点。每条数据要回看配对对象、评价分母和区间。

| 核心比较 | 核验结果和边界 |
| --- | --- |
| HRSID DEV705 等量增删 I−F | D-FINE +0.811 mask AP、RF-DETR 两个拟合 +0.906/+0.838。D-FINE 自然预算相对原生 48.160→50.715 AP，属于另一比较。官方 1,962 图的 +1.216 AP 为历史回顾评估，曾用于选型且有精确图像重复 [D-FINE](../DFINE_E24_DEV705_FREE_ENDPOINT_20261120.md) · [重复拟合](../RFDETR_COMMON5203_DEV705_FROZEN_ACTUAL_20261122.md) · [官方集](../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md) |
| Instance-RSDD Gaofen-3 583 图 I−F | 两个 RF-DETR 冻结拟合 +0.483/+0.299 AP。Gaofen-3 的限距对照不足以把每个增量都归给解除上限 [证据](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md#主张与核验记录) |
| Instance-RSDD 同一 74 图保留集 I−L | RF-DETR +0.479 AP、+0.116 连续 IoU［0.008, 0.242］。D-FINE +0.344 AP、+0.133 连续 IoU［0.018, 0.251］。两个冻结流程共用目标影像。42 图为剔除共享传感器日期的嵌套敏感性，地域独立仍未核实 [RF-DETR](../code/LEARNED_BOUNDED_PAIRING_20261126/ACTUAL.md) · [D-FINE](../code/DFINE_RSDD74_COMMON_CAP_20261126/ACTUAL.md) |
| 一像元 Boundary IoU 和代价 | 74 图 L−F 两载体 +0.735/+0.639，I−L 的一、二像元区间跨零。CPU 从缓存点分数至 RLE 的 F/L/I 中位耗时 26.1/138.6/27.4 ms/图 [边界区间](../code/RSDD74_BOUNDED_BOUNDARY_20261126/ACTUAL.md) · [计时](../code/LEARNED_BOUNDED_PAIRING_20261126/rsdd_timing/ACTUAL.md) |
| 强对照和外域限制 | 同端点、同增删量的改装 PointRend、经典动态吸引力、原版/受控 Phoenix、SegRefiner-LR 与 SRNet 均有记录。74 图 PointRend 的 I−L 仅 +0.095 AP，得分头交互区间跨零。PSeg-SSDD 232 图 I−L +0.021 AP、+0.069 连续 IoU，排除 10 个历史拟合重合的 222 图结论相近。各自参考匹配与共享后的对照须明示 [PointRend](../code/RSDD74_POINTREND_COMMON_CAP_20261126/SCORE_ACTUAL.md) · [PSeg](../code/PSEG232_DFINE_COMMON_CAP_20261126/ACTUAL_SCORE.md) · [整掩膜](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md#强精修整掩膜补证) |
| 固定总量的配比及失败 | SSDD 183 图同图两套参考 PSeg/SPSD，三次拟合均值 +2.299/+1.751 AP。冻结 SAR-SSP 232 图上 −0.139 AP、−0.212 IoU［−0.547, 0.104］，漏分减少但过分割扩大。两套标注与重复拟合都共用各自评价影像 [三次拟合](../code/SPSD_PSEG_REPEAT_VARIABLE_SIGNED_BUDGET_SEEDS5298_5299_20261126/ACTUAL.md) · [冻结迁移](../code/SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/ACTUAL.md) |

当前首要贡献是 **固定增删数后的位置控制及其受限实证**。ROH 支撑这一比较，分数目标相对替代目标的增益程度随参考构造变化。配比分支提供同图双参考的独立问题与负迁移证据。跨数据集测试、同一队列的两条管线、重复点头拟合分别报告。当前证据限定在保留检测和允许写回的像元，漏检恢复、拓扑、物理尺寸、RD 图及通用 SOTA 排名各需相应验证。

## 图件交付与构图规则

| 正文图 | 编辑入口与检查点 |
| --- | --- |
| 1 整体网络 | [network overview](figures/method_atlas_20261126/nora_network_overview.svg)，由 draw_nora_network_overview.py 重建。表现每次使用一个冻结前端、原生 logits/响应与二值掩膜分路、候选属性直通，监督只进入拟合 |
| 2–4 采样、分数覆盖、ROH | [采样](figures/method_atlas_20261126/nora_native_sampling.svg) · [读出](figures/method_atlas_20261126/nora_fractional_readout.svg) · [ROH](figures/method_atlas_20261126/nora_roh_architecture.svg)。分别用 draw_nora_scene_atlas.py、build_roh_atlas.py、build_roh_architecture.py。逐层符号见正文，优先用叠加方片、栅格、仿射矩阵和残差分流表示真实结构。把检测器原生采样与 GT 部分面积分开 |
| 5–6 选择和配比 | [F/L/I](figures/journal_20261126/method_bounded_selection.svg) · [固定总量](figures/method_atlas_20261126/nora_signed_budget.svg)。各子图保留共用初始点，画清 F 已保留最大边、L 可重配、I 两侧独立。四次修改的示意图清楚标出 add/remove 和净面积 |
| 7 配比分解 | [效应图](figures/journal_20261126/dual_reference_budget_effect.svg)，plot_nora_evidence.py。此图是首次拟合，正文表 V 汇报三次拟合均值。固定总量、同图参考、区间和轴单位在图文对应 |
| 8–9 正负定性 | [74 图案例](figures/journal_20261126/rsdd74_common_cap_cases.svg) · [Gaofen-3 案例](figures/journal_20261126/gf3_matched_cases_binned.svg)。同一案例各方法统一裁切、尺度、GT、配对预算和图例。Fig. 8 比较 L/I 等量的正负中位案例，Fig. 9 比较相同动态吸引力预算的得分来源。保留实测 IoU 和失败案例 |

补材以实际 LaTeX 的图编号、includegraphics 和 caption 为准。复核流程与几何、Phoenix 受控写回、跨集汇总、同图参考敏感性、DEV705/PSeg/SegRefiner/SAR-SSP 真实案例、单侧消融。图件源码、绘图脚本、[十页可编辑 Draw.io](figures/method_atlas_20261126/NORA_method_atlas.drawio) 与最终 PDF 同步。若制作概念草图，随后用可编辑矢量重建。原生图的重建脚本可能依赖上一代已审 PDF 的字样和剪裁锚点，换输入前核对。现成裁片若只需重排，可保留嵌入的原生图层与轮廓。论文实验图使用已核对的实测结果、裁片、掩膜和合法转换。

用户指定整体架构和模块图并重，色彩以奶油白、棕灰、橙和低饱和绿为主，避开大片深蓝。图内优先用图形表达操作，控制文字，单位、符号和图例保留。网络层用叠片或矩阵，不把标量 MLP 画成新卷积骨架。真实 SAR 裁片用于可授权的本机审阅版，公开图沿用示意或聚合矢量。标签用 GT outline、GT polygon、GT mask 等准确对象名，Phoenix threshold 表示 Phoenix 自身规则。示意值或空间池化在对应图注说明。参考 [GLDet 全文与图表研读](style_review/GLDET_FULL_STYLE_REVIEW_20261126.md) 与 [52 篇样本整理](../style_review/STYLE_REDESIGN_20261123.md) 的信息层级、画面节奏、公式和图表标题，保留自己的图形表达。

## 既有探索与任务状态

- **早期与训练。** 保留 MNE、GeoField、GeoKernel、TTA/soup、融合、BPR、重排及 N 系列的实际、预锁和失败记录。A/B 首 24 轮加续训 48 轮沿用原停止规则，续段优化器重建。第一组 B 较 A 增益 2.111 AP，仍低于可比强 E24 参考 0.274 AP。短跑按同预算配方判断。自主 UNet++ 的复合注意力、抗混叠下采样、门控细节上采样及旧权重资产曾核过，FIT109 CleanUNet++ 的局部补点与训练负结果已有记录。主稿用更有实证的冻结检测器写回。训练代码、历史模型、数据划分和合格入口从 [研究全史](../SAR_SEAL_RESEARCH_HANDOFF_20261126.md)、[固定 24+48 轮](../../docs/fixed_endpoint_joint_native_E24warm48_result_20261012.md)、[原生保留补点](../HRSID_FIT109_NATIVE_PRESERVING_ROI_ACTUAL_20261125.md) 进入。
- **数据及方向。** 公共数据优先用于通用验证。原生掩膜、HRSID JPG/PNG、Instance-RSDD 公开轮廓、PSeg/SPSD 同图参考、SAR-SSP 栅格和传感器身份均有记录。光学/AIS、幅相与复数 SAR、物理长度、RD 泛化是已探讨的扩展方向。现有论文使用解码 SAR 灰度和几何特征，按其已证范围表述。[数据入口](../../docs/DATA_ACCESS.md) · [参考准入](../code/PUBLIC_HULL_REFERENCE_SCOPE_20261126/ACTUAL.md) · [RD 适用范围](../RANGE_DOPPLER_TRANSFER_SCOPE_20261109.md)
- **最新几何分支。** DEV705 最小累计平方位移见证已经执行并结束。无标注规模检验 36 个候选、19 个前缀，实测仅改变 1/1,461 个有效固定匹配，低于预锁覆盖门。相对 L 的 mask AP −0.001385 点。[见证实测](../code/DEV705_MIN_CUMULATIVE_WITNESS_20261126/ACTUAL.md)
- **Umbra 前瞻外测。** U12 与补批 U24 具备冻结预测及回放 / 评分入口，尚无四份独立盲判 JSON 已验收记录。U12 每份至少 20 艘 / 8 组，U24 每份至少 30 艘 / 12 组，分别按原协议评分。用户可安排两位判读者。AI 三图探索性预试只用于核对轮廓定义，正式独立参考按冻结资格执行。[U12](../UMBRA12_DUAL_REVIEW_SCORE_READINESS_20261123.md) · [U24](../code/UMBRA_SUPPLEMENT_DUAL_REVIEW_SCORE_20261126/README.md)
- **复现与运行。** 证据编号、同分母与运行回执见 [主张与证据](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md)、[完整谱系](../U57_EXPERIMENT_LINEAGE_AND_EVIDENCE_SCOPE_20261021.md)、[固定端点复现](../code/RFDETR_MATCHED_POINT_REFINEMENT_20261120/README_FIXED_COUNT.md) 与 [根目录研究日志](../../README.md)。本机和 DGX 可按已有协议分工。训练结束以真实退出、更新数、来源和回执为准，按预锁规则决定下一项任务。中断、静态检查和科学负结果各有状态。当前续作优先完成稿图，不另起大规模训练。
- **写作和同步。** 英文直述技术、因果和观测，图注和表题短，方法与图内避免 author/editor 指代，完整种子、配置留补材和复现记录。审阅稿遵照项目规则禁止出现 ACKNOWLEDGMENT 式工具、润色或制图流程说明。维护作者信息占位。SCI-Skills 已安装，多智能体可协作审证据、正文与图件，主版本交给一处统稿。[项目发布范围](../../PUBLICATION_SCOPE.md) 与 [同步规则](../../docs/CONTINUOUS_SYNC.md) 适用。仅同步发布范围内文件，保护原始图像、标注、权重、运维凭据、第三方全文和运行绑定材料。

## 接手后的执行与验收

1. 以本页和现行两份 LaTeX 为基准，从摘要、引言、相关工作、Methods、Protocol、Results、Discussion 到 Conclusion 对照。逐段解决主张与证据错位、符号未定义、读出与训练混用、归因过强、旧稿结论串入等问题。保留原来的公式、指标和试验数值，在需要时通过证据链接核对。 
2. 独立复核正文 9 图和补材所有面板，然后修稿和修图。保留正负对照、固定预算、原始数值和标注来源，检查并修正字体、线宽、颜色、图例、阅读顺序、双栏尺寸与可编辑性。按同一输入对交付前后逐图比对。只在确实需要和已有授权的情况下用适用绘图工具建立新资产。
3. 编译公开与作者审阅版，逐页渲染检查图表与交叉引用、版面和字号。正文目标约 13 页，已有 12 页到结论 / 13 页含参考。细枝末节用补材，主文保留方法全链、针对性对照和负结果。扫描 LaTeX 与两份 PDF 的后台流程说明和残留草稿标记，验证版面没有溢出或缺引用，公开图和公开 PDF 不含原始裁片。新对话中直接给可读 PDF、可编辑图和精炼核验结果。正式提交须另行确认。