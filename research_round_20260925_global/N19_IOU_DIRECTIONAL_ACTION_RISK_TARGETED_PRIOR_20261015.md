# N19：IoU方向行动风险的定向先例核查

2026-10-15。**设计查新，未实施、未评价SAR；低新颖性/效用待检验；[pending Codex review]。** 使用 novelty-check 与 research-lit，仅浏览原始论文、作者/官方入口及既有公开普通N13文档；未读科学载荷、运行模型或操作Git。Codex MCP指定审查后端本会话未暴露，没有跨模型审查或正式审计结论。本轮为限时定向查证，不是穷尽查新，不给创新高分。

## 1. 待检验组合及项目内直接先例

候选保持PR单点二值动作在选点前固定、同C/K、等点成本、原分数类别/自然query顺序。三类概率为p_help、p_harm、p_noop；native P=0时用p_help−r·p_harm，P=1时用r·p_help−p_harm。r若可学习，必须推理可得、仅FIT监督/校准，禁止DEV GT或DEV阈值搜索。

**同一公式已在本项目[N13普通文档](N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)给出。** I−rU展开产生FN→TP:+1、FP→TN:+r、TP→FN:−1、TN→FP:−r，不能当作本轮新理论。N13还给出：真实标签/固定独立候选/exactK下内部r的五类排序是受限IoU oracle；即使概率完全校准且真实native IoU已知，概率加性代理仍可能不最大化期望最终IoU。不能把该oracle、Dinkelbach命名或新quality head当作最优性保证。

三个核心分别为：A. 指标驱动决策而非0.5阈值；B. 分式/非可分IoU与动作风险；C. 推理可得质量预测。可能工程差异仅是**冻结PR的四方向动作概率、native符号、固定稀疏预算和保留动作的具体整合**；本次没有找到并核实完全相同的全部组合，但这不构成首创证明。

## 2. 八项最相关原始先例

| 原始工作/状态 | 实际核到的技术事实 | 与本候选的关系及差异 |
|---|---|---|
| Werner Dinkelbach, *On Nonlinear Fractional Programming*, Management Science, 1967。[出版社](https://pubsonline.informs.org/doi/10.1287/mnsc.13.7.492)；读登记与摘要 | 分式目标与参数化优化的关系及相应算法早已建立。 | I−rU属于已知分式参数化思路；固定一个估计r并作概率topK，不等于执行Dinkelbach迭代，更不自动获得期望IoU最优性。 |
| Zachary Chase Lipton, Charles Elkan, Balakrishnan Narayanaswamy, *Thresholding Classifiers to Maximize F1 Score*, 2014。[原arXiv](https://arxiv.org/abs/1402.1892)；本轮核arXiv版本/摘要，未另核正式会议版 | 在已校准条件概率等条件下，F1最优阈值与最优F1相关，特殊情形为F1*/2。 | “0.5未必最优”不是创新；其分类决策不是固定PR动作的K点执行。F1*/2也不是本候选的固定r=.5。 |
| Sebastian Nowozin, *Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case*, CVPR 2014。[作者机构](https://www.microsoft.com/en-us/research/publication/optimal-decisions-from-probabilistic-models-the-intersection-over-union-case/)、[官方论文](https://openaccess.thecvf.com/content_cvpr_2014/papers/Nowozin_Optimal_Decisions_from_2014_CVPR_paper.pdf)；摘要及PDF检索段 | 从概率模型直接作IoU决策涉及组合困难，提出统计近似与参数线性规划；强调概率校准问题。 | 最直接的理论风险先例：局部边缘概率的线性效用不足以代表期望比值。该文不以当前冻结PR四方向行动/稀疏K为问题设置。 |
| Zixun Wang, Ben Dai, *RankSEG-RMA: An Efficient Segmentation Algorithm via Reciprocal Moment Approximation*, 2025。[原arXiv](https://arxiv.org/abs/2510.15362)、[官方PDF入口](https://openreview.net/pdf?id=4tRMm1JJhw)；作者仓库及官方PDF索引标NeurIPS2025，forum访问受验证限制 | 摘要将概率预测后按IoU/Dice直接决策与argmax/阈值区分；以倒数矩近似降低RankSEG计算开销，并构造逐像素分数。 | “不重训backbone、用概率后处理面向IoU”已有更直接先例。本候选只在固定PR可执行动作中受K约束选择，不是RankIoU/RankSEG-RMA复现；尚未证明更准确或更便宜。 |
| Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang, *Mask Scoring R-CNN*, CVPR2019。[原论文](https://arxiv.org/abs/1903.00241)、[正文](https://arxiv.org/html/1903.00241v1)；读摘要、相关工作及方法定位 | 从实例特征与预测mask回归mask IoU，校正质量与分类分数不一致，以更准确的mask排序改善AP。 | 预测实例mask质量本身不是创新。它主要重评分实例；本候选固定S0，仅把r用于像素动作风险，输出用途不同，但不足以独立构成贡献。 |
| Xiang Li等, *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection*, NeurIPS2020。[官方论文记录](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)、[方法正文](https://arxiv.org/html/2006.04388v1) | §3 QFL把类别和定位质量联合编码，以连续IoU标签训练；同时讨论质量分支训练/推理不一致。 | 质量感知概率/分数已有明确先例；QFL针对检测框类别质量，不是help/harm/noop或单点FP/FN动作效用，也不保证本方案的概率校准。 |
| Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang, *Acquisition of Localization Confidence for Accurate Object Detection*（IoU-Net）, ECCV2018。[原论文](https://arxiv.org/abs/1807.11590)；读摘要及接收说明 | 预测检测框与匹配GT的IoU，用于NMS及以预测IoU为目标的框细化。 | 推理时估计IoU并指导操作早已出现；它预测框定位质量，本候选若做mask质量和像素预算，需具体证明该用途有额外效用，不能把box IoU预测直接当mask监督。 |
| Ignacio M. De la Jara, Cristian Rodriguez-Opazo, Damith Ranasinghe, *Queries Knew More Than We Thought: Uncovering Latent Knowledge in Segmentation Models*（HYDRA）, 2026 arXiv预印本。[摘要](https://arxiv.org/abs/2609.20283)、[正文](https://arxiv.org/html/2609.20283v1)；读摘要、§4相关方法、§6相关工作及附录校准段 | 冻结输出缓存上的学习selector，显式keep-baseline/no-op；以训练GT计算动作IoU增益，部署用held-out校准margin抑制有害操作。SAM3版本有mask残差加入动作。 | “固定候选、学习行动收益、保留基线、控制误改”不是可独占的新机制。其动作主要为整mask候选/残差，不是当前PR逐点四方向和K配额。其校准扫阈值不可直接移到本项目DEV。页面显示2026-07-30提交但编号2609，日期存在不一致；接收状态/精确首次版本待复核，不依此确立优先权或引用其成绩作已验证基准。 |

## 3. 能说与不能说

可把它登记成“已知IoU方向风险对既有行动概率的预算适配”，用于检验少修/误改取舍。数学新颖性低，quality head原语新颖性低，整体组合差异尚未获得效用或非显然性证据。没有给出新理论、期望IoU最优、AP最优、首个质量感知selector、首次安全路由、外部SOTA或TGRS就绪结论。

F1=2J/(1+J)是同一二值结果的关系，不能据此互换期望F1与期望IoU最优化。r=.5是固定方向代价比，不是把原概率阈值设成.5；估计native质量也未必是最优修补后比值。三类CE输出不能未经核验就视为校准概率；IoU质量与四方向事件概率的校准是不同事项。max-IoU-to-any-GT、匹配GT质量与COCO匹配也不能混称，未匹配不赋真实空GT。

**最小继续建议（未执行）**：先完成N13已提出的固定r=.5机制对照，直接共享已完成N16 E24 UTILITY概率，与r=1原对称效用比较；PR、C、K、自然all100、S0、FP64 softmax、score>0及原tie规则全部固定，不增训练、不扫r、不引入DEV GT。先核r=1能复现原政策。比较完整187图官方AP及逐实例FN/FP修复与破坏、执行点数，而不是仅看加性代理。若r=.5无收益，不支持投入新quality head；也不证明所有质量条件化都无效。只有方向控制有可重复、可解释效用，再另行锁定是否训练r估计器，并明确质量头带来的额外容量、监督/校准数据、计算和部署成本。

这一建议是效用门，不宣称r=.5能证明“实例质量估计有效”；后者必须与相同概率/预算下固定r=.5比较。新解释或后处理收益也不能抹去N16/N17预锁STOP、事后多候选检验与DEV复用范围。

## 4. 实际查询记录与未覆盖范围

每核心至少三种表述，均已实际发出；搜索结果只用来定位原始来源，二手网页/搜索摘要不作独立性能证据。

- A：`"Thresholding Classifiers to Maximize F1 Score" optimal calibrated threshold`；`"Optimal Decisions from Probabilistic Models" "Intersection-over-Union"`；`"RankSEG" IoU optimal segmentation threshold 2024 2025 2026`。
- B：`Jaccard intersection over union fractional programming Dinkelbach segmentation optimal decisions`；`"Lovasz-Softmax" Jaccard structured loss 2018`；`IoU action utility budget sparse mask refinement 2024 2025 2026`。补检：`Jaccard Dinkelbach IoU refinement 2024 2025 2026`及`"IoU" "action" "budget" mask refinement 2024 2025 2026`。
- C：`"Mask Scoring R-CNN" mask IoU quality prediction CVPR 2019`；`"Generalized Focal Loss" "Quality Focal Loss" localization quality`；`"IoU-Net" "Acquisition of Localization Confidence" 2018`。补检：`"mask quality" "IoU" calibration 2024 2025 2026`。
- 最近六个月按2026-04-15至2026-10-15，arXiv限定与183天recency并行使用：`IoU F1 optimal decision segmentation threshold after:2026-04-15 before:2026-10-16`；`Jaccard Dinkelbach mask refinement action utility after:2026-04-15 before:2026-10-16`；`mask IoU quality estimation calibration after:2026-04-15 before:2026-10-16`；`"IoU" "optimal" "threshold" 2026 after:2026-04-15 before:2026-10-16`。检索混入旧文/邻近任务，不能将日期过滤当作完整覆盖。由此定位并实际读取上述HYDRA原文；其他近月结果未扩为泛综述。
- 2024—2026定向检索包含arXiv、OpenReview、NeurIPS/CVF，但**未逐一穷尽ICLR2025/2026、ICML2025/2026或NeurIPS2025全库**；Google Scholar/Semantic Scholar独立数据库、作者未索引实现、所有论文相关工作未完成穷尽扫描。Lovasz等被检索到的更宽结构化损失没有扩大核心表或冒称全面复核。部分CVF/论坛入口访问受阻，改读同作者arXiv原文；表格逐项保留实际阅读深度。

普通文档STOP；科学代码、协议绑定、模型训练、阈值选择和COCO评价均未执行。后续是否开展固定r=.5对照由Root另行按已有资格与停止规则决定。
