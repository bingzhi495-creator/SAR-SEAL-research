# N15 固定精修行动效用：五篇直接 primary 先例核查

2026-10-14。**普通研究文档 / 定向查新 / N15 假设未实施；本轮无科学执行、无Git发布。** 只读已发布的N15假设、N13 utility先例/目标备忘及在线原论文、作者代码；未读实际SCI、绑定SOURCE、RAW、SPEC、Q、数据、缓存或权重。不保存或分发第三方论文全文，不运行作者代码。由ROOT统一审阅、同步和决定后续预登记。

本轮应用research-lit和novelty-check的主张拆解、primary核验和差异判断；本轮定向子任务限定最多5篇、20分钟，未扫描本项目论文库/科研目录或写research-wiki。当前没有可调用的Codex MCP跨模型reviewer，因此**未完成该技能的跨模型核验**，不作穷尽性或首次证明。

## 直接影响候选和监督的结论

“预测哪里值得精修”“根据执行结果训练选择策略”“固定点预算”“候选感知置信度”和“先预测再稀疏激活”均已有直接先例。最强监督近例是 **PromptMoE：专家执行质量/遗憾/组合收益** 与 **SAM-Veteran：SAM实际输出IoU变化及精修/终止决策reward**；不能把“行动效用预测”本身当新的原语。PointRend和Mask Transfiner则分别代表不确定性采点与有监督的高频信息损失检测，两者的监督都不等于固定PR行动的净帮助/损害。

N15仍可检验一个有具体范围的差异：**同一冻结PR、相同GT无关候选域Ω和K、自然all100 query/score不变时，逐点预测该PR实际执行的help/noop/harm，是否优于同容量错误概率与简单候选感知排序；若scout先选K再调用PR，是否在预锁质量等效范围内减少真实总成本。** 这是待检验的工程/方法效用，不是已成立的创新、SAR领域空白或SOTA。

## 五篇核查表

|先例、状态和原始入口|监督label / 选择机制|与N15的具体差异、最直接否定风险|
|---|---|---|
|**PointRend: Image Segmentation as Rendering**，Kirillov、Wu、He、Girshick，CVPR2020，9799–9808。[CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)、[原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)、[作者代码](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)|point head用GT mask采样标签/BCE；选择根据coarse预测不确定性。作者代码训练先oversample随机点，再uncertainty top-k与随机点混合；推理grid top-k、迭代细分。不是learned help/harm目标。|Point预算/逐点MLP/粗预测反复注入均已存在。本项目固定native原图LOG支持、残差与冻结教师不是完整作者PointRend。若N15仅重写uncertainty排名，不能称新机制；必须比较同域、同K与同计算范围。|
|**Progressive Semantic Segmentation / MagNet**，Huynh等，CVPR2021，16755–16764。[CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Huynh_Progressive_Semantic_Segmentation_CVPR_2021_paper.html)、[原文§3.2/§4.1](https://openaccess.thecvf.com/content/CVPR2021/papers/Huynh_Progressive_Semantic_Segmentation_CVPR_2021_paper.pdf)、[作者代码](https://github.com/VinAIResearch/MagNet)|refinement模块以GT语义类别和CE训练；排序非learned action-label，而是旧预测uncertainty与候选certainty的乘积，median blur后top-k，再替换候选概率。候选已经算好。|已是候选感知、预算替换的强近例，但并非先选后仅算K个PR点。代码patch模式top-k后还筛除点，不保证所有模式最终exact K；概率替换亦不等于二值翻转。简单MagNet式排序若达到N15质量，则“改成utility分数”解释不足。|
|**Mask Transfiner for High-Quality Instance Segmentation**，Ke等，CVPR2022，4412–4421。[CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html)、[原文§3.1–3.3](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf)、[作者代码](https://github.com/SysCV/transfiner)|incoherence目标来自GT mask下/上采样损失的信息；原文nearest-neighbor缩放、XOR与邻域OR。检测器监督选择多尺度quadtree节点，再Transformer精修。不是当前P与GT逐点误差，更不是执行固定PR后净收益。|“学习哪里修”已存在；需空间/多尺度信息及相应实现预算。作者公开仓库标为reimplementation，其目标函数使用bilinear residue，与原文字面定义有版本差异。不能将冻结F插值/宽候选域称完整Transfiner，也不能因现有误差图不同就直接认定创新。|
|**PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting**，Price、Cote、Rundensteiner，CVPR2026，6325–6335。[CVF proceedings](https://openaccess.thecvf.com/CVPR2026?day=2026-06-05)、[官方原文§3.2.3](https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf#page=5)、[作者代码](https://github.com/sprice134/PromptMoE)|执行每expert后GT质量qe=(IoUe+BIoUe)/2，r_e=max_j q_j−q_e；stage1拟合实际regret的绝对误差。stage2冻结U，拟合pair gain q_ij−max(q_i,q_j)，再联合微调。路由选择少数expert，生成提示并让SAM重解码。|直接否定“效用/结果质量监督及稀疏激活首次”。这里是expert级、相对最好expert的质量/遗憾，非相对native的逐像素−1/0/+1；提示点数不是PR支持/翻转数。paper的best-pair规则与公开代码pair-marginal top-k不自动等价；regret到高U的映射/完整router训练链尚未核清，不能逐字照搬。|
|**SAM-Veteran: An MLLM-Based Human-like SAM Agent for Reasoning Segmentation**，Du、Li、Fan、Zhang、Pan、Zhang，**正式ICLR2026版**。[OpenReview](https://openreview.net/forum?id=oN55r8iJJW)、[正式PDF](https://openreview.net/pdf/69f488c8979689d03533b6ccaf32c652ebe483be.pdf)|MLLM生成初始框、观察SAM mask后提出精修点/终止。GRPO含实际mask IoU、精修/终止decision与IoU improvement RΔ；辅助任务使用人为FP/FN腐化mask。正式文稿还比较删除RΔ及用硬正改善reward替代。|不是普通confidence选择，实际行动结果反馈已有先例。其提示影响全mask、多轮、可终止，不是固定PR独立像素action或exact K同分数COCO评价。本轮**未核到可确认作者实现**，不能断言未开源；没有把其reward写成已核实同构三值help/harm CE。|

## 作者实现的定点核验与版本边界

- **PointRend**：直接只读[采样函数](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py)的`get_uncertain_point_coords_with_randomness/get_uncertain_point_coords_on_grid`，与[point loss及StandardPointHead](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py)的BCE、fine/coarse特征和逐层coarse concat。本轮读取main，**未锁新的commit、不执行**；论文经典机制由正式CVF版本支持。GT point occupancy loss不能写成固定refiner实际效用label。
- **MagNet**：复核固定`7f5b8760cb471230402015d80117559de0e38600`的[demo.py](https://github.com/VinAIResearch/MagNet/blob/7f5b8760cb471230402015d80117559de0e38600/demo.py#L160)和[geometry.py](https://github.com/VinAIResearch/MagNet/blob/7f5b8760cb471230402015d80117559de0e38600/magnet/utils/geometry.py#L82)。certainty是最高/次高类别概率差；先dense候选，再联合score、平滑、top-k和scatter；patch筛除会改变最终替换数。原文用CE训练refinement，50epochs；不以其mIoU/时间与本项目DEV AP直接排名。
- **Mask Transfiner**：复核`5b61fb53d8df5484f44c8b7d8415f398fd283ddc`的[GT目标生成](https://github.com/SysCV/transfiner/blob/5b61fb53d8df5484f44c8b7d8415f398fd283ddc/detectron2/modeling/roi_heads/mask_head.py#L86)，确认输入来自gt_masks、bilinear重建残差；[README](https://github.com/SysCV/transfiner/blob/5b61fb53d8df5484f44c8b7d8415f398fd283ddc/README.md)明确reimplementation。它不是预测P错误标签的原版等价替换。
- **PromptMoE**：官方11页PDF仅在RAM解析，第4/5页实际方法段已读；正式proceedings元数据由CVF June2026条目核实。[固定推理文件](https://github.com/sprice134/PromptMoE/blob/66d61d66238b372a5e38859864a1590260628ed4/PromptMoE.py#L41)确有utility/interaction和稀疏权重。[公开路由](https://github.com/sprice134/PromptMoE/blob/66d61d66238b372a5e38859864a1590260628ed4/PromptMoE.py#L103)采用pair-score logsumexp边缘再top-k；与paper最高无序pair+pair内softmax不同。论文§4.1训练预算为VOC2012 train的1464图80/20、384宽3层router、8/20/2epochs；本轮没有验证训练入口/权重与此声明的完整对应。
- **SAM-Veteran**：OpenReview直接页面/PDF本轮触发验证/403，正式PDF的primary搜索索引可读标题、作者、Published-at-ICLR2026页眉和reward段/消融；没有声称完整逐页读完或作者代码复现。更新此前N13“录用元数据待确认”的范围：目前可核正式版，不臆测poster类型或未公开实现状态。

## N15监督和最小否证设计建议（非实验登记）

首先冻结“行动”的精确函数与域。在Ω上使用将真正执行的`L + ungated PR residual`定义候选二值P'；若误取旧head的band-gated logits，会把LOG外行动记成noop，诊断对象已经不同。按native P、GT、P'同时保存a=FP→TN、b=FN→TP、c=TP→FN、d=TN→FP、z=noop。help/noop/harm三类加已知P可还原四种flip类型，但z多数的总accuracy不能证明utility可学；报告flip条件混淆、损害率、净修复与实例IoU。

对二值固定行动，`u=1[P≠GT]−1[P'≠GT]=δ(2e−1)`，δ是实际flip、e是native error。**label没有新增GT信息**。compressed scout descriptor未必决定δ，不能无条件把error-head和utility-head称完全等价；对应期望应保留δ与error的条件依赖。最强质量替代为同容量/输入/监督曝光的error head，加完整实际δ后按`δ[2p_error−1]`排序；它先算全域PR，必须单列成本，不伪称与先scout后K点执行计算相同。简单confidence排序/二类MagNet式native uncertainty×candidate certainty可作为不再训练的候选感知对照，须提前锁平滑和平局规则。

先区分任意GT编辑oracle与**固定PR行动oracle**。诊断时允许GT识别哪种行动有益，不是部署选择。后者在Ω无明显收益，则停止该refiner/候选域配方，不先训练scout。固定K的pixel utility不保证IoU/AP最优；a/b有不同IoU作用，forced harm/empty union/无匹配query的范围须保留。正式策略应覆盖all100 query、自然score、完整18700 COCO；374固定匹配只支持条件诊断，不能扩大到全部380或AP上限。

若有机会，再**另预锁**同refiner、Ω、K、划分、query匹配、初始化、seed、训练预算、scout容量与完整质量—成本指标的utility-vs-error对照；不是现在实施新训练。当前强E24 refiner已接触FIT，不能仅按后续scout拆分宣称refiner未见B；cross-fitting或另训A-only refiner的额外成本/行动变化都要披露。端到端账包含candidate M、标签生成、scout、F渲染、排序、实际PR(K)、传输/导出；固定K不等计算。

预期贡献只能是有证据的条件质量或稀疏执行成本差异。若error+actual flip或简单MagNet式适配质量等效，不能宣称新utility标签/检索机制优势；若oracle/coverage改善却AP不升，否定本配方；若要省计算但完整总时间/内存不降，也不支持成本贡献。已有N12覆盖22.53→24.67%但oracle仅+0.000728650、N13同K联合配方三seed负、N14更低trainBCE但同budget AP不正，均是必须保留的反证，不以更宽池或改名自动消除。

## 检索范围、证据等级与未完成项

本轮最多保留上述5篇；2025–2026定向查询涵盖segmentation refinement utility/reward/action/point budget及最近6个月近邻，优先原CVF、OpenReview、arXiv和作者GitHub。一般LLM奖励/文本refinement、同名MoE、非分割segment-level reward结果排除，不扩成模块目录。未命中完全同构实现不支持首次、领域空白或TGRS成熟。

A级：直接原文方法或作者代码读到具体目标/算子（PointRend代码，MagNet原文方法与代码，Transfiner原文目标与代码，PromptMoE官方方法PDF与推理代码）。B级：正式索引/primary搜索正文（SAM-Veteran正式出版/方法部分及代码未核）。C级：本文的N15差异、标签分析与后续建议。所有A级亦不等于独立复现或SAR效用。

本轮仅新建本普通文档；未修改N15候选/监督协议或任何实际绑定原件，未提交Git，未执行任何科学任务。ROOT后续同步与登记不等于假设、创新或收益已成立。
