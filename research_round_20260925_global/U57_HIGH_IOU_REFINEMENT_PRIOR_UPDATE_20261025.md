# 强粗掩膜精修损伤高 IoU：直接先例增量查新（2026-10-25）

状态：有界一手文献查新与可否定假设；未运行外部方法、未做科学计算，未确认新颖性。本页为该独占普通草稿的发布副本；未运行所列外部新方法。使用research-lit/novelty-check流程；本会话无可调用的跨模型Codex reviewer，未虚构交叉审阅，也不给无标尺创新分数。

本轮只读取Root指定的三份普通发布文档（U57_CURRENT_PROGRESS_AND_METHOD_EXPLANATION_20261024、U57_NEAREST_PRIOR_NOVELTY_CHECK_20261018、U57_DIRECT_PRIOR_INCREMENT_20261025）及公开原始论文/作者网页。指定N28文档在给定发布目录未找到，未转去运行目录搜索。未访问绑定SOURCE、资格链、GT/预测/权重载荷或SSH，未变更源码/配置。引用均为自写归纳，不复制第三方全文。

## 当前观测与可保留的推断

以下数字由Root提供，本子任务未重新计算或独立核验：内部DEV187图/380实例、100候选/图、冻结项目Mask2Former/R50 E24、同native候选/S0、统一COCO exact controls；native AP/AP75/AP50=67.071/83.843/97.009；固定R=68.099/85.213（AP50未提供）；作者固定commit `4bb7f95738c7a2e805f189210fb09e02a2197557` 的SAMRefiner ViT-H FP32/default5iters、seed2026102503=42.771/27.947/95.134。只支持此配置明显损伤高IoU指标，不证明SAM/SAMRefiner普遍失败。AP50较接近不能单独证明检测完全保持或具体损伤机制；GT来源独立payload审阅仍有限。

U57已知机制是固定最终实例mask embedding与粗logit条件化共享UNet++特征，零初始化头产生全域加性logit残差；保持候选、类别、分数和顺序。**零初始化保证起点等于native，不保证训练后的几何或IoU不退化；候选保持也不等于正确像素保持。** FULL/PLAIN同50轮还没有端点AP，不改当前配方、不提前判机制。

## 六个核心直接先例

| 原始工作、时间与代码 | 最相关技术/证据（均为论文自报，非本项目复现） | 对U57的具体差异与最小可否定检验 |
|---|---|---|
| **Lin等，SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement**。arXiv v1 2025-02-10/v2 03-16；ICLR2025。[论文§3.3、附录C.4/D.3](https://arxiv.org/html/2502.06756v2)、[作者仓库](https://github.com/linyq2117/SAMRefiner)。 | 粗mask挖掘点/框/Gaussian连续mask；基础版从SAM分支按预测IoU选择。++已明确指出缺少目标域先验会使IoU排序失配，通过IoU头LoRA与coarse-IoU排序伪监督适配，冻结mask生成。作者明确++依赖coarse质量、需额外训练，并非普适变体。 | U57直接学习稠密logit残差；它不是SAM提示精修/IoU排序头。不能把“目标域质量分支”当新思想。当前只做基础默认5轮，不能写作者最强配置或++失效。现有固定pair先看TP/FP/FN和面积；若日后取得分支数据，再单独预锁测“排序能否解释损失”，不向当前诊断临时加oracle。 |
| **Wang等，SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process**。NeurIPS2023；arXiv 2023-12-19。[正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html)、[方法/训练](https://arxiv.org/html/2312.12425)、[官方代码与权重](https://github.com/MengyuWang826/SegRefiner)。 | 离散coarse→fine状态转移，从coarse开始迭代，像素置信度控制软转移；默认6步。LR在LVIS训练，HR再用DIS5K/ThinObject-5K；非SAR训练free保证。它有模型无关实例/语义/二分割实验，适合作现成强替代。 | U57是带教师实例embedding的加性残差，非离散生成。仅换成扩散也不自动形成贡献。若后续选择本强替代，先固定同候选/S0、原图网格与模型权重版本，分清公开预训练直接迁移和FIT域适配；看高native-IoU实例中损伤/获益比例，不能只报均值。 |
| **Zhang等，RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features**。arXiv 2021-04-17；CVPR2021 pp6861–6869。[原文§3.3](https://arxiv.org/html/2104.08569)、[官方代码](https://github.com/zhanggang001/RefineMask)、[实际保留非边界的推理代码](https://raw.githubusercontent.com/zhanggang001/RefineMask/master/mmdet/models/roi_heads/refine_roi_head.py)。 | 高分辨率语义特征与多阶段RoI精修；后续阶段只监督边界，推理把上一阶段非边界logit拷回。故“保留强coarse主体、局部修边界”已有明确实现，不是新空白。 | U57残差是全域学习，无相同非边界拷回规则；RefineMask是受监督Mask R-CNN头，不可把随手移植称作者完整复现。先测现有改动是否大范围破坏native正确内部；若无内部损伤，不能用“保护内部”解释改进。边界宽度/区域定义须另行预锁，当前TPFPFN诊断不加新臂。 |
| **Price、Cote、Rundensteiner，PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting**。CVPR2026 June pp6325–6335；未核到更早arXiv。[CVF原始页](https://openaccess.thecvf.com/content/CVPR2026/html/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.html)、[作者代码/路由权重/结果](https://github.com/sprice134/PromptMoE)。 | IIP视觉专家提示图、DES稀疏选专家、PPE空间采点；针对固定提示规则对新域/误差模式脆弱。其作者结果表也含SegRefiner在若干非BIG数据集负ΔIoU，说明“精修会退化”已有系统先例。README仍写submitted，发表状态以CVF为准。 | DES选择提示专家，不是native/refined最终mask分支，也不保证最终不退化；U57没有此专家机制。若主张SAR需要特定局部信息，应先检验在何种误差群有效，而不是把动态选择改名。其IoU/BIoU不能改写成COCO AP；迁移同样需固定候选和评分。 |
| **Liu等，SAMWater: a prompt-guided multimodal framework for transferable water mapping with the segment anything model**。GIScience & Remote Sensing63(1),2745243；online2026-10-09。[原文](https://www.tandfonline.com/doi/full/10.1080/15481603.2026.2745243?af=R)、[官方代码](https://github.com/zkang-CN/SAMWater)。 | §5.1作者报告直接coarse+SAM使Kappa78.28→75.36、遗漏率14.85→34.12；Water-IoU只训练IoU-LoRA、冻mask生成，用同scene融合伪mask产生排序目标；代码明确改编SAMRefiner。SAM输入为S2 RGB，S1 SAR提供粗水体先验，另有SAM3文本光学提示。 | 这是最新遥感退化与目标域质量选择先例，但属于多源水体语义任务，不能解释纯SAR船实例。U57没有同图Water-IoU自适配。先用Root已固定pair的FN/GT、FP/GT、面积比检验是否真有前景抹除；若FN未升且面积不缩，撤回“漏分导致高IoU损失”的具体解释。 |
| **He等，UnfoldCRF: Structured Mask Refinement with Image-Conditioned Latent Regions**。arXiv v1 2026-09-27，当前预印本；未核到正式venue/已发布实现。[摘要](https://arxiv.org/abs/2609.33996)、[原文](https://arxiv.org/html/2609.33996)。 | 原mask作unary证据；局部pairwise与潜在region一致性，null状态允许弱一致区域退出。做输入/参数/迭代/监督匹配的recurrent控制，并报告made-worse比例；作者在COD10K报告11.7%→8.5%。限binary masks，instance级尚属future work；代码写will release。 | U57是全图UNet++条件残差，没有显式共享CRF能量；FULL/PLAIN参数尚未匹配。论文把容量/迭代控制与逐样本伤害率列为直接研究问题，不能把这些评价原则当全新机制。现有paired IoU可先报Δ分布/受损比例，后续按已有计划容量控制；不据其预印本数字排名SAR。 |

## SAMRefiner基础版与++：论文、代码和本次执行必须分开

[固定作者commit的基础函数](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py)默认`iters=5`、point/box/mask与negative point启用、`margin=0,gamma=4,strength=30`；最后按各row的SAM预测IoU argmax选mask，不含native identity候选作为最终回退。此为公开作者代码审读，不是对本地执行payload再审计。

[固定commit的++ README](https://raw.githubusercontent.com/linyq2117/SAMRefiner/4bb7f95738c7a2e805f189210fb09e02a2197557/SAMRefiner_plus/README.md)要求安装plus内带LoRA的`segment-anything`，不是调用基础函数便自动启用++。[作者DAVIS训练示例](https://raw.githubusercontent.com/linyq2117/SAMRefiner/4bb7f95738c7a2e805f189210fb09e02a2197557/SAMRefiner_plus/iou_adaption_davis585.py)只放开name含lora的参数；box-only、1iter，用候选与init_mask（非GT）的IoU排序，MarginRankingLoss margin=.02；SGD lr=.01、momentum=.9、wd=5e-4，默认1epoch，每5样本累积一步。脚本evaluation用多提示但1iter。GT只用于该示例评价；某些二手页面误说训练用真IoU，不应沿用。

该示例针对DAVIS-585，不是现成SAR FIT/DEV实例AP协议。若日后适配++，需在不接触DEV指标的FIT范围确定伪监督、适配预算和checkpoint，仍保留native S0作为外部候选分数；SAM内部IoU选分支与检测重评分是两件事。不能在本次基础版失败后直接运行++并把DEV择优写成预先确认的强比较。

## 旁证与检索范围

**Price等，Mastering SAM Prompts: A Large-Scale Empirical Study in Segmentation Refinement for Scientific Imaging，TMLR11/2025**，[原始论文](https://openreview.net/pdf?id=cWcTQMpqv6)、[作者代码](https://github.com/sprice134/MasteringSamPrompts)，已系统研究科学显微图像的提示组合并报告coarse-mask提示可能降低AP。它支持“域与提示相关、不能默认总会增益”，不能把其显微数据结论直接外推SAR。

局部误差/不确定性选择的更早直接家族还有[PointRend，CVPR2020](https://arxiv.org/html/1912.08193)、[Mask Transfiner，CVPR2022](https://arxiv.org/html/2111.13673)与[SegFix，ECCV2020](https://arxiv.org/abs/2007.04269)。Mask Transfiner只细化检测出的incoherent节点；SegFix以较可靠内部标签修边界；因此“只修不确定局部/沿边保留主体”的宽泛说法已有多条先例。无需为本轮查新逐一新开训练臂。

检索实际使用SAMRefiner+2025/2026、mask/segmentation refinement+coarse quality/preserve/uncertainty、SAR ship/SAM domain gap、作者/出版域限定，查看了2026-04至10月可检索条目。还核到[Prompting Foundation Models for Zero-Shot Ship Instance Segmentation in SAR Imagery，2026-04-20预印本](https://arxiv.org/abs/2604.17920)：SAR训练YOLOv11框提示SAM2，不是强coarse精修保形实验，不并入核心直接表。未找到与本项目同设置的“强纯SAR船native高IoU遭SAMRefiner损伤”论文，**未命中不等于不存在，也不构成发现首证或创新证明**。

## 可执行判断顺序（本稿不授权或运行新实验）

1. 先完成当前固定native IoU=.50匹配、同query的POSTHOC诊断；不重匹配、不加Ω/oracle，不把条件人口的IoU当整体新AP。记录全380 GT覆盖范围、未匹配/忽略者及受损和获益计数。
2. 如果主要FN增长且面积收缩，才保留“抹掉有效前景”假设；如果FP显著增长则另查外扩/背景吸入。若两者都有，要报告两类，不强行归一个机制。
3. 只有之后独立预锁、且分支原始数据可得时才检验“mask生成有好候选但排序选错”；若所有生成候选都不如native，排序适配不是充分解释。这个future诊断未执行，不回填当前结果。
4. 若有真实内部损伤，再决定是否需要一个RefineMask风格局部保留控制；若残差获益主要是边界而内部本来稳定，也不能靠命名再造创新。保留已有FULL/PLAIN终点与容量控制优先级，避免立即堆模块。
5. 最终需要同候选/S0的强自动精修替代；当前基础SAMRefiner是已运行点，++/PromptMoE/SegRefiner为待定候选。选定后锁协议，明确公开迁移、额外训练与资源成本。不能把某个负基线当作足够强的外部方法全面结论。


## Root实际补充（独立于文献自报）

[固定native几何诊断](U57_FIXED_NATIVE_GEOMETRY_DIAGNOSIS_ACTUAL_20261025.md)现已实际完成：374/380固定配对中，SAM的实例均值/中位数与pooled IoU均下降，典型FN增加且面积偏小，总FP同样增加；不再只停留于待检验的几何解释。该结果仍不识别作者内部三分支原因，不代表所有380 GT或全部配方。

本轮Root另核到CVF正式PromptMoE页面、SAMWater官方出版索引/正文、UnfoldCRF arXiv记录与SAMRefiner原文。基础SAMRefiner结果不代表++；后续强对照仍须固定协议、训练/预训练成本及适用范围。文献事实与本项目实测分开引用。
