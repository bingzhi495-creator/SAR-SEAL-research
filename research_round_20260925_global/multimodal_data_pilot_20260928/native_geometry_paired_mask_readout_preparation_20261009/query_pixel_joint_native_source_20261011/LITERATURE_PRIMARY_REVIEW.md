# 9441 参数 query/pixel 残差与质量评分：有界一手文献审查

核查日期：2026-10-11。范围仅公开论文、作者代码及本地 `.py` 接口；没有读取实验数据、权重、资格文件或结果，没有执行科学代码。下面列出八篇直接先例，区分发表论文和预印本；不是完整文献覆盖或新颖性证明。

**判断：结构原语的新颖性很弱。** 像素特征与实例条件预测的 MLP 融合、点监督、局部边界提示、残差修正、质量重排序，以及由同一概率场计算 soft-count 质量，均有直接先例。当前组合可以是待验证的工程贡献候选；贡献须落在明确的接入约束、输入信息、监督与决策关系，以及可重复的效用，而非给这些原语换名称。现有证据不足以称 TGRS-ready。

## 候选的实际比较单位

当前小头把 native F256 和 e0 分别投影为 16 维，连同原图 native logit L、sigmoid(L)、严格 L>0 的二值 support B、3×3 概率 contrast，组成 36 维输入，经 32 维 SiLU 和标量输出产生 u；共八个参数张量、9441 个标量。3×3 contrast 本身就是空间邻域信息，F 也包含原网络上下文，所以“原头没有空间信息”不成立。

要分别说明三件事：posterior logit L+u；受限几何 logit L+tanh(u)；从指定二值 support 与 posterior 算出的质量。后两者不必对应同一个决策 logit。联合训练 native 或放开幅度只是训练/部署选择，不是单独的创新证据。matched-query BCE 也不会自动校准全部 100 个推理 query；重复 query 可能有较高 GT IoU，不能因未匹配便把其质量标签固定为零。

## 八篇直接先例

| 一手论文及身份 | 核实的方法与官方代码位置 | 与候选的重合及实质差别 |
|---|---|---|
| **Mask Scoring R-CNN**，CVPR 2019，pp.6409–6418。[正式论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)；[作者全文](https://arxiv.org/pdf/1903.00241)。 | §3.2 用 RoI 空间特征和预测 mask 估计 mask IoU，回归 GT IoU，score 为 class score×predicted IoU。[官方 feature extractor](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/roi_maskiou_feature_extractors.py) 的 `forward` 对 mask max-pool 后与 RoI 特征 concat；[loss](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/loss.py) 为正样本 L2；[inference](https://github.com/zjhuang22/maskscoring_rcnn/blob/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/inference.py) 作 score 乘积。 | 质量预测和分类分数乘积已有。候选用像素 posterior 的预算函数而非另一个 IoU regressor。仅 e0+四统计量的 9441 标量对照缺少该论文的空间特征，不能代表完整 MSRCNN 或排除输入信息混杂。 |
| **PointRend: Image Segmentation as Rendering**，CVPR 2020，pp.9799–9808。[正式论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)；[作者全文](https://arxiv.org/pdf/1912.08193)。 | §3.1–3.2 把 fine-grained pixel feature 与实例 coarse prediction concat，交共享 MLP；训练混合不确定点和均匀点，推理逐级 subdivision。[官方 `StandardPointHead.forward`](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_head.py) 和 `roi_mask_point_loss` 直接实现 concat/MLP/BCEWithLogits；[sampling](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py) 实现不确定点选择。 | 像素条件 MLP 和 coarse-instance cue 点 BCE 已有。显式 e0 投影、附加 contrast、一次原图 dense 残差和预算评分是不同具体实现；不能将其泛称为首次 query-conditioned refinement。原论文用更细粒度特征及自适应 subdivision，候选不等于完整复现。 |
| **RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features**，CVPR 2021，pp.6861–6869。[正式论文](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html)；[作者全文](https://arxiv.org/pdf/2104.08569)。 | §3.1–3.3 用高分辨率 semantic feature/mask 与 instance feature/mask 多阶段融合；式1–3 定义边界区域与边界监督，式6 在推理保留内部、更新边界。[官方 `SFMStage`/`MultiBranchFusion`](https://github.com/zhanggang001/RefineMask/blob/main/mmdet/models/roi_heads/mask_heads/refine_mask_head.py) 含 concat、三支 dilated 3×3、upsampling 和 mask hints。 | 局部/多尺度特征、粗 mask 提示与边界修正已有。候选没有该额外语义分支与高分辨率 FPN 融合，不能把 native F 的双线性输出称为新高分辨率信息，也不能以冻结单层 F 的对照代表该方法。 |
| **Mask Transfiner for High-Quality Instance Segmentation**，CVPR 2022，pp.4412–4421。[正式论文](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html)；[作者全文](https://arxiv.org/html/2111.13673)。 | §3.1–3.3 用 learned incoherence 定位需细化区域，以多层 quadtree 的 pixel nodes、fine feature、coarse prediction、position 和局部邻域进入 transformer。[官方 `mask_head.py`](https://github.com/SysCV/transfiner/blob/main/detectron2/modeling/roi_heads/mask_head.py) 的 `get_incoherent_mask`、多层点处理和 `mask_loss_refine` 可核；发布版含实现差异，README 称其为 reimplementation。 | error-directed sparse refinement 与局部/非局部交互已有；其 transformer query 是 pixel node，不应混同 native object e0。候选的简单 pointwise 头没有 quadtree 或 learned incoherence；因而不是该架构复现，也不能把“加入邻域/attention”本身当新贡献。 |
| **SAMRefiner: Taming Segment Anything Model for Universal Mask Refinement**，ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html)；[作者全文](https://arxiv.org/html/2502.06756v1)。 | §3.2 从粗 mask 生成 point/box/mask prompts、用冻结 SAM 细化；§3.3 SAMRefiner++ 在 IoU MLP 末层作 LoRA-style adaptation，式4 是候选间 margin ranking，以与 coarse mask 最大 IoU 的候选为 j。[官方代码](https://github.com/linyq2117/SAMRefiner) 发布 `SAMRefiner_plus`；该具体适配实现未在本次逐行核完。 | 论文明确 ++ 只改变选择，不改变 mask generation。它不是共享 pixel posterior 的 soft-budget 头；使用 SAM 预训练与额外视觉表示，也不构成现有教师特征/参数预算下的公平直接控制。 |
| **FlowSeg: Dynamic Semantic Guidance for LLM-Conditioned Segmentation**，ICML 2026，PMLR 306:155588–155605。[正式论文入口](https://proceedings.mlr.press/v306/zhang26bb.html)；[作者全文](https://arxiv.org/html/2605.29461v1)。 | §3.4 式6给 boundary gate；式7以 compressed pixel feature+raw logit 产生 α·tanh 残差，再与 boundary gate 相乘加回 raw mask；§3.5 式10仍用其 Hungarian BCE/Dice/class supervision。[项目页](https://zkzhang98.github.io/FlowSeg_page/) 已核，但该页 Code 当时指向论文，未验证已发布实现。 | compressed feature+raw logit 的 bounded correction 已有。候选无 LLM 条件化、无该动态 query flow/boundary gate；L+u 与 L+tanh(u) 的目标区别可研究，但不是首次 residual 或 tanh correction。FlowSeg-inspired 小控制须明确非完整 FlowSeg 复现。 |
| **Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation**，Machine Learning 115:176，2026-07-13；早期 arXiv 版本为 2024。[期刊全文](https://link.springer.com/article/10.1007/s10994-026-07096-w)。 | §4.2 式22：SDC=2ΣpB/(Σp+ΣB)；B 可由同一 p 经式16阈值得到。方法是训练后、由已有概率输出直接估计质量，不另外训练 refiner。[官方 `SoftDice.metric`](https://github.com/gama-ufsc/soft_dice_confidence/blob/main/medical-imaging/src/confidence.py) 与 [`soft_dice`](https://github.com/gama-ufsc/soft_dice_confidence/blob/main/medical-imaging/src/metrics.py) 默认 eps=0。 | 下面的 IoU-budget 正是其 Dice 形式的代数变换，换 Dice 到 IoU 或给 eps 取新名字不是新公式。不同点在实例 all-query ranking、posterior 的训练来源与部署 support；语义 selective prediction 的理论不自动保证 matched-only head 的实例校准或 COCO AP。 |
| **UnfoldCRF: Structured Mask Refinement with Image-Conditioned Latent Regions**，arXiv v1，2026-09-27；未核实接收 venue。[版本记录](https://arxiv.org/abs/2609.33996)；[作者全文](https://arxiv.org/html/2609.33996v1)。 | §3.1 式1–4把 residual unary、image-conditioned local pairwise、latent region energy 展开为迭代推理；region 含 NULL state。§4/附录控制 input/encoder、参数和 stage、监督；报告 held-out generator 与 made-worse/Harm。论文承诺公开代码，本次未验证已发布实现。 | structured refinement 已有更强直接机制主张。候选不含这些 CRF 算子；该预印本更适合做证据标准：同输入/容量/深度控制、generator shift、错误修复和损伤一起报。其结果尚不能替代本项目公平复现，更不能当已认证 SOTA。 |

## 预算公式不能列作新贡献

令 I=ΣpB、A=ΣB、T=Σp，当前质量 q=I/(A+T−I+eps)。在 eps=0 时，SDC 式22的 D=2I/(A+T) 满足 q=D/(2−D)。若定义 denominator-only D_eps=2I/(A+T+eps)，同样精确成立。该步是代数，不是新增理论。[SDC 一手式22](https://link.springer.com/article/10.1007/s10994-026-07096-w)。

官方 SDC 代码在非零 eps 时使用 (2I+eps)/(A+T+eps)，因此不能声称其任意 eps 实现与当前函数逐位相同；默认 eps=0 一致。空 support 的约定、FP32 舍入及概率校准另须声明。c_ship×quality 的排序框架也已见 MSRCNN。用同一 p 定 support 并算质量已有直接先例；把 p 的来源换为学习残差，是监督/集成差别，不是公式新颖性。

## 最短强基线与可证伪差异

| 待检验的具体贡献 | 最少控制 | 能支持的结论与失败条件 |
|---|---|---|
| 显式 query 信息是否超出 coarse mask 身份提示 | 同 F、L/p/B/contrast、参数和点数，保留 16 维接口，把 e0 分支换为共享 image context；保留 L 的实例条件。 | 稳定差异才支持 e0 的额外作用；不是比较有无全部 query 条件。单教师/单 seed 或仅 AP 差异不能证明机制。 |
| posterior 估计与受限几何决策分离是否有实用价值 | 同容量/seed/点监督的 BCE(L+u) 与部署一致的 BCE(L+tanh(u))；同一模型分别交叉固定几何和评分。该控制借鉴 FlowSeg 原语，非完整复现。 | 报 FN/FP、boundary IoU、GT-best100、made-worse 和排名；cross 指标非可加、非独立因果。posterior 更好不保证 bounded geometry 更好，反之亦然。 |
| dense 质量预算是否优于直接质量学习而非多输入信息 | feature-informed scalar IoU 强对照应同用 F/e0 和 mask 信息，监督全部 query 对 GT 的最大 native-mask IoU，允许 duplicates 非零；披露容量、张量数、池化及计算差别。 | query+四统计量的同参数头只能回答较窄问题。若收益消失于 F-pooled 控制，不应把多输入信息收益归给预算公式；matched-only BCE 不足以证明全 query 校准。 |

公开先例并不要求所有整合都首次出现。可能成立的工程贡献，是在不新增基础模型或高分辨率特征的限制下，给出低容量、保留 native query/class/renderer 的实际接入，展示监督/几何/质量之间可重复且可解释的权衡。需有重复 seed、固定训练与评价规则、未用于选择的评估或新数据、足够训练证据，并明确算力成本。当前文献审查不提供效用、收敛、物理斑点机制或外部 SOTA 结论。

## 当前环境的可迁移性：仅 SOURCE 核查

- **最短形状/点监督基线是 PointRend-inspired adapter。** 已安装 MMDetection 的 [`MaskPointHead.forward`](</E:/RelocatedFromC/Users/houbingzhi/anaconda3/envs/seal_gpu/Lib/site-packages/mmdet/models/roi_heads/mask_heads/mask_point_head.py:97>) 接受 fine-grained feature 和 coarse feature，line112 concat；可以在同 native F、原图点和实际固定匹配上适配其数学头。原 `loss_and_target` 需要 RoI/sampling/GT 数据结构，不能直接换入现有 whole-frame matched path；要完整 PointRend 仍须保留其 fine FPN 与 subdivision，不可把受控移植命名为完整复现。
- **质量基线应保留空间特征。** 已安装 [`MaskIoUHead.forward`](</E:/RelocatedFromC/Users/houbingzhi/anaconda3/envs/seal_gpu/Lib/site-packages/mmdet/models/roi_heads/mask_heads/maskiou_head.py:94>) 对预测 mask sigmoid/max-pool 后与 spatial mask feature concat；其 target path 是 positive RoI 的 full-GT-area 修正，不等同全 100 query 的 max-GT 标签。需要明确适配输入和标签，不能只匹配标量参数数目。
- RefineMask 官方实现使用旧 MMDetection API 与额外 semantic/multi-scale 模块；MaskTransfiner 使用 Detectron2/quadtree 和额外依赖；SAMRefiner 引入 SAM 表示。它们不是当前冻结 M2F 环境的即插即用等预算对照。FlowSeg、UnfoldCRF 本次未核到可用官方实现，不能声称已具备完整可复现基线。

本次只核接口与公开方法，不运行这些组件；也不据文献审查决定启动新训练。对最新预印本的缺代码状态和未找到更同构论文，均不能推导“候选没有先例”。
