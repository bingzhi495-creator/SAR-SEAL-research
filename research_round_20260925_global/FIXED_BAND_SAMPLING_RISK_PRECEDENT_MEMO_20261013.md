# 固定可编辑带下的采样风险与全图掩膜质量：定向先例核查

日期：2026-10-13。状态：**普通研究备忘录；候选假设未执行，不是新结果或创新认证**。本次只读取独立 GitHub staged 的 N07 cache_fit.py、N10 train.py 普通 SOURCE 副本及公开原始论文/作者代码；未读取、复制、哈希或执行绑定 SOURCE、RAW/SPEC、数据/GT、权重、科学缓存，未运行任何科学数值。ROOT 负责现有资格链、实验选择及发布。本文不更新 C 的创新性结论，也不宣称最新文献穷尽。

## 1. 判断先行

PointRend、Mask2Former 和所核对的 MaskDINO sampled-mask 路径对不确定性偏置采样直接计算点 BCE / Dice，没有用逆点采样概率恢复全图均匀测度。这里的 importance sampling 是困难点选择策略，**不是统计学意义上已做重要性修正的全图风险估计**；但不修正并不等于算法有错误，它可能刻意优化困难区域。PointSup 的稀疏人工点监督也不能与“有完整 FIT 标签但用点减少计算”混为一谈。

本项目固定 |L|<=2 带与均匀带内采样使抽样测度可知，因而可讨论可复现的全图质量估计。但是：

- 只加逆概率 BCE 在当前头上仅产生**实例级梯度缩放**，不是新的带内空间建模。
- 精确固定区域统计量加上采样可编辑区域统计量，是已有分区 Monte Carlo / 多重重要性采样思想；加入原生全图质量锚点后也属于已知控制变量/差分估计。
- TP、预测质量的总和估计无偏，**不意味着 IoU/Dice 比值或其训练梯度无偏**。
- 现有缓存足以计算线性风险修正、采样几何质量及精确 GT 总面积，但**不足以声称有精确带外 native softmass / GT 交集**。后者需要另登记并核验冻结 FIT 统计量补充，本轮没有读取或生成它。

因此最合理的候选是：在固定原生表征、原图坐标、固定可编辑区域和固定评分的条件下，检验一个全图几何目标近似是否能减少小目标形状误差；价值只能由公平实验证明，不能先把它命名为已成立创新。

## 2. 本项目当前采样和优化的 SOURCE 事实

独立 staged SOURCE 路径，相对于 github_release_20261011：

- research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N07_frozen_E24_original_detail_point_refinement_source_20261012_01/cache_fit.py，sample_points 及 cache_image。
- 同一 preparation 下 N10_PR_FL91_direct_precedent_control_source_20261013_01/train.py，RECIPE、endpoint 与 train_epoch 中的 BCE 调用。

设一对已冻结匹配的 query/GT，其原图像素域为 Ω，N=|Ω|；B={x:|L(x)|<=2}，n_B=|B|。B 非空时，缓存先用 randint 从 B **有放回**抽 768 点，再用 randint 从 Ω 有放回抽 256 点，拼接为 m=1024 点，重复点保留；空 B 则 1024 点均匀来自 Ω。B 完全由原生 L 决定，不按 GT 选点。标签是这些整数原图像素上的 hard GT，F 是原生粗特征经固定 renderer 的采样，不是原始复数 SAR。

packet 的普通 SOURCE 定义包含 point_indices、native_logits、targets、raw_GT_area、band_counts，以及 replacement=True、duplicates_preserved=True；**源码中没有完整 FIT M 归档字段**。此处只核对字段定义，不读取 packet 实物。

N10 同一组缓存点跨所有轮复用；每个 GT 的 1024 点做 mean BCE，一实例一更新，无累积，固定 8 轮、13288 更新；AdamW lr=3e-4、weight_decay=1e-4、constant lr、no gradient clip。端点 BCE 是同一缓存的完整 1661 个实例平均，不能充当独立密集全 FIT 几何审计。

## 3. 直接点采样先例

| 先例 | 原始论文/官方实现实际核对 | 与候选的差别及不能推出的结论 |
|---|---|---|
| PointRend，CVPR 2020 | 官方 get_uncertain_point_coords_with_randomness：均匀候选插值 logits，按 uncertainty top-k 选一部分，再补均匀随机点。roi_mask_point_loss 直接 mean BCE，weight 只是忽略标签 -1；不是点包含概率。 | 训练重视难点已有直接先例。其 top-k 的选择密度不能简单写成“0.75/B+0.25/N”，也不是本项目的固定整数带均匀抽样。不能宣称发现了 PointRend 的错误。 |
| Mask2Former，CVPR 2022 | 官方 criterion 的 loss_masks 在 no_grad 下用上述不确定点坐标，后续 plain sampled BCE 与 sampled Dice。Dice 的和式没有逆采样概率。官方 matcher 独立使用所有 masks 共用的均匀随机坐标。 | 区分匹配成本采样与训练损失采样；不能把 matcher 的均匀点当作 loss 已被修正的证据。全图密集风险、偏置点风险与采样比值是不同目标。 |
| MaskDINO，CVPR 2023 | 官方 criterion 的 sampled-mask 路径复用不确定点采样，直接 sigmoid_ce_loss_jit 与 dice_loss_jit。 | 增加后续实例方法的代码证据，但不代表所有后续方法都如此。框损失/分类权重与点采样概率修正不是同一件事。 |
| PointSup / Implicit PointRend，CVPR 2022 | 官方 PointSup head 从人工点标注取坐标/标签，调用同一 roi_mask_point_loss。 | 稀疏标注监督不提供完整像素真值，也通常不提供每个像素的随机标注包含概率。不能移植“精确带外 GT 交集”而仍宣称只需点标注。 |
| Prior2Former，ICCV 2025 | 论文将 75% 预算用于高 evidential uncertainty，25% 随机点。作者 criterion 的 sampled binary/beta mask 路径使用 sampled BCE / Beta loss / Dice；带可选正标签抽样、类/正负平衡项，但没有由 sampler 返回点测度后进行逆概率质量估计。 | 不确定性与 Dice 联合设计已有近先例；其 evidential head/对称 Dice 不等于 HT 校正。对 optional 分支与论文默认配方的配置一致性未复现，不能把代码中所有选项说成论文主结果。 |

来源：[PointRend 官方采样](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py)、[官方点损失](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py)；[Mask2Former criterion](https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/modeling/criterion.py)、[matcher](https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/modeling/matcher.py)；[MaskDINO criterion](https://raw.githubusercontent.com/IDEA-Research/MaskDINO/main/maskdino/modeling/criterion.py)；[PointSup head](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointSup/point_sup/mask_head.py)、[Pointly-Supervised Instance Segmentation](https://arxiv.org/abs/2104.06404)；[P2F 论文](https://arxiv.org/html/2504.04841v2)、[作者 criterion](https://raw.githubusercontent.com/TUM-DAML/Prior2Former/master/Prior2Former/models/mask2former/modeling/criterion.py)、[作者采样代码](https://raw.githubusercontent.com/TUM-DAML/Prior2Former/master/Prior2Former/models/mask2former/detectron2/projects/point_rend/point_features.py)。这些 main/master URL 是本次机制核查入口，正式复现应锁定版本，本文不把浮动分支当作可复现实验版本。

## 4. 区域继承与统计修正的直接先例威胁

**区域整合先例。** RefineMask 的 BAR 在后续阶段按预测和 GT 边界的并集选择监督区域，计算区域平均 BCE；推理时在预测边界更新，带外继承前级 mask。作者源码 BARCrossEntropyLoss 明确先生成 boundary_region，再对 reduction=none 的 BCE 区域求和/区域像素数；合成时替换 pre_boundary 内值。故“只细化边界、保留其他位置”不是新原语。与当前候选不同的是：RefineMask 区域跨阶段依赖当前预测、训练区域用 GT 边界，目标是区域 BCE，而不是固定 native logit 带中的已知测度全图 soft-IoU 估计。[原论文 §3.3](https://arxiv.org/html/2104.08569v1)、[作者 BAR 源码](https://raw.githubusercontent.com/zhanggang001/RefineMask/main/mmdet/models/losses/cross_entropy_loss.py)。

**统计先例。** Horvitz–Thompson 已处理不等包含概率的有限总体总和估计；保留重复抽样的 draw-probability 加权应区别于去重集合 HT。本项目固定 768/256 分层配额的 pooled 公式更准确叫 deterministic-mixture / balance MIS 估计，不能把 1024 点谎称 iid 来自混合分布。Veach–Guibas 已给出多分布样本的无偏加权合并；不是新的采样校正定理。[HT 原论文](https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446)、[Veach–Guibas 作者论文入口](https://cs468.stanford.edu/papers/combine/)、[原论文 DOI](https://doi.org/10.1145/218380.218498)。

**学习与视觉中的重要性/差分先例。** survey training 的 HT risk minimization 已是研究对象；CVPR 2019 Histopathology 工作用 p/g 重新加权分割训练样本损失，虽其单位是图像/patch而非带内像素。2026-07 的 Scalable Model-Assisted Multi-Target Estimation 直接在检测/分割图像集合讨论精确模型总量加 sampled correction、importance sampling、difference/control-variate 与 ratio estimators。后者是测量而不是可微掩膜训练，不能说完全等价，但足以反驳“精确基线＋稀疏残差估计”为新统计思想。[HT risk 原论文](https://arxiv.org/abs/1610.03316)、[CVPR 2019 原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Robust_Histopathology_Image_Analysis_To_Label_or_to_Synthesize_CVPR_2019_paper.pdf)、[2026 预印本 §2.1](https://arxiv.org/html/2607.17581v1)。本次正文核查使用2026文献的arXiv版本；[作者arXiv摘要页](https://arxiv.org/abs/2607.17581)当前标注已被UAI 2026接收。接收状态来自作者页面，本次未另核会议正式proceedings，也未把该测量研究说成分割训练方法。

本次没有确认某篇分割训练论文逐项使用“冻结原图 |L|<=2＋固定带外精确质量＋带内 MIS＋原生评分保持”这一整个组合，**这一检索结果不是新颖性证明**。在已有区域细化与经典估计方法下，需要进一步查直接等价方法，并用误差证据说明组合为什么值得做。

## 5. 已知抽样测度：可以证明的线性部分

B 非空，定义 b=n_B/N、a=0.75+0.25b，以及 pooled 测度

q(x)=0.75·1_B(x)/n_B+0.25/N。

对固定 θ 的可积像素函数 f，保留全部重复点，

S_hat(f)=(1/1024) Σ_{j=1}^{1024} f(x_j)/q(x_j)。

虽然两段 draw 分布不同，期望按 768 个 band draws 和 256 个 uniform draws 分别求和，有 E[S_hat]=Σ_Ω f。若只估可编辑域，替换 f 为 1_B f。空 B 时定义 q=1/N；头没有可编辑点，候选不凭空产生 head 梯度。去重后若另用 HT，B 内包含概率是 π_B=1−(1−1/n_B)^768(1−1/N)^256，带外 π_O=1−(1−1/N)^256，估计器 Σ_{unique x}f(x)/π(x) 才是与这种去重设计匹配的形式；不能去重后沿用 draw q 权重。本文不建议为本候选改缓存去重。

variance 的核查须按两个 strata 的 draw 设计计算；“实际保存缓存的一个确定 seed”不是估计无偏性的数值证据。若使用重加权 self-normalized mean、裁剪权重或裁剪质量，目标与有限样本偏差均改变，必须另行命名和记录。

## 6. weighted BCE 的关键反证：当前配方只改实例尺度

头保持 L 不变的带外点有 ∂ℓ/∂θ=0。全图像素 mean BCE 的 MIS 权重为 w(x)=1/[N q(x)]。在 B 内所有点的权重相同：

w_B=b/a；带外 w_O=4。

因此，对**同一实例、同一实际缓存**，严格有

∇θ mean_j[w(x_j) BCE_j]=w_B ∇θ mean_j[BCE_j]。

这里不需要独立随机性假设；只需旧头门控保持、B 固定及 BCE reduction 一致。按整个原图均匀 BCE 作目标，w_B 会降低小 b 实例的相对梯度，未必有利于小船。若目标改成 B 内均匀 mean BCE，权重在可训练点为 1/a，也是实例标量；必须先明确想优化哪种总体，而不是看到“bias”便机械纠正。

AdamW 的自适应二阶矩、eps 和跨实例历史意味着实例尺度并不等价于一个简单恒定 learning rate。不能宣称 weighted BCE 没有实际作用，也不能把收益解释为采样校正产生了新局部特征。精确带外 BCE 只是 θ 无关常数，加入它本身不改变梯度。若做此控制，要报告它是实例/带面积权重控制，不是结构创新。

## 7. 几何候选必须使用一致的 soft mass

设 p0(x)=sigmoid(L(x))；pθ(x)=sigmoid(L(x)+1_B(x)uθ(x))，y(x) 是仅 FIT 的整数原图 GT。定义

T(θ)=Σ_Ω y pθ，P(θ)=Σ_Ω pθ，Y=Σ_Ω y；
soft-FP=P−T，soft-FN=Y−T。

D(θ)=(2T+ε)/(P+Y+ε)，J(θ)=(T+ε)/(P+Y−T+ε)。

ε 的单位是原图像素 mass，须固定在该尺度；不能直接沿用 1024 点 Dice 的 +1 后又把总量缩放为原图 mass 却称 smoothing 一致。

带外 T_O=Σ_O y p0、P_O=Σ_O p0 为 θ 无关常量。若它们已被独立核验，

T_hat=T_O+S_hat(1_B y pθ)，P_hat=P_O+S_hat(1_B pθ)，Y 精确；

将二者代入 D/J 得到的是 **采样 ratio surrogate**。固定域、teacher、匹配、renderer、标签定义均不变时才能一次计算 T_O/P_O。若 teacher/匹配/band 改动或改成逐轮 uncertainty top-k，不得复用旧常量，也不能继续沿用同一 q。

这不是 hard TP/FP/FN：hard native 0 阈值支持与 sigmoid mass 不能混用后仍称全图 soft Dice。可以另定义“带外 hard、带内 soft”的 hybrid surrogate，但必须承认它换了目标，密集审计也用同一公式。

几何比值中带外虽然不可训练，却参与分母与分子，使可编辑点的梯度幅度受完整 mask 状态影响；这才与区域 point-Dice 的差别有关。任何带外高置信度错误仍不可修复，不能承诺全图错误消失。

## 8. 当前缓存可做的质量近似，与缺少的统计量

利用现有 raw_GT_area=Y，可估 T_hat=S_hat(y pθ)、P_hat=S_hat(pθ) 后做 sampled D/J。无需新教师 forward，但 256 个全图点承担当量可能很大的带外贡献；不能把它叫精确带外锚定。

对微小 GT，T_hat 可超过 Y；P_hat>=T_hat 并不保证 D_hat/J_hat<=1。即使为正的 IoU 分母不至于自动变负，超过 1 的 ratio、强噪声与梯度波动仍可能出现。不得只因用了 exact Y 就宣称概率范围、稳定性或无偏 IoU 得到保证。clamp 到 [0,1]、TP 截断到 Y、增加 ε 均会改变估计与梯度；不能作为不披露的修复。

真正在固定带外精确求 mass 需要新的、前瞻登记的 frozen FIT 统计量补充，并验证与既有缓存的 query 匹配、native point logits、GT 面积、原图 renderer 一致；本 memo 不安排这项执行，也不假装现有 packet 已含它。

若已付出同一次完整 native/FIT 统计量扫描，可顺带定义更强的**原生总量差分锚定**，用 P0=Σ_Ωp0、T0=Σ_Ωyp0：

P_hat_delta=P0+S_hat(1_B(pθ−p0))；
T_hat_delta=T0+S_hat(1_B y(pθ−p0))。

cold u=0 时 sampled correction 精确为零，所以 ratio 精确回到原生 soft ratio；比仅精确带外而继续抽样带内原生 mass 的形式更有诊断价值。它是经典控制变量/差分估计，**不能称新估计定理**；近原生时减少 mass 方差是待检验判断，偏离原生后不保证持续较小方差，更不保证 ratio/训练梯度无偏。可能出现负估计 mass 或不合法 ratio 的风险仍须直接审计，不隐藏修补。

## 9. 不能跳过的偏差与适应性边界

1. 对固定 θ，正确线性估计的无偏性可成立；然而 E[X_hat/Z_hat]一般不等于 X/Z。denominator 远离零且满足展开条件时，二阶近似包含 −Cov(X_hat,Z_hat)/Z²+X Var(Z_hat)/Z³，这只是解释 bias 来源，不是本项目数值结论。把 ratio 自动微分，也通常不产生 dense-ratio gradient 的无偏估计。
2. 训练 θ 本身由同一缓存抽样决定，所以“固定 θ 对随机缓存无偏”不等于“适应训练后的端点无偏”。跨轮重复缓存可以保证公平和效率，但不能宣传每步是相对于全图的条件无偏 SGD。独立密集 FIT 子集审计与独立 sampler seed 是必要的诊断候选，不依赖 DEV GT 调整推理。
3. 稀疏小目标的 sampled targets 可能为零。exact Y 避免把真实非空实例当作没有 GT，但不能消除 sampled TP 的噪声。ratio 的方差、梯度方向/幅度、非法范围频率须分 GT 面积和 b 报告。
4. per-instance mean Dice、全 batch/global Dice、per-image pixel BCE 是不同总体。必须固定同一聚合定义，不能靠偷偷换 aggregation 收益后归因新估计器。
5. dense soft IoU/Dice 也不是 COCO AP 的一致性保证；固定 S0 排序下几何改动可能改善某些 IoU 阈值、损害另一些。需要完整 all100 支持、空 mask 规则和固定 scores 的既有评估约束。现阶段不宣称超越外部方法或 TGRS 充分性。

关于目标本身，Dice/Jaccard 与 CE 的差别已有理论及实验研究；soft-Dice 在固有不确定性下还可有体积偏差。它们支持“需验证目标匹配”，不是证明本项目必须用 Dice。[Dice/Jaccard 原论文](https://arxiv.org/abs/1911.01685)、[soft-Dice volume-bias 原论文](https://arxiv.org/abs/2211.04161)。

## 10. 最小且有判别力的未执行对照

这不是 launch 指令。ROOT 应先依据已核验误差判断是否值得登记。固定同一头/完整初始化、冻结表征与匹配、原图带、1024 个点、数据划分、seed、实例/图像顺序、optimizer 与固定终轮；结构/容量/推理计算量不得顺便增加。新增统计量生成成本另记，不能从效率账本抹除。不能挑 DEV 最优轮替代固定终轮。

| 对照 | 未执行定义 | 回答的问题 |
|---|---|---|
| A：旧 mean BCE | 完全原配方，作为配对重跑基准 | 当前可复现基线 |
| B：BCE + unweighted sampled Dice | 同一1024点 plain Dice，固定预登记 ε/λ | 普通几何损失是否已足以解释收益；避免把“第一次加 Dice”当新方法 |
| C：BCE + corrected global ratio | 与 B 同样 Dice family、同样 λ；用 known measure masses；如果有补充资格则采用 exact fixed part 或 native-delta anchor，清楚注明选哪一种 | 整个 corrected/anchored 组合是否优于 plain point-Dice；三臂筛选仅支持组合效用，不支持分别归因概率修正和精确锚定 |

若 C 相对 B 有可靠方向，才值得加一个最有区分力的结构不变对照：**同 q 修正的 sampled full-domain mass ratio，但带外仍采样、不使用精确锚点**，与 C 比较只隔离已知部分/差分 anchor。B 与此臂比较可检验“测度/全图定义”组合，仍须承认 exact Y 与 sampled Y 若不同也混入；要对概率修正单独归因，二臂应采用同一个 GT-area 定义和 ε 单位。不是为了凑创新点而自动展开四臂。

weighted BCE 是更便宜的尺度诊断候选，优先利用第6节等式认识其机制；若训练需用它，另配同 w_B 的实例 scalar BCE 作为实现/因果核对，不宣称新局部机制。同一 λ 的 BCE+Dice 是配方公平起点，但不同目标的梯度尺度仍可能改变 AdamW 轨迹；允许同范围 FIT 内配方检验时必须给所有臂相同搜索预算，不能只精调 C。

研究是否保留该方向，应看至少以下反证：

- C 不胜 B：不支持采样校正/anchor 提供额外效用，可能普通 Dice 足够；仅局部短预算不等于方法总体失效。
- C 只提高 sampled loss、密集 FIT 几何不改善：疑似 ratio bias、缓存适应或尺度问题。
- gains 只随 b / GT area 的权重变化出现，weighted/scalar BCE 即复现：实例重权重可以解释，不支持新的几何估计机制。
- 密集梯度/质量近似误差与非法 ratio 随 tiny GT 明显升高：不能称小船稳定改进；需要重新选择 surrogate 或采样量，不隐瞒截断。
- 一种子向好另一种子退化，或跨数据集趋势不保持：不支持稳定核心贡献。

预期的短预算、两 seed 仅用于前瞻筛选；充分训练、重复 seed、直接强先例和跨数据集验证仍决定最终证据层级。可以先用少量 FIT 的**密集 oracle loss 审计**区分目标本身与 MC 近似，明确它是诊断，不与稀疏方法混称相同算量；本次没有执行或安排任何这种数值工作。

## 11. 可以保留的候选表述

暂时只保留一个研究问题：**在公共 SAR 船舶实例分割的冻结原生 query carrier 上，当预测只能修改原图固定 logit 带时，能否通过与该支持域一致的全图几何风险近似，减少困难点损失与最终 mask 质量之间的差异，并在不改变 score/class/order 的情况下获得可复现收益？**

“支持域一致”不意味着 SAR 物理建模，当前 JPEG/F/e 不能证明使用了相位、极化或多模态。经典 Monte Carlo、HT、MIS、Dice、局部细化单独均不是本项目新原语。若公平实验只显示工程提速/额外 Dice 或实例尺度重权重收益，应按实际贡献写作；只有作用机制、强对照、样本近似质量、完整成本及跨数据集证据共同成立，才有理由讨论有技术差异的遥感任务整合。本文不判断达到 TGRS 投稿水平。

## 12. 核查范围及交付状态

查询围绕 point sampling propensity correction、importance-corrected segmentation risk、sampled Dice/IoU、boundary refinement 固定域及 exact-known-part/control-variate 估计。代码核查只针对上述公开函数及 staged sampler/BCE；未复现先例，未比较其报告 AP，未对源码做版本完整性认证，未把无关键词命中作为新颖性证据。部分老文献出版社/HTML访问失败时，仅使用可读的作者/原论文入口和明确段落；未将无法访问的正文内容补写成已核实事实。

普通 memo 已形成；所有方法与对照均未执行。GitHub 同步由 ROOT 基于最新远端、发布范围与当前绑定链负责；在远端核验前不报告已同步。
