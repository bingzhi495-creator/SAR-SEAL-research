# 当前 C 条件点头的直接先例、可证伪机制与贡献边界

日期：2026-10-13。状态：定向查新与普通设计备忘录；文中后续对照均未执行。本文不改变 N10 已登记协议，不增加运行权限，不读取或哈希绑定科学原件。数值结论由 ROOT 的实际终止资格与独立 CPU 复核另行决定。

## 1. 先给判断

当前 C 是“共享点 MLP + 实例条件偏置 + 原生 logit 跳接 + 原始不确定带内残差”的一个具体实现。把实例向量拼接到全连接输入，进而改变 ReLU 门控，已有明确原理论述；不能据此宣称新的条件调制原语、首个实例条件掩膜解码器或首个 SAR 实例条件分割方法。FiLM 原文 §3 直接说明了条件输入拼接与条件偏置的等价关系，并将其归入 γ=1 的特例。[FiLM，AAAI 2018，原论文 §2–3](https://ojs.aaai.org/index.php/AAAI/article/download/11671/11530)

值得继续检验的候选是一个范围更窄的集成问题：对于已经训练好的查询式 SAR 分割载体，能否利用其冻结的空间特征与实例表示，修正原始坐标中尚可由这些表示区分的几何误差，同时保留查询、分类、排序与原生评分。这里存在与完整 PointRend、Implicit PointRend 和端到端动态掩膜系统可描述的技术差异，但“可描述的差异”尚不是“有价值且新的研究贡献”。关键还需排除更简单的实例阈值偏移、额外容量及短预算优化路径等解释。

本备忘录没有给创新打数值分，也没有把没检索到相同命名视为首次。定向检索包含 2018–2026 的直接机制先例和近期 SAR/遥感工作；它不构成对全球最新论文、专利、在审稿件或未公开实现的穷尽查新。

## 2. 当前实现究竟计算什么

只核对了独立 staged N07 的 point_heads.py 普通副本。C 沿用 562→64→64→1、两次 ReLU，LN 均无 affine 且 eps=1e-5，最后一层权重和偏置为零初始化。输入按顺序为 LN(F256)、LN(e256)、clamp(L,-8,8)/8、49 维 patch；C 内部将 patch 置零。F/e/L 均 detach。不能因为类名带 OriginalDetail 就宣称 C 使用原 JPEG 或 7×7 图像细节。

令 x 为原图位置、q 为查询，f(x)=LN(F(x))，z_q=LN(e_q)，c_q(x)=clamp(L_q(x),-8,8)/8；将第一层权重按输入块拆成 A、B、a、P。则在实数代数下：

\[
h_{1,q}(x)=\operatorname{ReLU}(A f(x)+B z_q+a c_q(x)+b_1),
\]
\[
h_{2,q}(x)=\operatorname{ReLU}(U h_{1,q}(x)+b_2),\quad
u_q(x)=w^\top h_{2,q}(x)+b_3,
\]
\[
\widetilde L_q(x)=L_q(x)+{\bf1}\{|L_q(x)|\le 2\}\,u_q(x).
\]

P×0 消失。上述块拆分解释的是函数，未断言拆开矩阵乘法后与既有 FP32 GEMM 按位相等；任何计算重排都要重新资格检验。

显式实例条件等价为第一层偏置 b_{1,q}=b_1+Bz_q。B、A、U、w 是所有实例共享的训练参数；当前实现没有生成每实例完整的动态权重，也没有新增 F⊙e 乘积通路。原生 L 已经来自载体的 e·F 掩膜场，因此去掉显式 z_q 输入也不等于去掉全部实例信息。

“只加偏置，所以没有空间作用”同样不成立。在固定激活区域内，令 D1、D2 为 ReLU 的对角门控，固定 z、c，将 f 视为独立点头输入时，残差对 f 的局部偏导为（这是函数响应分析；实现 detach，因此不是输入反向传播，也不是整个教师 e·F 映射的全导数）

\[
\frac{\partial u_q}{\partial f}
=w^\top D_{2,q}(x)\,U\,D_{1,q}(x)\,A.
\]

z_q 可以移动激活阈值，从而使不同实例采用不同的分段空间响应；这是非线性条件交互，属于已知条件偏置机制。它不能反过来被命名为新的显式乘法调制或全动态参数网络。

信息与约束边界也很明确：

- 当前 C 无新增原始图像、邻域 patch、显式坐标或实例间竞争模块。载体 F/e 可能已经包含图像、上下文和位置信息；不能把点头输入的限制误读为整个模型没有上下文。
- 对同一查询，若两个位置具有相同的 f、L，点头必然给出相同 u。它不能从现有输入中恢复已经丢失、不可辨别的信息。
- 原生 |L|>2 的错误保持不变。带内 u 未被幅值约束，可能产生不连续、碎片或新空掩膜；带外恒等性不等于拓扑、安全性或整体性能保证。
- 固定原生评分用于隔离几何变化，不能被宣传为改善了置信度校准。新的前景/空掩膜仍保留原 S0；查询或分类错误不是本阶段的主要可干预对象。
- 40,257 是存储参数数目，49 维零输入对应 3,136 个对函数无贡献的首层权重，剩余 37,121 只是剔除这些零列后的参数槽位计数，并非严格的有效函数维数。AdamW 仍可衰减零列；偏置、非线性重参数化与优化路径也要计入解释。

## 3. 核心主张与直接先例

| 待审主张 | 直接先例及实际重叠 | 当前可保留的差异 | 判断 |
| --- | --- | --- | --- |
| 拼接实例向量形成条件空间响应是新机制 | FiLM 已明确给出 concat→条件偏置、γ=1、ReLU 阈值改变；DeepSDF 已使用共享网络的形状 latent 与查询位置拼接 | 条件来自已训练 SAR 查询向量；点输入是载体特征和 logit，不是形状坐标 | 原语主张不成立。[FiLM](https://ojs.aaai.org/index.php/AAAI/article/download/11671/11530)，[DeepSDF，CVPR 2019，§4/Fig.3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf) |
| 共享 MLP 在不确定位置细化掩膜是新机制 | PointRend 是最直接的点级精细化先例，使用图像特征与粗掩膜条件 | C 后置于冻结查询载体、显式拼接 e、只在原始 L 的固定闭带内加残差；没有完整 coarse-to-fine PointRend 管线 | 属于明确适配差异，不能据此称新的点渲染原理。[PointRend，CVPR 2020](https://arxiv.org/abs/1912.08193) |
| 实例条件点函数/动态掩膜首次出现 | Implicit PointRend 为每对象生成点函数参数；CondInst 使用实例动态掩膜卷积；AdaptIS 使用实例条件归一化 | C 的完整权重共享，仅首层偏置依赖实例；仍使用原生粗 logit，没有完整动态参数头 | 不能称首个条件点头；也不能把 C 等同于这三种完整系统。[Implicit PointRend，CVPR 2022](https://arxiv.org/abs/2104.06404)，[CondInst，ECCV 2020](https://arxiv.org/abs/2003.05664)，[AdaptIS，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Sofiiuk_AdaptIS_Adaptive_Instance_Selection_Network_ICCV_2019_paper.html) |
| SAR 物理信息驱动的实例条件修正是首次 | 2024 动态关键点 SAR 论文已将几何/强散射信息隐式编码进动态掩膜参数，并形成实例注意图 | 当前 e 是载体 latent，没有新增可核验的强散射、方向或物理参数；后置冻结修正与该端到端方法不同 | 不能把 e 命名为新 SAR 物理表示；应用贡献须另举误差证据。[作者机构库原摘要与 DOI](https://napier-repository.worktribe.com/output/3687229/sar-ship-instance-segmentation-with-dynamic-key-points-information-enhancement) |
| 保留查询和原生评分的后置窄带修正本身足够形成论文创新 | 残差、稀疏不确定区计算、冻结适配各有广泛先例；评分保持也是常见比较策略 | 这里有一套具体原图 renderer/all100/S0 不变的兼容契约，便于公平隔离几何效应 | 可作为系统设计与实验完整性优点；单独不能证明学术新颖性或 TGRS 水平 |

DeepSDF 的任务是 3D SDF 形状表示，它证明 latent 拼接的通用机制已有先例，并不意味着它已经解决 SAR 实例掩膜修正。不同任务间需要保持这个边界。[原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)


## 3a. 作者点头代码核查与完整系统差异

直接阅读的作者 [detectron2 point_head.py](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py) 中：

- StandardPointHead 使用共享 Conv1d(kernel=1)+ReLU，并在 COARSE_PRED_EACH_LAYER 启用时，每个隐藏层后重新拼接 coarse；最终 predictor 也接收 coarse。标准最终层使用 std=0.001 的正态权重，C/PR91 的精确零输出初始化是本项目保持原生冷态的适配，不能冒称作者原配方。
- ImplicitPointHead 的 _parse_params 拆出每实例各层的完整 W/b，_dynamic_mlp 按实例执行这些线性层。开启位置编码时，bbox 相对点坐标经随机 Fourier 编码再与图像特征拼接。这里的参数动态化比 C 的首层条件偏置范围更大；缩小为 tiny 动态头还需计入参数生成器成本，不可伪装成完整官方 IPR。

形式上，IPR 是 u_q(x)=MLP_{\theta_q}(features(x),PE_q(x))，θ_q 由实例参数头生成；C 则共享 A/U/w，仅 β_q=b+Bz_q 是实例相关。上述为便于比较的概括式，不替代其完整训练系统。[IPR 原论文 §4](https://arxiv.org/html/2104.06404v2)

[CondInst 作者 DynamicMaskHead](https://raw.githubusercontent.com/aim-uofa/AdelaiDet/master/adet/modeling/condinst/dynamic_mask_head.py) 的 parse_dynamic_params 和 mask_heads_forward_with_coords 验证了全 W/b 动态参数以及相对坐标输入，利用 grouped 1×1 conv 分实例计算。原常用 8 维 mask feature +2 维相对坐标、三个层10→8→8→1对应169动态参数；这不是包含 controller/主干的总模型参数数。[controller 作者源码](https://raw.githubusercontent.com/aim-uofa/AdelaiDet/master/adet/modeling/condinst/condinst.py)，[原论文 §2](https://arxiv.org/html/2003.05664v3)

[AdaptIS 原文 §3](https://arxiv.org/html/1909.07829v1) 采用选中点的 embedding 经 controller 产生 AdaIN 条件，另有 Relative CoordConv。不能把它简化成“与 C 完全相同的输入 concat”；其具体归一化和位置条件不同。原文给出的作者仓库源码本轮未成功读取，故不编造代码类名。[作者仓库](https://github.com/saic-vul/adaptis)

采样和推理的差异也不能省略。PointRend 原实例配方混合不确定点与随机点；推理以粗网格逐级放大并重新选择当前不确定点。IPR 的全掩膜监督配方采用均匀训练点，推理同样做网格/不确定点细分。C 固定原始 L 的 abs(L)<=2 带，只作一次残差；带不是迭代更新后的 uncertainty ranking。把几个通用点头接到同一冻结载体是合理结构控制，完整先例的 fine/P2 特征、box坐标、参数头与训练预算则需要另一个完整系统比较阶段。[PointRend 原文 §3–4](https://arxiv.org/html/1912.08193v2)，[IPR 原文 §4](https://arxiv.org/html/2104.06404v2)

作者 main/master 链接在查阅时可访问但没有固定提交，只用于原理查新。后续若真正复现，应另行固定版本、配置和完整成本；本轮没有下载或运行第三方科学代码。

## 4. SAR 与近期遥感工作对定位的影响

O2Former 将方向感知和多尺度查询生成放在主干/查询交互阶段。当前 C 若附加在这类载体上，已有的方向信息可能来自载体，不能把其全部收益计为新点头的物理建模。[O2Former 原论文](https://arxiv.org/html/2506.11913v1)

近期 GCIE-Net 的出版方正文检索片段给出 query embedding 经 MLP 与空间特征按通道相乘求和的掩膜生成式。这说明 SAR 查询—空间表征融合本身不能作为首次贡献。本轮完整页面访问受限，未核查其代码、完整消融或重现结果，不据其表格数值判断当前模型优劣。[GCIE-Net，Remote Sensing，2026，§3.2](https://www.mdpi.com/2072-4292/18/15/2512)

RSPrompter 已将自动学习的提示与基础分割模型用于遥感实例分割，并包含 SSDD，作者仓库也提供冻结 decoder 等配置说明。因此“给冻结大模型增加轻量实例条件适配器”不能作为不加限定的首次定位。[原论文摘要](https://arxiv.org/abs/2306.16269)，[作者代码](https://github.com/KyanChen/RSPrompter)

ReSAM 的原文提供点监督 Refine–Requery–Reinforce 和实例表征对齐，并在 HRSID-Inshore 等数据上报告 mIoU/F1。它是近期“条件—细化—遥感”语境的重要边界，但监督、提示来源、数据子集和指标与当前 fully supervised/all100 COCO AP 不同；既不能用其结果否定 C，也不能把两者数值并列作 SOTA 排名。[ReSAM，2026 原论文](https://arxiv.org/html/2511.21606v2)

DACF-Net 是需要保持在后续 SAR 对照清单中的近期几何细化工作。出版方页访问受限，作者 GitHub 地址可访问；本轮没有完成其几何模块的源码级核查，因此不将其列为“已证明与 C 完全同构/完全不同”，也不直接比较 AP。[出版方论文](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)，[作者仓库](https://github.com/yang199809/DACF-Net)

2024 动态关键点论文的全文本轮遇到机构库/出版方访问限制；已核对作者机构库原摘要与完整书目信息（JSTARS 17:11365–11385，DOI 10.1109/JSTARS.2024.3383779）。这里仅使用摘要直接支持的“几何/强散射→动态掩膜参数、实例注意”事实，不把其他非原始转载的细节当成核验过的方法公式。

## 5. N10 C 对 PR-FL-91 能回答什么

当前 PR 是 PointRend 式冻结载体适配对照，不是作者完整 PointRend 重现。它以 LN(F)+粗 logit 为点输入，三层 91 通道，粗 logit 在每层再注入；C 则两层 64 通道且首层额外有 LN(e)。两者原生冷态函数相同、缓存/采样/损失/训练轮数/种子与图像顺序配对、S0/查询规则相同，是有判别力的固定预算结构比较。

但是，显式 e、深度、宽度、coarse 再注入以及随机初始化张量形状同时变化。即使 C 在两种子上胜出，最多先支持“此固定预算和载体上，C 适配比此 PR 适配有效”；不能直接支持“e 已被证明是唯一原因”“胜过官方 PointRend”“PointRend 方法失效”或“SAR 物理机制已成立”。PR 最后一层直接输入 coarse logit，冷态后可较直接地学习带内比例修正，这也使短预算优化路径与 C 不同。

N10 的 +0.30 AP 点等内部资源优先级规则若触发，只负责决定后续资源安排；它不是统计显著性、机制证明、外部 SOTA 或发表标准。此备忘录不读取本轮 SCI 结果，也不预判 C/PR 的方向。

## 6. 优先可证伪的两个后续控制（仅设计，未执行）

以下均需另登记全新 SOURCE/资格；不修改运行中或冻结的 C/PR 源码，不自动延长已登记实验。原生载体、L、原始闭带、S0、查询、缓存点、full1024 分母 BCE、AdamW、固定端点和种子/图像顺序应一致。不能把带内点改成重新归一化损失，否则效应混入损失权重。

### A. 无显式 e 的 C-e0

沿用相同 562→64→64→1 与每个 seed 的 C 冷初始化，仅在训练及评价都将显式 e 输入置零；F、原生 L、native 跳接与带门保留。这是针对“显式 z_q 是否提供增益”的最窄功能删除，不是无实例信息基线，因为 L 仍携带 q 的原生条件。

反证方向：若充分可比预算和重复种子下，C-e0 与 C 的 AP/匹配几何效应相近，则没有证据要求显式 e 通路。若 C-e0 较差，只支持当前配方中显式通路的效用；不证明 concat 原语创新，也不排除额外活跃参数/优化路径的作用。

容量警戒：除 patch 的 3,136 权重外，e 块的 16,384 权重也不参与函数，剩余槽位为 20,737。名义参数和执行的稠密 GEMM 可相同，但有效输入容量减少；不称严格容量匹配。需要因果解释时，后续可增加容量取向的 F/L-only 加宽对照，并同时列存储参数、活跃相关槽位、执行 MAC 和测量时延。现有 PR 是一个近似参数/算量结构对照，不能替代所有机制对照。

### B. 实例条件标量偏移 δ(e) 对 C

\[
\widetilde L_q^{\rm offset}(x)=L_q(x)
+{\bf1}\{|L_q(x)|\le2\}\,g_\psi(\operatorname{LN}(e_q)).
\]

g 用两层 64 通道 ReLU scalar MLP，输出零初始化；或在相同 C 初始化路径下只使用 e 块及共享后续层。保留原生 L 的跳接/带门，去掉 residual 网络的空间 F 和 coarse c 输入。这里 residual 对同一实例是常数，但其掩膜移动仍随原生 L 的空间结构变化，不能误称“不会改变空间几何”。

最重要的反证：若 δ(e) 在配对 AP 与固定 native-query 的匹配像素分析上解释了 C 的大部分收益，则不能把收益归为必须依赖 F/e 的空间条件分类。N09 的一个全局 FIT δ 已被检验，并不等价于已检验这个实例条件 δ；全局 δ 无法替代它。

此控制的表达能力低于完整 C，e-only 256→64→64→1 为 20,673 活跃相关参数。若更简单的偏移达到同样收益，是有力的机制反证和简化机会；若它较差，则还需考虑容量差异，不能直接宣布空间机制成立。不要故意使用极小线性 δ 来把对照变弱。

这两项比再堆一个没有明确机制问题的新网络更能改变当前结论。短预算无差异不能证明长期无效；长预算也必须对 C 和对照同时分配。观察 AP 前锁定实质差异/等价判据，并报告全部种子，不根据 DEV 结果换阈值、挑 best 或补选“好看”的图。

### 廉价诊断与更强验证的边界

固定训练好的 C 后，交换同图两个查询的显式 e 而保留 F/L，可以检查输出是否依赖正确的 e 配对；它会产生分布外输入，只是敏感性诊断，不是公平训练基线，也不是因果创新证据。若需要稳定机制结论，要另行登记从训练起就破坏配对、保留边际统计的控制，并说明单实例图像、匹配查询与其他查询之间的分布限制。

门控可视化、Bz 范数或注意图也不是证明。更有意义的是：C 相对于 A/B 的增益是否集中于原生不确定带内、是否在小目标/相邻实例/岸边杂波这些实际误差上减少错误，而非只统一扩张；使用官方 COCO 与固定 native-query 匹配诊断分别陈述，不把后者当官方 AP。

## 7. 可以保留的贡献候选及停止夸大条件

唯一有依据保留的候选表述是：

> 针对查询式 SAR 船舶分割载体中原生表征尚可区分的几何误差，研究冻结空间—实例表示的后置非线性修正，在原始坐标的固定不确定带内保留查询与评分契约，并通过直接点头先例及实例偏移控制，检验其效果和成本。

这是待检验的系统整合与误差机制候选，未确认“首次”，不把已有 MLP/LN/concat/ReLU/残差分别计为三个创新点。若其优点只是可复现实验和部署便利，可肯定工程价值，但仍需证明为什么遥感任务需要这种组合、为什么强直接先例不足，以及所得结论是否跨划分/载体/数据集稳定。

若 A 或 B 解释了收益，应接受更简单的模型或缩小论文主张；若 PR 适配更好，应优先使用更强基线，不将它归咎于文章包装或额外数据。若长预算差异消失，记录预算依赖；若跨数据集消失，记录适用范围。任何这些结果都不等于项目失败，但它们会改变贡献的正确表述。

没有足够依据在本轮凭空提出一个“已证明不同于所有先例”的新结构模块。结构上的下一步应由已核验的残余误差驱动，而不是为了凑三个模块。

## 8. 工程折叠机会的真实边界

由于 e_q 对查询内所有点相同，可在实数代数下预计算 β_q=B·LN(e_q)+b1，逐点只计算 A·LN(F)+a c+β_q；零 patch 块也可移除。还可考虑缓存共享 F 投影。这能减少重复条件计算，是可测试的工程优化机会，不是新条件机制；当前 C 没有实现这些折叠，不得预报加速倍数。

FP32 求和顺序、内存读写、点集合构建、LN、renderer、all100 扫描及 F 投影都影响实际耗时。新实现须独立验证掩膜/输出一致性，单列载体训练/缓存生成/实际 teacher forward 的完整成本。当前归档字段评价的 native_forward_calls=0 只表示这一评价阶段没有 teacher 前向，不能称部署端无需载体或整体推理零成本。

## 9. 查新方法与证据等级

- 实现事实：独立 staged SOURCE 的文本核对；没有读取绑定实验载荷、运行训练、数值探测或哈希科学文件。
- 明确原理先例：FiLM 原 PDF/HTML；DeepSDF 原论文检索正文；PointRend/Implicit PointRend/CondInst 的原论文与作者代码由独立子审阅补核；AdaptIS 只核了原文机制，作者代码未成功读取。
- SAR 先例：O2Former 原 HTML；2024 动态关键点作者机构库原摘要；RSPrompter 原摘要/作者代码；GCIE-Net 出版方正文检索片段；ReSAM 原 HTML；DACF-Net 出版方信息及作者仓库。
- 未核事项：受访问限制的完整 SAR 方法/消融与全部近期同类实现，没有独立复现外部结果，没有跨数据集或盲 TEST 新结果。不可据此宣称“最新穷尽”“外部方法没有做过”或“已达到投稿级创新”。
- 检索同时用条件偏置、latent-conditioned point function、动态掩膜参数、point refinement、SAR 动态关键点及 2024–2026 查询/细化等表述，避免只搜索拟议方法名称。可用工具未提供 skill 指定的 Codex 跨模型 reviewer；本轮采用独立子审阅的原论文/代码核查，不冒称完成了指定跨模型调用。
