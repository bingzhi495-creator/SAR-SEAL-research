# 修补行动收益与计算分配：直接先例补充

2026-10-13；**普通研究文档，文献查新与静态推导，后续假设未执行。当前 N13 两臂、三种子、48 epoch 配方及冻结源码均不改。**

本轮只读普通 `N13_LEARNED_EDIT_UTILITY_NEXT_HYPOTHESIS_20261013.md` 与在线 primary 来源，最多保留六份新增直接工作；不读取科学载荷、RAW、SPEC、Q、缓存、GT 或绑定源码。已知错误图、PointFlow、Mask Transfiner、Mask Scoring 等内容不重复扩展。以下是有边界的新增查新，不是穷尽证明；不把外部报道数值与内部 DEV 比较。

## 六份最近邻工作及核验边界

| 工作、发表状态及原始入口 | 已有机制、作者代码及与当前候选的具体差异 |
|---|---|
| **Dynamic Video Segmentation Network（DVSNet）**；CVPR 2018 已发表，官方 proceedings 为 6556–6565 页；预印本 2018-04-03。[CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_Dynamic_Video_Segmentation_CVPR_2018_paper.html)、[原文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Dynamic_Video_Segmentation_CVPR_2018_paper.pdf)、[arXiv](https://arxiv.org/abs/1804.00931) | §3.2–3.4：轻量 DN 从 flow 特征预测廉价 warp 与昂贵 segmentation 输出的像素一致率，运行时不先求两路完整输出，再决定区域走哪一路；监督是两模型一致率，**不是与 GT 比较的净修补收益**。已是“先预测执行结果，再节约昂贵计算”的直接先例。[作者仓库](https://github.com/XUSean0118/DVSNet)明确为修改版 TensorFlow 实现，并说明 I/O、布局与原文差别，不能视为论文速度的原实现复测。候选区别是静态 SAR query、固定 PR、逐点有符号支持分配，而非视频重计算或关键帧更新。 |
| **SegBlocks**；TPAMI 45(2):2400–2411，2023，作者单位发表记录可核；预印本 2020-11-24，v2 为 2022-08-05。[作者单位记录](https://homes.esat.kuleuven.be/~konijn/publications/pub.html)、[DOI](https://doi.org/10.1109/TPAMI.2022.3162528)、[作者原文 HTML](https://arxiv.org/html/2011.12025v2) | §III-A、式(5)–(9)：每块选高/低分辨率，REINFORCE reward 结合该块任务损失相对全图损失与高分辨率块比例偏离目标的惩奖。**不是实际候选前后损失差**，但按任务反馈和成本分配计算已经存在。[作者代码](https://github.com/thomasverelst/segblocks-segmentation-pytorch)提供 CUDA 分块及 BlockPad，真实稀疏执行是贡献的一部分。候选只修改冻结原生 mask 的支持，不改变骨干分辨率；应证明具体质量—成本差异，不能以“预算约束策略”称首次。 |
| **BlockCopy**；ICCV 2021 已发表，官方原文。[CVF 原文](https://openaccess.thecvf.com/content/ICCV2021/papers/Verelst_BlockCopy_High-Resolution_Video_Processing_With_Block-Sparse_Feature_Propagation_and_Online_ICCV_2021_paper.pdf) | §3、式(8)–(12)：选执行/复用视频块，reward 为输出 information gain 与目标执行比例成本项；在线自蒸馏策略训练不使用 GT。§3.4 的实例 IG 来自当前/前帧预测 mask 的 IoU、score 等变化，语义 IG 使用 KL，**不是与 GT 比较的真实修补净收益**。论文包含实例分割，优化骨干/RPN，头照常运行，是计算分配/在线策略的近例，而非 `δ(2e−1)` 标签等价先例。[作者仓库](https://github.com/thomasverelst/blockcopy-video-processing-pytorch)可核分块框架及任务入口；本轮未执行代码。候选无视频时序复用，监督来自固定 PR 的 GT 净变化。 |
| **IteR-MRL：Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning**；CVPR 2020 已发表；预印本 2019-11-23。[CVF 原文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liao_Iteratively-Refined_Interactive_3D_Medical_Image_Segmentation_With_Multi-Agent_Reinforcement_Learning_CVPR_2020_paper.pdf)、[arXiv](https://arxiv.org/abs/1911.10334) | §3：voxel agent 根据原图、前轮概率及交互提示，选择概率调整动作；reward 使用修改前后的交叉熵相对收益，允许依赖相邻 voxel。已有“行动后净收益训练修补策略”。与候选区别为 3D 人机交互、多步概率动作，非自动静态 exact K 二值修改。CVF PDF 直接 fetch 本轮返回403，但官方索引中的方法正文和作者摘要可核；本轮未确认作者原代码，不能据此声称无代码或实现等价。 |
| **SAM-Veteran: An MLLM-Based Human-like SAM Agent for Reasoning Segmentation**；官方 OpenReview 托管文稿页眉**作者声明为 ICLR 2026 conference paper**，本轮仅核到该 primary PDF 的搜索索引正文片段，未下载逐页读完；未独立取得 forum 的 venue/decision 元数据，正式录用状态保留此限制。[具体 forum](https://openreview.net/forum?id=oN55r8iJJW)、[具体原文 PDF](https://openreview.net/pdf?id=oN55r8iJJW)、[官方内容寻址 PDF](https://openreview.net/pdf/69f488c8979689d03533b6ccaf32c652ebe483be.pdf) | primary 索引中的 §3.1、§3.2.2、Table 1：MLLM 根据当前 SAM mask 提出正/负提示点或停止；reward 直接包括行动前后 **ΔIoU** 的分档奖励与停止判断，非仅错误点概率。这是“质量改善指导点修补”及“判断是否值得继续”的近例。候选不同于文本推理、大模型、SAM 全局重解码及迭代停止，而是冻结 PR、固定 query/类别/分数与同 K 支持。forum/API/PDF 直接访问受 challenge，API 实际403；未用第三方汇总替代录用证据，也未确认作者代码。 |
| **RL-Axial: Learning When to Compute Global Context for Multimodal Segmentation**；Cognitive Computation 18,113，**2026-10-02 version of record 已发表**，出版社明确日期。[出版社全文](https://link.springer.com/article/10.1007/s12559-026-10661-z) | 式(7)–(9)：每影像块选 off/row/col/row+col，reward 为相对 off 的质量增量；**式(9)训练 J 是 soft-IoU**，摘要/总体讨论称 ΔmIoU，不能混写。组标准化策略梯度中有放回抽 K=8 次动作，不是 point 配额。成本分析是分支加权估计，未含完整选择器/路由开销，reward 也没有成本惩罚。RGB/IRRG+DSM 语义分割，非 SAR 实例精修；本轮未找到可确认作者代码。已有遥感质量反馈计算调度，应区别候选的固定修补支持及实际全流程成本。 |

SAR 范围：限定检索没有新增确认“预计修补行动净收益选择支持”的 SAR 直接工作。已有 PointFlow、Poly-BRBLE 等接近边界/不确定性精修已在此前综述讨论，仍须作为机制对照背景。这次没有确认新直接先例，不证明领域空白、组合首次或方法可发表；RL-Axial 也不能因为是遥感而视为 SAR 先例。

## 标签不是新信息，误差与动作收益仍须分开核查

对二值固定候选，`e=1[P!=GT]`、`δ=1[P!=P']`，严格有 `u=δ(2e−1)`；不变类恰为 δ=0，改变点上的修复/破坏就是当前错/对。收益标签本身不提供不能由错误与候选变化表达的新信息。完整输入 X 决定 P/P' 时，条件均值可因子化；压缩描述符 Z 未必如此。既有方案已经加入复用同一个8451参数 error head 的强替代 `δ_actual[2(p_FP+p_FN)−1]`，该后处理不增加训练更新，**但需要全域实际 PR 候选**，不能忽略此成本。

本轮六篇未确认恰好使用上述固定二值 PR、同 K、同输入容量与这一因子化对照的完整组合；这不是首次证明。DVSNet 已表明无需先执行昂贵候选即可预测路径结果，BlockCopy/SegBlocks 已表明收益与计算分配有直接先例，SAM-Veteran 则比像素错误标签更直接对齐行动 ΔIoU。

## IoU 目标错配及一个未来可否证突破口

固定一个非空 GT，令 T=原 TP、U=|GT|+原 FP>0。联合修改为 ΔTP、ΔFP，且 U+ΔFP>0，则

\[
\Delta\operatorname{IoU}=
\frac{U\Delta TP-T\Delta FP}{U(U+\Delta FP)}.
\]

因此真实 IoU 改善的精确条件是 `U ΔTP−T ΔFP>0`；像素净修复是 `ΔTP−ΔFP`，并不等价。例：|GT|=100、TP=80、FP=20，删11个FP却损失10个TP，像素净修复+1，但 IoU 从80/120降至70/109。该例是代数说明，不是实验结果。

**一个限定的未来假设**：在同一冻结 PR、同 K 和全部原生预测保留下，利用共享实例质量估计调整前景/背景修改的收益代理，在保持任务质量时避免全域 PR 计算。只在当前固定 N13 结果完成后另行预登记；不改原 utility 文档或本轮已锁实验。

可用已有三类预测 p_help/p_harm 构造内部排序代理。设 q=T/U 为真实原生 IoU：0→1时，修复/破坏对 `ΔTP−qΔFP` 的权重为 +1/−q；1→0时为 +q/−1。部署只能使用共享、在训练划分中拟合的 **估计** q_hat，不得使用 GT。于是候选控制为

\[
s_{student}=[(1-P)+P\hat q]p_{help}
-[(1-P)\hat q+P]p_{harm}.
\]

同8451参数 error head 的实际候选强替代，可相应写为

\[
s_{flip}=\delta_{actual}
\{(1+\hat q)p_{error}-[\hat q+(1-\hat q)P]\}.
\]

这是 **instance-quality-conditioned proxy**，不是 E[ΔIoU|Z]、IoU最优解或AP最优策略：真实 T/U 与局部改变相关，随机分母使 E[比率]不等于比率E；估计 q_hat 也不等于真实 IoU。对预测概率构造的加性代理，即使真实原生 q 已知，也不能保证最大化期望最终 IoU；但对真实逐点标签、固定独立二值候选和等成本 exact K 的特殊情形，五类优先序恰是完整单实例 IoU oracle，且不需要真实 q。[限定证明与完全校准概率反例](N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)。AP 另含阈值、排序及匹配；原生预测类别/分数/order 均应固定，禁止把调分收益混入。质量估计新增的容量、拟合曝光与成本必须共享并单列，不能仍把总体称8451参数。

该假设只检验两件相关事实：质量权重是否比同一 utility head 的像素收益排序更合适，以及预估支持后仅在所选 K 点执行 PR 是否比全域实际 flip 强替代更经济。各排序使用同一 PR、输入约束、A/B划分、点样本、种子及 native K，不因增加后处理而多训第三个 selector；若质量估计另有训练，要重新预登记其共享预算。真实 q 只能用于训练/离线机制诊断，不能作为部署收益。

成本账须包含公共 native forward、全域 selector、距离描述符、质量估计、排序、点采集、实际 PR 及导出；同时报告 refinement 增量成本，防止公共骨干掩盖选择器开销。预估支持路径为 `C_select+C_quality+C_sort+C_PR(K)+C_IO`，实际 flip 路径为 `C_error+C_quality+C_PR(全部query×图像域)+C_sort+C_IO`，不先假定哪条快。K个支持不等于K个真实翻转，也不等于固定耗时；记录修复、破坏、不变与选择/实际翻转数。

未来预登记质量等效区间、配对不确定性与成本收益门槛后，使用独立评价集验证：若完整AP/高阈值IoU不能保持在锁定质量区间，或端到端时间未降低，拒绝本数据及配方下“保持质量而省全域PR”的效用假设；代理相关性、较少MAC或单独正像素收益均不能替代这一结论。跨数据集失稳须保留，而非重新调权后称泛化。可能贡献是具体的有限容量校准、质量权重和真正稀疏执行所形成的可复现质量—成本收益，不是新名字、新标签、全新风险理论或投稿承诺。

## 核验状态

已实际浏览 primary 论文/出版社、作者 arXiv 及作者仓库入口；代码只读在线文本/README，未执行。三份作者仓库已明确链接；其余未确认作者代码，未将第三方实现当作者原件。SegBlocks 的作者 HTML 与 RL-Axial 出版社正文可读；部分 CVF/OpenReview 机制来自其 primary 搜索索引正文，未据此声称已下载逐页阅读。SAM-Veteran 录用元数据仍待独立核验。本文仅含概述、链接及独立代数分析，不收录第三方全文。普通文档形成后 STOP，由 ROOT 审阅及同步；同步不等于假设成立。


## 2026-10-13 有限SAR补核：提示分割与作者代码状态

本轮多智能体限定三条primary检索，ROOT另实际打开作者arXiv摘要与DACF作者仓库目录/README。查询原文为 `SAR ship instance segmentation reinforcement utility budget adaptive computation 2025 2026`、`SAR ship instance segmentation mask refinement point uncertainty error prediction 2025 2026`、`"DACF-Net" "SAR" instance segmentation 2026`。前两条限制arxiv.org、ieeexplore.ieee.org、tandfonline.com、mdpi.com、sciencedirect.com、link.springer.com、github.com；第三条限制tandfonline.com、github.com、arxiv.org、ieeexplore.ieee.org。这是本轮定向补核，不能当作穷尽查新或无先例证明。

- [Prompting Foundation Models for Zero-Shot Ship Instance Segmentation in SAR Imagery](https://arxiv.org/abs/2604.17920)，Mansour、Sica、Schmitt，arXiv v1日期2026-04-20。ROOT所读摘要为SAR数据训练YOLOv11检测框，再以框提示SAM2生成mask，无像素mask监督；当前发表venue和作者代码未确认。这与当前候选的固定修补器正负收益预测及sameK支持分配在所读机制上不同；此判断限于摘要，没有声称已逐页核对全文。其zero-shot范围不能扩写为整个系统未经SAR训练，也不能与当前全FIT mask监督方案按不同指标直接排名。
- [DACF作者仓库](https://github.com/yang199809/DACF-Net)本次可见文件仍为.gitignore、LICENSE、README；[作者README](https://raw.githubusercontent.com/yang199809/DACF-Net/main/README.md)表示代码在整理以供发布。因此仍不可记为已取得可运行的完整作者基线，更不可记为已公平复现。此处不从README中的投稿措辞推断当前出版社发表状态。

本轮没有取得会实质改变现有sameK utility强对照的新直接SAR行动净收益/学习预算分配证据；部分出版社正文访问受限，不能排除全文机制漏检。检索未命中不支持首次、领域空白、核心创新或TGRS成熟结论。继续保留既有稠密实际flip/error强替代及同等优化成本要求；当前N13两臂、训练长度与评价规则不改。该补核仅存摘要概述、检索范围和链接，不收录第三方论文全文。

## 2026-10-14：真实标签oracle与预测概率代理的范围更正

已完成[固定K二值IoU理论核对](N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)：真实逐点五类顺序在明确条件下恰为完整单实例IoU oracle；预测概率即使完全校准、真实原生q与U已知，也无期望IoU最优保证。此为理论更正和人工反例，未开展新SAR实验。后续如登记收益选择器，应增补共享同一概率头的固定r=1/2后处理对照，以检验估计质量权重的必要性；当前N13配方不改。
