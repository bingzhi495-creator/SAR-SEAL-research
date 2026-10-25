# U57：固定实例条件掩膜精修的最近直接先例查新

## 2026-10-25直接先例与新证据补充

[本轮增量查新](U57_HIGH_IOU_REFINEMENT_PRIOR_UPDATE_20261025.md)确认基础SAM与++需要区分；目标域质量排序、非边界保留、动态提示专家与结构化局部精修已有直接先例。不能以默认SAM负结果、零初始化或固定候选本身建立创新。[固定配对实测](U57_FIXED_NATIVE_GEOMETRY_DIAGNOSIS_ACTUAL_20261025.md)已证明该条件人口存在几何损伤，当前FULL/PLAIN仍待固定50轮端点。下文为早期查新快照，不因这次局部结果追溯升级新颖性。


访问日期：**2026-10-18**。 **SegS-YOLO补核日期：2026-10-25（北京时间）**；其余条目沿用原访问范围。状态：**有界一手文献查新，未复现外部方法，未形成新颖性认证或实验结论。** 使用 novelty-check / research-lit 流程；本会话无可调用的跨模型 Codex reviewer，未虚构交叉审阅结果。不按论文数量评价研究进度，不给缺乏标尺的“创新分数”。

本次先阅读普通 [U57 训练计划](U57_DGX_ORIGINAL_UPDATE_FULL_FIT_TRAINING_PLAN_20261018.md)、GitHub 发布副本的实例 adapter 源码及完整主干构造说明，再对照既有 U40/N23 综述。没有访问科学载荷、运行状态、原始回执、资格链、绑定执行源码、图像、标注或权重，没有导入机器学习库、运行模型或 SSH。本文仅保存自写归纳和链接，不转存第三方全文。

## 查新的实际对象

当前方法不是从头生成新 proposal 的实例分割器，而是在固定教师候选上，用完整改造 UNet++ 的共享图像特征预测实例 logit 残差。普通 adapter 源码实际关系是：

```text
F = feature_projection(U(RGB400))
H_q = F + broadcast(query_projection(e_q))
        + mask_projection(bilinear_upsample(M200_q))
delta_q = zero_initialized_1x1_head(ReLU(H_q))
refined_logit_q = native_logit_q + fixed_geometry_render(delta_q)
```

`F` 每图计算一次；条件同时包括教师实例 embedding `e_q` 与其粗 logit `M200_q`；外部原生 logit、候选身份、类别及分数保持既定协议。FULL 的 H/4 方向双支与注意力、嵌套解码是图像特征构造，不能把 `model_new1` 的另一种路由叙述混入 FULL。U57 不引入新的网络模块，只计划以原数学更新在 DGX 独立从头验证 FULL/PLAIN。

需要查新的三个候选主张分别是：①复合 UNet++ 特征组织具有额外效用；②实例条件与粗掩膜共同帮助修正原生掩膜；③保持候选和评分不变时，这种结构整合在公共 SAR 数据上有可靠收益。前两项的广义思想已有明确先例；第三项目前仍须实验建立，不能由接口存在推出。

## 保留的直接先例

下表“重叠/差异/必须比较”为本次对公开方法与本项目普通代码的分析，不是原作者对本项目的评价。未重新做外部 AP 排名，也不将异划分、异监督或异预算数字混成一张 SOTA 表。

| 工作与证据等级 | 已知重叠与具体差异 | 对本项目的比较要求 |
| --- | --- | --- |
| **Xu 等，DiffSARShipInst**，ISPRS JPRS 223:440–455，2025；已发表。核到[出版社原文索引](https://www.sciencedirect.com/science/article/pii/S0924271625000887)、[作者仓库](https://github.com/XiaowoXuSAR/DiffSARShipInst)及[原 mask head 源码](https://raw.githubusercontent.com/XiaowoXuSAR/DiffSARShipInst/main/diffsarshipinst/head.py)；本轮出版社正文直开失败。 | 原方法已有多尺度特征增强、适应船体长宽特征的多非对称卷积注意力及按实例生成参数的 IAMR；共享特征与实例条件掩膜生成不是本项目首创。当前方法在项目Mask2Former/R50 E24固定输出上另用RGB UNet++学习加性残差，保留原评分及候选；这不构成DiffSARShipInst的同协议复现，不能用原生E24的AP代替该外部方法结果。 | **必须有原教师未精修的同一输出控制**，报告整体流水线额外训练/计算成本；原教师与“教师+refiner”的差异不能全归因某个 UNet++ 模块。原生控制应对应实际Mask2Former/R50 E24；DiffSAR属于外部直接先例，作者方法的独立同协议比较仍待完成，不能以现有原生AP冒充。 |
| **Yang、Zhang、An、Li、Chen，DACF-Net**，IJDE 19(2)，2026-09-08 上线；已发表。核[出版社目录及日期](https://www.tandfonline.com/toc/tjde20/19/2)、[原文官方索引](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)与[作者仓库](https://github.com/yang199809/DACF-Net)；正文直开 403，方法依据官方域已索引正文段落。 | MSFFM 多分支大核/条带卷积、CLFIM 高语义调制低层以及 DCPBlock 空间/通道建模，直接覆盖“方向/尺度/语义融合改善 SAR 实例掩膜”的广义主张。其主干—FPN—几何 proposal—HTC 与本项目固定 query 的嵌套 UNet++ 残差接口不同。 | 不能把方向、多尺度或高低层融合本身称首次。结构效用还需区分普通条带卷积/额外容量与自适应方向采样。仓库本次只核到 README、LICENSE、.gitignore，声明实现待发布；不能称已取得可运行官方模型，协议适配也须与作者复现分开。 |
| **Zhang等，SegS-YOLO**，IEEE JSTARS 19:2658–2679，2026；2025-11-20上线。[官方PDF](https://ieeexplore.ieee.org/ielx8/4609443/11278119/11261833.pdf)已核§III-B-3及§IV-B、V-A。 | DSCPKI已结合CAA上下文/条带注意、并行多尺度卷积与DSConv，用于SAR实例分割；本项目则固定外部候选和分数，以UNet++及教师条件预测加性logit残差。 | 不称首次将DSConv或其与上下文/多尺度整合用于SAR实例掩膜。原文评价为mask AP；LS-SSDD仅定性。实际复现与同协议比较尚未完成，不据此判当前整体结构无贡献。 |
| **Prytula、Tsiporenko、Zeynalli、Fishman，IAUNet: Instance-Aware U-Net**，CVPR Workshops 2025:4778–4787；已发表的 workshop 工作。核[CVF 原论文索引](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/papers/Prytula_IAUNet_Instance-Aware_U-Net_CVPRW_2025_paper.pdf)、[作者 arXiv v1 方法](https://arxiv.org/html/2508.01928v1)、[项目页](https://slavkoprytula.github.io/IAUNet/)及[仓库](https://github.com/SlavkoPrytula/IAUNet)。 | 完整 U-Net、轻量卷积 pixel decoder、多尺度实例 query 细化、SE/CoordConv 及高分辨率 mask 特征已结合。因而“把 U 形网络变成实例感知网络”不是空白。它学习并更新 query、端到端预测类别/掩膜；本法使用固定外部 query 与粗 mask 的相加条件小头，不进行 Transformer query 更新。 | 应准确引用此结构类别先例，不能用“不是 SAR”免除技术比较；但医学 AP 不可直接与 SAR AP 排名。本次默认 `project_page` 分支是网站，另读到的 `master` README 也描述网站，未核到可运行训练源码；不据此断言所有分支均无代码。 |
| **Lin 等，SAMRefiner**，ICLR 2025；已发表。核[正式论文索引](https://proceedings.iclr.cc/paper_files/paper/2025/file/90800f46f84381b7891e1378ee850013-Paper-Conference.pdf)、[作者 arXiv v2 §3](https://arxiv.org/html/2502.06756v2)及[官方代码](https://github.com/linyq2117/SAMRefiner)。正式 PDF 本轮直开失败，方法阅读来自作者 v2。 | 已将外部粗 mask 和图像用于自动精修：由粗 mask 提取点/框/连续 mask 提示，并从图像前景均值形成 query embedding 以调整框；无需人工测试点。其 SAMRefiner++ 另有 IoU 排序适配。它不是本项目的固定教师 embedding + native logit 残差，也不使用本项目 UNet++。 | 至少一个通用“同粗 mask 的自动精修”强对照应进入最终证据；SAMRefiner 是现成候选。评估必须仅从预测 mask 生成提示，不用 GT 点/框。若比较基础 SAMRefiner，应保持原教师分数与候选映射，避免把 IoU 重评分收益混入掩膜修正。外部预训练和额外资源单列。 |
| **Price、Cote、Rundensteiner，PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting**，CVPR 2026:6325–6335；已发表。核[CVF 正式索引](https://openaccess.thecvf.com/content/CVPR2026/html/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.html)、[官方补充材料索引](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Price_PromptMoE_A_Segmentation_CVPR_2026_supplemental.pdf)与[作者仓库](https://github.com/sprice134/PromptMoE)。正文/PDF 直开失败，依据正式域索引摘要及作者说明。 | 图像+粗 mask 驱动专家引导图、动态专家选择和点提示精修，说明截至 2026，“图像信息辅助粗掩膜精修”和动态选择亦已有直接方法。它通过提示驱动 SAM；本项目以固定 query 的稠密卷积残差直接输出修正，不含该专家选择机制。 | 可作为最新通用精修参考，不能把其 IoU/BIoU 改写成 COCO mask AP。仓库仍写 submitted，但正式发表状态以 CVF 为准。它与 **arXiv:2511.18116 同名异常检测工作不是同一篇**，不可混用作者、机制或结果。 |

## 已经明确不是空白的部分

嵌套 dense skip 来自 [Zhou 等 UNet++ 原作](https://arxiv.org/abs/1807.10165)；其[扩展作者论文](https://arxiv.org/abs/1912.05074)亦已有 Mask R-CNN++ 实例分割用途。通道后接空间注意来自 [Woo 等 CBAM，ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)，SimAM 来自 [Yang 等，ICML 2021](https://proceedings.mlr.press/v139/yang21o.html)。[Qi 等 Dynamic Snake Convolution，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_Dynamic_Snake_Convolution_Based_on_Topological_Geometric_Constraints_for_Tubular_ICCV_2023_paper.html)已有细长/弯曲结构自适应感受野及多视角融合。自主决定它们的连接和落点可以形成整合贡献候选，不能将算子改名或“首次用于本文件”视为学术创新。

SAR 内部的方向与 query 结合亦已有 O²Former；本轮重新读到[作者 v1 方法](https://arxiv.org/html/2506.11913v1)，与项目已有 [U40 先例补核](U40_DIRECT_PRIOR_UPDATE_20261017.md)去重，不另堆一个条目。上表不构成全文穷尽搜索；CascadePSP、PointRend、RefineMask、SegRefiner 等既有精修家族不能因未重复列入而视作不存在。

2026-10-18时SegS-YOLO仍是待核线索；2026-10-25已通过出版社登记元数据与IEEE官方PDF补核其发表身份和DSCPKI方法，见[增量核查](U57_DIRECT_PRIOR_INCREMENT_20261025.md)。原先“精确结构、作者归属或发表年未核”的表述保留为历史访问情况，不再代表当前证据状态。作者与用户的身份关系尚未确认，不依姓名推断。

## 当前能够保留的差异与应撤回的主张

可准确描述的技术差异是：在固定 SAR 教师候选及原生 logits 上，用完整改造的 UNet++ 全图特征，与固定实例 embedding、粗 mask 进行明确的条件融合，输出零初始化加性残差；原检测分数和候选身份保持不变。这是可核查的接口与计算图差异，**尚未证明是新的有效机制**。

不得把“注意力+方向+多尺度”拆成三个已成立创新点；不得称首次 query-conditioned U-Net、首次图像+粗 mask 精修或首次动态选择；不得将 U52 的共享梯度累积工程候选改名为算法创新。没有强对照和跨集证据，也不能把公共 SAR 的一次涨点直接写成普适改善或 TGRS 投稿级成功。

## 一个最小决定性对照建议

**在 U57 的既定 FULL/PLAIN 完整终点之后，优先只增一个容量/计算匹配的普通结构控制 PLAIN-C。** 沿用同一 UNet++ 嵌套拓扑、实例 adapter、固定教师输出、数据、训练监督、每臂 37,050 更新及固定评估规则；只用普通残差卷积扩大 PLAIN 的特征容量，使有效参数和实测计算成本尽量接近 FULL，匹配方式、允许偏差和共同 state 初始化规则须在看结果前锁定，无法同时匹配则报告双重差距。不能用额外未参与 forward 的参数或无效计算凑“匹配”。

这个对照首先否证最直接的替代解释：收益只是多约 5.92M 参数或更多计算，而非 FULL 的特定组织。若 FULL 不能稳定优于 PLAIN-C，应收缩为整体实现效用，不给方向/注意力机制背书；若稳定超过，只支持该组织超出普通加宽的效用，仍需针对性消融才可归因某个模块。重复种子和跨公共数据集为后续确认，不以单 seed 判断首创。该建议是待预锁设计，不改变当前 U57 配方、不表示已经开跑，也不要求为查新另加无休止 smoke。

原教师控制与通用自动精修对照属于最后论文不可缺少的比较维度，上述“只增一个”仅指优先用于解释 FULL/PLAIN 结构差异的最小新增训练臂，不能被理解为一条对照就足以证明胜过所有直接先例。

## 检索覆盖与限制

本次使用了 SAR instance segmentation / fixed-query or instance-conditioned mask refinement / U-Net 与 UNet++ attention / dynamic snake 等多组措辞，并加入 2024–2026、2026 和作者/正式域限制；还检查了最近六个月可检索的 2026 论文线索。搜索发现页只作定位，技术结论仅依据上文原论文、作者说明/代码或正式出版域。确实核到 2026 的 DACF-Net 与 PromptMoE，故不能说“2026 没有可靠工作”。

本次没有穷尽 IEEE、arXiv、OpenReview 或所有 2026 顶会；正文拒绝访问和未核实现的范围已逐项标明。未从不可靠镜像认证 2026 新纪录，也没有将搜索未命中解释为不存在。最近先例足以限制宽泛首创说法，尚不足以断言上述精确整合必然没有贡献；最终判断取决于其针对何种真实误差、相对何种直接替代以及何种公平证据。

2026-10-25另补核[教师身份与数据接触范围](U57_TEACHER_IDENTITY_AND_DATA_EXPOSURE_SCOPE_20261025.md)：当前原生教师按公开恢复合同归为项目Mask2Former/R50 E24，修正早先与DiffSAR外部先例混用的称呼；未改训练与数值。
