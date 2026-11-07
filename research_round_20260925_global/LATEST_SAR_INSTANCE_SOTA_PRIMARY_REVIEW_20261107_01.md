# 2025–2026 SAR 船舶实例分割直接先例核验与下一轮候选

核验日期：2026-11-07。本文为独立普通研究文档；没有读取运行中的科学载荷、SOURCE、原始图像或权重，没有执行训练。以下均为公开文献报告值或待检验设计，不是本项目新实验结果。

## 1. 判断与证据边界

本轮确认了五项真正涉及 SAR 船舶**实例分割**的 2025–2026 直接先例，另补充一项 SAM 适配工作。最重要的更新是：方向查询增强、SAM 的频率/形态适配、检测框提示基础模型、DEIM 加实例掩膜分支以及全局/通道融合，都已有相当直接的先例。不能把这些组合本身作为未经限定的新颖性主张。

公开新论文的数量不等于可立即执行的完整作者实现数量。本轮未核验到下表五项 SAR 专用方法的完整作者代码、配置、可用训练权重与实现许可证的组合；这不表示它们不存在，只表示截至核验时尚不能据此承诺作者级复现。最可执行的下一步是使用有官方实现和权重的 RF-DETR-Seg，以及官方 SAM2 构成透明的新基线；先获得本项目协议下的真实强基线，再做小规模、针对性修改。

**禁止横向混算：** COCO mask AP50:95、mask AP50、按场景分别计算的 AP、mIoU、Dice 和检测框 AP/召回分别保存。外部论文分数不是本项目胜负依据；完整作者代码未取得时，实施名称应写“本项目透明适配/重建”，而非“完全复现”。

证据标签：A=原文方法与实验已读；B=出版方原文局部/检索缓存已读；C=出版方元数据与摘要可核、数值协议未完整取得。网页受限保留为访问限制，不据此补造实验。

## 2. 直接证据矩阵

| 工作及书目信息 | 任务与技术差异 | 经核验的指标（均为作者报告） | 数据、初始化与预算 | 代码、权重和许可证核验 | 证据 |
|---|---|---|---|---|---|
| **O²Former: Direction-Aware and Multi-Scale Query Enhancement for SAR Ship Instance Segmentation**；Fei Gao, Yunrui Li, Xiaoyu He, Jinping Sun, Jun Wang；2025-06-13 arXiv v1；本轮未核实正式录用 [P1] | Mask2Former 上的 OQG 多尺度查询生成、OAEM 方向卷积与极坐标位置融合；真实例分割 | SSDD 离岸/近岸 mask AP=71.9/63.2；HRSID=71.7/43.1。**四个场景子集分数，不能当整集 AP 或简单平均** | R50 ImageNet→HRSID→SSDD；batch8、AdamW 1e-4、500epoch，300/400衰减；官方比例划分。文中场景计数和 HRSID 总数有不一致，须取精确列表；尺寸分组采用64²大目标分界，不能默认标准COCO96² | 未定位作者官方完整实现、权重或实现许可；检索到的同名车道线 O2Former 与本文无关 | A |
| **GCIE-Net: A Global and Channel Information-Enhanced Network for Ship Instance Segmentation in SAR Images**；Wei Luo, Jinhui Lan, Yiliang Zeng, Wei Xiong；Remote Sensing18(15),2512；2026-08-01 [P2] | DEIM-D-FINE/HGNetv2 加查询×像素嵌入掩膜分支；MGEA 分组空洞卷积/高效注意力，CGFF 跨尺度通道融合 | PSeg-SSDD mask AP/AP50/AP75=69.7/98.4/87.0；HRSID=63.9/91.1/79.2。HRSID AP 与 LFG-Net 并列，不能称严格全面领先 | SSDD80/20，HRSID65/35；**不使用HGNetv2预训练权重**；AdamW、batch8、300epoch，四阶段DEIM训练及多尺度，基础640输入 | 本轮未找到作者官方代码/权重；论文CC BY4.0不自动授权未公开实现。方法、实验阅读来自出版方页面缓存和MDPI提供的原文镜像 [P2m] | A（出版方原文镜像） |
| **Prompting Foundation Models for Zero-Shot Ship Instance Segmentation in SAR Imagery**；Islam Mansour, Francescopaolo Sica, Michael Schmitt；arXiv2604.17920v1，2026-04-20；正式出版状态未核实 [P3] | SAR上训练YOLOv11检测器，以预测框提示冻结SAM2 Hiera-L；非整个系统零SAR监督 | SSDD232张：meanIoU0.637、Dice0.764、框召回0.892；**没有可比整集COCO mask AP结果**。1像素膨胀后的relaxedIoU另列，不能混入原指标 | SAM2不做mask微调；检测阈值0.5，最多3候选取预测IoU最高；检测训练AdamW+cosine；确切检测器规模、种子、全预算未核实。没有HRSID实验 | 文中官方链 https://github.com/IslamAlam/hybrivision 本次返回404；SAM2自身官方实现可用 [C1]，因此可透明重建、不可写已取得整篇作者代码 | A |
| **SAR Image Target Segmentation Guided by the Scattering Mechanism-Based Visual Foundation Model (SARSAM)**；Chaochen Zhang, Jie Chen, Zhongling Huang, Hongcheng Zeng, Zhixiang Huang, Yingsong Li, Hui Xu, Xiangkai Pu, Long Sun；Remote Sensing17(7),1209，2025-03-28 [P4] | SAM的M-Adapter形态/分组卷积、AWST自适应小波软阈值，以及框提示器；真实例目标掩膜 | 结论中的“mIoU提升>10%、APmask50提升>5%”属于**GT框提示**条件；不能代替自动预测提示结果或mask AP50:95。绝对整集maskAP尚未完整取得 | SSDD/HRSID；精确划分、初始化、epochs尚未完整核实；13.5%可训练参数和10.1%微调FLOPs不能表述为整个系统大小/速度 | 出版方Data Availability未给可核验代码链；未确认完整代码、权重和实现许可；文章CC BY不等于代码开放 | B |
| **联合微分物理场与结构化图推理的SAR复杂场景舰船实例分割方法（DSR-Net）**；何少炜、熊伟、姚力波；电子学报2026在线1–11页，2026-07-22录用、2026-08-19页面更新 [P5] | 微分物理场/FDE增强与结构化图推理/HSR关系建模；属于实例分割直接先例 | 正文受限，精确maskAP、HRSID指标与方差**未知**，不进入数值排名 | 出版方可检索到SSDD1160、928/232；初始化、完整预算及精确评估代码未知 | 本轮未核验作者代码、权重或实现许可；不补造实验表 | C/B局部 |
| **SAMSAR: A modified SAM architecture for oceanic ship segmentation of satellite SAR images using CNN-based Cross-Fused Attention**；Mahdi Rahimi, Saeed Sharifian；Expert Systems with Applications284,127852，2025-07-25 [P6] | SAM适配器+CNN局部分支及交叉融合注意力；与上行SARSAM是两篇不同工作 | 本轮未取得完整mask评估协议，绝对AP未知；仅作为SAM/CNN融合直接先例，不列入maskAP排行榜 | 完整划分、提示方式、训练预算和种子待核验 | 本轮未核验官方代码/权重与实现许可；不得仅凭摘要判定可运行 | C |

## 3. 至少两项原文深读后的可用结论

### O²Former：方向查询确有先例，但复现成本与比较资格须处理

原文的方法和消融支持“方向与多尺度查询互补”的作者内实验解释；然而 ImageNet→HRSID→SSDD 的初始化链及500epoch使其不适合与当前不同配方的短预算结果直接比大小。尤其不能把其离岸71.9和本项目整集maskAP放在同一胜负表。若后续采用方向信息，应冻结查询/预算等因素，设置方向信息、普通位置编码和等容量卷积的对照，才判断改进是否来自方向建模。出处：[P1]的III、IV节及消融表。

### YOLOv11→SAM2：是实际可构成新强基线的思路，但论文没有提供maskAP证据

原文方法非常清楚：检测器给框，SAM2给掩膜，候选按其预测IoU选择；这适合快速检验基础模型在SAR上的边界质量。其定性失败包含邻近舰船合并，说明“逐框提示独立分割”未必解决密集实例互斥。实际操作必须以**预测框**做主要评估、独立保存检测分数和SAM质量分数；GT框只能是诊断上限。还需在本项目COCO评估器重新计算maskAP，不能把0.637 mIoU解释为63.7 AP。出处：[P3]的3、4节。

### GCIE-Net：现代实时Transformer已进入SAR实例分割，不应把“DEIM+mask”包装成新创新

原文给出了完整四阶段训练和从零训练设定，值得作为当代方法设计先例。其MGEA单独加入时小目标AP下降、而中大目标改善；联合CGFF后才得到最终结果。因此“全局注意力普遍增强小船”不是原文支持的结论。基于此提出的高分辨率细节路由仍只能作为候选假设，需要本项目真实误差验证。该文对RF-DETR-Seg的报告也不能代表其所有官方规模、预训练与充分调参条件。出处：[P2m]第4节、表3/4/6/7。

## 4. 建议优先运行的两个候选

这两项按**可核验可执行性和研究区分度**排序，不宣称它们已经在本项目涨点。论文报告值不能替代本项目基线。

### 候选A：RF-DETR-Seg 官方实例分割基线 → 选择性高分辨率残差修正

- 可执行起点：[C2]官方代码、分割模型与配置入口；截至核验时README确认分割Nano至2XLarge及包为Apache2.0。先固定一个规模和版本，保留官方初始化，不在比较过程中换版本、输入尺度或训练配方。官方COCO吞吐/精度只说明通用模型能力，不能外推SAR排名。
- **最小修改（待检验）**：保持检测查询、候选框和分类分数路径相同，只对存在掩膜边界不确定性的实例，在更高分辨率浅层特征上预测零初始化残差；路由依据仅来自预测与图像。若基线错误主要是漏检或错误框，而不是掩膜细节，此设计优先级立即降低。
- 直接先例：GCIE-Net的查询×像素掩膜和跨尺度融合；O²Former多尺度查询；通用CV中的PointRend/掩膜细化已是已知范式。“高分辨率+残差”本身不能申报首次出现，必须进一步核对选择机制和SAR误差特征的具体差异。
- **最强替代解释**：额外像素、参数、推理计算，或插值/阈值改变造成收益。因此必须设等容量普通卷积、无条件全实例细化，以及相同分辨率/计算预算对照；同时报maskAP、boxAP、AP75、耗时和显存。
- 决策：官方基线先跑通并训练；同预算对照进入筛选，再按统一长预算确认。未经重复不赋予“创新有效”标签。

### 候选B：预测检测框 → 官方SAM2.1 → 邻近实例约束提示

- 可执行起点：复用现有合格预测检测框，加[C1]官方SAM2；无需重训mask即可得到独立基础模型基线。SAM2.1与[P3]使用SAM2不能混称逐项作者复现。SAM2提供Apache2.0主体代码/模型与BSD3可选组件，依其官方许可证保存声明。
- **最小修改（待检验）**：仅当验证集错误分析确认邻船掩膜合并时，使用其他**预测**实例中心/框产生排他负点，并约束一个实例的候选选择。不能使用GT邻船点、GT掩膜或测试集反馈。先比较无训练提示策略，再判断是否需要可训练小适配器。
- 直接先例：[P3]已经是检测器+SAM2；SARSAM/SAMSAR及RSPrompter等已覆盖SAR/遥感适配、局部结构与自动提示。因此自动框提示、频率适配或CNN/SAM融合不能单独成为新贡献。
- **最强替代解释**：提示数量增加、检测框更好、掩膜截断/NMS或阈值调参引起收益。固定预测框和分数，加入同提示数随机/中心负点、简单框裁剪/掩膜互斥的对照；完整流水线也单独评估。处理提示外小船时保留召回分析。
- 代价边界：SAM2编码可按图像复用，但总体速度、模型体积和细化次数仍需实测；不以冻结参数少推导整个系统高效。

两个候选都只是“强基线+可证伪修改”。若改动没有增益，应保留统一预算的完整失败记录；若短预算增益存在，下一步是相同初始化、预算与评估的长训及重复种子，不是先写机制解释。

## 5. 公平验证与阶段出口

1. **先建立可比协议**：相同官方数据列表/掩膜版本、训练与验证隔离、COCO maskAP评估器、输入尺度、有效batch、预训练记录、数据增强与推理规则。不得将外部文献AP直接作为同条件对照。
2. **同预算筛选**：基线与修改共同使用短预算；保存完整学习曲线与耗时。仅可结论“该预算与配方下相对表现”，不宣布方法总体失效。
3. **晋级条件**：主要maskAP有明确可重复收益，AP75/密集邻船或小目标诊断符合预设假设，且增益无法由容量/分辨率/提示数对照解释；再启动统一长预算。
4. **确认阶段**：至少3个重复种子，报告均值、离散程度与配对比较；验证集选择模型与阈值，测试集只作最终评估。短预算和长预算各自独立成表。
5. **跨数据集证据**：SSDD和HRSID的同域训练分别确认；跨数据集迁移与同域训练分开列。不得把HRSID→SSDD预训练与从零SSDD作未经说明的机制归因。
6. **论文出口**：明确一个核心问题、一项可证伪机制、与直接先例的技术差异及主要证据。实验收益、部署效率和适用边界均可形成贡献，不强求凑三个模块；尚未达到以上证据时不能认定TGRS投稿成熟。

## 6. 原始出处与访问记录

- [P1] [O²Former arXiv记录](https://arxiv.org/abs/2506.11913)，[v1原文HTML](https://arxiv.org/html/2506.11913v1)。方法、实验设置与按场景表均已读。
- [P2] [GCIE-Net出版方](https://www.mdpi.com/2072-4292/18/15/2512)，[DOI](https://doi.org/10.3390/rs18152512)。出版方直接页面多次429；可检索出版方缓存方法/消融。
- [P2m] [MDPI提供的GCIE-Net原文镜像](https://www.researchgate.net/publication/411150644_GCIE-Net_A_Global_and_Channel_Information-Enhanced_Network_for_Ship_Instance_Segmentation_in_SAR_Images)。页面明确“Publisher Full-text provided by MDPI”；本文只保存自有摘要和链接，没有复制全文。方法与实验原文已读。
- [P3] [Mansour等arXiv记录](https://arxiv.org/abs/2604.17920)，[v1原文HTML](https://arxiv.org/html/2604.17920v1)。方法、实验与指标定义已读；[文中作者代码链](https://github.com/IslamAlam/hybrivision)于本次核验返回404。
- [P4] [SARSAM出版方](https://www.mdpi.com/2072-4292/17/7/1209)，[DOI](https://doi.org/10.3390/rs17071209)。出版方缓存结论、讨论和数据声明已读；完整正文直接获取受限。作者名单/发布日期另与出版方提交的Crossref元数据核对。
- [P5] [DSR-Net出版方](https://www.ejournal.org.cn/cn/article/doi/10.12263/DZXB.20260699)，[DOI](https://doi.org/10.12263/DZXB.20260699)。书目信息可读；正文VIP限制，局部出版方检索缓存可见。
- [P6] [SAMSAR出版方](https://www.sciencedirect.com/science/article/abs/pii/S0957417425014745)，[DOI](https://doi.org/10.1016/j.eswa.2025.127852)。出版方摘要检索可读，完整直开403；不以二手综述补数值。
- [C1] [Meta官方SAM2代码、权重与许可入口](https://github.com/facebookresearch/sam2)。本次仅核验公开页面，没有下载权重或执行代码。
- [C2] [Roboflow官方RF-DETR实例分割代码及许可](https://github.com/roboflow/rf-detr)。本次仅核验公开页面；实际实施必须保存commit与具体模型，而不是跟随develop浮动。

最后状态：文献核验完成至上述范围；全部候选修改为待执行/待检验；无新实验收益、无新SOTA宣称、无TGRS成功宣称。
