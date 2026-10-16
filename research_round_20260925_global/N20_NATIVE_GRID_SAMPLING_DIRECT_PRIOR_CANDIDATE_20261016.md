# N20 原生网格取点：直接先例与覆盖率诊断候选

2026-10-16。状态：**条件候选、未经实验**。独立查证后，Root 复核了下列作者原稿、代码与既有普通设计记录。本备忘的文献与设计核查未访问当前绑定科学源码、权重、数据、缓存或资格链；当前 N20 执行配方未改，没有新增训练、收益或已确认瓶颈结论。

来源边界：两次 resize 的前提仅转述自已读 [N20 设计状态文档第 17 行](N20_TRUE_FEATURE_NEIGHBOR_SCOUT_DESIGN_AND_IMPLEMENTATION_STATUS_20261015.md)：分组 FP32 bilinear 至 800×800、左上裁剪、再 bilinear 至原图，两次 align_corners=False；该文第 19 行定义渲染后整数 ±1 九邻及 clamp。**这不是对当前冻结执行源码的重新核验。** 关联科学任务实际终止且满足原资格规则后，仍须核实当前执行所绑定的渲染/裁剪/索引/边界路径是否与上述文档一致，再确定精确组合插值支持。

| 已核直接先例与发表身份 | 覆盖内容及边界 |
| --- | --- |
| Chen、Liu、Wang，*Learning Continuous Image Representation With Local Implicit Image Function*（LIIF），CVPR 2021，已发表。[正式页](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html)、[作者原稿](https://arxiv.org/pdf/2012.09161)、[作者代码](https://github.com/yinboc/liif/blob/main/models/liif.py) | 原论文式(3)为原生 latent 网格 3×3 邻域拼接，式(4)为四个邻近 latent 的局部预测集成；代码含 `F.unfold(feat, 3, padding=1)`。这是最直接的“按原生特征网格取九邻”先例。任务为连续图像表示/超分，不等于冻结 SAR ERROR scout。 |
| Zhu 等，*Deformable DETR: Deformable Transformers for End-to-End Object Detection*，ICLR 2021 Oral，已发表；首个预印本 2020。[作者论文页及会议标注](https://arxiv.org/abs/2010.04159)、[正式会议页](https://openreview.net/forum?id=gZ9hCDWe6ke)、[作者代码](https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py) | 已有多尺度稀疏取点；作者代码的二维参考点分支以每层 `(W,H)` 归一化偏移，初始化半径按 1…K 展开。四维框分支另按框尺度归一化，不能混写；学习后不保证每点始终跨单元。正式会议页本轮触发浏览器校验，发表身份另由作者 arXiv 标注核对。 |
| Hu 等，*Learning Implicit Feature Alignment Function for Semantic Segmentation*（IFA），ECCV 2022，已发表。[正式论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890479.pdf)、[作者仓库](https://github.com/hzhupku/IFA) | 式(1)、(4)使用各层最近原生 latent、相对坐标及其编码来独立预测点，覆盖绕开预先双线性对齐、多尺度坐标条件化。正文并非每层 3×3 展开，不能把它说成同一种九邻。 |
| Cheng、Parkhi、Kirillov，*Pointly-Supervised Instance Segmentation*，CVPR 2022，已发表（预印本 2021），含 Implicit PointRend。[正式页](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)、[作者原稿 §4.1–4.3](https://arxiv.org/pdf/2104.06404) | 原稿明确 PointRend 的点特征来自同点双线性插值；Implicit PointRend 保留这类特征，新增实例条件点头与框内位置编码。它不是解决“九点同插值单元”的直接对照，不能作为该机制已被检验的证据。 |

当前可保留的研究问题是：当前冻结 teacher、固定中心与监督下，渲染后像素九邻是否大量共享同一原生支持，原生步长/跨单元取点是否增加有用观察。跨格点邻域与多尺度采样原语已有先例，不能单独主张创新。两份已读文档未覆盖 LIIF/IFA 这条直接关系，因此本备忘补充的是先例与可证伪前提，不是新机制成立证据。

数学边界：对**单次**双线性插值、九点确实共享同四个输入角点、且尚未做非线性变换时，九个原始向量处于这四向量张成的空间，条件秩至多 4。当前文档描述的是两次 resize 加裁剪；第二阶段同单元不自动等于共享四个原生 F 向量，应追踪两阶段组合支持。该上界不能推广到各点 LN 后。真实共享覆盖率、实际 F 的数值秩及 LN 后几何均未测量。

最有竞争力的替代解释：较大步长只增加普通上下文/感受野，或改善输入协方差与优化条件；不必由“解除低秩瓶颈”解释。中心复制控制匹配名义参数量，但不能排除这些解释。

下一步仅一个最小证伪诊断（未来候选，当前未执行）：依当前 N20 停止规则决定是否推进；关联任务终止并完成源码资格复核后，在未来另行预锁的 FIT 子集上由**精确两阶段坐标链**列出每个九邻的原生支持索引和组合权重，汇总同四角覆盖率、不同支持数与插值系数矩阵的秩（代数诊断）；clamp 点单列。先检验“像素九邻普遍共享四角”这一必要前提；若不成立，就撤回由共享四角推出低秩的解释。即便成立，也仅支持几何冗余前提，不能证明 LN 后低秩、路由信息不足或 AP 收益；后续是否比较原生步长须另行预锁，不能据此改当前 N20。

执行边界：子集规模、“普遍共享”的量化定义、统计口径与容差尚未预锁，本备忘不是可直接执行的实验注册。组合系数矩阵描述插值代数结构，不替代实际 FP32 特征数值秩或渲染结果的逐位核验。是否推进诊断及后续训练，仍服从当前 N20 的预锁停止规则。

复核范围：本次为定向先例补充，不能作为穷尽查新或首次性证明。Root 复核时部分 CVF 页面未能取回正文，LIIF 和 Pointly 的会议身份另核作者 arXiv 标注（[LIIF](https://arxiv.org/abs/2012.09161)、[Pointly](https://arxiv.org/abs/2104.06404)），方法内容核作者原稿；Deformable DETR 正式页访问限制见表内记录。IFA 的最近 latent 及多层坐标定义核 ECVA 正文式(1)、(4)。
