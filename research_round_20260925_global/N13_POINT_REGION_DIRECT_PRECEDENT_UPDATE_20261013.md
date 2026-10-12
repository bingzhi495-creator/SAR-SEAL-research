# N13 同预算区域精修：直接先例定点更新

日期：2026-10-13。状态：**普通文献与机制判断备忘录；STOP，交 ROOT 同步。** 本轮先读独立发布副本已有综述，再核对下列六项原论文/作者实现。只读取普通发布文档与在线 primary 文本；没有读取、哈希或执行任何 SCI、运行/将运行的 SOURCE、SPEC、Q、缓存、权重或 GT。没有修改当前源码、配方或训练臂。

## 结论

此次有限检索**未确认完整相同做法**：同一个 point head、每 query 同 K，按域内 opposite-class Chebyshev distance、abs(L)、flat index 的字典序选支持，再做768区域有放回点＋共同256全图均匀点、区域外继承原生输出。这个结论仅限本轮查到的原文与实现，不能证明首次组合，也不是发表级新颖性认证。

最接近的原语已有清楚先例：不确定点 MLP、75%困难点＋25%随机点、边界区域监督、带外继承、冻结粗模型稀疏精修和距离引导选点。N13 的可讨论差异集中在**原图每实例编辑配额不变时，几何距离与 logit 不确定性的支持分配**。它值得公平检验；但距离排序本身是很强的常规边界基线，不能仅给 lexsort 或采样配方命名便算核心创新。

## 已有范围与本轮新增事实

先读：[掩膜精修综述](../docs/mask_refinement_prior_review_20261011.md)、[N07直接先例](../docs/N07_point_patch_refinement_direct_prior_art_20261012.md)、[监督与采样补查](../docs/sampling_supervision_direct_prior_update_20261012.md)、[N11区域假设](N11_REGION_CHOICE_NEXT_HYPOTHESIS_20261013.md)、[固定带采样备忘录](FIXED_BAND_SAMPLING_RISK_PRECEDENT_MEMO_20261013.md)。它们已覆盖 Mask Transfiner、BPR、Implicit PointRend、Prior2Former、SAM-REF、Phoenix及有关 SAR 方法。本轮不重新堆叠这些摘要，不用不同协议的论文数字做排名。

| 重点先例与可核日期 | 本轮核对的机制/代码事实 | 与 N13 的具体差异、强替代意义 |
|---|---|---|
| **PointRend，CVPR2020**；arXiv v1 **2019-12-17**，v2 **2020-02-16**。[日期](https://arxiv.org/abs/1912.08193) | 作者训练采样先均匀过采样连续坐标，插值 logits 后 uncertainty top-k，再加独立均匀点；默认 importance ratio0.75、oversample3。推理按当前输出不确定性逐级细分选点。[采样函数](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py)、[配置](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/config.py) | 75/25及点 MLP均非新原语。N13是在冻结原图整数支持内均匀有放回采样，并固定训练缓存；不是作者随机候选 top-k 或动态细分的完整复现。受控 PR 适配能检验区域效用，不能代替完整 PointRend 对比。 |
| **RefineMask，CVPR2021**；arXiv v1 **2021-04-17**。[日期](https://arxiv.org/abs/2104.08569) | BAR后期监督取预测边界与 GT 边界并集，区域平均 BCE；训练默认width2，继承/推理width1。作者 block target 用方形卷积、归一化后0.1阈值、零填充，label2覆盖label1。[固定文件版本 bf81cac…作者loss](https://raw.githubusercontent.com/zhanggang001/RefineMask/bf81cac8090115e80c690cd94c4e25a1e6550603/mmdet/models/losses/cross_entropy_loss.py)、[推理继承](https://raw.githubusercontent.com/zhanggang001/RefineMask/main/mmdet/models/roi_heads/refine_roi_head.py) | 只修边界、区域外继承已有直接先例。N13用预测 P/L选区域，不用 GT 边界选点；精确域内 Chebyshev排序不是该阈值 block operator。原图width2只能叫适配；原作者RoI多级特征与尺度不同。 |
| **SharpContour，CVPR2022**；arXiv v1 **2022-03-24**。[日期](https://arxiv.org/abs/2203.13312) | 粗轮廓顶点沿法向离散搜索，实例动态参数点分类器判断内外，迭代移动轮廓；不是密集全图点排序。[原文方法](https://arxiv.org/html/2203.13312v1)、[作者项目页](https://xyzhang17.github.io/SharpContour/) | 边界几何＋实例点分类已有强先例。N13没有顶点位移/动态参数，属于固定像素支持残差。项目页仍写代码“Coming soon”；本轮不能标成已核得可运行作者代码。 |
| **SparseRefine，ECCV2024**；arXiv v1 **2024-07-26**。[日期](https://arxiv.org/abs/2407.19014)、[正式会议入口](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8392_ECCV_2024_paper.php) | 冻结粗模型，熵阈值选择像素，稀疏高分辨率 RGB 特征提取与 gated ensemble。实际作者selector是softmax entropy>threshold；refiner先clone原 logits，仅在mask内赋新值。[selector](https://raw.githubusercontent.com/mit-han-lab/sparserefine/master/sparserefine/models/selectors/entropy.py)、[refiner](https://raw.githubusercontent.com/mit-han-lab/sparserefine/master/sparserefine/models/refiner.py) | 冻结载体、稀疏区域、带外保持均已有实现。N13是实例query/同K、共享原生F点头，没有高分辨率RGB稀疏骨干；阈值密度与每query预算不同。同K也不等于同延迟，需计入distance/full sort成本。 |
| **SAMRefiner，ICLR2025**；arXiv v1 **2025-02-10**，v2 **2025-03-16**。[日期](https://arxiv.org/abs/2502.06756)、[正式会议入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html) | 作者extract_points用FastGeodis，lamb0、iterations2；正点取距背景**最大**位置，负点反向距前景且限制bbox后取**最大**，ties取首坐标。默认1正＋1负提示、迭代重新生成，默认五轮。[utils](https://raw.githubusercontent.com/linyq2117/SAMRefiner/main/utils.py)、[流程](https://raw.githubusercontent.com/linyq2117/SAMRefiner/main/sam_refiner.py) | 距离引导点已有实现，但选的是稳健内/外提示，不是boundary-min同K修补像素；SAM能修改整mask。不能因有distance keyword就判为N13完全相同，也不能忽略其几何选点先例。 |
| **SeSAM / Do Instance Priors Help Weakly Supervised Semantic Segmentation?**；arXiv v1 **2026-04-13**，本轮只按预印本处理。[日期](https://arxiv.org/abs/2604.11170) | 归一化Euclidean距背景构造采样概率，bbox网格内选K个cell各取一点，主配置K5；Table4直接比较confidence、DT-max中心、boundary、前景均匀与skeleton。[原文§3.2、§4.3、实现段](https://arxiv.org/html/2604.11170v1) | 这是近期几何/置信度/覆盖选点的直接近邻，说明“比较边界与置信度”不是空白。但它采SAM prompts并做弱标签扩展，不是N13共享点头的可编辑支持。未核得论文作者代码入口，不称实现复现已验证。 |

除 RefineMask 上述固定文件版本外，main/master代码链接是**2026-10-13检查时的公开文本**，不是锁定的实验复现版本；正式实施作者对照还须固定版本、配方与输入。

检索包含2024—2026及**2026-04-13至2026-10-13**窗口，定向词覆盖 point/mask refinement、boundary-distance sampling、uncertainty、fixed-budget、Chebyshev、lexsort；SeSAM满足该窗口日期。既有Phoenix索引仍保留。本轮是六项定点机制查新，未作多数据库系统穷尽、所有被引文献递归审阅或跨模型创新认证。

## 方法差异必须准确写

N13的K是每query原生 `|L|<=2` 像素数，**跨query可变**；1024是每实例监督点数，两者不是同一预算。在有限L下，按 `(abs(L),flat)` 取exact K与旧B严格同集合；不能把这两种写成两个独立竞争方法。二值预测的常规entropy按不确定性排序，在精确数学下也只是abs(L)的单调变换，换名称不能增加强对照。

几何排序 `(D,abs(L),flat)` 可以理解为：先纳入所有较近Chebyshev距离层，仅在最后一个未完全纳入的距离层按abs(L)和flat裁切。因此目前是**边界优先、层内不确定性打破并列**，不是已学习的几何/不确定性联合选择器。域外不造边界、全空/全满时D无异类而回退、deterministic flat ties是必要的定义与复现细节，本身不足以证明学术贡献。

设计好的部分是：同教师/point head/初始化/监督量/每query K，GT仅用于监督；共享uniform与rank draws降低配对随机差异，固定原生分数/全部query便于隔离几何。它们使正负结果更可解释。工程可核验性值得保留，但不能单独作为新分割机制或TGRS就绪证据。

主要不足是：几何操作仍是常规自适应边界带；D没有SAR物理信息，离预测轮廓很远的高置信FN仍可漏掉，错误小岛的边界也会被优先。改善条件oracle不代表同一共享点头可实现，少数匹配对改善不代表全部GT/AP。现阶段准确定位应是“受控区域选择假设及其效用待检验”，不把采样、距离算子、点头拆成三个已成立创新点。

## 真正可能改变后续判断的控制（不新增当前训练臂）

当前已锁N13两臂、种子与训练预算保持不变。以下仅是取得可重复训练证据后，针对机制主张的下一步候选，须另登记：

1. **普通几何替代能否解释收益**：同K、同点头比较 `(D,flat)` 与当前 `(D,abs(L),flat)`；它直接检验层内不确定性是否必要。再用RefineMask原作者block区域适配及预锁成本/预算曲线，避免只赢弱logit阈值便说胜过边界方法。若去掉abs(L)不受影响，就承认收益主要来自常规几何支持。
2. **预算与覆盖的替代解释**：同K随机支持或预锁网格覆盖支持、相同有放回/共同uniform监督；与原图boundary-distance配额比较。它检验是否只是换到新位置/增加覆盖而非边界机制。SeSAM的prompt结果只能说明此替代有文献依据，不能代替SAR控制结果。
3. **GT导向训练的强先例差异**：若要声称优于RefineMask式区域监督，单独明确作者预测/GT边界并集与N13预测only的差别；不能把多给GT边界选点产生的收益归因于推理几何规则，也不能给一方更弱配方后宣称完整方法优越。
4. **实际收益与代价**：配对种子/足够训练、全预测固定评价、FP/FN修复及新破坏像素、高IoU与跨公共数据集、选择/采样/训练/推理成本同时成立。若仅条件coverage/oracle提高而实际mask效用没有重复收益，保留支持诊断而撤回有效修补主张。

这些建议不要求现在分叉训练。优先完成既有公平两臂，避免同时换骨架、损失与区域造成归因不清。即使正收益成立，投稿贡献仍要落到“先例遗漏了什么错误、哪条具体规则与强常规替代有可测差异、在什么数据/预算内有效”；本备忘录不承诺首创、SOTA或TGRS发表水平。
