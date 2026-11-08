# SAR 船舶实例分割原文与源码核验续查

检索截至 2026-11-08。覆盖图优化、逆问题、雷达散射统计、结构预训练及直接分割对照。以正式出版记录或固定版本作者稿追溯代码，再核对实际函数、配置、许可证和输入前提。源码核验是静态阅读。文献各自的任务、协议和分数分开记录。未公开、不可检索及当前访问受限的来源仍有覆盖缺口。 [前一轮 30 条来源及方法索引](PRIMARY_METHODS_SOURCE_AUDIT_AND_NEXT_TEST_20261107.md)

## 数学图优化与可运行对照

**可微 Power Watershed，2026-10-28 预印本。** 已核读原文 §2–4、Table 1–3 及作者固定代码。四邻域代价权重由 $w_{uv}=\exp(-p c_{uv})$ 给出，以固定标记和一元项确定各标签概率。有限 $p$ 时按标签求稀疏线性系统，隐式反传再解一次伴随系统。较大的 $p\Delta c$ 恶化条件数并使梯度集中。原文对 PanNuke 的实例 F1@0.5 报告从连接域 U-Net 的 0.656 到 0.795，同等标记监督下达到与调参后的普通分水岭相当的结果。少量标注条件另有 0.015–0.020 F1 增益，适用范围是作者的核实例实验。纯图像梯度代价表现弱，错误边缘可直接把轮廓引向错误位置。对应实现为 `pw_layer.py:ImplicitRW, ImplicitRWMulti, MultiRWLayer`、`pw_fast.py:power_watershed_multi_fast` 和 `exp_instance_pannuke.py:InstanceSegmenter, predicted_markers, ws_labels`。代码 BSD-3-Clause，正文、图和结果表在作者仓库另行采用 CC BY-SA 4.0。本地保存原 PDF、三个代码文件和许可证及 SHA-256 清单。 [作者原文](https://arxiv.org/pdf/2610.38516v1) · [线性求解源码](https://github.com/hugues-talbot/differentiable-power-watershed/blob/691edd36c7d4d424346b1c74bc35d90c3dc2fb06/experiments/pw_layer.py) · [实例实验](https://github.com/hugues-talbot/differentiable-power-watershed/blob/691edd36c7d4d424346b1c74bc35d90c3dc2fb06/experiments/exp_instance_pannuke.py) · [许可证](https://github.com/hugues-talbot/differentiable-power-watershed/blob/691edd36c7d4d424346b1c74bc35d90c3dc2fb06/LICENSE)

**Falcon，2025 预印本。** 作者 §3.3 在 ViT 图上用分式二次变换与正则项交替求 K 路 normalized cut，§3.5 用 RGB 和可选深度细化。其作者链接在本轮公开 GitHub API 返回 404，无法核实可执行源码或许可。全局无监督分区及光学深度边缘先例，对当前固定 SAR 实例候选的评估另设任务。 [原文](https://arxiv.org/html/2504.05613v1) · [原文所列代码入口](https://github.com/KordingLab/Falcon)

**ARANet、LFG-Net 与 YOLO26。** ARANet 作者接受稿 §III-B、Fig. 3–4 的浅层 Gabor、平滑 Laplacian 和高通加权融合已经覆盖“多方向高频边缘 + SAR 骨架”的通用表述，本轮未核得可执行作者源码。LFG-Net 的作者 Apache-2.0 实现中已核到 LFCP、ROI 交互、DCT 掩膜编码和训练入口。R50 配置使用 128×128 掩膜和 700 DCT 系数，作者 HRSID 入口硬编码，复训需重配同协议数据和旧版 MMDetection 依赖。本轮取得其 IEEE 正式摘要与源码，全文和作者权重仍未核得。YOLO26n-seg 是额外的低成本通用对照，作者预印本报告 2.7M 参数、COCO mask AP 33.9（端到端）和 34.4（NMS），对应代码有 `Segment26`、`Proto26`、`v8SegmentationLoss`。官方 v8.4.0 权重 6,719,965 B，发布 SHA-256 为 `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`，代码 AGPL-3.0。正式比较须预锁后处理、训练预算、初始化和原图 RLE 评价。 [ARANet 正刊](https://doi.org/10.1109/JSTARS.2026.3720719) · [作者接受稿](https://www.researchgate.net/publication/411814256_ARANet_A_Unified_Representation_Alignment_Framework_for_SAR_Ship_Instance_Segmentation) · [LFG-Net IEEE 记录](https://doi.org/10.1109/TGRS.2022.3188677) · [LFG-Net 掩膜头](https://github.com/Evarray/LFG-Net/blob/8c6344b23a5c2a72ae7bb6a95f954645db154533/mmdet/models/roi_heads/mask_heads/crsb.py) · [YOLO26 原文](https://arxiv.org/html/2606.03748v1) · [固定源码](https://github.com/ultralytics/ultralytics/blob/f2d3aed634a5b0e4828024718d4a61ab2f83fb19/ultralytics/nn/modules/head.py#L324) · [官方权重](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt)

## 物理、逆问题与数据前提

| 方法 | 核验结果 | 当前显示灰度的适用性 |
| --- | --- | --- |
| γ-Bridge，2026-07 | 原文 §III 给出按 looks 参数化的 Gamma 观测桥、Gamma–Lévy 分解及解析后验。公开作者仓库 API 当前 404，未读到可执行代码和许可。 | 依赖线性强度、乘性 Gamma 与可解释 looks。单独检验恢复指标和下游 AP。 [作者原文](https://arxiv.org/html/2607.22719v1) |
| Geometry-Calibrated Closed-Form Shrinkage，2026-08 | 原文 §III 与补充给出 log–Yeo–Johnson、相似补丁分组 SVD 和长宽比校准收缩。作者仓库当前 404。 | 灰度上能重建经验去噪对照。物理噪声解释要核辐射映射、噪声分布及小船轮廓保真。 [作者原文](https://arxiv.org/html/2608.15028v1) |
| PELNet，ISPRS JPRS 2026 | 作者固定树的 `reconstruct.py` 实施复 3×3 旋转与重构。README 声称 MIT，固定树无许可证文件，GitHub 许可字段未确认。 | 需要复极化数据与通道标定。SSDD/HRSID 显示灰度缺少极化矩阵。 [正式论文](https://doi.org/10.1016/j.isprsjprs.2026.08.040) · [固定源码](https://github.com/shurongz/PhyDecomp/blob/bd44cdb6ef0eb2008c91fafbf6ee31f0b6192080/reconstruct.py#L114) |
| SDUDNet，IEEE JSTARS 2025 | §III 用非配对干净光学和 SAR 构造噪声伪样本，训练/网络/测试源码可见。已见合成和真实权重文件。论文写 80 轮，固定版代码默认 30 轮。文章的 CC BY 4.0 与代码许可分开，固定树未核到许可证文件。 | 可检验训练期外部数据先验。以相同数据量、骨架和增强的 SAR-only 对照分离先验效用，并独立核查小目标。 [正式论文](https://doi.org/10.1109/JSTARS.2025.3568854) · [作者全文](https://www.researchgate.net/publication/391687101_Speckle-Driven_Unsupervised_Despeckling_for_SAR_Images) · [固定训练源码](https://github.com/BFY-official/SDUDNet/blob/0c799911e84ef79c5fbe6f58f44cb8ee033048a1/train.py#L210) |
| SemDNet，ESWA 2026 卷期 | 作者实现 `RefinedNet` 需要六通道语义输入，公开测试入口给该输入随机张量并读取未随源码提供的权重。公开 `train.py` 只覆盖粗恢复支路，代码许可未核。 | 论文为去斑任务，当前实现缺忠实推理链。 [正式论文](https://doi.org/10.1016/j.eswa.2025.129200) · [公开测试入口](https://github.com/BFY-official/SemDNet/blob/b0265ae7e0dfe81fae5c8430d6e62a180f7d1cb3/test_refinednet.py#L38-L63) |
| PolSARtools，SoftwareX 2026 | 作者全文和 GPLv3 源码可核验。全极化 H/A/α 入口从 T3/C3 矩阵执行特征分解。 | 可作为复极化分解参考。当前灰度无需参与该链。 [作者原文](https://authors.library.caltech.edu/records/8fx6z-wbt88) · [固定源码](https://github.com/polsartools/polsartools/blob/1e9889d6850b07dcb6567400297bc551503cd7dc/polsartools/polsar/fp/h_a_alpha_fp.py) |
| OSP，2026-05 预印本 | 附录 C.3 为范围压缩后的相干脉冲流处理。论文承诺公开，当前未核到其实际训练代码。上游 Maya4 另有公开原始/中间级数据和 GPL-3.0-or-later API。 | 只在拥有复脉冲、几何和采样链的分支复现，聚焦质量独立评价。 [OSP 原文](https://arxiv.org/html/2605.10340v1) · [Maya4 固定 API](https://github.com/sirbastiano/Maya4/blob/c5357765d42d14cbd269524e26dff1fd766a55ab/maya4/api.py) |

即便输入真为单视指数散斑 $N$，严格递增的显示映射 $N^2$ 也能把均方值与方差之比 $E[N]^2/\operatorname{Var}(N)$ 从 1 改为 $4/(24-4)=0.2$，实际 looks 未改变。该反例说明，显示灰度的局部 ENL 代理不能单独验证物理 looks。灰度标准化如果进入对照，保持支持域和标记固定，另用区域成员打乱来区分边缘对齐与强度分布效应。

## 基础模型与预训练重合

**SARATR-X-v2。** 作者 v2 Table II 明示预训练包含 HRSID 4,623 图、SSDD 1,044 图。原文称继承的集合排除测试样本，缺与本项目留出精确对齐的逐图证据。原文约 36 万图与发布页 50 万图的权重谱系有待对齐。下游 GFL 指标是水平框检测。代码的 `SAR_Lay`、`SAR_Layer`、`My_SAR_feature` 实现多个尺度的对向区域对数比和融合，源码许可 MIT，权重另有许可和申请流程。可以用当前灰度独立研究结构目标或固定描述子，以相同骨架、通道、计算预算和原灰度旁路控制增益。对数比消除的是两侧共享的乘性增益。显示映射、量化和局部非均匀扰动仍需单独控制。 [作者 v2](https://arxiv.org/html/2607.23238v2#S4.SS1) · [结构算子源码](https://github.com/waterdisappear/SARATR-X-v2/blob/bbb09a66dec5af902b739d8ca0ebae6585b95925/pre-training/models/masked_autoencoder.py#L36) · [权重发布页](https://huggingface.co/waterdisappear/SARATR-X-v2)

**CrossEarth-SAR。** 作者原文 §3.2–3.3 用描述子引导稀疏专家，训练期可用光学伪标签，公开实验为地物语义分割。固定版的 `compute_sar_descriptor` 从三通道 ImageNet 反归一化后取均值再取 log1p。方向直方图通过 `grad_angle.flatten()` 汇总整批再扩展到 B 个样本，因此 B>1 时同一方向熵进入每张图，须做“同一张图与不同伴随样本”对照。ENL 是所用对数显示灰度的经验代理。固定树未核到根许可证，发布权重许可未确认。 [作者原文](https://arxiv.org/html/2603.12008v1) · [固定描述子源码](https://github.com/VisionXLab/CrossEarth-SAR/blob/693282718295dc81a9f4004885857459170ce4b8/CrossEarthSAR/models/backbones/dino_v2_sar_moe_balance.py#L280) · [固定权重入口](https://huggingface.co/conquer997/CrossEarth-SAR/blob/65afdd0c83d5da36b3027907755808865d5c0e05/crossearthsar_vits.pth)

## 下一项可证伪的诊断

现有交叉实验显示，等结构排序头单独编辑匹配候选时 AP 和 AP75 持平，未匹配候选上的编辑重现该头的损失。均值和不确定性两臂的匹配与未匹配效果方向不同。下一步先用固定原生预测计数可观测的候选重叠，并把 GT 只用于核对真实相邻实例和错误类型。只在 FIT 探索子集中核验重叠是否与错误相关。按图像或同源组配对，在同一重叠带固定候选、原始分数、标记与限额，先比较原始掩膜、逐像素归属、普通标记分水岭。有限 $p$ 随机游走作为额外机制对照，报告条件数代理、迭代数、收敛残差和 CPU/GPU 延迟。若重叠解释不了错误，或普通分水岭得到相当的匹配 IoU、AP75 与未匹配候选效应，证据指向降低图求解优先级。若使用无邻船或未知置信候选产生标记，只能由预测和图像生成，GT 种子单列上限。该诊断尚未执行，已停止的排序头作为已完成对照。 [FIT 交叉实测](RFDETR_E24_FIT_MATCH_CROSSREAD_ACTUAL_20261108.md)
