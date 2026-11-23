# 如何把 SAR 船舶轮廓实验写清楚：17 篇遥感论文的图、表与公式

2026-11-23。样本包括 13 篇 *IEEE Transactions on Geoscience and Remote Sensing*（TGRS），以及 JSTARS、《雷达学报》、*Remote Sensing* 和 CVPR 各 1 篇。选文优先考虑原图与轮廓对照、传感器与标注来源、受控实验，以及版面上如何连接公式与观测。下文页码从所链接 PDF 的第一页起算，可能与正式页码不同。SSDD 使用出版社 HTML 中的图号。16 篇 PDF 均逐篇渲染过所述图表所在的一页，其余相关页核对了可检索的正文与图表说明。每条分别记录原文所示内容及本稿可采用的做法。这里没有进行跨论文性能比较。

## TGRS：从一个可见的误差写起

1. **Xi Yang 等，*A Robust One-Stage Detector for Multiscale Ship Detection With Complex Background in Massive SAR Images*，2022。** PDF 第 8 页 Fig. 6、7 在相同场景中反复使用同一裁窗，各方法名称放在小图下。复杂背景与小船各占一组，图注交代真值、检出、漏检和虚警的颜色。轮廓图可沿用固定裁窗和固定误差图例，给每幅图留足像素去判断船侧多写还是少写。[正文 PDF](https://iip.tongji.edu.cn/pdf/ABO.pdf#page=8) · [DOI](https://doi.org/10.1109/TGRS.2021.3128060)

2. **Yuqun Yang 等，*AR²Det: An Accurate and Real-Time Rotational One-Stage Ship Detector in Remote Sensing Images*，2022。** PDF 第 4 页 Fig. 1 将框和分数预测、中心预测及分数调整画成不同分支，后接专图解释局部操作。第 9 页 Fig. 8 再分别给出海上、近岸、多尺度、复杂背景实例。本文方法总图应让读者沿单条路径看清检测分数从哪里来、在哪一步原样保留，以及最终哪个变量会改变。[正文 PDF](https://www.comp.hkbu.edu.hk/~ymc/papers/journal/TGRS-2021-3092433-publication-version.pdf#page=4) · [DOI](https://doi.org/10.1109/TGRS.2021.3092433)

3. **Marino 与 Hajnsek，*Statistical Tests for a Ship Detector Based on the Polarimetric Notch Filter*，2015。** PDF 第 14 页 Fig. 5 对照两次 Rotterdam 的 RADARSAT-2 采集；图注说明日期、极化通道和 dB 显示。相邻 Table II 另列 TanDEM-X 对照场景的 UTC 时间、地点、入射角和地距分辨率。我们的影像比较同样应说明像元尺度及灰度显示方式，来源不提供的参数单独标明。[作者稿 PDF](https://dspace.stir.ac.uk/retrieve/47bcbd47-13e9-4a20-81e3-d77f7e5620ca/Marino_etal_CFAR.pdf#page=14) · [DOI](https://doi.org/10.1109/TGRS.2015.2402312)

4. **Geng Yang 等，*Algorithm/Hardware Codesign for Real-Time On-Satellite CNN-Based Ship Detection in SAR Imagery*，2022。** PDF 第 12 页 Figs. 11–12 用相同格式展示离岸和近岸小船；Table IV 逐阶段列检测精度、模型规模与计算量。比较后处理成本时也应拆开「执行一次模型推理」与「在已有输出上重新分配像素」，并注明计时对象和单位。[作者稿 PDF](https://www.sfu.ca/~zhenman/files/J10-TGRS2022-OSCAR-RT.pdf#page=12) · [DOI](https://doi.org/10.1109/TGRS.2022.3161499)

5. **Han 等，*Align Deep Features for Oriented Object Detection*，2022。** PDF 第 5 页 Fig. 4 在四张同尺度网格上标出不同卷积的采样位置；Fig. 5 衔接计算模块。同页 Eq. (7) 把真值框与锚框的量分别写出，并说明角度区间；第 7 页 Table II 用相同任务比较这些算子。最适合本文借鉴的是重复网格：在两张完全相同的船体边界上，只改变选中的增加点与删除点，让空间约束可以逐格复核。[作者稿 PDF](https://arxiv.org/pdf/2008.09397#page=5) · [DOI](https://doi.org/10.1109/TGRS.2021.3062048)

6. **Chen 等，*Remote Sensing Image Change Detection With Transformers*，2022。** PDF 第 8 页 Fig. 5 的每组从配准的输入图和真值开始，后续方法共用白色 TP、黑色 TN、红色 FP、绿色 FN；不同数据集各有一组。船体轮廓分析也可将「真值内增加、真值外增加、真值内删除、真值外删除」置于同一图例，同时加边线或纹理，保证灰度印刷可读。[作者稿 PDF](https://arxiv.org/pdf/2103.00208#page=8) · [DOI](https://doi.org/10.1109/TGRS.2021.3095166)

7. **Cong Zhang 等，*Efficient Inductive Vision Transformer for Oriented Object Detection in Remote Sensing Imagery*，2023。** PDF 第 13 页 Table IV 改变计算预算，Fig. 9 给出层内复杂度分布；第 14 页 Fig. 11 再把精度与计算复杂度并列比较。对受限编辑实验，图轴应写清预算究竟是可写像元、实际改写像元，还是模型开销，三者需分别计量。[正文 PDF](https://crabwq.github.io/pdf/2023%20EIVT.pdf#page=13) · [DOI](https://doi.org/10.1109/TGRS.2023.3292418)

## TGRS：把监督信号、固定条件和评价单位放在读者眼前

8. **Ma 等，*SAM-Assisted Remote Sensing Imagery Semantic Segmentation With Object and Boundary Constraints*，2024。** PDF 第 5 页 Fig. 4 把对象图和边界图各配一张像素数值示意，旁边的 Fig. 5 解释对象一致性损失，Eq. (2) 才组合分割、对象与边界三个损失项。本文也应先画清一个多边形切过单元格时的面积分数，再写监督目标；推理分支只接实际可得的输入。[作者稿 PDF](https://arxiv.org/pdf/2312.02464#page=5) · [DOI](https://doi.org/10.1109/TGRS.2024.3443420)

9. **Liu 等，*PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images*，2025。** PDF 第 9 页 Table IV 逐项考查 HRSID-inshore 上的自训练、原型正则及负提示校正；Table V 另列式 (7) 的匈牙利匹配采用余弦、L1 或 L2 距离时在 1、2、3 个提示点下的结果。应把「改变端点配对」和「改变每侧编辑数量」列作不同对照，并在列标题上直接标预算。[作者稿 PDF](https://arxiv.org/pdf/2409.13401#page=9) · [DOI](https://doi.org/10.1109/TGRS.2025.3529031)

10. **Di Wang 等，*Advancing Plain Vision Transformer Toward Remote Sensing Foundation Model*，2023。** PDF 第 6 页 Fig. 4 仅展示第 *i* 个注意力头里窗口如何改变；第 12 页 Fig. 9 再用实际图像对照不同方法生成的窗口。本文可先用带编号的小网格给出规则，再在真实船例上按相同符号展示选中的像元，保持示例和证据各自的身份。[作者稿 PDF](https://arxiv.org/pdf/2208.03987#page=6) · [DOI](https://doi.org/10.1109/TGRS.2022.3222818)

11. **Di Wang 等，*An Empirical Study of Remote Sensing Pretraining*，2023。** PDF 第 6 页 Tables II–IV 依次给出小规模筛查、训练轮次检验和进入后续微调的候选，第 9 页 Fig. 5 又分别画训练损失与测试准确率。本文可用同样的顺序陈述源域选择和已冻结的目标域验证，并指出哪个决定在哪个数据集上作出。[作者稿 PDF](https://arxiv.org/pdf/2204.02825#page=6) · [DOI](https://doi.org/10.1109/TGRS.2022.3176603)

12. **Ying 等，*DGMA²-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection*，2024。** PDF 第 10 页 Table II 明列参数量、FLOPs、在 LEVIR-CD 上训练一个 epoch 的时间以及该数据集的 FPS；第 11 页 Tables III、IV 分别考查两个模块。需要报告推理成本时，给出实际设备、每幅图的候选数、调用次数与计时范围，尤其留意小候选直接通过的分支。[作者稿 PDF](https://piurilabs.di.unimi.it/Papers/tgrs24.pdf#page=10) · [DOI](https://doi.org/10.1109/TGRS.2024.3390206)

13. **Wei Liu 等，*A Multitask CNN-Transformer Network for Semantic Change Detection From Bitemporal Remote Sensing Images*，2024。** PDF 第 8 页 Tables I–IV 按任务和数据集分表，第 10 页 Fig. 9 展开不同方法在 SECOND 和 HRSCD 上的类别混淆矩阵，图注解释缩写。本文的 AP 与固定匹配船体 IoU 同样需要各自的评价单位；逐船几何分析可以解释误差方向，AP 的变化仍须结合匹配阈值和分数排序。[正文 PDF](https://uwaterloo.ca/geospatial-intelligence/sites/default/files/uploads/documents/a_multitask_cnn-transformer_network_for_semantic_change_detection_from_bitemporal_remote_sensing_images.pdf#page=10) · [DOI](https://doi.org/10.1109/TGRS.2024.3486787)

## 跨刊对照：交代海域、采集和标注的含义

14. **Shin、Yang 与 Marino，*Small Fishing Vessel Detection Using Polarimetric Observables Fusion With Dual-Polarization Sentinel-1 SAR Data*，JSTARS，2025。** PDF 第 3 页 Fig. 1 左侧为 Sentinel-1 影像覆盖范围，右侧为 6 个研究区及 AIS、V-Pass 航迹，Table I 紧接着列具体场景和成像时刻。这是地理实验证据的好版式：先说明样本从哪里来，再说明哪些独立采集支持统计推断。缺少采集标识时，应直接交代可核分组的层级。[作者稿 PDF](https://www.storre.stir.ac.uk/retrieve/6360660d-be11-4a37-8a92-f0a7ac0fce0b/Small_Fishing_Vessel_Detection_Using_Polarimetric_Observables_Fusion_with_Dual-Polarization_Sentinel-1_SAR_Data.pdf#page=3) · [DOI](https://doi.org/10.1109/JSTARS.2025.3624445)

15. **徐从安等，*RSDD-SAR：SAR 舰船斜框检测数据集*，《雷达学报》，2022。** PDF 第 3 页 Table 2 列原始数据，图 1 对照标注方式，图 2 示意航向和长宽比；第 5 页图 6、7 分别解释切片及人工修正。采用其后续实例资源时，数据段宜分开介绍原始场景、切片及当前使用的轮廓标注，并给出相应来源。该文图 3 定义的斜框本身不能充当轮廓真值。[出版社 PDF](https://files.sciengine.com/pdfs/2095-283X/23A1C428928844EC809FB12C4F5A7A6B.pdf#page=3) · [DOI](https://doi.org/10.12000/JR22007)

16. **Tianwen Zhang 等，*SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis*，Remote Sensing，2021。** 出版社正文 Fig. 12 按 RBox 测量宽高、角度等量，Fig. 13 改用 PSeg 的面积、周长及船体占整图的面积比例。本文讨论编辑数量相对于小船的尺度时，应写清分母是原始掩膜、标注多边形还是整张影像，并以原始像素或平方像素标单位。[出版社正文及图 12–13](https://www.mdpi.com/2072-4292/13/18/3690) · [DOI](https://doi.org/10.3390/rs13183690)

17. **Xin Zhang 等，*RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark*，CVPR，2025。** PDF 第 5 页 Fig. 4 按类别展示方向、长宽比等属性，第 6 页 Table 2 对不同 SAR 数据集明确标出是否提供旋转框。对细长或近邻船体的结果，宜先报告形态分布及其分母，再解释分组表现；不同数据集标的是框还是轮廓，应在表中一眼可辨。[CVF 正文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.pdf#page=5) · [DOI](https://doi.org/10.1109/CVPR52734.2025.00695)

## 本稿最值得采用的三项设计

- **方法图用一只船解释一次决定。** 从 S²A-Net 的重复网格与 Ma 等的逐格数值图借鉴布局。用同一组候选格和分数演示两种写法，只在选中位置加记号；旁列两侧编辑数及净面积变化。图注定义坐标、箭头方向和格内数字，再说明哪些信息只供训练。
- **定性图留一个能检验解释的反例。** 参照 BIT 的误差矩阵，在同一裁窗中并排放原图、轮廓真值、原掩膜、改动位置和局部放大图。可列一例恢复船体、一例侵入码头或漏掉船侧，并注明实例选择规则。零值及微小差异不通过每张图单独拉伸颜色放大。
- **结果段围绕可检验的差别展开。** 先交代相同输入、相同候选和纳入统计的单位，再用一张表给出配对差值及其区间。随后用新增和删除各自落在轮廓内外的面积解释连续 IoU 变化。一个被轮廓穿过的像元可以同时贡献内、外两项。若 AP 与它方向不同，另查匹配阈值和候选排序。影像块、船体和采集场景的样本数分别给出，便于读者判断结论的适用范围。
