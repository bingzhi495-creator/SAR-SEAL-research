# 17 篇检测、遥感表征与稳健性论文的呈现方法

逐篇打开原始 PDF，核对下列编号对应的正文、图注或表注，并渲染、目视核对每篇至少一个代表页。页码按链接所示 PDF 从首页计数。取样关注论文如何说明对照、预算、抽样和测量。图注、数值和视觉素材均留在原文。

## 检测与公平对照

| 论文 | 核对位置与原文做法 | 在 SAR 稿件中的用法 |
|---|---|---|
| 01 Carion 等，DETR（ECCV 2020） | Fig. 2（p.7）将骨干、编码器、解码器及预测头排成一条主线。Eqs. (1)–(2)（p.5）先给匹配成本和最优分配，再给依赖该分配的训练损失。Table 1（p.9）分列原有 Faster R-CNN、配方增强的 Faster R-CNN 与 DETR。 | 把用于几何评估的原生固定匹配写成独立定义，然后引用它定义配对差。比较表标出复现配方、冻结载体和文献报告值。 [原文](https://arxiv.org/pdf/2005.12872) |
| 02 Zhu 等，Deformable DETR（ICLR 2021） | Fig. 2（p.5）在同一示意中画出参考位置、偏移和被采样的特征。Eqs. (2)–(3)（pp.5–6）从单层推广到多层，式后定义注意力头、特征层和采样点索引。Fig. 3 与 Table 1（p.8）把收敛曲线和不同训练轮数的实测放在同页。 | 画局部像素集合及索引时让图、正文与算式共用记号。短预算与长训结果按更新步数分开，曲线旁注明每个末端点的实际预算。 [原文](https://arxiv.org/pdf/2010.04159) |
| 03 Zhang 等，DINO（ICLR 2023） | Fig. 5（p.10）用同一视觉语法并排画三种查询初始化，Fig. 6 随后比较两种框更新路径。Tables 1–2（pp.12–13）按训练长度分表，Table 4（p.15）显式列出 QS、CDN、LFT 的组合及小目标 AP。 | 在配对、独立和有界对照图中保留相同原生候选坐标与符号，只改变实际选点约束。消融表写明组合及 AP 小目标分量，记录已有组合中方向相反的结果。 [原文](https://arxiv.org/pdf/2203.03605) |
| 04 Zhao 等，RT-DETR（CVPR 2024） | Fig. 4（p.5）给总结构，Fig. 5 给局部融合块，两图的模块名相互对应。Table 3（p.8）以 AP、参数、延迟对应 Fig. 3 候选编码器。Table 4 用查询质量比例解释候选选择，Fig. 1 的延迟轴写明 T4、TensorRT、FP16。 | 总图中标出局部写回部件，局部图沿用总图名字。测时表标出 GPU、精度、是否包含解码和传输，以及进入写回的候选数。效用与开销同时报告。 [原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf) |
| 05 Zong 等，Co-DETR（ICCV 2023） | Fig. 4（p.3）用独立分支明确只在训练时启用的辅助头。Tables 2–3（p.6）对普通与强基座分块。§4.4（p.8）先声明消融所用骨干和默认头数，再解释组件、头类型和数量变化。 | 图中只保留实际推理路径，训练专用部分用独立语义和线型。正文按固定候选和相同编辑次数的控制结果解释机制，载体改变后的外部结果另列。 [原文](https://openaccess.thecvf.com/content/ICCV2023/papers/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.pdf) |
| 06 Li 等，ViTDet（ECCV 2022） | Fig. 2（p.5）用相同尺度标记对单层简化金字塔和 FPN 式实现作并排对照。Tables 2–3（p.9）把结构改动与显存、运行时间并列。Fig. 3（p.12）以三个同系列小图呈现参数、FLOPs、实际测试时间与精度的关系，对照由作者实现。 | 先将实际可写像素和在固定候选内重新匹配两类差别画清。若绘制成本图，同一组点用相同颜色和标记，单独给有效写回次数与实测时间。 [原文](https://arxiv.org/pdf/2203.16527) |

## 表征、尺度与数学写法

| 论文 | 核对位置与原文做法 | 在 SAR 稿件中的用法 |
|---|---|---|
| 07 Liu 等，Swin Transformer（ICCV 2021） | Fig. 3（p.4）包含总网络和两种交替窗口块，明确指向 Eq. (3)。Eq. (4)（p.5）式后给 $Q,K,V$ 的大小与相对位置偏置。Table 4（p.8）用多任务列显示移除窗口移动和换位置编码后的变化。 | 统一图中 $M_0,S^+,S^-,k$ 在公式与正文的用法，紧邻首次定义写出空间域、维度和可行集合。机制结果在多个数据配方中以相同列结构报告。 [原文](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf) |
| 08 He 等，MAE（CVPR 2022） | Figs. 2–3（p.2）统一使用输入、预测、真值的次序，分别呈现同域与迁移案例。Table 1（p.5）表注先集中定义未改动的训练配方，分小表实施单因素比较。Fig. 7（p.6）曲线的每个点来自一套完整训练日程。 | 各案例固定列顺序、图例和裁剪范围。用表头或表注一次写明共同条件。消融包含短预算时明确其独立运行条件，完整训练以独立身份列出。 [原文](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) |
| 09 Oquab 等，DINOv2（TMLR 2024） | Fig. 3 和 §3（p.4）把图像嵌入、去重和检索逐级画出来。Table 15（p.30）列出数据集、使用的划分及纳入方式；Table 6（p.13）清楚区分冻结特征上的线性探针与域泛化指标。 | 给一张数据资格图供审稿人追查原采集到最终测试窗的筛选。把公开标签来源、去重层级、是否重新拟合和各数据子集的纳入方式放在同一视图。 [原文](https://arxiv.org/pdf/2304.07193) |
| 10 Cong 等，SatMAE（NeurIPS 2022） | Fig. 2 与 Eq. (2)（p.4）对应时间嵌入时年、月、小时的编码，Fig. 3（p.5）在相同时间序列上把一致与独立掩蔽并排画。Tables 4–5（p.7）分别列时间和光谱配置。 | 论文涉及传感器、日期或极化时分别定义字段，并按实际对照显示输入。保留像素结构一致的并排示意来解释如何控制同一组候选。 [原文](https://proceedings.neurips.cc/paper_files/paper/2022/file/01c561df365429f33fcd7a7faa44c985-Paper-Conference.pdf) |
| 11 Reed 等，Scale-MAE（ICCV 2023） | Eq. (2)（p.3）给标准位置编码，Eq. (4) 和 Fig. 3（p.4）让新增的 GSD 比率与同像素大小、不同地面范围的示意对应。Fig. 5（p.6）用共同坐标比较尺度，Table 6（p.8）检查相对 GSD 变化。 | 在论文中把原生 512 像素画布、候选掩膜面积像素数和米制分辨率写清。只有当来源给出实际地面采样距离时才报告物理尺度结果。 [原文](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf) |

## 数据资格、统计与写作

| 论文 | 核对位置与原文做法 | 在 SAR 稿件中的用法 |
|---|---|---|
| 12 Koh 等，WILDS（ICML 2021） | Fig. 1（p.1）通过样本三元组定义域，分别画出域泛化和子群分布变化；Fig. 2（p.2）用统一尺度的谱系图并列任务、域和数据规模。Table 1（p.7）表注明确每项 ID/OOD 的定义。 | 将实例、图块、父景、传感器日期组画成不同统计层级。标注 74 与 42 为嵌套，表注写清每个域或划分的实际定义、分母和限制。 [原文](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf) |
| 13 Recht 等，ImageNet-v2（ICML 2019） | Fig. 1（p.2）以原测试集为横轴、新测试为纵轴，同一模型对应一点，并画恒等线和 Clopper–Pearson 95% 区间。§4、Table 2（p.7）把标注频率抽样策略及结果并排。 | 当比较同法跨来源重复时，以原指标和新指标成对展示，并交代筛选规则的影响。复用“恒等线表示无变化”的视觉惯例，区间继续采用本研究的采集组配对重抽样。 [原文](https://proceedings.mlr.press/v97/recht19a/recht19a.pdf) |
| 14 Hendrycks 等，Many Faces of Robustness（ICCV 2021） | Fig. 1（p.2）用影像样例说明不同类型的分布偏移；Tables 1–3（pp.6–7）保留各任务具体指标；Table 4（p.8）用符号汇总跨数据证据，并在正文说明某种改动的效用依赖偏移类型。 | 用可观察现象命名定性分组，保留收益和退化案例。用一张效应方向总表提示适用范围，AP、连续 IoU 和置信区间原始数值继续可见。 [原文](https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf) |
| 15 Ovadia 等，Predictive Uncertainty under Dataset Shift（NeurIPS 2019） | 评价指标段（p.4）逐项定义 Brier 与校准指标。Fig. 1（p.5）分列准确率、Brier 与置信度相关结果，并在图注注明阴影为 10 次运行的标准误。Figs. 2–3（pp.6–7）进一步分拆扰动和样本覆盖变化。 | 把全候选 AP 点估计与固定匹配的采集组 IoU 区间分别标注。若要称某个字段为可靠的概率，给出独立的校准证据及其分母。 [原文](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) |
| 16 Gadre 等，DataComp（NeurIPS 2023 Datasets and Benchmarks） | Fig. 1（p.3）把可改变的数据选取和固定训练评价顺序放在同一流程。Table 2（p.4）列骨干、MACs、数据池和看过的样本。附录 Figs. 4–5（p.28）检查跨评价集近重复，包括变形、裁剪与压缩。 | 先锁定原始候选和写回接口，再变动策略。在数据资格附表分别说明精确 RGB 去重、同父景同切片位置及跨极化或近重复核验。 [原文](https://arxiv.org/pdf/2304.14108) |
| 17 Ravuri 等，Radar Nowcasting（Nature 2021） | Fig. 1（p.3）并排画简化系统与一个降水雷达案例，行列固定时间和方法。Fig. 3（p.5）按多个空间聚合尺度展示两类 CRPS，扩展数据 Figs. 5–8（pp.15–18）保留其他数据划分与区域结果。 | 每个 SAR 案例保留同一灰度视窗、真值边界与方法列。主文展示固定的代表性选例，补充材料展示其他场景、误差类别与强度。 [原文](https://www.nature.com/articles/s41586-021-03854-z.pdf) |

## 可直接用于稿件的规则

1. **方法图与公式成对落地。** 一张总图定位冻结检测器、原生候选和掩膜写回。局部图拆出边界支持、端点分数和写回条件。先定义原生前景 $M_0$、合法增删支持 $S^+,S^-$、选中端点 $A,D$ 与实际预算 $k$。例如，以 $A\subseteq S^+\subseteq M_0^c$、$D\subseteq S^-\subseteq M_0$ 和 $\lvert A\rvert=\lvert D\rvert=k$ 说明面积守恒，再给 $M'=(M_0\setminus D)\cup A$。同一图的坐标、颜色、变量与公式保持对应。
2. **公平性写入表头。** 每组对照列出数据划分、载体、分数来源、是否重训、训练长度和实际增删量。主结果表优先全候选 AP、AP75、AP90、AR100。几何表给固定原生匹配数、子集增删覆盖、连续 IoU 和按真实轮廓面积归一的 FP/FN。在区间图上标记采集组单位和实用阈值。缺失配方或外部引用值在表注中准确注明。
3. **让异常结果有解释入口。** 定性图统一底图、裁剪、真值、方法列和尺度。同一组按预先说明的数值规则选正负案例，例如按组内中位改善选例。正文按观察、机制假设、对应消融的顺序写，保留结果不一致和覆盖不足的情形。把已观察到的性能差、可辨识的作用和仍需验证的解释分开写。
4. **提前交代统计单位。** 明确几何变化来自同一原生关联的配对估计。说明每个父景权重和按传感器日期或采集组重抽样的方法。嵌套敏感性分析用同一尺度展示，正文交代与主样本的共享关系。
