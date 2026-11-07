# 形状、拓扑、最优传输与等变性的一手补缺

2026-11-07。范围限六项代表方法，核对原论文、作者关联与真实实现。下文适用性和否定条件属于本项目的研究判断，尚无项目运行、误差统计或收益证据。所有源码通过网络作静态阅读，本文仅保留自有摘要和链接。

六项均可提供工具，但直接给基础模型添加已知几何损失、LoRA、卷积适配器、Sinkhorn 或等变卷积，技术差异仍不足以单独支持核心创新。对公共单通道显示 SAR，当前最值得辨别的是小船掩膜的轮廓遗漏、实例内断裂、邻船实例混淆，以及低分辨率表示造成的细节丢失。它们需要不同的监督和对照。

## 六项原始工作与实际源码

### 1. clDice，CVPR 2021

Suprosanna Shit、Johannes C. Paetzold 等，*clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*。通过预测和 GT 的骨架与对方区域之间的覆盖率，构成拓扑精确率和敏感度，再作调和平均。它面向管状结构。该代理损失的理论条件与连续体、数字图像及软骨架近似之间的关系须保留，不能由一次有限训练推出任意实例掩膜必然保持拓扑。 [原论文](https://arxiv.org/abs/2003.07311)

作者仓库固定提交 47d31a6cc4a8101b1ffe8052994821961e57af9f。已实际读取 cldice_loss/pytorch/cldice.py 的 soft_cldice.forward、soft_dice_cldice.forward，以及 soft_skeleton.py 的 SoftSkeletonize.soft_skel。前者的 iter_ 被保存，但实际构造固定 SoftSkeletonize(num_iter=10)。复现必须记录这一源码行为。exclude_background=True 会删除第一个通道，单前景 sigmoid 掩膜直接这样使用会删掉唯一通道。默认 torch.sum 跨批次聚合，还会使实例大小和骨架长度影响贡献。许可证为 MIT。 [损失真实源码](https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/cldice.py#L6-L61) · [骨架真实源码](https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/soft_skeleton.py#L5-L50)

对细长船可以作为实例内部断裂的辅助监督。它无法单独判定两个预测实例是否重复覆盖或合并，也不会准确约束船宽。先比较原生 BCE/Dice、同实例加权方式的普通 Dice，以及相同训练预算的 clDice。若骨架分数改善同时 mask AP、轮廓质量或邻船分离恶化，应否定其对目标任务的效用。宽船和有歧义的断续标注也需要单独检查。

### 2. Topograph，ICLR 2025

Laurin Lux、Alexander H. Berger、Alexander Weers、Nico Stucki 等，*Topograph: An efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation*。它联合预测和 GT 构建四类配对图像，以连通分量形成图，在图中识别拓扑关键错误区域，再聚合对应概率或交叉熵。其图论结论与定义的数字拓扑、背景边界和损失为零等条件相关，有限训练时仍需实际验证。原实验包括道路及医学图像，属于语义分割证据。 [正式会议页面](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html) · [作者方法第 2 节](https://arxiv.org/html/2411.03228)

作者给出的仓库固定提交 77ca4c4fd5d13e8b94e8337c82552baf07808b2e。已读取 losses/topograph.py 的 label_regions、create_graph、get_critical_nodes、TopographLoss.forward。它接受 [batch,classes,H,W]，按通道 argmax 后逐类处理，可选 4/8 连通和附加背景边框。源码导入编译模块 Topograph、SciPy 和 NetworkX，且可建立多进程池。单前景 logit 直接按单通道 argmax 会得到常量标签，需要按方法契约构造前景/背景表示。许可证为 MIT。 [关键图函数](https://github.com/AlexanderHBerger/Topograph/blob/77ca4c4fd5d13e8b94e8337c82552baf07808b2e/losses/topograph.py#L37-L269) · [真实前向](https://github.com/AlexanderHBerger/Topograph/blob/77ca4c4fd5d13e8b94e8337c82552baf07808b2e/losses/topograph.py#L429-L601)

公共船舶实例可按匹配的单实例二值掩膜考察此损失。将所有船合成语义前景会丢失实例身份，接触的两艘船也可能在语义连通度上没有错误。各实例独立连通又不保证跨实例互斥。简单对照至少包括边界监督、普通连通分量后处理，以及相同预算的 clDice。若收益依赖特定 4/8 连通选择或修补公开 GT，缺乏原始官方评价下的收益，应降低主张。

### 3. Boundary Loss，MIDL 2019

Hoel Kervadec、Jihene Bouchtiba、Christian Desrosiers、Eric Granger、Jose Dolz、Ismail Ben Ayed，*Boundary loss for highly unbalanced segmentation*。该工作将边界距离目标转为预测概率和 GT 符号距离图的区域积分，缓解小前景的区域监督不平衡。它为几何监督提供直接基线，与给船舶额外加边缘损失高度相关。 [正式原论文](https://proceedings.mlr.press/v102/kervadec19a.html)

作者仓库固定提交 171c32d88a4ce59af8be46fb88b96d3637b9515b。losses.py 第 98–118 行的 SurfaceLoss.__call__ 即 BoundaryLoss，对概率和距离图逐像素乘积求均值。utils.py 第 294–310 行的 one_hot2dist 计算符号距离，支持显式像素间距。损失要求通道概率为 simplex。独立 sigmoid 掩膜需要显式前景/背景概率契约。此版本距离转换中仍有 np.bool，未来复现需先验证与所选 NumPy 的兼容性。许可证为 MIT。 [损失实现](https://github.com/LIVIAETS/boundary-loss/blob/171c32d88a4ce59af8be46fb88b96d3637b9515b/losses.py#L98-L118) · [距离图实现](https://github.com/LIVIAETS/boundary-loss/blob/171c32d88a4ce59af8be46fb88b96d3637b9515b/utils.py#L294-L310)

该方法直接适配显示域掩膜，不需要幅度定标或相位。公共显示图若没有可信地面像素间距，应按图像像素单位定义距离，并检查尺度归一化。对照包括原生 BCE/Dice、普通边界带重加权和相同额外计算的密集掩膜监督。RF-DETR 的点监督与密集距离图监督计算预算不同，收益可能来自监督覆盖变化。若效果仅由增加密集像素或轮廓带权重解释，应将其作为配方收益。

### 4. OTA，CVPR 2021

Zheng Ge、Songtao Liu、Zeming Li、Osamu Yoshie、Jian Sun，*OTA: Optimal Transport Assignment for Object Detection*。将 GT 及背景视为供应方，锚点视为需求方，以分类和框回归损失定义成本，用 Sinkhorn 求熵正则传输方案，并用动态正样本配额作标签分配。原任务和指标为目标检测。 [原论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_OTA_Optimal_Transport_Assignment_for_Object_Detection_CVPR_2021_paper.pdf)

作者仓库固定提交 2c85b4d0f9031396854aae969330dde2ab5eacbd。已读取 ResNet-50 配方 fcos.py 的 FCOS.get_ground_truth 和 SinkhornDistance.forward。前者含背景供应、IoU 动态配额和空间约束。后者在 log 域迭代，输出的传输方案有 .detach()，之后方案按 GT 行最大值重新缩放并作离散分配。它依赖旧版 cvpods 检测框架，复现所需工作超过替换一个损失函数。许可证为 Apache-2.0。 [真实分配](https://github.com/Megvii-BaseDetection/OTA/blob/2c85b4d0f9031396854aae969330dde2ab5eacbd/playground/detection/coco/ota.res50.fpn.coco.800size.1x/fcos.py#L233-L345) · [真实 Sinkhorn](https://github.com/Megvii-BaseDetection/OTA/blob/2c85b4d0f9031396854aae969330dde2ab5eacbd/playground/detection/coco/ota.res50.fpn.coco.800size.1x/fcos.py#L665-L713)

若公共 SAR 的失败集中在相邻 GT 与候选预测的分配冲突，OT 可作为分配对照。RF-DETR 的原生一对一匹配和 OTA 的动态多正样本语义有明显差异，改成后者会同时改变重复预测、正样本数和推理消重。至少固定同一成本，比较原生 Hungarian、同正样本数的简单多分配，以及 OT，并单列正样本和查询预算。若仅因增加正样本或改变成本取得收益，无法归因于运输算法。像素空间 OT 若将多船合并成一个分布，也会丢失实例身份。

### 5. escnn，ICLR 2022

Gabriele Cesa、Leon Lang、Maurice Weiler，*A Program to Build E(N)-Equivariant Steerable CNNs*。使用满足群作用约束的核基来构造等变网络。作者原页面确认 ICLR 2022 及其代码关联。它提供构造数学保证所需的表示类型约束。 [作者论文页](https://maurice-weiler.gitlab.io/publication/2022_01_escnn/) · [原论文 PDF](https://openreview.net/pdf?id=0qf71T2bZGm)

作者仓库固定提交 9ad44cc37d694d9c805c4fef4d16722a254b3bf2。已读取 escnn/nn/modules/conv/r2convolution.py 的 R2Conv.forward、check_equivariance，并核对 FieldType.transform 和 gspaces.rot2dOnR2。这些代码操作带明确群表示的特征。源码包含离散采样与插值误差的检查。许可证文件已取得，未来使用须沿用其再分发条件。 [真实卷积](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/escnn/nn/modules/conv/r2convolution.py#L23-L349) · [特征变换](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/escnn/nn/field_type.py#L242) · [二维旋转群](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/escnn/gspaces/r2.py#L144) · [许可证](https://github.com/QUVA-Lab/escnn/blob/9ad44cc37d694d9c805c4fef4d16722a254b3bf2/LICENSE)

单通道灰度可以采用标量场输入，输出掩膜也应按空间坐标变换。普通 DINO 骨架前后接一个等变模块，通常不足以让整个实例系统等变，位置编码、降采样、匹配及后处理均需满足相应条件。优先用方形图像的 90° 旋转组诊断，避免任意角插值的混淆，所有组使用相同旋转增广。对照包括同参数或计算量的普通卷积适配器、旋转一致性训练，以及单列计算成本的旋转 TTA。图像旋转只是给定成像结果的坐标变换，不能证明跨雷达视角、极化或散射改变的稳健性。

### 6. TopoLoRA-SAM，2026 预印本

Salim Khazem，*TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation*。SAM 编码器加入 LoRA，图像嵌入接空间卷积适配器，并可加 clDice。包含视网膜、息肉和 SL-SSDD 海陆语义分割。它已直接覆盖“基础分割模型 + 轻量适配 + 拓扑损失”的组合。arXiv 记录提交日为 2026-01-05，HTML 正文另写 2026-08-24，此处保留两种日期来源，仅按预印本列出。 [原稿](https://arxiv.org/abs/2601.02273) · [方法和实验](https://arxiv.org/html/2601.02273)

作者原稿链接 salimkhazem/Seglab，固定提交 7de3a5fbad992363e690190eb82dffbe8a3c1a0b。已取得实际 SAMPEFTNet.forward、ConvAdapter、inject_lora、cldice_loss 及默认配置。实现为冻结编码器中的 FFN 线性层 LoRA、嵌入上的残差深度可分离卷积适配器和二值解码。默认配置 rank 为 8，原稿中另讨论 rank 16，未来复现必须明确版本与配方。SLSSDDDataset 读入 RGB 图及灰度二值标签。许可证为 MIT。 [真实模型](https://github.com/salimkhazem/Seglab/blob/7de3a5fbad992363e690190eb82dffbe8a3c1a0b/seglab/models/sam_peft/model.py#L18-L79) · [真实拓扑损失](https://github.com/salimkhazem/Seglab/blob/7de3a5fbad992363e690190eb82dffbe8a3c1a0b/seglab/models/sam_peft/topo_losses.py#L22-L55) · [真实默认配方](https://github.com/salimkhazem/Seglab/blob/7de3a5fbad992363e690190eb82dffbe8a3c1a0b/configs/models/sam_topolora.yaml) · [数据接口](https://github.com/salimkhazem/Seglab/blob/7de3a5fbad992363e690190eb82dffbe8a3c1a0b/seglab/data/sl_ssdd.py)

其 Dice 和二值语义任务不能作船舶 COCO mask AP 对比。该先例给出的最小消融结构仍应保留，包括解码器训练、仅 LoRA、仅同容量适配器、组合后加普通边界监督，以及组合后加拓扑监督。若项目只是复用这些组件并换数据集，应据实列为迁移或强基线。

## 公共船舶实例掩膜的决定性边界

| 候选误差 | 最小可否证诊断 | 首要简单对照与停止条件 |
|---|---|---|
| 细长船局部遗漏或断裂 | 固定匹配后的单实例误差，按 GT 面积、长宽比及表示分辨率分层，检查断裂是否已由栅格化产生 | 普通边界带权重、同实例加权 Dice、增加等量监督点。若密集监督或采样已解释收益，拓扑机制缺乏增量证据 |
| 相邻船混淆或重复覆盖 | 保留 GT 实例身份，分别统计一对多、一对零、重复预测及匹配变化，固定置信度和匹配规则 | 原生匹配、同成本和同正样本预算的简单分配。仅改善语义连通度不能支持实例分离结论 |
| 方向敏感 | 同图及 GT 作 90° 坐标旋转，预测逆变换后按实例集合匹配，保留准确度及稳定性两项结果 | 所有组同旋转增广、普通适配器、计算量匹配 TTA。若误差来自插值或额外预训练差异，应重新定位机制 |
| 公开 GT 的边界或连接歧义 | 在官方原始标签不变的前提下，检查 4/8 连通、边界容忍带和训练分辨率对诊断结论的敏感性 | 固定官方 mask AP 作主评估。若排序只在特定容忍带、连通规则或后处理标签下成立，应收缩结论 |

几何损失只能利用给定图像与标注。公开显示 SAR 的单个亮点、断续回波和人工轮廓之间未必有唯一物理船体对应，当前没有证据允许强制每个标签无孔或连续。按前景分布归一化的 OT 也可能掩盖面积错误，按实例连通的拓扑损失可能忽略跨实例关系。所有这些解释都需在相同骨架、初始化、训练长度、验证选择规则、推理成本及重复种子下检验，不能从损失名称推出作用。

后续最小入口是先在强基线的固定表示层级诊断可复现的轮廓、断裂或匹配错误。若主要错误来自掩膜空间分辨率和监督覆盖，优先检验同预算的表示或监督改动。只有简单几何基线后仍存在明确剩余错误，才值得构造更具体的拓扑或分配机制。本补缺没有认定以上任何路线达到投稿级创新。
