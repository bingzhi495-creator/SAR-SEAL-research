# GLDet 原论文核查与实例分割迁移判断

核查日期：2026-10-14。状态：**已阅读用户提供的20页原PDF并视觉核对方法图与主要表格；本项目迁移仍为待检验设计，未执行GLDet或新增模块训练**。只使用外部PDF、公开网页和ROOT提供的当前接口信息，没有读取项目私有SOURCE/RAW/SPEC/Q、实际数据或权重缓存。本文保存引用、复述和独立分析，不上传论文PDF、截图或全文抽取文本。ROOT正在推进的context-memory单模块首试保持原方案。

## 1. 来源、发表状态和代码可用性

Hao Chang, Xiongjun Fu, Ping Lang, Kunyi Guo, Jian Dong, Shibo Chang. **GLDet: Real-Time SAR Ship Detector Based on Global Semantic Information Enhancement and Local Gradient Information Mining**. *IEEE Transactions on Geoscience and Remote Sensing*, 63, article 5209020, pp.1-20, 2025. [IEEE正式记录](https://ieeexplore.ieee.org/document/10962196/)，[DOI:10.1109/TGRS.2025.3559551](https://doi.org/10.1109/TGRS.2025.3559551)。原PDF首页记载2025-04-07接受、2025-04-10发表、2025-04-21当前版本；[Crossref出版社登记元数据](https://api.crossref.org/works/10.1109/TGRS.2025.3559551)匹配题名、作者、卷号与IEEE文档10962196。

作者完整公开实现**尚未核验到**。原PDF没有作者代码链接。公开GitHub仓库名检索找到 [changhao123/GLDet](https://github.com/changhao123/GLDet)，仅含标题为“AIRSAR”的README；[changhao123/GLDet1](https://github.com/changhao123/GLDet1)为空仓。仅凭用户名与仓库名相近，不能确认其作者身份，更不能称已取得官方实现。当前没有可核验的GLDet源码、完整配置、训练日志或权重；检索未找到与本题名匹配的arXiv记录，不能据此断言不存在其他未索引版本。

下文方法和结果的一级证据均是用户提供的正式PDF，按页码、图号、表号定位；网页用于来源核查。Mamba算子性质另用其[原作者论文](https://arxiv.org/abs/2312.00752)核对。没有使用宣传文章或二手论文的数字替代GLDet原文。

## 2. 它确实包含实例分割，而非只有bbox检测

GLDet是完整的一阶段SAR网络：**Stem/多级GLD backbone → PAFPN → 检测/动态kernel mask/旋转框头**。它不是在固定教师的单个采样点上直接加一个注意力模块。

原文第9页、图4的实例分割分支包括kernel head和mask feature head：kernel head生成每个实例的三层动态卷积参数，作用于多尺度mask特征及两个相对坐标通道；mask用Dice监督，同时训练检测分支。旋转框任务另改角度分支、box coder及Rotated IoU损失。因而论文的mask AP是整网、检测匹配、分数及mask头共同形成的指标，不能归因于一个局部梯度模块，也不能直接迁入本项目固定query/score/support的mask残差评价。

这种整合可作为方法贡献的直接先例。已知Mamba、RepVGG、channel attention与动态kernel的组合并非因此自动无效；应比较具体结合方式、训练和效果。本项目仍需说明对这一先例的差异，不能将重新命名或复用模块当作已经成立的创新。

## 3. Global Information Augmentation Encoder 的结构与公式

依据第6-8页、Algorithm 1、图5：输入为`X ∈ R^(B×H×W×C)`，先将**有序完整栅格**展平为`B×(HW)×C`，LayerNorm后经Mamba并加残差：

\[
X_g=\mathrm{Mamba}(\mathrm{LN}(\mathrm{flatten}_{HW}(X)))+\mathrm{flatten}_{HW}(X).
\]

图5的Mamba有一条Linear+SiLU门控支路，另一条Linear→Conv1D→SiLU→Selective SSM支路；两者相乘后Linear投影。Algorithm 1用状态更新/读出表达SSM：

\[
\dot h(t)=\bar A h(t)+\bar B x(t),\qquad y(t)=\bar C h(t)+\bar D x(t).
\]

这里表达的是沿序列传播上下文。离散实现需要对应步长、input-dependent选择参数及扫描方向，不能把无序采样点的排列当作自然图像顺序。

有两处实现依据要保留为未确认：原文式(1)展开的`h1 = A B h0 + B x1`与通常状态递推不一致，不能按该式逐字实现；式(3)将Selective SSM写成固定卷积核的解释也需要谨慎。Mamba原作者明确指出input-dependent选择使高效固定卷积形式不再适用，而采用递归/并行scan。参见[Mamba原论文](https://arxiv.org/abs/2312.00752)。这限定了文字公式的实施可靠性，不否定GLDet表格中的整网结果。

backbone渠道按表I为：Stem 3→12→12→12→24，Stage1-4为48/96/192/384；含stride2下采样和各级GLD块。局部模块处理的是这些空间特征，并非256维采样点向量的任意重排。

## 4. Local Gradient Information Miner 的确切内容

LGIM分为 **GIM→DSA**（第8-9页、图5-6）。

### GIM

图6给出的算子是滴状/多分支RepVGGBlock级联，每个RepVGGBlock由`3×3卷积 + 1×1卷积 + identity`组成，并融合多级支路输出。它以空间卷积产生不同的局部feature maps，后交给DSA。

原文把这些结果称为gradient feature maps，文字解释了特征对loss变化的贡献；但Algorithm 1、图5-6与方法段没有给出显式Sobel、Laplacian、`∇I`、`∇F`或在线`∂loss/∂X`的前向计算步骤，没有一个可照抄的梯度算子公式。**不能把标题中的gradient直接理解成“对SAR原图求梯度后送入点头”，也不能自动设计一个需要GT loss梯度的推理支路再称其为原GIM。** 作者源码尚缺，具体卷积groups、支路融合及是否采用RepVGG推理重参数化还未核验。

### Directed Self-Attention（DSA）

第8页正文描述三个`3×3 depthwise conv`支路。两支路分别沿H/W方向平均池化，重排为分组channel表达，做channel cross-covariance/attention；第三支路为被加权的空间特征。原文称head数为16，attention矩阵每头为`(C/Head)×(C/Head)`，避免构造`HW×HW`像素attention。

式(4)可概括为：`X1,X2,X3=DConv3×3(X)`，`X_hat=A_H A_W X3`。为说明维度，下面是**按文字补全转置后的解释式，不冒充作者已经发布的实施公式**：

\[
Q_W=\mathrm{reshape}(\mathrm{Avg}_W X_1),\quad K_W=\mathrm{reshape}(\mathrm{Avg}_W X_2),\quad A_W=\mathrm{softmax}(Q_W K_W^T),
\]
\[
Q_H=\mathrm{reshape}(\mathrm{Avg}_H X_1),\quad K_H=\mathrm{reshape}(\mathrm{Avg}_H X_2),\quad A_H=\mathrm{softmax}(Q_H K_H^T).
\]

W方向pool后保留H，H方向pool后保留W；channel matrix乘`X3`的分组空间表达。原式(4)没有清晰标出transpose、缩放、归一化、softmax轴等实施细节，图5与文字维度也需由真实源码进一步消歧。这里是**方向pool形成的channel关联**，不是UCMNet的逐行/逐列空间cross-attention，两者不能混称。

### Decoder和neck

式(5)是LayerNorm、Linear→DWConv→GeLU→Linear与残差组成的FFN；DWConv继续需要空间邻域。式(6)表示C3/C4/C5间的top-down upsample与bottom-up stride conv融合；图4使用Concat标注，而式(6)用加法，这也是精确复现需要源码确认的一处。

## 5. 实验配方和主要mask结果

原文第10页：CUDA12.1、PyTorch2.1.2、OpenMMLab，单3090（原文写GTX 3090），AdamW，初始lr=0.004、batch16、score threshold=0.01；声明所有消融和比较训练500epochs。检测/实例分割输入800×800；旋转框任务512×512。使用COCO AP50:95/50/75与size分组评价。

未给出可核验的train/val/test文件清单或明确划分比例，也未完整报告初始化/预训练、学习率调度、weight decay、数据增强、seed和多次重复统计。论文声明同条件比较，应保留这一正面信息；在缺配置和日志时，其完全同划分、同初始化与所有外部专用方法重训仍为未确认事项，不能擅自把外部成绩归因为数据优势或包装。

第10页数据数量存在正文/表II不一致：SSDD ships为2456/2587，HRSID instances为16951/16969；这提示版本清单需要核验，不能自行挑一数值当其实际实验划分。第10页正文把APMask/APBbox对应任务反写，下列数值按清楚的表头和mask定义读取。

### SSDD消融，表III（第12页，已视觉核对）

| Mamba | GIM | DSA | mask AP | mask AP75 | bbox AP | FLOPs G | Params M | FPS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 无 | 无 | 无 | 65.4 | 80.6 | 66.6 | 20.5 | 22.8 | 267.1 |
| 有 | 无 | 无 | 67.5 | 82.9 | 67.8 | 21.6 | 22.6 | 255.3 |
| 有 | 有 | 无 | 69.1 | 85.8 | 71.7 | 38.1 | 23.4 | 250.8 |
| 有 | 无 | 有 | 67.9 | 83.9 | 69.2 | 22.6 | 19.8 | 250.1 |
| 无 | 有 | 有 | 69.7 | 86.7 | 72.7 | 43.7 | 22.6 | 248.7 |
| 有 | 有 | 有 | 71.3 | 89.1 | 75.7 | 45.1 | 22.6 | 246.6 |

在该论文配方下，Mamba相对全去除配置+2.1 mask AP；在Mamba上加GIM为+1.6；再加DSA为+2.2，而只在Mamba上加DSA为+0.4。结果支持局部提取与channel关联的组合价值候选，不能推出DSA单独必有收益。该表不是完整2³因子设计，缺GIM-only与DSA-only；配置参数量/计算量并不完全相同，完整45.1G相对20.5G约2.20倍。增益不能未经对照完全归因于gradient机制，仍要区分容量、卷积局部性及融合设计。

### 关键基线，表IV/VI（第12/15页）

| 数据/模型 | mask AP | mask AP75 | mask AP-small | Params M | FPS |
|---|---:|---:|---:|---:|---:|
| SSDD GLDet | 71.3 | 89.1 | 68.7 | 22.6 | 246.6 |
| SSDD RTMDet-tiny | 65.7 | 83.0 | 64.4 | 6.9 | 254.3 |
| SSDD HTC | 59.5 | 73.2 | 58.9 | 97.1 | 11.7 |
| SSDD DCTC | 67.4 | 83.8 | 64.4 | 51.3 | 34.0 |
| SSDD RSPrompter-query | 67.3 | 84.3 | 未报 | 未报 | 未报 |
| HRSID GLDet | 58.6 | 66.0 | 55.9 | 22.6 | 99.7 |
| HRSID RTMDet-tiny | 50.2 | 55.8 | 48.4 | 6.9 | 107.4 |
| HRSID HTC | 55.3 | 66.7 | 55.1 | 97.1 | 7.4 |
| HRSID DCTC | 55.1 | 66.2 | 54.8 | 51.3 | 20.0 |
| HRSID full-level context SE ROI Detector[43] | 57.9 | 69.5 | 57.3 | 107.6 | 5.3 |

这是可信、值得参考的整网精度/效率组合，但不是所有指标最好：HRSID GLDet比Detector[43]的总体mask AP高0.7，AP75低3.5，小目标AP低1.4；也不是比RTMDet-tiny更快，代价是更多参数换精度。原文介绍了Mask2Former，但结果表没有其可比训练行。

GLDet完整模型22.6M/45.1G不能与本项目新增点头约4万参数直接作“更轻量”比较；本项目还包含冻结教师及其前向成本。FPS没有充分说明预处理/NMS/I/O/计时batch/warmup，且同800输入下SSDD与HRSID值不同，故不得移用为本项目硬件下的推理速度承诺。当前本项目68.x的旧DEV指标与GLDet不同划分、不同网络、不同评估载体；没有统一规则不能宣称胜负。

近岸/离岸结果见表VII；GF-3与Terra-SAR大图泛化主要是切片拼接及可视化（第15-17页），没有独立标注大图的量化误差/重复统计。可作为定性迁移线索，不能升级成充分跨域SOTA证明。讨论中的SARATR-X融合是作者提出的进一步方向，不能当已经完成的融合收益。

## 6. 对当前mask细化的直接可用性

当前接口由ROOT提供：每个训练样本有K=1024个F256/native L/e/target/regionflag。旧PR是F非仿射LN、native c在全部四个linear处注入。ROOT提供的region_logits摘录将patch49构造为全零；保留patch ABI不等于已有真实空间patch输入。

| GLDet内容 | 是否能原样进入当前点头 | 缺什么 |
|---|---|---|
| GIAE / Mamba图像scan | 否 | 有序完整HW栅格、扫描方向、空间索引；对无序K点做scan会引入任意排序关系 |
| GIM 3×3/1×1/identity多分支 | 否 | 同一点周围真实图像或feature邻域、边界padding/坐标变换规则 |
| DSA H/W pooling channel covariance | 否 | 行列结构及覆盖范围；仅把256维向量重排为伪网格不具有原意义 |
| FFN / channel projection的一般思想 | 可写逐点适配，但不是原GLD | 它本身不增加局部或全局观测信息，收益仍要同容量/预算对照 |
| dynamic kernel mask+relative coords | 不是本轮残差head插件 | 实例kernel生成、多尺度mask feature和实际相对坐标，会改变原预测头和评价载体 |

检测任务并非迁移障碍的唯一理由：论文自身已有实例分割实验，所以局部空间信息用于mask细化是合理假设。真正障碍是当前输入缺邻域和有序图像结构，以及完整backbone级效果不能自动转成采样点head效果。也不能把当前context-memory解释为已经实现GLDet的global/local机制。

## 7. 可执行的后续建议（设计，当前不启动）

先完成ROOT正在开展的单一context-memory配对试训与锁定DEV AP评价；本综述不改变它的共同父权重、old LOG支持、固定cache、8→24新阶段、optimizer或分数规则。

若下一轮选择从GLDet借鉴一个模块，建议先选**局部多分支卷积的信息是否有用**，而不是一次导入Mamba+GIM+DSA+新head。它比当前F+c路由更直接提供缺失的空间观测，但必须先形成新独立空间cache并核验：每点原坐标/变换、真实3×3或7×7邻域、padding/有效区域mask、强度或feature归一化、采样和regionflag。只用推理时可得到的图像/冻结feature；GT仅供原训练监督，不进入邻域输入。

首先只选一种输入：冻结F邻域适配，或原SAR intensity patch适配。两者不是同一方法。冻结F邻域更接近GLDet的feature GIM，但存储/计算更大；intensity patch较廉价，却是另外的原图局部支路，不能称原GIM复现。当前全零patch需替换为实际观测会产生新cache/source资格，不能在旧绑定cache中补列或改写。

在有资格的新载体上，可把小型3×3/1×1/identity局部分支仅用来生成零初始化的center-point residual，旧query/score/LOG支持/K及教师保持固定。对照至少为共同父模型继续训练与单局部模块，后续需要同参数plain卷积/MLP和邻域置乱或中心复制对照，以区分真实空间信息、额外容量及额外训练。具体层数/宽度/预算先由真实cache与错误分析确定，本稿没有实现或运行它。

若未来要重新训练完整GLDet或换教师backbone，则是另一研发目标：需确认作者实现、mask head、split、预训练、500轮配方及统一Mask2Former/RTMDet等强基线，而不是在当前点头试训中顺带替换。单模块已有收益后再融合的用户要求适用于这些方向；不能据论文71.3或局部试验增益直接宣称本项目达到TGRS投稿水平。

## 8. 证据等级与发布边界

- **PRIMARY_PDF_READ / TABLES_VISUALLY_CHECKED**：方法结构、表III/IV/VI主要mask数字、论文声明预算和局限，来自用户提供的20页正式PDF。
- **PUBLISHER_METADATA_VERIFIED**：题名、作者、DOI、IEEE文档号与发表状态，通过原PDF和出版社登记核查。
- **AUTHOR_CODE_NOT_VERIFIED**：未取得可确认归属且含实际实现的作者仓库；不承诺精确复现。
- **SEGMENTATION_TRANSFER_DESIGN_ONLY**：当前点头适配和新增空间cache都是推论/待检验设计，本子任务未启动任何科学训练。
- 只同步本分析文档及引用。第三方PDF、完整抽取文本、页面截图、实际数据、权重和缓存不进入发布范围。ROOT负责README导航与持续同步；本子任务未自行提交或报告“已同步”。
