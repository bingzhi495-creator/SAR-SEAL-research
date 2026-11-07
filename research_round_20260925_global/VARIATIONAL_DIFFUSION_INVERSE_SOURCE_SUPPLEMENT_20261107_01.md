# 变分、扩散逆问题、SAR低秩与实例竞争：一手实现补充核验

日期：2026-11-07。范围：6项工作、4个要求的研究族，加一项2026真实SAR去斑。仅网络原文与静态源码核验；未读取活SOURCE、Ops、原始数据或权重，未SSH，未安装依赖，未执行科学任务。下述“源码存在”不等于已在本项目跑通。

## 1. 对当前任务最重要的输入边界

公共SSDD/HRSID中用于实例分割的显示图，不能未经核实当作辐射定标强度、单视幅度、复数SLC或欠采样回波。若显示过程包含未知对数压缩、裁剪、饱和与量化，原始噪声分布和相位信息通常不能由8位PNG反演出来。

Gamma乘性噪声的常见强度模型是 `y=x*n, n~Gamma(L,1/L)`；幅度对应平方根关系。若x、y是强度、L已知且条件满足，其负对数似然包含 `L*log(x)+L*y/x`。不能把同一Gamma概率直接套在显示灰度、幅度和复数数据上，也不能由任意异质港口区域的均值方差比就认定真实视数L。

因此这些研究族的优先用途不同：**图竞争更接近当前实例分割输出；去斑可作有独立对照的图像处理候选；压缩感知或物理后验重建在缺少测量算子/观测数据时不具备直接实施条件。** 下游maskAP、邻船合并/小船漏检才是本任务证据，PSNR、ENL或视觉更平滑不能替代它。

## 2. 六项证据矩阵

| 工作、原始出处 | 真实实现与固定版本 | 输入假设及本项目适用性 | 核验结果 |
|---|---|---|---|
| **RED**：Romano、Elad、Milanfar，《The Little Engine that Could: Regularization by Denoising》，SIAM J.Imaging Sciences10(4):1804–1844，2017；[原文](https://arxiv.org/html/1611.02862v3)，[出版方](https://doi.org/10.1137/16M1102884) | 作者实验仓库[google/RED](https://github.com/google/RED)，commit `31142ab55ad37c25f6704f5bfe81e7fec39360f0`；已读`RunFP`、`RunADMM`的实际迭代，不只README；Apache2.0主体，另依赖TNRD/MEX | 演示是已知线性H下模糊/下采样加AWGN；需要H、H转置、噪声参数和去噪器。把H设成身份只能称图像域正则化，不是SAR回波一致性恢复；不能给任意学得D宣称显式凸能量/全局收敛 | 核心代码可读；MATLAB旧依赖与TNRD外链未执行核验，没有下载权重 |
| **DPIR / PnP-HQS**：Kai Zhang、Yawei Li、Wangmeng Zuo、Lei Zhang、Luc Van Gool、Radu Timofte，《Plug-and-Play Image Restoration with Deep Denoiser Prior》，TPAMI；[原文](https://arxiv.org/html/2008.13751v2) | [作者cszn/DPIR](https://github.com/cszn/DPIR)，commit `15bca3fcc1f3cc51a1f99ccf027691e278c19354`；实际`data_solution`/`pre_calculate`及`main_dpir_deblur.main`；MIT主体 | HQS分离数据项与DRUNet非盲高斯去噪先验；原文允许迭代中非高斯结构残差，因此不能简单说“Gaussian denoiser只能处理高斯图”。但其FFT数据解依赖已知卷积/采样算子及所定义数据项，不能把未知SAR显示处理链伪装成已知H | 源码与模型获取入口存在；未取得/加载模型，不作本机兼容或SAR有效保证 |
| **2025 SAR-DDPM评估与聚合**：Alec Paul、Andreas Savakis，《On Denoising Diffusion Probabilistic Models for Synthetic Aperture Radar Despeckling》，Sensors25(7):2149，2025-03-28；[出版原文存档](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991316/) | 原文Data Availability直链[asp6244/SAR-DDPM-Aggregation](https://github.com/asp6244/SAR-DDPM-Aggregation)，commit `b0c87578fb4a23020af4f762f8a44330b1e1a584`；已读`SynthSARDataset.__getitem__`、`GaussianDiffusion.p_sample_loop`和`sarddpm_test.main`；MIT主体 | 光学灰度经平方→Gamma(1,1)→开方，clip并量化，生成单视幅度式合成噪声；真实SAR另按显示图加载。其合成噪声和真实显示SAR的差异不能由“physics”标签消除；不是已校准的当前数据后验 | 真正2025论文+真实代码。README标题仍是旧稿名“Sample Aggregation Methods…”，以正式论文为准；最终SAR微调权重本轮未核验。论文有DDPM低于单次U-Net及生成错误结构的反例 |
| **SAR非局部加权低秩**：Jing Fang、Shaohai Hu、Xiaole Ma，《A Boosting SAR Image Despeckling Method Based on Non-Local Weighted Group Low-Rank Representation》，Sensors18(10):3448，2018-10-13；[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6210930/) | 原文算法1/2、式25–29的加权低秩/奇异值阈值迭代可读；感谢部分提及取得他人WGLRR代码，**不能视作公开作者实现**。未找到可核验作者仓库/commit/实现许可 | 完全发展斑点、视数相关的对数域偏差处理、相似patch组近低秩；概率相似度有幅度/强度区分。8位显示灰度不能默认满足模型。小船孤立亮点可能被低秩/异常抑制当成噪声 | **论文级可核验，作者代码未知；不得登记成已取得可跑基线**。原文也承认块匹配/SVD计算昂贵 |
| **Mutex Watershed / 实例互斥图划分**：Steffen Wolf等，《The Mutex Watershed: Efficient, Parameter-Free Image Partitioning》，ECCV2018；[原文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper.pdf)；[后续目标函数原文](https://arxiv.org/abs/1904.12654) | 论文作者原链接重定向到[sciai-lab/mutex-watershed](https://github.com/sciai-lab/mutex-watershed)，commit `2641514dd38b2d1913c42d564bc96bc54ac1e636`；实际C++ `MutexWatershed::repulsive_mst_cut`、`dam_constrained_merge`、`add_dam_edge`及pybind接口已读；本轮未定位显式仓库许可证 | 输入是带吸引/排斥证据及偏移的图，不是原始SAR强度；可用于候选区域的邻船分离，但需可靠同实例/异实例关系与背景域。单纯图划分不提供船类别/漏检恢复，不等于SAR实例检测器 | 有真实算法实现；C++14/xtensor/旧pybind依赖未编译。未核实许可前不复制进发布库；不把其特定目标的最优性外推成任意multicut全局最优 |
| **2026 GLCNet**：Yang Yang、Jiangong Xu、Yuchuan Bai、Liangyu Chen、Junli Li、Jun Pan、Mi Wang，《Self-supervised global−local collaborative network for real SAR despeckling》，IJAEOG146:105135，2026；[出版方](https://doi.org/10.1016/j.jag.2026.105135) | 出版方直链[yangyang12318/LGCN](https://github.com/yangyang12318/LGCN)，commit `274af0e58d3dd565fd1f7a24d8f21105687636e7`；`MWCNN.forward`、`CentralMaskedConv2d.forward`、`GuidedDetailFusion.forward`及`MWCNNModel.train_step`已读；GPL3 | 单图自监督、wavelet-shuffle削弱斑点相关性、全局/局部分支与细节门控；不同传感器分别训练。盲点去噪须防相关噪声/信息泄漏。其附录地物**语义mIoU**不是船舶maskAP，也不证明SSDD/HRSID直接有效 | 实际公开源码存在，但默认分支有明显静态维度风险、loss重复、配置/权重发布说明不完整；**未获可运行资格**，详见第4节 |

固定commit由公开GitHub commits API核对；表内核心文件又按固定commit的raw URL读回。没有克隆或执行第三方代码。对WGLRR保留代码缺口，没有用伪代码凑“实现已核验”。

## 3. 核心函数证据与数学边界

### RED / DPIR

RED的[`RunFP.m`](https://github.com/google/RED/blob/31142ab55ad37c25f6704f5bfe81e7fec39360f0/minimizers/RunFP.m)实质交替调用Denoiser，并解含 `H'*H/sigma²+lambda*I` 的系统；[`RunADMM.m`](https://github.com/google/RED/blob/31142ab55ad37c25f6704f5bfe81e7fec39360f0/minimizers/RunADMM.m)明确有数据解、去噪器内迭代及对偶变量更新。算法原语是现成技术，改成“SAR正则化模块”不能据此主张新颖。

对一般可微D，`rho(x)=0.5*xᵀ(x-D(x))`的梯度是 `x−0.5*(D(x)+J_D(x)ᵀx)`；简化成`x−D(x)`需要额外条件。Reehorst与Schniter的[原始澄清研究](https://arxiv.org/abs/1806.02296)说明实际常见去噪器不具备所需Jacobian对称性。因此不能把任意UNet/SAM/DRUNet插入后直接宣称原显式目标、凸性与收敛定理全部保留；平衡点解释与具体收敛条件须另证。

DPIR的[`data_solution`](https://github.com/cszn/DPIR/blob/15bca3fcc1f3cc51a1f99ccf027691e278c19354/utils/utils_sisr.py#L61)包含FFT、下采样分块和实数逆FFT；[`main_dpir_deblur`](https://github.com/cszn/DPIR/blob/15bca3fcc1f3cc51a1f99ccf027691e278c19354/main_dpir_deblur.py#L155)交替数据解与带噪声水平图的DRUNet。**有迭代、有FFT不是“物理数据一致性”充分条件**；必须知道观测算子和似然为何适用于输入数据。

### 2025扩散：实现能读，不能假定比简单网络更有效

实际[`SynthSARDataset.__getitem__`](https://github.com/asp6244/SAR-DDPM-Aggregation/blob/b0c87578fb4a23020af4f762f8a44330b1e1a584/scripts/datasets.py#L97)给出了合成观测流程；[`GaussianDiffusion.p_sample_loop`](https://github.com/asp6244/SAR-DDPM-Aggregation/blob/b0c87578fb4a23020af4f762f8a44330b1e1a584/guided_diffusion/gaussian_diffusion.py#L405)保存每步`pred_xstart`；[`sarddpm_test.py`](https://github.com/asp6244/SAR-DDPM-Aggregation/blob/b0c87578fb4a23020af4f762f8a44330b1e1a584/scripts/sarddpm_test.py#L118)支持最终/末若干样本平均。论文的DSIFN合成实验里U-Net PSNR26.25、SSIM0.676，而100步SAR-DDPM为24.59/0.584：仅说明作者该数据/配方，不能判定扩散总体无效，也不能反过来预设扩散必涨点。

代码及论文中的按GT挑选最高PSNR时间步是oracle诊断，不能用于当前测试集部署结果；如将来采用此家族，时间步/平均窗口应仅在训练内部验证集固定，测试只运行选定策略。生成结构与真实小船不一致的风险，应通过原始图旁路与实例召回/边界/合并误差对照，不靠视觉选图。

### 图竞争

[`src/main.cpp`](https://github.com/sciai-lab/mutex-watershed/blob/2641514dd38b2d1913c42d564bc96bc54ac1e636/src/main.cpp#L427)确实实现了吸引边合并和互斥边登记，Python绑定位于同文件末尾。其输入含图像形状、边偏移、吸引通道数和互斥步长，边列表的置信顺序由调用端提供。它解决实例关系分离的图问题，**不要求SAR Gamma强度似然**，所以比盲套图像逆问题更接近当前输出；但从模型特征构建可靠有符号边仍是未知部分。

将来只在已确认存在邻船合并/重叠错误时评估，并对照简单mask-score逐像素argmax互斥、普通分水岭及相同候选约束；固定原始分数/框，显式记录是否增加/拆分实例。若收益只来自分数阈值或减少掩膜面积，应否定“图建模更好”解释。

## 4. GLCNet作者实现的静态资格问题

出版方正文与代码名顺序不同：论文为**GLCNet**，仓库为**LGCN**。实际方法函数名为`MWCNN`，因此不能只搜“GLCNet.py”。以下为静态推断，未执行，不能记作已观测运行报错：

1. [`network/all.py`](https://github.com/yangyang12318/LGCN/blob/274af0e58d3dd565fd1f7a24d8f21105687636e7/network/all.py#L93)：默认`in_ch=1,base_ch=64`时，`pd_img1`为DWT后4通道，`x=head_g(pd_img1)`为256通道。`mask=zeros_like(x)`后，`x_fft=fftn(pd_img1)`仍4通道，二者在`x_fft*mask`不能按默认配置广播。
2. 同函数局部分支`head_l`期待1通道，而频域张量从DWT分支产生；全局输出经IWT回全分辨率，局部分支尺寸/通道仍需独立核对。不能只改一行就报告忠实复现。`iwt_init`也硬编码新建float CUDA张量，对CPU、dtype和设备不是通用实现。
3. [`model/all.py`](https://github.com/yangyang12318/LGCN/blob/274af0e58d3dd565fd1f7a24d8f21105687636e7/model/all.py#L29)的`loss3`两个0.5项完全相同，均比较`output_end1`和`output_end`；不能代作者猜测第二项应是什么并称作者原配方。
4. 仓库README仍写训练、测试及权重将在录用后发布；当前虽有训练文件，不能由此推断完整最终配置和正式权重已齐。未核验到可用作者训练权重。

若确实想实施，需要先在独立副本记录“本项目修复/重建”，逐项核对论文张量流、损失、配置和作者更新；**不让修仓库成为阻挡强实例分割基线的长任务**。原文的10,000次迭代、真实噪声训练值得借鉴，但不提供当前任务直接涨点证据。

## 5. 执行排序与保留为OPEN的部分

1. 目前没有本轮错误统计，所有家族保持候选。图竞争只有在邻船分离确为主要误差时优先；可先形成不依赖新传感器数据的有限诊断。
2. 2025扩散有实代码，但训练/多步推理和自然图先验迁移成本高，不能作为“快速必涨点”默认主线。先保留简单U-Net/浅层平滑及原图旁路的强对照。
3. GLCNet是2026相关直接先例和可参考的盲点/去相关设计，作者默认源码暂不合格；不按README或论文漂亮数值认定能跑。
4. RED/PnP可作为已知数学工具及对照，当前缺真实观测算子/定标噪声模型，不能提出未经支持的SAR物理反演贡献。把去噪后灰度当成新增“真实模态”也不成立。
5. 非局部低秩已覆盖所需研究族，但本轮没有取得可核验作者实现，保持代码OPEN。CS/稀疏展开的欠采样回波重建输入在当前公共显示图任务中不具备，故没有用不适用的重建网络凑可跑候选。

论文前提、源码可读、实现可跑、数据适配、下游有效是五个不同证据层级。本补充完成6项范围内的一手核验，不宣称所有家族已闭合，不产生新实验数值或新科学运行授权。
