# 数学、物理与雷达信号处理原始证据审阅：公共 SAR 船舶实例分割

版本：2026-11-07；性质：研究设计审阅，未执行实验，未取得新指标，未认定创新成立或达到投稿水平。

## 1. 本次结论与适用范围

当前最值得先验证的是**有限点监督的分配是否与细小、细长实例的实际误差相匹配**。其次可研究**实例尺度归一化的有界距离场边界监督**。它们都可使用现有灰度图和实例标注。不能据此推断它们一定涨点；前景采样、边界采样、距离场和重要性估计都是已有原语，贡献只能来自被证实的误差机制、相对直接先例的具体改进与公平证据。

本次覆盖 12 个方法族，追溯经典原始论文、2025–2026 年新论文与作者/官方实现；并非穷尽全世界数学、物理、雷达方法，也不是跨论文排行榜。查新截至 2026-11-07。搜索包括 CFAR/local contrast、clutter Gamma/Wishart/K/G0、SAR despeckling、self-supervised intensity/SLC、differentiable rendering、signed distance/boundary/topology、Sinkhorn、point sampling/importance correction。关键结论只使用原始论文、作者页面/仓库、官方实现；未找到源码或无法取得全文处明确列缺口。

**数据边界是方法筛选的前提：**本报告以 SSDD/HRSID 的公开 8-bit 显示灰度 JPEG/PNG 为当前输入，没有核验为线性功率或线性幅度，没有可靠的 looks、辐射标定、成像传递函数、极化协方差、原始回波、相位、多时相配准或多视角 CAD 几何。读入 PNG 后做 FFT 不能恢复原始 SAR 相位；复制灰度通道也不能得到极化或多模态观测。灰度梯度、秩、距离和方向是可用的图像/标注量，但不能自动解释为散射系数、物理船长或真实 PSF。

必须特别注意的近先例：

- **CaSA-DETR，2026-11-05**：在 RT-DETRv2 的学习特征上计算 guard/reference 局部均值、标准差并形成软门控，再加多尺度 SACA。因此“CFAR 启发局部软门控 + 多尺度上下文 + 现代检测器”已经有很近先例；文章自己也区分 CFAR-inspired 与有恒虚警保证的 CFAR。作者代码本轮未核得。[原文](https://www.mdpi.com/2072-4292/18/21/3799)
- **OccFormer，ICCV 2023**：已经为稀疏、小类前景修改 Mask2Former 的匹配和监督点采样。不能把类/前景均衡采样当首次提出。[原文 §3.3.4](https://arxiv.org/html/2304.05316)
- **SDS-SAR，ISPRS 2026**：强相关的“仅强度图自监督去斑”新论文属实，但当前作者仓库内核是明确标注的 3×3 中值占位模型，不能按论文方法复现已经就绪处理。[论文](https://www.sciencedirect.com/science/article/pii/S0924271625004642)，[固定代码](https://github.com/YYF121/SDS-SAR/blob/82541722491054b1ca6605012e0561bcf0cc1d6b/src/models/placeholder_model.py)

## 2. 证据等级

- **P+C**：读到原始论文方法部分，读到对应作者/官方源码实际函数；不等于执行成功或复现了指标。
- **P+Cw**：原文与作者实现/正式软件入口可核验，但只读到包装、README 或部分源文件。
- **P**：原文/作者页面可核验；作者源码未知、未取得或不适合当前任务。
- **H**：本报告提出的待检验假设、推导或实验设计；不属于已发表结论。

至少 4 组达到原文方法与实际实现交叉阅读：Speckle2Void、MuLoG、Boundary Loss、clDice、PointRend、Mask2Former、OccFormer；另实际阅读 S3DIP、GeomLoss、RadarSimPy 和 MERLIN/SAR2SAR 包装实现。所有第三方代码只做阅读，未安装/导入/训练；未读取本项目运行中的科学载荷、原始数据和权重。

## 3. 12 个方法族的适用性与最小实验

### F01. 乘性散斑、Gamma / Wishart / K / G0 杂波模型

**数学对象与假设。**在线性强度、充分发展散斑及适当多视条件下，可以写成 I=R·S，S~Gamma(L, rate=L)，E[S]=1，Var[S]=1/L。多通道样本协方差 C=(1/L)Σ z_l z_l^H 在复高斯及 looks 等条件下建模为复 Wishart；单通道灰度不是多维协方差观测。将随机纹理与 speckle 相乘会产生更重尾模型：Gamma 纹理对应 K 类模型，逆 Gamma 纹理对应 G0 类模型。尾部能拟合不等于假设已被检验，也不自动提供空间独立性。[MuLoG §II–III](https://arxiv.org/pdf/1704.05335)，[Speckle2Void §IV-A](https://arxiv.org/html/2007.02075)，[Frery 等原文 DOI](https://doi.org/10.1109/36.581981)

**当前适用性。**物理似然仅能条件适用：显示映射 T、量化、饱和、JPEG 与空间相关改变分布。对 Y=T(I)，物理密度至少要考虑 T 的变换；非可逆裁剪/量化后更不能直接套连续 Gamma 像素似然。未知 T 时，不应强制用 Gamma NLL 训练或把 ENL 当真实 looks。允许把局部偏度、分位数、变异系数作为经验特征，但名称不能冒充已标定物理统计。

**源码与风险。**本轮没有取得 K/G0 原论文作者训练实现；Speckle2Void 与 MuLoG 的 likelihood 实现见 F05/F07。局部重尾参数拟合需要足够、近均质且没有船污染的样本，小窗口病态和模型误配风险大。

**最小改动与反证。**先仅做训练集外的统计适用性研究设计：线性元数据、变换敏感性、相关长度与预测尾部覆盖；现阶段不安排物理 NLL 主路线。若简单亮度/秩变换后收益消失，或实测尾概率与声称的 Pfa 不符，否定“物理恒虚警/辐射似然成立”，不能否定局部经验特征本身可能有用。

### F02. CA/OS-CFAR 与稳健局部对比

**数学对象。**CA-CFAR 用参考窗估计背景，指数强度背景、独立同分布参考样本及 CUT 等条件下，阈值系数 α=N(Pfa^(-1/N)−1)。OS-CFAR 以排序统计替代均值，意在减轻多目标等污染，仍需相应统计/秩分布标定。局部比值 I/(mean_ref+ε)、局部分位秩、(I−median_ref)/(MAD_ref+ε) 是相关的经验统计，不能混为相同的恒虚警证明。[Rohling 原文 DOI](https://doi.org/10.1109/TAES.1983.309350)，[官方参考实现文档](https://radarsimx.github.io/radarsimpy/api/process.html)

**源码阅读。**RadarSimPy commit 5a07616212142b673643f40bada49434c5e6c0dd，processing.py 的 cfar_ca_2d、os_cfar_threshold、cfar_os_2d。CA2D 明确排除 guard 区、归一化参考核、区分 linear/squarelaw；same 卷积边缘补零会改变边界统计。这是官方第三方实现，不是 Rohling 本人的代码。[固定源码](https://github.com/radarsimx/radarsimpy/blob/5a07616212142b673643f40bada49434c5e6c0dd/src/radarsimpy/processing.py)

**当前适用性与成本。**灰度局部对比可用；物理 Pfa 不可直接声称。常数个均值窗可以 O(HW)，逐窗排序约 O(HW·M log M)；稳健分位数实现可能比卷积慢。岸线/密集船污染参考窗、黑边 padding 和饱和是主要风险。

**最小修改。**在同一个 backbone 的固定位置、同参数量控制下，比较均值比值、median/MAD、局部分位秩与同尺寸普通卷积；保留原灰度分支。CFAR 门控直接先例 CaSA-DETR 必须作相关工作和强约束。[2026 原文](https://www.mdpi.com/2072-4292/18/21/3799) 若收益只来自额外容量、默认归一化或背景裁剪，不能归因为雷达机制；需要船为空图虚警、近岸/外海、饱和和灰度单调变换等分组证据。本轮不把它排在创新首位。

### F03. Lee / Gamma-MAP 局部自适应去斑

Lee 类方法根据局部均值、方差和噪声模型，在局部平均与观测之间收缩，可概括为 Rhat=μ+w(I−μ)；具体 w 取决于数据形式与模型。Gamma-MAP 用 Gamma backscatter/noise 等先验得到局部 MAP；不能把任意“均值加残差”命名为新散射模块。[Lee 原文](https://doi.org/10.1109/TPAMI.1980.4766994)，[Speckle2Void 关联原始模型说明](https://arxiv.org/html/2007.02075)

作者原始源码本轮未核得；没有把网上同名几行代码当作作者实现。当前未知线性强度与 L，适合作经验滤波对照，不能作已验证物理最优估计。均值/方差可用积分图 O(HW)，过强平滑容易抹掉小船尖端、散射中心和密集实例间隙。

最小实验是固定不训练的输入预处理或 residual 双视图，仅增加一种滤波，不与新 backbone 一起变；与 Gaussian/median、等强度平滑及直接加通道控制比较。下游 mask AP、AP75、小目标召回和边界误差决定是否有用，不用“图看起来干净”判成功。若恢复图船尖端/间隙丢失或没有同预算下游收益，放弃该配方；短预算负结果只限定于该预算。

### F04. Nonlocal / NL-SAR / SAR-BM3D

NL-SAR 按统计 patch 相似度构造扩展邻域，并在多个候选估计之间自适应选择；可处理 intensity、polarimetric/interferometric channels，但相应输入与统计仍需匹配。SAR-BM3D 将相似 patch 分组、变换域 LLMMSE/wavelet shrinkage 与后续估计结合，并非普通 BM3D 随意加 SAR 名称。[NL-SAR 作者页面](https://www.charles-deledalle.fr/pages/nlsar.php)，[SAR-BM3D 作者软件说明](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt)

NL-SAR 作者仓库入口和 v0.9/2018 版本可核验，本轮未成功读其 C 算法内核。SAR-BM3D 作者发布 v1.0/2013，入口 SARBM3D_v10，README 指向 Parrilli、Poderico、Angelino、Verdoliva 的 TGRS 50(2):606–616, 2012；本轮读取 README，未把二进制 MATLAB 包说成完整开放源代码。其 README 限制 nonprofit usage，发布衍生代码前另核许可证。

当前只能作为经验预处理路线，不保证数据统计匹配。patch 搜索成本近 O(HW·search_area·patch_area)，可能有大量 CPU 预处理；不能从训练计时中隐去。最小修改是预先规定一种参数、训练/验证/测试同规则滤波，单独报告离线成本；同时比较不滤波和同强度简单滤波。跨数据集若小船召回下降、边界变厚或收益只来自视图平滑，不能证明 nonlocal 雷达建模更优。

### F05. MuLoG：矩阵对数、统计保真与 Plug-and-Play ADMM

MuLoG 用 matrix log / 通道变换接入 Gaussian denoiser，再用统计数据项迭代校正。在线性强度模型下，log I=log R+log S，其中 E[log S]=ψ(L)−log L、Var[log S]=ψ1(L)；取 log 并不会把低 L 的偏斜噪声精确变成高斯，也不能忽略 log bias。[原文 §II–IV](https://arxiv.org/pdf/1704.05335)，[作者软件与 2022 扩展](https://www.charles-deledalle.fr/pages/mulog.php)

本轮通过作者推荐的 PyPI 正式包 **mulog 0.0.4** 读取源码，未安装或执行。wheel SHA256=749bc1b205fd20ba15442445c08a896bba800f90413e10a21c7a695a9ed05be4；Git commit 未核得，不能虚填。读到 mulog_algorithm.py:preset_mulog / iter_mulog（要求 Wishart field 和 number_looks，含 conditioning 与 log 通道初始化）；prox.py:prox_fisher_tippet / prox_wishart_fisher_tippet（Newton 数据项）；pnp_admm.py:iter_pnp_admm（denoiser、prox、residual 更新）；tools.py 的 log bias/std。[正式包元数据](https://pypi.org/project/mulog/0.0.4/)

当前无极化/干涉协方差，不能拿复制的灰度作为矩阵 MuLoG 输入；单通道版本仍需正确的强度/looks 解释。迭代 ADMM、Newton 和非局部 denoiser 可能明显增加成本。最小可比方案是验证过输入前提后，以固定迭代的单通道 MuLoG 作预处理对照；不能现在就作为“新的可微物理主干”。其主要价值是提醒我们保留统计 fidelity 与变换偏差，不是现成的新发明。

### F06. MERLIN / SAR2SAR：有数据条件的自监督

MERLIN 从真实 SLC 的实、虚分量建立独立训练信号，其独立性涉及 SAR transfer-function 条件及预处理；SAR2SAR 借助多时相并补偿变化，避免把真实变化学成噪声。[MERLIN 原文 §II–III/附录](https://arxiv.org/html/2110.13148)，[SAR2SAR 原文](https://arxiv.org/abs/2006.15037)

作者原始 MERLIN GitLab 入口已在原文核验，本轮连接未取得内核。实际读到 hi-paris/deepdespeckling commit b9bf54a013933a8dbfa63d5e02cd5b0a7acd7efc 的 merlin/merlin_denoiser.py:preprocess_noisy_image、denoise_image_kernel、preprocess_denoised_image，以及 sar2sar/sar2sar_denoiser.py:load_model / denoise_image。前者明确读取两个实虚通道、分别估计后合成为 amplitude。此包是包装实现，不能冒称原作者训练源码。[固定包装代码](https://github.com/hi-paris/deepdespeckling/blob/b9bf54a013933a8dbfa63d5e02cd5b0a7acd7efc/deepdespeckling/merlin/merlin_denoiser.py)

当前没有输入前提，暂不列本轮主实验。人为添加随机 phase/人工分裂同一 JPEG 不会恢复 MERLIN 证明中的独立观测。若未来取得公开有许可的 SLC，可另设数据分支；跨 chip 分割提升必须另行验证，不能由去斑论文指标推断。

### F07. Speckle2Void 与 2025–2026 单图自监督去斑

Speckle2Void 的盲点 CNN 预测 inverse-Gamma 条件先验参数，结合 Gamma noise 得 posterior mean (β+L·y)/(α+L−1)，训练依赖盲点/去相关防止复制噪声。原文指出空间相关与盲点大小之间有细节–去噪折衷。[原文 §IV](https://arxiv.org/html/2007.02075)

作者代码 commit ca4e83e7b8c9880dd12805430d0de2fd9c990f2b，Speckle2Void.py 的 build_inference / loss 实现 posterior、likelihood 与 TV；不是通用“不需任何统计条件”的 Noise2Void。[固定源码](https://github.com/diegovalsesia/speckle2void/blob/ca4e83e7b8c9880dd12805430d0de2fd9c990f2b/Speckle2Void.py)

新进展需要区别成熟度：

- **S3DIP，PRL 2025**：以 DIP 同时拟合 clean branch 与非负噪声矩阵，加入 soft-histogram 统计约束及 SAR-BM3D/FANS 的 guidance。作者 commit ea018dab23ef44ec2106b182661d3bec921fc0d4，已读 s3dip_real.py 与 hist_loss_broadcast.py；前者联合优化 net/noise，后者用三角软分箱。每图优化与 guidance 预处理是实际额外成本。[原文](https://doi.org/10.1016/j.patrec.2025.02.021)，[作者源码](https://github.com/IAPP-Group/S3DIP/tree/ea018dab23ef44ec2106b182661d3bec921fc0d4)
- **SDS-SAR，ISPRS 2026**：RA-SAMPLE 从 intensity 构造训练对，适合认真追踪；但已读作者源码是 medianBlur placeholder，论文实际实现尚未核得，不能列为已复现。
- **2026 score/log-domain preprint**：Heo 的论文声称近高斯变换与较快推理；作者实现本轮未知，不能凭新年份或摘要列为 SOTA。[原文](https://arxiv.org/abs/2601.14334)

未知显示映射让物理统计条件依然不成立。最小方案是把可取得、可匹配数据前提的方法作为独立预处理对照；先检验对标注小船与间隙的保真。比拼 mask AP 必须同一分割器、同训练预算、完整算入预处理。ENL 变大可能只是过平滑，不能单独证明下游有效。

### F08. 可微 SAR 成像 / 散射 / 逆渲染

有原始 echo 或已知几何时，成像可以抽象为 y=A_g(ρ)+n，物理一致性约束依赖 A_g 与观测。2025 inverse rendering 原文用 DSM、backscatter map、雷达观测几何与多视角图，按 surface patches/range-cell contributions 建立可微渲染；2026 OptiX 模拟还需要 glTF 场景、频率/带宽、视角、粗糙度等。它们不等价于在 JPEG 网络内加傅里叶卷积。[2025 原文 §II–III](https://arxiv.org/html/2502.10492)，[2026 原文](https://www.mdpi.com/2072-4292/18/20/3579)

作者源码本轮未核得；2024 DSR 角度反演原文也依赖给定 3D target model，不能从当前单视灰度自然得到。[DSR 原文](https://arxiv.org/html/2401.01165) RadarSimPy 读到 range_fft / doppler_fft / MUSIC 等实现入口，但这类 FFT/阵列处理需要时间、阵元或相干相位数据，不能把二维图像坐标当雷达时间–阵列样本。

暂不作为现有输入的最小改动。若未来另获公开 geometry/multi-view/SLC，先用真实配对验证 forward consistency 和 domain gap，再研究分割；仿真视觉逼真不能替代真实跨数据集验证。该路线数据与计算成本高，不能以物理词汇包装未满足输入的实验。

### F09. 各向异性 / 曲率 / Level-set / SDF 边界几何

图像域可以用 ∂u/∂t=div(D∇u) 表达各向异性扩散，level-set 常用 κ=div(∇φ/|∇φ|) 描述曲率；这些是经典图像数学，不是仅 SAR 才有。原始 SRAD 用 speckle-sensitive diffusion，依然有噪声统计条件。对当前最可靠的可用量是**标注生成的像素距离场**。Boundary Loss 将 contour distance 写成区域积分 L_boundary≈mean[φ_GT(x) p(x)]，以便训练网络。[Boundary Loss 原文 §2](https://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf)，[SRAD 原文](https://doi.org/10.1109/TIP.2002.804276)

作者代码 commit 171c32d88a4ce59af8be46fb88b96d3637b9515b，losses.py:SurfaceLoss/BoundaryLoss 用 probs×dist_map 平均；utils.py:one_hot2dist 用 EDT 构造正负距离并支持 spacing。本轮未核得 SRAD 原作者软件。[固定 Boundary Loss 源码](https://github.com/LIVIAETS/boundary-loss/blob/171c32d88a4ce59af8be46fb88b96d3637b9515b/losses.py)

距离以像素表示，不冒称米。EDT 可预计算 O(HW)，训练乘法 O(HW) 或固定 K 点；曲率/细分 PDE 会增加迭代与梯度不稳定风险。最小方案是按已匹配实例加入低权重、有界/尺度归一的距离场项，保持 matcher、backbone、点总数不变；直接对照原始 Boundary Loss、单纯 BCE 前景加权、boundary BCE。若只改善厚边界指标却降低 AP75/小船召回，或收益对 ±1 pixel 标注扰动极敏感，就限制或否定该配方。边界带宽、缩放增强与 mask rasterization 必须预先定义，不能验证集逐图调。

### F10. 骨架、图拓扑与 clDice

clDice 用预测 skeleton 落入 GT 的 precision、GT skeleton 落入预测的 sensitivity 的调和均值，soft 版本用 min/max pooling 迭代近似 skeleton。原文拓扑保证有前景/背景 skeleton 与包含关系条件；不能把优化 soft loss 说成任意概率 mask 的无条件拓扑保证。[原文 §2–4/附录](https://arxiv.org/html/2003.07311)

作者 commit 47d31a6cc4a8101b1ffe8052994821961e57af9f，读 cldice_loss/pytorch/cldice.py 和 soft_skeleton.py。实现有需注意的实际差异：虽然构造器接收 iter_，此版本创建 SoftSkeletonize 时硬编码 num_iter=10，不能未经检查以为传 iter_=3 就只执行 3 次。[固定源码](https://github.com/jocpae/clDice/blob/47d31a6cc4a8101b1ffe8052994821961e57af9f/cldice_loss/pytorch/cldice.py)

当前可使用 mask 数学，但船体不是天然血管/道路网络；不能合并所有船后要求整体连通。仅在已匹配的单实例细长断裂确为误差时，作为轻权重对照；成本 O(iter·HW)，对毛刺、裁边、标注孔洞敏感。必须同时看 false bridges、merged instances、holes、AP75，不能只看连通率。若提升只是填平间隙并粘连邻船，机制失败。

### F11. 最优传输 / Sinkhorn 几何距离

以 μ=Σ a_i δ_xi、ν=Σ b_j δ_yj 表示边界/形状点，熵正则 OT 最小化 Σπ_ij C_ij+ε KL(π,a⊗b)。Sinkhorn divergence 减去两项 self-cost；“debiased”是消除正则化 self-bias，并不等于对有限采样无偏。[Feydy 等原文](https://proceedings.mlr.press/v89/feydy19a.html)

作者 GeomLoss commit 00e493f36bd6cc8471526e59afc07193a1926e47，已读 _legacy/samples_loss.py:SamplesLoss、sinkhorn_samples.py:sinkhorn_tensorized/online、多尺度入口，以及 sinkhorn_divergence.py:sinkhorn_loop / sinkhorn_cost。tensorized 路径显式构造跨集和 self-cost 矩阵；online 路径借 KeOps 隐式计算，不等于时间必然线性。源码里的 “UNBIASED”注释不能解释成随机监督比值无偏。[固定源码](https://github.com/jeanfeydy/geomloss/blob/00e493f36bd6cc8471526e59afc07193a1926e47/src/geomloss/_legacy/sinkhorn_samples.py)

当前像素几何可用。balanced OT 要考虑质量/归一化；把每个实例归一成概率会丢失面积信息，必须保留区域损失。朴素 O(T·K²) 时间、O(K²) 存储，很容易比固定 K 的 BCE 贵。最小对照为每个已匹配实例固定少量边界点的 OT auxiliary，并比较 Chamfer/SDF、相同采样点和面积损失，完整计成本。若只因更大边界采样预算或面积重权得到收益，不可归因于 transport；未优于更便宜的 SDF 时不优先部署。

### F12. 不确定点采样 / 分层抽样 / 重要性校正

**直接先例。**PointRend §3.1 原文明确是 biased training sampling；实现从 3K 候选中选最不确定的 βK，再补随机点。Mask2Former 的匹配使用所有候选共享的均匀随机点，监督则每个已匹配实例独立用 uncertainty+random 点。OccFormer §3.3.4 已用类别频率修正采样以保留 sparse foreground。它们不是本报告的新方法。[PointRend](https://arxiv.org/html/1912.08193)，[Mask2Former](https://arxiv.org/html/2112.01527)，[OccFormer](https://arxiv.org/html/2304.05316)

已读作者源码：

- detectron2 commit 247691a1c292654867deb35cabb27f18d75a1363，point_features.py:get_uncertain_point_coords_with_randomness、point_sample。先双线性插值 logits 再算 −abs(logit) 不确定性；先算粗格 uncertainty 再插值不等价。[固定源码](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend/point_rend/point_features.py)
- Mask2Former commit 9b0651c6c1d5b3af2e6da0589b719c514ec0d69a，criterion.py:loss_masks / sigmoid_ce_loss / dice_loss；matcher.py:memory_efficient_forward。该实现直接在抽到点上求 BCE 均值和 Dice 比值，没有 per-point 逆概率校正。[criterion](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/criterion.py)，[matcher](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/matcher.py)
- OccFormer commit 44aee79434d6aedca0d97fe62965032505f583e7，mmdet_utils.py:sample_valid_coords_with_frequencies / get_uncertain_point_coords_3d_with_frequency；mask2former_occ.py:loss_single。multinomial 使用 replacement=False，随后 BCE/Dice 点损失未见按点 inclusion probability 的逆权修正；实例类别权重不是该修正。[采样源码](https://github.com/zhangyp15/OccFormer/blob/44aee79434d6aedca0d97fe62965032505f583e7/projects/mmdet3d_plugin/occformer/mask2former/base/mmdet_utils.py)

**不同问题不能混称。**Focal 的 (1−p_t)^γ 与 GDL 的 inverse-volume² 改变优化目标；不应当“采样无偏校正”的替代称谓。保留它们作为强对照，检验候选收益是否仅仅重权正样本。[Focal 原文](https://arxiv.org/html/1708.02002)，[GDL 原文](https://arxiv.org/html/1707.03237)

**反例也必须保留。**PartDistillation 的 CVPR 2023 作者补充材料改为相同 12,544 个全随机监督点，理由是聚类伪分割边界本身有噪声。2021 IVC 已研究角点/随机/边界偏移点监督；边界点更多并不自然等于更好。[PartDistillation 补充原文](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cho_PartDistillation_Learning_Parts_CVPR_2023_supplemental.pdf)，[2021 原文](https://www.sciencedirect.com/science/article/pii/S0262885621000743)

当前可研究，推理无须新输入；但“uncertainty 被 SAR clutter 吸引、漏掉小船边界”只是待测假设。额外构建 dense proposal maps、EDT、索引与采样并不免费，固定最终 K 不代表同计算量。必须同时报告监督点、候选点、GT 预计算、实际 step time/显存。

## 4. 采样数学：可以证明什么，不能证明什么（H）

以下是本报告对拟议实验的数学约束，尚未变成本项目实现或结果。

### 4.1 先固定目标，再谈无偏

离散 N 个像素中心的 BCE 目标是 L=(1/N)Σ_i ℓ_i。若独立、有放回抽 i_k~q，q_i>0，则：

Lhat=(1/K)Σ_k ℓ_i_k/(N q_i_k)，E[Lhat|当前样本、当前参数]=L。

这是 Hansen–Hurwitz 风格的重要性均值。若 q 依赖当前 θ，但抽样及权重都 stop-gradient，条件期望梯度恢复 (1/N)Σ∇ℓ_i。前提是权重准确、所有目标点有正支持、没有另外 clip/自归一化；权重 clipping 或以 Σw 再归一化通常改变有限样本无偏性。

无放回设计要用 inclusion probability π_i=Pr(i 被纳入样本)，HT=(1/N)Σ_{i∈S} ℓ_i/π_i。torch.multinomial(replacement=False) 的初始归一化 q_i 通常不是 π_i；不能直接用 1/(Nq_i) 并宣称 HT 无偏。[HT 原始论文](https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446)

PointRend/Mask2Former 的原生 point_sample 使用连续坐标，所读 Mask2Former 实现对 prediction 与 GT 都做双线性采样。本轮主任务对 RF-DETR 1.12.0 的独立源码核验则显示：prediction 为 bilinear，而 _sample_target_masks_at_points 对 GT 使用 nearest。不能把两者混写成相同标签目标。均匀连续点估计的是对应插值/标签约定下的单位方形损失积分，它不一定等于像素中心的离散 BCE。实验必须先冻结具体载体、GT/prediction 插值与 padding 约定；改为像素中心目标属于目标变化，不能称完全保持原生目标。

### 4.2 固定预算分层更容易审计

用有符号像素距离 d_i（内负外正）形成互斥层：F={d<−w}（前景内部）、B={|d|≤w}（双侧边界带）、G={d>w}（远背景）。每层均匀抽 K_s>0 点，Σ_s K_s=K，n_s 为层大小：

Lhat_strat=Σ_s (n_s/N)·mean_{k∈s} ℓ_i_k。

这在每层内部均匀、目标支持完整时对离散 BCE 无偏；不是把三层损失各乘 1/3。固定 K_s 后需要定义空层回退、小层有/无放回、cropped instance 与 padding 的规则。若保留连续目标，可以将像素单元分层后在每单元内均匀抖动，并按层面积校正，同时保持该载体实际的 GT/prediction 插值约定。

这里的连续方案必须保留所选载体实际约定：所读 Mask2Former 的 GT/prediction bilinear，或主任务已核 RF-DETR 1.12.0 的 GT nearest/prediction bilinear。自适应 q 可使用 uniform floor，例如 q_i=ε/N+(1−ε)q'_i。不可把 deterministic top-k 的 score softmax 当作它真实的抽样概率。若使用候选筛选与二阶段采样，必须计入所有阶段的真实纳入/抽样设计。

### 4.3 Dice、方差与目标变化

分别无偏估计 Dice numerator/denominator，不意味着它们的比值无偏，E[Ahat/Bhat]通常不等于 A/B。原生 Mask2Former（以及主任务已核 RF-DETR）中的 Dice 与 BCE 共享 uncertainty+uniform 点，而不是全均匀 Dice。如果保持原生 K 个 Dice 点，再为候选 BCE 另抽 K 个点，实际最多涉及 2K 个点，并非固定总点数；若把两者预算拆为 K_Dice+K_BCE=K，又改变了原生点数和方差。共享新 K 点则同时改变 Dice 的采样目标。第一轮宜限额诊断 BCE estimator 与点覆盖；进入训练前必须明确选择哪种设计、为基线提供同样总点/候选/时间预算，并单独审计 Dice 改变，不能预称保持原生训练目标或成本。

对有放回的梯度 g_i=∇ℓ_i，有 E||ghat−gbar||²=(1/K)[Σ||g_i||²/(N²q_i)−||gbar||²]；理想 q_i∝||g_i||。不确定性或正边界覆盖只是可计算 surrogate，不保证方差更低。应该实测 dense/coarse-grid gradient 参照、方差/余弦、权重 ESS、各层梯度份额与正边界命中。uniform 小实例正样本零命中的概率可计算为 (1−A/N)^K；该公式不证明 native uncertainty 一定更差。

## 5. 优先两条候选研究假设

### H1（优先）：保持明确 BCE 目标的实例几何分层监督

**假设。**在固定总点数与相同训练配方下，细小/细长实例的正边界有效覆盖不足是可重复的实际误差；按实例的内部–双侧边界–背景分层并做明确面积/概率校正，能改善 gradient estimator 的覆盖–方差折衷，提升 AP75/小船边界质量，推理开销不增加。

**与直接先例的差异候选。**相对 PointRend 的纯预测 uncertainty、OccFormer 的类别频率采样，尝试联合实例几何分层与明确的 BCE 目标定义，并在 display-SAR 的细长小实例上验证机制。差异是待检验组合与作用机制，不是重新发明分层抽样；本轮未找到完全同式先例不等于证明世界上不存在。当前已知组合不足以单独支撑 TGRS 核心创新，优先级仅为限额诊断，须先排除原生 Dice/总点数/插值差异混杂。

**启动前诊断。**在有资格运行时记录 native 采样点落入 F/B/G 的数量、正边界命中率、面积/细长度分组、未命中实例比例与 held-out mask errors；仅从训练/诊断划分得出设计。若不足不存在，不能保留这个机制叙述硬凑论文。

**最小对照次序。**

1. Native uncertainty+random；同 K 全均匀；同 K 前景/背景均衡但不加边界。
2. 三层但不校正；三层面积校正 BCE；matcher 固定。Dice 必须按 §4.3 预先选择共享新点、分拆预算或额外点的设计，对照总点/候选/时间成本一致；不能简单声称 Dice 与成本保持原生。
3. 同 K 的 class-frequency/OccFormer-style adaptation；weighted BCE/Focal；原始 Boundary Loss。
4. 比较固定配额与基于可观测 variance proxy 的配额；若增加 proposal 计算，给基线相等候选/时间预算。
5. 短预算所有对照同长度；至少入围 baseline/最优对照/候选进入同等长训、重复种子和第二数据集。

**否证。**校正版本不优于 uniform、仅未校正版本受益，说明更可能是目标重权；若加大 K 就抹掉收益，则范围应限定为低点预算；若只有一粒种子或单一数据划分受益，不认定稳定；若充分长训无收益，不能以短训优势声称普遍 SOTA。±1 pixel 边界扰动/不同 rasterization 导致崩溃时需要减弱硬边界采样。

### H2（次优）：实例尺度归一、窄带有界的距离场监督

**假设。**若主要误差是沿船体窄边/端点的形变和密集邻船间隙，而非分类/检测漏召回，原始全图 SDF 距离可能被大实例尺度、远背景与 raster 边界误差主导；使用每实例截断归一的 d'_i=clip(d_i/s,−τ,τ)，并限制有效窄带，可改善该折衷。

**技术差异候选与约束。**相对原始 SurfaceLoss 的全图绝对距离，测试 instance-normalized bounded narrow band；它仍是已知 SDF/边界损失的任务化改进，尚不足以直接叫新理论。s 只能来自像素 mask/box 的预定义尺度（如 sqrt(area)，带下限），不冒称物理船长；归一后对小目标噪声放大的风险必须单独验证。

**对照。**原始 SurfaceLoss；只截断；只尺度归一；同边界点的 BCE 重权；H1 sampler 固定；无新增损失。先与每实例 equalized loss 及简单 perimeter/area scaling 比较，区分新几何设计与隐含的小实例加权。若尝试方向张量/OT，作为后续独立实验，不首轮叠加。

**否证。**没有稳定改善 AP75/细长与密集分组；只对训练标注的边界风格过拟合；外部公开数据不保留；比简单 boundary BCE 更贵但不更好。任一发生都应降低优先级。H2 在 H1 证据明确之前只保留设计，不同时大规模扫超参数。

## 6. 原文—源码核验台账

| 方法 | 原文/方法定位 | 实际阅读文件或入口 | 固定版本 | 状态 |
|---|---|---|---|---|
| RadarSimPy CFAR | Rohling 1983；官方 process 文档 | processing.py:cfar_ca_2d, os_cfar_threshold, cfar_os_2d | 5a076162… | 官方第三方实现；未运行 |
| Speckle2Void | §IV-A–F | Speckle2Void.py:build_inference, loss | ca4e83e7… | P+C；未运行 |
| MuLoG | §II–IV；2022 扩展入口 | mulog_algorithm.py; prox.py; pnp_admm.py; tools.py | PyPI0.0.4, wheel SHA256 上文 | P+C；commit 未核；未安装 |
| MERLIN/SAR2SAR | MERLIN §II–III/附录；SAR2SAR 原文 | merlin_denoiser.py; sar2sar_denoiser.py | b9bf54a0… | 仅包装源码，原作者训练内核未取得 |
| S3DIP | §2–3.2 | s3dip_real.py; hist_loss_broadcast.py | ea018dab… | P+C；未运行 |
| SDS-SAR | RA-SAMPLE，ISPRS2026 | src/models/placeholder_model.py; README | 82541722… | 占位实现，论文内核未发布/未核得 |
| Boundary Loss | §2 | losses.py:SurfaceLoss,GeneralizedDice; utils.py:one_hot2dist | 171c32d8… | P+C；未运行 |
| clDice | §2–4/附录 | cldice.py; soft_skeleton.py | 47d31a6c… | P+C；iter_ 硬编码差异 |
| PointRend | §3.1 | point_features.py:point_sample, get_uncertain_point_coords_with_randomness | 247691a1… | P+C；未运行 |
| Mask2Former | §3.3 | criterion.py:loss_masks; matcher.py:memory_efficient_forward | 9b0651c6… | P+C；未运行 |
| OccFormer | §3.3.4 | mmdet_utils.py; mask2former_occ.py:loss_single | 44aee794… | P+C；未运行 |
| GeomLoss/Sinkhorn | AISTATS2019 §2–3 | _legacy/samples_loss.py, sinkhorn_samples.py, sinkhorn_divergence.py | 00e493f3… | P+C；未运行 |
| NL-SAR | 作者说明/文档 | toolbox v0.9 入口、手册 | v0.9 | 内核未取得 |
| SAR-BM3D | 作者 v1.0 README 指向原文 | README；SARBM3D_v10 入口 | v1.0/2013 | 仅文档；非完整源码核验 |
| 可微成像/渲染 | 2025 §II–III；2026 simulation | 原文；代码未核得 | 未知 | P；当前缺输入 |
| CaSA-DETR | 2026-11-05 §1.2 及方法描述 | 原文；作者代码未核得 | 未知 | P；强近先例 |

完整 commit、链接及每族适用条件见配套 JSON。未列“复现成功”、未引用外部数值作本项目公平对照，也未把任何方案自动判为发表级创新。

## 7. 研究执行约束

只在既有数据划分、初始化、augmentation、训练步数、有效 batch、优化器/学习率、评估阈值与 mask encoding 全部记录后比较。分割 mask AP 与检测 box AP 分开；除总体 AP，还预先定义小目标、细长度、近邻密集度、边界、ship-free false alarms 的分组，避免看结果后重选分组。

候选方法不能独享更多种子、调参次数、预训练数据或更久训练；外部 SOTA 和自研方法用相同证据标准。短预算是筛选证据，长预算才回答收敛与最终性能。长训失败必须保存，不以“还没调好”无限拖延；同时不能因不公平短预算断言一个架构总体无效。

本轮只形成研究设计与实现阅读，不接触运行绑定源码，不启动实验、不下载权重，不改变现有训练资格链。后续由主任务根据资格和资源安排实验与公开同步；本报告本地形成不等于已上传 GitHub。


## 8. 明确的 OPEN 覆盖边界

“12 个方法族”是本轮按当前任务组织的检索范围，不代表每个子原语都已完成原文与作者内核交叉审阅。下列项目是**后续覆盖项 OPEN**，不能写成已系统验证。

| 家族/链路 | 典型数学对象 | 所需观测 | 本轮状态与当前决策 |
|---|---|---|---|
| 匹配滤波、脉冲压缩 | 已知发射波形与接收信号的相关/卷积 | 复数时序、带宽/采样/波形 | OPEN：未系统追原文作者实现；不是对显示图滤波就能声称脉冲压缩 |
| 距离–多普勒、RMA、CSA、BP / SAR 成像 | 带几何的相位历史与成像算子 | 原始 echo 或相应可重聚焦数据、轨道/几何 | OPEN：本轮仅读 RadarSimPy FFT 入口，不是完成四种 SAR 成像算法核验；当前不适用 |
| 相干积累、STAP、MVDR、MUSIC | 相位一致的快拍/阵列协方差、空时自适应谱估计 | 多脉冲/多阵元复数快拍和标定 | OPEN：RadarSimPy 中定位到 DOA 入口，不宣称已深读这些算法原始论文；当前显示 chip 不满足输入 |
| CFAR | 参考样本估计与阈值/秩统计 | 线性统计解释或明确的经验特征解释 | 本轮已读 CA/OS 官方实现，物理保证仍受数据假设限制；不同于前面相干链路 |
| 变分 / PDE | fidelity+regularizer；扩散、曲率、level-set | 图像与边界/统计假设 | PARTIAL：Boundary Loss 与 MuLoG 深读；一般 level-set、SRAD 原作者源码未核得 |
| 稀疏 / 压缩感知 | 稀疏系数、测量算子 A、l1/相关先验 | 观测与已知 A；或明确图像域稀疏表示 | OPEN：未完成当前最新 SAR 原文+作者内核核验；不把显示图 FFT 当欠采样原始测量 |
| 低秩 / 稀疏分解 | 矩阵/张量的相关结构、rank/sparse 正则 | 时空/patch 组织及适当低秩前提 | OPEN：未完成原文源码查证；动态船只、多源显示映射会破坏某些前提 |
| PnP / RED | denoiser 诱导先验与数据项迭代 | 正确 forward model / 数据项 | PARTIAL：MuLoG 的 PnP ADMM 实际读到；S3DIP 仅以去斑输出 guidance，不等于 RED 内环；通用 RED 原文源码未完成 |
| 扩散 / score 去斑 | score/迭代生成或恢复过程 | 训练分布、噪声/变换假设 | PARTIAL：2026 preprint 原文摘要已核，2025 DDPM 作者入口已发现；内核尚未读，不认 SOTA/可复现就绪 |
| 最优传输 | measure matching、Sinkhorn | 可用几何点/权重 | 本轮 GeomLoss 与原文已读；最新所有 OT 变体未穷尽 |
| 图谱 / 几何 / 拓扑 | 图邻接、谱算子、距离、骨架/同伦 | 可构造的图与明确先验 | PARTIAL：clDice/SDF 已读；图拉普拉斯谱、持久同调及最新图实例分割未完成原文源码核验 |

本轮最确定的收益是**排除输入不满足的物理路线、识别近先例和防止“有仓库即有真实实现”的误判**。它使后续实验更有针对性，不能代替实验本身。

