# 经典雷达信号处理来源补录：六项可核查链路及输入边界

核查日期：2026-11-07。目标：为主要经典处理家族建立“原理/官方文档—真实源码—输入—许可—执行缺口”的证据入口。本补录只作联网静态审阅及普通文档保存；未访问 SSH、Ops、活动 SOURCE、项目数据、权重，未运行仿真、重建、训练或评估。不声称这些第三方实现已被本项目验证或属于原算法作者代码。

本轮限定 **6 项**：①RDA/CSA/ω-k 成像家族；②BP；③脉压/匹配滤波；④MVDR/Capon；⑤MUSIC；⑥稀疏 SAR/复数稀疏恢复。MVDR 与 MUSIC 满足阵列方法至少两项；本轮没有另外给 STAP 设第七项，亦不宣称已覆盖完整 STAP 实现。

## 1. 输入先决条件及可用程度总表

“源码存在”指读到实际计算语句，区别于 README、函数签名或占位符；“候选可执行入口”仍需后续独立环境与输入验证。本轮所有条目均为 **未执行**。

| 项 | 主要真实输入 | 已核实现与性质 | 当前公共灰度 SAR 船舶图能否直接使用 |
| --- | --- | --- | --- |
| 1 RDA/CSA/ω-k | 快时间×慢时间的复数回波/相位历史；载频、波形、采样率/PRF、轨迹/速度、参考距离等 | PySimSAR RDA；denkywu MATLAB RDA/经典CSA；RITSAR含Stolt的ω-k；均为后续独立实现，非原算法作者代码 | **不能**。成像后的JPEG/PNG没有完整相干回波、采集几何与相位；对像素做FFT不是重新SAR聚焦 |
| 2 BP | 复数距离压缩回波，每脉冲相位中心位置、像素3D坐标、中心频率、距离bin间隔、运动补偿参考距离 | NVIDIA官方MatX算子与CUDA kernel，BSD-3-Clause | **不能**。输入不是图像像素；有CPHD/距离压缩相位历史时是独立成像支线 |
| 3 脉压/匹配滤波 | 接收复数采样与已知发射参考波形，采样时钟/门延迟等 | SciPy官方复数线性互相关；PySimSAR LFM频域参考相关 | **不能解释为物理脉压**。普通图像卷积可以计算，但没有发射波形及回波时间轴时不是雷达距离压缩 |
| 4 MVDR/Capon | 相干阵列快拍或其复协方差、阵列几何/波长/导向矢量 | RadarSimPy官方processing.py真实Capon代码，GPL-3.0 | **不能**。邻近图像像素没有自动等价的阵元相位/空间导向信息 |
| 5 MUSIC | 同上，另需信号数/子空间维度、噪声及阵列模型假设 | RadarSimPy官方processing.py真实特征分解与伪谱代码，GPL-3.0 | **不能**。把普通特征通道当相干阵元，不构成MUSIC物理超分辨率 |
| 6 稀疏SAR/复数恢复 | SAR回波和已知正/伴随测量算子；或独立相干InSAR复数干涉图与字典 | 原作者SPGL1完整通用BPDN求解器；作者ComCSC实际复数稀疏代码但有helper缺口 | 原始反演**不能**。现有幅度图可另做明确的图像域稀疏先验，但不能声称恢复实测相位/新观测 |

当前可借鉴的是明确标为图像域的局部结构、正则化及鲁棒估计思想；若要提出真实物理层模块，必须先合法取得对应原始输入和元数据，建立独立任务及评价协议，不能强塞到当前实例mask训练。

## 2. 第1项：RDA、CSA、ω-k/RMA类成像

### 2.1 原理与官方/作者技术资料

- RDA的基本链路是距离压缩、距离徙动校正、方位压缩。实现项目作者的[PySimSAR成像数学文档](https://github.com/music-sustech/PySimSAR/blob/4864c26af205c3571d1dc2d62a603e62553b9e7f/docs/math/image-formation.md)列出输入物理量及算法步骤，并引用Cumming与Wong的SAR处理教材。这是**软件作者文档**，不是最初提出RDA者发布的代码。
- CSA原始来源入口：Raney、Runge、Bamler、Cumming、Wong，*Precision SAR Processing Using Chirp Scaling*，1994，[DOI](https://doi.org/10.1109/36.298008)。作者机构[DLR档案](https://elib.dlr.de/205/)确认题名、作者、年份，但该旧档案文献类型/期刊字段混杂，不能照搬为新的会议出版结论；其全文不在该页开放。本轮另读到作者UBC机构提供的后续原始论文[A Chirp Scaling Approach for Processing Squint Mode SAR Data](https://sar.ece.ubc.ca/papers/Davidson%2BCumming%2BIto_1996.pdf)，用于对照chirp scaling技术脉络。
- 频率—波数迁移原始来源：Cafforio、Prati、Rocca，*SAR Data Focusing Using Seismic Migration Techniques*，1991；[作者机构档案](https://re.public.polimi.it/handle/11311/524809)。它说明频率—波数方法及几何适用条件；不能据此将任何叫omega_k的仓库函数当作忠实原实现。

### 2.2 真实源码与固定版本

| 实现入口 | 固定commit与许可 | 已读计算内容 | 可执行性/边界 |
| --- | --- | --- | --- |
| PySimSAR `RangeDopplerAlgorithm.process/range_compress/azimuth_compress` | [music-sustech/PySimSAR](https://github.com/music-sustech/PySimSAR/tree/4864c26af205c3571d1dc2d62a603e62553b9e7f)，`4864c26af205c3571d1dc2d62a603e62553b9e7f`；根LICENSE MIT | [range_doppler.py](https://github.com/music-sustech/PySimSAR/blob/4864c26af205c3571d1dc2d62a603e62553b9e7f/pySimSAR/algorithms/image_formation/range_doppler.py)：读取RawData.echo，调用波形距离压缩，构造PhaseHistoryData，再完成方位处理/插值RCMC | 实际Python计算代码已读；需要该项目的Radar/Trajectory/RawData对象。未安装、未运行 |
| 教学MATLAB RDA脚本 `RDA_imaging.m` | [denkywu仓库](https://github.com/denkywu/SAR-Synthetic-Aperture-Radar/tree/8a68c5673edace4c8a9cde3c1d3c5fd02d326500)，`8a68c5673edace4c8a9cde3c1d3c5fd02d326500`；未核到明确根许可证，**许可未知** | 相对路径 `1-SAR成像算法/RD算法/1-2014.10.10. 正侧视，RDA/RDA_imaging.m`：点目标回波仿真；`H_range=conj(S_ref)`距离匹配、sinc插值RCMC、方位匹配及IFFT | 有完整计算语句与参数，但属于第三方教学复现；许可未清不能直接并入发布代码 |
| 教学MATLAB经典CSA脚本 `CSA_imaging.m` | 同上commit，许可未知 | 相对路径 `1-SAR成像算法/CS算法/1-正侧视，点目标仿真/CSA_imaging.m`：原始复回波、RD域变标相位`s_sc`、二维频域距离压缩/SRC/一致RCMC、方位滤波和附加相位校正 | 是可静态核对的经典相位变标计算，不把其点目标结果外推为所有高斜视实测数据精度 |
| RITSAR `omega_k(phs,platform,taylor=20,upsample=6)` | [dm6718/RITSAR](https://github.com/dm6718/RITSAR/tree/0e36d2ed50a92304c9b2c92ad1e1e46e042011e5)，`0e36d2ed50a92304c9b2c92ad1e1e46e042011e5`；根LICENSE正文MIT | [ritsar/imgTools.py](https://github.com/dm6718/RITSAR/blob/0e36d2ed50a92304c9b2c92ad1e1e46e042011e5/ritsar/imgTools.py)：方位FFT、二维匹配相位、明确的Stolt插值，再二维图像变换 | 第三方公开实现，函数注释指向Carrara等教材；不是Cafforio原代码。旧Python代码出现整数除法索引/旧API，当前Python兼容性未经验证 |

**命名核查的实际发现。** PySimSAR同commit的[chirp_scaling.py](https://github.com/music-sustech/PySimSAR/blob/4864c26af205c3571d1dc2d62a603e62553b9e7f/pySimSAR/algorithms/image_formation/chirp_scaling.py)自述采用“bulk相位补偿+residual插值”；[omega_k.py](https://github.com/music-sustech/PySimSAR/blob/4864c26af205c3571d1dc2d62a603e62553b9e7f/pySimSAR/algorithms/image_formation/omega_k.py)采用精确迁移因子后的RCMC插值。它们有代码，但不能不核对运算就称为纯相位、无插值的canonical CSA，或标准二维Stolt ω-k原算法。故本补录把经典CSA与Stolt分别追到上表另两个明确计算入口，保留PySimSAR实现差异，而非只依README列名。

**所需输入。** 原始回波应保存复相位、脉冲序列/快时间；参数至少包括载频、chirp rate/带宽与脉宽、采样率、PRF、轨迹或速度、参考斜距和模式。缺参数会导致徙动、相位及几何错误。聚焦后幅度图经取模、压缩和量化后，不能无损倒推出这些信息。

## 3. 第2项：时域反投影BP及可核官方GPU实现

**官方技术入口。** [NVIDIA MatX sar_bp API](https://nvidia.github.io/MatX/api/signalimage/radar/sar_bp.html)与[官方CPHD实例流程](https://github.com/NVIDIA/MatX/blob/0ac86644a82d63e578f695e335fb2d01e1cb06ae/examples/sarbp/README.md)。这是NVIDIA发布的软件实现与技术文档，**不是BP最初提出者的原始代码**。

固定仓库：[NVIDIA/MatX](https://github.com/NVIDIA/MatX/tree/0ac86644a82d63e578f695e335fb2d01e1cb06ae)，commit `0ac86644a82d63e578f695e335fb2d01e1cb06ae`；BSD-3-Clause。已经静态读到三个实际层次：

1. [operators/sar_bp.h](https://github.com/NVIDIA/MatX/blob/0ac86644a82d63e578f695e335fb2d01e1cb06ae/include/matx/operators/sar_bp.h)：`sar_bp(initial_image,range_profiles,platform_positions,voxel_locations,range_to_mcp,params)`；输入有编译期二维complex约束。
2. [transforms/sar_bp.h](https://github.com/NVIDIA/MatX/blob/0ac86644a82d63e578f695e335fb2d01e1cb06ae/include/matx/transforms/sar_bp.h)：`sar_bp_impl`，分配/选择精度及相位LUT并启动kernel。
3. [kernels/sar_bp.cuh](https://github.com/NVIDIA/MatX/blob/0ac86644a82d63e578f695e335fb2d01e1cb06ae/include/matx/kernels/sar_bp.cuh)：`SarBp`、距离/插值权重计算与相位校正，按像素—脉冲累积。不同算术精度与局部Taylor近似属于工程选择，必须同时评估数值误差。

所需输入是 **[脉冲,距离bin]复数距离压缩数据**，每个脉冲相位中心XYZ、成像像素XYZ、至运动补偿点距离、相位参考中心频率与距离bin间隔；坐标和长度单位必须一致。官方示例从CPHD预处理至该输入，不能把普通8-bit图当作range profiles。

可执行状态：实际算子和kernel存在，未编译/运行；需要匹配CUDA/MatX构建及有效CPHD输入。示例里的设备基准是作者特定数据/精度/尺寸的结果，不外推为本项目DGX Spark训练加速。即使数据可公开下载，归档许可也与BSD代码许可分开。本轮未下载CPHD或数据样例。

## 4. 第3项：脉冲压缩/匹配滤波

**原理与官方文档。** 对已知参考波形作共轭时反匹配，等价于接收信号与参考的互相关。官方[SciPy correlate文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)明确给出复共轭定义和matched-filter示例；它是成熟通用信号处理库，不能写成经典雷达算法原作者实现。

主入口：[scipy/scipy](https://github.com/scipy/scipy/tree/c818f76bf5be4aa8418ddcc3f7915de457eb0754)，commit `c818f76bf5be4aa8418ddcc3f7915de457eb0754`，BSD-3-Clause；实际源码[scipy/signal/_signaltools.py](https://github.com/scipy/scipy/blob/c818f76bf5be4aa8418ddcc3f7915de457eb0754/scipy/signal/_signaltools.py)，`correlate(in1,in2,mode='full',method='auto')`。已核到实际direct/FFT分支及共轭、交换输入后反转共轭、full输出零填充处理。雷达使用时接收复数快时间采样作为in1，实际参考chirp或相位码作为in2；不能套用示例中的普通矩形波当作任意传感器发射波形。

项目集成参考：PySimSAR同上MIT commit，[pySimSAR/waveforms/lfm.py](https://github.com/music-sustech/PySimSAR/blob/4864c26af205c3571d1dc2d62a603e62553b9e7f/pySimSAR/waveforms/lfm.py)的`LFMWaveform.generate/range_compress`，读取保存的复chirp，计算echo FFT × conj(reference FFT)后IFFT，支持一维/二维回波。它以echo长度作为FFT长度，使用前必须明确循环相关与线性相关、有效区间及range-gate约定，不能忽略边缘环绕；SciPy的full线性相关可作数值参照。

需要复数接收时序、发射参考、采样率、时间零点/门延迟；距离bin物理标定还需要传播速度与系统时延。仅有聚焦幅度图不具备这些条件。图像滤波可能改变分割特征，但应称图像域滤波，不称物理距离脉压。

## 5. 第4项：MVDR/Capon

**一手技术依据。** [RadarSimPy官方API](https://radarsimx.github.io/radarsimpy/api/process.html)明确说明ULA复协方差、阵元间距以波长为单位及扫描角。原始文献入口：J. Capon，*High-resolution frequency-wavenumber spectrum analysis*，1969，[DOI](https://doi.org/10.1109/PROC.1969.7278)。本轮技术实现依据是官方API及源码；未声称已获取IEEE原始全文。

固定仓库：[radarsimx/radarsimpy](https://github.com/radarsimx/radarsimpy/tree/5a07616212142b673643f40bada49434c5e6c0dd)，commit `5a07616212142b673643f40bada49434c5e6c0dd`；根[LICENSE](https://github.com/radarsimx/radarsimpy/blob/5a07616212142b673643f40bada49434c5e6c0dd/LICENSE)为 **GPL-3.0**。读取[processing.py](https://github.com/radarsimx/radarsimpy/blob/5a07616212142b673643f40bada49434c5e6c0dd/src/radarsimpy/processing.py)的`doa_capon(covmat,spacing=0.5,scanangles=range(-90,91))`：

- 由ULA阵元位置与sin(theta)生成复导向矢量。
- 为协方差增加固定1e-9的对角项，使用伪逆。
- 计算畸变无失真的最小方差权重与输出功率谱；返回角度谱dB。

它是RadarSimPy维护者的后续实现，不能称Capon本人源码。固定极小对角载荷不是对所有数据幅值/样本量都稳健；协方差条件数、阵列标定误差、干扰与训练快拍数量要单独核验。

输入是同时/相干校正后的多阵元复快拍，或在合适局部平稳假设下估计的Hermitian协方差；需要阵列形状、间距/波长及角度约定。当前函数只内建ULA扫描，不能直接套任意二维阵列或SAR平台几何。灰度图的空间协方差可以作为图像统计，但不提供真实阵列相位关系。

## 6. 第5项：MUSIC

**一手技术依据。** 同一[官方API](https://radarsimx.github.io/radarsimpy/api/process.html)和固定源码；原始文献入口：R. O. Schmidt，*Multiple emitter location and signal parameter estimation*，IEEE TAP，1986，[DOI](https://doi.org/10.1109/TAP.1986.1143830)。本轮IEEE页面要求浏览器验证，故不把受限原始全文列为已精读；技术条件由官方实现文档和代码核对。

真实函数：[processing.py](https://github.com/radarsimx/radarsimpy/blob/5a07616212142b673643f40bada49434c5e6c0dd/src/radarsimpy/processing.py)中的`doa_music(covmat,nsig,spacing=0.5,scanangles=range(-90,91))`；同commit `5a07616212142b673643f40bada49434c5e6c0dd`，GPL-3.0。已读到Hermitian eigendecomposition、取噪声子空间、ULA导向矢量、噪声投影范数倒数伪谱以及find_peaks选峰，具有真实计算实现。

要记录一个实现细节：该版本伪谱使用 **1/||E_noiseᴴ a(theta)||**，并用10log10归一化；经典常见功率式为投影范数平方的倒数。两者理想情况下峰位置相关，但谱幅标度不同，不能不核定义就比较伪谱dB数值。这是静态公式差异记录，本轮未做数值测试，不据此判断算法错误或精度。

必须有复数相干协方差、阵列导向模型及合理的nsig（小于阵元数）。相干多径、有限快拍、色噪声、阵列误差可能破坏标准子空间假设；是否需要空间平滑、预白化或其他修正属于另行设计。MUSIC的角度超分辨依赖观测与模型，不能给普通灰度图凭空创造空间分辨率。

## 7. 第6项：稀疏SAR重建与复数稀疏恢复

本项明确区分 **原始成像逆问题** 与 **成像后复数InSAR相位恢复**。二者都是真实雷达稀疏方法，但任务与输入不同。

### 7.1 原始回波反演：SAR测量模型 + 原作者SPGL1求解器

SAR原始论文：[Fang、Xu、Zhang、Hong、Wu，Fast Compressed Sensing SAR Imaging based on Approximated Observation](https://arxiv.org/abs/1302.3120)。该原文记录JSTARS 2014，7(1):352–363，[DOI](https://doi.org/10.1109/JSTARS.2013.2263309)。核心是以聚焦过程的近似逆构造观测模型，在稀疏正则化框架中迭代求解，避免直接巨大稠密测量矩阵。它需要采样掩膜、复数回波、系统参数及相应成像算子；不是从JPEG恢复物理相位。

实际公开通用优化入口是**SPGL1原作者**Ewout van den Berg、Michael P. Friedlander发布的[官方文档](https://friedlander.io/spgl1/)与[mpf/spgl1](https://github.com/mpf/spgl1/tree/361a5980667288857e4f4f84c53b536ddfac1d53)，commit `361a5980667288857e4f4f84c53b536ddfac1d53`，根COPYING为 **LGPL-2.1**。已读：

- [spg_bpdn.m](https://github.com/mpf/spgl1/blob/361a5980667288857e4f4f84c53b536ddfac1d53/spg_bpdn.m)：`spg_bpdn(A,b,sigma,...)`，求解min ||x||₁，约束||Ax-b||₂≤sigma。
- [spgl1.m](https://github.com/mpf/spgl1/blob/361a5980667288857e4f4f84c53b536ddfac1d53/spgl1.m)：实际求解迭代和`Aprod(x,mode)`，支持显式矩阵或正/伴随函数句柄；官方文档明确支持复数域。

**授权与实现关系不能混同。** SPGL1是其求解器作者的公开实现，**不是Fang等SAR论文的作者代码，也不是其fast iterative thresholding的逐步复现**。可用它建立同测量模型下的BPDN对照，但需要另有准确的SAR A与Aᴴ、采样及噪声界。当前本轮未取得Fang论文完整作者源码，故“全SAR成像链已可直接启动”状态是**未完成**；求解器本身有真实公开完整代码。不得用A=I图像阈值处理冒充原始SAR欠采样反演。

### 7.2 雷达专用作者代码：ComCSC/ComCSC-GR相位恢复

原始论文：[Kang、Hong、Liu、Baier、Yokoya、Demir，Learning Convolutional Sparse Coding on Complex Domain for Interferometric Phase Restoration](https://arxiv.org/abs/2003.03440)，[DOI](https://doi.org/10.1109/TNNLS.2020.2979546)。作者仓库[jiankang1991/ComCSC](https://github.com/jiankang1991/ComCSC/tree/6532e451927ac7c7c31a5cb72b2404084b681323)，固定commit `6532e451927ac7c7c31a5cb72b2404084b681323`。

已静态阅读`train_cpx_dic.m`、`sporco-m0.0.9/DictLearn/cbpdndl.m`与`sporco-m0.0.9/SparseCode/cbpdngr.m`。后者真实进行FFT卷积、复共轭、ADMM更新、稀疏收缩和梯度正则化；输入是**带相位的复数干涉图/复字典**。它用于InSAR相位恢复，不是二维船舶实例分割或原始SAR回波聚焦。README明确以SPORCO为基础修改，不能把全部SPORCO代码归为论文作者原始开发。

可执行性与许可必须保留缺口：已读函数调用`Shrinkcomp`，但本commit完整树未找到同名函数文件，也未在已读文件中找到定义；未经独立运行不能保证MATLAB路径上存在外部依赖。仓库没有已核实的根级统一许可证；所携带`sporco-m0.0.9/License`是LANL三条款BSD风格许可，**不据此擅自认定新增修改均有相同授权**。因此状态是“实际SAR稀疏实现已定位，helper及修改许可待核”，不能宣传开箱即用。作为同行研究与源码入口已成立，当前灰度任务不能直接复用其物理输入。

本轮没有为了填表额外列一个未读代码的低秩/RPCA模型；稀疏家族已经覆盖原始反演求解入口和雷达专用复数稀疏代码，同时明确何处尚缺完整链路。

## 8. 检索中剔除的假就绪入口

这些不是新增第七项，而是六项来源审计的反例记录。

- TorchSAR固定commit `05a46610d68bc884743a483565279f361ade5384`（Apache-2.0）中的[matched_filter.py](https://github.com/aisari/torchsar/blob/05a46610d68bc884743a483565279f361ade5384/torchsar/sharing/matched_filter.py)的`chirp_mf_td/chirp_mf_fd`直接抛`TypeError('Not opened yet!')`；[pulse_compression.py](https://github.com/aisari/torchsar/blob/05a46610d68bc884743a483565279f361ade5384/torchsar/sharing/pulse_compression.py)的`mfpc_throwaway`同样是占位，不能当可运行脉压实现。
- 同仓库[complex_image.py](https://github.com/aisari/torchsar/blob/05a46610d68bc884743a483565279f361ade5384/torchsar/sparse/complex_image.py)的`CICISTA.step`使用Y-X及复收缩，未出现SAR测量正/伴随算子；最多是图像域阈值迭代入口，不能冒充原始回波稀疏SAR反演。
- 搜索返回的SAR缩写还可能指specific absorption rate（MRI），STAP也可能指软件追踪或密码学。与合成孔径雷达无关的同名结果已排除；不把搜索命中数当有效文献覆盖率。

## 9. 研究取舍与状态

1. 对当前SSDD/HRSID等灰度实例分割，不接入需要不存在的相干输入的RDA/BP/MVDR/MUSIC/相位恢复。可以用其建模思想形成图像域假设，但命名、监督与贡献都必须落在实际可观测数据上。
2. 若后续确有合法原始复回波和几何，先选择一条独立可验证成像支线。建议从明确输入、许可与实现的MatX BP或基础RDA建立数值参照，报告PSLR/ISLR、聚焦/相位误差与成本；这些指标不能替代实例mask AP。
3. 阵列路线先验证协方差/阵列/信号数假设；稀疏路线先验证A/Aᴴ的伴随一致性与采样模型，再讨论稀疏先验的收益。任何从幅度推测相位或几何的模型输出都必须标为估计，不能当实测真值。
4. 所有固定commit只证明查到相应源码版本；未运行、未下载数据或权重、未复现原论文数值。源码许可不自动覆盖原始雷达数据；本文件不复制第三方全文/源文件，只给链接、原创总结与状态。
5. 本地补录完成；由主任务按既有发布范围同步。此处不声称GitHub已经同步，也不承诺后台训练或持续传输。

