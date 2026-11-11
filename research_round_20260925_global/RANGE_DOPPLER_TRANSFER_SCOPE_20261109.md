# 距离–多普勒图迁移的适用范围与最小验证

现有 HRSID 实测来自已聚焦的空间 SAR 幅度图和逐船轮廓。距离–多普勒（RD）表示的物理坐标和目标监督可能不同。可验证 RF-DETR 特征初始化与掩膜网格对齐规则的可移植性。已验收的 HRSID 增益只覆盖原空间图像配对。 [HRSID 作者数据](https://github.com/chaozhong2010/HRSID) · [原项目配对](HRSID_FIT_RESIZE_PHASE_PAIRED_TRAIN_PROTOCOL_20261109.md)

| 数据含义 | 合适的输出与输入路径 |
| --- | --- |
| SAR 成像中间 RD 数据 | 对具备原复数回波和成像参数的输入，先完成距离徙动校正、方位匹配滤波及逆 FFT，在聚焦空间图像上验证船舶轮廓。仅有 RD 幅度图时，反演所需相位可能缺失。 [Sentinel-1 官方聚焦流程](https://sentiwiki.copernicus.eu/web/s1-processing#slc-processing-algorithms) |
| 海面监视雷达的距离–速度图 | 目标为距离和径向速度格点上的回波峰或回波簇。多个船体可以落在相同或相邻格点，单张 RD 的空间区分度受限。按需要用角度或时间轨迹增加区分信息。 [TI 距离–速度处理](https://software-dl.ti.com/ra-processors/esd/MMWAVE-L-SDK/05_05_00_02/exports/api_guide_xwrL64xx/DOPPLERPROC_PAGE.html) |
| ISAR 的距离–多普勒图 | 若完成运动补偿与横向定标，可将多普勒维转到横距并作为另一成像域验证。成像投影与散射仍有变化，须取得对应实例标注。 [ISAR 横距定标](https://doi.org/10.1109/LGRS.2009.2021990) |

## 第一项公平实验

1. 先确定传感器、处理阶段、距离和速度坐标、相干积累时长、格点分辨率以及可用真值。优先使用原始复数数组或经定标的功率矩阵。固定 FFT、窗函数、零多普勒位置和归一化。图和标签用同一个物理网格变换，并把输出回映射到原格点。HRSID 的中心采样可作为 A/B 选项，实际效用由 RD 对照决定。
2. 若有独立航迹或目标中心，先做回波检测。同一 RD 输入下比较 CFAR、通用初始化的 RD 模型、用 HRSID 主干初始化的同一 RD 模型。模型组保持目标域训练量、检测头、数据和评分一致，按航次或采集序列分组留出，防止近邻帧跨组。冻结现有 SAR 权重的直接推理只有在输出任务一致时才作为独立诊断。
3. 对航迹和中心真值，主报固定虚警水平下的检出率、距离误差和径向速度误差。使用 AIS 时先核对时间同步、自运动补偿和径向投影。只有经独立验证的逐船回波掩膜可以用于 RD mask AP。若要主张中心对齐是可移植的机制，应另在相同 RD 初始化和预算下比较原生及中心对齐，两臂处理同样的监督实例。
4. 若两船的距离和径向速度重叠，单帧 RD 的可分性须单列。已有 RD 检测和多视图语义分割的公开对照可分别参考 DAROD 与 MVRSS。 [DAROD 作者代码](https://github.com/colindecourt/darod) · [MVRSS 原文](https://openaccess.thecvf.com/content/ICCV2021/papers/Ouaknine_Multi-View_Radar_Semantic_Segmentation_ICCV_2021_paper.pdf)

## 已核对的数据入口

| 入口 | 可用于 | 标签与输入边界 |
| --- | --- | --- |
| 全息凝视雷达海上低可观测目标 | 船舶回波检测的候选，含摩托艇、海杂波和配套真值 | 发布页报告 S 波段实测回波及真值 TR，逐实例 RD 掩膜尚未核实。先查说明和最小样本。 [作者数据页](https://www.radars.ac.cn/web/data/getData?dataType=MSLOTD) |
| 海上目标多源观测 MTDSP | 多类船舶回波研究与时空关联 | 作者报告七类船舶、28 组，含 X 波段相参回波、雷达/AIS 航迹、光学及红外。原标签与回波段需要重建对应关系，逐船 RD 掩膜资格待核。 [论文](https://doi.org/10.12000/JR25001) · [作者数据页](https://www.radars.ac.cn/web/data/getData?dataType=DatasetofRadarDetectingSea) |
| St Andrews 海上毫米波数据 | 原始 FMCW 回波处理的可复现候选 | 作者门户列出 CC BY 原始记录，配套论文做多普勒特征分类与异常检测。 [数据](https://doi.org/10.17630/2bb28a35-11a7-4d86-8464-75288e6b66ce) · [原文](https://doi.org/10.1049/rsn2.12518) |
| CARRADA | 跑通 RD 输入、标注变换和代码 | 作者数据为车载雷达与同步相机，含道路目标 RD 标注，任务范围是车、行人和骑行者。 [论文](https://arxiv.org/abs/2005.01456) · [作者代码](https://github.com/valeoai/carrada_dataset) |
| WaterScenes | 海面 4D 雷达点云和相机的候选 | 作者发布雷达点云点级标注和相机像素标注。由点云投影成的距离–速度图保留稀疏检测点，不能替代原始密集 RD 功率和独立逐船掩膜。 [作者数据格式](https://github.com/WaterScenes/WaterScenes#radar-format) |

RD 属于另一个验证问题。已有海面公开入口提供真实回波和检测真值线索，本轮只核对来源。船体分割主线仍用空间 SAR 的 HRSID 与 PSeg-SSDD 验证。

## 2026-11-10 补充核验

### 两个直接相关的公开入口

| 数据 | 已核对 | 第一项用途 |
| --- | --- | --- |
| KuRALS-PD | 海面子集按论文表 4 有 2 条序列、203 帧，含船艇。官方下载页可访问并列出 `KuRALS_PD.zip`，发布说明另提供 FFT 后复数 RD。论文用 GPS 投影、CA-CFAR 与局部峰修正构造位置标注。所核对版本的 `rdDense` 在峰值中心周围生成固定 3 个多普勒格点 × 5 个距离格点的窗口，边界裁剪。已核对入口、原文和生成源码，实际压缩包内容待核。 [论文](https://doi.org/10.3390/rs18010173) · [下载入口](https://github.com/lihua199710/KuRALS#prepare-the-kurals-dataset) · [固定窗口生成源码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L43-L67) | 验证 RD 输入、船艇定位及固定虚警下的检出率。作者像素指标描述对发布标注的拟合。判断真实回波边界需要独立的回波级标注。 |
| OpenSAR Insight 船舶数据 | 15 景 Sentinel-1，发布方说明同一地理区域配有 L0 RAW、距离压缩中间量、L1 SLC 和 GRD，含船位与框 XML。作者报告部分框有偏移或偏小，并列出人工筛选子集。 [发布页](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset) · [标注模式](https://opensarinsightweb.web.uah.es/dark_vessel.html) · [作者模型](https://huggingface.co/opensar-insight/vessel-detection-model) | 用同景配对验证星载 SAR 从聚焦图向成像前表示的迁移。先评估切片级船舶存在，再在独立核对坐标映射后评估定位。成像前数据存在维度和成像几何差异，地理共址只提供关联入口。 |

OpenSAR Insight 固定发布版的三个 XML 压缩包也已核对。训练、验证、测试分别含 2,671、311、1,065 个 XML，对应 10、2、3 个互不重叠的 SAR 产品。船舶行分别为 2,980、340、1,626，具备全局和局部距离压缩框的分别为 2,952、338、1,617。这个核验只支持场景产品隔离。船舶跨场景身份与空标签裁块中的完整目标覆盖待核。距离压缩框可用于同域定位，进入方位 FFT 后需要单独建立 RD 真值。 [固定发布版](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a) · [成像前坐标转换说明](https://github.com/ESA-PhiLab/OpenSARInsight/blob/main/backend/dataset_generation_scripts/SLC_to_FullRaw/README.md#L136)

### 实验决策

- 如果输入是海面监视雷达 RD，第一轮用单通道功率和独立 RD 输出头，保留距离与多普勒原始轴及单位。以同骨干、同头、同数据、同训练预算比较通用预训练和 HRSID 预训练。库中 RF-DETR 若需要方形输入，优先固定原格点分辨率后按距离切片或填充，并记录输出逆映射。CFAR 与作者 KuRALS-Net 在相同留出数据上重评。按完整采集序列留出，两个海面序列只够做初步方向检验。无独立完整真值时，显式记录未核实目标区及背景区，避免将未知目标计为虚警。
- 预训练迁移与掩膜采样机制分开检验。前者主报船艇检出率、每帧或每百万有效格点虚警、距离与径向速度误差。优先对独立 GPS／AIS 坐标评分并预设匹配容差，分别报告对发布标签的指标，因为其生成使用了 CFAR。后者只在有独立 RD 回波支持掩膜时做原生／中心采样配对，固定输入变换、初始化、实例集合及预算。HRSID 的正向结果仅验证空间图像监督。
- 若目标是原论文的空间船体掩膜，同时取得同一采集的复数数据和空间标注后，再比较空间单支与空间加多普勒信息的分支。对齐、模态贡献和计算预算要分别控制。已有 D-MFPN 用复数 SAR 的多普勒参数辅助船舶检测，DAMD-Net 用各向异性 RD 特征做海上目标识别。融合或轴向建模本身已有直接先例。 [D-MFPN 原文](https://doi.org/10.3390/rs15030626) · [DAMD-Net 原文](https://doi.org/10.3390/rs18203509)

本次证据等级为公开文献、发布说明与固定源码核验。RD 域的模型表现待实测。

## 2026-11-10 补充研究

HRSID 的训练中心采样修正空间掩膜与图像的相对像素中心。在两组已关闭的 DEV705 模型上，逐图等量非空候选的 IoU 0.90 匹配净增 98、70。RD 的索引原点取决于雷达处理约定，须重新核对物理轴与格点。 [HRSID 实测](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#关闭模型的候选覆盖与图内分配诊断)

### 零多普勒格点合成核验

均匀慢时间采样、相同 PRF、偶数长度的完整移位频轴，其中心频率为 `f(k) = (k − N/2) × PRF / N`。使用 NumPy 1.26.4，以 PRF 640 Hz 构造 64 点源网格和 32 点目标网格，对 −160、0、+160 Hz 的 64 点理想复数单频分别做矩形窗 FFT，计算归一化功率，再比较两种索引选取。

| 对照 | 目标格点对应源索引 | 目标零频位置读到的源频率 | 三个合成峰归一化功率 |
| --- | --- | ---: | --- |
| 物理频率匹配 | 2j | 0 Hz | 1、1、1 |
| 普通图像中心最近邻 | 2j+1 | +10 Hz | 4.16×10⁻³¹、0、4.16×10⁻³¹ |

目标零多普勒是索引 16。按频率取源索引 32，普通图像中心规则 `floor((16 + 0.5) × 64 / 32)` 取 33。两种索引在全部 32 个目标位置不同。合成峰刻意位于原始格点，功率差无真实雷达外推意义。对已形成谱图取样与改变相干积累时长重新估计谱是不同过程。FFT 网格间距和真实分辨能力分开记录。 [多普勒采样与零填充](https://www.mathworks.com/help/radar/ug/doppler-estimation.html) · [SAR RD 轴标定](https://www.sandia.gov/research/publications/details/determining-the-axes-of-a-range-doppler-image-2025-02-28/)

第一轮保留原生格点，按连续距离窗切片或填充，并在输出和评分中排除填充区域。需要重采样时记录轴原点、坐标间距、零多普勒位置和速度符号，统一映射输入、监督与输出。只有完整、未解模糊的等间距多普勒 FFT 主区间可考虑按 PRF 周期处理边界。裁剪、解模糊或拼接的速度轴应核对边界语义。距离轴保持有限，禁止跨轴旋转。公开 TMVA-Net 的配置与加载入口直接用原生 RD 标签，只列翻转增强。若目标域训练未触发标签缩放，HRSID 的原修正缺少生效入口。 [TMVA-Net 配置](https://github.com/valeoai/MVRSS/blob/368c2c892d8e6076c59cb21fd1056d472887990d/mvrss/config_files/tmvanet.json) · [数据加载](https://github.com/valeoai/MVRSS/blob/368c2c892d8e6076c59cb21fd1056d472887990d/mvrss/loaders/dataloaders.py#L104-L154)

若完整、偶数点且已移位的多普勒轴需要模拟速度取反，可比较原数组翻转、保持零速度的反射及关闭该增强。64 点合成核对了全轴，物理频率反射索引 `(N−j) mod N` 保持零频索引 32，数组反转 `N−1−j` 将其移到 31。该对照只在明确以速度取反解释增强时成立。对多视图模型，输入和标签须统一映射。 [FFT 频率定义](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html) · [MVRSS 翻转实现](https://github.com/valeoai/MVRSS/blob/368c2c892d8e6076c59cb21fd1056d472887990d/mvrss/loaders/dataloaders.py#L249-L276)

### 最小判别实验

| 假设 | 公平对照 | 判据 |
| --- | --- | --- |
| SAR 权重有迁移价值 | 用同一功率输入适配器和新 RD 检测头，比较通用初始化、HRSID 原生骨干和 HRSID 中心对齐骨干。固定目标域样本、顺序、预算和评分。对照同一留出数据上的 CFAR、KuRALS-Net。 | 对独立位置真值比较固定虚警水平的船艇检出率、距离及径向速度误差。按序列整体留出，保留经核实的海杂波负帧。KuRALS-PD 两条海面序列可互换留出做试点。跨采集或跨传感器外推另用独立域。 |
| 标签采样机制可移植 | 若有独立回波掩膜，固定 RD 初始化、输入变换、共同监督实例，比较原生栅格、图像中心规则与物理轴对齐。保留不缩放的原生格点参照。 | 先比较位置偏差和小回波保留，再比较同一原生网格 GT 的掩膜 AP。KuRALS-PD 固定矩形用于弱标签试点。 |

若 RD 来自星载 SAR 成像前阶段，可在 OpenSAR Insight 同景数据上先比较聚焦域与距离压缩域检测，再独立校准方位 FFT 后的定位标签。机载未聚焦 RD 的 CFAR 船舶检测已有直接先例，复数 SAR 距离压缩域的幅相融合也已有报道。若论文保留船体轮廓任务，可在取得同景空间掩膜与复数数据后比较空间单支和空间加多普勒特征，统一按空间实例掩膜评价。 [机载 RD CFAR](https://elib.dlr.de/127354/) · [复数距离压缩域方法](https://doi.org/10.12000/JR25185) · [同景数据发布](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset)

当前证据为公开来源核验和合成格点复算。真实海面 RD 训练、检测指标和跨域结论待实测。

## 首轮海面 RD 试点的样本资格与评分边界

HRSID 三组同预算 E24 配对在固定 DEV705 上的平均 mask AP 为 50.809→55.698，提高 4.888 个百分点。现有干预修改训练掩膜的采样，输入和网络结构沿用同一方案。配对中保留的监督实例仍有差异，共同实例的机制检验待完成。可把两组权重用作 RD 初始化候选，也可在 RD 另行检验物理格点对齐，分别判断两个问题。 [三种子结果](HRSID_FIT_RESIZE_PHASE_REPEAT_SEEDS_PROTOCOL_20261109.md#种子-5203-的固定-dev705-实测及三种子判定)

### KuRALS-PD 海面输入的前置核验

- 作者 Google Drive 目录元数据给出的处理包大小为 5,033,459,295 字节。两次有时限的下载端点探测均超时，当前未读取 ZIP 内容或确认包内样本。已核对发布入口与固定作者源码。 [作者发布说明](https://github.com/lihua199710/KuRALS#prepare-the-kurals-dataset)
- 海面分支按目标 ID 选择波束，用原始幅度做 CFAR 峰修正。生成模型输入时先将零多普勒行置零，对宽、窄脉冲各自按帧归一化，再拼接为 64×800 并移位。代码实际填充距离列 5 至 678（零起点，含端点），其中 119 至 136 为两路混合区。正式评分须先核对发布数组与代码是否一致，再排除拼接外的填充。经目标条件选波束的结果用于所选波束内的定位评估。近零速度单列，验证保留零多普勒的变体须从源复数 RD 构造。 [选波束、置零与拼接](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L282-L366)
- 位置字段取自 `tgtDistance`、`tgtVelocity`，其独立性要追溯到同步 GPS 或其他外部参考。发布论文的矩形尺寸描述与当前固定源码的 3×5 生成函数有出入，实际归档的标注尺寸要读回样本确认。两条海面采集路径同日不同时段，船艇身份待核。 [论文第 3.4 节与表 4](https://doi.org/10.3390/rs18010173) · [标注字段](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L403-L446) · [采集路径](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L462-L482)

### 首轮执行顺序

1. 优先取得最小海面样本及说明，核对原生数组、轴起点与间隔、有效格点、目标标注谱系、采集分组和已验证海杂波。只具备发布修正峰时，先报相对于该峰的定位结果。固定虚警下的检出率需要可信负区和独立匹配真值，整域监视还需要未按目标筛选的采集。
2. 使用同一功率或幅度适配器、同一新 RD 中心检测头，三组只替换骨干初值。初值分别为通用预训练、HRSID 原生监督、HRSID 中心对齐监督。首轮在原生格点按距离切片或填充，统一目标域数据、更新预算与评价规则。CFAR 与作者基线使用同一留出输入及评分范围。整条采集序列互换留出，匹配容差和阈值用训练侧数据或物理分辨率预先冻结。有独立真值时报告一对一检出率、每帧虚警、距离与径向速度误差。确认跨传感器泛化需加入独立采集。
3. 只有取得独立回波支持掩膜才推进同初值、同实例的三组分割对照，比较原生格点、普通图像中心采样、基于距离与多普勒坐标的对齐。若只有可靠中心，可先检验连续物理坐标到格点的系统偏差和定位误差。跨分辨率的对照优先来自已标定采集，合成结果单列。
4. 独立海面补充候选为 MTDSP 的回波和 AIS，先核对同步。LSS-PR-1.0 发布 12 组快艇、11 组客轮和 2 组海杂波，目标回波围绕已知距离裁取，可补充目标窗口内的定位或海杂波评价。其被动双基地距离与多普勒应按采集几何解释。 [MTDSP 数据论文](https://doi.org/10.12000/JR25001) · [LSS-PR-1.0 数据论文](https://doi.org/10.12000/JR24145) · [发布数据](https://www.scidb.cn/detail?dataSetId=40321d5b84664e09b798e0e0ee078fe3)

若目标仍是输出空间船体轮廓，优先采用同景配准后由聚焦 SAR 提供空间位置、由成像前数据提供补充特征的设计，统一在空间掩膜上比较空间单支与融合支。OpenSAR Insight 提供同景关联和检测框入口，方位 FFT 后的真值和逐船空间轮廓需要另外核验。 [作者数据处理与模型](https://github.com/ESA-PhiLab/OpenSARInsight)

本节新增证据为发布元数据、固定源码和原始论文核对。真实海面 RD 训练尚未执行。

## 复数训练样本读回与迁移选型

对 OpenSAR Insight 固定发布版，只读取训练归档的有限字节。在距离压缩重标定包的首个 6 MiB 内完整读回训练块 1778 的 VH 数组，形状为 634×512，数据类型为 complex64，全部值有限。训练标注包中存在一个同编号 XML。聚焦和 RAW 训练归档各读取首个 2 MiB，分别确认 TIFF 与 DAT 成员头。仅复数距离压缩数组完成了完整成员读回，三个前缀读出的首成员并非同一地理块。 [固定数据版本](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a)

作者的处理源码以方位线、距离采样排列距离压缩数组。按第 0 轴对上述样本做正交归一化 FFT 和频轴移位，生成的 634×512 复数频谱数值均有限，零频索引为 317，相对能量差为 5.58×10⁻¹⁴，逆变换相对误差为 6.07×10⁻¹⁶。本项证明公开训练样本可接入方位 FFT。尚未校准该块的 PRF、TOPS 波束扫描与多普勒中心，也未把 XML 中的距离压缩框映射到频域。数据保持在内存，记录只保留格式与数值核验。 [固定作者处理源码](https://github.com/ESA-PhiLab/OpenSARInsight/blob/192ded687391b536180af29d59c07ace9cb0344f/backend/dataset_generation_scripts/l0_to_range_compressed/l0_to_range_compressed/scripts/range_compressed.py) · [Sentinel-1 官方处理流程](https://sentiwiki.copernicus.eu/web/s1-processing)

RCShip-1.0 可用于距离压缩域的框检测对照。其原论文采用方位时间域，公开后续研究也说明测试样本由 Sentinel-1 场景逆处理重建。真正的 SAR RD 检测已有 DLR 在实测机载数据上的 CFAR 与 Faster R-CNN，星载原始回波的船位预测也已有 U-Net 先例。比较时先区分采集域、轴定义与真值语义。 [RCShip 原文](https://doi.org/10.1109/LGRS.2024.3366749) · [FM-DETR 原文](https://doi.org/10.3390/rs18213669) · [DLR CFAR](https://elib.dlr.de/127354/) · [DLR Faster R-CNN](https://elib.dlr.de/189078/) · [ESA 原始回波方法](https://doi.org/10.1109/JSTARS.2024.3399021)

- **独立海面监视 RD**。多尺度编码和 UNet++ 嵌套跳连可以作为待验证的骨架选项。原方案的船体长条方向卷积与依赖 SAR 教师的残差掩膜头分别对应空间形状与原候选，优先从新定位头开始。固定目标域输入、样本、更新预算和评估，分别比较通用、HRSID 原生、HRSID 中心对齐的骨干初始化，并报告额外的源域训练。先验效用在完整序列留出上用固定虚警下的检出率和距离、径向速度误差验证。原生频谱须保留有效格点及近零速度目标的评估机会。
- **沿用空间船体掩膜任务**。把同次采集的局部多普勒特征配准回聚焦 SAR 的候选区域，重点检验独立定义的运动失焦场景。比较空间单支、常规物理重聚焦后接同一分割器、真实配对融合、同条件错配且同容量的融合。固定空间真值与评估规则，先验证输入配准，后比较整集和预先定义失焦组的 mask AP75/AP90 与虚警。OpenSAR Insight 现成标签支持船位或框核验。逐船轮廓以及重聚焦所需元数据需另行落实。多普勒参数融合和重聚焦已有先例，研究问题是经校准的同景信息对空间掩膜边界的独立增益。 [D-MFPN](https://doi.org/10.3390/rs15030626) · [SLC 重聚焦](https://doi.org/10.1049/joe.2019.0447)

本次更新的证据包括公开训练样本读回、单样本方位 FFT 数值核验及方法对照。真实 RD 模型训练与跨域效用待实测。

## 距离–多普勒输入的任务定义与新增核对

- **星载 SAR 成像中间域。** 横纵轴由斜距和方位频率构成。常规聚焦可在相干复数和成像参数可用时恢复空间坐标。仅保留功率时，方位 FFT 的相位平移信息会丢失。一次可复算的理想离散试验用 64×16 的复数阵列，把一个 4×3 非零块沿方位移动 13 格，两个空间支撑相差 24 格，而归一化方位 FFT 功率的最大差异相对峰值为 4.26×10⁻¹⁶。这个试验说明在所设模型下存在位置歧义。实测雷达的点扩散、波束和运动误差另需验证。 [Sentinel-1 聚焦流程](https://sentiwiki.copernicus.eu/web/s1-processing)
- **同景配对数据。** Maya4 公开 Sentinel‑1 Stripmap 的原始、距离压缩、距离徙动校正和聚焦四个阶段，可作为处理链与配准对照。应检查存储数组的坐标约定，或从距离压缩的方位时间序列构造方位 FFT。尚未核实 Maya4 提供逐船轮廓。OpenSAR Insight 的公开训练复数块已通过一次方位 FFT 数值核验，其发布标签目前支持船位或框试点。先做同景配对与复现常规聚焦，再决定能否验证轮廓。 [Maya4 论文](https://elib.dlr.de/227503/) · [Maya4 数据与代码](https://github.com/sirbastiano/Maya4) · [OpenSAR Insight](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset)
- **面向论文的主试验。** 任务定义为利用同次采集的 RD 信息改善聚焦域内的空间船体掩膜。先用原有空间模型产生候选，再在其对应的回波窗口提取经标定的多普勒特征，交给原掩膜头细化。比较空间单支、常规物理重聚焦接同一头、真实配对融合及同条件错配的等容量融合。按采集场景隔离，预先定义失焦子组，固定空间真值、匹配规则和预算，报告整体与子组的 AP75/AP90。D-MFPN 已验证空间和多普勒信息融合的船舶检测。当前待检验的是其对失焦船体实例边界的独立增益。 [D-MFPN 原文](https://doi.org/10.3390/rs15030626)
- **独立海面监视 RD。** 用距离和径向速度定位船艇，比较 CFAR 及同一定位头的通用初始化、HRSID 原生初始化和 HRSID 中心对齐初始化。按整序列留出，以独立船位或航迹报告固定虚警下的检出率和距离、速度误差。KuRALS-PD 的两个海面序列可做小试，目标条件选波束和 GPS/CFAR 构造的矩形标签限制其用途。可用 MTDSP 的独立航迹进一步核对。训练新头或用目标标签调阈值属于目标域适配。真实 RD 精度仍待实测。 [KuRALS](https://doi.org/10.3390/rs18010173) · [MTDSP](https://doi.org/10.12000/JR25001)

## 产品辨认与论文试验的进入条件

先读第二根轴及其单位，再决定迁移任务。NISAR 的 L1 RSLC 虽以 Range Doppler 命名，官方定义的栅格是斜距×零多普勒方位时间，仍是已聚焦的空间复图。真正的成像中间谱第二轴是方位多普勒频率，海面监视雷达 RD 的第二轴通常用频率或径向速度表示。三者各自的标签和迁移结论需要分开。ISAR 若具备运动补偿及横距定标，可单独作为成像域适配候选。 [NISAR 官方样例与规格](https://science.nasa.gov/mission/nisar/sample-data/) · [NISAR 栅格说明](https://nisar-docs.asf.alaska.edu/rslc/) · [Sentinel-1 产品流程](https://sentiwiki.copernicus.eu/web/s1-processing) · [ISAR 横距定标](https://doi.org/10.1109/LGRS.2009.2021990)

最小实现可共用多尺度骨干。聚焦空间输入保留原掩膜头，海面监视 RD 用原生轴适配和新的距离、速度定位头。若实际 RD 监督仍在原生格点且流程无需缩放，先验的掩膜重采样干预没有直接作用入口。源域骨干初始化收益与 RD 目标域的物理格点对齐收益分别设对照。相同分组和训练预算下，分别比较通用、HRSID 原生、HRSID 中心对齐骨干初始化，和原生格点与物理对齐消融。固定射频处理、目标真值、强基线和评分范围。源域 HRSID 的 +4.888 点与冻结 SSDD DEV187 的 +0.376 点均属空间掩膜实测。后者未达到预设支持门槛。 [HRSID 重复种子](HRSID_FIT_RESIZE_PHASE_REPEAT_SEEDS_PROTOCOL_20261109.md) · [SSDD 冻结结果](HRSID_TO_SSDD_FROZEN_DEV187_PROTOCOL_20261110.md#六臂实际评分及独立复算)

原论文优先保留空间船体轮廓任务。先用 OpenSAR Insight 的框和成像参数试验空间候选到局部回波窗口的配准。候选在所有比较臂共用，窗口由候选与传感器几何决定，目标真值只进入训练监督和独立评分。先在按采集隔离的验证侧核对配准误差、时间和频率标定，再决定是否进入分割比较。分割阶段需要同景复数数据、原生空间实例掩膜和足够多独立场景。最小模型是空间候选加局部多普勒特征，在原空间格点细化轮廓。对照包括空间单支、常规物理重聚焦、真实配对融合和同容量错配融合。错配控制在相近传感器、距离与杂波条件下打乱配对，排除同一实例及重叠窗口。固定原空间候选与分数的几何复算可进一步区分轮廓改进和候选变化。以整集及事先定义的运动失焦子组的 AP75/AP90、匹配轮廓质量及代价判断效用。已有多普勒融合和未聚焦 RD 检测先例，待检验的新增主张是同景回波信息对空间轮廓边界的独立收益。 [OpenSAR Insight 发布](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a) · [D-MFPN](https://doi.org/10.3390/rs15030626) · [DLR 未聚焦 RD 检测](https://elib.dlr.de/189078/)

海面 RD 的独立探索优先用 KuRALS-PD 验证数据接入与定位，并以独立同步航迹数据补充检验。只有经核对的背景与独立参考位置才支持固定虚警下的检出率，近零速度目标和同格目标分列。发布矩形只支持相应弱标签的性能表述。已有一份 OpenSAR Insight 训练复数数组通过 FFT 数值核验。RD 模型训练、真实定位收益与空间分割收益均待实测。 [KuRALS 原文](https://doi.org/10.3390/rs18010173) · [MTDSP](https://doi.org/10.12000/JR25001)

## 同源多普勒分支的信号来源对照

同一聚焦幅度图的频谱可以改变网络的表示和归纳偏置。由于频谱由该幅度图确定，它没有增加输入观测。利用同景复数数据时，仍应记录频域分支来自聚焦 SLC 还是成像前复数回波，以及它保留的相位、慢时间或方位频率信息。先验主张是这些信息有助于识别运动失焦下的轮廓误差，效用与机理均待实验。使用同次采集的两种表示时，记为同源多表示融合。 [Sentinel-1 聚焦流程](https://sentiwiki.copernicus.eu/web/s1-processing) · [多普勒特征船舶检测先例](https://www.mdpi.com/2072-4292/15/3/626)

先用已核对的空间候选关联局部回波，在统一空间网格上比较空间单支、常规重聚焦、真实配对融合和同容量错配融合。若真实配对有收益，再加等容量的幅度图派生频谱对照，并在同景复数 SLC 可用时加复数空间分支。两个补充对照分别检验空间幅度的重新表达和复数信息本身能否解释收益。输入谱源、配准容差、候选、失焦分组和评价规则在留出场景评分前固定。OpenSAR Insight 的公开框可用来做关联试点，独立实例掩膜及同景元数据核验通过后才进入轮廓实验。Copernicus 另发布 65 个未做星上 FDBAQ 压缩的 Sentinel-1C/D L0 产品，部分含船，适合作为工程数据候选。该发布需 CDSE S3 访问，逐船标签资格待核。 [OpenSAR Insight 数据](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset) · [Copernicus 数据公告](https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-1-sar-dataset-acquired-without-onboard-fdbaq-compression)

KuRALS-PD 报告的 203 帧是海面场景总数，需从归档核对船艇帧、非船帧及目标 ID。作者代码还包含海面 UAV 路径。海面分支按目标选择波束，两条海面序列只作所选波束内的定位试点。要衡量全域检出率与虚警，需具备可独立判断漏检及虚警的未筛选场景和真值。 [KuRALS 原文](https://www.mdpi.com/2072-4292/18/1/173) · [作者处理源码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py)

## 2026-11-11 有限训练样本读回与直接先例

固定 OpenSAR Insight 的公开发布版本。训练侧实际 HTTP 206 读回聚焦归档头部 24 MiB、距离压缩归档头部 12 MiB 和完整训练 XML 归档 466,628 字节。前两个 TAR.ZST 前缀只能枚举其完整成员。聚焦归档前缀含 196 个 GRD 成员，覆盖 98 个块编号。距离压缩前缀含同一 SAR 产品的 6 个 VH complex64 块，均有同编号训练 XML，6 个 XML 均未列船舶。其中最早一块包含前轮试点样本。两个归档的已读块编号交集为 0，未取得可供配准的完整阳性同景对。缺少船舶条目的 XML 也只表示发布侧未列标签，不能充当经核验的负样本。此次仅限训练侧，未读验证或测试归档。 [发布数据](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a) · [安全汇总](results/OPEN_SAR_RD_BOUNDED_TRAIN_INTAKE_ACTUAL_SAFE_20261111.json)

按作者的数组方向，对 6 个块各做第 0 轴的正交归一化 FFT、移位和逆变换。全部输入和输出有限，最大相对能量差 9.04×10⁻¹⁴，最大相对逆变换误差 6.08×10⁻¹⁶。数值结果仅支持输入可读及 FFT 实现，尚未核准 PRF、方位频率原点、目标回波关联或任何模型效用。六块来自同一产品，不能作为六次独立采集。 [作者处理流程](https://github.com/ESA-PhiLab/OpenSARInsight/tree/192ded687391b536180af29d59c07ace9cb0344f/backend/dataset_generation_scripts/l0_to_range_compressed)

Qiang 等于 2026 年已从候选区提取局部多普勒信息，联合聚焦强度在运动失焦条件下做船舶框检测，并在 50 景真实 GF-3 场景上评估。这里的待检验主张应进一步限定为**在真实留出场景上，经标定的同景局部回波能否改善固定候选的空间实例轮廓**。对同一批候选比较聚焦单支、物理重聚焦接同一掩膜头、等容量错配及真实配对。真实配对首先用同编号船舶阳性块确认原始产品、极化、坐标方向、距离门和方位窗口。可用轻量残差投影把局部多普勒特征接到原掩膜头，初始状态等价于聚焦单支。若配准或独立掩膜资格未通过，结论停留在输入和检测定位。 [Qiang 等，2026](https://doi.org/10.3390/rs18081166)

独立海面监视 RD 优先做回波定位。复用骨干，输入为保留原生距离与速度格点的对数功率，输出为新的回波定位头。用同一头和训练预算比较通用、HRSID 原生和 HRSID 中心对齐初始化，再以 CFAR 作为同场物理基线。用 KuRALS-PD 做两条海面序列的工程小试，用具备同步航迹和可核验背景的采集进行更完整的检出率及虚警评估。若要求零样本，须预先固定输入映射、权重及后处理，任何使用目标标注的拟合和阈值选择单独记为适配。 [KuRALS 原文](https://doi.org/10.3390/rs18010173) · [MTDSP](https://doi.org/10.12000/JR25001)

## 固定一对阳性候选

以作者固定版本人工筛选清单的训练项连接已发布的训练 XML。203 个唯一 VH 筛选块均有对应 XML。首个满足船舶阳性、L1 与 L0 burst 字段及两域位置要求的候选为 TRAIN 395，VH。该 XML 记录 1 艘船，具备聚焦域位置和框、距离压缩域全局与局部框。候选确认只依据训练元数据。两个 TRAIN 归档的 9 字节尾部均由 HTTP 206 实测，未发现标准 Zstd 随机访问索引标记，训练目录也未公开成员索引。此核验未排除其他独立帧结构或发布方的外部索引。 [人工筛选表](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/blob/e90a0c0465cfb8484b4f356ef968d976054f453a/yolo_dataset_filtered.csv) · [Zstd 格式](https://github.com/facebook/zstd/blob/dev/contrib/seekable_format/zstd_seekable_compression_format.md)

后续优先使用官方成员接口、索引或小包，只取同编号 SLC-VH 与复数距离压缩 VH 数组。须匹配产品、burst、极化、轴顺序和框有效范围。聚焦归档中的 GRD 与复数 SLC 分开识别。距压框仍属于方位时间域。已核对候选尚无两幅配对数组，未读取 DEV 或 TEST。此分支的资源优先级低于 HRSID 同监督实例的主实验及 SSDD 验证。

## 先判定迁移的对象，再选择基准

独立海面监视 RD 的问题是“空间 SAR 学到的特征能否帮助定位船舶回波”。最小模型采用同一 RD 适配器、共享的多尺度主干结构和重新初始化的距离–速度定位头。三组分别使用通用预训练、HRSID 原生监督及 HRSID 中心对齐监督的骨干权重。每组共用目标域训练样本、头初始化、训练步数和验证阈值选择规则，按完整采集序列留出。若首轮发现收益，再预设低标注和完整标注两个预算。只在低标注或早期检查点观察到的差异，表述为该条件下的样本效率或训练效率证据。独立采集及重复训练支持的结果才用于更广的迁移表述。船载高频地波雷达已有改进 Mask R-CNN 在实测 RD 上做检测和回波实例分割的先例。KuRALS-PD 适合公开定位试点，其发布掩膜需按弱标签解释。 [船载地波雷达原文](https://www.ams-journal.org.cn/website/cn/article/doi/10.12362/j.issn.1671-6647.20231226001) · [KuRALS 原文](https://www.mdpi.com/2072-4292/18/1/173)

面向现有船体实例分割论文，更直接的问题是“同景局部多普勒能否修正失焦船的空间边界”。OpenSAR Insight 可用于原始距离压缩复数与聚焦图的同景配准试点。FAIR-CSAR V1.0 另提供 GF-3 L1 SLC 及包括 PRF 在内的成像元数据，适合试验从聚焦复数图逆处理提取局部多普勒特征。需把该输入记录为 SLC 派生表示，并保留直接使用 SLC 复数的对照。FAIR-CSAR 的标准监督为旋转框，可用于检测或坐标核验。分割评价需同景、独立的船体轮廓。已有工作在 FAIR-CSAR 实景上利用局部时频与多普勒信息改善框检测，因此实验应固定空间候选，验证真实配对相对于物理重聚焦及等容量错配对照的高 IoU 轮廓增益。 [FAIR-CSAR 作者发布](https://doi.org/10.57760/sciencedb.radars.00019) · [基准论文](https://doi.org/10.1109/TGRS.2024.3519891) · [真实 GF-3 框检测先例](https://doi.org/10.3390/rs18081166)

可复用的实现接点是 RF-DETR 现有掩膜头的最终查询投影。对每个固定空间候选，从经几何标定的回波窗口提取特征，只在该候选的掩膜查询向量上加一个零初始化投影。初始模型与空间单支等价，候选 ID、类别分数和空间特征图保持相同。真实配对与错配组使用同一分支容量，另以重聚焦结果检验常规物理处理的解释力。该接法仅完成静态定位，属于待检验设计。 [现有查询投影](code/HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py#L29-L36)

当前已核对复数训练块和 FFT 数值一致性。正样本同景配对、RD 模型训练和两条路线的实际性能仍待实测。

## 2026-11-11 模型接入与独立验证的边界

**独立 RD 回波定位。** 先确认第二轴是频率或径向速度，再记录格点原点、符号、间距、处理窗和数据是复数、功率还是已转分贝。首轮保留原生网格，填充区域从损失和评分排除。沿用 RF-DETR 多尺度骨干，使用相同的单通道功率适配与新定位头，在目标域用同样的训练数据、步数和阈值规则比较通用预训练、HRSID 原生监督和 HRSID 中心监督三个初始化。后两组之间的差值用于检验中心监督是否给特征迁移带来增量。两组初始化都来自空间分割。若 RD 输入没有掩膜缩放步骤，空间标签采样修正没有直接的干预入口。调过的 UNet++ 嵌套跳连可以作为另一套骨干试验，方向卷积需另设同容量、同预算消融。 [HRSID 配对](HRSID_FIT_RESIZE_PHASE_REPEAT_SEEDS_PROTOCOL_20261109.md) · [UNet++ 结构记录](U57_CURRENT_PROGRESS_AND_METHOD_EXPLANATION_20261024.md)

**KuRALS-PD 先做受控定位。** 论文报告的 203 帧覆盖两条海面采集序列。作者扫描处理通过目标字段选择波束，原始目录映射中船和无人机的派生序列可来自同一采集目录。按原采集目录分组，避免把同一采集拆到训练与验证。发布版仅用于目标波束已选条件下的定位对照，固定窗口标签只用于评估对弱标签的拟合。若要检验独立海面搜索，先取得由固定规则选取或完整扫描的波束及可核验的负帧，统一给 CFAR 和学习模型。用同步航迹的距离、径向速度及预定容差独立匹配，在相同评估区域上报告检出率、每帧虚警和定位误差。少数采集的互换留出用于试点。 [KuRALS 表 4](https://www.mdpi.com/2072-4292/18/1/173) · [扫描波束选择](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L258-L273) · [采集目录映射](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L421-L441)

**保留空间船体轮廓的论文路线。** 优先核对 OpenSAR Insight 同编号阳性训练对的产品、burst、极化、物理轴和候选窗口的映射。具备独立空间轮廓后，再用固定候选比较空间单支、物理重聚焦、等容量错配和真实配对融合。RD 特征只修改原空间掩膜查询，评价固定候选的轮廓质量与留出采集上的 AP75／AP90。此前完成的是六个复数块的 FFT 数值核验。尚无 RD 模型训练、零样本泛化或定位与轮廓增益结果。 [公开配对数据说明](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset) · [已核验范围](results/OPEN_SAR_RD_BOUNDED_TRAIN_INTAKE_ACTUAL_SAFE_20261111.json) · [运动失焦船舶框检测先例](https://www.mdpi.com/2072-4292/18/8/1166)

## 海面实测输入与首轮迁移对照

把 RD 按距离与频率或径向速度解释。若来自聚焦后的 NISAR RSLC，其第二轴为零多普勒方位时间，仍按空间复图处理。真正的海面监视 RD 测到的是相对径向速度。在实际分辨单元内相互重叠的回波可能无法从单帧功率图分开。单个 FFT 格点与实际分辨单元需要分别核对。近零速度、速度折叠和疑似多目标重叠在评分前标记。径向参考速度由同步航迹与雷达相对运动投影得到。若存在模糊，按传感器配置单列或先独立解模糊。 [TI 雷达处理说明](https://www.ti.com/lit/an/swra553a/swra553a.pdf) · [NISAR 栅格定义](https://nisar-docs.asf.alaska.edu/rslc/)

海面 RD 的第一项学习对照优先用 HRSID 种子 5202 已验收的固定共同监督配对作为来源。原生和中心两份权重的源监督实例、训练预算相同，适合检验中心监督是否改善可迁移特征。加同版本通用预训练组和同输入的 CFAR，三组学习模型共用原生格点对数功率、初始化相同的 RD 定位头、训练样本、预算和阈值选择规则。按整次采集分组，能确认的同船重复采集及重叠窗口放在同组。匹配容差在评分前由实际分辨率及航迹误差确定，用独立位置一对一匹配。主报固定虚警水平下的检出率、距离及径向速度误差。目标标注用于训练定位头或校准阈值时，结果记为适配。KuRALS-PD 的目标条件选波束仅支持受控定位。独立监视须核对同一波束规则及背景真值。 [HRSID 共同行实测](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测) · [KuRALS](https://doi.org/10.3390/rs18010173) · [MTDSP](https://doi.org/10.12000/JR25001)

若原论文继续输出空间船体轮廓，优先验证同景多普勒对运动失焦的作用。在同一批空间候选上使用按候选与几何选取的局部复数窗口，把多普勒特征接到现有 RF-DETR 掩膜查询的零初始化残差投影。用空间单支、物理重聚焦接同一头、同容量错配和真实配对四组，在独立逐船空间轮廓上比较高 IoU 的 AP 与边界质量。已有方法在同类输入上做船舶框检测。这个设计的待验证问题是同景信号能否改善固定候选的空间轮廓。公开训练块的方位 FFT 已核验。配对阳性数组、独立空间轮廓和真实 RD 模型分数仍待取得。 [查询投影入口](code/HRSID_FIT_NATIVE_POINT_UPDATE_20261110/cached_final_query.py#L29-L36) · [直接检测先例](https://www.mdpi.com/2072-4292/18/8/1166)


St Andrews 公开了十份原始 FMCW 文件。对三份附件的 4 KiB 前缀探测均返回 HTTP 403，未读取雷达数据。最小附件为水池海狮试验，最小海试附件对应海杂波与静止礁石。论文给出采集参数，当前仍缺少可核对的二进制格式与文件级船舶标签。此入口暂作为补充。 [公开文件](https://doi.org/10.17630/2bb28a35-11a7-4d86-8464-75288e6b66ce) · [作者论文](https://doi.org/10.1049/rsn2.12518)

### 岸基真实回波的输入核验

从 McMaster 官方目录取得 Dartmouth 第 18 号 CDF（15,732,284 字节）。读取首个 512 样本的 HH 复数窗，按文件的单极化 PRF 1,000 Hz 做慢时间 FFT，得到 512×14 的频率×距离矩阵。零频索引为 256，频轴为 −500 至 498.046875 Hz，频率采样间隔为 1.953125 Hz。距离门近边为 2,574 至 2,769 m，名义采样间距为 15 m，作者给出的距离分辨率为 30 m。矩形窗原始 IQ 的逐门能量最大相对误差为 9.73×10⁻¹⁶。以复指数矩阵直接计算 DFT，矩形窗和去均值 Hann 窗的总谱能量与 FFT 分别相差 6.66×10⁻¹⁶ 和 2.22×10⁻¹⁶，频轴和第 9 门峰索引一致。 [作者样本及使用条件](https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html) · [作者读取说明](https://soma.ece.mcmaster.ca/ipix/dartmouth/cdfhowto.html) · [执行脚本](code/IPIX_DARTMOUTH_RD_AXIS_20261111/ipix_dartmouth_18_hh.py) · [结果摘要](results/IPIX_DARTMOUTH_RD_AXIS_ACTUAL_SAFE_20261111.json)

目录第 9 门的目标为直径 1 m、包覆金属网的泡沫浮球。当前核验覆盖一个岸基采集窗口和真实海杂波，幅度标定、径向速度符号、独立速度真值、船舶标签和模型表现未验证。可按官方条件获取 CDF 与 `ipixload.m`，向脚本传入 `--cdf`、`--official-loader` 和新的 `--output` 路径。执行环境为 NumPy 1.26.4 与 SciPy 1.15.3。

### 阳性同景配对的获取路径

从固定发布版的训练 XML 核对 395 号候选对应 Sentinel‑1A L1 SLC 产品 `S1A_IW_SLC__1SDV_20200407T052956_20200407T053026_032017_03B2E2_18FA`、IW2、VH、L1 burst 9。作者转换器明确以零起点编号，ASF 单 burst 接口也使用零起点索引。官方 CDSE 目录已核对同名母产品在线。按 XML 的 burst 字段曾在 ASF 查到 `S1_202951_IW2_20200407T053022_VH_18FA-BURST`。后续足迹核验排除了这一候选，见[阳性同景数组读回](#train-395-阳性同景数组的实测读回)。第 5 burst 是后续地理足迹支持的目录候选，仍需核对官方有效行列与地理定位栅格。作者场景表第 2 景把该 SLC 映射到确切的 L0 RAW 产品 `S1A_IW_RAW__0SDV_20200407T052956_20200407T053029_032017_03B2E2_4F42`。ASF RAW 目录独立确认 ZIP 大小 1,642,382,079 字节。训练 XML 的 L0 burst 字段为 5。下一步可从作者发布成员取得对应距离压缩块，或按固定处理链从该 L0 产品提取 VH 回波。这一轮目录核验仅覆盖元数据。后续已取得作者训练归档的同编号配对数组，独立几何配准仍待实测。 [固定训练标签目录](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a/train) · [作者零起点定义](https://github.com/ESA-PhiLab/OpenSARInsight/blob/192ded687391b536180af29d59c07ace9cb0344f/backend/dataset_generation_scripts/L0_preparation/main/slc2raw-main/slc2raw/raw_data_params.py#L44) · [ASF 接口](https://sentinel1-burst-documentation.asf.alaska.edu/) · [ASF 精确目录行](https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_202951_IW2_20200407T053022_VH_18FA-BURST&output=geojson) · [CDSE 目录记录](https://catalogue.dataspace.copernicus.eu/odata/v1/Products(884ad94b-1595-5b26-a50f-7e97afd39ab9)) · [作者场景映射](https://github.com/ESA-PhiLab/OpenSARInsight/blob/192ded687391b536180af29d59c07ace9cb0344f/backend/dataset_generation_scripts/DVD_dataset_generation/scene_mapping.csv#L3) · [ASF RAW 目录](https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1A_IW_RAW__0SDV_20200407T052956_20200407T053029_032017_03B2E2_4F42&output=geojson)

## 被动海面 RD 的补充数据与迁移判定

《雷达学报》的 LSS-PR-1.0 使用说明确认，公开入口含实测复数 RD 和配套点航迹。其中快艇 12 组、客轮 11 组，另有高低海况海杂波各一组。船舶片段依照已知目标距离门截取，适合测已选门内的回波定位或识别，以及两组指定海况的杂波响应。由于采集选择条件和背景范围有限，这些片段的虚警率只对应被评估的窗口。使用说明要求注册、验证下载及科研引用。此次核对了作者说明，数组未实测。被动雷达位置与频率按收发站几何解释。与单站主动雷达比较时，先核实轨迹投影、距离定义和多普勒轴，按传感器分别报告误差。 [数据入口](https://doi.org/10.57760/sciencedb.14468) · [数据论文](https://doi.org/10.12000/JR24145) · [作者使用说明](https://www.radars.ac.cn/fileLDXB/cms/news/info/2024/11/c23b26b747dd4d12b2673ceb6bb93a4e/%E2%80%9C%E8%A2%AB%E5%8A%A8%E9%9B%B7%E8%BE%BE%E4%BD%8E%E6%85%A2%E5%B0%8F%E6%8E%A2%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%88LSS-PR-1.0%EF%BC%89%E2%80%9D%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.pdf)

首轮同传感器实验优先验证特征初始化。对 RF-DETR 使用相同定位头和同一物理格点，比较通用初始化、HRSID 原生监督权重和 HRSID 固定共同实例的中心对齐权重，同时对照 CFAR 与适用的作者模型。原生格点首轮不缩放，保持距离与多普勒轴。输出头在目标域训练后的结果记为迁移适配。若仅在有标签微调后观察到收益，结论限定于对应传感器、任务及标注预算。只有输出含义兼容且目标标签未参与适配和阈值选择时，才单独记录零样本结果。首轮按原始采集分组，并保留近零多普勒和距离–多普勒重叠样本的单独诊断。 [HRSID 固定共同实例实测](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测) · [KuRALS 原文](https://doi.org/10.3390/rs18010173)

如果任务仍是船体空间轮廓，近期验证优先级是取得一对阳性同景复数回波与聚焦图，实测映射精度并审查独立轮廓可用性。接入测试固定空间候选和分数，比较空间单支、常规重聚焦、同容量同条件错配与真实配对。先看对运动失焦候选的固定匹配 IoU 与边界误差，再看独立采集的 AP75 和 AP90。若真实配对的收益不能超过重聚焦和错配对照，证据不足以支持新增多普勒分支。OpenSAR Insight 已发布的框可用于配准检查和定位试点。论文的空间实例分割结论仍依赖逐船轮廓。 [同景数据](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset) · [运动失焦检测先例](https://doi.org/10.3390/rs18081166)

## TRAIN 395 阳性同景数组的实测读回

固定作者训练发布版的公共归档支持有界 HTTP 206。P321 实际读取聚焦前缀 640 MiB 和复数距离压缩前缀 512 MiB，按相同编号及高置信船舶 XML 匹配到 31 对 GRD／距压候选，获得 395 的 VH 文件。P323 在聚焦归档读到第 2,441,665,100 个压缩字节时取得同编号 SLC VH，随即停止。只保留匹配成员和聚合结果，训练原始数组及标注留在本地。两次原进程均退出 0。未开验证、测试或模型推理。 [固定训练发布](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a/train) · [前缀脚本](code/OPEN_SAR_TRAIN395_PAIR_20261111/positive_prefix.py) · [SLC 读取](code/OPEN_SAR_TRAIN395_PAIR_20261111/exact_slc.py) · [前缀汇总](results/OPEN_SAR_TRAIN395_POSITIVE_PREFIX_ACTUAL_SAFE_20261111.json) · [SLC 汇总](results/OPEN_SAR_TRAIN395_EXACT_SLC_ACTUAL_SAFE_20261111.json)

P324 逐字节校验并实测 512×512 的聚焦 SLC 与 691×512 的距离压缩数据，两份都是有限的 `complex64`。同编号 GRD 为 708×180 的 `uint16`，没有 GeoTIFF 映射标签，原 XML 的 SLC 框不能直接叠加。XML 的 SLC 与距离压缩框都映射到全局距离列 24149–24195，两边数组的本地零起点距离列均为 163–209。各自在标注方位窗内按距离求平均功率，最高列分别为 172 和 202，都落在标注距离区间。扩展左右各 16 列的对数功率相关为 −0.0448。峰位与相关性不足以支持逐像素直接对齐，且两域使用不同的成像阶段和方位窗口。复数距离压缩做方位 FFT 后，逆变换相对误差 6.72×10⁻¹⁶，固定一列的直接 DFT 对照相对误差 1.88×10⁻¹³。PRF 和多普勒零点尚未校准，距压框的纵轴仍为方位时间。 [读回代码](code/OPEN_SAR_TRAIN395_PAIR_20261111/pair_readback.py) · [实际读回](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json)

ASF 和 Copernicus 同一母产品的十个 IW2／VH 目录足迹独立核对一致。先前按 XML 的 `L1_burst_id=9` 直接查询得到的 ASF 零起点第 9 burst 排除了船位及全部四角。第 5 burst 包含船位和两角，另两角在足迹之外。更正外部元数据候选为第 5 burst；仍需读取官方有效行列与地理定位栅格，核准船位窗口及完整配准。公开训练归档提供了上述阳性复数配对，独立几何配准、模型收益和真实船体轮廓评分仍待实测。 [ASF 第 5 burst](https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_202947_IW2_20200407T053011_VH_18FA-BURST&output=geojson) · [Copernicus 第 5 burst](https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts(97973f8a-eb90-4aba-a626-e7ebf0d53065))

## 模型迁移决策与海面实例分割先例

**可检验的模型迁移。** 当前 RF-DETR 船舶实例分割模型的已验证改动是训练掩膜采样。在 HRSID 固定共同监督的 E24 开发集配对中，mask AP 为 51.434→56.072。RD 初始模型可以复用 RF-DETR 主干权重，或另行测试 UNet++ 的嵌套跳连。距离和速度有不同的单位、分辨率与边界条件。第一轮保留原生 RD 格点，只按距离窗切片或填充，记录零多普勒和速度符号，使用新的回波定位头。空间掩膜中心规则、空间旋转增强及原轮廓头的迁移效用分别待测。若后续有独立回波掩膜，先验证物理格点映射和往返误差，再测试采样改动。 [HRSID 共同行实测](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测) · [TI 多普勒轴处理](https://software-dl.ti.com/ra-processors/esd/MMWAVE-L-SDK/05_05_00_02/exports/api_guide_xwrL64xx/DOPPLERPROC_PAGE.html)

**海面 RD 的直接先例。** 张梦妮等用 2019 年渤海海试的 500 张船载高频地波雷达 RD 谱图构建 COCO 数据，以自适应候选特征提取生成标签，并用带 CBAM 的 Mask R-CNN 同时检测船只和预测回波谱掩膜。原文用同一海试图像构建划分，实测比较主要用 AIS 核对目标检测，未见独立采集划分、独立回波掩膜 AP 或公共数据及模型发布入口。其输出是 RD 格点上的回波谱形状。Akhtar 与 Norheim-Næss 另用奥斯陆峡湾实测 RD 序列结合目标观测角估计船长与速度，评价目标为参数估计。两个先例界定了可主张的差异，单纯将实例分割网络用到 RD 谱图已有实测工作。 [张梦妮等，2025，§2、§4](https://www.ams-journal.org.cn/website/cn/article/doi/10.12362/j.issn.1671-6647.20231226001) · [Akhtar 与 Norheim-Næss，2026](https://www.ffi.no/en/publications-archive/ship-length-and-velocity-estimation-using-range-doppler-maps-and-neural-networks)

**独立 RD 支线的判定。** 在 KuRALS-PD 的原始采集分组上做初步受控定位。学习模型共用输入、定位头、目标域监督和训练预算，比较通用初始化、HRSID 原生监督、HRSID 固定共同行中心监督。相同输入下复算 CFAR 与作者基线。HRSID 组相对通用组的变化检验源域预训练价值，两组 HRSID 的差值检验中心监督的增量。两条海面序列、203 帧与目标条件选波束约束了试点结论。固定版本的作者加载器在各序列内部按连续帧作 80／10／10 划分。跨采集试点应整段留出，并记录两条海面序列的采集关联。完整搜索检出率和虚警需另有可核验的搜索区域、同步参考位置和背景。单次新传感器上带标签微调的结果归为适配。 [KuRALS 数据论文](https://doi.org/10.3390/rs18010173) · [扫描与标签代码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py) · [作者划分代码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/loaders/dataset.py#L59-L91)

**空间船体分割主线。** 若输入来自同景 SAR 成像前数据，先用现有 TRAIN 395 阳性对核准几何、有效采样行及方位频率，再验证回波窗口能否对应聚焦图上的空间候选。现有 FFT 复算只确认数值链，未证实该映射。取得独立空间掩膜后，固定候选比较聚焦单支、常规物理重聚焦、同容量错配和真实配对融合。报告独立留出采集上的固定候选 IoU 与 mask AP75／AP90，预先定义运动失焦组。2026 年已有人用同景局部多普勒与聚焦强度改善失焦船舶框检测。本项目拟检验的增量为可复现的空间轮廓收益。当前尚无 RD 模型训练或泛化分数。 [TRAIN 395 读回](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [Qiang 等，2026](https://doi.org/10.3390/rs18081166)

## RD 试验的执行顺序与评分

先读取输入产品的两轴定义、信号单位、复数或功率形式和数据来源。NISAR RSLC 的零多普勒方位时间对应聚焦图像的空间坐标，可在独立逐船空间轮廓上固定权重和预处理检验零样本。海面雷达的第二轴是多普勒频率或径向速度，应以回波位置或频谱支撑为目标。SAR 成像前只有功率谱时存在方位位置歧义，见本记录的平移对照。 [NISAR RSLC 产品定义](https://nisar-docs.asf.alaska.edu/rslc/) · [Sentinel-1 成像流程](https://sentiwiki.copernicus.eu/web/s1-processing)

1. **先核对 RD 加载和监督。** CARRADA 提供汽车、行人和骑行者的实例 ID、RD 稀疏点、框和密掩膜。密掩膜由投影后的稀疏聚类点经圆形膨胀得到。若作为公开分割试点，沿用其序列级划分，先读回实例、坐标轴和重叠像素，再同时报告对发布密掩膜的 IoU 或 AP 及独立列出的稀疏点覆盖。两类指标共享标注来源，各自衡量对标签的拟合。作者原基线为语义分割，对照其实例输出需先统一为语义口径。道路场景用于检验 RD 数据和采样机制。 [CARRADA 原文与标注方法](https://arxiv.org/html/2005.01456#S4.SS3) · [作者下载页](https://arthurouaknine.github.io/codeanddata/carrada) · [MVRSS 代码](https://github.com/valeoai/MVRSS)
2. **海面试点先固定原生格点。** 三个 RF-DETR 主干初始化来自同版通用预训练、HRSID 固定共同行的原生臂及中心臂。接同一个重新初始化的 RD 定位头，输入用统一的原生格点功率表示，只切窗和填充，排除填充区并记录逆映射。首轮同时固定目标域数据访问顺序、更新步数、头初值、归一化和阈值规则。中心臂相对原生臂的差值用于检验源监督采样的增量。按原始采集分组，能确认的同船重复与重叠时间窗同组。KuRALS-PD 两段海面采集先做目标波束内定位，阈值须预设或只用训练侧校准。独立监视评价需要未筛选的搜索区域及可核验背景，再同输入重评 CFAR，按雷达分辨率和同步误差预锁距离与径向速度的匹配容差。主报固定虚警下的检出率及匹配误差，另列近零多普勒、同格重叠和弱回波。若先导有信号，扩大到独立采集并做重复种子，分别记录冻结主干与目标域微调。 [KuRALS 数据论文](https://doi.org/10.3390/rs18010173) · [同预算的 HRSID 固定共同行实测](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测)
3. **船体空间掩膜优先做同景配准。** 对已读回的 OpenSAR Insight 复数对，先从原始元数据核准 PRF、多普勒零点、有效采样行和独立地理控制点，记录对同一空间候选的往返坐标残差。用现有框核验定位，取得独立逐船空间掩膜后才评分。固定候选和候选分数，比较聚焦图单支、常规重聚焦、相同容量但错配回波、真实同景回波四组。在独立场景上比较固定候选 IoU、边界误差及 mask AP75／AP90，预先标出运动失焦组。真实配对同时超过重聚焦及错配对照，且重复实验保留方向，才支持同景回波带来独立的轮廓增益。 [同景数组实测](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [多普勒与聚焦图融合做框检测的先例](https://doi.org/10.3390/rs18081166)

本次核对覆盖普通源码、现有实测报告及数据原文。CARRADA 和 KuRALS-PD 尚无本轮数据读回或模型分数，当前项目也没有真实 RD 泛化指标。公开海事数据中尚未核实可用的独立逐船 RD 回波轮廓基准。

## 同景样本的母幅 burst 行列归属实测

从 Copernicus 官方目录实际读取同一母幅 IW2／VH 的全部 10 条记录。每条 `LinesPerBurst=1499`、`SamplesPerBurst=24497`，`Lines` 从 0 起依次增加 1499。按 CDSE 提取工具的行偏移公式和 ASF 的零起点文件序定义，测量文件第 5 个 burst 的名义行域为 [7495,8994)。ASF 第 5、9 个条目均与 CDSE 的绝对及相对编号、宽度和各自的方位时间吻合，跨目录时间差分别为 0.021 和 0.057 秒。首次向 ASF 合并查询返回 502，两个单条查询均返回 200。 [CDSE 提取工具](https://github.com/eu-cdse/utilities/blob/main/sentinel1_burst_extractor.sh) · [ASF API 定义](https://sentinel1-burst-documentation.asf.alaska.edu/#api-usage) · [CDSE 第 5 条目录](https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts(97973f8a-eb90-4aba-a626-e7ebf0d53065))

作者公开切片源码用一基行列写出 `SARData`，以减 1 的索引取数组。固定发布 XML 的角点与船中心采用相同的相对到绝对坐标关系。按此约定，裁块的零起点母幅行列为 8007–8518 和 23985–24496，列末尾恰在名义宽度内。船框位于 burst 内第 894–910 行和母幅第 24148–24194 列。把行起点整体多加 1，burst 归属仍为 5。CDSE 和 ASF 的第 5 个足迹均包含作者船中心和两个裁块角点，第 9 个均不包含。作者 XML 的 `L1_burst_id=9` 仍保留，当前证据无法解释该字段的来源或版本映射。此前所列共同距离列是 XML 的原字段值。这里列出的母幅零起点列只对应 SLC，距离压缩产品的绝对列映射待独立标定。 [固定训练标签发布](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset/tree/e90a0c0465cfb8484b4f356ef968d976054f453a/train) · [作者切片源码](https://github.com/ESA-PhiLab/OpenSARInsight/blob/192ded687391b536180af29d59c07ace9cb0344f/backend/dataset_generation_scripts/DVD_dataset_generation/main.py#L866-L931) · [原复算](code/OPEN_SAR_TRAIN395_PAIR_20261111/burst_geometry_readback.py) · [实测摘要](results/OPEN_SAR_TRAIN395_DUAL_BURST_GEOMETRY_ACTUAL_SAFE_20261111.json) · [CDSE 全 10 条查询](https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts?%24filter=ParentProductId%20eq%20%27884ad94b-1595-5b26-a50f-7e97afd39ab9%27%20and%20SwathIdentifier%20eq%20%27IW2%27%20and%20PolarisationChannels%20eq%20%27VH%27&%24top=12) · [ASF 第 5 条](https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_202947_IW2_20200407T053011_VH_18FA-BURST&output=geojson) · [ASF 第 9 条](https://api.daac.asf.alaska.edu/services/search/param?granule_list=S1_202951_IW2_20200407T053022_VH_18FA-BURST&output=geojson)

从 CDSE 匿名请求精确注释 XML 返回 401，ASF 对应 XML 请求返回 307 到官方登录入口。两个采集目录能定位候选及其名义网格。逐行有效样本、官方地理网格、L0 PRF 与多普勒零点仍待核验，不能由 SLC 行域或 burst 间隔代替。本次未读取新的雷达数组、运行模型或评分。 [官方逐行有效样本和地理网格规范](https://s1.pages.eopf.copernicus.eu/s1-l12-rp/main/pfs/level_1_product_specification.html) · [ASF 登录与单 burst 注释说明](https://sentinel1-burst-documentation.asf.alaska.edu/#authorization)

## 本轮迁移决策与 KuRALS-PD 的有效格点

**优先做同景轮廓试点。** 现有 RF-DETR 的正向证据来自聚焦空间图的训练掩膜采样。沿用空间候选和掩膜头，让经标定的同景复数回波只参与候选边界细化。在现有一对训练样本上先核对 PRF、有效采样、相位和坐标，并记录候选窗口往返的空间误差。若数据只有 RD 功率，方位位置可能不唯一。使用现有检测框完成配准试点后，需独立逐船空间轮廓才进入分割评价。固定候选和分数，比较聚焦单支、常规重聚焦、等容量错配和真配对，报告失焦子组与整体的高 IoU 轮廓质量。当前只有数据和坐标核验，未获得 RD 模型分数。 [同景数据读回](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [ESA 聚焦说明](https://sentiwiki.copernicus.eu/web/s1-processing)

**海面 RD 先验证定位与初始化。** 主干可以复用。以原生距离和多普勒网格、相同适配层与定位头对比通用预训练、HRSID 原生监督和 HRSID 同实例中心对齐监督三种初始化。分开报告冻结主干和目标域微调，按完整采集留出。两个海面序列只支持已选波束内的先导定位。若进入完整搜索评估，独立参考航迹、未选波束和可核验背景是固定虚警比较的前提。公开发布的空间检测框与峰值矩形各自适用相应定位任务。 [KuRALS 数据论文](https://doi.org/10.3390/rs18010173) · [OpenSAR Insight 数据卡](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset)

**固定版本的评分掩膜。** 按 KuRALS-PD 发布源码，海面扫描分支先创建 64×800 的零阵列，仅填充零起点距离列 5–678，再把已抑制的零多普勒行移到索引 32。由源码推算，按该生成器每帧有 43,136 个非填充格点，若要求格点包含未抑制的信号，则有 42,462 个。两者均为源码推算，实际包仍须读回确认。固定虚警分母需明确有效区，若用处理后图，零多普勒抑制行不提供原始零速回波证据。获得抑制前复数数组后，可对同一帧分别复算抑制前后性能。 [拼接与移位源码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L269-L334) · [论文 §3.3](https://doi.org/10.3390/rs18010173)

**论文与生成器的窗口规格有差异。** 论文 §3.4 描述的 KuRALS-PD 矩形半宽为距离 3 格、多普勒 5 格。固定版本生成器实际用多普勒 3 格 × 距离 5 格的完整窗口，并在边界裁剪。正式复现前需对实际发布数组、中心坐标和掩膜连通块逐样本核验，记录版本。以独立位置作为定位真值时，另报对 GPS／CFAR 派生标签的分数，避免把固定模板的拟合解释成真实回波边界。 [论文 §3.4](https://doi.org/10.3390/rs18010173) · [窗口源码](https://github.com/lihua199710/KuRALS/blob/abc63a685402178568c29031a3e65c8a95e98109/kurals/dataset_process/kuralspd_processing.py#L39-L62)

## 泛化结果的命名与主线选择

| 要验证的问题 | 最小证据 |
| --- | --- |
| 现有船体分割器能否零样本迁移 | 新产品仍输出已聚焦的空间图，目标语义仍为船体空间轮廓。预先确定归一化和坐标转换，锁定全部模型权重和阈值，在独立采集的轮廓真值上评分。 |
| SAR 训练能否提高海面 RD 定位 | 用目标域标签训练同一个新定位头，在同一输入和预算下比较通用骨干、HRSID 原生骨干及固定同监督实例的中心采样骨干。先冻结骨干对比特征迁移，再按相同预算解冻微调。按采集留出，区分已选波束内定位与完整搜索，后者用独立航迹匹配并报告固定虚警下检出率。 |
| RD 能否提高论文中的船体分割 | 保留聚焦图上的空间候选、类别分数及掩膜评价规则。先在已读回的同景复数配对上核准几何和有效采样，再在独立空间轮廓上比较聚焦单支、常规重聚焦、等容量错配及真实配对。错配优先在同一采集条件内做有约束的替换，匹配窗口尺寸和背景统计，避免模型凭传感器或海况辨认错配。整体 mask AP 与固定候选轮廓质量同时报告，按事先定义的失焦组分层。 |

机载 X 波段的直接检测先例采用距离压缩回波的方位 FFT，并用 AIS 与 CFAR 联合构造检测框。这支持把 RD 定位列为可行任务，也说明定位评估需要单列与候选生成独立的参考真值。已有同景多普勒辅助失焦船舶框检测，以及船载地波雷达回波实例分割。与这些先例的比较应落到空间轮廓、同景回波的增量及相应对照。 [机载 RD 检测原文](https://elib.dlr.de/189078/1/Ship_Detection_Based_on_Faster_R-CNN_Using_Range-Compressed_Airborne_Radar_Data.pdf) · [失焦船舶框检测](https://www.mdpi.com/2072-4292/18/8/1166) · [海面回波实例分割](https://www.ams-journal.org.cn/website/cn/article/doi/10.12362/j.issn.1671-6647.20231226001)

美国佛罗里达海峡 SeaSonde 档案另提供实测多普勒交叉谱和用于天线方向图检查的 AIS 位置记录，可作为独立海杂波与航迹对齐的候选。其海洋流场用途和 AIS 来源尚不足以确认逐帧目标覆盖。需要核对谱窗口时间、速度投影、覆盖及匹配率后，才能确定适用的评分范围。本轮仅核对发布目录与字段。当前优先使用已读回的同景配对试验空间候选几何，现有证据停留在数据链和公开先例层面。 [SeaSonde 数据目录](https://data.griidc.org/data/U1.x843.000%3A0001) · [同景配对实测](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json)


## 少标注迁移的对照与论文决策

如果 RD 指距离–多普勒图，直接复用主干结构、使用 SAR 权重初始化，以及让同景多普勒信息改善原空间掩膜，是三个不同实验。普通监视雷达 RD 的像素代表距离和频率或径向速度。两艘船在同一分辨单元内重叠时，单帧功率图的可分性受限。成像过程中的 SAR RD 则需要按具体产品识别方位多普勒的含义及聚焦映射。沿用调过的 UNet++ 时，可检验嵌套跳连和多尺度特征能否支持 RD 定位。只有目标域实际发生同类掩膜栅格变换，才检验训练掩膜的采样修正。 [HRSID 原始任务](https://github.com/chaozhong2010/HRSID) · [TI 距离与速度测量](https://www.ti.com/lit/an/swra553a/swra553a.pdf) · [Sentinel-1 聚焦](https://sentiwiki.copernicus.eu/web/s1-processing#slc-processing-algorithms)

**RD 权重迁移已有对照先例。** RiCL 在 CARRADA 的 RD 训练序列上预训练，再用 RADDet 标注微调 FCOS。其表 I 报告，使用全部 RADDet 训练标注时，框 mAP@0.5 从 39.7 到 41.0，使用 20% 时从 18.8 到 25.9。Qian 等表 IV 的 3D RAD 检测 AP@0.3 分别为随机初始化 0.311、CARRADA 预训练 0.337、UWCR 预训练 0.361、RADDet 同域预训练 0.391。两项研究均有目标域监督，数据来自车载雷达。RiCL 比较随机初始化，Qian 等在下游保留的是 RA 特征提取器。它们支持把目标域或相近 RD 预训练作为较强对照，也支持优先检验少标注下的迁移价值。 [RiCL 表 I](https://arxiv.org/html/2402.08427#S4.T1) · [Qian 等表 IV](https://magus.ink/homepage_static/paper/ieee26-qianql.pdf#page=4)

**首轮训练要能判别收益来自哪里。** 若要在海面 RD 做定位，用同一原生格点输入和同一新定位头，比对通用初始化、HRSID 原生初始化及 HRSID 同实例中心监督初始化。目标域采集数允许时，预先锁定少标注和全标注两档。按完整采集抽取训练子集，同档次使用相同训练序列、步骤、阈值校准数据和留出集。记录绝对分数与同档配对差值，分别报告冻结主干和微调。若扩大验证，再加入同任务 RD 预训练的同架构对照，避免把通常的雷达预训练收益归给 SAR 特定改动。KuRALS-PD 只先用于已选波束下的定位试点。CARRADA 的逐实例 ID 关联稀疏点及其扩张稠密区域，适合做道路 RD 流程对照。 [KuRALS 原文](https://doi.org/10.3390/rs18010173) · [CARRADA 标注构造](https://arxiv.org/html/2005.01456#S4.SS3)

**当前论文优先验证同景多普勒辅助船体轮廓。** 现有一对阳性数据可先检查空间候选与回波窗口能否可靠对应。通过几何核验且取得独立空间轮廓后，固定候选和分数，比较聚焦单支、常规重聚焦、等容量错配与真实配对，按预先划定的失焦组和独立采集报告轮廓 IoU、mask AP75 和 AP90。直接先例已经用同景多普勒做运动失焦船舶框检测。本项目要检验的增量是空间轮廓是否改善。当前已核验同景数组和 FFT，完整配准、RD 训练及模型收益仍待实测。 [阳性同景实测](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [失焦船舶框检测先例](https://doi.org/10.3390/rs18081166)

## 坐标核验与轮廓收益的判别试验

先根据数据轴确认任务。零多普勒地理定位的 Sentinel-1 SLC 仍是已聚焦的斜距–方位图。若数据轴确实是距离–径向速度，船舶实例对应回波簇，现有空间掩膜头的输出定义随之改变。试点先记录两个轴的名称与单位、采样间隔、数组来源、是否保存复数以及独立真值类型。 [Copernicus SLC 规范](https://sentiwiki.copernicus.eu/web/s1-processing)

同景轮廓试验加入已知配准偏移，检验候选边界改进是否随错位加重而衰减。先用独立物理定位参考估计配准残差，在留出评分前按原生格点固定可接受残差和偏移阶梯。推理时保持候选与分数相同，分别沿斜距与方位改变真实回波窗位置，保留同条件错配和常规重聚焦对照。主报固定候选的逐实例 IoU 差值和正式 mask AP75/AP90，按采集统计。现有 395 样本的近船距离剖面相关约 −0.045，这一项只比较原生剖面，独立配准仍待验证。 [同景数组实测](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [多普勒船舶框检测先例](https://doi.org/10.3390/rs18081166)

PeakConv 已在车载 RD 分割中学习以回波峰为中心的感受野，MARSS 在 CARRADA 使用多尺度和方向建模。自研 UNet++ 的双方向模块若改用于 RD，须先与同架构普通卷积比较，固定物理坐标和算力预算。这两项先例可用于道路任务比较，海面迁移效用须在独立采集上验证。 [PeakConv，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_PeakConv_Learning_Peak_Receptive_Field_for_Radar_Semantic_Segmentation_CVPR_2023_paper.html) · [MARSS，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.html)


## 从空间模型迁移到 RD 的首轮验收

2026-11-12 复核。现有自定义 UNet++ 的多尺度特征提取和嵌套跳连可作为可复用结构。现有 RF-DETR 的中心采样是训练监督干预。优先用已完成对照的同架构检查点检验权重迁移，待确定目标域确有同类格点变换后，再单独检验采样规则。当前没有真实 RD 模型分数。 [自定义 UNet++ 模块核查](UNETPP_MODULE_MODIFICATIONS_CONFIRMED_20261017.md) · [固定共同监督的训练对照](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测)

| 输入及目标 | 第一项可判别的验证 |
| --- | --- |
| 海面雷达的距离 × 径向速度 | 保留原生格点，先用功率输入、新定位头与原轴坐标回归。冻结主干比较通用预训练、HRSID 原生权重、HRSID 同实例中心监督权重，再按相同预算微调。只有把同一初始化、同一头和预算下的自定义 UNet++ 与普通 UNet++ 比较，才可将差值归于结构改动。按采集整段留出，独立航迹支持距离与速度误差，未筛选搜索域和经核对背景支持固定虚警下的检出率。 [KuRALS-PD 论文](https://doi.org/10.3390/rs18010173) |
| 同景 SAR 成像前复数数据与已聚焦图 | 沿用现有空间候选和掩膜头。按采集几何取局部回波，在聚焦图上细化边界。先用现有框和传感器元数据验配准，再取得独立空间实例掩膜。第一轮比较空间单支、常规重聚焦、同容量错配和真实配对。统一候选、分数和评价，检验预定义失焦组的固定候选 IoU 和整体 mask AP75／AP90。 [OpenSAR Insight 数据卡](https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset) · [失焦船舶框检测先例](https://doi.org/10.3390/rs18081166) |

**配准精度的数量级。** 对高宽为 $h,w$ 的理想矩形，保持形状和面积，沿两轴平移 $(d_x,d_y)$ 时，交集面积为 $(w-|d_x|)_+(h-|d_y|)_+$，再除以两矩形并集面积得到 IoU。仅沿短轴平移时，要达到 IoU $t$，平移绝对值上限为 $w(1-t)/(1+t)$，其中 $w$ 为短轴宽度。在原生空间掩膜格点上，3 像素宽目标对应 IoU 0.75 和 0.90 的单轴上限分别约 0.429、0.158 像素，5 像素宽时约 0.714、0.263 像素。四个值已代回公式复算。该算例只用于设定错位敏感性试验的量级，实际残差预算须在训练侧按船宽和原生轮廓复算，联合考虑两轴偏差及参考点误差。配准评分使用独立参考坐标，并报告原生格点和物理单位。 [项目的固定匹配位移复算](HRSID_DEV705_CONTINUOUS_AP_SENSITIVITY_20261111.md#hrsid-与-ssdd-的固定匹配对称位移)

**公开数据的使用范围。** CARRADA 可核验带实例 ID 的 RD 加载与分割流程，场景为道路，密掩膜由稀疏回波点膨胀。KuRALS-PD 的两条海面序列共 203 帧，可做已选波束内定位试点，发布矩形来自外部位置与 CFAR 峰值处理，适用于弱标签评价。RADIal 发布原始 ADC 和车辆位置字段，K-Radar 发布 4D 雷达张量和道路 3D 框，WaterScenes 发布雷达点云与图像实例标注，这三者的官方标注说明均未提供船舶 RD 像素轮廓。FM-DETR 的 RCShip-1.0 为距压方位时间域的单类框检测，测试数据由独立 Sentinel-1 场景逆处理重建，可作为另一项距压检测对照。 [CARRADA](https://arxiv.org/html/2005.01456#S4.SS3) · [KuRALS-PD](https://doi.org/10.3390/rs18010173) · [RADIal 标注](https://github.com/valeoai/RADIal#labels) · [K-Radar](https://arxiv.org/abs/2206.08171) · [WaterScenes](https://github.com/WaterScenes/WaterScenes) · [FM-DETR §2.1](https://www.mdpi.com/2072-4292/18/21/3669)

**当前执行决策。** 论文优先采用同景配准及空间轮廓路径。已有一对训练样本通过复数数组与 FFT 数值核验，完整配准和独立空间轮廓待落实。首轮先核对已有样本的逐行有效区、PRF 和频率原点，再验证空间候选窗口。海面 RD 定位可作为独立少标注适配试点。单一采集带标签微调只能报告该条件下的适配效果，跨采集与跨传感器各需独立数据验证。 [阳性同景读回](results/OPEN_SAR_TRAIN395_PAIR_READBACK_ACTUAL_SAFE_20261111.json) · [ESA SLC 处理](https://sentiwiki.copernicus.eu/web/s1-processing)

## 原生 RD 网格与模型接入核验

2026-11-12 补充。既有 HRSID 载体为 RF-DETR 1.12.0 Seg Nano，默认补丁宽度为 12。DINO 实现支持两轴分别按补丁对齐的矩形输入，对矩形 token 网格插值的是位置编码。公开 `predict()` 入口仍会缩放并按 RGB 归一化，既有空间训练还含缩放、裁剪与翻转。RD 试验拟使用底层 eager 入口，读入浮点功率，按训练采集锁定归一化，保留轴次序与零频位置。首轮以同一标量通道复制到三个输入通道来匹配原骨干，且三组初始化共用处理。训练和评分保留有效区及原始物理轴。骨干在内部注意力前未收到外层有效掩膜，需实际复核补边对临近输出的影响。若有掩膜输出，先恢复到填充后的网格，再裁去无效区，避免把全幅填充区域压缩进原生图。 [原载体配置](HRSID_COCO_NATIVE_FIT_E24_PROTOCOL_20261109.md) · [矩形位置编码](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/backbone/dinov2_with_windowed_attn.py#L324-L416) · [骨干掩膜传递](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/backbone/backbone.py#L201-L235) · [默认推理预处理](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/_prediction/core.py#L1074-L1100) · [输出后处理](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py#L252-L292)

合成 CPU 复算以 64×800 原生格点为例，将距离列按半开区间 [0,416) 和 [384,800) 切片，两个输入均补齐至 72×420，分别只接收全局距离列 [0,400) 和 [400,800) 的结果。裁补往返后 51,200 个 float32 格点逐位一致，每格只回填一次。24 个检查点覆盖零频、两端频格、全图边缘和重叠接缝。合成采样率设为 1,000 Hz，频轴直接公式与 FFT 格点在检查位置一致。这个试验只验索引和张量回填。实际模型前向、峰定位、窗口边缘效应和真实数据分数待测。首轮模型检查应另置相距 1、2、4 格点的同轴双峰和跨接缝单峰，检查下采样能否保留间距，评估补边对比与重叠结果一致性。海面定位的迁移对照固定原生格点及定位头，比较通用、HRSID 原生、HRSID 中心监督三种初始化。

同景校准另查 NASA CMR 两条精确母幅记录，聚焦 SLC 与 Level-0 RAW 分别命中 `G1719774182-ASF` 和 `G1719764450-ASF`。两条目录的场景级 `DOPPLER` 字段均为 `0`，目录没有 PRF。跟随 ASF 的 ISO XML 入口最终到 Earthdata 登录，两项均返回 HTTP 401。未读到精确注释，目录的 `0` 不用作方位 FFT 的多普勒零点。论文路线的首要数据条件仍是对应 burst 的逐行有效区、地理定位及同景相位和频率标定。 [NASA CMR SLC](https://cmr.earthdata.nasa.gov/search/concepts/G1719774182-ASF.umm_json) · [NASA CMR RAW](https://cmr.earthdata.nasa.gov/search/concepts/G1719764450-ASF.umm_json) · [Sentinel-1 处理说明](https://sentiwiki.copernicus.eu/web/s1-processing)

补查的 HAXR 包含港口雷达与 AIS，雷达数组按距离、方位组织，适用于位置或跟踪。CSIR Fynmeet 的实测小船和海杂波有作者报告，官方页面列出的原始 MAT 数据需向机构申请。目前两个入口均未确认提供独立逐船 RD 回波轮廓。 [HAXR 官方数据](https://zenodo.org/records/19824555) · [CSIR 官方报告](https://researchspace.csir.co.za/items/c72f2882-3afc-4e57-aeb8-8e5bf4963529/full)

伯明翰大学的 STREAM Portsmouth 海试公开了一部分 76–81 GHz 实测数据。公开包的说明明确列出扫描样例 `sea_4` 和固定凝视样例 `stare_4`，两项提供复数距离谱与 ZED 参考视频。完整采集表中另列有专门的海杂波凝视，需另向作者询问。它提供慢时间 FFT 构建真实 RD 的第二个候选，并可先检查视频覆盖与时间误差。发布资料未给出逐船 RD 像素掩膜。当前只核对发布说明，未下载或评分雷达数组。 [官方数据](https://edata.bham.ac.uk/626/) · [采集和格式说明](https://edata.bham.ac.uk/626/1/stream_sea_radar_dataset_1_description.pdf)
