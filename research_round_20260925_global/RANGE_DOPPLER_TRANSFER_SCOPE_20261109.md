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
