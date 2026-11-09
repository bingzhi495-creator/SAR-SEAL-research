# HRSID 跨数据集证据范围

官方作者发行 PNG 版本。已有独立核验覆盖官方归档中央目录的 5,604 个 PNG、同归档配套 `train2017` 和 `test2017` 标注及两个完整 PNG 样本。最初批量传输保留已验证的 64 MiB 前缀。11 月 9 日续取完成后，独立核验全部归档字节与成员 CRC。现有本地 JPG 版本包含 5,604 个逻辑图像和 5,601 个不同编码对象，已核对 CRC 与旧栅格试验输入。按文件名对应的标注字段曾完成语义核验，两版本未证明像素一致。报告应明确使用的编码和标注版本。 [HRSID 作者发行与论文入口](https://github.com/chaozhong2010/HRSID)

| 现有记录 | 可支持的范围 |
| --- | --- |
| 官方风格 TRAIN 的 3,642 图、11,047 个标注 | 已锁定 FIT2937/DEV705。文件名前缀分组 137 个，109/28 分配。该前缀是母景代理，母景独立性还需证据。 |
| TEST 的 1,962 图、5,922 个标注 | 历史文稿明确记载在这部分做过融合参数、种子与检查点选择。可作为冻结新配方后的外部数据集评估，历史接触需要披露。 |
| 全部 16,969 条标注栅格试验 | 尝试已完整，9 条 TRAIN、4 条 TEST 得到空掩膜。原非空门槛导致记录退出 1，其他解码错误为 0。Windows 与 DGX 编解码扩展不同，跨运行时的空掩膜计数需重新固定。 |
| 2 组跨 TRAIN/TEST 编码完全相同图像 | 两组标注均有差异。对目标域重训须预先锁定去重和原划分敏感性报告。SSDD FIT/DEV 的零编码交集也不能证明场景独立。 |

历史结果和新测试应单列。保持主协议的完整测试总体，空掩膜和跨划分重复的处理须在新输出前固定。冻结 SSDD 配方后，可以先执行 JPG 的零目标调参迁移。固定目标域训练是另一项可移植性验证，需要先完成 FIT-only 的栅格及真实增强核验。原始 FIT 中至少五张超过 100 个实例，最大 163。RF-DETR 的 13 个查询组分别匹配，默认每组 100 仍不足以对这些图实施全实例配对。若目标实验要求全配对，应在比较臂统一设定足够查询容量、明确成本并固定 COCO maxDets=100。 [历史测试使用披露](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/d2b38d430302153c55a2e5330cea296a9324dcdb/paper_tgrs/revision_tgrs_20260905/sec_exp.tex#L21) · [RF-DETR 原生分组匹配](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/matcher.py#L747-L771)

本次只核对既有普通来源与已闭合汇总，未打开 TEST 图像或标注正文。正式跨数据集效用还取决于已冻结的 SSDD 方法和公平对照。

## 2026-11-09 官方 PNG 续取及其他数据来源

作者页面把 JPG 近岸和离岸更新包与 PNG 包分列。PNG 的 Drive 文件 ID 为 `1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc`，归档总长 5,698,302,315 字节。备用百度网盘页面仅核到提取入口，文件身份尚未核实。 [作者下载说明](https://github.com/chaozhong2010/HRSID#dataset-download-link) · [官方 PNG](https://drive.google.com/file/d/1xgXi8KC3MDWuu7Yp4n2J-LOPYLRszHAc/view)

Windows 标准 HTTP 客户端实际取得两段完整 206 响应。64 MiB 边界后的首 1 MiB 用时 2.00 秒。另一个请求覆盖旧前缀末 4 KiB 及其后 16 MiB，用时 7.40 秒，旧末尾与新首 1 MiB 分别逐字节核对一致，版本时间和总长一致。已连续取得 80 MiB。完整归档及全成员 CRC 待核，未打开 ZIP 成员或评价模型。历史失败原件独立保留。 [实际传输记录](results/HRSID_PNG_STANDARD_HTTP_RESUME_PROBES_ACTUAL_SAFE_20261109.json)

| 核查来源 | 适用范围 |
| --- | --- |
| PsSAR-v1.0 | 复用 HRSID 图像和船舶实例，增加海陆背景标签。可用于近岸条件分层，仍承接 HRSID 的图像与历史测试接触。作者标记数据 GPL-3.0、基线代码 Apache-2.0。 [数据与许可](https://github.com/PerSARption/PsSAR-v1.0-Unlocking-SAR-Panoptic-Segmentation/blob/main/README_DATASET.md) |
| OSSDD | 公开 Sentinel-1 VV/VH 图块。作者将旋转框内部栅格化成二值图，适用于检测及框形状验证，船体轮廓精度仍需真实实例标注。 [标注原文](https://arxiv.org/html/2608.01963v1#S4) · [数据卡](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset#license-and-attribution) |
| FSS-Seg / PID-SAM | IJCAI 2026 论文报告基于 FUSARShip1.0 的七类像素语义数据，以 Dice、IoU、精度与 HD95 评价。论文用点级掩膜贡献和 PPO 优化提示，可作为点条件机制的直接先例之一。作者源码入口为 ZIP，训练配置与实例 ID 发布尚未核验。 [论文](https://ijcai.org/proceedings/2026/0161.pdf) · [作者代码](https://github.com/Fm336/PID-SAM) |

### 作者源码复核和后续实际传输

已取得固定提交的 PID.zip，96,928 字节，Git blob 为 99c5b6d3dae18ca77617379518c768c0e085ef08，ZIP 成员 CRC 通过。实物含 5 个 Python 源文件、7 个编译缓存、训练和测试样本列表。训练入口导入的船舶 trainer、LoRA 包装和通用 trainer 在压缩包中只有缓存版本，SAM 库和数据仍依赖外部输入。固定包没有发现许可证或环境锁定文件。仅静态读取，运行复现仍待补齐。 [固定作者源码](https://github.com/Fm336/PID-SAM/blob/2764fab108304ad5da62fbb5e6c4e65aeb4ece0a/PID.zip)

在先验证的 80 MiB 连续前缀上开启新的顺序传输。前 16 个完整 64 MiB 块已接收并逐块计算 SHA-256，连续归档已达 1,157,627,904 字节。既有归档和失败片段保持独立。全量接收、目录和成员 CRC 仍需在进程结束后独立核验。

### 完整官方 PNG 归档已核验

本轮从已核验的 80 MiB 前缀续取 84 块新数据，实际接收 5,614,416,235 字节。传输及最终目录核对耗时 637.69 秒。独立回读逐块核对范围和 SHA-256，完整归档为 5,698,302,315 字节，中央目录与此前确认的官方目录精确一致。ZIP 含 5,604 个 PNG、5 个 JSON 和 3 个目录条目，全部成员 CRC 通过。 [独立安全汇总](results/HRSID_OFFICIAL_PNG_FULL_ARCHIVE_INTEGRITY_ACTUAL_SAFE_20261109.json)

已确认配套编码归档的完整性。后续仍需实际像素与标注对齐、固定空栅格处理及评估协议。历史测试接触范围沿用既有记录。

## 第三套实例轮廓验证的新增查证

SARShipRES 的 IEEE 作者稿第三节 A 说明，使用 COCO API 从 HRSID 和 SSDD 取影像和实例，新增文本指代及散射属性，组成 6,764 个表达与掩膜配对。它可用于语言引导分割研究，和两套源数据共享影像及标签，无法提供独立场景盲测。 [IEEE 作者稿](https://xplorestaging.ieee.org/ielx8/4609443/4609444/11739338.pdf?arnumber=11739338)

Instance-RSDD 作者报告从 RSDD 选择 3,555 幅真实 SAR 图逐船标注像素轮廓，2,840 幅用于训练，715 幅用于评价。作者的公开说明链接到网盘，项目仅核对到 COCO 标注目录和说明文件入口。还需取得实际标注、图像对应关系及许可，核对官方评价成员、同母景裁块与 HRSID/SSDD 交集。原始 RSDD 公开的是旋转框，须使用作者额外标注的实例掩膜。只有上述证据通过后，才纳入冻结 A/B 模型的零调参直接迁移。 [原文](https://doi.org/10.3390/rs15051324) · [Instance-RSDD 作者入口](https://github.com/XIDIAN212Lab/SRNet/blob/master/readme.md#the-instance-rsdd-dataset) · [原始 RSDD](https://github.com/makabakasu/RSDD-SAR-OPEN)

已用作者公开提取码在未登录浏览器中打开分享，并读取完整的 `readme.txt` 预览。HTTP 200 的 UTF-8 预览正文与清单同为 1,145 字节，SHA-256 为 `9bb26c196be254fb1a15ed5dd599ee9dc113bd96be82a645b9f01c5a433bb8e6`。说明确认原图来自 RSDD-SAR，并给出两篇引用。说明未提供数据许可、逐图映射或划分细节。 [作者分享](https://pan.baidu.com/s/1_hIyZKErRDvvceiKDhRmvA) · [原始影像论文](https://doi.org/10.12000/JR22007) · [实例标注论文第 4.1.2 节](https://www.mdpi.com/2072-4292/15/5/1324#sec4dot1dot2-remotesensing-15-01324)

`annotation_coco` 的清单列出 `train.json` 1,933,970 字节及 `test.json` 540,834 字节。只选择训练标注尝试普通下载，服务要求登录，未取得训练标注字节。测试标注只核对清单，未打开。现有证据可确认作者分享入口与文件名，不能验证 COCO 字段、逐船多边形、图像对应或独立性。 [作者发布说明](https://github.com/XIDIAN212Lab/SRNet/blob/master/readme.md#the-instance-rsdd-dataset)

## 2026-11-10 独立外部数据入口与标签边界

| 来源 | 已核对内容 | 可采用的评价范围 |
| --- | --- | --- |
| Schwegmann SAR 船舶数据 | 作者说明以 MIT 发布船舶、疑似目标和海杂波裁块。数据论文第 IV 节报告 43 个 Sentinel-1 极化影像和 3 个 RADARSAT-2 影像，共 1,596 个船记录，像元间距 25 或 40 米。二值参考仅对应裁块中央的船，论文未明确逐像素描绘流程。发布库未含整景原始影像。 | 可用于冻结的中心船形状辅助检验。需要按采集分组处理跨极化重复，并采用适配目标窗口的评价。裁块其他船的完整标注与高分辨率船体真值未获支持。 [作者数据说明](https://gitlab.com/sarmaps/sar-ship-dataset/-/raw/9decec2a18b9d7d6fd2d057a435808ca67d2bad1/README.md) · [作者数据论文](https://sarmaps.gitlab.io/data/dataset_paper.pdf) |
| UEIKAP SAR | Zenodo 发布 Sentinel-1 SAR 包，许可标为 CC BY 4.0。作者会议记录确认项目使用船体与尾迹多边形。旧预检核对 515 张 TIFF 与 515 份 YOLO 多边形，class 0 有 760 个候选船体。class 1 与 COCO wake 对齐，class 0 的文字类名、逐实例绘制流程及完整性仍待核。32 个已映射采集家族中，25 个跨作者芯片划分。 | 旧记录按来源预留 18 个家族、108 图、139 个 class 0 候选，与另取的 QA50 家族分离。核对现有 QA50 v2 包仍在本地，179,666,088 字节。仅在公开标签语义、完整性及分组核验通过后，才以预留家族为单位做外测。官方芯片划分保留作作者口径。 [Zenodo 发布](https://zenodo.org/records/18391027) · [作者会议记录](https://www.iris.unina.it/handle/11588/1071901) |
| AIR-SARShip 与 LFG-Net | AIR-SARShip-1.0 发布论文明确为 31 幅 GF-3 影像与 VOC 矩形框。LFG-Net 作者报告的 AirSARShip 掩膜指标另有标注前提，公开仓库未定位可复用的实例掩膜与生成说明。 | 原始发布支持检测。按独立实例掩膜评价仍须另核数据。 [数据论文](https://doi.org/10.12000/JR19097) · [LFG-Net 作者仓库](https://github.com/Evarray/LFG-Net) |

Schwegmann 数据已实际从固定 GitLab 源取得首段 4 MiB。HTTP 206 声明完整正样本 JSON 为 289,068,441 字节，前缀 SHA-256 为 `b517a9442f5a1a7c89d45093c34fb2fb27705eba4420974cd42f09bd3b952411`。解出 23 条完整记录，覆盖两个产品字符串。全部包含配对的 101×101 与 21×21 有限数值影像和二值参考，前景面积均为 14–41 像素，两个尺寸下逐条面积相同。首轮读取 1 MiB 时按单层 JSON 列表解析失败。随后按实际双层列表解析四 MiB 前缀。未读取全包或执行模型。 [固定作者 JSON 入口](https://gitlab.com/sarmaps/sar-ship-dataset/-/raw/9decec2a18b9d7d6fd2d057a435808ca67d2bad1/json/ship_positives.json)

PSeg-SSDD 仍是当前配方重训的优先跨库路线。Instance-RSDD 的实际掩膜与许可待核，其 RSDD 源含 TerraSAR-X，须与 HRSID 的具体采集来源核对。第三套数据的质量核验与模型选择相互隔离。 [原始 RSDD 发布](https://github.com/makabakasu/RSDD-SAR-OPEN)

### UEIKAP 旧缺失文件头补齐

按原作者 ZIP 中央目录摘要和 515 张 TIFF 成员清单，从 12 个原 HTTP 429 成员分别取得 8,192 个压缩字节。共 14 次 HTTP 206 请求，含目录共 591,461 字节。原文件头 503／515 补至 515／515。按原 Copernicus 产品开始时间及 95% 足迹覆盖规则，12 张均唯一归入已有五个采集家族。重新向官方目录发起六次查询，12 张的归组完全复现。旧唯一归组覆盖 493／515 增至 505／515，仍有 10 张未唯一归组。32 个已映射家族中跨作者芯片划分的仍为 25 个。 [Copernicus OData 官方说明](https://documentation.dataspace.copernicus.eu/APIs/OData.html)

新增 12 张均在已有 QA 家族且与 18 个预留外测家族零交集。原 QA50、108 张预留图与 139 个候选船体的名单保持冻结。全库旧标注计数核对到 515 图均有 class 0、514 图兼有 class 1，无无标注图。该发布总体富集正例，空海面的虚警仍需别处验证。作者说明 SAR 船体多边形用于方位偏移估计，AIS 参与标注。这支持类语义解释，逐实例勾绘来源和图内标注完整性仍未核准。 [作者技术论文](https://doi.org/10.1109/MetroAeroSpace64938.2025.11114590)

本次限于缺失成员的文件头和产品目录，未解码 TIFF 像素、未读取原 QA50 或预留成员字节、未打开标签或执行模型。 [实际恢复汇总](results/UEIKAP_MISSING_TIFF_HEADER_RECOVERY_ACTUAL_SAFE_20261110.json) · [目录复查汇总](results/UEIKAP_MISSING_TIFF_CATALOGUE_READBACK_ACTUAL_SAFE_20261110.json)

### UEIKAP 十张未归组裁块的双目录复核协议

只用旧回执中仍未归组的 10 张裁块的 UTC 时间戳与 WGS84 足迹。原 18 个预留采集家族与 QA50 名单固定。按 QA50 编号升序取首个已有唯一归组的质检记录作为正对照。先确认两个公开目录都能按原规则复现正对照。该样本只校验查询与产品名归并。

Copernicus 和 ASF 分别检索同一卫星的 IW GRDH 产品，候选时间窗为裁块文件名时间前后各 120 秒。ASF 空间请求范围外扩 0.001 度，实际交集仍用原足迹。拒绝截断、未完成或无法解释的匹配产品结果。产品名解析出卫星、绝对轨道和 datatake；原始与 COG 表示去重。主规则保持原有精确起始时间和单产品覆盖至少 95%。两目录有相同的唯一标准化产品才列作新的精确匹配证据。查询失败与有效零命中分开。

扩展诊断在每个目录内按物理采集家族取足迹并集，在原裁块上计算覆盖率。历史单产品门槛沿用经纬度平面面积，并集门槛使用球面面积。只允许同卫星、同绝对轨道、同 datatake 且处于固定时间窗的产品相并。分别报告至少 95% 的并集候选与来自预留家族的正面积交叠、仅边界接触。并集候选只提供来源线索。预留家族如有任何接触，应单列复核，其余无命中也限于这两套目录与预先声明的查询范围。名单原件保持冻结，本轮不打开图像、实例标注或模型结果。 [Copernicus OData](https://documentation.dataspace.copernicus.eu/APIs/OData.html) · [ASF 查询](https://docs.asf.alaska.edu/api/keywords/) · [产品命名与切片](https://sentiwiki.copernicus.eu/web/s1-products)

[固定几何实现](code/UEIKAP_ACQUISITION_METADATA_20261110/ueikap_acquisition_footprint.py)
