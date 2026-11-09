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
