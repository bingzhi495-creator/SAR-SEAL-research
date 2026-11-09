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
