# 像素交换先例和外域标注核查

Atkinson 已用独立的前景和背景吸引力排序做等量交换，以保持每个粗像元的类别比例。Xu 与 Huang 进一步明确区分逐点和成对打分，在土地覆盖亚像元配置中用时空标签对像元对估计吸引力。论文的技术差异在于固定 SAR 检测候选、已预测的分数、边界点集、占据率头和每实例实际补删数，检验距离配对是否限制可修复的位置。同量对照支持保留这一具体结论。整框替代策略仍需报告。 [Atkinson 原文](https://www.asprs.org/wp-content/uploads/pers/2005journal/jul/2005_jul_839-846.pdf) · [Xu 与 Huang 原文](https://doi.org/10.1109/LGRS.2013.2268153) · [本轮实测](INSTANCE_RSDD_RESERVE_POINTREND_ACTUAL_20261123.md)

已有记录覆盖了 Instance-RSDD 的 125 个父景，包括历史测试的 112 个和后续预留的 13 个。可匿名获取的 NASTaR 提供 NovaSAR 影像、AIS 长宽和航向，可用于按过境划分的物理长度辅助验证。已读取一份 AIS 表头和一对 GeoTIFF 文件头，确认显示版为 512×512 uint8，原值版样本为 uint16。这里应以逐文件头为准。作者转换脚本的分割字段为空。另一项高分三号研究报告了 139 张带专家船体轮廓的裁块，来自三次采集，需要向作者申请。当前独立掩膜评价的现成路径仍是原先锁定的 12 个 Umbra 采集组的判读回传。 [NASTaR 数据目录](https://data.bris.ac.uk/data/dataset/2tfa6x37oerz2lyiw6hp47058) · [作者转换脚本](https://github.com/Benyaminhosseiny/NASTaR/blob/main/NASTaR_to_COCO_VOC.py) · [高分三号轮廓数据原文](https://doi.org/10.3390/rs16193669) · [Umbra 判读入口](UMBRA_PROSPECTIVE_GEC_ANNOTATION_ACTUAL_20261122.md)

## 标注文件复核与独立外测

项目目录收到的 Instance-RSDD 两份 COCO 与此前文件接收和评分绑定的 SHA-256 一致。作者 train 含 2,810 图和 3,147 个船体轮廓，test 含 740 图和 872 个轮廓。740 张测试图与 74 张额外父景图均按图像 ID 和名称核对到本地影像。测试 112 个父景和追加实验的 13 个父景共同覆盖该发行版的 125 个父景。同一文件的再次获取可以复现上述结果，新增切块仍需区分父景、传感器日期和像素重复。这次未产生新评分。[标注接收指纹](INSTANCE_RSDD_AUTHOR_FILES_ADMISSION_ACTUAL_20261122.md) · [主表独立复现](SAR_B5_MAIN_TABLES_REPRODUCTION_ACTUAL_20261123.md) · [新增 74 图](INSTANCE_RSDD_PARENT_RESERVE_ACTUAL_20261123.md)

作者 Test01 标注文件实计 60 图、207 条，全部分割字段缺失或为空。LMRD 使用中心坐标、长宽及角度标出旋转框。S3T-Net 的船舶提取实验使用 HRSID，GCIE-Net 使用 PSeg-SSDD 和 HRSID。这四项提供检测数据或既有数据上的方法对照。[Test01 原标注](https://github.com/Bighandsome-handsome/SAR-ShipDetection-Project/blob/b205e83d82f9868a1441118e132e900c5eccee2d/Test01/annotations/instances_val2017.json) · [LMRD 论文 §2.2](https://www.mdpi.com/2072-4292/18/10/1639) · [S3T-Net 原文](https://www.ygxb.ac.cn/cn/article/doi/10.11834/jrs.20265095) · [GCIE-Net 原文 §4](https://www.mdpi.com/2072-4292/18/15/2512)

PID-SAM 论文 §4.1 报告从高分三号 FuSARShip 1.0 衍生的 FSS-Seg，共 11,857 幅七类语义分割影像。作者库中的 PID.zip 提供训练代码和划分，已有核验记录显示读取每图单灰度掩膜。未发现从作者取得这套衍生掩膜的公开下载、标注许可或父景映射。SISS-YOLO 同样报告了研究者新增轮廓，其数据说明指向原始 OpenSARShip 2.0。两项保留为后续作者数据线索。Umbra 预锁队列的两份独立判读尚未回传。[PID-SAM 论文](https://ijcai.org/proceedings/2026/161) · [作者代码固定版本](https://github.com/Fm336/PID-SAM/blob/2764fab108304ad5da62fbb5e6c4e65aeb4ece0a/PID.zip) · [此前数据核查](HRSID_CROSSSET_EVIDENCE_BOUNDARY_20261108.md) · [SISS-YOLO 原文](https://www.mdpi.com/2072-4292/17/17/3118) · [Umbra 判读状态](UMBRA_PROSPECTIVE_GEC_ANNOTATION_ACTUAL_20261122.md)
