# SAR-SSP 作者档案验收

已接收作者发布的 `SLGSea_Datasets.zip`。本地大小等于分享清单的 866,821,756 字节。固定该文件的 SHA-256，逐项核验 2,324 个 SAR-SSP 影像、panoptic PNG 和标注 JSON 成员与作者索引。船舶是类别 0 的实例类，海和陆为区域类。每张图及标注尺寸匹配，PNG 实例值、面积和船舶框与 JSON 对应。[作者发布及数据说明](https://github.com/deng1218/SLGSea/blob/8225c680d3e991aa2cb5b22af3918b8704ec7b02/README.md#datasets)

| 作者划分 | 原生 500×500 图块 | 船舶实例 | 含船图块 | 母图组／含船组 | 接触图块边缘的船体 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 929 | 461 | 217 | 31／31 | 174 |
| Val | 232 | 139 | 48 | 31／23 | 48 |

全部 31 个母图号都出现在两个划分中。精确解码图像无重复，按作者母图号和坐标发现 **4 对跨划分图块的范围交叠**。同一发布包的 5,604 个 HRSID-PAN 与 1,160 个 SPSD 影像中，未找到与 SAR-SSP 相同尺寸且逐像元一致的解码 RGB。图块重裁、重新编码和不同波段版本仍需用母景信息核对。

坐标与检测框比对进一步确认，SAR-SSP 裁片来自 AIR-SARShip-1.0 的 31 个高分三号原景。历史 B1 的 SARDet-100K 初始化覆盖其中 27 景，该权重不可作为全量无接触外测对照。现行冻结 RF-DETR center5202 的项目链为官方 COCO 初始化和 HRSID FIT。正式结果应保留母景分组及与其他高分三号队列的采集级限制。[来源匹配及模型链](SAR_SSP_PROVENANCE_NOTE_20261125.md)

[真实验收回执](results/SAR_SSP_AUTHOR_INTAKE_SAFE_20261125.json) · [复现脚本](code/SAR_SSP_AUTHOR_INTAKE_20261125/audit_archive.py)

## 冻结原生模型推理

现行 COCO→HRSID 载体已经在全部 232 张官方验证图完成 GPU 推理，运行 78.1 秒，保留 22,741 个检测候选及 8,592 个面积合格的细化概率平面。独立 CPU 核验逐项复核原生掩膜和候选审计，一致通过。推理与回读未打开目标标注，未进行优化器更新。候选计数属于载体输出，船体覆盖与精度待固定规则评分。[冻结推理协议](code/SAR_SSP_AIR_FROZEN_20261125/CAPTURE_PRELOCK.md) · [推理回执](results/SAR_SSP_VAL232_RFDETR_CAPTURE_SAFE_20261125.json) · [独立回读回执](results/SAR_SSP_VAL232_RFDETR_NATIVE_VERIFY_SAFE_20261125.json)
