# 像素交换先例和外域标注核查

Atkinson 已用独立的前景和背景吸引力排序做等量交换，以保持每个粗像元的类别比例。Xu 与 Huang 进一步明确区分逐点和成对打分，在土地覆盖亚像元配置中用时空标签对像元对估计吸引力。论文的技术差异在于固定 SAR 检测候选、已预测的分数、边界点集、占据率头和每实例实际补删数，检验距离配对是否限制可修复的位置。同量对照支持保留这一具体结论。整框替代策略仍需报告。 [Atkinson 原文](https://www.asprs.org/wp-content/uploads/pers/2005journal/jul/2005_jul_839-846.pdf) · [Xu 与 Huang 原文](https://doi.org/10.1109/LGRS.2013.2268153) · [本轮实测](INSTANCE_RSDD_RESERVE_POINTREND_ACTUAL_20261123.md)

已有记录覆盖了 Instance-RSDD 的 125 个父景，包括历史测试的 112 个和后续预留的 13 个。可匿名获取的 NASTaR 提供 NovaSAR 影像、AIS 长宽和航向，可用于按过境划分的物理长度辅助验证。已读取一份 AIS 表头和一对 GeoTIFF 文件头，确认显示版为 512×512 uint8，原值版样本为 uint16。这里应以逐文件头为准。作者转换脚本的分割字段为空。另一项高分三号研究报告了 139 张带专家船体轮廓的裁块，来自三次采集，需要向作者申请。当前独立掩膜评价的现成路径仍是原先锁定的 12 个 Umbra 采集组的判读回传。 [NASTaR 数据目录](https://data.bris.ac.uk/data/dataset/2tfa6x37oerz2lyiw6hp47058) · [作者转换脚本](https://github.com/Benyaminhosseiny/NASTaR/blob/main/NASTaR_to_COCO_VOC.py) · [高分三号轮廓数据原文](https://doi.org/10.3390/rs16193669) · [Umbra 判读入口](UMBRA_PROSPECTIVE_GEC_ANNOTATION_ACTUAL_20261122.md)
