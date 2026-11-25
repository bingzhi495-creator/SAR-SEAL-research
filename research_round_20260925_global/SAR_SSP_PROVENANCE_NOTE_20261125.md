# SAR-SSP 来源场景与预训练重叠核验

2026-11-25。使用 SLGSea 作者压缩包中的 SAR-SSP 标注 JSON 和 GrokSAR 公开的 AIR-SARShip-1.0 检测框转换件核验来源。输出只包含汇总统计。

## 来源匹配

SAR-SSP 的 929 张训练裁片和 232 张验证裁片均有 `source_image_no`、`tile_x`、`tile_y`、`sea_state`。按裁片范围重建得到 30 景 3000 × 3000 和 1 景 4140 × 4140。两种尺寸和数量与 AIR-SARShip-1.0 原论文相符。原始影像为高分三号。现有元数据足以按原景分组，缺少独立采集产品 ID、时间和场景间地理关系。 [SLGSea 作者仓库](https://github.com/deng1218/SLGSea) · [AIR-SARShip-1.0 原论文](https://doi.org/10.12000/JR19097) · [GrokSAR 数据说明](https://github.com/GrokCV/GrokSAR/blob/master/docs/datasets.md)

将 SAR-SSP 两个划分的 600 条舰船 bbox 记录加上各自裁片偏移还原到原景坐标。将 461 条 AIR-SARShip-1.0 的四边形检测记录转换为轴对齐外接框。每条 SAR-SSP 记录在每个候选 AIR 原景内取得最大 IoU，满足阈值即对该候选原景计一票。不同 SAR-SSP 裁片或记录可投票匹配同一个 AIR 检测框，因此这里统计匹配记录，不估计唯一舰船数。每景将同编号的票数与 30 个其他编号中最高的票数比较。

| 轴对齐框 IoU 阈值 | 同编号匹配记录数 | 各景最佳异编号票数之和 | 同编号严格最高的原景数 |
|---:|---:|---:|---:|
| 0.30 | 427 | 7 | 31/31 |
| 0.50 | 279 | 1 | 28/31 |
| 0.75 | 49 | 0 | 19/31 |

尺寸和坐标给出的相互独立证据均支持 SAR-SSP 裁片来自 AIR-SARShip-1.0 的 31 个编号原景。验证集涵盖全部 31 个原景，其中 23 景、48 张裁片有舰船记录。训练、验证共享全部原景，裁片原点没有重复，且有 4 对跨划分裁片各共享 180,000 个像素的坐标足迹，占单裁片面积的 72%。这些是分组和裁片来源核验。SAR-SSP 掩膜的标注工艺、质量及发布许可需依据其自己的来源说明核实。

## 历史模型暴露与现行冻结链

早期 B1 的实际训练配置继承 SARDet-100K DenoDet GFL 的骨干。按 SARDet-100K 训练 COCO 文件和作者文件名映射核得，其 438 张 AIR-SARShip-1.0 训练裁片覆盖 SAR-SSP 的 27 个原景。将两套裁片按原景做矩形足迹相交，SAR-SSP 验证裁片中有 111/232 张覆盖相交区域。交换 SARDet 文件名中的两个坐标后为 107/232 张。两种结果均为基于裁片命名和坐标的诊断，未逐像素比对。B1及沿用该骨干的衍生权重不能作为全量 SAR-SSP 零样本外测对照。SARDet 元数据可从作者仓库指向的 Kaggle 发布页取得。 [SARDet-100K 论文，表 1](https://papers.nips.cc/paper_files/paper/2024/file/e7eb8128eb26eafbe901348df1dbacdc-Paper-Conference.pdf) · [SARDet-100K 作者发布](https://www.kaggle.com/datasets/greatbird/sardet-100k)

现行 RF-DETR center5202 E24 冻结载体的独立接收记录将 `011bd30e…` 与 Q164 初始化 `63ff93af…`、HRSID FIT 索引以及实际完成的 24 轮 / 17,640 次更新绑定。COCO Seg Nano 初始化实际回执将官方权重 `a44613a4…`、Q100 `40524fc6…` 和同一 Q164 状态绑定。两份回执支持该实际项目链未使用早期 B1 或 SARDet-100K 骨干。上游官方权重的完整预训练数据谱系超出这项核验范围。匹配的 PointRend 控制独立实际回执将 `7401de08…` 与同一载体、109 张 HRSID FIT 图、种子 5217 和 200 次更新绑定。 [COCO 初始化回执](results/RFDETR_COCO_SEG_NANO_OFFICIAL_START_SAFE_20261109.json) · [PointRend 实际拟合回执](results/RFDETR_POINTREND_SOURCE_FULL_FIT109_ACTUAL_SAFE_20261122.json)

现行主方法还使用同一载体上拟合的 1,321 参数占用头，正式冻结权重摘要为 `3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1`。复核原始拟合回执 `114ac529…` 及自然退出的独立接收回执 `476b0f3f…`，其唯一种子 5217 全 FIT 记录使用 109 张 HRSID FIT 源图、92 个有效监督源组及 200 次更新，拟合阶段的 DEV、TEST 与 AP 字段均为 false。 [占用头复核摘要](results/RFDETR_CENTER5202_FIT109_MATCHED_TRAINING_PUBLIC_SAFE_20261120.json) · [固定源证据的复核代码](code/INSTANCE_RSDD_ACQUISITION_RESERVE_20261123/verify_source_identities.py)

如果验收样本和阈值全程冻结且参与的每个权重均无该语料暴露，可将 SAR-SSP 作者验证集作为按 31 个 AIR 原景聚类的 GF-3 外部域对照。若另与既有 GF-3 评测合并，需要核对产品或影像重叠。若在 SAR-SSP 内训练或调参，则需按 `source_image_no` 重划，现有训练与验证并非原景独立。本核验没有推理、图像或掩膜解码，也没有计算 AP。

## 复核入口

运行 `python compare_provenance.py <SLGSea_Datasets.zip> <AIR-SARShip-1.0/dota>`。如需复核 SARDet 历史暴露，在本地另行提供 `--sardet-train` 和 `--sardet-mapping`，对应文件 SHA-256 分别为 `5fd26a7431a70a9126ae14ffefce3c1d775244f9b6b0c9b444dc3edbc7a6f863` 和 `e0d814f6661657679ac355f52728df94cb0eb02d00756b7a0d97ce50aa787f55`。代码不输出逐实例明细。
