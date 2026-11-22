# SAR 船舶实例分割投稿评议

**2026 年 11 月 22 日。论文已有外审前评阅基础。** HRSID 的证据支持明确、可复现的配对限制效应。以原生实例边界、同量归因为主线，已具备专业审阅和讨论是否送审的基础。独立采集评估暴露了原生匹配覆盖不足，正式投稿的跨域主张仍需限定。 [当前英文稿](paper_boundary_refinement_20261120/boundary_refinement.pdf) · [主张与证据矩阵](SAR_SHIP_MASK_TGRS_EVIDENCE_MATRIX_20261112.md) · [TGRS 作者说明](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/)

| 可写的贡献 | 当前证据 |
| --- | --- |
| 原生船舶边界误差的几何诊断 | 连续多边形和固定匹配区分候选框截断、增删位置与面积偏差。相同编辑量时，本船 IoU 的改善可能增加邻船侵入。D-FINE 的邻船规则在 FIT109 局部有效，DEV705 只有 7 个暴露匹配、3 个来源前缀，整体 AP 持平。 [边界误差](DEV705_MATCHED_PAIRING_GEOMETRY_MODIFIERS_20261121.md) · [邻船复验](FIT109_NATIVE_RIVAL_OCCUPANCY_ACTUAL_20261122.md) |
| 匹配编辑量的边界修正与归因 | 在相同端点、置信度、点头和逐候选编辑量下，独立选点较距离固定配对提高 D-FINE 的 0.811 AP 点，RF-DETR 两次下游拟合的 0.906／0.838 AP 点，以及 HRSID 历史官方划分的 1.216 AP 点。Atkinson 已有独立等量像元交换，RankSEG 已有排序与体积选择。本文的增量落在原生实例边界支持和同量机制证据。 [官方实测](HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md) · [第二检测器复验](RFDETR_COMMON5203_DEV705_FROZEN_ACTUAL_20261122.md) · [Atkinson 原文](https://doi.org/10.14358/PERS.71.7.839) |
| 冻结验证、适用范围和工程复现 | HRSID 官方原生 54.433→55.454 AP，三个点头均提高；两次检测器训练共享初始化，正式划分曾用于模型选择。PSeg-SSDD 历史官方迁移 +0.344 AP，同量机制 +0.081。232 图在输入已驻留条件下，CPU 精修均值 158.8→104.3 毫秒，输出逐位相同。 [冻结迁移](PSEG_SSDD_TEST232_DFINE_FROZEN_ACTUAL_20261121.md) · [等价取样](results/PSEG_TEST232_DFINE_EXACT_COST_ACCELERATION_SAFE_20261121.json) |

三个点头都在第二个检测器上复现了固定配对的影响。相对允许改变面积的普通同量写回，HRSID 官方平均 +0.324 AP，仍有一个点头退步；第二个检测器的连续 IoU 区间跨零。分数占据率相对 COCO 硬栅格有小幅优势，相对同一覆盖率的 0.50 硬化标签平均只增 0.051 AP，一个种子下降。面积和软标签的独立作用均应按这些证据陈述。 [分数标签控制](FIT109_FRACTIONAL_TARGETS_ACTUAL_20261122.md)

UEIKAP 的 108 张作者原图对应 139 个已标注船体及 18 个预留 Sentinel-1 采集组。冻结 D-FINE 在原生阈值仅匹配 2 个船体（1.44%），二者固定配对预算均为零。主比较的有效覆盖为 0 艘、0 组，差值和区间均为零。两例相同动态空间吸引力预算下，学习排序的连续 IoU 平均低 0.914 点。检查全候选也仅有 5 个达标。前 100 候选在原面积、至多八次真值交换的放宽条件下最多覆盖 4 个。该结果约束本次载体与编辑预算的可修复范围。独立学术评审认为限定至 HRSID 同头同量机制后，作者资料齐备即可尝试 TGRS 送审。项目建议优先取得独立且匹配充足的闭集证据，以提高投稿依据。 [UEIKAP 冻结实测](UEIKAP_RESERVED_HULL_ACTUAL_20261122.md)

在历史 HRSID DEV705 上，检测器训练未使用对应影像。同量学习排序较经典动态规则提高 2.428 AP、5.353 AP75 和 1.150 连续 IoU，27 组配对区间 [0.927, 1.397]。这种严格同量对照支持源域排序贡献。在历史 PSeg-SSDD 232 图上，经典空间吸引力是竞争性对照。定住逐候选实际编辑量后，学习排序相对动态经典规则 −0.026 AP、+0.279 AP75、−0.270 连续 IoU（按解码图像组等权，95% 区间 [−0.440, −0.098]）。去除历史重复后方向一致。主结论依旧是 HRSID 上的固定配对限制，应将学习占据率比经典规则更有效的主张限定于 HRSID 的冻结载体。 [HRSID DEV705 实测](HRSID_DEV705_CLASSICAL_ATTRACTION_ACTUAL_20261122.md) · [经典对照实测](PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md)

在历史 DEV187 的目标训练 RF-DETR 上，HRSID 冻结点头和 PSeg FIT109 点头相对动态空间吸引力分别提高 0.845 和 0.695 AP。两者的连续 IoU 区间均跨零。原生 70.694 AP 领先所有编辑组，目标点头相对原生下降 0.885 AP 和 0.470 连续 IoU 点，结束此同量配方。相同端点和交换量的真值几何上界仍高于原生 2.246 点，放宽次数多 0.032，开放面积多 1.018。稿件应保留 HRSID 已重复的同量效应，并报告目标域单种子重拟合的局限。[目标强载体实测](PSEG_RFDETR_FROZEN_SOURCE_HEAD_ACTUAL_20261122.md) · [目标点头重拟合](PSEG_RFDETR_TARGET_FIT109_ACTUAL_20261122.md)

## 投稿前最关键的补证

在采集来源可核验、此前未参加开发且有足量可编辑船舶的场景上，一次冻结原生、固定配对、同量独立、自然计数和直接强对照，按采集组报告 AP、连续 IoU 和失败范围。其中 Instance-RSDD 高分三号子集值得优先核验。已定位作者的实例标注目录，获取入口需要账户登录，图像须从原始 RSDD 发行获取。入库后核对场景身份、标注覆盖、许可与重叠。OSSDD 的掩膜由旋转框填充，VesselBench 目前给出检测框，二者适合其定位任务。 [Instance-RSDD 论文](https://www.mdpi.com/2072-4292/15/5/1324) · [作者发布](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset) · [OSSDD 标注协议](https://arxiv.org/html/2608.01963v1#S4.SS1) · [VesselBench](https://arxiv.org/html/2609.37003v1)

SMART-Ship 原文确认靠泊船舶多边形，当前可核入口需要申请。SISS-YOLO 报告由 Labelme 勾画的 Sentinel-1 船体，公开数据声明只有原始 OpenSARShip2.0 入口。两条路径目前缺作者轮廓文件或采集索引。Umbra 已按预锁顺序和空间间隔取回 12 个不同采集组的原生窗口，并完成两份离线匿名判读材料。该补充证据仍依赖独立人工轮廓与一致性。 [Umbra 已核验准备](UMBRA_PROSPECTIVE_GEC_ANNOTATION_ACTUAL_20261122.md) [SMART-Ship 原文](https://arxiv.org/html/2508.02384v2) · [SISS-YOLO 原文](https://www.mdpi.com/2072-4292/17/17/3118)

作者、单位、通讯信息和 ORCID 保留占位。原 A/B 长训练停止结果及全部历史试验仍见证据矩阵。 [研究收尾计划](PAPER_CLOSEOUT_PLAN_20261121.md)
