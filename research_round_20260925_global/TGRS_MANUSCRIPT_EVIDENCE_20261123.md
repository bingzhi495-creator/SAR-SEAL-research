# 边界精修论文审阅稿与证据索引

2026 年 11 月 23 日。IEEE 双栏英文稿面向 TGRS 学术审阅，作者信息按约定保留占位。正文沿着窄船栅格偏移、匹配预算检验、跨库结果、失效边界和复现条件展开。[英文 PDF](paper_boundary_refinement_20261120/boundary_refinement.pdf) · [LaTeX](paper_boundary_refinement_20261120/boundary_refinement.tex) · [图件及重建说明](paper_boundary_refinement_20261120/figures/README.md)

## 主张与核验记录

| 论文问题 | 直接证据 | 适用范围 |
| --- | --- | --- |
| 固定编辑量时，强制固定配对有什么代价 | HRSID DEV705 的 D-FINE、RF-DETR 5202／5203 分别提高 0.811、0.906、0.838 mask AP 点。5202 的 HRSID 官方分配重评三头均值 +1.216 AP。 | 同检测候选、同分数、同端点、同实际逐候选编辑量。HRSID DEV 有开发接触，官方分配为历史重评。[D-FINE](DFINE_E24_DEV705_FREE_ENDPOINT_20261120.md) · [RF-DETR 5202](RFDETR_CENTER5202_DEV705_MATCHED_RESULT_20261120.md) · [RF-DETR 5203](RFDETR_COMMON5203_DEV705_FROZEN_ACTUAL_20261122.md) · [官方集复算](HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md) |
| 固定配对的差异能否跨库重现 | Gaofen-3 583 图，两次 RF-DETR 拟合分别 +0.483 和 +0.299 AP。配对连续 IoU 分别 +0.259 [0.194, 0.322] 和 +0.196 [0.136, 0.253] 点。 | 两次检测器冻结外推，部分样本共享。各自原生匹配均为 425 个，贡献组分别为 68 和 67。[首次迁移](INSTANCE_RSDD_FROZEN_HRSID_ACTUAL_20261122.md) · [双侧分解](INDEPENDENT_SIDE_FACTORIAL_ACTUAL_20261123.md) · [第二次迁移](INSTANCE_RSDD_CARRIER5203_ACTUAL_20261123.md) |
| 主结果由全部个体共同驱动吗 | Gaofen-3 5203、头 5217 在动态吸引力预算下，425 个合格匹配中 268 个改善、154 个下降、3 个无差。图 4 按正、负两侧各自中位数选例，四例都做八次交换。 | 以连续 IoU、原生固定匹配和精确交集为依据。右列显示净交集位置变化，公开图只含局部空间聚合。[图件与选例规则](paper_boundary_refinement_20261120/figures/README.md) |
| 同端点排名和整区精修的结论是否一致 | Gaofen-3 上，占据率较同支持同量的 PointRend 提高 +0.312／+0.390 AP。整框 PointRend 比无候选上限的占据率高 0.200 AP，AR100 低 1.744 点。74 图上 Phoenix 同端点同量领先三头均值 0.308 AP，几何区间跨零。 | 同支持适配和整区输出分别报告。Phoenix 的同队列整图评价见下方补证。[PointRend 支持对照](INSTANCE_RSDD_POINTREND_SUPPORT_FACTOR_ACTUAL_20261123.md) · [双侧位置移植](POINTREND_SIGNED_QUOTA_SWAP_ACTUAL_20261123.md) · [Phoenix](INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md) |
| 现有失败如何约束泛化声明 | 去重 PSeg 与 UEIKAP 追加 342 图上，学习排名分别比同量动态吸引力低 0.275 和 0.053 连续 IoU 点。UEIKAP 预留 139 艘只有 17 艘具合格正预算的原生匹配。 | 未匹配轮廓在 UEIKAP 总体统计中计零，无全场景 AP。固定面积精修对漏检和面积不足作用受限。[PSeg](PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md) · [UEIKAP 追加](UEIKAP_ADDITIONAL_FAMILIES_ACTUAL_20261123.md) · [UEIKAP 覆盖](UEIKAP_RFDETR_CARRIER_ACTUAL_20261122.md) |
| 结论能否从作者数据复算 | 主表 3,434 项数值与独立重跑逐值一致。74 图在干净环境重跑，1,470 项数值逐值一致。 | 需要作者发布数据和项目指定冻结权重。74 图复现限定检测器 5202 与头 5217。[主表复现](SAR_B5_MAIN_TABLES_REPRODUCTION_ACTUAL_20261123.md) · [74 图复现](INSTANCE_RSDD_B5_REPRODUCTION_ACTUAL_20261123.md) |

## 强精修整掩膜补证

同一 Phoenix XL1 权重在 74 图上以公开 LVIS 面积规则整掩膜写回时，较同端点同量的 Phoenix 下降 **17.779 AP、9.825 连续 IoU 点**，几何区间 **[−14.014, −6.440]**。固定匹配的额外船外面积增加 33.726 点。按公开 COCO 面积规则仅改写 172 个候选，研究面积区间保持原生。这一结果已补进英文稿。结论限定于已评估的 Instance-RSDD 74 图，新的预注册采集结果仍需完成。[评分和复现](RSDD74_PHOENIX_WHOLEMASK_ACTUAL_20261123.md)

在保持同一端点、同一分数和逐候选 2k 次翻转时，严格等面积相对自由增删仅 **+0.190 AP、+0.083 连续 IoU [−0.191, 0.401]**。因此强对照中观察到的整掩膜过扩张不能归因于严格等面积这一项约束。连续 IoU 同时取决于净面积变化和真实交集变化，稿件分别给出其关系和实测。[同量配额消融](RSDD74_PHOENIX_AREA_ABLATION_ACTUAL_20261123.md)

固定同一完整 Phoenix 分数、原生候选与范围时，整图定额较整图阈值 **+9.393 AP、+9.462 连续 IoU [6.329, 13.283]**，船体外误分减少 33.131 个面积归一点。限定同实际翻转量后，窄端点较整图 **+0.309 AP、+0.281 IoU [−0.025, 0.666]**。这补足了写回范围和翻转规则的交叉对照，支持阻止该冻结配置的过量写回。[四组实测](RSDD74_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md)

同一操作对照迁移到 PSeg-SSDD 官方历史划分，以冻结 HRSID D-FINE 代替 RSDD 的 RF-DETR，仍得到 **+2.525 AP 和 +4.001 连续 IoU [2.644, 5.382]**。整图自由增删定额在 PSeg 低于原生 D-FINE 和经典空间吸引力。去重后两项主效应保留。证据支持两个历史数据上的精修过量问题，采集域泛化和跨库实用领先仍待核验。[PSeg 四组实测与核验](PSEG232_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md)

修订稿已纳入同端点补证。实测显示，PSeg 的窄域 Phoenix 在保持同一后验和总翻转量时，改用等量增删获得 **+1.842 AP 和 +1.151 IoU [0.701, 1.594]**。等量 Phoenix 的 18.222 AP 与同量静态吸引力的 18.417 AP 差 0.195 点，联合判据未支持任一方向。Instance-RSDD 的同类差异较小。两组结果要求将整体编辑量、两侧分配及端点排序分开论证，并保留载体与数据同时变化的条件。[同增删量实测](PSEG232_PHOENIX_BALANCED_RANK_ACTUAL_20261123.md)

## 正式投稿前的取舍

12 个先锁定的 Umbra 采集窗口已完成冻结检测、原生端点精修、作者 Phoenix 整掩膜及独立复算。双参考评分器通过合成测试。两份原始轮廓回传后，依预锁口径报告等量端点主比较和整掩膜次要对照，纳入采集组覆盖。 [盲测执行](UMBRA12_FROZEN_CARRIER_ACTUAL_20261123.md) · [作者整掩膜对照](UMBRA12_PHOENIX_WHOLE_ACTUAL_20261123.md) · [评分准备](UMBRA12_DUAL_REVIEW_SCORE_READINESS_20261123.md)

本稿已具备围绕固定预算边界决策开展专业审阅的证据。能优先保留的贡献是把编辑策略、评分和预算分开，并跨两个检测框架和两个高分三号载体得到可复核的效果。数学依据与经典分侧像素交换及后验概率排序有明确关系，技术贡献应落在任务定义、对照设计和原生栅格边界证据。分数标签单独带来稳定外部收益的证据仍弱，严格 AP90 的表示敏感性和 AIS 长度误差应保留。 [严格阈值复评](RSDD_STRICT_CONTINUOUS_ACTUAL_20261123.md) · [分数监督对照](RFDETR_FRACTIONAL_TIE_TRANSFER_ACTUAL_20261123.md) · [AIS 长度外测](GULF_AIS_GEOSCALE_ACTUAL_20261123.md)

若以 TGRS 正式投递为目标，最能改变判断的是一个事先锁定且此前未参与开发的采集队列，在同一冻结检测和标注下给出强精修的实际流程、同量排序和覆盖三类结果。作者信息及原图展示许可也需落实。TGRS 对遥感创新与实验报告的要求见[期刊作者指南](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/)。
