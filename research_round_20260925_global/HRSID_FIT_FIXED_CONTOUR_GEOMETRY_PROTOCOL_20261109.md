# HRSID FIT 固定查询的轮廓几何对照

原生候选筛选的严格缺口已通过全 164 查询检查。109 个主图组上原生 496 个有效 GT 有 37 个严格达标，全查询为 39。五张密集压力图在 0.90 全部未达标。本轮只研究这些图上原生候选已经达到 0.75、同图原生查询最佳掩膜仍低于 0.90 的情形。固定原生候选、原分数及其顺序，不新增前向或训练，不改变 FIT、DEV、TEST 角色。 [原始查询对照](HRSID_FIT_NATIVE_QUERY_RETENTION_PROTOCOL_20261109.md)

复用已闭合的 114 张图原始预测、冻结组清单及同一 FIT 栅格。按原生候选顺序以 0.75 阈值做贪心一对一，当前候选取未占用 GT 中的最大 IoU，平分按原标注顺序靠后者。只保留原生最佳 IoU 未到 0.90 的 GT，固定这条查询和 GT。复核主层为 234 对、压力层为 59 对。各个对照独立修改固定查询的二值掩膜，分数和配对保持不变。原严格匹配始终保持。诊断恢复指同一配对达到 0.90，不计算新的 COCO 匹配或 AP。

| 对照 | 候选改动 |
| --- | --- |
| 原样 | 恒等 |
| 整体平移 | 两轴整数偏移各 −2 到 2，拒绝导致前景出界的移位 |
| 整体厚度 | 对原掩膜做半径 1 或 2 的方形结构元腐蚀或膨胀，图外为背景 |
| 联合整体 | 先做厚度改动，再平移，包含两类单独操作和恒等 |
| 放宽预算的整体 | 同一联合家族，仅去掉像素改动数约束 |
| 局部边界上限 | 在原掩膜的 4 像素切比雪夫边界带内，真值引导补回漏分或去掉误分 |
| 全图编辑上限 | 同预算允许在整图补回漏分或去掉误分 |

有预算的对照统一使用 B = min(16, floor(原预测与 GT 并集面积 / 10))，每改变一个二值像素计一次。整体家族在允许操作中按真值 IoU 最大选取，平分优先更少改动，再用固定枚举序。局部带由原掩膜的半径 4 膨胀扣除半径 4 腐蚀得到，覆盖联合整体操作的可能改动位置。相同预算的最优有利编辑先补 FN，再去 FP。补 FN 增加交集且保持并集，去 FP 保持交集且减少并集，在 IoU 未达 1 时，同等编辑数优先补 FN 得到不低的 IoU。每种策略报告实际改动量、TP、FP、FN、达到 0.90 的计数和带内可编辑错误。另报多少配对存在至少一个非恒等的预算内整体动作、在统一改动预算内即使任意位置编辑也无法恢复的比例，以及按原 IoU 和 GT 面积分层的结果。放宽对照用于检查整体操作的最小步幅是否已超预算。

决策以 109 个主图组为单位做 4,096 次配对重采样，固定种子 5213，报告按合格配对数归一化的恢复率差区间。只有局部带至少恢复 20% 且覆盖至少 15 个图组、比预算内联合整体至少多恢复 10 个百分点、比放宽预算的联合整体至少多恢复 5 个百分点且配对区间下界大于零，才继续检验预测侧局部变形。若未通过且放宽的联合整体恢复至少 20% 并覆盖至少 15 组，优先把全局位移及厚度作为对照；若两类均不足，按全图上限区分预算和可编辑位置后重新检查误差机制。压力层独立报告，不参与主决策。所有上限依赖 GT。算子家族有限，结果只能约束本轮平移及方形结构元，不能排除旋转、方向尺度、亚像素重采样等其他整体解释。

Deep Snake 和 SharpContour 已学习轮廓位移，Boundary Patch Refinement 已学习局部边界修复，TIDE 已使用真值干预诊断掩膜误差。SSDD 早先的固定 16 点上限主要检查点前缀的可达性，本轮新增整体与局部家族在同一 HRSID 配对上的预算对照。若局部上限成立，后续仍须同监督、同候选、同编辑成本学习对照，才能判断能否转化为 AP。 [Deep Snake](https://openaccess.thecvf.com/content_CVPR_2020/html/Peng_Deep_Snake_for_Real-Time_Instance_Segmentation_CVPR_2020_paper.html) · [SharpContour](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_SharpContour_A_Contour-Based_Boundary_Refinement_Approach_for_Efficient_and_Accurate_Instance_Segmentation_CVPR_2022_paper.html) · [Boundary Patch Refinement](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html) · [TIDE](https://dbolya.com/tide/) · [SSDD 固定前缀](RFDETR_FIXED_MASK_RANK_WITNESS_AND_FIT_PROTOCOL_20261109.md)
