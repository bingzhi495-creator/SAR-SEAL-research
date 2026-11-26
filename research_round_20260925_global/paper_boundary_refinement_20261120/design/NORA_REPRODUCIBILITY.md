# NORA 的方法名称与实验编号

当前稿标题为 **NORA: Native-Grid Occupancy Ranking and Allocation for SAR Ship Instance Segmentation**。Ranking 指合法端点的占据率排序。Allocation 指在固定总编辑数下分配增加和删除。正文先报告等增删量的 F、L、I，再报告固定总量的可变配比。F 表示距离优先配对，L 表示在原最大单对距离下重配，I 表示两侧独立选点。这些写法对应原有的冻结预测与评价。

## 复现编号

| 正文别名 | 原实验标识及含义 | 结果和入口 |
| --- | --- | --- |
| RF-DETR fit A | 检测器下游训练种子 5202 | [HRSID DEV705](../../RFDETR_CENTER5202_DEV705_MATCHED_RESULT_20261120.md) · [原版官方集](../../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md) |
| RF-DETR fit B | 检测器下游训练种子 5203；与 A 共用训练数据及初始化 | [HRSID DEV705](../../RFDETR_COMMON5203_DEV705_FROZEN_ACTUAL_20261122.md) · [Instance-RSDD 冻结迁移](../../INSTANCE_RSDD_CARRIER5203_ACTUAL_20261123.md) |
| H1、H2、H3 | 标量占据头的 5217、5218、5219。只用 H1 的结果对应首个冻结点头。同侧消融表的 H1、H2 对应前两头。 | [同侧消融](../../INDEPENDENT_SIDE_FACTORIAL_ACTUAL_20261123.md) · [限距对照与采集分组](../../code/LEARNED_BOUNDED_PAIRING_20261126/ACTUAL.md) |
| 双参考表中的拟合 1、2、3 | 双输出占据头的 5297、5298、5299。配比分析中的初始拟合和近似目标复算均对应 5297。 | [三次拟合](../../code/SPSD_PSEG_REPEAT_VARIABLE_SIGNED_BUDGET_SEEDS5298_5299_20261126/ACTUAL.md) · [目标域冻结迁移](../../code/SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/ACTUAL.md) |

定性图中，HRSID DEV705 的交集分解和高分三号的学习对动态吸引力使用 H1。它们分别来自 RF-DETR fit A、fit B。高分三号和历史 Instance-RSDD 保留集的共同影像以及 HRSID 的历史评估暴露均按正文口径报告。图 6 的别名来自同一批汇总输入。D-FINE E24 的独立冻结流程沿用原标识。

## 细粒度复算记录

- HRSID 官方集上，H1–H3 的固定增删量 AP 增益依次为 1.052、1.297、1.299 点。自然增删量相对原生为 1.078、1.062、0.923 点。自然量下的小头独立写回相对大头可变配比为 +0.184、+0.963、−0.174 AP 点。[正式分配实测](../../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md)
- 高分三号的等增删量 AP 增益在 fit A 的两头上为 0.482、0.483 点，在 fit B 的三头上为 0.240、0.280、0.378 点。fit A 在自然量相对于原生的分头变化为 AP +0.587、+0.517，AP75 +1.438、+1.230，AP90 −0.094、−0.107 点。[占据头与数据视图](../../INSTANCE_RSDD_CARRIER5203_ACTUAL_20261123.md) · [等量对照](../../INDEPENDENT_SIDE_FACTORIAL_ACTUAL_20261123.md)

正文以拟合数、均值或范围描述重复结果。表内可分头比较的原始数值仍与原表一致。具体的划分、冻结载体和原始候选由上述记录、程序及聚合回执定位。[完整稿](../boundary_refinement_complete.pdf)
