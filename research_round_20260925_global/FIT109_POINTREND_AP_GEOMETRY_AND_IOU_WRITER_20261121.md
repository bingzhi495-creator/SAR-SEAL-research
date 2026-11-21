# FIT109 的 AP 与连续边界效应

原 PointRend 同载体实测的 B 与 J 共用头、置信度、排序和每候选编辑量。以下事后诊断使用同一预测，按官方 COCO 匹配逐阈值复算。 [原实测](RFDETR_POINTREND_FIT109_ACTUAL_20261121.md) · [COCO 评估实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) · [TIDE 误差分析先例](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/849_ECCV_2020_paper.php)

- B 相对 J 的 +0.938 AP 点集中在较低阈值。AP60 为 +2.613，AP75 为 +0.031。IoU 0.60 下有 16 个 FP→TP 和 5 个 TP→FP。匹配竞争可使掩膜保持相同的候选改变状态。
- J 缩小掩膜的 4,158 个候选，在两种掩膜替换次序取平均后对应 +0.937 AP 点。J 扩大掩膜的 2,096 个候选对应 +0.001 点。余下 484 个候选编辑相同。官方 101 点 AP 差值中，可达召回网格项为 +0.851 点，网格内平均插值精度项为 +0.088 点。
- 在 392 个原固定匹配和 92 个来源组上，B−J 的真值交集项为 +1.602 IoU 点，预测面积项为 −1.518 点，净效应为 +0.084。两个项来自同一对编辑结果的精确代数分解。J 的平均绝对面积误差较 B 低 0.434 个真值面积百分点，组配对区间 [−0.844, −0.029]。局部过度删除与总体面积偏差改善可以同时出现。 [逐阈值与几何分解](results/FIT109_POINTREND_AP_GEOMETRY_DIAGNOSTIC_SAFE_20261121.json) · [替换诊断与召回分解](results/FIT109_POINTREND_BJ_REPLACEMENT_DIAGNOSTIC_SAFE_20261121.json)

## 依据估计 IoU 的同量选择

保持 6,738 个面积合格候选的两侧端点和逐候选编辑量，复用五个来源留出点头。每个非原生臂改动 97,964 个像素。T 在同一后验上用固定阈值 3/7 排序；R 遍历可行的增删数量，以预测交集与预测并集的比值选取最大者。给定数量时对两侧分别排序，因此能精确求解这个代理目标。它使用的是后验质量估计上的比值，不假定比值等于真实 IoU 的条件期望。后验排序和期望重叠决策已有直接先例，本次检验固定原候选和实际 2k 次写回的局部可行集合。 [RankSEG](https://www.jmlr.org/papers/v24/22-0712.html) · [候选约束的期望 IoU](https://openaccess.thecvf.com/content_iccv_2015/papers/Ahmed_Optimizing_Expected_Intersection-Over-Union_ICCV_2015_paper.pdf)

| 策略 | AP | AP75 | AP90 | 连续 IoU |
| --- | ---: | ---: | ---: | ---: |
| 原生 N | 54.844 | 58.830 | 6.556 | 83.934 |
| 等量增删 B | 54.400 | 57.729 | 7.319 | 83.776 |
| 自由增删 J | 53.462 | 57.698 | 7.049 | 83.692 |
| 固定偏置 T | 53.996 | 58.069 | 6.459 | 83.738 |
| 预测比值 R | 54.350 | 58.073 | 7.207 | 83.761 |

R 相对 T 增加 0.354 AP 点，连续 IoU 增加 0.023 点，来源组区间为 [−0.065, 0.102]。相对 B 和 N 的 AP 变化为 −0.050 与 −0.494 点，未满足预登记条件。当前后验下，改变增删代价或比值目标不足以超过强对照。结果限于 FIT109 与该单种子拟合。

首个预测进程封存全 109 图的选择后，在调用评分时因局部变量重名退出 1。读取原封存预测的评分进程退出 0。修正变量名后的完整重放也退出 0，逐候选编辑文件与原件 SHA 一致，评分逐值一致。独立密集栅格复算核对新臂的 97,964 次编辑、十档 AP 和连续几何，最大数值差为 1.42×10⁻¹⁴ 点。 [实测](results/FIT109_MATCHED_IOU_WRITER_ACTUAL_SAFE_20261121.json) · [复算记录](results/FIT109_MATCHED_IOU_WRITER_REPLAY_SAFE_20261121.json) · [源码与预锁设置](code/FIT109_MATCHED_IOU_WRITER_20261121/README.md)
