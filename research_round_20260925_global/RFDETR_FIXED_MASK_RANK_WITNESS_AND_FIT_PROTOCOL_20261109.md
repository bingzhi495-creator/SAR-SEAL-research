# 严格小船掩膜的候选与排名复核

原固定 DEV187 的小目标 IoU≥0.90 一对一覆盖，在每图前 1/10/100 个保留掩膜分别为 E24 15/42/61、F 18/39/47、S 16/37/57。S 的 21 个独有小目标中 15 个位于第 11–100 位。两端点都有合格候选的 36 个小目标中，S 有 7 个提前、10 个推迟、19 个持平。总前十覆盖 S 比 F 多 2 个，其中小目标少 2 个、其余多 4 个。此处分别报告独立候选存在与一对一覆盖。 [原实验](RFDETR_E24_STRICT_OPTIMIZER_PAIR_ACTUAL_20261109.md) · [逐目标聚合复核](results/RFDETR_STRICT_FS_RANK_WITNESS_SAFE_20261109.json)

官方 COCO 的 AP90 贪心 TP 总量与最大一对一覆盖完全相等，E24/F/S 分别为 100/80/90。F 的第 38 个 TP 出现在全局第 122 位，S 在第 145 位，之前分别有 84、107 个 FP。较深的候选覆盖优势与较低的早期精度同时发生。F/S 的掩膜、保留数和分数均可能参与该差异。 [官方评分轨迹复核](results/RFDETR_STRICT_FS_OFFICIAL_AP90_RANK_SAFE_20261109.json)

## 单一假设与 FIT 预锁检验

在同一冻结 E24 候选、同一原图像素支持与训练标签上，检验两侧灰度秩能否比同侧支持的有符号归一化灰度均值提供额外的高 IoU 候选排序信息。只换一个标量，分别对有效边界带取有符号均值。两头共用原分数的 logit、预测面积、带宽覆盖率、logit 边界余量、有效支持率和截距。空带与未匹配候选照常保留。采用候选对原 FIT 非 crowd GT 的最大 IoU 在十个固定 COCO 阈值上的通过比例为软目标，固定 128 次全批梯度更新和相同 L2。分数为原分数乘预测质量。

沿用已封存的原图分组。516 张拟合图有 42,481 个候选，111 张留出图有 9,083 个候选，114 张原校准图不参与拟合或阈值选择。留出原生 AP 和 AP75 必须复现已有不编辑基准。整个选择器留出对拟合独立，冻结检测器已经见过 FIT。无 SSDD 原 DEV、正式 TEST 或新分割模型更新。 [固定候选来源](RFDETR_E24_FIT741_NATIVE_CUDA_CAPTURE_ACTUAL_20261108.md) · [候选源码](code/RFDETR_E24_FIT_BILATERAL_QUALITY_20261109/bilateral_quality_scout.py)

针对同一个小目标、且各候选只与该 GT 有 IoU≥0.50 的明确对应，比较 IoU≥0.90 候选和 0.50≤IoU<0.90 候选的相对得分。并列计半，先按 GT 汇总再等权平均。进入后续阶段需至少覆盖 20 个 GT、15 张图，秩相对原分数和学习均值中较强者的一致率至少高 2 个百分点，全候选 mask AP 相对两者至少高 0.1 个百分点，AP75 不低于原生且 AP90 不低于两者。任何门不满足即停止此固定配方。评分头的相同参数规模和同支持不代表两种标量相同，正结果只支持本特征的边际作用。

质量回归乘分已有 Mask Scoring R-CNN 直接先例。MCG 已用面积、紧致度、轮廓强度估计候选 IoU，SAM 已用阈值稳定性筛选掩膜。 Girón 等已使用非参数两侧秩统计检测 SAR 边缘。 [SAR 非参数边缘原文](https://arxiv.org/abs/1207.1915)项目自身 N26 也已在固定 R 掩膜上证实通用学习质量排序的效用。这里的可检验差异只在相同支持下秩统计相对普通均值的附加信息。 [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf) · [MCG 作者训练源码](https://github.com/jponttuset/mcg/blob/e72031d793abf8921e39a8ef3c20de2198c8b26f/full/scripts_training/rank_training.m#L110-L127) · [SAM 稳定性源码](https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/utils/amg.py#L156-L176) · [N26 历史证据](N26_FROZEN_DUALGT02_ACTUAL_RESULT_20261107.md)
