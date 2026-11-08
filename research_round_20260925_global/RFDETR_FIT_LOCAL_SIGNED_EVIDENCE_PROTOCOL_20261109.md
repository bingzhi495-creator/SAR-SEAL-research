# SSDD 原图边界的双侧纠错证据

上轮在相同候选与分数上观察到真值拟合系数的 117 个双侧修复配对。此前原灰度法线对应、既有秩头和原图 patch 都未通过各自固定门槛。本轮检验预测边界两侧的冻结语义特征差，是否比纯分数、常规点特征和实例全局向量更能定位应补入及应移除的原像素。 [有界动作实测](RFDETR_FIT_BOUNDED_ACTION_HEADROOM_PROTOCOL_20261109.md) · [灰度对应实测](RFDETR_E24_FIT_CONTOUR_CORRESPONDENCE_PROTOCOL_20261109.md)

从预测掩膜的原图 2 像素带生成候选点。每侧最多 128 个，按固定图像索引和查询种子从同带确定抽样。由最近反侧的像素中心确定朝前景的方向，在离反侧中心最近的半像素位置取边界锚。没有读取真值选点或定方向。统一用原生贪心配对建立诊断标签，标签为当前像素二值预测是否与真值不同。保留原已严格的配对，未匹配候选只计数。合成验证检查边界方向、稳定并列次序、固定补删预算和退化掩膜。

五组均用 6 个公共预测量和 32 个附加量、同宽 32 的单隐层 MLP、同初值、同 96 次全批 AdamW 与同组等权 BCE。公共量为中心 logit、反侧距离、原掩膜内外、原掩膜面积的对数、候选分数与位次。纯 logit 组用固定图像轴邻域 32 点。常规点组用冻结基底的中心 32 维。全局组用同中心 16 维加原实例全局对比的 16 维。法线组和切线组在同边界锚、同两像素半径采样，使用相同中心 16 维及相反位置的 16 维特征差。空间特征使用同一固定正交投影，常规点组代表 PointRend 风格的信息对照。各组同名义容量，采样计算和输入信息分别报告。另有无更新的原不确定性、零编辑参考和在同支持上用真值排顺序的机会读出。

484 个 FIT 拟合图、479 个编码组，排除此前 32 个诊断组。评分使用互斥的 114 个校准角色图、109 个编码组。该组曾参与旧筛选器校准，也在原载体训练范围内，本轮用于与新读出训练分组分离的探索性评价。原 111 图不读。锁定每个原配对每侧四个编辑，支持不足时按真实支持缩短，各组同位置集合和同侧计数。计算补入与移除的整数 0.90 裕量、FP/FN、双侧净改善与原严格失守。K1 为敏感性。图像编码组配对做 2,048 次重采样。

至少 60 个有配对的编码组，法线组两侧平均裕量均为正，相对常规点组的总裕量均值至少 +1，且相对所有预测组和零编辑的配对区间下界均大于 0，原严格失守不多于常规点组，才进入无真值全候选读出。若此门槛未通过，按原配方停止。诊断使用真值固定配对且强制同编辑预算，单独记录未匹配规模。无掩膜 AP、正式 DEV 或 TEST。后续全候选须加入预测侧弃权，保留零编辑选项。

PointRend 已用预测位置的细特征和粗分数，BPR 已用局部预测掩膜块，SegFix 与 SharpContour 已用方向或法线做边界修正，Mask Transfiner 已检测不一致区域。此实验隔离的是冻结语义下跨预测边界的局部对应效用。 [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) · [BPR](https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html) · [SegFix](https://arxiv.org/abs/2007.04269) · [SharpContour](https://arxiv.org/abs/2203.13312) · [Mask Transfiner](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html)

[实验源码](code/RFDETR_FIT_LOCAL_SIGNED_EVIDENCE_20261109/local_evidence.py) · [支持和有符号计算](code/RFDETR_FIT_LOCAL_SIGNED_EVIDENCE_20261109/local_math.py)

